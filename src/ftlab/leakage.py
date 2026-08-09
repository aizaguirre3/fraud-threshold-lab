"""Leakage guard.

Three distinct things can leak in a temporal fraud problem, and they need three
distinct checks:

1. **Identifier / target columns reaching the feature matrix.** ``isFraud`` is
   the obvious one; ``TransactionID`` and raw ``TransactionDT`` are the subtle
   ones. Both are monotone in time in IEEE-CIS, so a tree model will happily
   split on "row index > N" and score well in-sample while learning nothing
   transferable.

2. **Target-derived columns.** A feature accidentally built from the label
   betrays itself by separating the classes almost perfectly on its own.

3. **Future information inside behavioural aggregates.** Per-card frequency and
   amount statistics are the highest-value features in this dataset and the
   easiest place to leak: computing them over the full dataset lets a training
   row see the behaviour of its own card in the test period. Aggregates
   therefore carry provenance and are checked against the training index.

Every check raises :class:`LeakageError` rather than warning. A leak that only
warns is a leak that ships.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .config import AMOUNT, ID_COL, TARGET, TIME_COL


class LeakageError(AssertionError):
    """Raised when a feature matrix violates a leakage invariant."""


#: Columns that must never appear as model inputs.
#: ``TransactionAmt`` is deliberately *not* here -- the amount is known at
#: authorisation time and is a legitimate feature. It is also used as the
#: false-negative cost, but that is a downstream evaluation concern, not leakage.
FORBIDDEN_AS_FEATURE: frozenset[str] = frozenset({TARGET, ID_COL, TIME_COL})


def assert_no_forbidden_columns(
    X: pd.DataFrame, extra_forbidden: frozenset[str] | set[str] = frozenset()
) -> None:
    """Fail if the target, an identifier, or the raw time axis is a feature."""
    forbidden = set(FORBIDDEN_AS_FEATURE) | set(extra_forbidden)
    present = sorted(forbidden.intersection(X.columns))
    if present:
        raise LeakageError(
            f"Forbidden column(s) present in feature matrix: {present}. "
            "The target, row identifier, and raw transaction time must never be "
            "model inputs; derive bounded features (hour-of-day, day-of-week) instead."
        )


def single_feature_auc(x: pd.Series, y: pd.Series) -> float:
    """Orientation-free AUC of one feature against the target.

    Returns 0.5 for a feature that is constant or entirely missing, since such a
    column carries no separating information and cannot be a leak.
    """
    mask = x.notna()
    if mask.sum() < 2:
        return 0.5
    xv = x[mask]
    yv = y[mask]
    if yv.nunique() < 2 or xv.nunique() < 2:
        return 0.5
    auc = roc_auc_score(yv, xv)
    # A perfectly *inverted* feature leaks just as badly as a perfectly aligned one.
    return max(auc, 1.0 - auc)


def find_target_derived_columns(
    X: pd.DataFrame, y: pd.Series, max_abs_auc: float = 0.999
) -> list[tuple[str, float]]:
    """Return ``(column, auc)`` for any numeric column that all but perfectly
    separates the classes on its own."""
    suspects: list[tuple[str, float]] = []
    for col in X.columns:
        s = X[col]
        if not pd.api.types.is_numeric_dtype(s):
            continue
        auc = single_feature_auc(s, y)
        if auc >= max_abs_auc:
            suspects.append((col, float(auc)))
    return sorted(suspects, key=lambda t: -t[1])


def assert_no_target_derived(
    X: pd.DataFrame, y: pd.Series, max_abs_auc: float = 0.999
) -> None:
    """Fail if any single column separates the classes essentially perfectly."""
    suspects = find_target_derived_columns(X, y, max_abs_auc=max_abs_auc)
    if suspects:
        detail = ", ".join(f"{c} (auc={a:.4f})" for c, a in suspects)
        raise LeakageError(
            f"Column(s) separate the target almost perfectly and are probably "
            f"target-derived: {detail}"
        )


def assert_fit_window_is_train_only(
    fit_index: pd.Index | np.ndarray, train_index: pd.Index | np.ndarray
) -> None:
    """Fail if a statistic was fit on rows outside the training window.

    ``fit_index`` is the set of rows a behavioural aggregate was computed from.
    Any row in it that is not in ``train_index`` is future information.
    """
    fit = pd.Index(fit_index)
    train = pd.Index(train_index)
    outside = fit.difference(train)
    if len(outside) > 0:
        raise LeakageError(
            f"{len(outside)} row(s) used to fit an aggregate fall outside the "
            f"training window (e.g. {list(outside[:5])}). Behavioural aggregates "
            "must be computed from the training period only."
        )


def assert_splits_are_temporally_disjoint(
    train_time: pd.Series, val_time: pd.Series, test_time: pd.Series
) -> None:
    """Fail unless train strictly precedes validation, which strictly precedes test."""
    if not (train_time.max() <= val_time.min()):
        raise LeakageError(
            f"Train window overlaps validation: train max {train_time.max()} > "
            f"val min {val_time.min()}."
        )
    if not (val_time.max() <= test_time.min()):
        raise LeakageError(
            f"Validation window overlaps test: val max {val_time.max()} > "
            f"test min {test_time.min()}."
        )


def audit(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    max_abs_auc: float = 0.999,
    extra_forbidden: frozenset[str] | set[str] = frozenset(),
) -> dict:
    """Run every applicable check and return a summary for the run report."""
    assert_no_forbidden_columns(X, extra_forbidden=extra_forbidden)
    assert_no_target_derived(X, y, max_abs_auc=max_abs_auc)
    aucs = [single_feature_auc(X[c], y) for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    return {
        "n_features": int(X.shape[1]),
        "n_rows": int(X.shape[0]),
        "forbidden_check": "pass",
        "target_derived_check": "pass",
        "max_single_feature_auc": float(max(aucs)) if aucs else float("nan"),
        "amount_column_retained_as_feature": AMOUNT in X.columns,
    }
