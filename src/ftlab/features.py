"""Feature engineering across three families, with provenance on the aggregates.

Families
--------
1. **Transaction** -- amount and its shape (log, cents remainder), product code,
   card attributes, billing address, distances, email domains, and the raw
   ``C*``/``D*``/``M*``/``V*`` blocks the competition ships.
2. **Device and identity** -- the ``id_*`` block, device type and description,
   plus an explicit ``identity_present`` flag. That flag matters: identity is
   attached to only ~24% of rows, and those rows carry a materially different
   fraud rate, so absence is information rather than an inconvenience.
3. **Behavioural aggregates** -- per-card and per-address frequency and amount
   statistics, and each transaction's amount relative to its card's typical
   amount. These are the strongest features available and the easiest way to
   leak, so they carry the training index they were fit from and are checked
   against it.

Time is represented only as hour-of-day and day-of-week. Raw ``TransactionDT``
never enters the matrix -- it is monotone across the split and would let the
model identify the period rather than the behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import AMOUNT, ID_COL, TARGET, TIME_COL
from .leakage import assert_fit_window_is_train_only

#: Keys that behavioural aggregates are grouped by. ``card1`` is the closest
#: thing the anonymised data has to a card identifier; ``addr1`` is the billing
#: region. The pair captures "this card, used from its usual region".
AGGREGATE_KEYS: tuple[tuple[str, ...], ...] = (
    ("card1",),
    ("card2",),
    ("addr1",),
    ("card1", "addr1"),
    ("P_emaildomain",),
)

SECONDS_PER_DAY = 86400


def add_time_features(df: pd.DataFrame, time_col: str = TIME_COL) -> pd.DataFrame:
    """Hour-of-day and day-of-week, derived from the seconds offset.

    Bounded and periodic, so they carry over to a later window. The raw offset
    does not and is dropped before modelling.
    """
    out = df.copy()
    t = out[time_col]
    out["hour_of_day"] = ((t / 3600.0) % 24).astype(np.float32)
    out["day_of_week"] = ((t / SECONDS_PER_DAY) % 7).astype(np.float32)
    return out


def add_amount_features(df: pd.DataFrame, amount_col: str = AMOUNT) -> pd.DataFrame:
    """Log amount and the cents remainder.

    The cents remainder separates round-number amounts from precise ones, which
    behaves differently between card-testing traffic and ordinary retail.
    """
    out = df.copy()
    amt = out[amount_col].astype(np.float64)
    out["amt_log"] = np.log1p(amt).astype(np.float32)
    out["amt_cents"] = (amt - np.floor(amt)).round(4).astype(np.float32)
    out["amt_is_round"] = (out["amt_cents"] == 0).astype(np.int8)
    return out


@dataclass
class AggregateMaps:
    """Behavioural aggregates plus the training index they were fit from.

    Storing ``fit_index`` is what makes the leakage guard enforceable rather
    than aspirational: :func:`ftlab.leakage.assert_fit_window_is_train_only`
    can check the provenance instead of trusting the caller.
    """

    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    fit_index: pd.Index = field(default_factory=pd.Index)
    global_amount_mean: float = 0.0

    def column_names(self) -> list[str]:
        names: list[str] = []
        for prefix, tbl in self.tables.items():
            names.extend(tbl.columns.tolist())
            names.append(f"{prefix}__amt_ratio")
        return names


def _prefix(keys: tuple[str, ...]) -> str:
    return "grp_" + "_".join(keys)


def fit_behavioral_aggregates(
    train_df: pd.DataFrame,
    keys_list: tuple[tuple[str, ...], ...] = AGGREGATE_KEYS,
    amount_col: str = AMOUNT,
) -> AggregateMaps:
    """Compute per-group frequency and amount statistics from the training window only."""
    tables: dict[str, pd.DataFrame] = {}
    for keys in keys_list:
        if any(k not in train_df.columns for k in keys):
            continue
        p = _prefix(keys)
        g = train_df.groupby(list(keys), dropna=False, observed=True)[amount_col]
        tbl = pd.DataFrame(
            {
                f"{p}__count": g.size().astype(np.float32),
                f"{p}__amt_mean": g.mean().astype(np.float32),
                f"{p}__amt_std": g.std().astype(np.float32),
            }
        )
        tables[p] = tbl
    return AggregateMaps(
        tables=tables,
        fit_index=train_df.index,
        global_amount_mean=float(train_df[amount_col].mean()),
    )


def apply_behavioral_aggregates(
    df: pd.DataFrame,
    maps: AggregateMaps,
    keys_list: tuple[tuple[str, ...], ...] = AGGREGATE_KEYS,
    amount_col: str = AMOUNT,
) -> pd.DataFrame:
    """Attach fitted aggregates to any window.

    Groups never seen in training map to NaN, which is correct: a card first
    observed in the test period genuinely has no training history, and the model
    should be told that rather than handed an imputed average.
    """
    out = df.copy()
    for keys in keys_list:
        p = _prefix(keys)
        if p not in maps.tables:
            continue
        tbl = maps.tables[p]
        joined = out.join(tbl, on=list(keys))
        for c in tbl.columns:
            out[c] = joined[c].astype(np.float32)
        mean_col = f"{p}__amt_mean"
        with np.errstate(divide="ignore", invalid="ignore"):
            out[f"{p}__amt_ratio"] = (
                out[amount_col].astype(np.float32) / out[mean_col]
            ).replace([np.inf, -np.inf], np.nan).astype(np.float32)
    return out


def build_features(
    df: pd.DataFrame,
    maps: AggregateMaps,
    *,
    train_index: pd.Index | None = None,
) -> pd.DataFrame:
    """Full feature pipeline for one window.

    ``train_index`` is optional but recommended: passing it triggers the
    provenance check that the aggregates were fit on the training window.
    """
    if train_index is not None:
        assert_fit_window_is_train_only(maps.fit_index, train_index)
    out = add_time_features(df)
    out = add_amount_features(out)
    out = apply_behavioral_aggregates(out, maps)
    return out


def to_model_matrix(
    df: pd.DataFrame, categorical_cols: list[str] | None = None
) -> tuple[pd.DataFrame, list[str]]:
    """Drop non-features and coerce object columns to pandas ``category``.

    Returns the matrix and the list of categorical column names, which LightGBM
    consumes directly.
    """
    drop = [c for c in (TARGET, ID_COL, TIME_COL) if c in df.columns]
    X = df.drop(columns=drop)
    if categorical_cols is None:
        categorical_cols = X.select_dtypes(include="object").columns.tolist()
    for c in categorical_cols:
        if c in X.columns:
            X[c] = X[c].astype("category")
    return X, categorical_cols


def align_categories(
    train_X: pd.DataFrame, *others: pd.DataFrame
) -> tuple[pd.DataFrame, ...]:
    """Force later windows onto the training window's category vocabulary.

    Without this, a category present only in test gets its own code and silently
    means something different to the model than it did in training. Unseen
    levels become NaN, which LightGBM routes down its missing-value branch.
    """
    cat_cols = train_X.select_dtypes(include="category").columns
    aligned = []
    for other in others:
        o = other.copy()
        for c in cat_cols:
            if c in o.columns:
                o[c] = pd.Categorical(
                    o[c].astype(object), categories=train_X[c].cat.categories
                )
        aligned.append(o)
    return tuple(aligned)
