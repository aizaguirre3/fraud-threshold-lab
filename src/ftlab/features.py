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

from .config import AMOUNT, ID_COL, RANDOM_SEED, TARGET, TIME_COL
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


@dataclass
class TargetEncoderMaps:
    """Smoothed per-group fraud rates, plus out-of-fold values for training rows.

    Target encoding is the highest-value feature family available here and the
    only one that touches the label directly, so it needs two protections that
    the count/amount aggregates do not:

    * **Fit on the training window only.** Same provenance rule as the other
      aggregates, enforced the same way -- but here it actually bites. A card's
      mean amount barely moves when test rows contaminate it; a card's *fraud
      rate* computed over the test period is close to the answer itself.
    * **Out-of-fold within training.** A training row must not contribute to the
      encoding it is scored with. Without this the model reads each row's own
      label back out of the feature and the fit collapses onto it: in-sample
      performance looks superb and none of it transfers.

    Smoothing pulls small groups toward the global prior, so a card seen once
    with one fraud does not get an encoded rate of 1.0.
    """

    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    oof: pd.DataFrame = field(default_factory=pd.DataFrame)
    fit_index: pd.Index = field(default_factory=pd.Index)
    prior: float = 0.0
    smoothing: float = 20.0

    def column_names(self) -> list[str]:
        return [f"te_{p}" for p in self.tables]


def _smoothed_rate(
    counts: pd.Series, sums: pd.Series, prior: float, smoothing: float
) -> pd.Series:
    """Additive (Bayesian) smoothing toward the global fraud rate."""
    return (sums + smoothing * prior) / (counts + smoothing)


def fit_target_encoders(
    train_df: pd.DataFrame,
    keys_list: tuple[tuple[str, ...], ...] = AGGREGATE_KEYS,
    target: str = TARGET,
    smoothing: float = 20.0,
    n_folds: int = 5,
    seed: int = RANDOM_SEED,
) -> TargetEncoderMaps:
    """Fit smoothed target encodings, with out-of-fold values for the train rows."""
    from sklearn.model_selection import KFold

    y = train_df[target]
    prior = float(y.mean())

    tables: dict[str, pd.DataFrame] = {}
    for keys in keys_list:
        if any(k not in train_df.columns for k in keys):
            continue
        p = _prefix(keys)
        g = train_df.groupby(list(keys), dropna=False, observed=True)[target]
        rate = _smoothed_rate(g.size(), g.sum(), prior, smoothing)
        tables[p] = rate.to_frame(name=f"te_{p}").astype(np.float32)

    oof = pd.DataFrame(index=train_df.index)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for keys in keys_list:
        p = _prefix(keys)
        if p not in tables:
            continue
        col = pd.Series(np.nan, index=train_df.index, dtype=np.float32)
        for fit_pos, held_pos in kf.split(train_df):
            fit_rows = train_df.iloc[fit_pos]
            held_rows = train_df.iloc[held_pos]
            g = fit_rows.groupby(list(keys), dropna=False, observed=True)[target]
            fold_prior = float(fit_rows[target].mean())
            rate = _smoothed_rate(g.size(), g.sum(), fold_prior, smoothing)
            mapped = held_rows.join(rate.to_frame(name="_r"), on=list(keys))["_r"]
            col.loc[held_rows.index] = mapped.to_numpy(dtype=np.float32)
        oof[f"te_{p}"] = col

    return TargetEncoderMaps(
        tables=tables, oof=oof, fit_index=train_df.index,
        prior=prior, smoothing=smoothing,
    )


def apply_target_encoders(
    df: pd.DataFrame,
    maps: TargetEncoderMaps,
    train_index: pd.Index,
    keys_list: tuple[tuple[str, ...], ...] = AGGREGATE_KEYS,
) -> pd.DataFrame:
    """Attach target encodings: out-of-fold for training rows, full-fit elsewhere.

    Groups unseen during training stay NaN rather than falling back to the prior.
    A card with no training history genuinely has no observed fraud rate, and
    LightGBM handles the missing branch better than a silently imputed average.
    """
    assert_fit_window_is_train_only(maps.fit_index, train_index)
    out = df.copy()
    train_pos = out.index.intersection(train_index)

    for keys in keys_list:
        p = _prefix(keys)
        if p not in maps.tables:
            continue
        name = f"te_{p}"
        joined = out.join(maps.tables[p], on=list(keys))[name]
        out[name] = joined.astype(np.float32)
        # Training rows are overwritten with their out-of-fold value so no row
        # is scored using an encoding it helped produce.
        if len(train_pos) and name in maps.oof.columns:
            out.loc[train_pos, name] = maps.oof.loc[train_pos, name].to_numpy(
                dtype=np.float32
            )
    return out


def build_features(
    df: pd.DataFrame,
    maps: AggregateMaps,
    *,
    train_index: pd.Index | None = None,
    target_maps: "TargetEncoderMaps | None" = None,
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
    if target_maps is not None:
        if train_index is None:
            raise ValueError(
                "train_index is required when target_maps is supplied: target "
                "encodings need it to decide which rows get out-of-fold values."
            )
        out = apply_target_encoders(out, target_maps, train_index)
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
