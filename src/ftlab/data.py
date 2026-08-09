"""Loading, joining, and profiling the IEEE-CIS transaction and identity tables."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import AMOUNT, DATA_RAW, ID_COL, TARGET, TIME_COL

TRANSACTION_FILE = "train_transaction.csv"
IDENTITY_FILE = "train_identity.csv"

#: Marks rows where the identity table had no matching record. The IEEE-CIS
#: identity table covers only about a quarter of transactions, and *which*
#: transactions carry identity data is itself informative -- it reflects the
#: capture path the payment took. Dropping unmatched rows would discard 76% of
#: the data and the signal in the missingness along with it.
IDENTITY_PRESENT = "identity_present"


def _downcast(df: pd.DataFrame) -> pd.DataFrame:
    """float64 -> float32, int64 -> smallest safe int. Halves memory with no
    material precision cost for this feature set."""
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_float_dtype(s):
            df[col] = s.astype(np.float32)
        elif pd.api.types.is_integer_dtype(s) and col != ID_COL:
            df[col] = pd.to_numeric(s, downcast="integer")
    return df


def load_raw(data_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the transaction and identity CSVs."""
    d = Path(data_dir) if data_dir is not None else DATA_RAW
    tx_path = d / TRANSACTION_FILE
    id_path = d / IDENTITY_FILE
    missing = [p.name for p in (tx_path, id_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing raw data file(s) {missing} in {d}. Run `make data` "
            "(requires a Kaggle API token and acceptance of the IEEE-CIS "
            "competition rules)."
        )
    tx = _downcast(pd.read_csv(tx_path))
    ident = _downcast(pd.read_csv(id_path))
    return tx, ident


def join_transaction_identity(
    tx: pd.DataFrame, ident: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    """Left-join identity onto transaction, preserving unmatched rows.

    Returns the joined frame and a coverage report. The join is a left join on
    purpose: identity coverage is partial and non-random, so an inner join would
    both shrink the dataset and bias it toward one capture path.
    """
    n_tx = len(tx)
    matched_ids = set(ident[ID_COL])
    df = tx.merge(ident, on=ID_COL, how="left", validate="one_to_one")
    df[IDENTITY_PRESENT] = df[ID_COL].isin(matched_ids).astype(np.int8)

    n_matched = int(df[IDENTITY_PRESENT].sum())
    fraud_with_id = float(df.loc[df[IDENTITY_PRESENT] == 1, TARGET].mean())
    fraud_without_id = float(df.loc[df[IDENTITY_PRESENT] == 0, TARGET].mean())

    report = {
        "n_transactions": n_tx,
        "n_identity_rows": len(ident),
        "n_matched": n_matched,
        "identity_coverage_pct": round(100.0 * n_matched / n_tx, 2),
        "fraud_rate_with_identity_pct": round(100.0 * fraud_with_id, 3),
        "fraud_rate_without_identity_pct": round(100.0 * fraud_without_id, 3),
        "identity_missingness_is_signal": bool(
            abs(fraud_with_id - fraud_without_id) > 0.005
        ),
    }
    return df, report


def profile(df: pd.DataFrame) -> dict:
    """Class balance, missingness, cardinality, and time span."""
    n = len(df)
    n_fraud = int(df[TARGET].sum())
    miss = df.isna().mean().sort_values(ascending=False)
    obj_cols = df.select_dtypes(include="object").columns
    card = {c: int(df[c].nunique(dropna=True)) for c in obj_cols}

    dt = df[TIME_COL]
    amt = df[AMOUNT]
    fraud_amt = df.loc[df[TARGET] == 1, AMOUNT]

    return {
        "n_rows": n,
        "n_columns": int(df.shape[1]),
        "n_fraud": n_fraud,
        "fraud_rate_pct": round(100.0 * n_fraud / n, 3),
        "imbalance_ratio": round((n - n_fraud) / max(n_fraud, 1), 1),
        "time_span_days": round(float(dt.max() - dt.min()) / 86400.0, 1),
        "time_min_raw": int(dt.min()),
        "time_max_raw": int(dt.max()),
        "columns_over_50pct_missing": int((miss > 0.50).sum()),
        "columns_over_90pct_missing": int((miss > 0.90).sum()),
        "mean_missingness_pct": round(100.0 * float(miss.mean()), 2),
        "categorical_columns": len(obj_cols),
        "max_categorical_cardinality": max(card.values()) if card else 0,
        "total_amount": round(float(amt.sum()), 2),
        "total_fraud_amount": round(float(fraud_amt.sum()), 2),
        "fraud_amount_share_pct": round(100.0 * float(fraud_amt.sum() / amt.sum()), 3),
        "median_amount": round(float(amt.median()), 2),
        "median_fraud_amount": round(float(fraud_amt.median()), 2),
    }


def load_joined(data_dir: Path | None = None) -> tuple[pd.DataFrame, dict]:
    """Convenience: load, join, and profile in one call."""
    tx, ident = load_raw(data_dir)
    df, join_report = join_transaction_identity(tx, ident)
    report = {"join": join_report, "profile": profile(df)}
    return df, report


def write_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str))
