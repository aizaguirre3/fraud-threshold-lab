"""Export a self-contained alert stream for downstream consumers.

The downstream queue-simulation project (``alert-triage-sim``) needs scored
alerts with timestamps, amounts, outcomes, and an explanation of *why* each
alert scored high. It should not need the 710 MB Kaggle download, a Kaggle
account, or this project's feature pipeline to do that. So everything it needs
is written to one file here.

Attributions come from LightGBM's ``pred_contrib=True``, which returns exact
TreeSHAP values. That avoids taking a dependency on the ``shap`` package and is
faster, since the contributions are computed during prediction rather than
reconstructed afterwards.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import AMOUNT, ID_COL, TARGET, TIME_COL

#: How many contributing signals to record per transaction. Five is what fits in
#: a reviewer's field of view without scrolling, which is the constraint that
#: matters for the interface downstream.
TOP_K_SIGNALS = 5


def top_shap_signals(
    model, X: pd.DataFrame, top_k: int = TOP_K_SIGNALS
) -> tuple[pd.DataFrame, np.ndarray]:
    """Per-row top-``k`` feature attributions, ranked by absolute contribution.

    Returns a frame with ``signal_{i}_name`` / ``_value`` / ``_contribution``
    columns, plus the model's base (expected) value per row.
    """
    contrib = model.predict(X, pred_contrib=True)
    contrib = np.asarray(contrib)
    # LightGBM appends the base value as a final column.
    base = contrib[:, -1]
    contrib = contrib[:, :-1]

    feature_names = np.asarray(X.columns)
    order = np.argsort(-np.abs(contrib), axis=1)[:, :top_k]

    out: dict[str, np.ndarray] = {}
    rows = np.arange(len(X))
    values = X.to_numpy(dtype=object)
    for i in range(top_k):
        idx = order[:, i]
        # Feature values are rendered as strings. The top contributor for a
        # given row can be a float (TransactionAmt) or a category
        # (DeviceInfo = "ALE-L23 Build/HuaweiALE-L23"), so a single column holds
        # both types and no numeric dtype can represent it. These values exist
        # to be displayed to a reviewer, never to be computed on.
        vals = pd.Series(values[rows, idx], dtype=object)
        out[f"signal_{i + 1}_name"] = feature_names[idx]
        out[f"signal_{i + 1}_value"] = (
            vals.where(vals.notna(), None).astype("string").to_numpy()
        )
        out[f"signal_{i + 1}_contribution"] = contrib[rows, idx]
    return pd.DataFrame(out, index=X.index), base


def build_alert_stream(
    test_df: pd.DataFrame,
    X_test: pd.DataFrame,
    model,
    scores: dict[str, np.ndarray],
    top_k: int = TOP_K_SIGNALS,
) -> pd.DataFrame:
    """Assemble the full scored stream for the test window.

    Every scored transaction is exported, not only those above P1's operating
    threshold. The downstream simulator chooses its own alert cutoff and its own
    auto-disposition band, and it cannot explore either if the stream has
    already been filtered.
    """
    signals, base = top_shap_signals(model, X_test, top_k=top_k)

    stream = pd.DataFrame(
        {
            "transaction_id": test_df[ID_COL].to_numpy(),
            "transaction_dt": test_df[TIME_COL].to_numpy(),
            "amount": test_df[AMOUNT].to_numpy(dtype=float),
            "is_fraud": test_df[TARGET].to_numpy(dtype=int),
        },
        index=test_df.index,
    )
    for name, arr in scores.items():
        stream[name] = np.asarray(arr, dtype=float)
    stream["shap_base_value"] = base

    # A few interpretable descriptors so the reviewer interface can show context
    # without the consumer needing the feature pipeline.
    for col in ("ProductCD", "card4", "card6", "DeviceType", "identity_present"):
        if col in test_df.columns:
            stream[col] = test_df[col].astype(object).to_numpy()

    stream = pd.concat([stream, signals], axis=1)
    return stream.sort_values("transaction_dt").reset_index(drop=True)


def write_alert_stream(
    stream: pd.DataFrame, metadata: dict, out_dir: Path
) -> tuple[Path, Path]:
    """Write the stream as parquet plus a JSON sidecar describing its provenance.

    The sidecar carries the operating threshold, cost assumptions, and model
    metrics so the consumer can state where its inputs came from rather than
    restating them from memory.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    stream_path = out_dir / "alert_stream.parquet"
    meta_path = out_dir / "alert_stream_metadata.json"

    stream.to_parquet(stream_path, index=False)
    meta = dict(metadata)
    meta["n_rows"] = int(len(stream))
    meta["n_fraud"] = int(stream["is_fraud"].sum())
    meta["total_amount"] = float(stream["amount"].sum())
    meta["fraud_amount"] = float(stream.loc[stream["is_fraud"] == 1, "amount"].sum())
    meta["dt_min"] = int(stream["transaction_dt"].min())
    meta["dt_max"] = int(stream["transaction_dt"].max())
    meta["window_days"] = round(
        (float(stream["transaction_dt"].max()) - float(stream["transaction_dt"].min()))
        / 86400.0,
        2,
    )
    meta["columns"] = list(stream.columns)
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    return stream_path, meta_path
