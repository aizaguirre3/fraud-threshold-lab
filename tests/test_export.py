"""Alert stream export.

The export is the contract with the downstream queue simulator, so the schema
and the attribution ranking are what these tests pin down. The string-typed
signal values matter more than they look: the top contributor for a row can be a
float or a category, and a mixed-type column cannot be written to parquet at
all -- that failure took a full pipeline run to surface.
"""

from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from ftlab.config import AMOUNT, ID_COL, TARGET, TIME_COL
from ftlab.export import (
    TOP_K_SIGNALS,
    build_alert_stream,
    top_shap_signals,
    write_alert_stream,
)


@pytest.fixture
def fitted(synthetic_frame):
    """A small LightGBM plus the frame and matrix it was fit on."""
    from ftlab.features import add_amount_features, add_time_features, to_model_matrix

    feats = add_amount_features(add_time_features(synthetic_frame))
    X, _ = to_model_matrix(feats)
    y = feats[TARGET].to_numpy()
    model = lgb.LGBMClassifier(n_estimators=25, num_leaves=7, verbose=-1,
                               random_state=0)
    model.fit(X, y)
    return model, feats, X


def test_signal_columns_are_emitted_for_each_rank(fitted):
    model, _, X = fitted
    signals, base = top_shap_signals(model, X)
    for i in range(1, TOP_K_SIGNALS + 1):
        for suffix in ("name", "value", "contribution"):
            assert f"signal_{i}_{suffix}" in signals.columns
    assert len(signals) == len(X)
    assert len(base) == len(X)


def test_signals_are_ranked_by_absolute_contribution(fitted):
    """Rank 1 must dominate rank 2 in magnitude, regardless of sign."""
    model, _, X = fitted
    signals, _ = top_shap_signals(model, X)
    for i in range(1, TOP_K_SIGNALS):
        a = signals[f"signal_{i}_contribution"].abs()
        b = signals[f"signal_{i + 1}_contribution"].abs()
        assert (a >= b - 1e-12).all()


def test_signal_names_are_real_feature_names(fitted):
    model, _, X = fitted
    signals, _ = top_shap_signals(model, X)
    assert set(signals["signal_1_name"]).issubset(set(X.columns))


def test_shap_contributions_reconstruct_the_model_margin(fitted):
    """TreeSHAP is additive: base value plus all contributions equals the raw
    margin. Checking that confirms the base column was split off correctly."""
    model, _, X = fitted
    contrib = np.asarray(model.predict(X, pred_contrib=True))
    margin = model.predict(X, raw_score=True)
    np.testing.assert_allclose(contrib.sum(axis=1), margin, rtol=1e-6, atol=1e-6)


def test_signal_values_are_strings_so_parquet_can_type_the_column(fitted):
    """The top contributor may be numeric for one row and categorical for the
    next; a mixed-dtype object column is unwritable."""
    model, feats, X = fitted
    stream = build_alert_stream(
        feats, X, model, {"score_raw": np.zeros(len(X))}
    )
    for i in range(1, TOP_K_SIGNALS + 1):
        col = stream[f"signal_{i}_value"]
        assert all(isinstance(v, str) for v in col.dropna())


def test_alert_stream_round_trips_through_parquet(fitted, tmp_path):
    """The regression test for the failure that only appeared on real data."""
    model, feats, X = fitted
    stream = build_alert_stream(feats, X, model, {"score_raw": np.zeros(len(X))})
    path, meta_path = write_alert_stream(stream, {"source_project": "test"}, tmp_path)
    back = pd.read_parquet(path)
    assert len(back) == len(stream)
    assert list(back.columns) == list(stream.columns)


def test_alert_stream_carries_the_downstream_contract_columns(fitted):
    model, feats, X = fitted
    stream = build_alert_stream(feats, X, model, {"score_raw": np.zeros(len(X))})
    for col in ("transaction_id", "transaction_dt", "amount", "is_fraud",
                "score_raw", "shap_base_value"):
        assert col in stream.columns


def test_every_scored_transaction_is_exported_not_just_alerts(fitted):
    """The consumer picks its own threshold and auto-close band, and cannot
    explore either if the stream arrives pre-filtered."""
    model, feats, X = fitted
    stream = build_alert_stream(feats, X, model, {"score_raw": np.zeros(len(X))})
    assert len(stream) == len(feats)


def test_alert_stream_is_sorted_by_arrival_time(fitted):
    model, feats, X = fitted
    stream = build_alert_stream(feats, X, model, {"score_raw": np.zeros(len(X))})
    assert (stream["transaction_dt"].diff().dropna() >= 0).all()


def test_multiple_score_variants_are_carried(fitted):
    model, feats, X = fitted
    scores = {
        "score_raw": np.linspace(0, 1, len(X)),
        "score_isotonic": np.linspace(0, 0.5, len(X)),
    }
    stream = build_alert_stream(feats, X, model, scores)
    assert "score_raw" in stream.columns
    assert "score_isotonic" in stream.columns


def test_metadata_sidecar_records_stream_provenance(fitted, tmp_path):
    model, feats, X = fitted
    stream = build_alert_stream(feats, X, model, {"score_raw": np.zeros(len(X))})
    _, meta_path = write_alert_stream(
        stream, {"source_project": "fraud-threshold-lab", "operating_threshold": 0.03},
        tmp_path,
    )
    meta = json.loads(meta_path.read_text())
    assert meta["source_project"] == "fraud-threshold-lab"
    assert meta["operating_threshold"] == 0.03
    assert meta["n_rows"] == len(stream)
    assert meta["n_fraud"] == int(stream["is_fraud"].sum())
    assert meta["fraud_amount"] <= meta["total_amount"]
