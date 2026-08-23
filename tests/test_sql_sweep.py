"""Cross-implementation verification of the threshold sweep.

The sweep is the computation every operational number is read off, so it is
implemented three times: Python (canonical), SQL (here, executed), and SAS
(``sas/threshold_sweep.sas``, not executable on this machine). Two independent
implementations agreeing on the same data is stronger evidence than either
passing its own unit tests.

These tests build their own scored CSV rather than using the Kaggle-derived
one, so they run in CI where that data does not exist.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ftlab.config import CostConfig
from ftlab.sqlsweep import (
    compare_with_python,
    cost_optimal_row,
    read_scored,
    render_sql,
    run_sql_sweep,
)
from ftlab.threshold import sweep_thresholds

duckdb = pytest.importorskip("duckdb")


@pytest.fixture
def scored_csv(tmp_path):
    """A scored file shaped like the pipeline's output."""
    rng = np.random.default_rng(20260810)
    n = 4000
    y = rng.binomial(1, 0.05, n)
    score = np.clip(rng.beta(1.5, 20, n) + y * 0.25, 1e-6, 1 - 1e-6)
    amount = np.round(rng.lognormal(4.0, 1.1, n), 2)
    path = tmp_path / "test_scores.csv"
    pd.DataFrame(
        {
            "TransactionID": np.arange(n),
            "TransactionDT": np.sort(rng.integers(0, 2_000_000, n)),
            "TransactionAmt": amount,
            "isFraud": y,
            "score_raw": score,
        }
    ).to_csv(path, index=False)
    return path


def test_sql_and_python_agree_at_identical_thresholds(scored_csv):
    """The headline cross-check."""
    out = compare_with_python(scored_csv, n_checks=25)
    assert out["all_match"], out["checks"]
    assert out["thresholds_compared"] == 25


def test_every_confusion_column_agrees(scored_csv):
    out = compare_with_python(scored_csv, n_checks=15)
    for field in ("n_alerts", "tp", "fp", "fn", "precision", "recall",
                  "fraud_caught_dollars"):
        assert out["checks"][field], field


def test_sql_produces_more_operating_points_than_the_python_grid(scored_csv):
    """SQL walks every achievable cut; Python samples a grid."""
    out = compare_with_python(scored_csv, n_checks=5)
    assert out["sql_operating_points"] > out["python_grid_points"]


def test_sql_confusion_counts_sum_to_the_row_count(scored_csv):
    sweep = run_sql_sweep(scored_csv)
    total = sweep["tp"] + sweep["fp"] + sweep["fn"] + sweep["tn"]
    assert (total == 4000).all()


def test_sql_alert_volume_equals_tp_plus_fp(scored_csv):
    sweep = run_sql_sweep(scored_csv)
    assert (sweep["n_alerts"] == sweep["tp"] + sweep["fp"]).all()


def test_sql_recall_is_monotone_in_alert_volume(scored_csv):
    sweep = run_sql_sweep(scored_csv).sort_values("n_alerts")
    assert (sweep["recall"].diff().dropna() >= -1e-12).all()


def test_sql_total_cost_matches_its_definition(scored_csv):
    cost = CostConfig()
    sweep = run_sql_sweep(scored_csv, cost)
    expected = sweep["fraud_missed_dollars"] + sweep["fp"] * cost.cost_per_false_positive
    np.testing.assert_allclose(sweep["total_cost"], expected, rtol=1e-9)


def test_sql_review_hours_follow_handling_time(scored_csv):
    sweep = run_sql_sweep(scored_csv, CostConfig(minutes_per_review=6.0))
    np.testing.assert_allclose(
        sweep["review_hours"], sweep["n_alerts"] * 0.1, rtol=1e-9
    )


def test_sql_cost_optimum_is_at_least_as_good_as_the_python_grid(scored_csv):
    """SQL evaluates every cut point, so it cannot find a worse optimum than a
    501-point sample of the same curve."""
    cost = CostConfig()
    scored = read_scored(scored_csv)
    sql_best = cost_optimal_row(run_sql_sweep(scored_csv, cost))
    py = sweep_thresholds(
        scored["isFraud"].to_numpy(),
        scored["score_raw"].to_numpy(),
        scored["TransactionAmt"].to_numpy(dtype=float),
        cost,
        n_points=500,
    )
    py_cost = (py["fraud_missed_dollars"] + py["fp"] * cost.cost_per_false_positive).min()
    assert sql_best["total_cost"] <= py_cost + 1e-6


def test_no_tied_scores_share_a_cut_point(scored_csv):
    """A threshold admits every row sharing that score, so each emitted cut must
    be a distinct score."""
    sweep = run_sql_sweep(scored_csv)
    assert sweep["threshold"].is_unique


def test_render_sql_substitutes_every_placeholder(scored_csv):
    sql = render_sql(scored_csv, CostConfig())
    for placeholder in ("{scored}", "{cost_fp}", "{minutes_review}", "{score_col}"):
        assert placeholder not in sql


def test_round_trip_parsing_is_required_for_agreement(scored_csv):
    """Documents the trap: pandas' default CSV float parser is not correctly
    rounded, and the resulting last-bit differences look exactly like an
    off-by-one bug in the sweep."""
    default = np.sort(pd.read_csv(scored_csv)["score_raw"].to_numpy())
    exact = np.sort(read_scored(scored_csv)["score_raw"].to_numpy())
    con = duckdb.connect()
    try:
        dd = con.execute(
            f"SELECT score_raw FROM read_csv_auto('{scored_csv}') ORDER BY score_raw"
        ).fetch_df()["score_raw"].to_numpy()
    finally:
        con.close()
    assert np.array_equal(exact, dd)
    # The default parser is close but not bit-identical.
    assert np.allclose(default, dd, rtol=1e-12)
