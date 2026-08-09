"""Threshold-sweep correctness.

These are the tests that matter most: every operational claim in the README is
read off this table, so an error here would propagate into every reported
number. The confusion counts are checked against values worked out by hand
rather than against the implementation's own output.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ftlab.config import CostConfig
from ftlab.threshold import (
    best_f1_row,
    operational_table,
    sweep_thresholds,
    threshold_at_alert_volume,
    threshold_for_capacity,
)


def test_confusion_counts_are_exact_at_a_known_threshold(tiny_scores):
    y, scores, amounts = tiny_scores
    # At t=0.65 the flagged set is the top four scores: labels 1,1,0,1.
    sweep = sweep_thresholds(y, scores, amounts, thresholds=np.array([0.65]))
    row = sweep.iloc[0]
    assert row["n_alerts"] == 4
    assert row["tp"] == 3
    assert row["fp"] == 1
    assert row["fn"] == 1  # four positives total, three caught
    assert row["tn"] == 5


def test_confusion_counts_sum_to_n_at_every_threshold(tiny_scores):
    y, scores, amounts = tiny_scores
    sweep = sweep_thresholds(y, scores, amounts, n_points=60)
    total = sweep["tp"] + sweep["fp"] + sweep["fn"] + sweep["tn"]
    assert (total == len(y)).all()


def test_alert_volume_equals_tp_plus_fp(tiny_scores):
    y, scores, amounts = tiny_scores
    sweep = sweep_thresholds(y, scores, amounts, n_points=60)
    assert (sweep["n_alerts"] == sweep["tp"] + sweep["fp"]).all()


def test_precision_and_recall_match_their_definitions(tiny_scores):
    y, scores, amounts = tiny_scores
    sweep = sweep_thresholds(y, scores, amounts, n_points=60)
    s = sweep[sweep["n_alerts"] > 0]
    np.testing.assert_allclose(s["precision"], s["tp"] / s["n_alerts"], rtol=1e-12)
    np.testing.assert_allclose(s["recall"], s["tp"] / y.sum(), rtol=1e-12)


def test_false_alarm_rate_is_one_minus_precision(tiny_scores):
    y, scores, amounts = tiny_scores
    sweep = sweep_thresholds(y, scores, amounts, n_points=60)
    s = sweep[sweep["n_alerts"] > 0]
    np.testing.assert_allclose(s["false_alarm_rate"], 1.0 - s["precision"], rtol=1e-12)


def test_dollar_columns_are_exact(tiny_scores):
    y, scores, amounts = tiny_scores
    # Top four flagged: amounts 100, 200, 50, 300; fraud among them 100+200+300=600.
    sweep = sweep_thresholds(y, scores, amounts, thresholds=np.array([0.65]))
    row = sweep.iloc[0]
    assert row["fraud_caught_dollars"] == pytest.approx(600.0)
    total_fraud = amounts[y == 1].sum()  # 100+200+300+400 = 1000
    assert row["fraud_missed_dollars"] == pytest.approx(total_fraud - 600.0)
    assert row["dollar_recall"] == pytest.approx(600.0 / 1000.0)


def test_review_hours_follow_the_configured_handling_time(tiny_scores):
    y, scores, amounts = tiny_scores
    four = sweep_thresholds(y, scores, amounts, CostConfig(minutes_per_review=4.0),
                            thresholds=np.array([0.65]))
    eight = sweep_thresholds(y, scores, amounts, CostConfig(minutes_per_review=8.0),
                             thresholds=np.array([0.65]))
    assert four.iloc[0]["review_hours"] == pytest.approx(4 * 4 / 60)
    # Doubling handling time doubles the staffing requirement at fixed volume.
    assert eight.iloc[0]["review_hours"] == pytest.approx(2 * four.iloc[0]["review_hours"])


def test_recall_is_monotone_non_increasing_in_threshold(tiny_scores):
    y, scores, amounts = tiny_scores
    sweep = sweep_thresholds(y, scores, amounts, n_points=200).sort_values("threshold")
    assert (sweep["recall"].diff().dropna() <= 1e-12).all()


def test_alert_volume_is_monotone_non_increasing_in_threshold(tiny_scores):
    y, scores, amounts = tiny_scores
    sweep = sweep_thresholds(y, scores, amounts, n_points=200).sort_values("threshold")
    assert (sweep["n_alerts"].diff().dropna() <= 0).all()


def test_threshold_zero_flags_everything_and_above_one_flags_nothing(tiny_scores):
    y, scores, amounts = tiny_scores
    sweep = sweep_thresholds(y, scores, amounts, n_points=40)
    assert sweep["n_alerts"].max() == len(y)
    assert sweep["n_alerts"].min() == 0


def test_ties_at_the_boundary_are_all_flagged_together():
    """Rows sharing the boundary score must fall on the same side of it."""
    y = np.array([1, 0, 1, 0])
    scores = np.array([0.5, 0.5, 0.5, 0.1])
    amounts = np.ones(4)
    sweep = sweep_thresholds(y, scores, amounts, thresholds=np.array([0.5]))
    assert sweep.iloc[0]["n_alerts"] == 3


def test_length_mismatch_is_rejected():
    with pytest.raises(ValueError, match="Length mismatch"):
        sweep_thresholds(np.array([0, 1]), np.array([0.1]), np.array([1.0, 2.0]))


def test_best_f1_row_maximises_f1(tiny_scores):
    y, scores, amounts = tiny_scores
    sweep = sweep_thresholds(y, scores, amounts, n_points=200)
    row = best_f1_row(sweep)
    assert row["f1"] == pytest.approx(sweep["f1"].max())


def test_threshold_at_alert_volume_finds_the_closest_row(tiny_scores):
    y, scores, amounts = tiny_scores
    sweep = sweep_thresholds(y, scores, amounts, n_points=200)
    row = threshold_at_alert_volume(sweep, 5)
    assert abs(row["n_alerts"] - 5) <= abs(sweep["n_alerts"] - 5).min()


def test_threshold_for_capacity_respects_the_hour_budget(tiny_scores):
    y, scores, amounts = tiny_scores
    budget = 0.2  # hours -> 3 reviews at 4 minutes each
    row = threshold_for_capacity(sweep_thresholds(y, scores, amounts, n_points=200), budget)
    assert row["review_hours"] <= budget + 1e-12


def test_operational_table_hits_each_recall_target(tiny_scores):
    y, scores, amounts = tiny_scores
    sweep = sweep_thresholds(y, scores, amounts, n_points=200)
    table = operational_table(sweep, recall_targets=(0.25, 0.5, 0.75))
    assert len(table) == 3
    for _, r in table.iterrows():
        assert r["recall"] >= r["recall_target"] - 1e-12


def test_operational_table_picks_the_cheapest_row_meeting_the_target(tiny_scores):
    y, scores, amounts = tiny_scores
    sweep = sweep_thresholds(y, scores, amounts, n_points=200)
    table = operational_table(sweep, recall_targets=(0.5,))
    row = table.iloc[0]
    cheaper = sweep[(sweep["recall"] >= 0.5) & (sweep["n_alerts"] < row["n_alerts"])]
    assert cheaper.empty


def test_perfect_scores_give_perfect_precision_and_recall():
    y = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    sweep = sweep_thresholds(y, scores, np.ones(4), thresholds=np.array([0.5]))
    row = sweep.iloc[0]
    assert row["precision"] == pytest.approx(1.0)
    assert row["recall"] == pytest.approx(1.0)
    assert row["fp"] == 0
