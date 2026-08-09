"""Cost-sensitive selection.

The cost logic decides the recommended operating point, so these tests check
the arithmetic against values computed independently rather than trusting the
module to agree with itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from ftlab.config import CostConfig
from ftlab.cost import (
    add_cost_columns,
    compare_cost_vs_f1,
    cost_ratio_sensitivity,
    optimal_threshold,
    sensitivity_summary,
)
from ftlab.threshold import sweep_thresholds


def test_review_cost_per_alert_matches_hourly_rate_and_handling_time():
    c = CostConfig(analyst_hourly_cost=35.0, minutes_per_review=4.0)
    assert c.review_cost_per_alert == pytest.approx(35.0 * 4 / 60)


def test_cost_per_false_positive_adds_labour_and_friction():
    c = CostConfig(analyst_hourly_cost=35.0, minutes_per_review=4.0,
                   friction_cost_per_fp=5.0)
    assert c.cost_per_false_positive == pytest.approx(35.0 * 4 / 60 + 5.0)


def test_total_cost_is_missed_fraud_dollars_plus_false_alarm_cost(tiny_scores):
    y, scores, amounts = tiny_scores
    c = CostConfig()
    sweep = sweep_thresholds(y, scores, amounts, c, n_points=80)
    scored = add_cost_columns(sweep, c)
    expected = scored["fraud_missed_dollars"] + scored["fp"] * c.cost_per_false_positive
    np.testing.assert_allclose(scored["total_cost"], expected, rtol=1e-12)


def test_false_negative_cost_uses_actual_amounts_not_a_constant():
    """Two datasets identical except in fraud amount must differ in FN cost."""
    y = np.array([1, 0, 0, 0])
    scores = np.array([0.1, 0.9, 0.8, 0.7])  # the fraud is ranked last, so it is missed
    cheap = sweep_thresholds(y, scores, np.array([10.0, 1, 1, 1]),
                             thresholds=np.array([0.5]))
    dear = sweep_thresholds(y, scores, np.array([10_000.0, 1, 1, 1]),
                            thresholds=np.array([0.5]))
    c = CostConfig()
    assert add_cost_columns(cheap, c).iloc[0]["fn_cost"] == pytest.approx(10.0)
    assert add_cost_columns(dear, c).iloc[0]["fn_cost"] == pytest.approx(10_000.0)


def test_optimal_threshold_actually_minimises_total_cost(tiny_scores):
    y, scores, amounts = tiny_scores
    c = CostConfig()
    sweep = sweep_thresholds(y, scores, amounts, c, n_points=200)
    best = optimal_threshold(sweep, c)
    assert best["total_cost"] == pytest.approx(add_cost_columns(sweep, c)["total_cost"].min())


def test_raising_false_positive_cost_never_lowers_the_optimal_threshold(tiny_scores):
    """Monotonicity: if false alarms get more expensive, you do not alert more."""
    y, scores, amounts = tiny_scores
    sweep = sweep_thresholds(y, scores, amounts, n_points=300)
    sens = cost_ratio_sensitivity(sweep, fp_costs=(1.0, 5.0, 25.0, 100.0, 500.0))
    assert (sens["threshold"].diff().dropna() >= -1e-12).all()
    assert (sens["n_alerts"].diff().dropna() <= 0).all()


def test_a_negligible_false_positive_cost_pushes_toward_alerting_on_everything():
    y = np.array([1, 0, 1, 0, 0, 0, 1, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
    amounts = np.array([500.0, 10, 500, 10, 10, 10, 500, 10])
    sweep = sweep_thresholds(y, scores, amounts, n_points=300)
    cheap = optimal_threshold(sweep, cost_per_fp=0.001)
    dear = optimal_threshold(sweep, cost_per_fp=10_000.0)
    assert cheap["recall"] >= dear["recall"]
    assert cheap["n_alerts"] >= dear["n_alerts"]


def test_compare_cost_vs_f1_reports_a_non_negative_penalty(tiny_scores):
    """The F1 point cannot beat the cost point on cost -- that is what optimal means."""
    y, scores, amounts = tiny_scores
    sweep = sweep_thresholds(y, scores, amounts, n_points=300)
    cmp = compare_cost_vs_f1(sweep)
    assert cmp["cost_penalty_of_using_f1_threshold"] >= -1e-9
    assert cmp["cost_optimal"]["total_cost"] <= cmp["f1_optimal"]["total_cost"] + 1e-9


def test_comparison_records_the_assumptions_it_used():
    """A reported optimum is meaningless without the assumptions behind it."""
    y = np.array([1, 0, 1, 0])
    sweep = sweep_thresholds(y, np.array([0.9, 0.6, 0.4, 0.1]), np.ones(4))
    cmp = compare_cost_vs_f1(sweep, CostConfig(analyst_hourly_cost=42.0))
    assert cmp["assumptions"]["analyst_hourly_cost"] == 42.0
    assert cmp["assumptions"]["false_negative_cost"] == "actual transaction amount"


def test_sensitivity_summary_reports_the_span_of_the_recommendation(tiny_scores):
    y, scores, amounts = tiny_scores
    sweep = sweep_thresholds(y, scores, amounts, n_points=300)
    summary = sensitivity_summary(cost_ratio_sensitivity(sweep))
    lo, hi = summary["threshold_range"]
    assert lo <= hi
    assert summary["fp_cost_fold_range"] > 1.0
