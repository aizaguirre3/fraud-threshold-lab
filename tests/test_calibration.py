"""Calibration metrics and post-hoc recalibration."""

from __future__ import annotations

import numpy as np
import pytest

from ftlab.calibration import (
    calibration_metrics,
    compare_at_fixed_threshold,
    compare_at_matched_alert_volume,
    expected_calibration_error,
    fit_isotonic,
    fit_platt,
    reliability_curve,
)


def test_a_perfectly_calibrated_model_has_near_zero_ece():
    rng = np.random.default_rng(7)
    p = rng.uniform(0.05, 0.95, size=40_000)
    y = rng.binomial(1, p)
    assert expected_calibration_error(y, p, n_bins=20) < 0.01


def test_a_systematically_overconfident_model_has_large_ece():
    """Predict 0.9 everywhere while the truth happens 30% of the time."""
    rng = np.random.default_rng(7)
    y = rng.binomial(1, 0.3, size=20_000)
    p = np.full(len(y), 0.9)
    assert expected_calibration_error(y, p, n_bins=10) == pytest.approx(0.6, abs=0.02)


def test_ece_is_non_negative(tiny_scores):
    y, scores, _ = tiny_scores
    assert expected_calibration_error(y, scores, n_bins=4) >= 0.0


def test_unknown_binning_strategy_is_rejected():
    with pytest.raises(ValueError, match="binning strategy"):
        expected_calibration_error(np.array([0, 1]), np.array([0.2, 0.8]),
                                   strategy="nonsense")


def test_reliability_curve_bins_cover_every_row():
    rng = np.random.default_rng(3)
    p = rng.uniform(0, 1, 5000)
    y = rng.binomial(1, p)
    curve = reliability_curve(y, p, n_bins=10)
    assert curve["count"].sum() == len(y)
    assert (curve["gap"] == curve["mean_predicted"] - curve["observed_rate"]).all()


def test_calibration_metrics_report_the_prediction_bias():
    y = np.array([0, 0, 0, 1])          # 25% observed
    p = np.array([0.9, 0.9, 0.9, 0.9])  # 90% predicted
    m = calibration_metrics(y, p, n_bins=2)
    assert m["mean_prediction_bias"] == pytest.approx(0.65)
    assert m["observed_rate"] == pytest.approx(0.25)


def test_platt_scaling_preserves_the_ranking():
    """Strictly monotone by construction -- this is why matched-volume FP counts
    cannot move, and the README says so."""
    rng = np.random.default_rng(11)
    s_val = rng.uniform(0, 1, 3000)
    y_val = rng.binomial(1, s_val * 0.4)
    platt = fit_platt(s_val, y_val)
    s_test = rng.uniform(0, 1, 800)
    out = platt.transform(s_test)
    assert np.array_equal(np.argsort(np.argsort(s_test)), np.argsort(np.argsort(out)))


def test_platt_improves_a_badly_scaled_score():
    rng = np.random.default_rng(5)
    n = 20_000
    true_p = rng.uniform(0.02, 0.5, n)
    y = rng.binomial(1, true_p)
    inflated = np.clip(true_p * 3.0, 0, 1)  # overconfident but correctly ranked
    platt = fit_platt(inflated, y)
    before = expected_calibration_error(y, inflated)
    after = expected_calibration_error(y, platt.transform(inflated))
    assert after < before


def test_isotonic_output_stays_in_the_unit_interval():
    rng = np.random.default_rng(9)
    s = rng.uniform(0, 1, 4000)
    y = rng.binomial(1, s * 0.5)
    out = fit_isotonic(s, y).transform(np.array([-5.0, 0.0, 0.5, 1.0, 5.0]))
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_isotonic_is_non_decreasing():
    rng = np.random.default_rng(9)
    s = rng.uniform(0, 1, 4000)
    y = rng.binomial(1, s * 0.5)
    out = fit_isotonic(s, y).transform(np.linspace(0, 1, 200))
    assert (np.diff(out) >= -1e-12).all()


def test_matched_alert_volume_holds_false_positives_flat_under_platt():
    """The documented negative result, asserted so it cannot silently change."""
    rng = np.random.default_rng(13)
    n = 6000
    y = rng.binomial(1, 0.05, n)
    raw = np.clip(rng.beta(2, 20, n) + y * 0.25, 0, 1)
    amounts = rng.uniform(10, 500, n)
    platt = fit_platt(raw, y)
    out = compare_at_matched_alert_volume(y, raw, platt.transform(raw), amounts, 300)
    assert out["raw"]["n_alerts"] == out["calibrated"]["n_alerts"]
    assert out["fp_reduction"] == 0


def test_fixed_threshold_comparison_detects_the_volume_change():
    """Where calibration does bite: a 0.5 cutoff read as a probability."""
    rng = np.random.default_rng(17)
    n = 5000
    y = rng.binomial(1, 0.05, n)
    overconfident = np.clip(rng.beta(2, 6, n) + y * 0.3, 0, 1)
    platt = fit_platt(overconfident, y)
    out = compare_at_fixed_threshold(y, overconfident, platt.transform(overconfident), 0.5)
    assert out["calibrated"]["n_alerts"] < out["raw"]["n_alerts"]
    assert out["fp_reduction"] > 0
