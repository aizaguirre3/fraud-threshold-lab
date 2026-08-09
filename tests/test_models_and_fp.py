"""Discrimination metrics, imbalance handling, and false-positive characterization."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ftlab.config import AMOUNT, TARGET
from ftlab.fpanalysis import (
    build_segment_frame,
    cluster_false_positives,
    univariate_fp_lift,
    write_fp_profiles,
)
from ftlab.models import discrimination_metrics, undersample_indices


# ------------------------------------------------------------- metrics


def test_pr_auc_baseline_equals_the_positive_rate():
    """PR AUC is only readable against its no-skill floor, so the floor is reported."""
    y = np.array([1] * 30 + [0] * 970)
    scores = np.random.default_rng(0).uniform(size=1000)
    m = discrimination_metrics(y, scores)
    assert m["pr_auc_no_skill_baseline"] == pytest.approx(0.03)


def test_a_random_scorer_lands_near_the_baseline_on_pr_auc():
    rng = np.random.default_rng(2)
    y = rng.binomial(1, 0.035, 40_000)
    m = discrimination_metrics(y, rng.uniform(size=len(y)))
    assert m["pr_auc"] == pytest.approx(0.035, abs=0.01)
    assert m["roc_auc"] == pytest.approx(0.5, abs=0.02)


def test_roc_auc_flatters_an_imbalanced_problem_more_than_pr_auc_does():
    """The README's claim, asserted: at a 3.5% base rate a mediocre model can post
    a respectable ROC AUC while its PR AUC stays low."""
    rng = np.random.default_rng(4)
    n = 60_000
    y = rng.binomial(1, 0.035, n)
    scores = np.clip(rng.normal(0.3, 0.15, n) + y * 0.25, 0, 1)
    m = discrimination_metrics(y, scores)
    assert m["roc_auc"] > 0.80
    assert m["pr_auc"] < m["roc_auc"] / 2


def test_a_perfect_scorer_reaches_one_on_both_metrics():
    y = np.array([0, 0, 1, 1])
    m = discrimination_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert m["roc_auc"] == pytest.approx(1.0)
    assert m["pr_auc"] == pytest.approx(1.0)


def test_pr_auc_lift_is_relative_to_the_baseline():
    y = np.array([0] * 90 + [1] * 10)
    m = discrimination_metrics(y, np.concatenate([np.zeros(90), np.ones(10)]))
    assert m["pr_auc_lift_over_baseline"] == pytest.approx(m["pr_auc"] / 0.10)


# ---------------------------------------------------------- undersampling


def test_undersampling_keeps_every_positive():
    rng = np.random.default_rng(1)
    y = rng.binomial(1, 0.03, 10_000)
    keep = undersample_indices(y, ratio=3.0)
    assert set(np.flatnonzero(y == 1)).issubset(set(keep))


def test_undersampling_hits_the_requested_negative_ratio():
    rng = np.random.default_rng(1)
    y = rng.binomial(1, 0.03, 10_000)
    keep = undersample_indices(y, ratio=3.0)
    n_pos = int((y[keep] == 1).sum())
    n_neg = int((y[keep] == 0).sum())
    assert n_neg == pytest.approx(3 * n_pos, rel=0.01)


def test_undersampling_is_reproducible_under_a_fixed_seed():
    y = np.random.default_rng(1).binomial(1, 0.05, 5000)
    assert np.array_equal(undersample_indices(y, seed=99), undersample_indices(y, seed=99))


def test_undersampling_cannot_request_more_negatives_than_exist():
    y = np.array([1, 1, 0])
    keep = undersample_indices(y, ratio=100.0)
    assert len(keep) == 3


# ------------------------------------------------------- FP characterization


def _flagged_fixture(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Flag every 'C' product plus a random tail, so one segment is over-flagged."""
    rng = np.random.default_rng(0)
    y = frame[TARGET].to_numpy()
    flagged = (frame["ProductCD"] == "C").to_numpy() | (rng.uniform(size=len(frame)) < 0.05)
    return y, flagged


def test_segment_frame_covers_the_interpretable_attributes(synthetic_frame):
    from ftlab.features import add_amount_features, add_time_features

    f = add_amount_features(add_time_features(synthetic_frame))
    seg = build_segment_frame(f)
    assert "ProductCD" in seg.columns
    assert "amount_bucket" in seg.columns
    assert "hour_bucket" in seg.columns
    assert len(seg) == len(f)


def test_missing_categoricals_become_an_explicit_level(synthetic_frame):
    seg = build_segment_frame(synthetic_frame)
    assert seg["DeviceType"].isna().sum() == 0
    assert "__missing__" in set(seg["DeviceType"])


def test_fp_lift_denominator_uses_legitimate_transactions_only(synthetic_frame):
    y, flagged = _flagged_fixture(synthetic_frame)
    lift = univariate_fp_lift(synthetic_frame, y, flagged, min_segment_size=1)
    row = lift[(lift["attribute"] == "ProductCD") & (lift["value"] == "C")].iloc[0]
    legit_c = int(((y == 0) & (synthetic_frame["ProductCD"] == "C")).sum())
    assert row["legit_transactions"] == legit_c


def test_fp_lift_identifies_the_over_flagged_segment(synthetic_frame):
    y, flagged = _flagged_fixture(synthetic_frame)
    lift = univariate_fp_lift(synthetic_frame, y, flagged, min_segment_size=1)
    top = lift.iloc[0]
    assert top["attribute"] == "ProductCD" and top["value"] == "C"
    assert top["lift_vs_base"] > 1.5


def test_fp_lift_is_rate_over_base_rate(synthetic_frame):
    y, flagged = _flagged_fixture(synthetic_frame)
    lift = univariate_fp_lift(synthetic_frame, y, flagged, min_segment_size=1)
    np.testing.assert_allclose(
        lift["lift_vs_base"],
        lift["false_alarm_rate"] / lift["base_false_alarm_rate"],
        rtol=1e-10,
    )


def test_small_segments_are_excluded(synthetic_frame):
    y, flagged = _flagged_fixture(synthetic_frame)
    lift = univariate_fp_lift(synthetic_frame, y, flagged, min_segment_size=100_000)
    assert lift.empty


def test_clustering_partitions_every_false_positive(synthetic_frame):
    from ftlab.features import add_amount_features, add_time_features

    f = add_amount_features(add_time_features(synthetic_frame))
    f["grp_card1__count"] = 10.0
    f["grp_card1__amt_ratio"] = 1.0
    f["grp_addr1__count"] = 20.0
    y, flagged = _flagged_fixture(f)
    clusters, labels = cluster_false_positives(f, y, flagged, n_clusters=3)
    n_fp = int((flagged & (y == 0)).sum())
    assert len(labels) == n_fp
    assert clusters["n_false_positives"].sum() == n_fp
    assert clusters["share_of_false_positives"].sum() == pytest.approx(1.0)


def test_clustering_degrades_gracefully_when_there_are_too_few_false_positives(synthetic_frame):
    y = synthetic_frame[TARGET].to_numpy()
    flagged = np.zeros(len(y), dtype=bool)
    clusters, labels = cluster_false_positives(synthetic_frame, y, flagged, n_clusters=5)
    assert clusters.empty and labels.empty


def test_prose_is_generated_from_the_tables(synthetic_frame):
    """The README text must be derived from the numbers, not written alongside them."""
    y, flagged = _flagged_fixture(synthetic_frame)
    lift = univariate_fp_lift(synthetic_frame, y, flagged, min_segment_size=1)
    op = {
        "threshold": 0.05,
        "n_alerts": int(flagged.sum()),
        "fp": int((flagged & (y == 0)).sum()),
        "false_alarm_rate": 0.9,
        "n_legit": int((y == 0).sum()),
        "base_false_alarm_rate": float(flagged[y == 0].mean()),
    }
    text = write_fp_profiles(lift, pd.DataFrame(), op)
    assert f"{op['n_alerts']:,}" in text
    assert "base rate" in text


# ------------------------------------------------------- logistic baseline


def test_winsorizer_caps_extreme_standardised_values():
    """Heavy-tailed anonymised columns reach 500 sigma; the linear model must not
    spend its capacity there."""
    from ftlab.models import LOGISTIC_CLIP_SIGMA, _clip_sigma

    out = _clip_sigma(np.array([-900.0, -1.0, 0.0, 2.5, 513.0]))
    assert out.max() == pytest.approx(LOGISTIC_CLIP_SIGMA)
    assert out.min() == pytest.approx(-LOGISTIC_CLIP_SIGMA)


def test_winsorizer_leaves_ordinary_values_untouched():
    from ftlab.models import _clip_sigma

    ordinary = np.array([-2.0, -0.5, 0.0, 1.25, 3.0])
    np.testing.assert_allclose(_clip_sigma(ordinary), ordinary)


def test_logistic_baseline_reports_its_convergence_status(synthetic_frame):
    """A baseline that silently failed to converge would overstate the margin the
    primary model beats it by, so the status is part of the reported metrics."""
    from ftlab.features import add_amount_features, add_time_features
    from ftlab.models import fit_logistic_baseline
    from ftlab.features import to_model_matrix

    f = add_amount_features(add_time_features(synthetic_frame))
    X, cats = to_model_matrix(f)
    y = f[TARGET].to_numpy()
    r = fit_logistic_baseline(X[:1200], y[:1200], X[1200:1600], y[1200:1600],
                              X[1600:], y[1600:], cats)
    assert set(["converged", "n_iter", "max_iter", "clip_sigma"]) <= set(r.metrics)
    assert isinstance(r.metrics["converged"], bool)
    assert r.metrics["n_iter"] <= r.metrics["max_iter"]
