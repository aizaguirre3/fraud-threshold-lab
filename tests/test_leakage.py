"""Leakage guard.

The handoff requires the guard to exist "in code and in a test". Each check
gets a test that constructs the leak it is supposed to catch and asserts the
guard raises -- a guard that has never been shown to fire is not a guard.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ftlab.config import ID_COL, TARGET, TIME_COL, SplitConfig
from ftlab.features import build_features, fit_behavioral_aggregates
from ftlab.leakage import (
    LeakageError,
    assert_fit_window_is_train_only,
    assert_no_forbidden_columns,
    assert_no_target_derived,
    assert_splits_are_temporally_disjoint,
    audit,
    single_feature_auc,
)
from ftlab.split import make_time_split


def test_target_column_in_features_is_caught():
    X = pd.DataFrame({"a": [1, 2, 3], TARGET: [0, 1, 0]})
    with pytest.raises(LeakageError, match="Forbidden column"):
        assert_no_forbidden_columns(X)


def test_row_identifier_in_features_is_caught():
    X = pd.DataFrame({"a": [1, 2, 3], ID_COL: [10, 11, 12]})
    with pytest.raises(LeakageError, match=ID_COL):
        assert_no_forbidden_columns(X)


def test_raw_time_axis_in_features_is_caught():
    """TransactionDT is monotone across the split; as a feature it encodes the period."""
    X = pd.DataFrame({"a": [1, 2, 3], TIME_COL: [100, 200, 300]})
    with pytest.raises(LeakageError, match=TIME_COL):
        assert_no_forbidden_columns(X)


def test_clean_feature_matrix_passes():
    X = pd.DataFrame({"a": [1, 2, 3], "hour_of_day": [1.0, 2.0, 3.0]})
    assert_no_forbidden_columns(X)  # must not raise


def test_target_derived_column_is_caught():
    y = pd.Series([0, 1, 0, 1, 0, 1, 1, 0])
    X = pd.DataFrame({"innocent": [1, 2, 3, 4, 5, 6, 7, 8], "sneaky": y * 10 + 1})
    with pytest.raises(LeakageError, match="target-derived"):
        assert_no_target_derived(X, y)


def test_inverted_target_derived_column_is_also_caught():
    """A perfectly anti-correlated column leaks exactly as badly."""
    y = pd.Series([0, 1, 0, 1, 0, 1, 1, 0])
    X = pd.DataFrame({"inverted": 1 - y})
    with pytest.raises(LeakageError):
        assert_no_target_derived(X, y)


def test_a_merely_informative_column_is_not_flagged():
    """The guard must not fire on a genuinely good feature, or it is unusable."""
    rng = np.random.default_rng(0)
    y = pd.Series(rng.binomial(1, 0.3, size=500))
    x = y * rng.normal(1.0, 1.0, 500) + rng.normal(0, 1.0, 500)
    assert_no_target_derived(pd.DataFrame({"useful": x}), y)


def test_single_feature_auc_is_orientation_free():
    y = pd.Series([0, 0, 1, 1])
    aligned = pd.Series([1.0, 2.0, 3.0, 4.0])
    inverted = pd.Series([4.0, 3.0, 2.0, 1.0])
    assert single_feature_auc(aligned, y) == pytest.approx(1.0)
    assert single_feature_auc(inverted, y) == pytest.approx(1.0)


def test_single_feature_auc_of_a_constant_column_is_one_half():
    y = pd.Series([0, 1, 0, 1])
    assert single_feature_auc(pd.Series([7.0] * 4), y) == pytest.approx(0.5)


def test_aggregate_fit_outside_the_training_window_is_caught():
    train = pd.Index([0, 1, 2, 3])
    fit_on_everything = pd.Index([0, 1, 2, 3, 4, 5])
    with pytest.raises(LeakageError, match="outside the training window"):
        assert_fit_window_is_train_only(fit_on_everything, train)


def test_aggregate_fit_on_the_training_window_passes():
    assert_fit_window_is_train_only(pd.Index([0, 1, 2]), pd.Index([0, 1, 2, 3]))


def test_overlapping_split_windows_are_caught():
    train_t = pd.Series([1, 2, 3, 9])
    val_t = pd.Series([4, 5])
    test_t = pd.Series([6, 7])
    with pytest.raises(LeakageError, match="overlaps validation"):
        assert_splits_are_temporally_disjoint(train_t, val_t, test_t)


def test_disjoint_split_windows_pass():
    assert_splits_are_temporally_disjoint(
        pd.Series([1, 2, 3]), pd.Series([4, 5]), pd.Series([6, 7])
    )


def test_the_real_feature_pipeline_passes_its_own_audit(synthetic_frame):
    """End-to-end: aggregates fit on train only, then audited."""
    split = make_time_split(synthetic_frame, SplitConfig())
    maps = fit_behavioral_aggregates(synthetic_frame.loc[split.train_idx])
    feats = build_features(synthetic_frame, maps, train_index=split.train_idx)
    X = feats.drop(columns=[TARGET, ID_COL, TIME_COL])
    report = audit(X.select_dtypes(include=[np.number]), feats[TARGET])
    assert report["forbidden_check"] == "pass"
    assert report["max_single_feature_auc"] < 0.999


def test_aggregates_fit_on_the_full_frame_are_rejected_by_build_features(synthetic_frame):
    """The realistic mistake: fitting card statistics before splitting."""
    split = make_time_split(synthetic_frame, SplitConfig())
    leaky_maps = fit_behavioral_aggregates(synthetic_frame)  # fit on everything
    with pytest.raises(LeakageError, match="outside the training window"):
        build_features(synthetic_frame, leaky_maps, train_index=split.train_idx)
