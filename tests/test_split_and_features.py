"""Time-aware split and feature construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ftlab.config import AMOUNT, ID_COL, TARGET, TIME_COL, SplitConfig
from ftlab.features import (
    add_amount_features,
    add_time_features,
    align_categories,
    apply_behavioral_aggregates,
    fit_behavioral_aggregates,
    to_model_matrix,
)
from ftlab.split import make_time_split, split_frames


# ----------------------------------------------------------------- split


def test_split_windows_are_chronological(synthetic_frame):
    split = make_time_split(synthetic_frame, SplitConfig())
    tr, va, te = split_frames(synthetic_frame, split)
    assert tr[TIME_COL].max() <= va[TIME_COL].min()
    assert va[TIME_COL].max() <= te[TIME_COL].min()


def test_split_windows_partition_the_frame(synthetic_frame):
    split = make_time_split(synthetic_frame, SplitConfig())
    total = len(split.train_idx) + len(split.val_idx) + len(split.test_idx)
    assert total == len(synthetic_frame)
    assert len(split.train_idx.intersection(split.val_idx)) == 0
    assert len(split.val_idx.intersection(split.test_idx)) == 0


def test_split_respects_the_requested_proportions(synthetic_frame):
    split = make_time_split(synthetic_frame, SplitConfig(train_frac=0.6, val_frac=0.2))
    n = len(synthetic_frame)
    assert len(split.train_idx) == pytest.approx(0.6 * n, rel=0.05)
    assert len(split.val_idx) == pytest.approx(0.2 * n, rel=0.10)


def test_fractions_that_leave_no_test_window_are_rejected(synthetic_frame):
    with pytest.raises(ValueError, match="non-empty test window"):
        make_time_split(synthetic_frame, SplitConfig(train_frac=0.8, val_frac=0.2))


def test_out_of_range_fractions_are_rejected(synthetic_frame):
    with pytest.raises(ValueError, match="strictly in"):
        make_time_split(synthetic_frame, SplitConfig(train_frac=1.5, val_frac=0.1))


def test_split_summary_records_the_boundaries(synthetic_frame):
    split = make_time_split(synthetic_frame, SplitConfig())
    s = split.summary()
    assert s["train_rows"] + s["val_rows"] + s["test_rows"] == len(synthetic_frame)
    assert s["train_end_dt_raw"] <= s["val_end_dt_raw"]


# -------------------------------------------------------------- features


def test_time_features_are_bounded_and_periodic(synthetic_frame):
    out = add_time_features(synthetic_frame)
    assert out["hour_of_day"].between(0, 24).all()
    assert out["day_of_week"].between(0, 7).all()


def test_raw_time_is_not_promoted_into_a_feature(synthetic_frame):
    """Derived time features must not simply re-encode the raw offset."""
    out = add_time_features(synthetic_frame)
    assert out["hour_of_day"].corr(out[TIME_COL]) < 0.5


def test_amount_features_match_their_definitions(synthetic_frame):
    out = add_amount_features(synthetic_frame)
    np.testing.assert_allclose(
        out["amt_log"], np.log1p(synthetic_frame[AMOUNT].astype(float)), rtol=1e-5
    )
    assert out["amt_cents"].between(0, 1).all()
    assert set(out["amt_is_round"].unique()) <= {0, 1}


def test_round_amounts_are_flagged():
    df = pd.DataFrame({AMOUNT: [100.00, 99.99, 50.0, 12.34]})
    out = add_amount_features(df)
    assert out["amt_is_round"].tolist() == [1, 0, 1, 0]


def test_aggregates_are_computed_from_the_training_rows_only(synthetic_frame):
    split = make_time_split(synthetic_frame, SplitConfig())
    train = synthetic_frame.loc[split.train_idx]
    maps = fit_behavioral_aggregates(train)
    tbl = maps.tables["grp_card1"]
    # Recompute one group by hand from the training rows.
    card = train["card1"].iloc[0]
    expected = train.loc[train["card1"] == card, AMOUNT].mean()
    assert tbl.loc[card, "grp_card1__amt_mean"] == pytest.approx(expected, rel=1e-4)
    assert maps.fit_index.equals(train.index)


def test_aggregate_counts_reflect_training_volume_not_total_volume(synthetic_frame):
    split = make_time_split(synthetic_frame, SplitConfig())
    train = synthetic_frame.loc[split.train_idx]
    maps = fit_behavioral_aggregates(train)
    card = train["card1"].iloc[0]
    train_count = int((train["card1"] == card).sum())
    full_count = int((synthetic_frame["card1"] == card).sum())
    assert maps.tables["grp_card1"].loc[card, "grp_card1__count"] == train_count
    assert train_count < full_count  # the card does appear later, and is not counted


def test_groups_unseen_in_training_get_missing_aggregates(synthetic_frame):
    train = synthetic_frame.iloc[:100]
    maps = fit_behavioral_aggregates(train)
    later = synthetic_frame.copy()
    later.loc[later.index[-5:], "card1"] = 999_999  # a card that never trained
    out = apply_behavioral_aggregates(later, maps)
    assert out.loc[out["card1"] == 999_999, "grp_card1__count"].isna().all()


def test_amount_ratio_compares_the_row_to_its_group_mean(synthetic_frame):
    maps = fit_behavioral_aggregates(synthetic_frame)
    out = apply_behavioral_aggregates(synthetic_frame, maps)
    ratio = out[AMOUNT] / out["grp_card1__amt_mean"]
    np.testing.assert_allclose(
        out["grp_card1__amt_ratio"].dropna(), ratio.dropna(), rtol=1e-4
    )


def test_model_matrix_drops_target_identifier_and_raw_time(synthetic_frame):
    X, cats = to_model_matrix(synthetic_frame)
    for col in (TARGET, ID_COL, TIME_COL):
        assert col not in X.columns


def test_category_alignment_maps_unseen_levels_to_missing():
    train = pd.DataFrame({"c": pd.Categorical(["a", "b", "a"])})
    test = pd.DataFrame({"c": pd.Categorical(["a", "z"])})
    (aligned,) = align_categories(train, test)
    assert list(aligned["c"].cat.categories) == ["a", "b"]
    assert aligned["c"].isna().sum() == 1  # 'z' was never trained on


def test_category_alignment_keeps_shared_levels_on_the_same_code():
    """Without alignment a level can silently mean something different in test."""
    train = pd.DataFrame({"c": pd.Categorical(["x", "y", "z"])})
    test = pd.DataFrame({"c": pd.Categorical(["z", "y"])})
    (aligned,) = align_categories(train, test)
    assert list(aligned["c"].cat.categories) == list(train["c"].cat.categories)
    assert aligned["c"].cat.codes.tolist() == [2, 1]


# ------------------------------------------------------- target encoding


def test_target_encoding_is_smoothed_toward_the_prior(synthetic_frame):
    """A group seen once with one fraud must not encode to 1.0."""
    from ftlab.features import fit_target_encoders

    maps = fit_target_encoders(synthetic_frame, smoothing=20.0)
    for tbl in maps.tables.values():
        col = tbl.iloc[:, 0]
        assert col.max() <= 1.0
        assert col.min() >= 0.0
        # Smoothing keeps every group well inside the extremes.
        assert col.max() < 0.95


def test_target_encoding_records_the_prior_it_smoothed_toward(synthetic_frame):
    from ftlab.features import fit_target_encoders

    maps = fit_target_encoders(synthetic_frame)
    assert maps.prior == pytest.approx(synthetic_frame[TARGET].mean())


def test_stronger_smoothing_pulls_groups_closer_to_the_prior(synthetic_frame):
    from ftlab.features import fit_target_encoders

    weak = fit_target_encoders(synthetic_frame, smoothing=1.0)
    strong = fit_target_encoders(synthetic_frame, smoothing=500.0)
    prior = weak.prior
    key = next(iter(weak.tables))
    weak_spread = (weak.tables[key].iloc[:, 0] - prior).abs().mean()
    strong_spread = (strong.tables[key].iloc[:, 0] - prior).abs().mean()
    assert strong_spread < weak_spread


def test_target_encoding_carries_its_fit_window(synthetic_frame):
    from ftlab.features import fit_target_encoders

    split = make_time_split(synthetic_frame, SplitConfig())
    maps = fit_target_encoders(synthetic_frame.loc[split.train_idx])
    assert maps.fit_index.equals(split.train_idx)


def test_target_encoding_fit_outside_the_training_window_is_rejected(synthetic_frame):
    """The mistake that inflates PR AUC by 21.9%: encode before splitting."""
    from ftlab.features import apply_target_encoders, fit_target_encoders
    from ftlab.leakage import LeakageError

    split = make_time_split(synthetic_frame, SplitConfig())
    leaky = fit_target_encoders(synthetic_frame)  # fit on everything
    with pytest.raises(LeakageError, match="outside the training window"):
        apply_target_encoders(synthetic_frame, leaky, split.train_idx)


def test_training_rows_receive_out_of_fold_encodings(synthetic_frame):
    """A training row must not be scored with an encoding it helped produce.

    The out-of-fold value differs from the full-fit value precisely because the
    row's own label was excluded from it.
    """
    from ftlab.features import apply_target_encoders, fit_target_encoders

    split = make_time_split(synthetic_frame, SplitConfig())
    train = synthetic_frame.loc[split.train_idx]
    maps = fit_target_encoders(train)
    out = apply_target_encoders(synthetic_frame, maps, split.train_idx)

    name = maps.column_names()[0]
    prefix = name[len("te_"):]
    full_fit = synthetic_frame.join(maps.tables[prefix], on=["card1"])[name]
    oof = out.loc[split.train_idx, name]
    differing = (oof - full_fit.loc[split.train_idx]).abs() > 1e-9
    assert differing.mean() > 0.5  # most training rows must differ


def test_non_training_rows_use_the_full_fit_encoding(synthetic_frame):
    from ftlab.features import apply_target_encoders, fit_target_encoders

    split = make_time_split(synthetic_frame, SplitConfig())
    maps = fit_target_encoders(synthetic_frame.loc[split.train_idx])
    out = apply_target_encoders(synthetic_frame, maps, split.train_idx)

    name = maps.column_names()[0]
    prefix = name[len("te_"):]
    expected = synthetic_frame.join(maps.tables[prefix], on=["card1"])[name]
    np.testing.assert_allclose(
        out.loc[split.test_idx, name].to_numpy(dtype=float),
        expected.loc[split.test_idx].to_numpy(dtype=float),
        rtol=1e-6, equal_nan=True,
    )


def test_groups_unseen_in_training_get_no_encoding(synthetic_frame):
    from ftlab.features import apply_target_encoders, fit_target_encoders

    split = make_time_split(synthetic_frame, SplitConfig())
    maps = fit_target_encoders(synthetic_frame.loc[split.train_idx])
    frame = synthetic_frame.copy()
    frame.loc[frame.index[-5:], "card1"] = 999_999
    out = apply_target_encoders(frame, maps, split.train_idx)
    assert out.loc[frame["card1"] == 999_999, "te_grp_card1"].isna().all()


def test_build_features_requires_a_train_index_for_target_encoding(synthetic_frame):
    from ftlab.features import build_features, fit_behavioral_aggregates, fit_target_encoders

    maps = fit_behavioral_aggregates(synthetic_frame)
    tmaps = fit_target_encoders(synthetic_frame)
    with pytest.raises(ValueError, match="train_index is required"):
        build_features(synthetic_frame, maps, target_maps=tmaps)
