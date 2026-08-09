"""Time-aware train / validation / test split.

The split is chronological, not random. In IEEE-CIS ``TransactionDT`` is a
seconds offset from an unspecified reference date and the labelled file spans
roughly six months. Cutting it at time quantiles produces three contiguous,
non-overlapping windows:

    |------------ train (70%) ------------|-- val (15%) --|-- test (15%) --|

Two things this buys, both of which a random split destroys:

* Behavioural aggregates fit on the training window cannot see a card's future
  behaviour, because the future is physically in a different frame.
* The evaluation answers the question an operator actually has -- "how will the
  model do on next month's traffic" -- rather than "how well does it interpolate
  within a period it has already seen".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import TIME_COL, SplitConfig
from .leakage import assert_splits_are_temporally_disjoint


@dataclass(frozen=True)
class TimeSplit:
    """Row-index masks plus the boundaries that produced them."""

    train_idx: pd.Index
    val_idx: pd.Index
    test_idx: pd.Index
    train_end_dt: float
    val_end_dt: float
    time_col: str = TIME_COL

    def summary(self) -> dict:
        return {
            "time_col": self.time_col,
            "train_rows": len(self.train_idx),
            "val_rows": len(self.val_idx),
            "test_rows": len(self.test_idx),
            "train_end_dt_raw": float(self.train_end_dt),
            "val_end_dt_raw": float(self.val_end_dt),
            "train_end_day": round(float(self.train_end_dt) / 86400.0, 2),
            "val_end_day": round(float(self.val_end_dt) / 86400.0, 2),
        }


def make_time_split(df: pd.DataFrame, cfg: SplitConfig | None = None) -> TimeSplit:
    """Split ``df`` chronologically into train / validation / test.

    Ties on the boundary timestamp are pushed into the *earlier* window, so the
    windows stay disjoint in row terms even when many transactions share a
    second.
    """
    cfg = cfg or SplitConfig()
    if not 0 < cfg.train_frac < 1 or not 0 < cfg.val_frac < 1:
        raise ValueError("train_frac and val_frac must each lie strictly in (0, 1).")
    if cfg.train_frac + cfg.val_frac >= 1.0:
        raise ValueError(
            f"train_frac ({cfg.train_frac}) + val_frac ({cfg.val_frac}) must be < 1 "
            "so that a non-empty test window remains."
        )

    t = df[cfg.time_col]
    train_end = float(np.quantile(t, cfg.train_frac))
    val_end = float(np.quantile(t, cfg.train_frac + cfg.val_frac))

    train_mask = t <= train_end
    val_mask = (t > train_end) & (t <= val_end)
    test_mask = t > val_end

    for name, mask in (("train", train_mask), ("val", val_mask), ("test", test_mask)):
        if not mask.any():
            raise ValueError(f"Time split produced an empty {name} window.")

    split = TimeSplit(
        train_idx=df.index[train_mask],
        val_idx=df.index[val_mask],
        test_idx=df.index[test_mask],
        train_end_dt=train_end,
        val_end_dt=val_end,
        time_col=cfg.time_col,
    )
    assert_splits_are_temporally_disjoint(
        t[train_mask], t[val_mask], t[test_mask]
    )
    return split


def split_frames(
    df: pd.DataFrame, split: TimeSplit
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        df.loc[split.train_idx],
        df.loc[split.val_idx],
        df.loc[split.test_idx],
    )
