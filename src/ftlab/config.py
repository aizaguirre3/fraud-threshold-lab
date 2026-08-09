"""Central configuration: paths, seeds, split boundaries, and cost assumptions.

Every number a reviewer might question lives here rather than being buried in a
function body, so the assumptions behind a reported result are inspectable in
one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

RANDOM_SEED = 20260808

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"
FIGURES = ARTIFACTS / "figures"
TABLES = ARTIFACTS / "tables"


@dataclass(frozen=True)
class SplitConfig:
    """Time-aware split boundaries, expressed as quantiles of ``TransactionDT``.

    IEEE-CIS ships a labelled ``train`` file and an unlabelled competition
    ``test`` file. Because the competition test labels are not public, the
    train/validation/test split here is carved out of the labelled file along
    the time axis. A random split would let the model see future transactions
    on the same card that it is asked to score, which inflates every metric.
    """

    train_frac: float = 0.70
    val_frac: float = 0.15
    # test_frac is the remainder (0.15)
    time_col: str = "TransactionDT"

    @property
    def test_frac(self) -> float:
        return 1.0 - self.train_frac - self.val_frac


@dataclass(frozen=True)
class CostConfig:
    """Cost assumptions for cost-sensitive threshold selection.

    Defaults chosen with the project owner:

    - ``analyst_hourly_cost``: fully loaded cost of a fraud review analyst.
    - ``minutes_per_review``: average manual review handling time.
    - ``friction_cost_per_fp``: proxy for the customer cost of a false alarm
      (declined transaction, call-in, abandonment). This one is genuinely not
      measurable from the data; the sensitivity sweep in ``cost.py`` exists
      precisely because it is an assumption rather than a measurement.

    False-negative cost is *not* a constant: it is the actual transaction
    amount of the missed fraud, read per row from ``TransactionAmt``.
    """

    analyst_hourly_cost: float = 35.0
    minutes_per_review: float = 4.0
    friction_cost_per_fp: float = 5.0

    @property
    def review_cost_per_alert(self) -> float:
        """Labour cost of a single manual review, in dollars."""
        return self.analyst_hourly_cost * (self.minutes_per_review / 60.0)

    @property
    def cost_per_false_positive(self) -> float:
        """Total cost of one false alarm: review labour plus customer friction."""
        return self.review_cost_per_alert + self.friction_cost_per_fp


@dataclass(frozen=True)
class ModelConfig:
    """LightGBM hyperparameters. Deliberately modest -- the operational layer,
    not model sophistication, is the contribution of this project."""

    n_estimators: int = 800
    learning_rate: float = 0.05
    num_leaves: int = 64
    min_child_samples: int = 100
    subsample: float = 0.8
    subsample_freq: int = 1
    colsample_bytree: float = 0.6
    reg_lambda: float = 1.0
    early_stopping_rounds: int = 50
    n_jobs: int = -1

    def as_lgbm_kwargs(self) -> dict:
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "min_child_samples": self.min_child_samples,
            "subsample": self.subsample,
            "subsample_freq": self.subsample_freq,
            "colsample_bytree": self.colsample_bytree,
            "reg_lambda": self.reg_lambda,
            "n_jobs": self.n_jobs,
            "random_state": RANDOM_SEED,
            "verbose": -1,
        }


@dataclass(frozen=True)
class Config:
    split: SplitConfig = field(default_factory=SplitConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    seed: int = RANDOM_SEED


DEFAULT = Config()

TARGET = "isFraud"
AMOUNT = "TransactionAmt"
ID_COL = "TransactionID"
TIME_COL = "TransactionDT"


def ensure_dirs() -> None:
    for d in (DATA_PROCESSED, ARTIFACTS, FIGURES, TABLES):
        d.mkdir(parents=True, exist_ok=True)
