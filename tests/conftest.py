"""Synthetic fixtures.

The tests deliberately do not touch the Kaggle data. They run in CI where that
data is unavailable, and more importantly the metric and cost logic is worth
testing against inputs whose correct answers can be worked out by hand.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ftlab.config import AMOUNT, ID_COL, TARGET, TIME_COL


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)


@pytest.fixture
def tiny_scores() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ten rows with hand-checkable confusion counts.

    Sorted by score descending, the labels are 1,1,0,1,0,0,1,0,0,0.
    """
    y = np.array([1, 1, 0, 1, 0, 0, 1, 0, 0, 0])
    scores = np.array([0.95, 0.90, 0.85, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10])
    amounts = np.array([100.0, 200.0, 50.0, 300.0, 25.0, 75.0, 400.0, 10.0, 20.0, 30.0])
    return y, scores, amounts


@pytest.fixture
def synthetic_frame(rng: np.random.Generator) -> pd.DataFrame:
    """A small frame shaped like the real one: time axis, amount, card keys, target."""
    n = 2000
    dt = np.sort(rng.integers(86_400, 86_400 * 120, size=n)).astype(np.int64)
    card1 = rng.integers(1000, 1050, size=n)
    addr1 = rng.integers(100, 110, size=n)
    amount = np.round(rng.lognormal(mean=4.0, sigma=1.0, size=n), 2)

    # Fraud probability rises with amount and is concentrated on a few cards.
    logit = -3.5 + 0.35 * np.log1p(amount) + 0.9 * np.isin(card1, [1001, 1002, 1003])
    p = 1 / (1 + np.exp(-logit))
    y = rng.binomial(1, p)

    return pd.DataFrame(
        {
            ID_COL: np.arange(n),
            TIME_COL: dt,
            AMOUNT: amount.astype(np.float32),
            "card1": card1,
            "card2": rng.integers(200, 220, size=n),
            "addr1": addr1,
            "P_emaildomain": rng.choice(["a.com", "b.com", "c.com"], size=n),
            "ProductCD": rng.choice(["W", "C", "R"], size=n),
            "card4": rng.choice(["visa", "mastercard"], size=n),
            "card6": rng.choice(["debit", "credit"], size=n),
            "DeviceType": rng.choice(["mobile", "desktop", None], size=n),
            "identity_present": rng.integers(0, 2, size=n).astype(np.int8),
            TARGET: y,
        }
    )
