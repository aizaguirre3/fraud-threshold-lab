"""Calibration measurement and post-hoc recalibration.

Discrimination and calibration are different properties. Discrimination asks
whether the model puts fraud above non-fraud in the ranking. Calibration asks
whether a score of 0.30 means the thing happens 30% of the time. A model can be
excellent at the first and badly wrong at the second, and the second is what
determines false-positive volume whenever somebody sets a threshold by reading
the number as a probability -- which is what everyone does.

Recalibrators are fit on the **validation** window and evaluated on **test**.
Fitting a calibrator on the data it is scored against reports the calibrator's
memory, not its quality.

A note on what recalibration can and cannot do, which the results here bear out:
Platt scaling is strictly monotone, so it cannot change the ranking, and
therefore cannot change *which* transactions are flagged at a matched alert
volume. At matched volume the false-positive count is identical by
construction. Calibration pays off at a *fixed probability threshold*, where it
changes how many transactions clear the bar at all. Both comparisons are
computed below, because reporting only the favourable one would be dishonest.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss



def expected_calibration_error(
    y_true: np.ndarray, prob: np.ndarray, n_bins: int = 20, strategy: str = "quantile"
) -> float:
    """Weighted mean gap between predicted probability and observed frequency.

    ``strategy='quantile'`` puts equal counts in each bin. Under heavy imbalance
    that is the right default: equal-width bins leave the upper bins nearly
    empty, and an ECE dominated by bins holding a handful of rows is noise.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(prob, dtype=float)
    n = len(p)
    if n == 0:
        return float("nan")

    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    elif strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        raise ValueError(f"Unknown binning strategy {strategy!r}.")
    if len(edges) < 2:
        return float(abs(p.mean() - y.mean()))

    idx = np.clip(np.searchsorted(edges, p, side="left") - 1, 0, len(edges) - 2)
    total = 0.0
    for b in range(len(edges) - 1):
        mask = idx == b
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        total += (cnt / n) * abs(p[mask].mean() - y[mask].mean())
    return float(total)


def reliability_curve(
    y_true: np.ndarray, prob: np.ndarray, n_bins: int = 20, strategy: str = "quantile"
) -> pd.DataFrame:
    """Per-bin mean predicted probability against observed fraud rate."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(prob, dtype=float)
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.searchsorted(edges, p, side="left") - 1, 0, len(edges) - 2)

    rows = []
    for b in range(len(edges) - 1):
        mask = idx == b
        if not mask.any():
            continue
        rows.append(
            {
                "bin": b,
                "bin_low": float(edges[b]),
                "bin_high": float(edges[b + 1]),
                "count": int(mask.sum()),
                "mean_predicted": float(p[mask].mean()),
                "observed_rate": float(y[mask].mean()),
                "gap": float(p[mask].mean() - y[mask].mean()),
            }
        )
    return pd.DataFrame(rows)


def calibration_metrics(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 20) -> dict:
    """ECE (both binnings), Brier score, and the aggregate over/under-prediction."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(prob, dtype=float)
    return {
        "ece_quantile": expected_calibration_error(y, p, n_bins, "quantile"),
        "ece_uniform": expected_calibration_error(y, p, n_bins, "uniform"),
        "brier": float(brier_score_loss(y, p)),
        "mean_predicted": float(p.mean()),
        "observed_rate": float(y.mean()),
        "mean_prediction_bias": float(p.mean() - y.mean()),
    }


@dataclass
class Recalibrator:
    """A fitted post-hoc calibration map."""

    name: str
    model: object

    def transform(self, scores: np.ndarray) -> np.ndarray:
        s = np.asarray(scores, dtype=float)
        if self.name == "platt":
            return self.model.predict_proba(s.reshape(-1, 1))[:, 1]
        return np.clip(self.model.predict(s), 0.0, 1.0)


def fit_platt(val_scores: np.ndarray, y_val: np.ndarray) -> Recalibrator:
    """Logistic regression on the raw score. Strictly monotone: ranking preserved."""
    lr = LogisticRegression(max_iter=1000)
    lr.fit(np.asarray(val_scores, dtype=float).reshape(-1, 1), np.asarray(y_val))
    return Recalibrator("platt", lr)


def fit_isotonic(val_scores: np.ndarray, y_val: np.ndarray) -> Recalibrator:
    """Isotonic regression. Weakly monotone -- it can flatten score ties, so it
    may reorder within tied blocks, but it cannot invert the ranking."""
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(np.asarray(val_scores, dtype=float), np.asarray(y_val, dtype=float))
    return Recalibrator("isotonic", iso)


def _top_k_confusion(
    y: np.ndarray, scores: np.ndarray, amounts: np.ndarray, k: int
) -> dict:
    """Confusion counts for the k highest-scoring rows.

    Selection is by rank rather than by a threshold grid. A grid cannot express
    "exactly k alerts" for two different score distributions -- the nearest
    achievable volumes differ, and comparing 295 alerts against 300 is not a
    matched-volume comparison. Ranking gives exactly k for both.
    """
    y = np.asarray(y).astype(int)
    s = np.asarray(scores, dtype=float)
    a = np.asarray(amounts, dtype=float)
    n = len(y)
    k = int(np.clip(k, 0, n))
    order = np.argsort(-s, kind="mergesort")[:k]
    tp = int(y[order].sum())
    fp = k - tp
    total_pos = int(y.sum())
    return {
        "threshold": float(s[order[-1]]) if k > 0 else float("inf"),
        "n_alerts": k,
        "tp": tp,
        "fp": fp,
        "precision": (tp / k) if k else float("nan"),
        "recall": (tp / total_pos) if total_pos else float("nan"),
        "fraud_caught_dollars": float(a[order][y[order] == 1].sum()) if k else 0.0,
    }


def compare_at_matched_alert_volume(
    y_test: np.ndarray,
    raw_scores: np.ndarray,
    calibrated_scores: np.ndarray,
    amounts: np.ndarray,
    target_alerts: int,
    cost=None,
) -> dict:
    """False positives at equal alert volume, raw versus calibrated.

    This is the comparison the handoff asks for. Expect it to come out flat for
    a monotone recalibrator -- that is the correct answer, not a bug, and the
    README says so.
    """
    raw = _top_k_confusion(y_test, raw_scores, amounts, target_alerts)
    cal = _top_k_confusion(y_test, calibrated_scores, amounts, target_alerts)
    return {
        "target_alerts": int(target_alerts),
        "raw": raw,
        "calibrated": cal,
        "fp_reduction": raw["fp"] - cal["fp"],
        "note": (
            "A strictly monotone recalibration cannot change which rows are "
            "flagged at a matched alert volume, so a zero difference here is "
            "the expected result rather than a null finding."
        ),
    }


def compare_at_fixed_threshold(
    y_test: np.ndarray,
    raw_scores: np.ndarray,
    calibrated_scores: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Alert volume and false positives at a fixed probability threshold.

    This is where calibration actually changes the operation: the threshold is
    read as a probability, and an overconfident model clears it far too often.
    """
    y = np.asarray(y_test).astype(int)

    def _at(p: np.ndarray) -> dict:
        flag = np.asarray(p) >= threshold
        tp = int(((flag) & (y == 1)).sum())
        fp = int(((flag) & (y == 0)).sum())
        n_alerts = int(flag.sum())
        return {
            "n_alerts": n_alerts,
            "tp": tp,
            "fp": fp,
            "precision": (tp / n_alerts) if n_alerts else float("nan"),
            "recall": tp / max(int((y == 1).sum()), 1),
        }

    raw = _at(raw_scores)
    cal = _at(calibrated_scores)
    return {
        "threshold": threshold,
        "raw": raw,
        "calibrated": cal,
        "fp_reduction": raw["fp"] - cal["fp"],
        "alert_volume_change": cal["n_alerts"] - raw["n_alerts"],
    }
