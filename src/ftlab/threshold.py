"""Threshold sweep -- the core contribution.

A classifier emits a score. A review operation needs a decision, and the
decision is a threshold. This module computes, across the full threshold range,
everything a review manager needs to choose one:

    at this threshold you flag N transactions, need H analyst hours, catch X% of
    fraud worth $Y, and Z% of your alerts are false alarms.

The sweep is exact, not sampled: counts come from cumulative sums over the
score-sorted arrays, so a 500-point sweep over 88k rows costs one sort.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CostConfig

SWEEP_COLUMNS = [
    "threshold",
    "n_alerts",
    "alert_rate",
    "review_hours",
    "tp",
    "fp",
    "fn",
    "tn",
    "precision",
    "recall",
    "f1",
    "false_alarm_rate",
    "fraud_caught_dollars",
    "fraud_missed_dollars",
    "dollar_recall",
]


def _threshold_grid(scores: np.ndarray, n_points: int) -> np.ndarray:
    """A grid dense where the scores actually are.

    A uniform grid on [0, 1] wastes most of its points on empty score ranges
    when the distribution piles up near zero, which it always does under class
    imbalance. Mixing in score quantiles keeps resolution where operating points
    live.
    """
    uniform = np.linspace(0.0, 1.0, n_points // 2)
    qs = np.linspace(0.0, 1.0, n_points - n_points // 2)
    quantile = np.quantile(scores, qs)
    grid = np.unique(np.concatenate([uniform, quantile, [0.0, 1.0 + 1e-12]]))
    return grid


def sweep_thresholds(
    y_true: np.ndarray,
    scores: np.ndarray,
    amounts: np.ndarray,
    cost: CostConfig | None = None,
    n_points: int = 500,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Compute the operational table across the threshold range.

    Parameters
    ----------
    y_true, scores, amounts
        Aligned arrays for one evaluation window. ``amounts`` drives the dollar
        columns and the false-negative cost.
    cost
        Supplies ``minutes_per_review``, which converts alert volume into
        analyst hours. Default 4 minutes per review.
    """
    cost = cost or CostConfig()
    y = np.asarray(y_true).astype(np.int64)
    s = np.asarray(scores, dtype=np.float64)
    a = np.asarray(amounts, dtype=np.float64)
    if not (len(y) == len(s) == len(a)):
        raise ValueError(
            f"Length mismatch: y={len(y)}, scores={len(s)}, amounts={len(a)}."
        )
    n = len(y)
    total_pos = int(y.sum())
    total_fraud_dollars = float(a[y == 1].sum())

    grid = _threshold_grid(s, n_points) if thresholds is None else np.asarray(thresholds, float)

    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    a_sorted = a[order]
    cum_tp = np.concatenate([[0], np.cumsum(y_sorted)])
    cum_tp_dollars = np.concatenate([[0.0], np.cumsum(y_sorted * a_sorted)])

    asc = np.sort(s)
    n_alerts = n - np.searchsorted(asc, grid, side="left")
    n_alerts = np.clip(n_alerts, 0, n)

    tp = cum_tp[n_alerts]
    caught_dollars = cum_tp_dollars[n_alerts]
    fp = n_alerts - tp
    fn = total_pos - tp
    tn = n - n_alerts - fn

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(n_alerts > 0, tp / n_alerts, np.nan)
        recall = tp / total_pos if total_pos else np.full_like(tp, np.nan, dtype=float)
        f1 = np.where(
            (precision + recall) > 0, 2 * precision * recall / (precision + recall), 0.0
        )
        dollar_recall = (
            caught_dollars / total_fraud_dollars if total_fraud_dollars else np.nan
        )

    df = pd.DataFrame(
        {
            "threshold": grid,
            "n_alerts": n_alerts.astype(np.int64),
            "alert_rate": n_alerts / n,
            "review_hours": n_alerts * (cost.minutes_per_review / 60.0),
            "tp": tp.astype(np.int64),
            "fp": fp.astype(np.int64),
            "fn": fn.astype(np.int64),
            "tn": tn.astype(np.int64),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_alarm_rate": 1.0 - precision,
            "fraud_caught_dollars": caught_dollars,
            "fraud_missed_dollars": total_fraud_dollars - caught_dollars,
            "dollar_recall": dollar_recall,
        }
    )
    return df[SWEEP_COLUMNS]


def best_f1_row(sweep: pd.DataFrame) -> pd.Series:
    """The F1-maximising operating point."""
    valid = sweep.dropna(subset=["f1"])
    return valid.loc[valid["f1"].idxmax()]


def threshold_at_alert_volume(sweep: pd.DataFrame, n_alerts: int) -> pd.Series:
    """The sweep row whose alert volume is closest to ``n_alerts``.

    Matched-alert-volume comparison is the only fair way to compare two scoring
    models operationally: it holds the staffing cost fixed and asks which model
    spends it better.
    """
    idx = (sweep["n_alerts"] - n_alerts).abs().idxmin()
    return sweep.loc[idx]


def threshold_for_capacity(sweep: pd.DataFrame, analyst_hours: int | float) -> pd.Series:
    """The lowest threshold whose review load fits within ``analyst_hours``.

    This is the staffing-constrained question: given the team I have, how low
    can I set the bar?
    """
    feasible = sweep[sweep["review_hours"] <= analyst_hours]
    if feasible.empty:
        return sweep.loc[sweep["review_hours"].idxmin()]
    return feasible.loc[feasible["threshold"].idxmin()]


def operational_table(
    sweep: pd.DataFrame, recall_targets: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9)
) -> pd.DataFrame:
    """A short table keyed by recall target -- the form a review manager reads.

    Rather than dumping 500 rows, answer "what does it cost me to catch 70% of
    fraud?" for a handful of targets.
    """
    rows = []
    for target in recall_targets:
        feasible = sweep[sweep["recall"] >= target]
        if feasible.empty:
            continue
        # Cheapest way to hit the target: fewest alerts among rows that reach it.
        row = feasible.loc[feasible["n_alerts"].idxmin()].copy()
        row["recall_target"] = target
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["recall_target", *SWEEP_COLUMNS])
    out = pd.DataFrame(rows).reset_index(drop=True)
    return out[["recall_target", *SWEEP_COLUMNS]]
