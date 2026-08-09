"""Cost-sensitive threshold selection.

The F1-optimal threshold treats a missed fraud and a false alarm as equally
regrettable. They are not. A missed fraud costs the transaction amount; a false
alarm costs a few minutes of analyst time plus whatever the customer friction is
worth. At a 3.5% positive rate those two are separated by more than an order of
magnitude, so the two optima land in different places.

Total expected cost at threshold t:

    C(t) = sum(amount of fraud missed at t) + n_false_positives(t) * c_fp

where ``c_fp`` = analyst review labour + customer friction proxy.

The friction term is an assumption nobody can measure precisely, which is why
:func:`cost_ratio_sensitivity` exists. The finding that matters is not the
optimal threshold itself but how much it moves when the assumption changes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CostConfig
from .threshold import best_f1_row, sweep_thresholds


def add_cost_columns(
    sweep: pd.DataFrame,
    cost: CostConfig | None = None,
    *,
    cost_per_fp: float | None = None,
) -> pd.DataFrame:
    """Attach expected-cost columns to a threshold sweep.

    ``cost_per_fp`` overrides the configured value, which the sensitivity sweep
    uses to vary the assumption without rebuilding the sweep.
    """
    cost = cost or CostConfig()
    c_fp = cost.cost_per_false_positive if cost_per_fp is None else cost_per_fp
    out = sweep.copy()
    out["cost_per_fp_used"] = c_fp
    out["fp_cost"] = out["fp"] * c_fp
    out["fn_cost"] = out["fraud_missed_dollars"]
    out["review_labour_cost"] = out["review_hours"] * cost.analyst_hourly_cost
    out["total_cost"] = out["fp_cost"] + out["fn_cost"]
    return out


def optimal_threshold(
    sweep: pd.DataFrame,
    cost: CostConfig | None = None,
    *,
    cost_per_fp: float | None = None,
) -> pd.Series:
    """The cost-minimising operating point."""
    scored = add_cost_columns(sweep, cost, cost_per_fp=cost_per_fp)
    return scored.loc[scored["total_cost"].idxmin()]


def compare_cost_vs_f1(
    sweep: pd.DataFrame, cost: CostConfig | None = None
) -> dict:
    """Quantify, in dollars, the gap between the cost-optimal and F1-optimal points.

    They will not coincide. The size and direction of the gap is the whole point:
    it says what optimising a symmetric accuracy metric costs the business.
    """
    cost = cost or CostConfig()
    scored = add_cost_columns(sweep, cost)
    cost_row = scored.loc[scored["total_cost"].idxmin()]
    f1_row = best_f1_row(scored)

    return {
        "cost_optimal": {
            "threshold": float(cost_row["threshold"]),
            "n_alerts": int(cost_row["n_alerts"]),
            "review_hours": float(cost_row["review_hours"]),
            "recall": float(cost_row["recall"]),
            "precision": float(cost_row["precision"]),
            "total_cost": float(cost_row["total_cost"]),
            "fraud_missed_dollars": float(cost_row["fraud_missed_dollars"]),
            "fp_cost": float(cost_row["fp_cost"]),
        },
        "f1_optimal": {
            "threshold": float(f1_row["threshold"]),
            "n_alerts": int(f1_row["n_alerts"]),
            "review_hours": float(f1_row["review_hours"]),
            "recall": float(f1_row["recall"]),
            "precision": float(f1_row["precision"]),
            "total_cost": float(f1_row["total_cost"]),
            "fraud_missed_dollars": float(f1_row["fraud_missed_dollars"]),
            "fp_cost": float(f1_row["fp_cost"]),
        },
        "cost_penalty_of_using_f1_threshold": float(
            f1_row["total_cost"] - cost_row["total_cost"]
        ),
        "extra_alerts_from_cost_threshold": int(
            cost_row["n_alerts"] - f1_row["n_alerts"]
        ),
        "extra_review_hours_from_cost_threshold": float(
            cost_row["review_hours"] - f1_row["review_hours"]
        ),
        "assumptions": {
            "analyst_hourly_cost": cost.analyst_hourly_cost,
            "minutes_per_review": cost.minutes_per_review,
            "review_cost_per_alert": round(cost.review_cost_per_alert, 4),
            "friction_cost_per_fp": cost.friction_cost_per_fp,
            "cost_per_false_positive": round(cost.cost_per_false_positive, 4),
            "false_negative_cost": "actual transaction amount",
        },
    }


def cost_ratio_sensitivity(
    sweep: pd.DataFrame,
    cost: CostConfig | None = None,
    fp_costs: tuple[float, ...] | None = None,
) -> pd.DataFrame:
    """Sweep the false-positive cost and watch where the optimum goes.

    Reported alongside the mean fraud amount, because the meaningful quantity is
    the *ratio* of false-positive cost to typical fraud loss, not either alone.
    """
    cost = cost or CostConfig()
    if fp_costs is None:
        fp_costs = (1.0, 2.5, 5.0, 7.33, 10.0, 15.0, 25.0, 50.0, 100.0)

    rows = []
    for c_fp in fp_costs:
        row = optimal_threshold(sweep, cost, cost_per_fp=c_fp)
        rows.append(
            {
                "cost_per_fp": c_fp,
                "threshold": float(row["threshold"]),
                "n_alerts": int(row["n_alerts"]),
                "alert_rate": float(row["alert_rate"]),
                "review_hours": float(row["review_hours"]),
                "recall": float(row["recall"]),
                "precision": float(row["precision"]),
                "fraud_missed_dollars": float(row["fraud_missed_dollars"]),
                "total_cost": float(row["total_cost"]),
            }
        )
    out = pd.DataFrame(rows)
    return out


def sensitivity_summary(sens: pd.DataFrame) -> dict:
    """How stable is the recommendation across the plausible cost range?"""
    return {
        "fp_cost_range": [float(sens["cost_per_fp"].min()), float(sens["cost_per_fp"].max())],
        "fp_cost_fold_range": float(
            sens["cost_per_fp"].max() / max(sens["cost_per_fp"].min(), 1e-9)
        ),
        "threshold_range": [float(sens["threshold"].min()), float(sens["threshold"].max())],
        "recall_range": [float(sens["recall"].min()), float(sens["recall"].max())],
        "alert_rate_range": [
            float(sens["alert_rate"].min()),
            float(sens["alert_rate"].max()),
        ],
        "review_hours_range": [
            float(sens["review_hours"].min()),
            float(sens["review_hours"].max()),
        ],
    }


def evaluate_from_scores(
    y_true: np.ndarray,
    scores: np.ndarray,
    amounts: np.ndarray,
    cost: CostConfig | None = None,
    n_points: int = 500,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Convenience: sweep, compare against F1, and run the sensitivity analysis."""
    cost = cost or CostConfig()
    sweep = sweep_thresholds(y_true, scores, amounts, cost, n_points=n_points)
    comparison = compare_cost_vs_f1(sweep, cost)
    sens = cost_ratio_sensitivity(sweep, cost)
    return sweep, comparison, sens
