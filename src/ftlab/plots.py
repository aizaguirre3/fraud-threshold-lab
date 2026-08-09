"""Figures. Matplotlib only, no seaborn, no styling dependencies."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

FIGSIZE = (9, 5.5)
DPI = 140


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_threshold_curves(sweep: pd.DataFrame, path: Path, operating: float | None = None) -> Path:
    """Precision, recall and F1 against threshold, with alert volume on a twin axis."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(sweep["threshold"], sweep["precision"], label="Precision", lw=2)
    ax.plot(sweep["threshold"], sweep["recall"], label="Recall", lw=2)
    ax.plot(sweep["threshold"], sweep["f1"], label="F1", lw=2, ls="--")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Precision / Recall / F1")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(sweep["threshold"], sweep["n_alerts"], color="grey", alpha=0.6, label="Alerts")
    ax2.set_ylabel("Alert volume (transactions)")
    ax2.set_yscale("symlog")

    if operating is not None:
        ax.axvline(operating, color="crimson", ls=":", lw=1.5)
        ax.text(operating, 1.0, "  operating point", color="crimson", fontsize=8, va="top")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center right", fontsize=9)
    ax.set_title("Detection quality and alert volume across the threshold range")
    return _save(fig, path)


def plot_review_hours(sweep: pd.DataFrame, path: Path) -> Path:
    """The staffing view: fraud caught in dollars against analyst hours required."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    s = sweep.sort_values("review_hours")
    ax.plot(s["review_hours"], s["fraud_caught_dollars"], lw=2)
    ax.set_xlabel("Analyst review hours required (test window)")
    ax.set_ylabel("Fraud dollars caught")
    ax.grid(alpha=0.3)
    ax.set_title("Fraud dollars recovered per analyst hour invested")
    ax.ticklabel_format(style="plain", axis="y")
    return _save(fig, path)


def plot_cost_curve(scored: pd.DataFrame, path: Path) -> Path:
    """Total expected cost against threshold, decomposed into its two terms."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(scored["threshold"], scored["total_cost"], lw=2, label="Total expected cost")
    ax.plot(scored["threshold"], scored["fn_cost"], lw=1.5, ls="--", label="Missed fraud ($)")
    ax.plot(scored["threshold"], scored["fp_cost"], lw=1.5, ls="--", label="False alarm cost ($)")
    best = scored.loc[scored["total_cost"].idxmin()]
    ax.axvline(best["threshold"], color="crimson", ls=":", lw=1.5)
    ax.plot([best["threshold"]], [best["total_cost"]], "o", color="crimson")
    ax.annotate(
        f"min cost ${best['total_cost']:,.0f}\n@ t={best['threshold']:.4f}",
        xy=(best["threshold"], best["total_cost"]),
        xytext=(12, 24),
        textcoords="offset points",
        fontsize=9,
        color="crimson",
    )
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Expected cost (USD, test window)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title("Cost-sensitive threshold selection")
    return _save(fig, path)


def plot_cost_sensitivity(sens: pd.DataFrame, path: Path) -> Path:
    """How far the recommended operating point moves as the FP cost assumption varies."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    ax1.plot(sens["cost_per_fp"], sens["threshold"], "o-", lw=2)
    ax1.set_xscale("log")
    ax1.set_xlabel("Assumed cost per false positive (USD, log scale)")
    ax1.set_ylabel("Cost-optimal threshold")
    ax1.grid(alpha=0.3)
    ax1.set_title("Optimal threshold vs FP cost assumption")

    ax2.plot(sens["cost_per_fp"], sens["recall"], "o-", lw=2, label="Recall")
    ax2.plot(sens["cost_per_fp"], sens["alert_rate"], "s-", lw=2, label="Alert rate")
    ax2.set_xscale("log")
    ax2.set_xlabel("Assumed cost per false positive (USD, log scale)")
    ax2.set_ylabel("Rate")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=9)
    ax2.set_title("Operating consequences of the same sweep")
    fig.suptitle("Sensitivity of the recommendation to an unmeasurable assumption", y=1.02)
    return _save(fig, path)


def plot_reliability(curves: dict[str, pd.DataFrame], path: Path) -> Path:
    """Reliability diagram for raw and recalibrated scores.

    Two panels, because one is not enough under class imbalance. On a linear
    axis, 19 of 20 equal-count bins fall below 0.08 and the informative region
    collapses against the origin. The log-log panel is where the behaviour at
    the scores that actually occur is legible; the linear panel keeps the
    high-score tail in view and stops the log axis from exaggerating it.
    """
    fig, (ax, axl) = plt.subplots(1, 2, figsize=(12.5, 5.8))

    lim = 0.0
    for name, c in curves.items():
        if c.empty:
            continue
        ax.plot(c["mean_predicted"], c["observed_rate"], "o-", lw=1.8, ms=4, label=name)
        lim = max(lim, float(c["mean_predicted"].max()), float(c["observed_rate"].max()))
    lim = min(1.0, lim * 1.1) if lim > 0 else 1.0
    ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.6, label="perfect calibration")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraud rate")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title("Linear scale (full range)")

    lo = 1.0
    for name, c in curves.items():
        if c.empty:
            continue
        m = (c["mean_predicted"] > 0) & (c["observed_rate"] > 0)
        axl.plot(c.loc[m, "mean_predicted"], c.loc[m, "observed_rate"],
                 "o-", lw=1.8, ms=4, label=name)
        if m.any():
            lo = min(lo, float(c.loc[m, "mean_predicted"].min()),
                     float(c.loc[m, "observed_rate"].min()))
    lo = max(lo * 0.7, 1e-6)
    axl.plot([lo, lim], [lo, lim], "k--", lw=1, alpha=0.6, label="perfect calibration")
    axl.set_xscale("log")
    axl.set_yscale("log")
    axl.set_xlim(lo, lim)
    axl.set_ylim(lo, lim)
    axl.set_xlabel("Mean predicted probability (log)")
    axl.set_ylabel("Observed fraud rate (log)")
    axl.grid(alpha=0.3, which="both")
    axl.legend(fontsize=9)
    axl.set_title("Log scale (where 19 of 20 bins live)")

    fig.suptitle(
        "Reliability diagram, test window (20 equal-count bins). "
        "Points above the diagonal mean the model under-predicts risk.",
        y=1.02,
    )
    return _save(fig, path)


def plot_pr_curves(curves: dict[str, tuple[np.ndarray, np.ndarray]], baseline: float, path: Path) -> Path:
    """Precision-recall curves, with the no-skill baseline drawn in."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for name, (recall, precision) in curves.items():
        ax.plot(recall, precision, lw=2, label=name)
    ax.axhline(baseline, color="k", ls="--", lw=1, alpha=0.6,
               label=f"no-skill baseline ({baseline:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title("Precision-recall, test window")
    return _save(fig, path)


def plot_fp_lift(lift: pd.DataFrame, path: Path, top_n: int = 15) -> Path:
    """Which legitimate populations absorb the false alarms."""
    if lift.empty:
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.text(0.5, 0.5, "No segments met the minimum size threshold",
                ha="center", va="center")
        ax.axis("off")
        return _save(fig, path)
    top = lift.head(top_n).iloc[::-1]
    labels = [f"{r.attribute}={r.value}" for r in top.itertuples()]
    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.35 * len(top))))
    ax.barh(labels, top["lift_vs_base"], color="steelblue")
    ax.axvline(1.0, color="k", ls="--", lw=1)
    ax.set_xlabel("False-alarm rate relative to base rate (1.0 = base)")
    ax.grid(alpha=0.3, axis="x")
    ax.set_title("Legitimate populations most over-flagged at the operating threshold")
    return _save(fig, path)
