"""Quantify what a naive validation setup buys you in illusory performance.

The README claims a random split inflates every metric. This script measures it
instead of asserting it, and decomposes the inflation into two independent
mistakes that a careless pipeline makes together:

A. **Time-aware split, aggregates fit on the training window.**
   The honest setup, and what the rest of this project reports.

B. **Random split, aggregates fit on the random training subset.**
   Isolates the *temporal* leak alone. The model scores a card's early
   transactions having already seen its later ones, because rows from the same
   card land on both sides of a random split.

C. **Random split, aggregates fit on the entire dataset.**
   The fully naive pipeline: compute per-card statistics up front, then split.
   Adds a *feature provenance* leak on top of the temporal one, so every
   training row's features are partly built from test-period behaviour.

D. **Time-aware split, aggregates fit on the entire dataset.**
   The realistic near-miss, and the reason this project carries provenance on
   its aggregates at all. An analyst who correctly splits on time can still
   compute per-card statistics before splitting and never notice. Isolating this
   cell is what says whether the guard earns its complexity.

All four use identical features, model configuration, seed, and split
proportions. The only differences are how rows are assigned and what window the
behavioural aggregates were fit from.
"""

from __future__ import annotations

import json
import time
import warnings

import numpy as np
import pandas as pd

from ftlab import config, models
from ftlab.config import TARGET
from ftlab.data import load_joined
from ftlab.features import (
    align_categories,
    apply_behavioral_aggregates,
    add_amount_features,
    add_time_features,
    fit_behavioral_aggregates,
    to_model_matrix,
)
from ftlab.split import make_time_split


def _silence_accelerate() -> None:
    for msg in ("divide by zero encountered in matmul",
                "overflow encountered in matmul",
                "invalid value encountered in matmul"):
        warnings.filterwarnings("ignore", message=msg, category=RuntimeWarning)


def _build(df: pd.DataFrame, maps) -> pd.DataFrame:
    """Feature pipeline without the provenance assertion, so the leaky variants
    can be constructed deliberately."""
    out = add_time_features(df)
    out = add_amount_features(out)
    return apply_behavioral_aggregates(out, maps)


def _fit_and_score(feats, tr_idx, va_idx, te_idx, cfg, label):
    tr, va, te = feats.loc[tr_idx], feats.loc[va_idx], feats.loc[te_idx]
    X_tr, cats = to_model_matrix(tr)
    X_va_raw, _ = to_model_matrix(va, cats)
    X_te_raw, _ = to_model_matrix(te, cats)
    X_va, X_te = align_categories(X_tr, X_va_raw, X_te_raw)

    fit = models.fit_lightgbm(
        X_tr, tr[TARGET].to_numpy(),
        X_va, va[TARGET].to_numpy(),
        X_te, te[TARGET].to_numpy(),
        cfg.model, name=label,
    )
    return fit.metrics["test"]


def main() -> None:
    _silence_accelerate()
    cfg = config.DEFAULT
    config.ensure_dirs()
    t0 = time.time()

    print("Loading data ...")
    df, _ = load_joined()
    tsplit = make_time_split(df, cfg.split)

    # Random split with the same proportions, for a like-for-like comparison.
    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(df.index.to_numpy())
    n = len(perm)
    n_tr = len(tsplit.train_idx)
    n_va = len(tsplit.val_idx)
    r_tr = pd.Index(np.sort(perm[:n_tr]))
    r_va = pd.Index(np.sort(perm[n_tr:n_tr + n_va]))
    r_te = pd.Index(np.sort(perm[n_tr + n_va:]))

    results = {}

    print("[A] time-aware split, aggregates fit on the training window ...")
    maps_a = fit_behavioral_aggregates(df.loc[tsplit.train_idx])
    feats_a = _build(df, maps_a)
    results["A_time_split_honest"] = _fit_and_score(
        feats_a, tsplit.train_idx, tsplit.val_idx, tsplit.test_idx, cfg, "A")

    print("[B] random split, aggregates fit on the random training subset ...")
    maps_b = fit_behavioral_aggregates(df.loc[r_tr])
    feats_b = _build(df, maps_b)
    results["B_random_split_clean_aggregates"] = _fit_and_score(
        feats_b, r_tr, r_va, r_te, cfg, "B")

    print("[C] random split, aggregates fit on the entire dataset ...")
    maps_c = fit_behavioral_aggregates(df)
    feats_c = _build(df, maps_c)
    results["C_random_split_leaky_aggregates"] = _fit_and_score(
        feats_c, r_tr, r_va, r_te, cfg, "C")

    print("[D] time-aware split, aggregates fit on the entire dataset ...")
    # feats_c already carries whole-dataset aggregates; only the split differs.
    results["D_time_split_leaky_aggregates"] = _fit_and_score(
        feats_c, tsplit.train_idx, tsplit.val_idx, tsplit.test_idx, cfg, "D")

    honest = results["A_time_split_honest"]
    summary = {
        "variants": results,
        "inflation_vs_honest": {
            k: {
                "roc_auc_absolute": v["roc_auc"] - honest["roc_auc"],
                "pr_auc_absolute": v["pr_auc"] - honest["pr_auc"],
                "pr_auc_relative_pct": 100.0 * (v["pr_auc"] - honest["pr_auc"]) / honest["pr_auc"],
            }
            for k, v in results.items() if k != "A_time_split_honest"
        },
        "split_sizes": {
            "train": int(n_tr), "val": int(n_va), "test": int(n - n_tr - n_va),
        },
        "seed": cfg.seed,
        "runtime_seconds": round(time.time() - t0, 1),
    }

    (config.TABLES / "split_comparison.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 78)
    print(f"{'variant':<40} {'ROC AUC':>9} {'PR AUC':>9} {'PR vs honest':>14}")
    print("-" * 78)
    for k, v in results.items():
        delta = "" if k == "A_time_split_honest" else \
            f"{100.0 * (v['pr_auc'] - honest['pr_auc']) / honest['pr_auc']:+.1f}%"
        print(f"{k:<40} {v['roc_auc']:>9.4f} {v['pr_auc']:>9.4f} {delta:>14}")
    print("=" * 78)
    print(f"\nDone in {summary['runtime_seconds']}s -> artifacts/tables/split_comparison.json")


if __name__ == "__main__":
    main()
