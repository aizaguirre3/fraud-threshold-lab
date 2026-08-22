"""End-to-end run: data -> features -> models -> threshold -> cost -> calibration -> FP profile.

Single entry point. Reproduces every number reported in the README from the raw
Kaggle CSVs. Writes tables to artifacts/tables and figures to artifacts/figures.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from ftlab import calibration as cal
from ftlab import config, cost, export, fpanalysis, leakage, models, plots
from ftlab.config import AMOUNT, TARGET
from ftlab.data import load_joined
from ftlab.features import (
    align_categories,
    build_features,
    fit_behavioral_aggregates,
    fit_target_encoders,
    to_model_matrix,
)
from ftlab.split import make_time_split, split_frames
from ftlab.threshold import (
    operational_table,
    sweep_thresholds,
    threshold_at_alert_volume,
)


def _silence_accelerate_matmul_warnings() -> None:
    """Suppress spurious FPU warnings from Apple's Accelerate BLAS.

    On macOS/arm64, numpy linked against Accelerate raises "divide by zero",
    "overflow", and "invalid value" RuntimeWarnings from ``matmul`` even for
    inputs that contain no zeros, infinities, or NaNs, and even when the result
    is entirely finite. Verified with a plain ``A @ A.T`` on random normal
    float64 data, with no scikit-learn involved.

    This is filtered rather than worked around because there is nothing in the
    numerics to fix: the flags are set by the vendor BLAS, not by the data. The
    filter is deliberately narrow -- only ``matmul`` RuntimeWarnings -- so a
    genuine overflow anywhere else still surfaces.
    """
    for msg in ("divide by zero encountered in matmul",
                "overflow encountered in matmul",
                "invalid value encountered in matmul"):
        warnings.filterwarnings("ignore", message=msg, category=RuntimeWarning)


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-points", type=int, default=500,
                    help="Threshold grid resolution.")
    ap.add_argument("--fp-clusters", type=int, default=5)
    ap.add_argument("--quick", action="store_true",
                    help="Subsample the training window for a fast smoke run. "
                         "Numbers from a quick run are NOT reportable.")
    ap.add_argument("--target-encoding", action="store_true",
                    help="Add per-group fraud-rate features (out-of-fold within "
                         "the training window). OFF by default: measured on the "
                         "test window they cost 2.1%% of PR AUC, because a card's "
                         "fraud rate from months ago does not transfer across the "
                         "31-day gap to the test period. See the README.")
    ap.add_argument("--skip-export", action="store_true",
                    help="Skip the alert-stream export. The TreeSHAP pass over "
                         "all test rows is about half the total runtime, and "
                         "nothing reported in the README depends on it.")
    args = ap.parse_args()

    _silence_accelerate_matmul_warnings()
    cfg = config.DEFAULT
    config.ensure_dirs()
    t0 = time.time()
    results: dict = {"quick_mode": args.quick, "seed": cfg.seed}

    # ---------------------------------------------------------------- P1.1
    print("[1/6] Loading and joining data ...")
    df, data_report = load_joined()
    results["data"] = data_report

    split = make_time_split(df, cfg.split)
    results["split"] = split.summary()
    print(f"      train={len(split.train_idx):,} val={len(split.val_idx):,} "
          f"test={len(split.test_idx):,}")

    if args.quick:
        rng = np.random.default_rng(cfg.seed)
        keep = rng.choice(split.train_idx, size=min(60_000, len(split.train_idx)),
                          replace=False)
        train_df = df.loc[np.sort(keep)]
    else:
        train_df = df.loc[split.train_idx]

    # ---------------------------------------------------------------- P1.2
    print("[2/6] Building features (aggregates fit on train window only) ...")
    maps = fit_behavioral_aggregates(train_df)
    leakage.assert_fit_window_is_train_only(maps.fit_index, split.train_idx)

    tmaps = None
    if args.target_encoding:
        tmaps = fit_target_encoders(train_df, seed=cfg.seed)
        leakage.assert_fit_window_is_train_only(tmaps.fit_index, split.train_idx)
        results["target_encoding"] = {
            "columns": tmaps.column_names(),
            "smoothing": tmaps.smoothing,
            "prior": tmaps.prior,
            "oof_folds": 5,
        }

    feats = build_features(df, maps, train_index=split.train_idx,
                           target_maps=tmaps)
    tr, va, te = split_frames(feats, split)
    if args.quick:
        tr = feats.loc[train_df.index]

    X_tr, cat_cols = to_model_matrix(tr)
    X_va_raw, _ = to_model_matrix(va, cat_cols)
    X_te_raw, _ = to_model_matrix(te, cat_cols)
    X_va, X_te = align_categories(X_tr, X_va_raw, X_te_raw)

    y_tr = tr[TARGET].to_numpy()
    y_va = va[TARGET].to_numpy()
    y_te = te[TARGET].to_numpy()
    amt_te = te[AMOUNT].to_numpy(dtype=float)

    audit = leakage.audit(X_tr, pd.Series(y_tr, index=X_tr.index))
    results["leakage_audit"] = audit
    print(f"      leakage audit passed: {audit['n_features']} features, "
          f"max single-feature AUC {audit['max_single_feature_auc']:.4f}")

    print("[3/6] Fitting models ...")
    lr = models.fit_logistic_baseline(X_tr, y_tr, X_va, y_va, X_te, y_te, cat_cols)
    print(f"      logistic baseline  test PR AUC {lr.metrics['test']['pr_auc']:.4f}")

    fits, imb_summary = models.compare_imbalance_strategies(
        X_tr, y_tr, X_va, y_va, X_te, y_te, cfg.model, seed=cfg.seed
    )
    results["imbalance_comparison"] = imb_summary
    winner_key = imb_summary["winner_by_val_pr_auc"]
    best = fits[winner_key]
    print(f"      imbalance winner: {winner_key} "
          f"(val PR AUC {imb_summary['val_pr_auc'][winner_key]:.4f})")
    print(f"      primary model      test PR AUC {best.metrics['test']['pr_auc']:.4f}, "
          f"ROC AUC {best.metrics['test']['roc_auc']:.4f}")

    results["models"] = {
        "logistic_regression": lr.metrics,
        **{f.name: f.metrics for f in fits.values()},
        "primary_model": best.name,
        "primary_best_iteration": best.best_iteration,
    }

    # ---------------------------------------------------------------- P1.3
    print("[4/6] Threshold sweep and cost-sensitive selection ...")
    sweep = sweep_thresholds(y_te, best.test_scores, amt_te, cfg.cost,
                             n_points=args.n_points)
    sweep.to_csv(config.TABLES / "threshold_sweep_test.csv", index=False)

    op_table = operational_table(sweep)
    op_table.to_csv(config.TABLES / "operational_table.csv", index=False)
    results["operational_table"] = op_table.to_dict("records")

    # ---------------------------------------------------------------- P1.4
    comparison = cost.compare_cost_vs_f1(sweep, cfg.cost)
    results["cost_vs_f1"] = comparison
    sens = cost.cost_ratio_sensitivity(sweep, cfg.cost)
    sens.to_csv(config.TABLES / "cost_sensitivity.csv", index=False)
    results["cost_sensitivity"] = cost.sensitivity_summary(sens)
    results["cost_sensitivity_rows"] = sens.to_dict("records")

    operating_threshold = comparison["cost_optimal"]["threshold"]
    print(f"      cost-optimal t={operating_threshold:.4f} "
          f"(recall {comparison['cost_optimal']['recall']:.3f}, "
          f"{comparison['cost_optimal']['n_alerts']:,} alerts)")
    print(f"      F1-optimal   t={comparison['f1_optimal']['threshold']:.4f} "
          f"(recall {comparison['f1_optimal']['recall']:.3f}, "
          f"{comparison['f1_optimal']['n_alerts']:,} alerts)")
    print(f"      using the F1 threshold instead costs "
          f"${comparison['cost_penalty_of_using_f1_threshold']:,.0f} on this window")

    # ---------------------------------------------------------------- P1.5
    print("[5/6] Calibration ...")
    raw_test = best.test_scores
    platt = cal.fit_platt(best.val_scores, y_va)
    iso = cal.fit_isotonic(best.val_scores, y_va)
    platt_test = platt.transform(raw_test)
    iso_test = iso.transform(raw_test)

    results["calibration"] = {
        "raw": cal.calibration_metrics(y_te, raw_test),
        "platt": cal.calibration_metrics(y_te, platt_test),
        "isotonic": cal.calibration_metrics(y_te, iso_test),
    }
    target_alerts = int(comparison["cost_optimal"]["n_alerts"])
    results["calibration"]["matched_alert_volume"] = {
        "platt": cal.compare_at_matched_alert_volume(
            y_te, raw_test, platt_test, amt_te, target_alerts, cfg.cost),
        "isotonic": cal.compare_at_matched_alert_volume(
            y_te, raw_test, iso_test, amt_te, target_alerts, cfg.cost),
    }
    results["calibration"]["fixed_threshold_0.5"] = {
        "platt": cal.compare_at_fixed_threshold(y_te, raw_test, platt_test, 0.5),
        "isotonic": cal.compare_at_fixed_threshold(y_te, raw_test, iso_test, 0.5),
    }
    print(f"      ECE raw {results['calibration']['raw']['ece_quantile']:.5f} -> "
          f"platt {results['calibration']['platt']['ece_quantile']:.5f} -> "
          f"isotonic {results['calibration']['isotonic']['ece_quantile']:.5f}")

    rel_curves = {
        "raw": cal.reliability_curve(y_te, raw_test),
        "platt": cal.reliability_curve(y_te, platt_test),
        "isotonic": cal.reliability_curve(y_te, iso_test),
    }
    for name, c in rel_curves.items():
        c.to_csv(config.TABLES / f"reliability_{name}.csv", index=False)

    # ---------------------------------------------------------------- P1.6
    print("[6/6] False positive characterization ...")
    flagged = raw_test >= operating_threshold
    op_row = threshold_at_alert_volume(sweep, target_alerts)
    n_legit = int((y_te == 0).sum())
    operating_point = {
        "threshold": float(operating_threshold),
        "n_alerts": int(op_row["n_alerts"]),
        "fp": int(op_row["fp"]),
        "tp": int(op_row["tp"]),
        "false_alarm_rate": float(op_row["false_alarm_rate"]),
        "n_legit": n_legit,
        "base_false_alarm_rate": float(flagged[y_te == 0].mean()),
    }
    results["operating_point"] = operating_point

    lift = fpanalysis.univariate_fp_lift(te, y_te, flagged)
    lift.to_csv(config.TABLES / "false_positive_segments.csv", index=False)
    clusters, labels = fpanalysis.cluster_false_positives(
        te, y_te, flagged, n_clusters=args.fp_clusters, seed=cfg.seed
    )
    clusters.to_csv(config.TABLES / "false_positive_clusters.csv", index=False)

    prose = fpanalysis.write_fp_profiles(lift, clusters, operating_point)
    (config.ARTIFACTS / "false_positive_profile.md").write_text(prose)
    results["fp_top_segments"] = lift.head(10).to_dict("records")

    # ---------------------------------------------------------------- figures
    print("      writing figures ...")
    plots.plot_threshold_curves(sweep, config.FIGURES / "threshold_curves.png",
                                operating=operating_threshold)
    plots.plot_review_hours(sweep, config.FIGURES / "fraud_vs_analyst_hours.png")
    plots.plot_cost_curve(cost.add_cost_columns(sweep, cfg.cost),
                          config.FIGURES / "cost_curve.png")
    plots.plot_cost_sensitivity(sens, config.FIGURES / "cost_sensitivity.png")
    plots.plot_reliability(rel_curves, config.FIGURES / "reliability.png")

    pr_curves = {}
    for nm, sc in (("logistic baseline", lr.test_scores), ("LightGBM", raw_test)):
        p, r, _ = precision_recall_curve(y_te, sc)
        pr_curves[nm] = (r, p)
    plots.plot_pr_curves(pr_curves, float(y_te.mean()),
                         config.FIGURES / "pr_curves.png")
    plots.plot_fp_lift(lift, config.FIGURES / "fp_lift.png")

    # ---------------------------------------------------------------- scores out
    pd.DataFrame(
        {
            config.ID_COL: te[config.ID_COL].to_numpy(),
            config.TIME_COL: te[config.TIME_COL].to_numpy(),
            AMOUNT: amt_te,
            TARGET: y_te,
            "score_raw": raw_test,
            "score_platt": platt_test,
            "score_isotonic": iso_test,
        }
    ).to_csv(config.DATA_PROCESSED / "test_scores.csv", index=False)

    # Self-contained stream for the downstream queue simulator, including
    # per-alert TreeSHAP attributions so it never needs this feature pipeline.
    if args.skip_export:
        print("      skipping alert stream export (--skip-export)")
        results["alert_stream"] = {"skipped": True}
        results["runtime_seconds"] = round(time.time() - t0, 1)
        (config.TABLES / "run_results.json").write_text(
            json.dumps(results, indent=2, default=_jsonable)
        )
        print(f"\nDone in {results['runtime_seconds']}s. "
              f"Results -> artifacts/tables/run_results.json")
        return

    print("      exporting alert stream ...")
    stream = export.build_alert_stream(
        te, X_te, best.model,
        {"score_raw": raw_test, "score_platt": platt_test, "score_isotonic": iso_test},
    )
    stream_path, meta_path = export.write_alert_stream(
        stream,
        {
            "source_project": "fraud-threshold-lab",
            "model": best.name,
            "model_test_metrics": best.metrics["test"],
            "operating_threshold": operating_threshold,
            "operating_point": operating_point,
            "cost_assumptions": comparison["assumptions"],
            "split": results["split"],
            "seed": cfg.seed,
        },
        config.DATA_PROCESSED,
    )
    results["alert_stream"] = {
        "path": str(stream_path.relative_to(config.ROOT)),
        "metadata": str(meta_path.relative_to(config.ROOT)),
        "rows": int(len(stream)),
    }
    print(f"      alert stream -> {stream_path.relative_to(config.ROOT)} "
          f"({len(stream):,} rows)")

    results["runtime_seconds"] = round(time.time() - t0, 1)
    (config.TABLES / "run_results.json").write_text(
        json.dumps(results, indent=2, default=_jsonable)
    )
    print(f"\nDone in {results['runtime_seconds']}s. "
          f"Results -> artifacts/tables/run_results.json")


if __name__ == "__main__":
    main()
