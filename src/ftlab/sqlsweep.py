"""Run the SQL threshold sweep and compare it against the Python one.

The point of this module is cross-implementation verification. The threshold
sweep is the single most consequential computation in the project -- every
operational number is read off it -- so it is implemented independently in SQL
and checked against the Python result on the real scored data. Two
implementations agreeing on 88,581 rows is much stronger evidence of
correctness than either one passing its own unit tests.

The SAS program in ``sas/`` implements the same algorithm and cannot be
executed here (no licence on the build machine). Verifying the SQL is what
makes the shared logic tested rather than merely written down.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROOT, CostConfig

SQL_PATH = ROOT / "sql" / "threshold_sweep.sql"


def render_sql(
    scored_csv: Path,
    cost: CostConfig | None = None,
    score_col: str = "score_raw",
    sql_path: Path | None = None,
) -> str:
    """Substitute parameters into the SQL template."""
    cost = cost or CostConfig()
    template = (sql_path or SQL_PATH).read_text()
    return (
        template.replace("{scored}", str(scored_csv))
        .replace("{cost_fp}", repr(cost.cost_per_false_positive))
        .replace("{minutes_review}", repr(cost.minutes_per_review))
        .replace("{score_col}", score_col)
    )


def run_sql_sweep(
    scored_csv: Path,
    cost: CostConfig | None = None,
    score_col: str = "score_raw",
) -> pd.DataFrame:
    """Execute the SQL sweep in DuckDB and return the result."""
    import duckdb

    sql = render_sql(scored_csv, cost, score_col)
    con = duckdb.connect()
    try:
        return con.execute(sql).fetch_df()
    finally:
        con.close()


def sql_row_at_alert_volume(sweep: pd.DataFrame, n_alerts: int) -> pd.Series:
    """The sweep row closest to a given alert volume."""
    idx = (sweep["n_alerts"] - n_alerts).abs().idxmin()
    return sweep.loc[idx]


def cost_optimal_row(sweep: pd.DataFrame) -> pd.Series:
    return sweep.loc[sweep["total_cost"].idxmin()]


def read_scored(scored_csv: Path) -> pd.DataFrame:
    """Read the scored CSV with a round-trip-exact float parser.

    ``float_precision="round_trip"`` is load-bearing, not decoration. pandas'
    default CSV float parser is fast but not correctly rounded: on this file it
    produces doubles differing from DuckDB's in the last few bits, which is
    enough to move a threshold comparison across a row boundary and make the
    two implementations disagree on alert counts by one. That looked exactly
    like an off-by-one bug in the sweep logic and is not -- it is a CSV parsing
    difference. With round-trip parsing the two engines agree bit for bit.
    """
    return pd.read_csv(scored_csv, float_precision="round_trip")


def compare_with_python(
    scored_csv: Path,
    cost: CostConfig | None = None,
    score_col: str = "score_raw",
    n_checks: int = 25,
) -> dict:
    """Compare the two implementations at identical thresholds.

    Matching on threshold rather than alert volume is what makes this an
    equality test: the SQL walks all 87,526 achievable operating points while
    the Python sweep evaluates a 501-point grid, so comparing "closest row"
    compares different operating points and proves nothing.
    """
    from .threshold import sweep_thresholds

    cost = cost or CostConfig()
    scored = read_scored(scored_csv)
    sql = run_sql_sweep(scored_csv, cost, score_col)

    # Sample cut points spread across the whole sweep, then evaluate the Python
    # implementation at exactly those thresholds.
    picks = sql.iloc[
        np.linspace(0, len(sql) - 1, min(n_checks, len(sql))).astype(int)
    ].reset_index(drop=True)
    thresholds = picks["threshold"].to_numpy()

    py = sweep_thresholds(
        scored["isFraud"].to_numpy(),
        scored[score_col].to_numpy(),
        scored["TransactionAmt"].to_numpy(dtype=float),
        cost,
        thresholds=thresholds,
    ).reset_index(drop=True)

    merged = picks.merge(py, on="threshold", suffixes=("_sql", "_py"))
    checks = {
        "n_alerts": bool((merged["n_alerts_sql"] == merged["n_alerts_py"]).all()),
        "tp": bool((merged["tp_sql"] == merged["tp_py"]).all()),
        "fp": bool((merged["fp_sql"] == merged["fp_py"]).all()),
        "fn": bool((merged["fn_sql"] == merged["fn_py"]).all()),
        "fraud_caught_dollars": bool(
            np.allclose(
                merged["fraud_caught_dollars_sql"],
                merged["fraud_caught_dollars_py"],
                rtol=1e-9,
            )
        ),
        "precision": bool(
            np.allclose(merged["precision_sql"], merged["precision_py"], rtol=1e-9)
        ),
        "recall": bool(
            np.allclose(merged["recall_sql"], merged["recall_py"], rtol=1e-9)
        ),
    }

    sql_best = cost_optimal_row(sql)
    return {
        "thresholds_compared": int(len(merged)),
        "all_match": all(checks.values()),
        "checks": checks,
        "sql_operating_points": int(len(sql)),
        "python_grid_points": 501,
        "sql_cost_optimal": {
            "threshold": float(sql_best["threshold"]),
            "n_alerts": int(sql_best["n_alerts"]),
            "recall": float(sql_best["recall"]),
            "total_cost": float(sql_best["total_cost"]),
        },
    }
