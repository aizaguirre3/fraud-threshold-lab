"""Phase 1.1 -- load, join, profile, and split. Writes artifacts/tables/data_profile.json."""

from __future__ import annotations

import json

from ftlab import config
from ftlab.data import load_joined, write_report
from ftlab.split import make_time_split


def main() -> None:
    config.ensure_dirs()
    df, report = load_joined()
    split = make_time_split(df, config.DEFAULT.split)
    report["split"] = split.summary()

    for name, idx in (
        ("train", split.train_idx),
        ("val", split.val_idx),
        ("test", split.test_idx),
    ):
        sub = df.loc[idx]
        report["split"][f"{name}_fraud_rate_pct"] = round(
            100.0 * float(sub[config.TARGET].mean()), 3
        )
        report["split"][f"{name}_fraud_dollars"] = round(
            float(sub.loc[sub[config.TARGET] == 1, config.AMOUNT].sum()), 2
        )

    write_report(report, config.TABLES / "data_profile.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
