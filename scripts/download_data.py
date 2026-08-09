"""Fetch the IEEE-CIS training tables from Kaggle.

Requires a Kaggle API token at ``~/.kaggle/kaggle.json`` and acceptance of the
competition rules at https://www.kaggle.com/c/ieee-fraud-detection/rules .
Without the rules acceptance the API returns HTTP 403 with an explanatory
message, which this script surfaces rather than swallowing.

Only the *labelled* training tables are downloaded. The competition test files
have no public labels, so they cannot be used for evaluation here -- the
time-aware split is carved out of the labelled file instead.
"""

from __future__ import annotations

import io
import json
import sys
import urllib.request
import zipfile
from base64 import b64encode
from pathlib import Path

from ftlab.config import DATA_RAW

BASE = "https://www.kaggle.com/api/v1/competitions/data/download/ieee-fraud-detection"
FILES = ("train_transaction.csv", "train_identity.csv")


def _credentials() -> tuple[str, str]:
    path = Path.home() / ".kaggle" / "kaggle.json"
    if not path.exists():
        sys.exit(
            f"No Kaggle token at {path}. Create one at kaggle.com/settings "
            "-> API -> Create New Token."
        )
    cfg = json.loads(path.read_text())
    return cfg["username"], cfg["key"]


def download(name: str, dest: Path) -> None:
    user, key = _credentials()
    auth = b64encode(f"{user}:{key}".encode()).decode()
    req = urllib.request.Request(
        f"{BASE}/{name}", headers={"Authorization": f"Basic {auth}"}
    )
    try:
        with urllib.request.urlopen(req) as resp:
            payload = resp.read()
    except urllib.error.HTTPError as exc:  # noqa: F821
        body = exc.read().decode(errors="replace")
        if exc.code == 403:
            sys.exit(
                f"Kaggle returned 403 for {name}. Accept the competition rules at "
                "https://www.kaggle.com/c/ieee-fraud-detection/rules and retry.\n"
                f"Server said: {body}"
            )
        sys.exit(f"Kaggle returned HTTP {exc.code} for {name}: {body}")

    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        zf.extractall(dest)
    print(f"  {name} -> {dest / name}")


def main() -> None:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = DATA_RAW / name
        if target.exists():
            print(f"  {name} already present, skipping")
            continue
        print(f"  downloading {name} ...")
        download(name, DATA_RAW)
    print("Raw data ready in", DATA_RAW)


if __name__ == "__main__":
    main()
