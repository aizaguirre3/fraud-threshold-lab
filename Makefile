PY := .venv/bin/python
UV := uv

.PHONY: all setup data profile run split-comparison test lint clean

## Full reproduction from raw data to every reported number.
all: setup data run test

setup:
	$(UV) venv --python 3.12
	$(UV) pip install -e ".[dev]"

## Download IEEE-CIS. Requires ~/.kaggle/kaggle.json and acceptance of the
## competition rules at kaggle.com/c/ieee-fraud-detection/rules.
data:
	$(PY) scripts/download_data.py

profile:
	$(PY) scripts/profile_data.py

run:
	$(PY) scripts/run_pipeline.py

## Measure what a naive validation setup would have reported.
split-comparison:
	$(PY) scripts/split_comparison.py

test:
	$(PY) -m pytest

clean:
	rm -rf artifacts/figures/* artifacts/tables/* data/processed/*
