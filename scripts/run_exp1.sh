#!/bin/bash
set -e
cd "$(dirname "$0")/.."
echo "Running exp1: EDA"
uv run python experiments/exp1_eda.py
echo "Done. Outputs in experiments/exp1_outputs/"
