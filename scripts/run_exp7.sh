#!/bin/bash
set -e
cd "$(dirname "$0")/.."
echo "Running exp7: Self-training v2 (domain adaptation)"
mkdir -p experiments/exp7_outputs
uv run python experiments/exp7_st_v2.py 2>&1 | tee experiments/exp7_outputs/exp7_log.txt
echo "Done. Outputs in experiments/exp7_outputs/"
