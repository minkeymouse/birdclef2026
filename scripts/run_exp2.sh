#!/bin/bash
set -e
cd "$(dirname "$0")/.."
echo "Running exp2: SED Baseline"
uv run python experiments/exp2_sed_baseline.py 2>&1 | tee experiments/exp2_outputs/exp2_log.txt
echo "Done. Outputs in experiments/exp2_outputs/"
