#!/bin/bash
set -e
cd "$(dirname "$0")/.."
echo "Running exp10: Ensemble + post-processing evaluation"
mkdir -p experiments/exp10_outputs
uv run python experiments/exp10_ensemble.py 2>&1 | tee experiments/exp10_outputs/exp10_log.txt
echo "Done. Outputs in experiments/exp10_outputs/"
