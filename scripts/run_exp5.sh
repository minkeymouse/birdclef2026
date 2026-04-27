#!/bin/bash
set -e
cd "$(dirname "$0")/.."
echo "Running exp5: Self-training on unlabeled soundscapes"
mkdir -p experiments/exp5_outputs
uv run python experiments/exp5_self_training.py 2>&1 | tee experiments/exp5_outputs/exp5_log.txt
echo "Done. Outputs in experiments/exp5_outputs/"
