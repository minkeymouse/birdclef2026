#!/bin/bash
set -e
cd "$(dirname "$0")/.."
echo "Running exp8: Focal loss + label smoothing"
mkdir -p experiments/exp8_outputs
uv run python experiments/exp8_focal_loss.py 2>&1 | tee experiments/exp8_outputs/exp8_log.txt
echo "Done. Outputs in experiments/exp8_outputs/"
