#!/bin/bash
set -e
cd "$(dirname "$0")/.."
echo "Running exp6: EfficientNet-B3 + 20s chunks"
mkdir -p experiments/exp6_outputs
uv run python experiments/exp6_b3_20s.py 2>&1 | tee experiments/exp6_outputs/exp6_log.txt
echo "Done. Outputs in experiments/exp6_outputs/"
