#!/bin/bash
set -e
cd "$(dirname "$0")/.."
echo "Running exp3: SED v2 (20s, attention head, wave mixup)"
mkdir -p experiments/exp3_outputs
uv run python experiments/exp3_sed_v2.py 2>&1 | tee experiments/exp3_outputs/exp3_log.txt
echo "Done. Outputs in experiments/exp3_outputs/"
