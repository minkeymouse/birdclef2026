#!/bin/bash
set -e
echo "=== exp11: 5s chunk baseline ==="
mkdir -p experiments/exp11_outputs
cd /data/birdclef2026
uv run python experiments/exp11_5s_baseline.py 2>&1 | tee experiments/exp11_outputs/exp11_log.txt
echo "=== exp11 complete ==="
