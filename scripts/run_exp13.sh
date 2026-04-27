#!/bin/bash
set -e
echo "=== exp13: Multi-backbone training ==="
mkdir -p experiments/exp13_outputs
cd /data/birdclef2026
uv run python experiments/exp13_multi_backbone.py --backbone all 2>&1 | tee experiments/exp13_outputs/exp13_log.txt
echo "=== exp13 complete ==="
