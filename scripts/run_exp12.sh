#!/bin/bash
set -e
echo "=== exp12: Inference optimization sweep ==="
mkdir -p experiments/exp12_outputs
cd /data/birdclef2026
uv run python experiments/exp12_inference_opt.py 2>&1 | tee experiments/exp12_outputs/exp12_log.txt
echo "=== exp12 complete ==="
