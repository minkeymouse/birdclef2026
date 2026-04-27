#!/bin/bash
set -e
echo "=== exp14: Waveform augmentation + background mixing ==="
mkdir -p experiments/exp14_outputs
cd /data/birdclef2026
uv run python experiments/exp14_aug_v2.py 2>&1 | tee experiments/exp14_outputs/exp14_log.txt
echo "=== exp14 complete ==="
