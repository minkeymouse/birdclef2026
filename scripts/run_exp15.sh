#!/bin/bash
set -e
echo "=== exp15: Proper self-training (2025 winner recipe) ==="
mkdir -p experiments/exp15_outputs
cd /data/birdclef2026
uv run python experiments/exp15_proper_st.py 2>&1 | tee experiments/exp15_outputs/exp15_log.txt
echo "=== exp15 complete ==="
