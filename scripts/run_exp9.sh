#!/bin/bash
set -e
cd "$(dirname "$0")/.."
echo "Running exp9: Enhanced augmentation + 2nd round self-training"
mkdir -p experiments/exp9_outputs
uv run python experiments/exp9_aug_st2.py 2>&1 | tee experiments/exp9_outputs/exp9_log.txt
echo "Done. Outputs in experiments/exp9_outputs/"
