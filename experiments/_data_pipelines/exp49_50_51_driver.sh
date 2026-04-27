#!/bin/bash
# Sequentially run exp49c (taxon gate retrain), exp50 (exp47 w/ 2025 BG),
# exp51 (27-class head w/ 2025 BG). Logs each to exp4?_outputs/driver.log.
set -u
cd /data/birdclef2026

echo "=== DRIVER START $(date) ==="

echo ""
echo ">>> exp49c: taxon gate retrain $(date)"
uv run python -u experiments/exp49c_taxon_gate_2025_2026.py \
    > experiments/exp49_outputs/49c_train.log 2>&1
RC49=$?
echo "exp49c exit code: $RC49"

echo ""
echo ">>> exp50: exp47 w/ 2025 BG $(date)"
mkdir -p experiments/exp50_outputs
uv run python -u experiments/exp50_exp47_with_2025bg.py \
    > experiments/exp50_outputs/train.log 2>&1
RC50=$?
echo "exp50 exit code: $RC50"

echo ""
echo ">>> exp51: 27-class w/ 2025 BG $(date)"
mkdir -p experiments/exp51_outputs
uv run python -u experiments/exp51_27head_with_2025bg.py \
    > experiments/exp51_outputs/train.log 2>&1
RC51=$?
echo "exp51 exit code: $RC51"

echo ""
echo "=== DRIVER DONE $(date) ==="
echo "exit codes: 49c=$RC49  50=$RC50  51=$RC51"
