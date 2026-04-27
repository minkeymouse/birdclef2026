#!/bin/bash
set -e
cd "$(dirname "$0")/.."
mkdir -p experiments/exp4_outputs

# Phase 1: Perch extraction (CPU only — separate process to avoid TF/CUDA conflict)
if [ ! -f experiments/exp4_outputs/perch_cache/_done.flag ]; then
    echo "Phase 1: Extracting Perch v2 soft labels (CPU)..."
    CUDA_VISIBLE_DEVICES="" uv run python experiments/exp4_extract_perch.py 2>&1 | tee experiments/exp4_outputs/perch_log.txt
fi

# Phase 2: KD training (GPU)
echo "Phase 2: Training with Knowledge Distillation..."
uv run python experiments/exp4_perch_kd.py 2>&1 | tee experiments/exp4_outputs/exp4_log.txt

echo "Done. Outputs in experiments/exp4_outputs/"
