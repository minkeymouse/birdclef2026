#!/usr/bin/env bash
# Sequence multiple LB submissions with different lever configs.
# Each iteration: edit notebook → push kernel → wait dev → submit
set -e
export KAGGLE_API_TOKEN=${KAGGLE_API_TOKEN:?must be set}

NB="/data/birdclef2026/notebooks/birdclef-2026-perch-distill/notebook.ipynb"
LOGDIR="/data/birdclef2026/experiments/_scratch_logs"

submit_with_message() {
  local kernel_v="$1"
  local msg="$2"
  echo "[$(date +%H:%M)] Submitting kernel v${kernel_v}..."
  uv run kaggle competitions submit -c birdclef-2026 -f submission.csv \
    -k ultimatumgame/birdclef-2026-perch-distill -v "$kernel_v" \
    -m "$msg" 2>&1 | tail -3
  sleep 5
}

push_and_wait() {
  local label="$1"
  echo "[$(date +%H:%M)] Pushing kernel for $label..."
  uv run kaggle kernels push -p /data/birdclef2026/notebooks/birdclef-2026-perch-distill 2>&1 | tail -2

  # Get the new version number (must wait for upload to register)
  sleep 10
  local kv
  for try in 1 2 3 4 5; do
    kv=$(uv run kaggle kernels list -m -s ultimatumgame/birdclef-2026-perch-distill 2>&1 | grep -oE "version[: ]+[0-9]+" | head -1 | grep -oE "[0-9]+" || echo "")
    if [ -n "$kv" ]; then break; fi
    sleep 5
  done

  echo "[$(date +%H:%M)] $label kernel pushed, polling for dev complete..."

  while true; do
    sleep 60
    s=$(uv run kaggle kernels status ultimatumgame/birdclef-2026-perch-distill 2>&1 | grep -oE "(KernelWorkerStatus\.[A-Z]+|complete|error|failed|cancel)" | head -1)
    case "$s" in
      *complete*|*COMPLETE*) echo "[$(date +%H:%M)] $label dev OK"; break;;
      *error*|*ERROR*|*failed*|*FAILED*|*cancel*)
        echo "[$(date +%H:%M)] $label dev FAILED with $s — skipping submission"
        return 1
        ;;
    esac
  done

  return 0
}

echo "=== LB SWEEP started at $(date) ==="
echo "Already submitted: v38 (LR α=0.3 β=0.1)"
echo ""
echo "Plan: 4 more slots with orthogonal hypothesis tests"
echo ""
