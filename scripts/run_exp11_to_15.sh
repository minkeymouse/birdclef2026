#!/bin/bash
# Master runner: exp11 → exp12 → exp13 → exp14 → exp15 (sequential)
# Total estimated time: ~8 hours on RTX 5090
set -e

echo "=========================================="
echo "Starting exp11-15 batch run"
echo "Start time: $(date)"
echo "=========================================="

# exp11 must run first (foundation for exp12, exp13, exp15)
bash scripts/run_exp11.sh
echo "exp11 done at $(date)"

# exp12 depends on exp11 weights; exp13 depends on exp11 mel cache
# Run exp12 first (fast, ~15 min) then exp13 (slow, ~3 hours)
bash scripts/run_exp12.sh
echo "exp12 done at $(date)"

bash scripts/run_exp13.sh
echo "exp13 done at $(date)"

# exp14 is independent but runs after to not compete for GPU
bash scripts/run_exp14.sh
echo "exp14 done at $(date)"

# exp15 depends on exp11 weights
bash scripts/run_exp15.sh
echo "exp15 done at $(date)"

echo "=========================================="
echo "All experiments complete!"
echo "End time: $(date)"
echo "=========================================="

# Print summary
echo ""
echo "=== RESULTS SUMMARY ==="
for exp in 11 12 13 14 15; do
    results="experiments/exp${exp}_outputs/exp${exp}_results.json"
    if [ -f "$results" ]; then
        echo "--- exp${exp} ---"
        cat "$results"
        echo ""
    fi
done
