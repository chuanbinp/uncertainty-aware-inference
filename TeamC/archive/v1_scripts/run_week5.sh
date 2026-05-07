#!/bin/bash
# Week 5 convergence phase: Pareto analysis + routing simulation
# Assumes evaluation results already exist from Weeks 2-4.
# Run from project root: bash TeamC/run_week5.sh
set -e

cd "$(dirname "$0")"

echo "========================================"
echo "Week 5: Pareto Analysis + Routing Simulation"
echo "========================================"

# Step 0: Verify results exist
echo ""
echo "--- Checking for result files ---"
FOUND=0
for dir in results ../TeamA/results ../TeamB/results; do
    if [ -d "$dir" ]; then
        COUNT=$(find "$dir" -name "*.json" 2>/dev/null | wc -l)
        echo "  $dir: $COUNT JSON files"
        FOUND=$((FOUND + COUNT))
    else
        echo "  $dir: not found"
    fi
done

if [ "$FOUND" -eq 0 ]; then
    echo ""
    echo "ERROR: No result files found. Run eval scripts first (run_sweep.sh)."
    exit 1
fi
echo "Total: $FOUND result files"

# Step 1: Entropy analysis (with cross-team data)
echo ""
echo "--- Step 1: Entropy Analysis (cross-team) ---"
python entropy_analysis.py --include_all_teams

# Step 2: Cross-model comparison
echo ""
echo "--- Step 2: Cross-Model Comparison ---"
python cross_model_compare.py

# Step 3: Pareto frontier analysis
echo ""
echo "--- Step 3: Pareto Frontier Analysis ---"
python pareto_script.py

# Step 4: Routing simulation
echo ""
echo "--- Step 4: Routing Simulation ---"
python routing_simulation.py

echo ""
echo "========================================"
echo "Week 5 analysis complete."
echo "Results:"
echo "  Entropy analysis:   results/analysis/"
echo "  Pareto frontiers:   results/pareto/"
echo "  Routing simulation: results/routing/"
echo "========================================"
