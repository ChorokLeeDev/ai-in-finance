#!/bin/bash
# Day 3: Complete SHAP Analysis and Correlation
#
# This script:
# 1. Runs SHAP for 4 classification tasks (40-60 min)
# 2. Computes n=12 correlation analysis (1 min)
#
# Total runtime: ~50-70 minutes

set -e

echo "========================================"
echo "Day 3: SHAP Analysis and Correlation"
echo "========================================"
echo "Started at: $(date)"
echo ""

# Change to correct directory
cd /Users/i767700/Github/ai-in-finance/papers/conformal_covid

# Step 1: Classification SHAP
echo ""
echo "Step 1: Running SHAP for classification tasks (4 tasks)..."
echo "========================================"
echo ""

bash run_shap_classification.sh

echo ""
echo "✓ Classification SHAP complete!"
echo ""

# Step 2: Correlation
echo ""
echo "Step 2: Computing n=12 correlation..."
echo "========================================"
echo ""

/Users/i767700/Github/ai-in-finance/.venv/bin/python3 code/analyze_n12_correlation.py

echo ""
echo "✓ Correlation analysis complete!"
echo ""

# Final summary
echo ""
echo "========================================"
echo "Day 3: COMPLETE!"
echo "========================================"
echo "Completed at: $(date)"
echo ""
echo "Results:"
echo "  - SHAP: results/shap/"
echo "  - Correlation: results/n12_correlation_results.csv"
echo "  - Figure: results/figure_n12_correlation.pdf"
echo "  - Table: results/table_n12_correlation.tex"
echo ""
echo "Next: Review results and update paper!"
echo ""
