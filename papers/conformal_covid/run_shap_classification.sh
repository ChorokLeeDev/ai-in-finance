#!/bin/bash
# Run SHAP Analysis for All 4 Classification Tasks
# Day 3 of Quick Start (n=12) Plan
#
# Tasks:
# 1. rel-trial/study-outcome (NEW - from Day 2)
# 2. rel-trial/study-adverse (EXISTING - from n=8)
# 3. rel-trial/site-success (EXISTING - from n=8)
# 4. rel-f1/driver-dnf (NEW - from Day 2)
#
# Runtime: ~10-15 minutes per task = 40-60 minutes total
#
# Usage:
#   bash run_shap_classification.sh

set -e  # Exit on error

PYTHON="/Users/i767700/Github/ai-in-finance/.venv/bin/python3"
SCRIPT="code/compute_shap_classification.py"
OUTPUT_DIR="results/shap"
LOG_DIR="logs"

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

echo "========================================"
echo "SHAP Analysis - Classification Tasks"
echo "Day 3 of Quick Start (n=12) Plan"
echo "========================================"
echo ""
echo "Tasks to analyze:"
echo "  1. rel-trial/study-outcome (NEW)"
echo "  2. rel-trial/study-adverse (EXISTING)"
echo "  3. rel-trial/site-success (EXISTING)"
echo "  4. rel-f1/driver-dnf (NEW)"
echo ""
echo "Output: $OUTPUT_DIR"
echo "Logs: $LOG_DIR"
echo ""
echo "Estimated runtime: 40-60 minutes"
echo ""
echo "========================================"
echo ""

# Task 1: study-outcome (NEW)
echo "Task 1/4: rel-trial/study-outcome"
echo "========================================"
LOG_FILE="$LOG_DIR/shap_study-outcome.log"
if [ -f "$OUTPUT_DIR/shap_rel-trial_study-outcome.pkl" ]; then
    echo "✓ Results already exist, skipping..."
else
    echo "Running SHAP analysis..."
    $PYTHON $SCRIPT \
        --dataset rel-trial \
        --task study-outcome \
        --subsample 10000 \
        --seed 42 \
        2>&1 | tee "$LOG_FILE"
    echo "✓ Completed"
fi
echo ""

# Task 2: study-adverse (EXISTING - check if already computed)
echo "Task 2/4: rel-trial/study-adverse"
echo "========================================"
LOG_FILE="$LOG_DIR/shap_study-adverse.log"
if [ -f "$OUTPUT_DIR/shap_rel-trial_study-adverse.pkl" ]; then
    echo "✓ Results already exist, skipping..."
else
    echo "Running SHAP analysis..."
    $PYTHON $SCRIPT \
        --dataset rel-trial \
        --task study-adverse \
        --subsample 10000 \
        --seed 42 \
        2>&1 | tee "$LOG_FILE"
    echo "✓ Completed"
fi
echo ""

# Task 3: site-success (EXISTING - check if already computed)
echo "Task 3/4: rel-trial/site-success"
echo "========================================"
LOG_FILE="$LOG_DIR/shap_site-success.log"
if [ -f "$OUTPUT_DIR/shap_rel-trial_site-success.pkl" ]; then
    echo "✓ Results already exist, skipping..."
else
    echo "Running SHAP analysis..."
    $PYTHON $SCRIPT \
        --dataset rel-trial \
        --task site-success \
        --subsample 10000 \
        --seed 42 \
        2>&1 | tee "$LOG_FILE"
    echo "✓ Completed"
fi
echo ""

# Task 4: driver-dnf (NEW)
echo "Task 4/4: rel-f1/driver-dnf"
echo "========================================"
LOG_FILE="$LOG_DIR/shap_driver-dnf.log"
if [ -f "$OUTPUT_DIR/shap_rel-f1_driver-dnf.pkl" ]; then
    echo "✓ Results already exist, skipping..."
else
    echo "Running SHAP analysis..."
    $PYTHON $SCRIPT \
        --dataset rel-f1 \
        --task driver-dnf \
        --subsample 10000 \
        --seed 42 \
        2>&1 | tee "$LOG_FILE"
    echo "✓ Completed"
fi
echo ""

echo "========================================"
echo "SHAP Analysis Complete!"
echo "========================================"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  1. Verify all 4 SHAP result files exist"
echo "  2. Run correlation analysis: python analyze_n12_correlation.py"
echo "  3. Update paper with n=12 results"
echo ""
