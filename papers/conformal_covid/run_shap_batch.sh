#!/bin/bash

# Run SHAP analysis for all 6 remaining tasks
# Uses project's venv Python

PYTHON=/Users/i767700/Github/ai-in-finance/.venv/bin/python3
SCRIPT=/Users/i767700/Github/ai-in-finance/papers/conformal_covid/code/analyze_feature_importance.py
RESULTS=/Users/i767700/Github/ai-in-finance/papers/conformal_covid/results/shap

echo "Starting SHAP analyses for 6 tasks..."
echo "Python: $PYTHON"
echo "Script: $SCRIPT"
echo ""

# Run each task
for task in sales-group sales-payterms item-plant item-shippoint sales-incoterms item-incoterms; do
    echo "Running $task..."
    $PYTHON "$SCRIPT" --dataset rel-salt --task "$task"

    # Check if successful
    if [ -f "$RESULTS/shap_rel-salt_${task}.pkl" ]; then
        echo "✓ $task complete"
    else
        echo "✗ $task failed"
    fi
    echo ""
done

echo "All done! Checking results..."
ls -lt "$RESULTS"/shap_rel-salt_*.pkl | wc -l | xargs echo "Total SHAP files:"
