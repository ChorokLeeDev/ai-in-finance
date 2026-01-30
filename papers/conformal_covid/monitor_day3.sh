#!/bin/bash
# Monitor Day 3 SHAP execution every 10 minutes
#
# Checks:
# - SHAP results files being created
# - Log file progress
# - Completion status

LOG_FILE="papers/conformal_covid/monitor_day3.log"
CHECK_INTERVAL=600  # 10 minutes

echo "Starting Day 3 monitoring (checking every 10 minutes)..." | tee -a "$LOG_FILE"
echo "Started at: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

while true; do
    echo "=== Check at $(date) ===" | tee -a "$LOG_FILE"

    # Count SHAP result files
    SHAP_COUNT=$(ls results/shap/shap_*.pkl 2>/dev/null | wc -l | tr -d ' ')
    echo "SHAP results: $SHAP_COUNT/12 complete" | tee -a "$LOG_FILE"

    # Check for classification SHAP files
    echo "" | tee -a "$LOG_FILE"
    echo "Classification SHAP progress:" | tee -a "$LOG_FILE"

    if [ -f "results/shap/shap_rel-trial_study-outcome.pkl" ]; then
        echo "  ✓ study-outcome" | tee -a "$LOG_FILE"
    else
        echo "  ⏳ study-outcome (running...)" | tee -a "$LOG_FILE"
    fi

    if [ -f "results/shap/shap_rel-trial_study-adverse.pkl" ]; then
        echo "  ✓ study-adverse" | tee -a "$LOG_FILE"
    else
        echo "  ⏳ study-adverse" | tee -a "$LOG_FILE"
    fi

    if [ -f "results/shap/shap_rel-trial_site-success.pkl" ]; then
        echo "  ✓ site-success" | tee -a "$LOG_FILE"
    else
        echo "  ⏳ site-success" | tee -a "$LOG_FILE"
    fi

    if [ -f "results/shap/shap_rel-f1_driver-dnf.pkl" ]; then
        echo "  ✓ driver-dnf" | tee -a "$LOG_FILE"
    else
        echo "  ⏳ driver-dnf" | tee -a "$LOG_FILE"
    fi

    echo "" | tee -a "$LOG_FILE"

    # Check correlation results
    if [ -f "results/n12_correlation_results.csv" ]; then
        echo "✓ Correlation analysis COMPLETE!" | tee -a "$LOG_FILE"
        echo "" | tee -a "$LOG_FILE"
        echo "=======================================" | tee -a "$LOG_FILE"
        echo "🎉 DAY 3 COMPLETE! 🎉" | tee -a "$LOG_FILE"
        echo "=======================================" | tee -a "$LOG_FILE"
        echo "Completed at: $(date)" | tee -a "$LOG_FILE"
        echo "" | tee -a "$LOG_FILE"
        echo "Results available at:" | tee -a "$LOG_FILE"
        echo "  - results/n12_correlation_results.csv" | tee -a "$LOG_FILE"
        echo "  - results/figure_n12_correlation.pdf" | tee -a "$LOG_FILE"
        echo "  - results/table_n12_correlation.tex" | tee -a "$LOG_FILE"
        echo "" | tee -a "$LOG_FILE"
        break
    fi

    # Check main execution log
    if [ -f "day3_execution.log" ]; then
        echo "Latest from execution log:" | tee -a "$LOG_FILE"
        tail -3 day3_execution.log | tee -a "$LOG_FILE"
    fi

    echo "" | tee -a "$LOG_FILE"
    echo "Next check in 10 minutes..." | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"

    # Wait 10 minutes
    sleep $CHECK_INTERVAL
done

echo "Monitoring complete at $(date)" | tee -a "$LOG_FILE"
