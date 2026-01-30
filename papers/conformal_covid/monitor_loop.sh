#!/bin/bash
# Monitor classification experiments every 10 minutes
# Runs in background and alerts when experiments complete

LOG_FILE="papers/conformal_covid/monitor_loop.log"
CHECK_INTERVAL=600  # 10 minutes in seconds

echo "Starting experiment monitoring (checking every 10 minutes)..." | tee -a "$LOG_FILE"
echo "Started at: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

while true; do
    echo "=== Check at $(date) ===" | tee -a "$LOG_FILE"

    # Run the monitoring script
    bash papers/conformal_covid/monitor_experiments.sh | tee -a "$LOG_FILE"

    # Check if both experiments are complete
    STUDY_COMPLETE=$(grep -c "✓ Completed all 50 seeds" papers/conformal_covid/study_outcome.log 2>/dev/null || echo "0")
    DRIVER_COMPLETE=$(grep -c "✓ Completed all 50 seeds" papers/conformal_covid/driver_dnf.log 2>/dev/null || echo "0")

    if [ "$STUDY_COMPLETE" -gt 0 ] && [ "$DRIVER_COMPLETE" -gt 0 ]; then
        echo "" | tee -a "$LOG_FILE"
        echo "🎉 BOTH EXPERIMENTS COMPLETE! 🎉" | tee -a "$LOG_FILE"
        echo "Completed at: $(date)" | tee -a "$LOG_FILE"
        echo "" | tee -a "$LOG_FILE"
        echo "Next steps:" | tee -a "$LOG_FILE"
        echo "1. Check results in papers/conformal_covid/results/conformal/" | tee -a "$LOG_FILE"
        echo "2. Run Day 3 SHAP analysis: bash papers/conformal_covid/run_shap_classification.sh" | tee -a "$LOG_FILE"
        echo "" | tee -a "$LOG_FILE"
        break
    fi

    echo "" | tee -a "$LOG_FILE"
    echo "Next check in 10 minutes..." | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"

    # Wait 10 minutes
    sleep $CHECK_INTERVAL
done

echo "Monitoring complete at $(date)" | tee -a "$LOG_FILE"
