#!/bin/bash
# Monitor running experiments

echo "=========================================="
echo "EXPERIMENT MONITORING"
echo "=========================================="
echo ""

# Check study-outcome
echo "1. STUDY-OUTCOME (rel-trial)"
echo "-------------------------------------------"
if [ -f "papers/conformal_covid/study_outcome.log" ]; then
    echo "Latest progress:"
    tail -5 papers/conformal_covid/study_outcome.log
    echo ""
    echo "Seeds completed:"
    grep -c "Seed.*coverage_drop" papers/conformal_covid/study_outcome.log || echo "0"
else
    echo "No log file yet"
fi

echo ""
echo "2. DRIVER-DNF (rel-f1)"
echo "-------------------------------------------"
if [ -f "papers/conformal_covid/driver_dnf.log" ]; then
    echo "Latest progress:"
    tail -5 papers/conformal_covid/driver_dnf.log
    echo ""
    echo "Seeds completed:"
    grep -c "Seed.*coverage_drop" papers/conformal_covid/driver_dnf.log || echo "0"
else
    echo "No log file yet"
fi

echo ""
echo "=========================================="
echo "RESULTS"
echo "=========================================="

# Check if results exist
if [ -f "papers/conformal_covid/results/aps_rel-trial_study-outcome.json" ]; then
    echo ""
    echo "✓ study-outcome COMPLETE:"
    cat papers/conformal_covid/results/aps_rel-trial_study-outcome.json
fi

if [ -f "papers/conformal_covid/results/aps_rel-f1_driver-dnf.json" ]; then
    echo ""
    echo "✓ driver-dnf COMPLETE:"
    cat papers/conformal_covid/results/aps_rel-f1_driver-dnf.json
fi

echo ""
echo "=========================================="
echo "To check again: bash monitor_experiments.sh"
echo "=========================================="
