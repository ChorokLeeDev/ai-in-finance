#!/bin/bash
# Launch all 8 retraining experiments in parallel
# 4 frequencies × 2 tasks = 8 total experiments

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
OUTPUT_DIR="papers/conformal_covid/results/retraining"
LOG_DIR="papers/conformal_covid/logs/retraining"

mkdir -p "$LOG_DIR"

echo "========================================================================"
echo "LAUNCHING RETRAINING EXPERIMENTS"
echo "========================================================================"
echo "Time: $(date)"
echo "Repo root: $REPO_ROOT"
echo "Output: $OUTPUT_DIR"
echo "Logs: $LOG_DIR"
echo ""

# Set PYTHONPATH
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

# Task list
CATASTROPHIC_TASK="sales-shipcond"
ROBUST_TASK="sales-office"

# Frequency list
FREQS=("none" "1M" "3M" "6M")

# Launch experiments in parallel
PIDS=()

for FREQ in "${FREQS[@]}"; do
    # Catastrophic task
    LOG_FILE="$LOG_DIR/retrain_${FREQ}_${CATASTROPHIC_TASK}.log"
    echo "Launching: $CATASTROPHIC_TASK @ $FREQ → $LOG_FILE"

    nohup python3 "$SCRIPT_DIR/retraining_experiment.py" \
        --dataset rel-salt \
        --task "$CATASTROPHIC_TASK" \
        --freq "$FREQ" \
        --seed 42 \
        > "$LOG_FILE" 2>&1 &

    PIDS+=($!)

    # Robust task
    LOG_FILE="$LOG_DIR/retrain_${FREQ}_${ROBUST_TASK}.log"
    echo "Launching: $ROBUST_TASK @ $FREQ → $LOG_FILE"

    nohup python3 "$SCRIPT_DIR/retraining_experiment.py" \
        --dataset rel-salt \
        --task "$ROBUST_TASK" \
        --freq "$FREQ" \
        --seed 42 \
        > "$LOG_FILE" 2>&1 &

    PIDS+=($!)
done

echo ""
echo "Launched 8 experiments with PIDs: ${PIDS[@]}"
echo ""
echo "Monitor progress:"
echo "  tail -f $LOG_DIR/*.log"
echo ""
echo "Check completion:"
echo "  ls -lh $OUTPUT_DIR/"
echo ""
echo "Expected output: 8 PKL files + 8 JSON files"
echo "Estimated time: 2-4 hours (running in parallel)"
echo ""
echo "========================================================================"
