# Batch Execution Plan - CPU Optimized

**Date:** 2025-12-26 16:12
**Strategy:** Run 4 experiments at a time to manage CPU usage

---

## Batch 1: Catastrophic Task (RUNNING NOW)

**Status:** ✅ **RUNNING**
**PIDs:** 2709, 2710, 2711, 2712
**Started:** 16:12 KST
**Expected completion:** ~18:00-19:00 (1.5-2 hours)

**Experiments:**
1. PID 2709: sales-shipcond, none
2. PID 2710: sales-shipcond, 1M
3. PID 2711: sales-shipcond, 3M
4. PID 2712: sales-shipcond, 6M

**CPU usage:** ~120% total (4 × 30% each)

---

## Batch 2: Robust Task (PENDING)

**Status:** ⏳ **Will run after Batch 1 completes**
**Expected start:** ~18:00-19:00
**Expected completion:** ~19:30-21:00

**Experiments:**
5. sales-office, none
6. sales-office, 1M
7. sales-office, 3M
8. sales-office, 6M

---

## Monitoring Batch 1

### Quick status check:
```bash
ps -p 2709,2710,2711,2712 | wc -l
# 5 = all running (1 header + 4 processes)
# 1 = all finished
```

### Check completion:
```bash
ls papers/conformal_covid/results/retraining/*.pkl | wc -l
# Target after Batch 1: 4
# Target after Batch 2: 8
```

### Monitor progress:
```bash
# Check all logs
for freq in none 1M 3M 6M; do
    echo "=== $freq ==="
    tail -3 papers/conformal_covid/logs/retraining/retrain_${freq}_sales-shipcond.log 2>/dev/null
done
```

---

## When to Launch Batch 2

### Option 1: Automatic (recommended)
Wait for Batch 1 to complete, then manually run:

```bash
cd /Users/i767700/Github/ai-in-finance
export PYTHONPATH=/Users/i767700/Github/ai-in-finance:$PYTHONPATH

TASK="sales-office"
LOG_DIR="papers/conformal_covid/logs/retraining"

for FREQ in none 1M 3M 6M; do
    LOG_FILE="$LOG_DIR/retrain_${FREQ}_${TASK}.log"
    echo "Launching: $TASK @ $FREQ"

    nohup python3 papers/conformal_covid/code/retraining_experiment.py \
        --dataset rel-salt \
        --task "$TASK" \
        --freq "$FREQ" \
        --seed 42 \
        > "$LOG_FILE" 2>&1 &

    echo "  PID: $!"
done

echo "Batch 2 launched!"
ps aux | grep retraining_experiment.py | grep -v grep
```

### Option 2: Check and auto-launch
```bash
# Run this in a loop
while true; do
    RUNNING=$(ps -p 2709,2710,2711,2712 | wc -l)
    if [ "$RUNNING" -eq "1" ]; then
        echo "Batch 1 complete! Launching Batch 2..."
        # Run launch command here
        break
    fi
    echo "Batch 1 still running... $(date)"
    sleep 300  # Check every 5 minutes
done
```

---

## Expected Results

### Batch 1 (Catastrophic Task):
- **none:** ~22% coverage (baseline failure)
- **1M:** ~70% coverage (monthly retraining helps!)
- **3M:** ~50% coverage (quarterly helps)
- **6M:** ~35% coverage (bi-annual moderate help)

### Batch 2 (Robust Task):
- **All frequencies:** ~99.8% coverage (no degradation, retraining unnecessary)

---

## Timeline

| Time | Event |
|------|-------|
| 16:12 | ✅ Batch 1 launched (4 experiments) |
| 16:15-16:30 | Baseline (none) completes |
| 17:00-18:00 | Quarterly/bi-annual (3M, 6M) complete |
| 18:00-19:00 | Monthly (1M) completes (slowest) |
| **~19:00** | **Batch 1 complete, launch Batch 2** |
| 19:15-19:30 | Batch 2 baseline completes |
| 20:00-21:00 | Batch 2 all complete |
| **~21:00** | **All 8 experiments complete!** |

**Total time:** ~5 hours (vs 2-3 hours for parallel)
**CPU impact:** ~120% (vs ~240% for 8 parallel)

---

## Current Status

**Batch 1:** 🔄 RUNNING (PIDs 2709-2712)
**Batch 2:** ⏳ PENDING
**Next action:** Wait ~2 hours, then launch Batch 2
