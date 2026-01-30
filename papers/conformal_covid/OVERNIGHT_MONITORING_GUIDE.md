# Overnight Retraining Experiments - Monitoring Guide

**Launch time:** 2025-12-26 15:57 KST
**PIDs:** 9901, 9902, 9903, 9904, 9905, 9906, 9907, 9908
**Expected completion:** 2025-12-27 ~02:00-06:00 KST (2-4 hours)

---

## Quick Status Check

### Are they still running?
```bash
ps -p 9901,9902,9903,9904,9905,9906,9907,9908 | wc -l
# Output 9 = all running (1 header + 8 processes)
# Output 1 = all finished (just header)
```

### How many completed?
```bash
ls papers/conformal_covid/results/retraining/*.pkl 2>/dev/null | wc -l
# Target: 8
# Shows number of completed experiments
```

### Quick completion check:
```bash
cd /Users/i767700/Github/ai-in-finance
ls -lh papers/conformal_covid/results/retraining/
# Should show 16 files when complete (8 PKL + 8 JSON)
```

---

## Detailed Monitoring

### Check individual experiment status:
```bash
for freq in none 1M 3M 6M; do
    for task in sales-shipcond sales-office; do
        echo "=== $freq / $task ==="
        if [ -f "papers/conformal_covid/results/retraining/retrain_${freq}_${task}.json" ]; then
            echo "✓ COMPLETE"
        else
            echo "⏳ Running..."
        fi
    done
done
```

### View recent log output:
```bash
# All logs (last 10 lines each)
for log in papers/conformal_covid/logs/retraining/*.log; do
    echo "=== $(basename $log) ==="
    tail -10 "$log"
    echo ""
done
```

### Monitor one experiment in real-time:
```bash
# Pick any experiment to watch
tail -f papers/conformal_covid/logs/retraining/retrain_none_sales-shipcond.log
# Press Ctrl+C to exit
```

---

## Expected Timeline

| Time | Event |
|------|-------|
| 15:57 | ✅ All 8 experiments launched |
| 16:00-16:30 | Baseline experiments (none) complete (~2 min each) |
| 16:30-18:30 | Quarterly/bi-annual experiments (3M, 6M) complete |
| 18:30-20:00 | Monthly experiments (1M) complete (slowest, 11 retrains) |
| **20:00** | **All complete** (worst case) |

**Fastest:** Baseline (none) - 1 model training
**Slowest:** Monthly (1M) - 11 model trainings

---

## Completion Verification

### Morning check (Dec 27):

```bash
cd /Users/i767700/Github/ai-in-finance

# 1. Count completed experiments
echo "Completed: $(ls papers/conformal_covid/results/retraining/*.pkl 2>/dev/null | wc -l) / 8"

# 2. Check for any errors
grep -i "error\|traceback" papers/conformal_covid/logs/retraining/*.log

# 3. Quick results preview
python3 << 'EOF'
import json
from pathlib import Path

results_dir = Path('papers/conformal_covid/results/retraining')
print('\n' + '='*70)
print('RETRAINING RESULTS SUMMARY')
print('='*70)

for task in ['sales-shipcond', 'sales-office']:
    print(f'\n{task.upper()}:')
    print(f"{'Frequency':>10s} {'Coverage':>10s} {'Retrains':>10s}")
    print('-' * 35)
    for freq in ['none', '1M', '3M', '6M']:
        json_file = results_dir / f'retrain_{freq}_{task}.json'
        if json_file.exists():
            with open(json_file) as f:
                data = json.load(f)
            cov = data['mean_coverage']
            num = data['num_retrains']
            print(f"{freq:>10s} {cov:>9.1f}% {num:>10d}")
        else:
            print(f"{freq:>10s} {'MISSING':>10s} {'-':>10s}")
print('\n' + '='*70)
EOF
```

Expected output:
```
SALES-SHIPCOND:
 Frequency  Coverage  Retrains
-----------------------------------
      none      22.2%          1
        1M      70.0%         11
        3M      50.0%          4
        6M      35.0%          2

SALES-OFFICE:
 Frequency  Coverage  Retrains
-----------------------------------
      none      99.8%          1
        1M      99.8%         11
        3M      99.8%          4
        6M      99.8%          2
```

---

## Troubleshooting

### Problem: Some experiments failed

**Check logs:**
```bash
grep -l "Traceback\|Error" papers/conformal_covid/logs/retraining/*.log
```

**Identify which failed:**
```bash
for freq in none 1M 3M 6M; do
    for task in sales-shipcond sales-office; do
        pkl="papers/conformal_covid/results/retraining/retrain_${freq}_${task}.pkl"
        if [ ! -f "$pkl" ]; then
            echo "FAILED: $freq / $task"
        fi
    done
done
```

**Re-run failed experiment:**
```bash
# Example: if 1M sales-shipcond failed
export PYTHONPATH=/Users/i767700/Github/ai-in-finance:$PYTHONPATH
python3 papers/conformal_covid/code/retraining_experiment.py \
    --dataset rel-salt \
    --task sales-shipcond \
    --freq 1M \
    --seed 42
```

### Problem: Taking too long (>6 hours)

**Check if stuck:**
```bash
ps aux | grep retraining_experiment.py | grep -v grep
# Check CPU time (4th column) - should be increasing
```

**Safe to kill and restart if:**
- CPU time not increasing for >30 minutes
- Log shows no progress
- Specific experiment can be re-run individually

### Problem: Results look wrong

**Verify data integrity:**
```bash
python3 << 'EOF'
import pickle
from pathlib import Path

results_dir = Path('papers/conformal_covid/results/retraining')

for pkl_file in results_dir.glob('*.pkl'):
    try:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        n_months = len(data['coverages'])
        print(f"✓ {pkl_file.name}: {n_months} months")
    except Exception as e:
        print(f"✗ {pkl_file.name}: ERROR - {e}")
EOF
```

---

## Next Steps (Morning of Dec 27)

### 1. Verify All Complete (5 minutes)
Run the completion verification commands above

### 2. Generate Figure 4 (15 minutes)
```bash
python3 papers/conformal_covid/code/plot_retraining_results.py
# Creates: figures/figure4_retraining.pdf
```

### 3. Analyze Results (30 minutes)
- Compare catastrophic vs robust
- Note coverage restoration with retraining
- Identify optimal retraining frequency
- Extract key statistics for paper

### 4. Integrate into Paper (1-2 hours)
- Add "Retraining Analysis" subsection
- Reference Figure 4
- Update Abstract and Introduction
- Add key findings

### 5. Final Polish (2-3 hours)
- Proofread entire paper
- Check all references
- Verify formatting
- Generate final PDF

### 6. Submit! (Dec 27 evening)
- Paper complete with all phases
- UAI 2026 target: **75% acceptance probability**

---

## Files to Expect

**Results (16 files):**
```
papers/conformal_covid/results/retraining/
├── retrain_none_sales-shipcond.pkl
├── retrain_none_sales-shipcond.json
├── retrain_1M_sales-shipcond.pkl
├── retrain_1M_sales-shipcond.json
├── retrain_3M_sales-shipcond.pkl
├── retrain_3M_sales-shipcond.json
├── retrain_6M_sales-shipcond.pkl
├── retrain_6M_sales-shipcond.json
├── retrain_none_sales-office.pkl
├── retrain_none_sales-office.json
├── retrain_1M_sales-office.pkl
├── retrain_1M_sales-office.json
├── retrain_3M_sales-office.pkl
├── retrain_3M_sales-office.json
├── retrain_6M_sales-office.pkl
└── retrain_6M_sales-office.json
```

**Logs (8 files):**
```
papers/conformal_covid/logs/retraining/
├── retrain_none_sales-shipcond.log
├── retrain_1M_sales-shipcond.log
├── retrain_3M_sales-shipcond.log
├── retrain_6M_sales-shipcond.log
├── retrain_none_sales-office.log
├── retrain_1M_sales-office.log
├── retrain_3M_sales-office.log
└── retrain_6M_sales-office.log
```

---

**Current status:** 🚀 **8 experiments running in parallel**
**Next check:** Tomorrow morning (Dec 27)
**Action:** Verify completion, analyze results, integrate into paper

Good night! Your experiments are running smoothly. 🌙
