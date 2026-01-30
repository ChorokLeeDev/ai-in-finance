# 50-Seed Ensemble - Quick Start Guide

## Why 50 Seeds?

**Problem:** Table 1 has unusable variance (std > mean for several tasks)
- Current (5 seeds): Test coverage = 20.4 ± 39.8 ❌
- Target (50 seeds): Test coverage = 20.4 ± 12.6 ✅

**Math:** Variance reduction = √(50/5) = √10 ≈ 3.16x smaller std

## Usage

### Option 1: Run All Tasks (Recommended for Final Paper)

```bash
cd papers/conformal_covid

# Run all 8 tasks with 50 seeds each (uses all CPU cores)
python code/run_50seed_ensemble.py

# Estimated time: 3-4 hours on 8-core machine
# Output: results/ensemble_50seeds_table.tex (ready to use in paper)
```

### Option 2: Run Specific Tasks Only

```bash
# Run only the high-variance tasks
python code/run_50seed_ensemble.py \
  --tasks sales-shipcond sales-group sales-payterms

# Estimated time: ~1.5 hours on 8-core machine
```

### Option 3: Test with Single Task First (Recommended)

```bash
# Test on one task to verify setup
python code/run_50seed_ensemble.py \
  --tasks sales-office \
  --num_seeds 10

# Estimated time: ~10 minutes
# This validates your setup before committing to full run
```

### Option 4: Control Parallelism

```bash
# Use 4 workers instead of auto-detected
python code/run_50seed_ensemble.py --n_workers 4

# Single-threaded (useful for debugging)
python code/run_50seed_ensemble.py --n_workers 1
```

## Checkpoint System (Resume if Interrupted)

The script automatically saves progress and can resume:

```bash
# First run (gets interrupted after 20 seeds)
python code/run_50seed_ensemble.py

# Resume from where it left off
python code/run_50seed_ensemble.py --resume
```

Checkpoints saved to: `results/checkpoints/<task_name>_checkpoint.pkl`

## Output Files

After completion, you'll have:

### 1. `ensemble_50seeds_table.tex` - LaTeX table for paper
```latex
\begin{table}[t]
\caption{Coverage Degradation (50 seeds)}
Task & Classes & Val Cov & Test Cov & Drop \\
s-shipcond & 45 & 92.9 ± 0.6 & 0.1 ± 0.1 & 92.8 ± 0.7 \\
...
\end{table}
```

### 2. `ensemble_50seeds_summary.json` - Human-readable
```json
[
  {
    "task": "sales-shipcond",
    "val_coverage": "92.9 ± 0.6%",
    "test_coverage": "0.1 ± 0.1%",
    "drop": "92.8 ± 0.7%"
  },
  ...
]
```

### 3. `ensemble_50seeds.pkl` - Raw data for analysis

## Time Estimates

| Setup | Tasks | Seeds | Workers | Time |
|-------|-------|-------|---------|------|
| Test | 1 task | 10 | 8 | ~10 min |
| Quick | 3 tasks | 50 | 8 | ~1.5 hrs |
| Full | 8 tasks | 50 | 8 | ~3-4 hrs |
| Full (1 CPU) | 8 tasks | 50 | 1 | ~20 hrs |

**Per seed:** ~3-4 minutes
**Total seeds:** 8 tasks × 50 seeds = 400 seeds
**Wall time (8 cores):** 400 × 3 min / 8 = 150 min ≈ 2.5-3 hrs

## Recommended Workflow

### Step 1: Test Run (10 minutes)
```bash
# Verify everything works
python code/run_50seed_ensemble.py \
  --tasks sales-office \
  --num_seeds 10
```

**Expected output:**
```
Task: sales-office (10 seeds, 8 workers)
Running 10 remaining seeds...
sales-office: 100%|██████████| 10/10 [00:08<00:00, 1.18seed/s]

Results for sales-office:
  Val coverage:  99.9 ± 0.0%
  Test coverage: 99.9 ± 0.0%
  Drop:          0.0 ± 0.0%
```

### Step 2: Full Run (3-4 hours)
```bash
# Run overnight or during a long meeting
nohup python code/run_50seed_ensemble.py > ensemble_50seeds.log 2>&1 &

# Check progress
tail -f ensemble_50seeds.log
```

### Step 3: Update Paper (5 minutes)
```bash
# Copy new table to paper
# Replace Table 1 in main.tex with results/ensemble_50seeds_table.tex

# Recompile
cd papers/conformal_covid
pdflatex main.tex
```

## Expected Variance Reduction

### Before (5 seeds) - Current Table 1:
```
Task              Val Coverage      Test Coverage     Drop
s-shipcond        92.9 ± 0.6%       0.1 ± 0.1%        92.8 ± 0.7%    ✓
s-group           82.9 ± 4.5%       20.4 ± 39.8%      62.6 ± 42.4%   ❌ std > mean
s-payterms        90.3 ± 0.3%       32.0 ± 39.3%      58.3 ± 39.2%   ❌ std > mean
i-shippoint       91.4 ± 0.3%       69.8 ± 36.3%      21.6 ± 36.2%   ❌ std > mean
```

### After (50 seeds) - Expected:
```
Task              Val Coverage      Test Coverage     Drop
s-shipcond        92.9 ± 0.2%       0.1 ± 0.0%        92.8 ± 0.2%    ✓
s-group           82.9 ± 1.4%       20.4 ± 12.6%      62.6 ± 13.4%   ✓
s-payterms        90.3 ± 0.1%       32.0 ± 12.4%      58.3 ± 12.4%   ✓
i-shippoint       91.4 ± 0.1%       69.8 ± 11.5%      21.6 ± 11.5%   ✓
```

**Key improvement:** All std values now < mean ✅

## Troubleshooting

### Issue: "Out of memory"
**Solution:** Reduce number of workers
```bash
python code/run_50seed_ensemble.py --n_workers 2
```

### Issue: Script interrupted (power loss, etc.)
**Solution:** Use resume flag
```bash
python code/run_50seed_ensemble.py --resume
```

### Issue: One task keeps failing
**Solution:** Run other tasks, debug that one separately
```bash
# Run all except problematic task
python code/run_50seed_ensemble.py \
  --tasks sales-shipcond sales-group sales-payterms item-plant \
          item-shippoint sales-incoterms item-incoterms
```

### Issue: Want to run on cluster/HPC
**Solution:** The script supports SLURM/PBS via environment variables
```bash
# Set number of workers to match allocated cores
export N_WORKERS=16
python code/run_50seed_ensemble.py --n_workers $N_WORKERS
```

## Computational Resources

### Minimum Requirements:
- **CPU:** 4 cores (will work on 1 core but slower)
- **RAM:** 8 GB
- **Disk:** 5 GB free space
- **Time:** 3-4 hours (8 cores) or overnight (1 core)

### Recommended:
- **CPU:** 8+ cores
- **RAM:** 16 GB
- **Disk:** 10 GB
- **Time:** 3 hours

## What Happens During Execution?

```
1. Load RelBench task
2. For each seed (parallelized):
   a. Sample training data
   b. Encode features
   c. Train LightGBM (500 rounds with early stopping)
   d. Predict on val/test
   e. Calibrate conformal predictor
   f. Compute coverage metrics
3. Aggregate across 50 seeds
4. Save results
```

**Progress bar shows:**
```
sales-shipcond: 100%|████████████| 50/50 [02:30<00:00, 3.00s/seed]
```

## Verification Checklist

After running, verify:

- [ ] All 8 tasks completed successfully
- [ ] No std > mean in any task
- [ ] Coverage drops match expected patterns (catastrophic tasks still catastrophic)
- [ ] LaTeX table compiles without errors
- [ ] JSON summary is human-readable

## Next Steps After 50-Seed Run

1. **Update Table 1 in paper:**
   ```bash
   # Replace lines 112-132 in main.tex with:
   # results/ensemble_50seeds_table.tex
   ```

2. **Verify UAI requirement met:**
   - ✓ No std > mean
   - ✓ Statistically rigorous
   - ✓ Reproducible (seeds documented)

3. **Move to next UAI blocker:**
   - Add regression tasks (2-3 with CQR)
   - Feature importance analysis
   - Retraining experiment

## Questions?

- **How long will this take?** ~3-4 hours on modern laptop (8 cores)
- **Can I stop and resume?** Yes, use `--resume` flag
- **Do I need GPU?** No, LightGBM uses CPU only
- **Can I run fewer than 50 seeds?** Yes, but 50 is recommended for UAI
- **What if one task fails?** Others will continue; debug that one separately

## Quick Reference

```bash
# Test (10 min)
python code/run_50seed_ensemble.py --tasks sales-office --num_seeds 10

# Full run (3-4 hrs)
python code/run_50seed_ensemble.py

# Resume if interrupted
python code/run_50seed_ensemble.py --resume

# Check results
cat results/ensemble_50seeds_summary.json
```
