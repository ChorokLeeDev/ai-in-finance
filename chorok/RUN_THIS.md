# 🚀 Quick Start: Train All Ensembles

## Option 1: Direct Python (Recommended)

```powershell
# In PowerShell terminal
python chorok/train_parallel.py --yes
```

**Time**: ~21 minutes
**Workers**: 7 parallel processes
**Output**: Real-time progress in terminal

---

## Option 2: PowerShell Script (Background with Log)

```powershell
# Run in background and save to log file
powershell -File chorok/run_training.ps1
```

**Advantages**:
- Saves output to log file: `chorok/results/training_YYYYMMDD_HHMMSS.log`
- Can monitor with: `Get-Content chorok/results/training_*.log -Wait`

---

## Option 3: Windows Terminal (Keep Window Open)

1. Open Windows Terminal
2. Run:
   ```powershell
   cd C:\Users\hippo\relbench
   python chorok/train_parallel.py --yes
   ```
3. Minimize window and continue working

---

## Monitor Progress

**Check status anytime:**
```powershell
python chorok/check_ensemble_status.py
```

**Expected output:**
```
[OK] item-plant           [5/5 seeds] - COMPLETE
[!!] item-shippoint       [3/5 seeds] - INCOMPLETE  # Training...
...
Phase 2: 62.5% COMPLETE
```

---

## After Training Completes

**1. Verify all tasks complete (should be 100%):**
```powershell
python chorok/check_ensemble_status.py
```

**2. Run Phase 3 (Uncertainty Quantification):**
```powershell
python chorok/temporal_uncertainty_analysis.py
```

**3. Run Phase 4 (Correlation Analysis):**
```powershell
python chorok/compare_shift_uncertainty.py
```

---

## Troubleshooting

**If training fails:**
- Just re-run the same command
- Already completed models will be skipped automatically

**If too slow:**
```powershell
# Reduce workers (less parallel, more stable)
python chorok/train_parallel.py --yes --workers 3

# Or even faster (less accurate but OK for UQ)
python chorok/train_parallel.py --yes --num-trials 3 --sample-size 5000
```

**If you want to test first:**
```powershell
# Dry run (show what will be trained)
python chorok/train_parallel.py --dry-run
```

---

## Configuration

Current settings (optimized for speed):
- **Parallel workers**: 7 (auto-detected CPU count - 1)
- **Sample size**: 10,000 (3x faster than 50k)
- **Num trials**: 5 (2x faster than 10)
- **Total time**: ~21 minutes (vs 7 hours sequential)

These settings provide sufficient quality for UQ research while being much faster.

---

## What's Being Trained

**Missing ensembles:**
- item-shippoint: 5 models
- item-incoterms: 5 models
- sales-group: 4 models (already has 1)
- sales-payterms: 5 models
- sales-shipcond: 5 models
- sales-incoterms: 5 models

**Total**: 29 models

**Already complete:**
- item-plant: 5 models ✓
- sales-office: 6 models ✓

---

**Ready?** Just run this:

```powershell
python chorok/train_parallel.py --yes
```
