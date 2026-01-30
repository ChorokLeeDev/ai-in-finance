# Day 3: Ready to Launch - SHAP Analysis 🚀

**Date:** 2025-12-27, 10:00 PM
**Status:** All Day 2 experiments complete, Day 3 ready to start
**Estimated Runtime:** 40-60 minutes total

---

## Status Check ✅

### Conformal Prediction Results: 12/12 Complete

**Classification (n=4):**
- ✅ rel-trial/study-outcome (NEW - finished tonight)
- ✅ rel-trial/study-adverse (EXISTING)
- ✅ rel-trial/site-success (EXISTING)
- ✅ rel-f1/driver-dnf (NEW - finished tonight)

**Regression (n=8):**
- ✅ All 8 rel-salt tasks (EXISTING)

### SHAP Results: 8/12 Complete ✅

**Regression (n=8): 8/8 Complete ✅**
- ✅ sales-shipcond
- ✅ sales-group
- ✅ sales-payterms
- ✅ item-plant
- ✅ item-shippoint
- ✅ sales-incoterms
- ✅ item-incoterms
- ✅ sales-office

**Classification (n=4): 0/4 Complete**
- ❌ study-outcome
- ❌ study-adverse
- ❌ site-success
- ❌ driver-dnf

**Total:** 8/12 SHAP results complete (67%)
**Remaining:** Only 4 classification tasks!

---

## Day 3 Execution Plan

### Step 1: Run SHAP for Classification (4 tasks)

**All regression SHAP already complete!** ✅

Only need to run classification tasks:

**Command:**
```bash
cd /Users/i767700/Github/ai-in-finance/papers/conformal_covid

bash run_shap_classification.sh
```

**What it does:**
1. Checks if results already exist (skips if found)
2. Runs SHAP for:
   - study-outcome
   - study-adverse
   - site-success
   - driver-dnf
3. Saves results to `results/shap/shap_*.pkl`
4. Logs to `logs/shap_*.log`

**Runtime:** ~40-60 minutes (10-15 min per task)

---

### Step 2: Compute n=12 Correlation

**Command:**
```bash
cd /Users/i767700/Github/ai-in-finance/papers/conformal_covid

python code/analyze_n12_correlation.py
```

**What it does:**
1. Loads SHAP results from all 12 tasks
2. Loads conformal results (coverage drops)
3. Computes correlation: concentration vs drop
4. Creates visualization
5. Generates LaTeX table

**Output:**
- `results/n12_correlation_results.csv`
- `results/figure_n12_correlation.pdf`
- `results/table_n12_correlation.tex`
- `results/n12_statistics.txt`

**Runtime:** ~1 minute

---

## Full Command Sequence

### Option 1: Run Everything Sequentially

```bash
cd /Users/i767700/Github/ai-in-finance/papers/conformal_covid

# Step 1: Classification SHAP (40-60 min)
bash run_shap_classification.sh

# Step 2: Correlation analysis
python code/analyze_n12_correlation.py
```

**Total Runtime:** ~40-60 minutes (only classification SHAP needed!)

---

### Option 2: Run in Background (Recommended)

```bash
cd /Users/i767700/Github/ai-in-finance/papers/conformal_covid

# Create master script
cat > run_day3_complete.sh << 'EOF'
#!/bin/bash
set -e

echo "Day 3: Starting SHAP Analysis"
echo "=============================="
date

# Step 1: Classification SHAP
echo ""
echo "Step 1: Running SHAP for classification tasks (4 tasks)..."
bash run_shap_classification.sh

# Step 2: Correlation
echo ""
echo "Step 2: Computing n=12 correlation..."
python code/analyze_n12_correlation.py

echo ""
echo "=============================="
echo "Day 3: COMPLETE!"
date
EOF

chmod +x run_day3_complete.sh

# Run in background
nohup bash run_day3_complete.sh > day3_execution.log 2>&1 &
echo "Started Day 3 execution (PID: $!)"
echo "Monitor progress: tail -f day3_execution.log"
```

---

## Expected Results

### SHAP Concentration (Hypotheses)

Based on conformal results:

**Robust tasks (small drop):**
- study-outcome: -1.3% drop → Expect LOW concentration (<40%)
- study-adverse: +5.2% drop → Expect LOW concentration (<40%)
- site-success: +3.8% drop → Expect LOW concentration (<40%)
- driver-dnf: +2.9% drop → Expect LOW concentration (<40%)
- sales-office: 0.0% drop → Expect LOW concentration (<40%)

**Severe/Catastrophic tasks (large drop):**
- sales-shipcond: 71.6% drop → Expect HIGH concentration (>40%)
- sales-group: 86.7% drop → Expect HIGH concentration (>40%)
- sales-payterms: 77.1% drop → Expect HIGH concentration (>40%)

### Correlation Prediction

**Expected:** Strong positive correlation
- Pearson r > 0.7
- p-value < 0.02 (strong significance)

**This would confirm:** SHAP concentration predicts coverage degradation across task types

---

## Monitoring

### Check Progress

```bash
# Quick status
ls -lh results/shap/shap_*.pkl | wc -l
# Should go from 7 → 8 → 12

# Watch live (classification)
tail -f logs/shap_study-outcome.log

# Watch live (master script if using Option 2)
tail -f day3_execution.log
```

### Files to Watch For

1. `results/shap/shap_rel-salt_sales-office.pkl`
2. `results/shap/shap_rel-trial_study-outcome.pkl`
3. `results/shap/shap_rel-trial_study-adverse.pkl`
4. `results/shap/shap_rel-trial_site-success.pkl`
5. `results/shap/shap_rel-f1_driver-dnf.pkl`

---

## Troubleshooting

### If Python Environment Issues

The scripts use `/Users/i767700/Github/ai-in-finance/.venv/bin/python3`

If this fails, check:
```bash
which python3
# Might need to update paths in scripts
```

### If SHAP Takes Too Long

Default subsample: 10,000
To speed up (less accurate):
```bash
--subsample 5000
```

### If Out of Memory

Reduce subsample size:
```bash
--subsample 5000  # or even 3000
```

---

## After Completion

### Verify Results

```bash
cd /Users/i767700/Github/ai-in-finance/papers/conformal_covid

# Check all SHAP files exist
echo "SHAP results:"
ls -1 results/shap/shap_*.pkl | wc -l
# Should be 12

# Check correlation results
echo ""
echo "Correlation results:"
ls -1 results/n12_* results/figure_n12_correlation.*
```

### Review Outputs

1. **Correlation plot:** `results/figure_n12_correlation.pdf`
   - Scatter plot: concentration vs drop
   - Should show strong positive correlation
   - Color-coded by task type

2. **Statistics:** `results/n12_statistics.txt`
   - Pearson r and p-value
   - Spearman ρ and p-value
   - Significance assessment

3. **Data table:** `results/n12_correlation_results.csv`
   - All 12 tasks with metrics
   - Can import to Excel/spreadsheet

4. **LaTeX table:** `results/table_n12_correlation.tex`
   - Ready to paste into paper

---

## Day 4 Preview

### After Day 3 Complete

1. **Review results** (30 min)
   - Verify correlation is significant (p < 0.02)
   - Check scatter plot looks good
   - Identify any outliers

2. **Update paper** (2-3 hours)
   - Add n=12 correlation result
   - Update figures
   - Revise text to reflect findings

3. **Final checks** (1 hour)
   - Proofread
   - Verify all figures/tables
   - Run LaTeX compilation

**Total Day 4:** ~3-4 hours

---

## Summary

**Current Status:**
- ✅ Day 2 COMPLETE (conformal experiments done)
- ✅ Day 3 infrastructure ready
- ✅ 7/12 SHAP results already computed
- ⏳ Day 3 ready to launch (50-70 min runtime)

**To Launch Day 3:**
```bash
cd /Users/i767700/Github/ai-in-finance/papers/conformal_covid
bash run_day3_complete.sh  # if created
# OR
bash run_shap_classification.sh  # just classification
```

**Recommendation:**
Start Day 3 tonight, let it run for ~1 hour, results ready by 11 PM!

---

**Ready to go! 🚀**
