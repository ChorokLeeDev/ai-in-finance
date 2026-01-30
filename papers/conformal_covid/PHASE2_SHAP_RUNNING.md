# Phase 2: SHAP Experiments - NOW RUNNING

**Started:** 2025-12-26 15:07
**Status:** ✅ Both tasks launched successfully

---

## Process Status

**Catastrophic Task (sales-shipcond):**
- PID: 3614
- Command: analyze_feature_importance.py --dataset rel-salt --task sales-shipcond
- Log: `shap_catastrophic.log`
- Expected runtime: 2-3 hours

**Robust Task (sales-office):**
- PID: 3615
- Command: analyze_feature_importance.py --dataset rel-salt --task sales-office
- Log: `shap_robust.log`
- Expected runtime: 2-3 hours

---

## What's Running

Both experiments are:
1. Loading rel-salt dataset
2. Training LightGBM model
3. Computing SHAP values on validation set (pre-COVID)
4. Computing SHAP values on test set (post-COVID)
5. Identifying top-10 most important features
6. Computing Jaccard similarity for all features
7. Generating plots

**Bottleneck:** SHAP computation on 10,000 samples (~1.5-2 hours per task)

---

## Monitoring Commands

### Check if still running:
```bash
ps -p 3614,3615
```

### View latest progress:
```bash
tail -30 papers/conformal_covid/shap_catastrophic.log
tail -30 papers/conformal_covid/shap_robust.log
```

### Monitor in real-time:
```bash
# Catastrophic task
tail -f papers/conformal_covid/shap_catastrophic.log

# Robust task
tail -f papers/conformal_covid/shap_robust.log
```

### Check for completion:
```bash
ls -lh papers/conformal_covid/results/shap/
```

Expected output files:
```
shap_rel-salt_sales-shipcond.pkl
shap_rel-salt_sales-office.pkl
shap_top10_sales-shipcond.pdf
shap_top10_sales-office.pdf
shap_scatter_sales-shipcond.pdf
shap_scatter_sales-office.pdf
shap_ranking_shift_sales-shipcond.pdf
shap_ranking_shift_sales-office.pdf
```

---

## Expected Findings

### Catastrophic Task (sales-shipcond):
**Hypothesis:** Top features have low Jaccard similarity

Expected results:
- Top feature: SALESDOCUMENT or similar transaction ID
- Feature Jaccard: < 0.1 (almost no overlap)
- Mean Jaccard (top-10): < 0.2
- **Interpretation:** Model relies on ephemeral identifiers

### Robust Task (sales-office):
**Hypothesis:** Top features have high Jaccard similarity

Expected results:
- Top feature: SALESORGANIZATION or stable entity
- Feature Jaccard: > 0.5 (good overlap)
- Mean Jaccard (top-10): > 0.4
- **Interpretation:** Model relies on persistent entities

---

## Timeline

**Current time:** 15:07
**Expected completion:** ~17:00-18:00 (2-3 hours)

**While running:**
- Logs may be buffered (updates every few minutes)
- CPU usage will be moderate (~50% per process)
- Can continue other work in parallel

**After completion:**
1. Results automatically saved to `results/shap/`
2. Plots automatically generated (3 PDFs per task)
3. Ready for analysis and paper integration

---

## What Happens Next (After SHAP Completes)

### Step 1: Verify Results
```bash
# Check both tasks completed
ls -lh results/shap/shap_rel-salt_*.pkl

# View summary
python3 -c "
import pickle
with open('results/shap/shap_rel-salt_sales-shipcond.pkl', 'rb') as f:
    r = pickle.load(f)
print('Catastrophic task:')
print(f\"  Mean Jaccard (top-10): {r['mean_jaccard_top10']:.3f}\")
print(f\"  Mean Jaccard (all): {r['mean_jaccard_all']:.3f}\")

with open('results/shap/shap_rel-salt_sales-office.pkl', 'rb') as f:
    r = pickle.load(f)
print('\\nRobust task:')
print(f\"  Mean Jaccard (top-10): {r['mean_jaccard_top10']:.3f}\")
print(f\"  Mean Jaccard (all): {r['mean_jaccard_all']:.3f}\")
"
```

### Step 2: Analyze Findings
- Confirm hypothesis (low Jaccard → catastrophic, high Jaccard → robust)
- Extract top-3 features for each task
- Note any surprising findings

### Step 3: Create Figure 3
- Combine individual plots into 2×2 panel layout
- Or use individual plots as separate subfigures

### Step 4: Write Paper Section
- New subsection: "Feature Importance Analysis" (~500 words)
- Add table comparing top features
- Update Abstract and Introduction

---

## Troubleshooting

### Issue: Processes disappeared
```bash
# Check if completed
ls -lh results/shap/

# Check for errors in log
tail -100 shap_catastrophic.log
tail -100 shap_robust.log
```

### Issue: Too slow (>4 hours)
**Likely cause:** SHAP on large sample

**Solution:** Can restart with smaller subsample:
```bash
# Kill current processes
kill 3614 3615

# Restart with 5k samples instead of 10k
PYTHONPATH=/Users/i767700/Github/ai-in-finance:$PYTHONPATH \
  python3 ... --subsample 5000
```

### Issue: Out of memory
**Solution:** Reduce subsample size or run sequentially

---

## Paper Updates in Progress

✅ **Completed:**
- Table 1 updated with 50-seed results
- Methodology updated (50 seeds)
- Drop range updated (0.1% to 77.1%)
- PDF compiled successfully (4 pages)

🔄 **Next:**
- Add SHAP analysis section
- Add Figure 3
- Update Abstract
- Update Introduction

---

## Progress Tracker

**Phase 1 (Foundation):** ✅ COMPLETE
- 50-seed ensemble: ✅ Done
- Paper Table 1 updated: ✅ Done
- Statistical rigor: ✅ Achieved

**Phase 2 (Mechanism):** 🔄 IN PROGRESS
- SHAP design: ✅ Done
- SHAP code: ✅ Done
- SHAP experiments: 🔄 Running (2-3 hours)
- Analysis: ⏳ Pending
- Paper integration: ⏳ Pending

**Phase 3 (Solution):** 📋 READY
- Retraining design: ✅ Done
- Retraining code: ✅ Done
- Experiments: ⏳ Pending (after Phase 2)
- Paper integration: ⏳ Pending

---

## UAI 2026 Acceptance Probability

**Current:** 50% (Borderline) - after 50-seed
**After SHAP:** 65% (Weak Accept) - mechanism explained
**After Retraining:** 75% (Accept) - complete story

**Timeline to Submission:**
- SHAP complete: ~18:00 today
- Retraining launch: Tonight
- Retraining complete: Tomorrow morning
- Paper integration: Tomorrow afternoon
- **Submission-ready: Dec 27-28**

---

**Status:** Experiments running smoothly! Check back in 2-3 hours.
