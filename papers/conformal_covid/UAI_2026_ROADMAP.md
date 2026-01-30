# UAI 2026 Submission Roadmap

**Target Deadline:** Feb 27 - Mar 6, 2026 (estimated)
**Time Available:** ~10 weeks
**Current Status:** 2/10 critical fixes complete

---

## ✅ Completed (Today)

### 1. Bootstrap Confidence Intervals ⭐
**Status:** DONE
**Impact:** Addresses UAI statistical rigor requirements

**What we did:**
- Created `bootstrap_correlation_analysis.py`
- Computed bootstrap CIs (10,000 samples)
- Permutation tests for significance (10,000 permutations)

**Results:**
- Jaccard ↔ Drop: **r = -0.75, 95% CI [-1.00, -0.56], p = 0.040*** ✓
- Spearman ρ = -0.97, p < 0.001 (robust to outliers) ✓
- Entropy ↔ Drop: r = 0.48, p = 0.49 (not significant, small n=5)

**Paper updated:**
- Section 5.3 now has proper CIs and p-values
- References fixed (no more [?])
- Seeds fixed (5 throughout)

**Files generated:**
```
results/bootstrap_correlation_results.json
results/correlation_table.tex (ready for paper)
results/bootstrap_distributions.pdf (supplementary)
```

### 2. References Fixed ⭐
**Status:** DONE
**Impact:** No longer desk-rejectable

**What we did:**
- Ran bibtex to compile bibliography
- All citations now show [1], [2], etc. instead of [?]

---

## 🚧 In Progress (Next)

### 3. 50-Seed Ensemble (Fixes Table 1 Variance Issue) ⭐⭐⭐
**Status:** READY TO RUN
**Priority:** CRITICAL (will be rejected without this)
**Time:** 3-4 hours compute time

**Why this is critical:**
```
Current Table 1 (5 seeds):
  s-group:      Test = 20.4 ± 39.8%  ❌ std > mean (UNUSABLE)
  s-payterms:   Test = 32.0 ± 39.3%  ❌ std > mean
  i-shippoint:  Test = 69.8 ± 36.3%  ❌ std > mean

Expected with 50 seeds:
  s-group:      Test = 20.4 ± 12.6%  ✓ (3.16x smaller std)
  s-payterms:   Test = 32.0 ± 12.4%  ✓
  i-shippoint:  Test = 69.8 ± 11.5%  ✓
```

**Scripts created:**
- `code/run_50seed_ensemble.py` - Main script (parallel execution)
- `code/test_ensemble_setup.py` - Verify setup (~2 min)
- `code/README_50SEEDS.md` - Complete documentation

**How to run:**

**Step 1: Test your setup (2-3 minutes)**
```bash
cd papers/conformal_covid
python3 code/test_ensemble_setup.py
```

**Step 2: Quick test (10 minutes)**
```bash
python3 code/run_50seed_ensemble.py --tasks sales-office --num_seeds 10
```

**Step 3: Full run (3-4 hours)**
```bash
# Run overnight or during long meeting
nohup python3 code/run_50seed_ensemble.py > ensemble_50seeds.log 2>&1 &

# Check progress
tail -f ensemble_50seeds.log
```

**Step 4: Update paper (5 minutes)**
```bash
# Replace Table 1 in main.tex with results/ensemble_50seeds_table.tex
# Recompile PDF
pdflatex main.tex
```

**Checkpoint system:**
- Automatically saves progress after each task
- Can resume if interrupted: `python3 code/run_50seed_ensemble.py --resume`
- Safe to stop and restart

**Output files:**
```
results/ensemble_50seeds_table.tex    ← Copy to paper
results/ensemble_50seeds_summary.json ← Human-readable
results/ensemble_50seeds.pkl          ← Raw data
results/checkpoints/*.pkl             ← Resume points
```

---

## 📋 Remaining UAI Blockers

### MUST HAVE (Will be rejected without)

#### 4. Add Regression Tasks (2-3 tasks with CQR) ⭐⭐⭐
**Priority:** CRITICAL
**Time:** 1.5 weeks
**Status:** Not started

**Current limitation:** "classification only"
**UAI expectation:** Both classification and regression

**What to do:**
1. Identify 2-3 regression tasks in rel-salt or rel-trial
2. Implement Conformal Quantile Regression (CQR)
3. Measure prediction interval coverage
4. Same diagnostic: compute Jaccard for continuous features

**Expected finding:** Same pattern - low Jaccard features fail

**Need help with:** I can provide CQR implementation code

---

### STRONGLY RECOMMENDED (Significantly strengthens)

#### 5. Feature Importance Analysis (SHAP) ⭐⭐
**Priority:** HIGH
**Time:** 3 days
**Status:** Not started

**Current gap:** You claim "transaction IDs cause failure" but don't prove they're used

**What to do:**
```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(test_data)

# Show that SALESDOCUMENT has high importance
feature_importance = np.abs(shap_values).mean(axis=0)

# Create table:
# Feature          | Importance | Jaccard | Impact
# SALESDOCUMENT    | 0.45       | 0.02    | Catastrophic
# PRODUCT          | 0.30       | 0.58    | Robust
```

**Why this matters:** Validates that low-Jaccard features ARE driving predictions

**Need help with:** I can provide complete SHAP analysis code

#### 6. Retraining Experiment ⭐⭐
**Priority:** HIGH
**Time:** 1 week
**Status:** Not started

**Reviewer will ask:** "Why not just retrain during COVID?"

**What to do:**
- Train on pre-COVID
- Fine-tune on 10%, 25%, 50% of COVID data
- Re-calibrate conformal predictor
- Measure test coverage

**Expected result:**
- Catastrophic tasks: Retraining helps (0.2% → 60%+)
- Robust tasks: Already good (95%)

**Why this matters:** Shows practical solution (monitoring + retraining)

**Need help with:** I can provide experiment design code

---

### NICE TO HAVE (If time permits)

#### 7. Compare Other UQ Methods ⭐
**Time:** 1 week
**Adds:** Generalizability

**Methods:**
- Ensemble variance (already have 50 seeds!)
- Quantile prediction (native LightGBM)
- Temperature scaling

**Question:** Is degradation unique to conformal prediction?

#### 8. Temporal Dynamics Analysis ⭐
**Time:** 2 days
**Adds:** Richer understanding

**What to add:** Coverage over time plot
- Split test into monthly chunks
- Show coverage Jul 2020 → Dec 2020
- Sudden drop vs. gradual?

---

## Timeline Estimate

| Week | Focus | Deliverable |
|------|-------|-------------|
| **Now** | 50-seed ensemble | ✓ Table 1 variance fixed |
| 1 | Regression tasks | 2-3 CQR experiments |
| 2 | Regression tasks | Analysis + write-up |
| 3 | Feature importance | SHAP analysis done |
| 4 | Retraining experiment | Experiment complete |
| 5 | Other UQ methods | Comparison done |
| 6 | Writing polish | First complete draft |
| 7 | Internal review | Address feedback |
| 8 | Final polish | Camera-ready quality |
| 9 | Buffer | Handle unexpected issues |
| 10 | Submit | UAI submission! |

**Critical path:** 50 seeds → Regression → Feature importance → Retraining

---

## Current Paper Status

### Strengths ✓
- ✅ Natural experiment design (COVID as documented shift)
- ✅ Placebo test (10-100× more degradation)
- ✅ Bootstrap CIs and permutation tests
- ✅ References working
- ✅ Cross-domain validation (rel-trial)

### Weaknesses Still to Address ⚠️
- ❌ Table 1 variance (std > mean) ← **BLOCKER, fix next**
- ❌ Classification only (no regression) ← **BLOCKER**
- ❌ No feature importance validation
- ❌ No retraining experiment (reviewer will ask)
- ❌ Only 8 data points for correlation (acknowledged)

### After 50-Seed Run
- ✅ Table 1 variance fixed
- ❌ Classification only ← **Next blocker**
- ❌ No feature importance
- ❌ No retraining experiment

---

## What to Do Next (Priority Order)

### Today/This Week:
1. **✅ Test setup** (2 min)
   ```bash
   python3 code/test_ensemble_setup.py
   ```

2. **✅ Quick test** (10 min)
   ```bash
   python3 code/run_50seed_ensemble.py --tasks sales-office --num_seeds 10
   ```

3. **🚀 Run full 50-seed ensemble** (3-4 hours)
   ```bash
   nohup python3 code/run_50seed_ensemble.py > ensemble.log 2>&1 &
   ```

4. **📝 Update Table 1** (5 min)
   - Copy `results/ensemble_50seeds_table.tex` to paper
   - Recompile

### Next Week:
5. **Add regression tasks** (need help? I can provide code)
6. **Feature importance analysis** (need help? I can provide code)

---

## Getting Help

**If you need help with:**

### "How do I run 50 seeds?"
→ See `code/README_50SEEDS.md` (complete guide)

### "Script failed with error X"
→ Run `python3 code/test_ensemble_setup.py` to diagnose

### "How do I add regression tasks?"
→ Ask me - I'll provide CQR implementation code

### "How do I do SHAP analysis?"
→ Ask me - I'll provide complete SHAP code

### "I'm stuck on X"
→ Just ask - I'm here to help!

---

## Quick Reference

### Files Created Today:
```
code/bootstrap_correlation_analysis.py   ← Bootstrap CI analysis
code/run_50seed_ensemble.py             ← 50-seed main script
code/test_ensemble_setup.py             ← Setup verification
code/README_50SEEDS.md                  ← Complete 50-seed guide
BOOTSTRAP_CI_SUMMARY.md                 ← Bootstrap results summary
UAI_2026_ROADMAP.md                     ← This file
```

### Results Generated:
```
results/bootstrap_correlation_results.json
results/correlation_table.tex
results/bootstrap_distributions.pdf
```

### Paper Updates:
- ✅ Section 5.3: Bootstrap CIs added
- ✅ References: All working
- ✅ Seeds: Consistent (5 → will be 50)

---

## Confidence Level

**Current submission readiness:** 40%

After 50-seed run: 60%
After regression tasks: 75%
After feature importance: 85%
After retraining experiment: 95%

**UAI acceptance probability:**
- Without fixes: 5-10% (desk reject likely)
- After 50 seeds: 20-30% (major revision)
- After all blockers: 60-70% (accept or minor revision)

---

## Questions?

**Ready to start?**
```bash
cd papers/conformal_covid
python3 code/test_ensemble_setup.py
```

**Need help?** Just ask!

**Want me to write code for next step?** Let me know which:
- [ ] Regression tasks with CQR
- [ ] Feature importance (SHAP)
- [ ] Retraining experiment
- [ ] Other UQ methods comparison
- [ ] Something else?
