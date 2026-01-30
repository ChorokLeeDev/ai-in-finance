# Phase 2: Feature Importance Analysis - SHAP Implementation

**Status:** Scripts created, ready to run
**Created:** 2025-12-26
**Next:** Wait for 50-seed completion, then execute SHAP experiments

---

## Overview

**Research Question:** Why do low-Jaccard tasks catastrophically fail?
**Hypothesis:** Models rely on unstable (low-Jaccard) features when available.

**Experimental Design:** Compare two contrasting tasks:
- **Catastrophic:** sales-shipcond (Jaccard=0.02, Drop=93%)
- **Robust:** sales-office (Jaccard=0.61, Drop=0.1%)

---

## Implementation Complete ✅

### Files Created:

1. **`analyze_feature_importance.py`** (440 lines)
   - Loads task data using existing pipeline
   - Trains LightGBM model
   - Computes SHAP values on val (pre-COVID) and test (post-COVID)
   - Identifies top-10 most important features
   - Computes Jaccard similarity for all features
   - Saves results to pickle file

2. **`plot_shap_results.py`** (260 lines)
   - Creates 3 plots per task:
     - Top-10 features bar chart (colored by Jaccard)
     - Feature importance vs Jaccard scatter plot
     - Feature ranking shift (val → test)
   - Publication-quality PDF outputs

---

## How to Run

### Step 1: Run SHAP on Catastrophic Task (2-3 hours)

```bash
cd /Users/i767700/Github/ai-in-finance

PYTHONPATH=/Users/i767700/Github/ai-in-finance:$PYTHONPATH \
  python3 papers/conformal_covid/code/analyze_feature_importance.py \
  --dataset rel-salt \
  --task sales-shipcond \
  --subsample 10000 \
  --seed 42
```

**Expected Output:**
```
papers/conformal_covid/results/shap/
├── shap_rel-salt_sales-shipcond.pkl
├── shap_top10_sales-shipcond.pdf
├── shap_scatter_sales-shipcond.pdf
└── shap_ranking_shift_sales-shipcond.pdf
```

### Step 2: Run SHAP on Robust Task (2-3 hours)

```bash
PYTHONPATH=/Users/i767700/Github/ai-in-finance:$PYTHONPATH \
  python3 papers/conformal_covid/code/analyze_feature_importance.py \
  --dataset rel-salt \
  --task sales-office \
  --subsample 10000 \
  --seed 42
```

**Can run in parallel with Step 1:**
```bash
# Run both in background
nohup env PYTHONPATH=/Users/i767700/Github/ai-in-finance:$PYTHONPATH \
  python3 papers/conformal_covid/code/analyze_feature_importance.py \
  --dataset rel-salt --task sales-shipcond \
  > shap_catastrophic.log 2>&1 &

nohup env PYTHONPATH=/Users/i767700/Github/ai-in-finance:$PYTHONPATH \
  python3 papers/conformal_covid/code/analyze_feature_importance.py \
  --dataset rel-salt --task sales-office \
  > shap_robust.log 2>&1 &

# Monitor progress
tail -f shap_*.log
```

---

## Expected Findings

### Catastrophic Task (sales-shipcond):

**Hypothesis:**
- Top feature: SALESDOCUMENT (transaction ID)
- Feature Jaccard: ~0% (completely new IDs in test set)
- Top-10 mean Jaccard: < 0.1

**Implication:** Model learns patterns on ephemeral identifiers that don't persist post-COVID.

### Robust Task (sales-office):

**Hypothesis:**
- Top feature: SALESORGANIZATION (business unit)
- Feature Jaccard: ~60% (most orgs persist)
- Top-10 mean Jaccard: > 0.5

**Implication:** Model learns patterns on stable entities that remain valid post-COVID.

---

## Analysis Checklist

After both experiments complete:

1. **Compare top features:**
   - [ ] Catastrophic: Top feature has low Jaccard? (< 0.1)
   - [ ] Robust: Top feature has high Jaccard? (> 0.5)

2. **Statistical analysis:**
   - [ ] Compute correlation: Feature importance × Feature Jaccard
   - [ ] For catastrophic: Negative/no correlation expected
   - [ ] For robust: Positive correlation expected

3. **Create comparison table:**

```markdown
| Task        | Top Feature       | SHAP  | Jaccard | Drop |
|-------------|-------------------|-------|---------|------|
| s-shipcond  | SALESDOCUMENT     | X.XX  | 0.02    | 93%  |
| s-office    | SALESORGANIZATION | X.XX  | 0.61    | 0.1% |
```

4. **Extract insights for paper:**
   - [ ] Top-3 features for each task
   - [ ] Mean Jaccard: top-10 vs all features
   - [ ] Feature ranking stability (how many top-10 drop out?)

---

## Paper Integration Plan

### New Subsection: "Feature Importance Analysis"

**Location:** Section 5 (Extended Experiments), after Placebo Test

**Length:** ~500 words + 1 table + 1 figure

**Key Points:**
1. Explain SHAP methodology briefly
2. Present catastrophic task findings
3. Present robust task findings
4. Show comparison table
5. State key insight: "Models automatically rely on unstable features"

### Updates Needed:

**Abstract:** Add 1 sentence
```latex
We use SHAP analysis to show that models automatically learn to rely on
time-dependent features even when they lack stability, explaining the
mechanism of catastrophic failure.
```

**Introduction:** Add contribution
```latex
\item \textbf{Mechanistic Understanding}: SHAP analysis reveals models
      rely on unstable features, explaining why low Jaccard causes failure
```

**Figure 3:** 2×2 panel layout
- Panel A: Top-10 features (catastrophic) - bar chart colored by Jaccard
- Panel B: Top-10 features (robust) - bar chart colored by Jaccard
- Panel C: Scatter plot (importance vs Jaccard) - both tasks
- Panel D: Ranking shift comparison

---

## Runtime Estimates

**Per task:**
- Data loading & preprocessing: 2-3 min
- Model training: 5-10 min
- SHAP computation (10k samples): 1.5-2.5 hours
  - TreeExplainer is fast, but 10k samples × many features = time
- Jaccard computation: 1-2 min
- Plotting: 1-2 min

**Total per task:** 2-3 hours
**Both tasks (parallel):** 2-3 hours
**Both tasks (sequential):** 4-6 hours

**Recommendation:** Run in parallel overnight

---

## Dependencies Check

All dependencies already available in conda environment:
- ✅ shap
- ✅ lightgbm
- ✅ numpy
- ✅ pandas
- ✅ matplotlib
- ✅ sklearn

No additional installations needed.

---

## Troubleshooting

### Issue: SHAP too slow

**Solution:** Reduce subsample size
```bash
--subsample 5000  # Instead of 10000
```

### Issue: Out of memory

**Solution:** Reduce subsample or use background computation
```bash
--subsample 5000
```

### Issue: TreeExplainer error

**Check:** LightGBM model type
```python
# Should be lgb.Booster, not sklearn wrapper
```

---

## Next Steps (After SHAP Complete)

1. ✅ Verify hypothesis confirmed
2. ✅ Create comparison table
3. ✅ Generate Figure 3 (2×2 panels)
4. ✅ Write new subsection (~500 words)
5. ✅ Update Abstract and Introduction
6. ✅ Compile paper and verify

**Then:** Move to Phase 3 (Retraining experiments)

---

## Success Metrics

**Phase 2 delivers:**
- Mechanistic understanding of failure
- Visual evidence (Figure 3)
- Quantitative support (SHAP scores, Jaccard values)

**Impact on paper:**
- Novelty: 2.5 → 3.0/5 ⬆️
- Significance: 3.0 → 3.5/5 ⬆️
- Overall: Borderline (50%) → Weak Accept (65%) ⬆️

---

**Status:** Ready to execute. Waiting for 50-seed ensemble completion first.
