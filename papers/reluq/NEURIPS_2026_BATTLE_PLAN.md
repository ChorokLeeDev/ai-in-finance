# NeurIPS 2026 Battle Plan - Aggressive Strategy

**Target:** NeurIPS 2026 Main Conference (May 2026 deadline)
**Backup:** KDD 2026 (February 2026 deadline)
**Current Probability:** 35-45% → **Target: 75-80%**
**Timeline:** 20 weeks (Dec 23, 2025 - May 15, 2026)

---

## Strategy: Front-Load Critical Gaps, Add Novelty Mid-Way

### Success Criteria for NeurIPS
- ✅ Strong baselines (SHAP comparison)
- ✅ Fixed methodology (permutation or alternative)
- ✅ Novelty boost (active learning)
- ✅ Broad validation (8+ domains, classification + regression)
- ✅ Scale demonstration (100K samples)
- ✅ Modern UQ methods (conformal prediction)

### KDD Fallback (Week 6 Checkpoint)
If progress is slow or results are weak, submit strong KDD paper with:
- SHAP baseline
- 6 domains
- Regression only
- Core FK attribution validated

---

## Phase 1: Critical Fixes (Week 1-6) - MAKE OR BREAK

### Week 1-2: SHAP Baseline Comparison ⚠️ HIGHEST PRIORITY

**Why Critical:** Reviewers will reject without SOTA baseline comparison.

**Experiment Design:**
```python
# Four attribution methods, same data:
methods = [
    'TreeSHAP-Individual',    # Standard SHAP on all features
    'TreeSHAP-FK',           # SHAP with predefined FK groups
    'TreeSHAP-Correlation',  # SHAP with correlation clusters
    'RelUQ-Permutation',     # Your current method
]

# Three evaluation metrics:
metrics = {
    'stability': 'Spearman correlation across 5 random seeds',
    'attribution_error_corr': 'Spearman(attribution, error_impact)',
    'actionability': 'Manual score: can practitioner act on this?',
}

# Test on all 6 domains:
domains = ['rel-salt', 'rel-trial', 'rel-f1', 'rel-hm', 'rel-avito', 'rel-stack']
```

**Expected Outcome:**
| Method | Stability | Attr-Error ρ | Actionability |
|--------|-----------|--------------|---------------|
| SHAP-Individual | 0.85 | 0.70 | Low (24 features) |
| SHAP-FK | 0.88 | 0.85 | High (5 FK groups) |
| SHAP-Corr | 0.60 | 0.65 | Medium (unstable groups) |
| RelUQ-Perm | 0.93 | 0.90 | High (5 FK groups) |

**Key Message:** "FK grouping improves SHAP stability and actionability"

**Code to Write:**
- `experiments/shap_baseline_comparison.py`
- `experiments/correlation_clustering.py`
- `analysis/compare_attribution_methods.py`

**Deliverable:** Table + Figure for main paper

**Time Budget:** 10 days

---

### Week 2-3: Fix Permutation Method ⚠️ CRITICAL

**The Problem:** Your notes show permutation-based UQ attribution gave equal values (33.33% for 3 groups).

**Root Cause Analysis:**

From your experiment notes:
> "For bootstrap ensembles, permuting features doesn't significantly change ensemble variance because ensemble variance comes from model diversity (different bootstrap samples)"

**This is theoretically incorrect for the following reason:**

Ensemble variance = Epistemic uncertainty should reflect "what if the model saw different data?"

When you permute FK group X:
- Model predictions SHOULD change (you break X→Y relationship)
- Model disagreement SHOULD increase (models trained on different bootstraps react differently)

**Why it failed:** Likely implementation issue, not fundamental problem.

**Debugging Steps:**

1. **Verify base uncertainty is non-zero:**
   ```python
   base_uncertainty = ensemble_variance(X)
   assert base_uncertainty > 0, "Zero variance = no diversity"
   ```

2. **Verify permutation changes predictions:**
   ```python
   pred_base = ensemble_predict(X)
   pred_perm = ensemble_predict(permute(X, fk_group='RESULTS'))
   assert not np.allclose(pred_base, pred_perm), "Permutation has no effect"
   ```

3. **Verify permutation increases uncertainty:**
   ```python
   unc_base = ensemble_variance(X)
   unc_perm = ensemble_variance(permute(X, fk_group='RESULTS'))
   print(f"Base: {unc_base}, Permuted: {unc_perm}, Ratio: {unc_perm/unc_base}")
   # Should see ratio > 1.0 for important FK groups
   ```

**Three Scenarios:**

**Scenario A: It works after debugging** (60% probability)
- Keep permutation method
- Add to paper: "Ensemble subsampling rate 0.7-0.8 is critical for variance"
- Proceed with current approach

**Scenario B: It partially works** (30% probability)
- Works for some domains (SALT, Trial) but not others (Stack)
- This is actually GOOD - it's another way to detect EP domains
- Add diagnostic: "If permutation doesn't increase uncertainty, not EP domain"

**Scenario C: It fundamentally doesn't work** (10% probability)
- Drop permutation entirely
- Switch to SHAP-based approach:
  ```python
  # Use SHAP values variance as uncertainty attribution
  shap_values = [shap_explainer(model_i, X) for model_i in ensemble]
  uncertainty_attr = np.var(shap_values, axis=0)  # Variance of SHAP values
  fk_attr = group_by_fk(uncertainty_attr)
  ```

**Deliverable:** Working permutation method OR justified alternative

**Time Budget:** 7 days

---

### Week 3-4: Add 2 More EP Domains

**Target Domains:**

1. **Banking: Transaction Fraud Detection**
   - Database: CUSTOMER → ACCOUNT → TRANSACTION → MERCHANT
   - Task: Predict transaction amount (regression) or fraud (classification)
   - EP Structure: Customer attributes → Account risk → Transaction pattern
   - Dataset: Kaggle IEEE-CIS Fraud Detection (relational subset)

2. **Manufacturing: Equipment Failure Prediction**
   - Database: FACTORY → MACHINE → SENSOR → READING
   - Task: Predict time-to-failure or failure probability
   - EP Structure: Factory config → Machine specs → Sensor health
   - Dataset: NASA Turbofan Engine Degradation or Azure Predictive Maintenance

**Alternative if above unavailable:**

3. **Insurance: Claims Amount Prediction**
   - Database: POLICY → CUSTOMER → CLAIM → PROVIDER
   - EP Structure: Policy terms → Customer risk → Claim severity

**Validation Criteria for Each Domain:**
- ✅ Spearman ρ ≥ 0.80 (attribution-error correlation)
- ✅ Top FK identified correctly in noise injection test
- ✅ Domain experts agree FK ranking makes sense (if possible)

**Deliverable:** 2 new domains with full validation

**Time Budget:** 10 days

---

### Week 5-6: KDD Paper Draft + Checkpoint

**KDD Submission Decision Point (Feb 15, 2026):**

**Submit to KDD if:**
- ❌ SHAP baseline shows RelUQ is NOT better (ρ similar or worse)
- ❌ Permutation method cannot be fixed
- ❌ Only 6 domains validated (no new domains added)
- ❌ Falling behind schedule (>2 weeks delay)

**KDD Paper Scope (Conservative Version):**
- Title: "Schema-Guided Uncertainty Attribution for Enterprise ML"
- Focus: Practical tool for ERP/clinical/retail domains
- Core contribution: FK grouping for stability + actionability
- 6 domains, regression only, ensemble-based
- Target: Applied data mining audience (perfect fit)

**Skip KDD if:**
- ✅ SHAP baseline shows RelUQ is clearly better
- ✅ Permutation method works reliably
- ✅ 8+ domains validated
- ✅ On schedule

**Deliverable:** Decision + KDD draft (if submitting) OR continue to Phase 2

**Time Budget:** Week 6 checkpoint

---

## Phase 2: Novelty Boost (Week 7-12) - DIFFERENTIATION

### Week 7-10: FK-Level Active Learning ⭐ GAME CHANGER

**Why This Matters:**

This transforms your paper from "attribution method" to "full system for data quality optimization."

Reviewers will see: "Oh, this isn't just explaining uncertainty, it's actionable."

**Experimental Design:**

**Setup:**
```python
# Start with sparse training data (20% of full dataset)
# Iteratively add data based on FK-level uncertainty

strategies = {
    'RelUQ-FK-Guided': 'Add samples from highest-uncertainty FK group',
    'Random': 'Add random samples (baseline)',
    'Uncertainty-Sampling': 'Add highest-uncertainty samples (standard AL)',
    'Diversity-Sampling': 'Add diverse samples (coverage baseline)',
}

# Simulate 10 iterations of data acquisition
# Each iteration: add 10% more data (200 samples)
# Measure: uncertainty reduction, accuracy gain, cost efficiency
```

**FK-Guided Acquisition Algorithm:**
```python
def fk_guided_acquisition(X_pool, ensemble, fk_groups, budget=200):
    # 1. Compute FK-level uncertainty attribution
    fk_attribution = compute_fk_attribution(ensemble, X_pool)

    # 2. Prioritize FK groups
    top_fk = max(fk_attribution, key=fk_attribution.get)

    # 3. Sample from top FK group
    samples_in_fk = [x for x in X_pool if x.fk_group == top_fk]

    # 4. Within FK, use uncertainty sampling
    uncertainties = [ensemble_variance(x) for x in samples_in_fk]
    selected = top_k(samples_in_fk, uncertainties, k=budget)

    return selected
```

**Evaluation Metrics:**
1. **Uncertainty Reduction:** How much does uncertainty drop per sample added?
2. **Accuracy Gain:** MAE improvement per sample added
3. **Cost Efficiency:** If FK groups have different collection costs, optimize uncertainty/cost ratio
4. **Convergence Speed:** Iterations to reach target accuracy

**Expected Results:**

| Strategy | Uncertainty @ 50% data | Samples to 90% accuracy | Cost (if FK-weighted) |
|----------|------------------------|-------------------------|------------------------|
| Random | 0.15 | 5000 | $5000 |
| Uncertainty Sampling | 0.12 | 4200 | $4200 |
| Diversity Sampling | 0.13 | 4500 | $4500 |
| **RelUQ FK-Guided** | **0.08** | **3500** | **$2800** |

**Key Insights to Highlight:**

1. **Table-Level Acquisition:** "Acquire data for TABLE X" is more actionable than "acquire sample 12437"
2. **Cost-Aware:** Different tables have different collection costs (e.g., customer data is cheap, lab tests are expensive)
3. **Practical Impact:** Enterprises care about ROI - "30% fewer samples for same accuracy = 30% cost reduction"

**Figures to Generate:**
- Learning curves (accuracy vs samples added)
- Uncertainty reduction curves
- Cost-efficiency frontier (if costs available)
- Heatmap: FK group contribution over iterations

**Deliverable:** Full active learning section + 3-4 figures

**Time Budget:** 4 weeks (this is substantial work)

---

### Week 11-12: Classification Extension

**Current Gap:** Only regression tasks

**Extension:** Use ensemble disagreement for classification

**Method:**
```python
# For classification task
def ensemble_disagreement(x, ensemble):
    predictions = [model.predict_proba(x) for model in ensemble]
    # Disagreement = variance in predicted probabilities
    disagreement = np.var(predictions, axis=0).sum()
    return disagreement

# FK attribution for classification
def fk_attribution_classification(X, ensemble, fk_groups):
    base_disagreement = mean([ensemble_disagreement(x) for x in X])

    attribution = {}
    for fk in fk_groups:
        X_perm = permute(X, fk_group=fk)
        perm_disagreement = mean([ensemble_disagreement(x) for x in X_perm])
        attribution[fk] = perm_disagreement - base_disagreement

    return normalize(attribution)
```

**Datasets:**
- rel-trial: study-outcome (binary classification)
- rel-f1: driver-dnf, driver-top3 (binary classification)
- Banking: fraud detection (binary classification)

**Validation:**
- Compute attribution-error correlation (use misclassification increase instead of MAE increase)
- Expected ρ ≥ 0.80 for EP domains

**Deliverable:** Classification results table + 1 figure

**Time Budget:** 2 weeks

---

## Phase 3: Polish & Scale (Week 13-16) - ROBUSTNESS

### Week 13-14: Scale Validation (100K Samples)

**Current Gap:** Only 3K-10K samples tested

**Experiment:**
```python
# Use SALT or synthetic EP dataset
# Scale up to 100,000 samples

scale_tests = [1_000, 5_000, 10_000, 50_000, 100_000]

for n in scale_tests:
    X_train = sample(n)
    ensemble = train_ensemble(X_train)
    attribution = compute_fk_attribution(ensemble, X_test)

    # Measure:
    # - Runtime (should scale linearly)
    # - Stability (should improve with more data)
    # - Attribution accuracy (should plateau)
```

**Expected Results:**
- Runtime: O(n) scaling (show plot)
- Stability: Plateaus at n≥10K
- Attribution: Consistent across scales

**Deliverable:** Scale validation figure + runtime table

**Time Budget:** 2 weeks

---

### Week 15-16: Conformal Prediction Integration

**Why Important:** Conformal prediction is state-of-the-art for UQ in 2025

**Method:**
```python
from mapie.regression import MapieRegressor

# Train ensemble with conformal calibration
ensemble_conformal = [
    MapieRegressor(estimator=model, method='plus', cv=5)
    for model in base_ensemble
]

# FK attribution on prediction interval width
def fk_attribution_conformal(X, ensemble_conformal, alpha=0.1):
    # Base prediction intervals
    intervals = [model.predict(X, alpha=alpha) for model in ensemble_conformal]
    base_width = mean([interval[:, 1] - interval[:, 0] for interval in intervals])

    # Permute FK and measure interval change
    attribution = {}
    for fk in fk_groups:
        X_perm = permute(X, fk_group=fk)
        intervals_perm = [model.predict(X_perm, alpha=alpha) for model in ensemble_conformal]
        perm_width = mean([interval[:, 1] - interval[:, 0] for interval in intervals_perm])
        attribution[fk] = perm_width - base_width

    return normalize(attribution)
```

**Validation:**
- Compare conformal attribution vs ensemble variance attribution
- Expected: High correlation (both measure uncertainty)
- Benefit: Conformal gives calibrated intervals (more rigorous)

**Deliverable:** Conformal comparison table

**Time Budget:** 2 weeks

---

## Phase 4: Writing & Submission (Week 17-20)

### Week 17-18: Paper Writing

**Structure (NeurIPS Format):**

1. **Abstract** (150 words)
   - Problem: UQ attribution for enterprise ML
   - Solution: FK-level grouping + Error Propagation Hypothesis
   - Results: ρ ≥ 0.85 on 8 domains, 30% cost reduction via active learning

2. **Introduction** (1.5 pages)
   - Hook: "80% of enterprise ML uses relational databases, but UQ research ignores this"
   - Gap: Feature-level attribution is unstable and not actionable
   - Contribution: Schema-guided attribution + active learning framework

3. **Related Work** (1 page)
   - UQ methods (ensembles, MC Dropout, conformal)
   - Attribution methods (SHAP, permutation importance)
   - Active learning (standard vs FK-level)
   - Relational learning (RelBench, GNNs)

4. **Method** (2 pages)
   - FK grouping algorithm
   - Attribution via permutation (or SHAP)
   - Error Propagation Hypothesis (formal definition)
   - Active learning extension

5. **Experiments** (3 pages)
   - 8 domains: 6 EP (SALT, Trial, F1, H&M, Banking, Manufacturing), 2 non-EP (Amazon, Stack)
   - Baselines: SHAP-Individual, SHAP-FK, SHAP-Correlation, Random
   - Main result: Attribution-error correlation table
   - Active learning: Learning curves + cost analysis
   - Classification: Results on 3 binary tasks
   - Scale: 100K sample validation
   - Conformal: Comparison table

6. **Discussion** (1 page)
   - When it works: EP domains (define criteria)
   - When it fails: Non-EP domains (explain why)
   - Practical impact: Cost reduction, data quality optimization

7. **Conclusion** (0.5 pages)
   - Summary + limitations + future work

**Appendix:**
- Detailed algorithms
- Additional ablations
- Domain descriptions
- Hyperparameters

**Figure Budget (8 main figures):**
1. Overview pipeline
2. SHAP baseline comparison (stability + attribution-error)
3. Domain validation (8 domains, ρ comparison)
4. Active learning learning curves
5. Cost efficiency analysis
6. Classification results
7. Scale validation
8. Conformal comparison

**Time Budget:** 2 weeks

---

### Week 19: Figure Generation & Polish

**All Figures Must Be:**
- Publication quality (vector graphics, 300 DPI)
- Consistent style (colors, fonts, layout)
- Clear legends and labels
- Accessible (colorblind-friendly palette)

**Use Seaborn + Matplotlib:**
```python
import seaborn as sns
sns.set_context("paper")
sns.set_style("whitegrid")
palette = sns.color_palette("colorblind")
```

**Time Budget:** 1 week

---

### Week 20: Submission

**Pre-Submission Checklist:**
- ✅ Anonymized (no author names, affiliations)
- ✅ 9 pages main + unlimited appendix
- ✅ References formatted correctly
- ✅ Figures referenced in text
- ✅ Code released (anonymous GitHub)
- ✅ Supplementary material (proofs, extra results)
- ✅ Reproducibility checklist completed
- ✅ Ethics statement (if applicable)

**Supplementary Material:**
- Code (anonymized repo)
- Full results tables
- Additional figures
- Proofs of theorems
- Dataset descriptions

**Submit to NeurIPS 2026:** May 15, 2026 (estimated deadline)

---

## Risk Management & Contingencies

### Red Flags to Watch For

**Week 6 Checkpoint (KDD Decision):**
- If ≥2 critical experiments failed → Submit to KDD
- If on schedule but weak results → Submit to KDD
- If strong results but behind schedule → Skip KDD, continue to NeurIPS

**Week 12 Checkpoint (NeurIPS Feasibility):**
- If Phase 1+2 complete with strong results → High confidence NeurIPS
- If only Phase 1 complete → Medium confidence (60-65%)
- If Phase 1 incomplete → Pivot to KDD or defer to NeurIPS 2027

### Parallel Strategies

**Don't Block on Sequential Experiments:**

Can run in parallel:
- Week 1-2: SHAP baseline (Person A) + Permutation debugging (Person B)
- Week 3-4: Banking domain (Person A) + Manufacturing domain (Person B)
- Week 7-10: Active learning (Person A) + Classification extension (Person B)

If you have collaborators, parallelize. If solo, prioritize P0 tasks first.

---

## Success Metrics

### Minimum Viable NeurIPS Paper
- ✅ 8+ domains validated
- ✅ SHAP baseline showing RelUQ is better
- ✅ Active learning showing 20%+ efficiency gain
- ✅ Classification extension working
- ✅ Clear scope (EP domains only, honest about limitations)

### Strong NeurIPS Paper (Target)
- ✅ All above +
- ✅ 100K scale validation
- ✅ Conformal prediction support
- ✅ Cost-aware active learning with real enterprise cost data
- ✅ Domain expert validation (if possible)

### Exceptional NeurIPS Paper (Stretch)
- ✅ All above +
- ✅ Causal attribution theory (from future directions)
- ✅ Real-world deployment case study
- ✅ Open source library released

---

## Estimated Probabilities

| Milestone | NeurIPS Acceptance |
|-----------|-------------------|
| Current state | 35-45% |
| + Phase 1 (Week 6) | 55-65% |
| + Phase 2 (Week 12) | 70-75% |
| + Phase 3 (Week 16) | 75-80% |
| + Stretch goals | 80-85% |

**Conservative estimate: 75% if all critical experiments succeed**

---

## Next Actions (This Week)

### Day 1-2: SHAP Baseline Setup
1. Install SHAP library
2. Load all 6 current domains
3. Run TreeSHAP on rel-salt as proof-of-concept
4. Verify: Can we group SHAP values by FK?

### Day 3-4: Permutation Debugging
1. Reproduce "equal attribution" bug
2. Instrument code to trace where it fails
3. Test hypothesis: subsampling rate, ensemble diversity
4. Document findings

### Day 5-7: First Comparison Results
1. Full SHAP vs RelUQ comparison on rel-salt
2. Generate comparison table
3. If RelUQ is clearly better → proceed
4. If not → pivot to SHAP-based FK attribution

**Your immediate task:** Start with SHAP baseline this week. This is the make-or-break experiment.

---

## KDD Backup Plan (If Needed)

**KDD 2026 Submission (Week 6 alternative):**

**Title:** "FK-Level Uncertainty Attribution for Enterprise Machine Learning"

**Scope:**
- Focus on practical value for enterprises
- 6 domains (SALT, Trial, F1, H&M, Avito, Stack)
- Regression tasks only
- Core FK attribution validated
- SHAP comparison (if available)

**Target Audience:** Applied data miners, enterprise ML engineers

**Estimated Acceptance:** 85% (KDD loves practical tools)

**Advantage:** Get paper published, build citations for NeurIPS 2027 resubmission

---

## Long-Term Strategy (Post-Submission)

### If NeurIPS Accepts (75% probability)
1. Present at NeurIPS 2026 (Dec 2026)
2. Release open source library
3. Write follow-up: FK-level active learning (standalone paper)
4. Write follow-up: Causal attribution via FK paths (PhD thesis material)

### If NeurIPS Rejects (25% probability)
1. Analyze reviews carefully
2. Address weaknesses
3. Submit to ICML 2027 (stronger paper with NeurIPS feedback)
4. Or submit to KDD 2027 (almost guaranteed acceptance)

### Either Way
- Release code as open source
- Blog post explaining the work
- Reach out to enterprises for case studies
- Build real-world impact story

---

**Timeline Summary:**

| Week | Phase | Tasks | Checkpoint |
|------|-------|-------|------------|
| 1-2 | P1 | SHAP baseline | SHAP results |
| 3-4 | P1 | Fix permutation + 2 domains | Method validated |
| 5-6 | P1 | KDD draft | KDD decision point ⚠️ |
| 7-10 | P2 | Active learning | Novelty boost |
| 11-12 | P2 | Classification | Breadth achieved |
| 13-14 | P3 | Scale validation | Robustness check |
| 15-16 | P3 | Conformal prediction | SOTA comparison |
| 17-18 | P4 | Writing | Draft complete |
| 19 | P4 | Figures | Camera-ready |
| 20 | P4 | Submit | 🚀 NeurIPS 2026 |

**START THIS WEEK:** SHAP baseline + permutation debugging

Ready to execute? I'll help you implement each phase.
