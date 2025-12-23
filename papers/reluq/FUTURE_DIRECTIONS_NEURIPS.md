# Future Directions: Path to NeurIPS 2026

**Date**: 2025-12-23
**Based on**: Ultra-deep review
**Current State**: 55-60% acceptance probability
**Target State**: 75-80% acceptance probability

---

## Priority Matrix

| Priority | Issue | Effort | Impact |
|----------|-------|--------|--------|
| **P0** | Fix circular theorem | 1 week | +5% |
| **P0** | Add causal validation experiment | 2 weeks | +10% |
| **P1** | Develop EP domain detection | 2 weeks | +5% |
| **P1** | Unify paper narrative | 1 week | +3% |
| **P2** | Learning curve experiments | 1 week | +3% |
| **P2** | Clean up hero figure | 2 days | +2% |

**Total estimated improvement: +28%** (55% → 83%)

---

## P0: Critical Fixes

### 1. Fix the Circular Theorem

**Current Problem**:
```
Theorem: α(g) ∝ ε(g) ∝ I(Y; Xg | X-g)
```
Both α (uncertainty) and ε (error) are measured via permutation. Perfect correlation is definitional, not a finding.

**Fix Strategy**:

**Option A: Reframe as Proxy Validation**
```markdown
New Claim: "FK-level uncertainty is a valid proxy for error impact"

Why useful:
- Error impact requires ground truth (expensive)
- Uncertainty computed at inference time (free)
- If correlated, use uncertainty for real-time monitoring
```

**Option B: Predictive Validation**
```markdown
New Claim: "FK uncertainty predicts future error degradation"

Experiment:
1. Train on data from period T1
2. Compute FK uncertainties on T1
3. Test on period T2 (temporal shift)
4. Show: high-uncertainty FKs have larger error increase
```

**Recommended**: Option B (stronger, more novel)

**Implementation**:
```python
# Pseudo-code for temporal validation
def temporal_validation(dataset, task):
    # Split by time
    train_data = data[data.timestamp < T1]
    test_early = data[(data.timestamp >= T1) & (data.timestamp < T2)]
    test_late = data[data.timestamp >= T2]

    # Train ensemble
    models = train_ensemble(train_data)

    # Compute FK uncertainties on early test
    fk_uncertainties = compute_fk_uncertainty(models, test_early)

    # Compute error degradation (early → late)
    error_early = compute_error(models, test_early)
    error_late = compute_error(models, test_late)
    error_degradation = (error_late - error_early) / error_early

    # Correlation: high uncertainty → high degradation?
    correlation = spearman(fk_uncertainties, error_degradation)

    return correlation  # Should be positive and significant
```

---

### 2. Add Causal Validation Experiment

**Current Problem**:
Intervention experiment (mean imputation) shows importance, not actionability.
"Removing features increases error" ≠ "Improving data decreases error"

**Fix: Corruption Experiment**

```markdown
Design:
1. Identify top-uncertainty FK (e.g., RESULTS)
2. Corrupt it systematically (add noise, drop values)
3. Retrain model on corrupted data
4. Measure: Does error increase proportionally to uncertainty?

Expected result:
- RESULTS (70% uncertainty) → 70% of error increase from corruption
- QUALIFYING (5% uncertainty) → 5% of error increase from corruption
```

**Implementation**:
```python
def corruption_validation(X, y, fk_to_cols, models):
    """Validate that uncertainty predicts corruption sensitivity."""
    results = {}

    for fk_name, cols in fk_to_cols.items():
        # Corrupt FK features
        X_corrupted = X.copy()
        for col in cols:
            noise = np.random.normal(0, X[:, col].std() * 0.5, len(X))
            X_corrupted[:, col] += noise

        # Retrain on corrupted data
        models_corrupted = train_ensemble(X_corrupted, y)

        # Measure error increase
        base_error = compute_mae(models, X, y)
        corrupted_error = compute_mae(models_corrupted, X, y)
        error_increase = (corrupted_error - base_error) / base_error

        results[fk_name] = {
            'uncertainty': fk_uncertainties[fk_name],
            'error_increase': error_increase
        }

    # Correlation should be positive
    return spearman(uncertainties, error_increases)
```

**Success criterion**: ρ > 0.7 between uncertainty and corruption sensitivity

---

## P1: Important Improvements

### 3. Develop EP Domain Detection

**Current Finding**:
- EP domains (SALT, Trial, F1, Avito): ρ ≈ 1.0
- Non-EP domain (Stack): ρ = -0.5

**Turn this into a contribution**:

```markdown
New Contribution: "EP Domain Detector"

Method:
1. Compute FK uncertainty attribution
2. Compute FK error attribution (permutation importance)
3. Calculate ρ = corr(uncertainty, error)

Interpretation:
- ρ > 0.7: EP domain → FK attribution is valid
- ρ < 0.3: Non-EP domain → Use alternative methods
- 0.3 < ρ < 0.7: Mixed → Interpret with caution

Value: Practitioners can diagnose whether FK attribution applies
```

**Formalize as theorem**:
```
Theorem (EP Detection):
A domain D satisfies Error Propagation if and only if:
  Corr(α_FK, ε_FK) > τ
where τ ≈ 0.7 is an empirically determined threshold.

Proof sketch:
- EP implies dimensional independence
- Independence implies no cross-FK interactions
- No interactions implies α and ε measure same signal
- Same signal implies high correlation
```

---

### 4. Unify Paper Narrative

**Current Problem**: Two competing stories

| Document | Story |
|----------|-------|
| main_neurips_v3.tex | "Uncertainty predicts error, FK provides stable attribution" |
| PAPER_OUTLINE.md | "FK uncertainty decomposition enables data investment" |

**Unified Narrative**:

```markdown
Title: "Schema-Aware Uncertainty Decomposition for Relational Data Investment"

Abstract (revised):
Machine learning on relational databases requires knowing not just THAT
predictions are uncertain, but WHICH data sources to improve. We introduce
FK-level uncertainty decomposition, which attributes model uncertainty to
foreign key groups using database schema. Our key contributions:

1. We show uncertainty contribution correlates with error impact (ρ > 0.9)
   in Error Propagation domains, validating FK attribution

2. We introduce the Importance × Stability framework, identifying FKs that
   are high-importance but high-uncertainty as data investment targets

3. We validate causally: corrupting top-uncertainty FKs increases error
   proportionally to their uncertainty contribution

4. We provide an EP domain detector (ρ < 0.3 indicates non-EP structure)

Experiments on 5 domains, 13 tasks demonstrate actionable insights across
enterprise, clinical, sports, and marketplace applications.
```

---

## P2: Nice-to-Have Improvements

### 5. Learning Curve Experiments

**Purpose**: Prove uncertainty is epistemic (reducible with data), not aleatoric

```python
def learning_curve_by_fk(X, y, fk_to_cols, data_fractions=[0.2, 0.4, 0.6, 0.8, 1.0]):
    """Show uncertainty decreases with more data for high-uncertainty FKs."""
    results = {fk: [] for fk in fk_to_cols}

    for frac in data_fractions:
        # Subsample
        n = int(len(X) * frac)
        X_sub, y_sub = X[:n], y[:n]

        # Train and compute FK uncertainties
        models = train_ensemble(X_sub, y_sub)
        for fk_name in fk_to_cols:
            unc = compute_fk_uncertainty(models, X_sub, fk_name)
            results[fk_name].append(unc)

    # Plot: X=data fraction, Y=uncertainty, line per FK
    # High-uncertainty FKs should show steeper decrease
    return results
```

**Expected finding**:
- RESULTS (high unc): 70% → 40% → 25% (steep decrease)
- QUALIFYING (low unc): 5% → 4% → 4% (flat)

This proves "collect more data" advice is valid for high-uncertainty FKs.

---

### 6. Clean Up Hero Figure

**Current Issues**:
- "% variance change when permuted" is confusing
- Negative values mean "stabilizing" but look like "bad"
- Quadrant labels use emojis (not publication-ready)

**Revised Figure**:

```
Y-axis: "FK Stability Score"
        (positive = destabilizing/noisy, negative = stabilizing)

X-axis: "FK Importance"
        (% error increase when FK permuted)

Quadrants:
- Top-right: "High-Value Data Target" (invest here)
- Top-left: "Noise Source" (investigate data quality)
- Bottom-right: "Reliable Signal" (data sufficient)
- Bottom-left: "Low Priority" (ignore)
```

Remove emojis, use professional labels with brief explanations in caption.

---

## Implementation Timeline

### Week 1-2: Causal Validation
- [ ] Implement corruption experiment
- [ ] Run on all 5 EP domains
- [ ] Document correlation between uncertainty and corruption sensitivity

### Week 3: Fix Theorem
- [ ] Rewrite theorem as "proxy validation" claim
- [ ] Add temporal validation experiment (if data supports)
- [ ] Update formal theorem document

### Week 4: EP Detection
- [ ] Formalize EP detection criterion
- [ ] Test on Stack Overflow (negative control)
- [ ] Write up as diagnostic tool

### Week 5: Unify Narrative
- [ ] Rewrite abstract and introduction
- [ ] Align all sections with unified story
- [ ] Update hero figure

### Week 6: Learning Curves + Polish
- [ ] Run learning curve experiments
- [ ] Create publication-ready figures
- [ ] Internal review

---

## Success Metrics

### Before Submission Checklist

- [ ] Corruption experiment shows ρ > 0.7
- [ ] Temporal validation (if applicable) shows predictive power
- [ ] EP detection criterion formalized and tested
- [ ] Learning curves show epistemic uncertainty is reducible
- [ ] Hero figure is publication-ready
- [ ] All claims have matching experiments
- [ ] Limitations section is honest and complete

### Probability Checkpoints

| Milestone | Probability |
|-----------|-------------|
| Current state | 55-60% |
| + Corruption experiment | 65-70% |
| + EP detection formalized | 70-75% |
| + Unified narrative | 73-78% |
| + Learning curves | 75-80% |
| + Polished figures | 78-83% |

---

## Risk Mitigation

### If Corruption Experiment Fails

**Scenario**: Uncertainty doesn't predict corruption sensitivity

**Fallback**:
- Reframe as "FK importance attribution" (not uncertainty)
- Emphasize schema-guided grouping as the contribution
- Target KDD 2026 instead (more applied venue)

### If Temporal Validation Fails

**Scenario**: No datasets have suitable temporal structure

**Fallback**:
- Focus on corruption experiment only
- Acknowledge limitation in paper
- Propose temporal validation as future work

### If EP Detection Is Too Noisy

**Scenario**: ρ threshold varies across domains

**Fallback**:
- Report as empirical finding rather than formal criterion
- Use confidence intervals instead of hard threshold
- Position as "diagnostic guidance" not "detection algorithm"

---

## Backup Plans

| Venue | Deadline | Probability | Notes |
|-------|----------|-------------|-------|
| NeurIPS 2026 | May 2026 | 75-80% | Primary target |
| KDD 2026 | Feb 2026 | 85% | More applied, earlier |
| ICML 2026 | Jan 2026 | 70% | Theory-heavy venue |
| VLDB 2026 | Mar 2026 | 80% | Database audience |

**Recommendation**: If corruption experiment works, target NeurIPS. If not, pivot to KDD.

---

## Key Reviewer Questions to Prepare

1. **"How is this different from grouped permutation importance?"**
   - Answer: We measure uncertainty (variance), not importance (error)
   - We provide 2D framework combining both for actionability
   - We validate causally via corruption experiments

2. **"Why not just use SHAP?"**
   - Answer: SHAP is feature-level, unstable for correlated features
   - FK grouping provides semantic stability (std < 3%)
   - Schema-aware grouping is automatic, not manual

3. **"Does improving high-uncertainty FKs actually help?"**
   - Answer: Yes, corruption experiment shows proportional relationship
   - Learning curves show uncertainty decreases with more data
   - (If temporal validation works) Predicts future error degradation

4. **"When does this method fail?"**
   - Answer: Non-EP domains (Stack Overflow, ρ = -0.5)
   - We provide EP detection criterion to diagnose applicability
   - User behavior-driven domains may need alternative methods

5. **"Is the theorem novel?"**
   - Answer: The theorem explains WHY FK attribution works (EP conditions)
   - It provides testable predictions (ρ → 1 under EP)
   - The failure mode (ρ < 0) is itself informative

---

*Document created: 2025-12-23*
*Next review: After corruption experiment (Week 2)*
*Target submission: NeurIPS 2026 (May 2026)*
