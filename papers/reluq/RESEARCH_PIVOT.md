# Research Pivot: FK Uncertainty Attribution

**Date**: 2025-12-23
**Status**: Core hypothesis FALSIFIED - pivoting to new directions
**Decision**: Document learnings, explore new angles

---

## Part 1: What We Tried and Why We're Abandoning It

### Original Hypothesis

> "FK-level uncertainty decomposition identifies which foreign key relationships
> contribute to model uncertainty, enabling targeted data investment decisions."

### The Core Claim

If FK_A has high uncertainty contribution, then:
1. Improving FK_A's data quality should reduce model error
2. FK_A is a good target for data investment

### Experiments Run

| Experiment | Goal | Result | Verdict |
|------------|------|--------|---------|
| **Corruption** | Show uncertainty predicts corruption sensitivity | ρ = -0.04 (p=0.82) | ❌ FAILED |
| **EP Detection** | Show uncertainty correlates with error | ρ = 0.09 (p=0.63) | ❌ FAILED |
| **Learning Curve** | Show uncertainty is epistemic | 4/6 passed | ✅ Partial |
| **Importance Validation** | Show error predicts corruption | ρ = 0.60 (p<0.001) | ✅ PASSED |

### Why It Failed

**The fundamental problem**: Ensemble variance (uncertainty) measures something DIFFERENT from permutation importance (error impact).

```
Uncertainty contribution:  "How much does ensemble disagree when FK is broken?"
Error attribution:         "How much does error increase when FK is broken?"

These are NOT the same thing.
```

**Empirical evidence**:
- Aggregate correlation: ρ = -0.04 (essentially zero)
- Only 2/6 domains showed any correlation
- The domains that "worked" (rel-trial, rel-event) may be spurious

### What We Learned

1. **Ensemble variance ≠ feature importance** in relational data
2. **Schema-guided grouping IS stable** (std < 3% across seeds)
3. **EP property is domain-specific** - not a universal law
4. **Error attribution DOES work** (ρ = 0.60) - but it's not novel

### Why We're Abandoning This Direction

| Reason | Details |
|--------|---------|
| **Core hypothesis falsified** | ρ = -0.04 between uncertainty and corruption |
| **Fallback not novel** | "FK importance" is standard practice |
| **Insufficient remaining contribution** | EP detection alone is too small |
| **Opportunity cost** | Other directions may be more promising |

---

## Part 2: What We're Keeping

### Validated Findings (Can be reused)

1. **Schema-guided grouping provides stability**
   - Feature-level: high variance across seeds
   - FK-level: std < 3%
   - This can be a minor contribution in another paper

2. **EP detection criterion**
   - Compute ρ(uncertainty, error)
   - ρ > 0.7 → EP domain
   - ρ < 0.3 → non-EP domain
   - Useful diagnostic tool

3. **Task-specific FK behavior**
   - Same FK can be important for one task, irrelevant for another
   - Example: EVENT_ATTENDEES (-69% for attendance, +27% for repeat)
   - Interesting observation for future work

4. **Experimental infrastructure**
   - `causal_validation_suite.py` - corruption experiments
   - `aggregate_validation_analysis.py` - multi-domain analysis
   - Can be reused for other hypotheses

### Code Assets

```
papers/reluq/experiments/
├── causal_validation_suite.py      # Reusable validation framework
├── aggregate_validation_analysis.py # Multi-domain aggregation
├── fk_active_learning.py           # FK-guided acquisition
├── create_2d_visualization.py      # Importance × stability plots
└── validation_results/             # All experimental data
```

---

## Part 3: New Angles to Explore

### Angle 1: FK-Level Conformal Prediction

**Hypothesis**: Conformal prediction intervals vary by FK group, enabling targeted uncertainty quantification.

**Method**:
1. Train conformal predictor on relational data
2. Compute prediction intervals
3. Decompose interval width by FK contribution
4. Validate: Do wide intervals correlate with actual errors?

**Novelty**: No one has applied conformal prediction with schema-aware decomposition

**Feasibility**: Medium (conformal prediction is well-understood)

**Key experiment**:
```python
# Compute FK-level conformal intervals
for fk in fk_groups:
    intervals_with_fk = conformal_predict(X)
    intervals_without_fk = conformal_predict(X_permuted_fk)
    fk_contribution = intervals_with_fk - intervals_without_fk
```

**Risk**: May fail for same reason as uncertainty (measuring different things)

---

### Angle 2: Causal FK Discovery

**Hypothesis**: FK relationships encode causal structure; we can discover which FKs have causal vs correlational relationships to the target.

**Method**:
1. Use causal discovery algorithms (PC, FCI) on FK-aggregated features
2. Compare discovered DAG with schema
3. Identify FKs with direct causal paths to target

**Novelty**: Connecting relational schema with causal discovery

**Feasibility**: Hard (causal discovery is noisy, requires assumptions)

**Key experiment**:
```python
# Discover causal structure
dag = pc_algorithm(X_by_fk)
# Compare with schema-implied structure
causal_fks = [fk for fk in schema_fks if dag.has_path(fk, target)]
```

**Risk**: Causal discovery may not work on observational relational data

---

### Angle 3: Temporal FK Uncertainty Dynamics

**Hypothesis**: FK uncertainty contribution changes over time, enabling early detection of distribution shift.

**Method**:
1. Compute FK uncertainty at time T1
2. Compute FK uncertainty at time T2 (after shift)
3. Measure: Which FKs show largest uncertainty increase?
4. Validate: Do these FKs have highest error increase?

**Novelty**: Temporal analysis of FK-level uncertainty

**Feasibility**: Medium (requires temporal data, rel-salt has COVID shift)

**Key experiment**:
```python
# Pre-shift
fk_unc_t1 = compute_fk_uncertainty(model, X_t1)
# Post-shift
fk_unc_t2 = compute_fk_uncertainty(model, X_t2)
# Change
delta_unc = {fk: fk_unc_t2[fk] - fk_unc_t1[fk] for fk in fks}
# Correlate with error change
```

**Risk**: May require uncertainty to work (which we just showed doesn't)

---

### Angle 4: FK-Aware Data Valuation

**Hypothesis**: Data points with rare FK values are more valuable for training.

**Method**:
1. Compute FK value frequency (how common is raceId=123?)
2. Train with/without rare FK values
3. Measure: Does removing rare FK values hurt more?
4. Use for data valuation/pricing

**Novelty**: Connecting data valuation with relational structure

**Feasibility**: Medium (data valuation is active area)

**Key experiment**:
```python
# Identify rare FK values
rare_fk_values = {fk: values with count < threshold}
# Leave-one-out on rare vs common
value_rare = loo_influence(rare_samples)
value_common = loo_influence(common_samples)
# Hypothesis: value_rare >> value_common
```

**Risk**: May just rediscover that rare data is valuable (known)

---

### Angle 5: Negative Result Paper

**Hypothesis**: None - this is a descriptive/warning paper

**Contribution**:
1. Document that FK uncertainty ≠ FK importance
2. Provide EP detection criterion as diagnostic
3. Warn practitioners not to use uncertainty for data investment
4. Release benchmark for future research

**Venue**: Workshop paper, TMLR (accepts negative results), or blog post

**Feasibility**: High (already have all the data)

**Value**: Prevents others from going down same dead end

---

### Angle 6: Focus on Causal Regimes Paper

**Status**: You have a separate paper in `papers/causal_regimes/`

**Hypothesis**: Causal structure changes across market regimes; detecting these changes improves trading

**Novelty**: Regime-aware causal discovery for finance

**Recommendation**: This may be more promising than continuing FK uncertainty

---

## Part 4: Recommended Next Steps

### Immediate (This Week)

1. ✅ Document what we tried (this file)
2. ✅ Commit all experimental code and results
3. ⬜ Brief advisor on findings (1-pager)

### Short-term (Next 2 Weeks)

| Priority | Action | Effort |
|----------|--------|--------|
| 1 | Evaluate Angle 1 (conformal) with quick experiment | 2 days |
| 2 | Evaluate Angle 3 (temporal) on rel-salt COVID data | 2 days |
| 3 | Decide: pivot or write negative result paper | 1 day |

### Medium-term (Next Month)

- If new angle works: pursue for NeurIPS 2026
- If no angle works: write workshop paper on negative result
- Consider focusing on causal regimes paper instead

---

## Part 5: Key Files Reference

### Experimental Results
```
papers/reluq/experiments/validation_results/
├── aggregate_analysis.json          # All correlations
├── aggregate_validation.png         # Key figure
├── *_validation.json                # Per-domain results
```

### Documentation
```
papers/reluq/
├── RESEARCH_PIVOT.md               # This file
├── FUTURE_DIRECTIONS_NEURIPS.md    # Original plan (now outdated)
├── experiments/VALIDATION_RESULTS_SUMMARY.md  # Detailed results
```

### Code
```
papers/reluq/experiments/
├── causal_validation_suite.py      # Main validation code
├── aggregate_validation_analysis.py # Analysis code
```

---

## Part 6: Lessons Learned

### What Went Right

1. **Rigorous validation before writing** - Saved us from publishing false claims
2. **Multi-domain testing** - Revealed the failure wasn't domain-specific
3. **Aggregate analysis** - 31 data points more reliable than per-domain
4. **Honest assessment** - Didn't rationalize failed results

### What Went Wrong

1. **Late validation** - Should have run corruption experiment earlier
2. **Assumed correlation** - Assumed uncertainty ≈ importance without testing
3. **Over-reliance on theory** - EP theorem was circular/tautological

### For Future Research

1. **Validate core hypothesis FIRST** - Before building on it
2. **Run causal experiments early** - Correlation ≠ causation
3. **Test on multiple domains** - Single domain results are unreliable
4. **Be skeptical of perfect correlations** - ρ = 1.000 was a red flag

---

*Document created: 2025-12-23*
*Status: Pivoting to new directions*
*Next review: After quick experiments on Angles 1 and 3*
