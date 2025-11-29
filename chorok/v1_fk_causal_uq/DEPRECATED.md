# DEPRECATED: FK-Causal-UQ Approach

**Date**: 2025-11-29
**Status**: NOT PURSUING

---

## What We Tried

**Hypothesis**: FK structure provides causal prior for uncertainty attribution.

**Method**: FK-Causal-UQ - use backdoor adjustment based on FK DAG to compute interventional uncertainty attribution.

**Claim**: This would be better than correlational methods (SHAP, LOO) because it's "causal."

---

## Why It Failed

### Empirical Test: FK-Causal-UQ vs Causal SHAP

We implemented Causal SHAP (Heskes et al., NeurIPS 2020) and compared on synthetic data with known ground truth:

| Method | ρ with Ground Truth | p-value |
|--------|---------------------|---------|
| Standard SHAP | 0.900 | 0.037 |
| **Causal SHAP** | **0.900** | **0.037** |
| FK-Causal-UQ (Ours) | 0.900 | 0.037 |
| True Interventional | 0.700 | 0.188 |

**Result: All permutation-based methods perform IDENTICALLY.**

### Root Cause Analysis

1. **Permutation is permutation** - Whether you "know" the DAG or not, shuffling feature values breaks the same correlations. The FK structure doesn't change what permutation does.

2. **Causal SHAP already exists** - Heskes et al. (2020) solved interventional feature attribution. Our method is a subset of their approach applied to uncertainty instead of predictions.

3. **No advantage from FK** - The FK structure tells us parent-child relationships, but:
   - Standard models already learn these correlations from data
   - Stratified permutation (our method) ≈ regular permutation in practice
   - The "backdoor adjustment" doesn't help when features are correlated

4. **Synthetic validation was misleading** - Our earlier ρ = 0.964 vs 0.741 comparison was between our method and LOO (retraining-based). When we compared to Causal SHAP (the right baseline), we matched exactly.

---

## What We Learned

### The Real Difference

| Comparison | Result | Meaning |
|------------|--------|---------|
| LOO vs Permutation | Different | Retraining ≠ Permutation |
| Causal SHAP vs Standard SHAP | Same | DAG knowledge doesn't help for permutation |
| FK-Causal-UQ vs Causal SHAP | Same | FK structure ≠ advantage |

### Key Insight

**The novel contribution we THOUGHT we had:**
- "FK structure provides causal advantage for UQ attribution"

**What's actually true:**
- FK structure provides NO advantage over Causal SHAP
- Permutation-based methods are equivalent regardless of DAG knowledge

---

## What's Still Valid

These findings from the FK-Causal-UQ work are still valid:

1. **Semi-synthetic validation (60% vs 0%)** - Our method detected shifted columns better than LOO, but this is shift detection, not attribution

2. **COVID case study** - CUSTOMERPAYMENTTERMS spike in Feb 2020 is a real observation

3. **Computational efficiency** - O(N × predict) vs O(|FK| × train) is real

4. **Theory** - The identification theorem is mathematically correct, it just doesn't provide practical advantage

---

## Code Archive

The following files are deprecated but preserved:

- `fk_causal_uq.py` - The method implementation
- `synthetic_causal_data.py` - Synthetic validation
- `compare_causal_shap.py` - The comparison that showed no advantage
- `THEORY.md` - Formal theory (correct but not useful)
- `RESEARCH_ROADMAP.md` - Original roadmap (archived)

---

## New Direction

See **[aggregation_uq/README.md](aggregation_uq/README.md)** for the new research direction:

**Aggregation-Aware Uncertainty Quantification**

This addresses a genuinely novel problem: ML models are miscalibrated for aggregation uncertainty in relational data.

---

## References

- Heskes, T., Sijben, E., Bucur, I. G., & Claassen, T. (2020). Causal Shapley Values: Exploiting Causal Knowledge to Explain Individual Predictions of Complex Models. NeurIPS 2020. https://arxiv.org/abs/2011.01625

---

*Last Updated: 2025-11-29*
