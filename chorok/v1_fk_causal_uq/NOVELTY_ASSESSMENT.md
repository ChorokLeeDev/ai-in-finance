# Novelty Assessment: Honest Evaluation

**Date**: 2025-11-29

---

## What We Have (Current State)

### Claimed Contribution
"First to combine FK structure with interventional UQ attribution"

### Evidence
- Synthetic validation: ρ = 0.964 (interventional) vs ρ = 0.741 (LOO)
- Theory: Identification theorem in THEORY.md
- Algorithm: fk_causal_uq.py (no retraining)

---

## Honest Assessment: What's NOT Novel

| Component | Prior Work | Our Addition |
|-----------|------------|--------------|
| Interventional attribution | Causal SHAP (2024), Causal Shapley (NeurIPS 2020) | Applied to UQ, not predictions |
| Relational causal discovery | RelFCI (2024), RelPC, Suna (VLDB 2024) | Use FK as given prior (no discovery) |
| UQ attribution | VFA (ensemble variance) | Causal, not correlational |
| Do-calculus | Pearl (1995, 2009) | Standard application |

### Critical Distinction: Causal Shapley vs Our Method

**Causal Shapley (Heskes et al., NeurIPS 2020):**
- Explains **predictions** f(x)
- Question: "Why did the model predict Y=3?"
- Uses causal graph for conditional sampling
- Paper: https://arxiv.org/abs/2011.01625

**Our FK-Causal-UQ:**
- Explains **uncertainty** U(f(x))
- Question: "Why is the model uncertain about its prediction?"
- Uses FK structure as causal prior
- No existing work does this combination

**This IS a genuine novelty**: Causal attribution for UQ (not just prediction) using FK structure.

### Weaknesses
1. **Synthetic validation only** - We generated the data, could be biased
2. **No comparison with actual Causal SHAP** - Just claimed difference
3. **Identification theorem not formally proved** - Just stated assumptions
4. **No coverage guarantees** - Unlike conformal methods

---

## How to Strengthen Novelty

### Priority 1: Rigorous Comparison with Causal SHAP ✅ DONE - NEGATIVE RESULT
**Why**: If we claim to be different, we must SHOW it empirically
**Results** (2025-11-29):

| Method | Correlation with Ground Truth |
|--------|------------------------------|
| Standard SHAP | ρ = 0.900, p = 0.037 ** |
| **Causal SHAP** | ρ = 0.900, p = 0.037 ** |
| FK-Causal-UQ (Ours) | ρ = 0.600, p = 0.285 |
| True Interventional | ρ = 0.700, p = 0.188 |

**CRITICAL FINDING**: Causal SHAP performs BETTER than our FK-Causal-UQ method.
- Standard SHAP and Causal SHAP: ρ = 0.900 (significant)
- Our method: ρ = 0.600 (not significant)
- Δρ = -0.300 (we are worse)

**Verdict**: NO NOVELTY - Causal SHAP already solves this problem
- [ ] ~~Implement actual Causal SHAP~~ → Done (compare_causal_shap.py)
- [x] Run on same synthetic data → Yes
- [x] Compare: Does Causal SHAP also get ρ ≈ 0.96? → YES, better than us
- [x] If yes → our novelty is weaker → **CONFIRMED: WEAK/NO NOVELTY**

### Priority 2: Semi-Synthetic Validation ✅ DONE
**Why**: Synthetic data is too controlled, real data has no ground truth
**Results** (5 columns tested with permutation shift):

| Shifted Column | LOO | Causal |
|----------------|-----|--------|
| CUSTOMERPAYMENTTERMS | ✗ | ✗ |
| SALESORGANIZATION | ✗ | **✓** |
| TRANSACTIONCURRENCY | ✗ | **✓** |
| DISTRIBUTIONCHANNEL | ✗ | ✗ |
| HEADERINCOTERMSCLASSIFICATION | ✗ | **✓** |

**LOO Accuracy: 0/5 (0%) | Causal Accuracy: 3/5 (60%)**

- [x] Take real SALT FK structure (sales-group task)
- [x] Inject KNOWN distribution shifts into specific columns
- [x] Test: Does our method correctly identify the shifted column?
- [x] **Result**: Causal method 60% accurate, LOO method 0% accurate!

### Priority 3: Formal Proof of Identification Theorem ✅ DONE
**Why**: A stated theorem without proof is not a contribution
**How**:
- [x] Write formal proof using do-calculus rules (THEORY.md §4.4)
- [x] Identify EXACTLY which assumptions are needed (Causal Sufficiency, FK=Causal, Acyclicity)
- [x] Show when identification FAILS (THEORY.md §4.3 - confounders, reversed FK, cycles)

### Priority 4: Add Conformal Wrapper
**Why**: Coverage guarantees are a concrete, verifiable contribution
**How**:
- [ ] Wrap FK-Causal-UQ in conformal prediction
- [ ] Provide intervals with guaranteed coverage (e.g., 90%)
- [ ] This is tangible and easy to verify

### Priority 5: Computational Complexity Analysis
**Why**: "No retraining" claim needs formal backing
**How**:
- [ ] Prove: LOO is O(|FK| × T_train), ours is O(N × T_predict)
- [ ] Show actual runtime comparison
- [ ] For large models, this is significant

---

## Revised Novelty Claims (Honest Version)

### Strong Claims (Can Defend)
1. **FK as causal prior eliminates discovery** - RelFCI/RelPC need to discover DAG, we use given FK structure
2. **Applied to UQ, not just prediction** - Causal SHAP explains predictions, we explain uncertainty
3. **No retraining required** - LOO retrains |FK| times, we don't

### Weak Claims (Need More Evidence)
1. **Better than Causal SHAP** - Must implement and compare
2. **Works on real data** - Only synthetic so far
3. **Identification theorem** - Need formal proof

### NOT Claims (Don't Overclaim)
1. ~~First interventional attribution~~ (Causal SHAP exists)
2. ~~First relational causal method~~ (RelFCI exists)
3. ~~Novel do-calculus application~~ (standard)

---

## Concrete Next Steps

```
Week 1: Causal SHAP Comparison
- Install causal-shap or implement
- Run on synthetic data
- Document: Are we actually better?

Week 2: Semi-Synthetic Validation
- Inject shifts into SALT FK structure
- Test detection accuracy
- Compare with baselines

Week 3: Formalize Theory
- Write proof of Theorem 1
- Document failure cases
- Identify minimal assumptions

Week 4: Conformal Wrapper
- Implement coverage guarantees
- Validate empirically
- This becomes a concrete contribution
```

---

## What Makes a Paper Publishable

| Criterion | Current Status | Needed |
|-----------|---------------|--------|
| Novel method | Partial (combination is new) | Compare with Causal SHAP |
| Theoretical contribution | Stated, not proved | Formal proof |
| Empirical validation | Synthetic only | Semi-synthetic + real |
| Reproducibility | Code exists | Package + documentation |
| Practical value | Claimed | Real case study |

---

## Bottom Line (Updated 2025-11-29)

### Causal SHAP Comparison Results

After rigorous comparison with Causal SHAP (Heskes 2020):

| Method | ρ with Ground Truth |
|--------|---------------------|
| Standard SHAP | 0.900 |
| Causal SHAP | 0.900 |
| **FK-Causal-UQ (Ours)** | **0.900** |
| True Interventional | 0.700 |

**Finding**: All permutation-based methods perform identically. FK structure provides **NO ADVANTAGE** for static attribution.

### What FK Structure Actually Provides

1. **NOT useful for**: Static uncertainty attribution (same as standard SHAP)
2. **POTENTIALLY useful for**: Distribution shift detection (semi-synthetic: 60% vs 0%)
3. **Useful for**: Computational efficiency (no retraining)

### Remaining Valid Contributions

1. **Semi-synthetic validation**: 60% vs 0% for shift detection (still valid)
2. **COVID case study**: CUSTOMERPAYMENTTERMS +2.11 in Feb 2020 (observational)
3. **Computational**: O(N × predict) vs O(|FK| × train)
4. **Theory**: Identification theorem (applies, but doesn't differentiate from Causal SHAP)

### Revised Assessment

**Current state**: The core hypothesis (FK structure helps) is NOT supported for attribution
**Valid contribution**: Distribution shift detection using FK-aware methods
**Not valid**: Claims of novelty over Causal SHAP for static attribution

**Honest verdict**: This is NOT a top venue paper. Could be a workshop paper focused specifically on:
- "FK-aware distribution shift detection in relational data"
- NOT "Causal uncertainty attribution" (already solved by Causal SHAP)

---

*Last Updated*: 2025-11-29 (after Causal SHAP comparison)
