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

### Priority 1: Rigorous Comparison with Causal SHAP
**Why**: If we claim to be different, we must SHOW it empirically
**How**:
- [ ] Implement actual Causal SHAP (use `causal-shap` package or implement)
- [ ] Run on same synthetic data
- [ ] Compare: Does Causal SHAP also get ρ ≈ 0.96?
- [ ] If yes → our novelty is weaker (just applied existing method)
- [ ] If no → genuine contribution (FK structure provides advantage)

### Priority 2: Semi-Synthetic Validation
**Why**: Synthetic data is too controlled, real data has no ground truth
**How**:
- [ ] Take real SALT FK structure
- [ ] Inject KNOWN distribution shifts into specific FKs
- [ ] Test: Does our method correctly identify the shifted FK?
- [ ] This provides "ground truth" on real data structure

### Priority 3: Formal Proof of Identification Theorem
**Why**: A stated theorem without proof is not a contribution
**How**:
- [ ] Write formal proof using do-calculus rules
- [ ] Identify EXACTLY which assumptions are needed
- [ ] Show when identification FAILS (negative results matter)

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

## Bottom Line

**Current state**: Promising direction with preliminary results
**Not ready for**: Top venue (NeurIPS/ICML main track)
**Might be ready for**: Workshop paper, arXiv preprint

**To reach top venue**:
1. Must compare with Causal SHAP empirically
2. Must validate on real/semi-synthetic data
3. Must prove identification theorem
4. Coverage guarantees would strengthen

---

*Last Updated*: 2025-11-29
