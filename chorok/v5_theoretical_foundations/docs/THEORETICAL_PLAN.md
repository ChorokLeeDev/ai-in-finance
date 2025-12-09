# Theoretical Plan: Deriving Intervention Effect Bounds

**Created**: 2025-12-09
**Goal**: Prove bounds on uncertainty reduction from data collection under FK hierarchy
**Target**: NeurIPS 2026 Main Conference (Probabilistic Methods / Theory)

---

## 1. The Goal

### What We Want to Prove

**Main Theorem (Informal)**:
> Collecting Δn samples for FK_k reduces posterior variance by at least f(n_k, Δn, K, σ),
> where the bound is **tighter** under hierarchical structure than independent estimation.

**Corollary (Practical)**:
> A closed-form ROI formula: plug in (current samples, planned samples, number of FKs) → guaranteed minimum uncertainty reduction.

---

## 2. Mathematical Setup

### 2.1 The Model

```
Level 0 (Hyperprior):
  σ²_FK ~ InverseGamma(a₀, b₀)     # Variance across FK importances

Level 1 (FK Importance):
  α_k | σ²_FK ~ N(0, σ²_FK)        # Importance of FK k, for k = 1, ..., K

Level 2 (Observations):
  y_ki | α_k ~ N(α_k, σ²_obs)      # Observed importance score i for FK k
                                    # i = 1, ..., n_k
```

**Why InverseGamma?** Conjugate with Normal likelihood → closed-form posteriors.

### 2.2 What We Observe

From permutation experiments, we get importance scores:
- y_k1, y_k2, ..., y_{k,n_k} for each FK k
- These are noisy measurements of the true importance α_k

### 2.3 What We Want to Estimate

- α_k: True importance of FK k
- Var(α_k | data): Our uncertainty about α_k

### 2.4 The Intervention Question

> If we collect Δn more samples for FK_k, how much does Var(α_k | data) decrease?

---

## 3. Known Results (Building Blocks)

### 3.1 Single FK, Known Variance (Textbook)

If α ~ N(0, τ²) and y_i | α ~ N(α, σ²) for i = 1, ..., n:

```
Posterior: α | y ~ N(μ_post, σ²_post)

where:
  μ_post = (τ²/(τ² + σ²/n)) · ȳ
  σ²_post = 1/(1/τ² + n/σ²) = τ²σ²/(τ²n + σ²)
```

### 3.2 Variance Reduction with More Samples

```
Var(α | n samples) = τ²σ² / (τ²n + σ²)
Var(α | n+Δn samples) = τ²σ² / (τ²(n+Δn) + σ²)

Ratio = Var(n+Δn) / Var(n) = (τ²n + σ²) / (τ²(n+Δn) + σ²)
```

### 3.3 Limiting Cases

**Case 1: Prior dominates (τ² << σ²/n)**
```
Var ≈ τ²  (constant, data doesn't help much)
```

**Case 2: Data dominates (τ² >> σ²/n)**
```
Var ≈ σ²/n  →  Var(n+Δn)/Var(n) ≈ n/(n+Δn)
```

### 3.4 Hierarchical Shrinkage (Efron-Morris)

With K groups sharing a prior:
```
Shrinkage factor: B_k = σ²_FK / (σ²_FK + σ²_obs/n_k)

Posterior mean: E[α_k | data] = B_k · ȳ_k + (1-B_k) · μ_global
Posterior var:  Var(α_k | data) ≈ B_k · σ²_obs/n_k
```

---

## 4. What's Novel (Our Contribution)

### 4.1 Gap in Existing Literature

| Existing | Missing |
|----------|---------|
| Posterior variance formula | **Bound on variance CHANGE** with Δn samples |
| Shrinkage factor for estimation | **Shrinkage benefit for intervention planning** |
| Hierarchical models in general | **FK-structured bounds specifically** |

### 4.2 Our Theorems to Prove

**Theorem 1 (Single FK Bound)**:
> For FK_k with n_k samples and prior variance τ², collecting Δn samples gives:
>
> Var(α_k | n_k + Δn) / Var(α_k | n_k) ≤ (n_k + σ²/τ²) / (n_k + Δn + σ²/τ²)

**Theorem 2 (Hierarchical Bound)**:
> Under hierarchical prior with K FKs and learned σ²_FK:
>
> Var(α_k | n_k + Δn)_hier ≤ Var(α_k | n_k + Δn)_indep × C(K, n_total)
>
> where C(K, n_total) < 1 quantifies the hierarchical benefit.

**Theorem 3 (ROI Formula)**:
> The minimum guaranteed uncertainty reduction from collecting Δn samples is:
>
> ΔVar/Var ≥ Δn / (n_k + Δn + σ²_obs/σ²_FK) × (1 + H(K))
>
> where H(K) > 0 is the hierarchical bonus depending on number of FKs.

---

## 5. Proof Strategy

### 5.1 Approach: Empirical Bayes + Conjugate Analysis

**Why this approach?**
- Full Bayesian with HalfNormal on σ_FK → intractable
- Empirical Bayes: estimate σ²_FK from data, then condition on it
- Conjugate (InverseGamma): closed-form posteriors

**Tradeoff**: Slightly less "pure" Bayesian, but tractable and practical.

### 5.2 Proof Outline for Theorem 1

1. Write posterior variance as function of n:
   ```
   V(n) = τ²σ² / (τ²n + σ²)
   ```

2. Compute ratio V(n+Δn) / V(n):
   ```
   R = (τ²n + σ²) / (τ²n + τ²Δn + σ²)
   ```

3. Show R ≤ n/(n+Δn) when τ² is large (data-dominated regime)

4. Derive the general bound including the prior term σ²/τ²

### 5.3 Proof Outline for Theorem 2

1. In hierarchical model, σ²_FK is estimated from all K FKs:
   ```
   σ̂²_FK = f(y_11,...,y_Kn_K)
   ```

2. The effective prior for α_k becomes:
   ```
   α_k | σ̂²_FK ~ N(0, σ̂²_FK)
   ```
   where σ̂²_FK is more accurate than single-FK estimate.

3. Show that Var(σ̂²_FK) decreases with K (more FKs = better estimate)

4. This tighter σ̂²_FK leads to tighter posterior variance on α_k

5. Quantify the improvement factor C(K, n_total)

### 5.4 Proof Outline for Theorem 3

1. Combine Theorems 1 and 2

2. Express in terms of observable quantities:
   - n_k: current samples for FK_k
   - Δn: planned additional samples
   - K: number of FKs
   - σ²_obs: observation noise (estimable from data)
   - σ²_FK: FK variance (estimable from data)

3. Derive closed-form lower bound on variance reduction

---

## 6. Technical Challenges

### 6.1 Challenge: σ²_FK is Random

**Problem**: In full Bayes, σ²_FK has a posterior distribution, not point value.

**Solution**:
- Use empirical Bayes (plug-in estimate)
- OR derive bounds that hold for all σ²_FK in high-probability region
- OR use PAC-Bayes framework

### 6.2 Challenge: Coupling Between FKs

**Problem**: α_1, ..., α_K are not independent given data (they share σ²_FK).

**Solution**:
- Condition on σ²_FK, then they ARE independent
- Marginalize later if needed

### 6.3 Challenge: Tightness of Bounds

**Problem**: Loose bounds are useless (e.g., "reduction ≥ 0%").

**Solution**:
- Validate bounds against empirical results
- F1: 84.5% reduction → bound should predict ≥ 70-80%
- SALT: 34.8% reduction → bound should predict ≥ 25-30%

---

## 7. Validation Plan

### 7.1 Theoretical Validation

- Check bounds reduce to known results in special cases
- Verify bounds are tight in limiting regimes
- Compare to Efron-Morris bounds (should be related)

### 7.2 Empirical Validation

Using our experimental results:

| Domain | n_k | Δn | Empirical Reduction | Bound Should Predict |
|--------|-----|----|--------------------|---------------------|
| F1 | 0 | 551 | 84.5% | ≥ 70% |
| SALT | 0 | 568 | 34.8% | ≥ 25% |

If bounds match, we have strong validation.
If bounds are too loose, we need tighter analysis.

### 7.3 Sensitivity Analysis

Test bounds across:
- Different K (number of FKs)
- Different n_k (current sample sizes)
- Different σ²_FK / σ²_obs ratios

---

## 8. Timeline

### Phase 1: Foundation (Week 1)
- [ ] Write formal model specification
- [ ] Derive single-FK posterior variance (known, verify)
- [ ] Compute variance ratio V(n+Δn)/V(n)

### Phase 2: Main Theorem (Week 2)
- [ ] Derive Theorem 1 (single FK bound)
- [ ] Set up hierarchical model with conjugate priors
- [ ] Derive σ²_FK posterior under InverseGamma prior

### Phase 3: Hierarchical Benefit (Week 3)
- [ ] Prove Theorem 2 (hierarchical improves bound)
- [ ] Quantify C(K, n_total) explicitly
- [ ] Derive the ROI formula (Theorem 3)

### Phase 4: Validation (Week 4)
- [ ] Validate bounds on F1 and SALT data
- [ ] Check tightness
- [ ] Revise if needed

### Phase 5: Paper Writing (Week 5-6)
- [ ] Write up theorems with full proofs
- [ ] Create figures showing bound vs empirical
- [ ] Position contribution relative to existing work

---

## 9. Success Criteria

### Minimum (Workshop-Level)
- [ ] Theorem 1 proven (single FK bound)
- [ ] Empirical validation on 2 domains
- [ ] Clear improvement over naive baseline

### Target (Main Conference)
- [ ] Theorem 2 proven (hierarchical benefit)
- [ ] Theorem 3: Closed-form ROI formula
- [ ] Bounds match empirical within 20%
- [ ] Novel contribution clearly articulated

### Stretch
- [ ] Extension to 3-level hierarchy (FK → Column → Value)
- [ ] Connection to active learning regret bounds
- [ ] PAC-Bayes style high-probability bounds

---

## 10. Key References to Study

### Must Read
1. Efron & Morris (1975) - "Data Analysis Using Stein's Estimator"
2. Gelman (2006) - "Prior distributions for variance parameters"
3. Berger (1985) - "Statistical Decision Theory and Bayesian Analysis" (Ch. 4)

### Should Read
4. Carlin & Louis (2000) - "Bayes and Empirical Bayes Methods"
5. Morris (1983) - "Parametric Empirical Bayes Inference"

### Nice to Have
6. PAC-Bayes literature (McAllester, Catoni)
7. Bayesian experimental design (Chaloner & Verdinelli)

---

## 11. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Bounds too loose | Medium | Start with tighter assumptions, relax later |
| Proofs too complex | Medium | Focus on simplest case first (conjugate) |
| Doesn't match empirical | Low | Our setup matches standard hierarchical Bayes |
| Not novel enough | Medium | Emphasize FK structure exploitation |

---

## 12. Next Immediate Step

**Start with Phase 1, Task 1**:

Write the formal model specification with exact distributions, parameters, and notation. This is the foundation everything else builds on.

---

*This plan will be updated as we progress through the derivation.*
