# Formal Model Specification & Derivations

**Created**: 2025-12-09
**Status**: Phase 1 Complete

---

## 1. Model Specification

### 1.1 Setup

We have K foreign keys. For each FK k, we want to estimate its "importance" α_k - how much it contributes to prediction uncertainty.

**Observations**: From permutation experiments, we get noisy importance scores:
- Run n_k permutations for FK k
- Each permutation gives a score y_{k,i}

### 1.2 Generative Model

```
┌─────────────────────────────────────────────────────────────┐
│  HIERARCHICAL BAYESIAN MODEL FOR FK IMPORTANCE              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Level 0 (Hyperprior):                                      │
│      σ²_FK ~ InverseGamma(a₀, b₀)                          │
│                                                             │
│  Level 1 (FK Importance):                                   │
│      α_k | σ²_FK ~ N(0, σ²_FK)    for k = 1, ..., K        │
│                                                             │
│  Level 2 (Observations):                                    │
│      y_{k,i} | α_k ~ N(α_k, σ²_obs)   for i = 1, ..., n_k  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 Notation

| Symbol | Meaning |
|--------|---------|
| K | Number of FKs |
| α_k | True importance of FK k |
| n_k | Number of observations for FK k |
| N = Σ_k n_k | Total observations |
| y_{k,i} | i-th observation for FK k |
| ȳ_k = (1/n_k) Σ_i y_{k,i} | Sample mean for FK k |
| σ²_FK | Prior variance of FK importances |
| σ²_obs | Observation noise variance |
| τ² = σ²_FK | (Alternative notation for prior variance) |
| σ² = σ²_obs | (Alternative notation for observation variance) |
| λ = σ²_obs / σ²_FK = σ²/τ² | Noise-to-signal ratio |

---

## 2. Single FK Posterior (Known σ²_FK)

### 2.1 Derivation

**Prior**:
```
α_k ~ N(0, τ²)
```

**Likelihood** (for n_k observations):
```
p(y_k | α_k) = Π_i N(y_{k,i} | α_k, σ²)
             ∝ exp(-n_k/(2σ²) · (ȳ_k - α_k)²)
```

The sufficient statistic is ȳ_k with effective likelihood:
```
ȳ_k | α_k ~ N(α_k, σ²/n_k)
```

**Posterior** (Normal-Normal conjugate):
```
α_k | y_k, σ²_FK ~ N(μ_k, V_k)
```

### 2.2 Posterior Parameters

**Precision adds**:
```
1/V_k = 1/τ² + n_k/σ²
```

**Posterior variance**:
```
V_k = 1 / (1/τ² + n_k/σ²)
    = τ²σ² / (σ² + n_k·τ²)
    = τ²σ² / (σ² + n_k·τ²)
```

**Reparametrized with λ = σ²/τ²**:
```
V_k = τ² · λ / (λ + n_k)
    = σ² / (λ + n_k)
```

**Posterior mean**:
```
μ_k = V_k · (n_k·ȳ_k/σ²)
    = [τ²σ² / (σ² + n_k·τ²)] · [n_k·ȳ_k/σ²]
    = n_k·τ² / (σ² + n_k·τ²) · ȳ_k
    = B_k · ȳ_k
```

**Shrinkage factor**:
```
B_k = n_k·τ² / (σ² + n_k·τ²)
    = n_k / (n_k + λ)
    = 1 / (1 + λ/n_k)
```

### 2.3 Summary Box

```
┌─────────────────────────────────────────────────────────────┐
│  SINGLE FK POSTERIOR                                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  α_k | y_k, σ²_FK ~ N(μ_k, V_k)                            │
│                                                             │
│  where:                                                     │
│      B_k = n_k / (n_k + λ)           [shrinkage factor]    │
│      μ_k = B_k · ȳ_k                 [posterior mean]      │
│      V_k = σ² / (n_k + λ)            [posterior variance]  │
│      λ = σ²_obs / σ²_FK              [noise-to-signal]     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Variance Reduction Formula

### 3.1 Variance as Function of n

```
V(n) = σ² / (n + λ)
```

This is a **decreasing function** of n (more data → less variance).

### 3.2 Ratio After Adding Δn Samples

```
V(n + Δn) = σ² / (n + Δn + λ)

V(n + Δn)     n + λ
───────── = ─────────────
  V(n)       n + Δn + λ
```

### 3.3 Variance Reduction (Percentage)

```
                    V(n + Δn)           n + λ
Reduction = 1 - ───────────── = 1 - ─────────────
                    V(n)             n + Δn + λ

                n + Δn + λ - n - λ       Δn
          = ───────────────────── = ─────────────
                 n + Δn + λ          n + Δn + λ
```

### 3.4 Key Formula

```
┌─────────────────────────────────────────────────────────────┐
│  VARIANCE REDUCTION FORMULA                                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│                        Δn                                   │
│  Reduction = ─────────────────                              │
│               n + Δn + λ                                    │
│                                                             │
│  where:                                                     │
│      n = current samples                                    │
│      Δn = additional samples                                │
│      λ = σ²_obs / σ²_FK                                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Analysis of the Formula

### 4.1 Limiting Cases

**Case 1: λ → 0 (precise observations, σ²_obs << σ²_FK)**
```
Reduction → Δn / (n + Δn)
```
This is the "ideal" case - maximum benefit from data.

**Case 2: λ → ∞ (noisy observations, σ²_obs >> σ²_FK)**
```
Reduction → 0
```
Data is too noisy to help - prior dominates.

**Case 3: n = 0 (starting from scratch)**
```
Reduction = Δn / (Δn + λ)
```

**Case 4: Large n (already have lots of data)**
```
Reduction ≈ Δn / n    (for n >> Δn, λ)
```
Diminishing returns.

### 4.2 Numerical Examples

| n | Δn | λ | Reduction | Interpretation |
|---|----|----|-----------|----------------|
| 0 | 100 | 1 | 99% | Starting fresh, λ=1 |
| 0 | 100 | 10 | 91% | Starting fresh, noisy |
| 0 | 100 | 100 | 50% | Starting fresh, very noisy |
| 100 | 100 | 1 | 50% | Doubling data |
| 100 | 100 | 10 | 48% | Doubling data, noisy |
| 500 | 100 | 1 | 17% | Already have lots |

### 4.3 Connecting to Empirical Results

**F1 Experiment**:
- n = 0, Δn = 551
- Observed reduction: 84.5%

From formula: 84.5% = 551 / (551 + λ)
```
551 + λ = 551 / 0.845 = 652
λ = 101
```

**Implied**: σ²_obs / σ²_FK ≈ 101

This means observations are about 100x noisier than the prior spread of FK importances. This is plausible given permutation variability.

---

## 5. Theorem 1: Single FK Bound

### Statement

**Theorem 1 (Variance Reduction Bound)**

Let α_k be the importance of FK k with prior α_k ~ N(0, σ²_FK).
Given n_k observations with noise variance σ²_obs, the posterior variance satisfies:

```
V(n_k) = σ²_obs · σ²_FK / (n_k · σ²_FK + σ²_obs)
```

Adding Δn samples reduces variance by exactly:

```
V(n_k + Δn)     n_k + λ
─────────── = ───────────────
  V(n_k)       n_k + Δn + λ
```

where λ = σ²_obs / σ²_FK.

Equivalently, the **reduction rate** is:

```
                       Δn
Reduction(n_k, Δn) = ─────────────────
                      n_k + Δn + λ
```

### Bounds

**Lower bound** (worst case, λ → ∞):
```
Reduction ≥ 0
```

**Upper bound** (best case, λ → 0):
```
Reduction ≤ Δn / (n_k + Δn)
```

### Proof

Direct computation from Normal-Normal conjugate posterior. See Section 2-3.  ∎

---

## 6. Hierarchical Case Setup

### 6.1 Why Hierarchy Helps

With K > 1 FKs, we can **estimate** σ²_FK from data:

```
σ̂²_FK = estimate from {ȳ_1, ..., ȳ_K}
```

Key insight: **Even FK k with few samples benefits from FKs with many samples** through the shared σ²_FK estimate.

### 6.2 Marginal Distribution of ȳ_k

Given the model:
```
α_k | σ²_FK ~ N(0, σ²_FK)
ȳ_k | α_k ~ N(α_k, σ²_obs/n_k)
```

Marginalizing over α_k:
```
ȳ_k | σ²_FK ~ N(0, σ²_FK + σ²_obs/n_k)
```

### 6.3 Empirical Bayes Estimator

**Equal sample sizes** (n_k = n for all k):

Let v = σ²_FK + σ²_obs/n. The MLE is:
```
v̂ = (1/K) Σ_k ȳ_k²

σ̂²_FK = (1/K) Σ_k ȳ_k² - σ²_obs/n
```

**Variance of estimator**:
```
Var(σ̂²_FK) ≈ (2/K) · (σ²_FK + σ²_obs/n)²
```

**Key**: Variance decreases as O(1/K).

### 6.4 The Hierarchical Benefit

```
┌─────────────────────────────────────────────────────────────┐
│  HIERARCHICAL BENEFIT                                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  With K FKs:                                                │
│      Var(σ̂²_FK) = O(1/K)                                   │
│                                                             │
│  Effect on posterior:                                       │
│      - K = 1: σ²_FK poorly estimated, high uncertainty     │
│      - K large: σ²_FK well estimated, formula is accurate  │
│                                                             │
│  Borrowing strength:                                        │
│      - Sparse FK benefits from data-rich FKs               │
│      - All FKs contribute to estimating σ²_FK              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 7. Toward Theorem 2

### 7.1 Total Variance Decomposition

By law of total variance:
```
Var(α_k | y) = E[Var(α_k | y, σ²_FK)] + Var[E(α_k | y, σ²_FK)]
             = E[V_k] + Var[μ_k]
```

- First term: Average posterior variance (as if σ²_FK known)
- Second term: Additional variance from uncertainty in σ²_FK

### 7.2 Approximation for Large K

When K is large, σ̂²_FK → σ²_FK, so:
```
Var(α_k | y) ≈ V_k(σ̂²_FK)
```

The plug-in approximation is accurate.

### 7.3 Theorem 2 (Preview)

**Theorem 2 (Hierarchical Benefit)**

Under the hierarchical model with K FKs, the effective posterior variance satisfies:

```
Var_hier(α_k | y) ≤ V_k(σ²_FK) · (1 + c/K)
```

for some constant c > 0.

**Implication**: As K → ∞, hierarchical is as good as knowing σ²_FK exactly.

**Proof**: Requires careful analysis of the second moment. [To be completed in Phase 2]

---

## 8. Summary of Phase 1

### Completed

✅ Formal model specification
✅ Single-FK posterior derivation
✅ Variance reduction formula: Reduction = Δn / (n + Δn + λ)
✅ Theorem 1 statement and proof
✅ Connection to empirical results (F1: λ ≈ 101)
✅ Setup for hierarchical case

### Key Results

**The ROI Formula (Single FK)**:
```
If you have n samples and collect Δn more:

                    Δn
Reduction = ─────────────────
             n + Δn + λ

where λ = σ²_obs / σ²_FK (estimable from data)
```

**Interpretation**:
- λ small → observations informative → big reduction
- λ large → observations noisy → small reduction
- n large → diminishing returns

### Next Steps (Phase 2)

- [ ] Complete Theorem 2 proof (hierarchical benefit)
- [ ] Derive the constant c in the bound
- [ ] Extend to unequal sample sizes
- [ ] Validate λ estimation from empirical data

---

*Document complete for Phase 1.*
