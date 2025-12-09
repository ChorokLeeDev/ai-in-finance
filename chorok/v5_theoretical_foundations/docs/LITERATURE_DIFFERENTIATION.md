# Literature Differentiation: Our Novel Contribution

**Created**: 2025-12-09
**Purpose**: Document how our work differs from existing literature

---

## 1. The Key Prior Work

### Gu et al. (2020) - "Borrowing Strength and Borrowing Index for Bayesian Hierarchical Models"

**Citation**: Computational Statistics & Data Analysis, 2020
**Link**: https://pmc.ncbi.nlm.nih.gov/articles/PMC7185234/

#### What They Did

| Aspect | Their Approach |
|--------|----------------|
| **Goal** | Measure how much borrowing happened after fitting model |
| **Method** | Distance between hierarchical vs independent posteriors |
| **Formula** | BI = d(F_ind, F_hier) / [d(F_ind, F_hier) + d(F_hier, F_pool)] |
| **Output** | Borrowing Index ∈ [0, 1] |
| **Question** | "How much information was shared across groups?" |

#### Their Key Results

1. **Borrowing Index**: Normalized measure of where hierarchical model falls between independence (0) and complete pooling (1)

2. **Sample Size Effect**: Smaller groups borrow proportionally more than larger groups

3. **Theorem**: Their distance approximation provides a lower bound on true distance

#### What They Explicitly Did NOT Do

From their paper:
> "Does not address **smart borrowing** — distinguishing correct from incorrect information sharing"

They acknowledge this as future work.

---

## 2. Other Related Work

### Efron & Morris (1975) - Shrinkage Estimators

- Proved shrinkage is admissible for K ≥ 3 groups
- Derived optimal shrinkage factor
- **Did NOT**: Give variance reduction as function of future samples

### Gelman (2006) - Prior Distributions for Variance Parameters

- How to set priors on σ²_FK
- Effect of prior choice on inference
- **Did NOT**: Prospective intervention planning

### NBER (2024) - Empirical Bayes in Labor Economics

- Modern applications of EB methods
- Value-added models with varying standard errors
- **Did NOT**: ROI formula for data collection decisions

---

## 3. The Gap in Literature

### What Exists

```
RETROSPECTIVE ANALYSIS
├── "How much did groups borrow from each other?" (Gu 2020)
├── "What's the optimal shrinkage factor?" (Efron-Morris)
└── "How does prior choice affect inference?" (Gelman)
```

### What's Missing

```
PROSPECTIVE DECISION-MAKING
├── "If I collect Δn more samples, how much will variance drop?" [MISSING]
├── "What's my ROI on data collection investment?" [MISSING]
└── "When is borrowing DANGEROUS?" [MISSING]
```

---

## 4. Our Differentiated Contribution

### 4.1 The Core Difference

| Dimension | Prior Work | Our Work |
|-----------|------------|----------|
| **Timing** | Retrospective (after fitting) | **Prospective (before collecting)** |
| **Question** | "How much did we borrow?" | **"How much WILL uncertainty drop?"** |
| **Purpose** | Model comparison | **Investment decision** |
| **Output** | Index (descriptive) | **ROI formula (prescriptive)** |
| **Assumption** | Borrowing is good | **Borrowing can be dangerous** |

### 4.2 Our Three Novel Contributions

#### Contribution 1: Prospective Variance Reduction Formula

**Existing**: Measure borrowing after the fact
**Ours**: Predict variance reduction before collecting data

```
FORMULA:
                    Δn
Reduction = ─────────────────
             n + Δn + λ

USAGE:
  "I have 100 samples. If I collect 200 more, variance drops by 200/(100+200+λ)"

  No need to fit model first. Just plug in numbers.
```

#### Contribution 2: Unknown Unknowns Detection

**Existing**: Assumes borrowing is beneficial
**Ours**: Identifies when borrowing is dangerous

```
DANGEROUS REGION:
  Low uncertainty + Low sample count = Unknown Unknown

  Model is confident but WRONG because it has never seen this region.
  Borrowing from other groups makes it MORE confident, not better.

DETECTION:
  Flag when: Var(α_k) is low BUT n_k is low
  This is the "confidently wrong" scenario.
```

#### Contribution 3: Hierarchical ROI for Data Investment

**Existing**: Compare models after fitting
**Ours**: Guide data collection decisions prospectively

```
DECISION FRAMEWORK:

  For each FK k:
    1. Current samples: n_k
    2. Estimated λ_k = σ²_obs / σ²_FK
    3. If collect Δn samples:
       Guaranteed reduction ≥ Δn / (n_k + Δn + λ_k)
    4. Cost of collecting Δn: $C
    5. Value of uncertainty reduction: $V

  → Prioritize FKs by V/C ratio
```

### 4.3 The "Smart Borrowing" Gap We Fill

Gu et al. (2020) explicitly noted they do NOT address "smart borrowing."

We address this through **Unknown Unknowns Detection**:

```
SMART BORROWING = Knowing when NOT to trust borrowed information

Case 1: FK_k has many samples, borrows little → SAFE (data-driven)
Case 2: FK_k has few samples, borrows a lot, other FKs similar → SAFE (valid borrowing)
Case 3: FK_k has few samples, borrows a lot, FK_k is DIFFERENT → DANGEROUS (unknown unknown)

We detect Case 3 by checking:
  - Is n_k small?
  - Is posterior variance suspiciously low?
  - Is FK_k structurally different from others?
```

---

## 5. Positioning Statement

### For the Paper

> "While prior work quantifies borrowing strength retrospectively — measuring how much information was shared after model fitting (Gu et al., 2020) — we derive **prospective bounds** for intervention planning. Our framework answers: given current data, what's the **guaranteed minimum uncertainty reduction** from collecting Δn additional samples?
>
> Furthermore, we identify when hierarchical borrowing becomes **dangerous**: the 'unknown unknowns' where models exhibit low uncertainty despite having never observed a region. In these cases, borrowing from other groups increases confidence without improving accuracy — precisely the scenario risk managers most need to detect."

### One-Sentence Differentiation

> **Gu et al. (2020)**: "How much did we borrow?" (retrospective, descriptive)
>
> **Our work**: "How much WILL uncertainty drop, and when is borrowing dangerous?" (prospective, prescriptive)

---

## 6. Technical Differentiation

### Their Theorem vs Our Theorem

**Gu et al. Theorem 1**:
```
d*₂(F_M1, F_M2) ≤ d₂(F̂_M1, F̂_M2)

"Our distance approximation lower-bounds the true distance"
```

**Our Theorem 1**:
```
Var(α_k | n+Δn) / Var(α_k | n) = (n + λ) / (n + Δn + λ)

"Variance reduction is exactly this function of sample size"
```

**Our Theorem 2** (to prove):
```
Var_hier(α_k) ≤ Var_indep(α_k) × C(K)

"Hierarchy gives tighter variance by factor C(K) < 1"

PLUS: Warning when this bound is UNRELIABLE (unknown unknowns)
```

### Key Difference in Output

| Their Output | Our Output |
|--------------|------------|
| Borrowing Index ∈ [0,1] | Variance reduction % |
| "You borrowed 60% of available information" | "Collect 200 samples → 45% less uncertainty" |
| Compare models | Make decisions |

---

## 7. Why This Matters for NeurIPS

### Reviewer Question: "How is this different from Gu et al. 2020?"

**Answer**:
1. **Different question**: They measure borrowing retrospectively; we predict variance reduction prospectively
2. **Different output**: They give a descriptive index; we give a prescriptive ROI formula
3. **Different assumption**: They assume borrowing is good; we identify when it's dangerous
4. **Different use case**: They compare models; we guide data investment decisions

### Reviewer Question: "What's the theoretical novelty?"

**Answer**:
1. **Prospective bound** on variance reduction (not just retrospective measurement)
2. **Unknown unknowns formalization**: When low variance + low samples = dangerous
3. **Hierarchical ROI formula**: Closed-form for data investment decisions

---

## 8. Summary

### The Literature Landscape

```
EXISTING WORK
│
├── Shrinkage Estimation (Efron-Morris 1975)
│   └── "Shrinkage is optimal for K ≥ 3"
│
├── Prior Specification (Gelman 2006)
│   └── "How to set priors on variance parameters"
│
├── Borrowing Measurement (Gu et al. 2020)
│   └── "How much information was shared?"
│
└── [GAP] Prospective Intervention Planning
    └── "How much WILL uncertainty drop? When is borrowing dangerous?"
```

### Our Contribution Fills the Gap

```
OUR WORK
│
├── Prospective Variance Reduction Formula
│   └── Reduction = Δn / (n + Δn + λ)
│
├── Unknown Unknowns Detection
│   └── Low variance + low samples = dangerous
│
└── Hierarchical ROI Framework
    └── Guide data investment decisions with guarantees
```

---

*This document establishes our differentiated contribution for v5.*
*Key reference: Gu et al. (2020) - we build on but differ from this work.*
