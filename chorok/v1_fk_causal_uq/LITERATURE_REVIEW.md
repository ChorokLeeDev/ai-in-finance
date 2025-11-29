# Literature Review: Causal UQ for Relational Data

**Phase 2 Research Foundation**

---

## Executive Summary

This review identifies a **clear research gap**: while causal methods exist for feature attribution (Causal SHAP) and relational causal discovery (RelFCI, RelPC), **no framework provides interventional uncertainty attribution using FK structure as causal prior**.

---

## 1. The Problem with Correlational Attribution

### 1.1 Standard SHAP is Correlational

> "SHAP fails to differentiate between causality and correlation, often misattributing feature importance when features are highly correlated."
> — [Causal SHAP (2024)](https://arxiv.org/html/2509.00846)

**Key issues**:
- Permutation-based methods assume feature independence
- When features are correlated, marginal sampling creates unrealistic instances
- Attribution to spuriously correlated features is non-zero

### 1.2 Conditional vs. Marginal (Interventional)

Two approaches exist:
1. **Observational** (conditional): E[f(X) | X_S = x_S]
2. **Interventional** (marginal): E[f(X) | do(X_S = x_S)]

> "The conditional approach is fundamentally unsound from a causal perspective. The marginal approach should be preferred."
> — [Causal Analysis of Shapley Values](https://www.researchgate.net/publication/383917927_Causal_Analysis_of_Shapley_Values_Conditional_vs_Marginal)

**Our LOO method is observational** — we remove features and observe change, but don't intervene causally.

---

## 2. Causal Feature Attribution

### 2.1 Causal SHAP Framework

[Causal SHAP](https://arxiv.org/html/2509.00846) integrates causal graphs into SHAP:
- Uses PC algorithm for causal discovery
- Uses IDA algorithm for causal strength quantification
- Reduces attribution to spuriously correlated features to zero

**Limitation**: Requires causal discovery from data (computationally expensive, may be wrong).

### 2.2 Do-Calculus for Attribution

[Pearl's do-calculus](https://academic.oup.com/biomet/article/112/1/asae043/7738083) provides:
- Axiomatization of interventional distributions
- Rules for computing P(Y | do(X)) from observational data
- Identification conditions for causal effects

**Our opportunity**: FK relationships provide a **known causal structure** — no discovery needed!

---

## 3. Relational Causal Discovery

### 3.1 FK Relationships Encode Causality

> "Foreign-key relationships are used to grasp causal dependencies between events."
> — [Causal Process Mining from Relational Databases](https://www.researchgate.net/publication/358690039_Causal_Process_Mining_from_Relational_Databases_with_Domain_Knowledge)

Key insight: **FK from Table A → Table B implies B's attributes can causally affect predictions involving A**.

### 3.2 Relational Causal Discovery Algorithms

| Algorithm | Source | Key Feature |
|-----------|--------|-------------|
| RelPC | [Learning Causal Models of Relational Domains](https://www.researchgate.net/publication/215992251_Learning_Causal_Models_of_Relational_Domains) | First algorithm for relational causal discovery |
| RelFCI | [Relational Causal Discovery with Latent Confounders](https://arxiv.org/abs/2507.01700) | Handles latent confounders |
| Suna | [VLDB 2024](https://dl.acm.org/doi/10.14778/3749646.3749684) | GPU-accelerated confounder discovery |

**Our advantage**: We don't need to discover the causal graph — FK structure IS the causal prior.

---

## 4. Epistemic Uncertainty Attribution

### 4.1 Current State

> "Although current methods mainly focus on explaining predictions, with some including uncertainty, they fail to provide guidance on how to reduce the inherent uncertainty."
> — [Ensured: Explanations for Decreasing Epistemic Uncertainty](https://arxiv.org/abs/2410.05479)

**Existing work**:
- [Robust Explanations Through Uncertainty Decomposition](https://arxiv.org/html/2507.12913): Decomposes into epistemic/aleatoric
- [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_Epistemic_Uncertainty_Quantification_For_Pre-Trained_Neural_Networks_CVPR_2024_paper.pdf): Gradient-based epistemic UQ for pre-trained models

### 4.2 Gap: Causal Uncertainty Attribution

No existing work provides:
1. **Interventional** (not observational) uncertainty attribution
2. **Per-FK** (not per-feature) granularity
3. **Causal identification** from FK structure

---

## 5. Interventional UQ Methods

### 5.1 Gaussian Process Framework

> "The spectral Interventional Mean Process (IMPspec) is a GP-based framework for quantifying uncertainty over causal functions."
> — [ICLR 2023](https://causalai.net/r87.pdf)

Provides:
- Uncertainty over causal effects (not just predictions)
- Works with continuous treatments

**Limitation**: Not designed for relational data or FK structure.

### 5.2 Conformal Prediction + Causality

[UAI 2024](https://proceedings.mlr.press/v244/) featured work on:
- Prediction intervals for counterfactual outcomes
- Coverage guarantees under hidden confounding

**Our opportunity**: Combine FK-causal structure with conformal prediction for guaranteed coverage.

---

## 6. Research Gap (Our Contribution)

| Existing Work | What They Have | What They Lack |
|--------------|----------------|----------------|
| Causal SHAP | Interventional attribution | UQ focus, FK structure |
| RelFCI/RelPC | Relational causal discovery | UQ application, intervention |
| VFA/Ensembles | Epistemic UQ | Causal attribution |
| IMPspec | Interventional UQ | Relational data |
| Conformal | Coverage guarantees | Causal attribution |

### Our Proposed Framework

**FK-Causal-UQ**: Interventional uncertainty attribution for relational data

| Component | Our Approach |
|-----------|--------------|
| Causal Structure | FK relationships (given, not discovered) |
| Attribution Type | Interventional (do-calculus) |
| Target Quantity | Epistemic uncertainty (not prediction) |
| Guarantees | Identification theorem + Conformal coverage |

---

## 7. Theoretical Foundation

### 7.1 FK → DAG Mapping

For relational schema R with tables T₁, ..., Tₙ and FK constraints:
- **Nodes**: Tables (or table-attribute pairs)
- **Edges**: FK constraints define directed edges
- **Direction**: FK from A→B means B is a parent of A in causal DAG

### 7.2 Interventional UQ Query

For FK relationship fk from table A to B:

```
UQ_causal(fk) = E[U(Y) | do(fk = x)] - E[U(Y)]
```

Where U(Y) is uncertainty measure (entropy, variance).

Contrast with LOO (observational):
```
UQ_loo(fk) = E[U(Y) | features_fk removed] - E[U(Y)]
```

### 7.3 Identification Theorem (To Prove)

**Conjecture**: Under FK structure, causal UQ is identifiable if:
1. No unmeasured confounders between FK-linked tables
2. FK direction reflects causal direction
3. No feedback loops in FK graph

---

## 8. Key Papers to Cite

### Causal Inference + ML
1. [CLeaR 2024](https://proceedings.mlr.press/v236/) - Conference on Causal Learning and Reasoning
2. [UAI 2024](https://proceedings.mlr.press/v244/) - Uncertainty in AI
3. [Causal Inference Survey](https://pmc.ncbi.nlm.nih.gov/articles/PMC11384545/) - Comprehensive 2024 survey

### Relational Causal Discovery
4. [RelFCI](https://arxiv.org/abs/2507.01700) - Latent confounders in relational data
5. [Suna (VLDB)](https://dl.acm.org/doi/10.14778/3749646.3749684) - Scalable confounder discovery

### Causal Attribution
6. [Causal SHAP](https://arxiv.org/html/2509.00846) - Causal feature attribution
7. [Causal Shapley Values (NeurIPS 2020)](https://dl.acm.org/doi/abs/10.5555/3495724.3496125) - Foundational work

### Uncertainty + Explanation
8. [Ensured (2024)](https://arxiv.org/abs/2410.05479) - Epistemic uncertainty in explanations
9. [Uncertainty Decomposition](https://arxiv.org/html/2507.12913) - Robust explanations via UQ

---

## 9. Next Steps

Based on this review:

1. **Formalize FK → DAG mapping** (mathematical notation)
2. **Define interventional UQ query** (contrast with LOO)
3. **State identification theorem** (conditions for identifiability)
4. **Prove LOO ≠ Interventional** (theoretically, not just empirically)
5. **Design experiment**: Synthetic data with known causal ground truth

---

**Last Updated**: 2025-11-29
