# Hierarchical Bayesian Intervention Analysis
## Structured Uncertainty Decomposition with Actionable Recommendations

**Author**: ChorokLeeDev
**Created**: 2025-12-09
**Status**: Design Document

---

## 1. Problem Statement

### Current State (v3 FK Attribution)
```
Level 1: ITEM 테이블이 37% 기여
Level 2: ITEM.weight 컬럼이 45% 기여
Level 3: weight가 [0, 0.5] 범위일 때 68% 기여
```
**문제**: "어디가 문제인지"는 알지만, "얼마나 바꾸면 되는지"는 모름

### Desired State (v4 Hierarchical Bayesian)
```
Level 1: ITEM 테이블 → 데이터 품질 개선 시 불확실성 -12% [95% CI: -15%, -9%]
Level 2: ITEM.weight → 결측치 보정 시 불확실성 -5% [95% CI: -7%, -3%]
Level 3: weight ∈ [0, 0.5] → 추가 데이터 수집 시 불확실성 -8% [95% CI: -12%, -4%]
```
**핵심**: Point estimate가 아닌 **credible interval**과 함께 intervention effect 제공

---

## 2. Why Bayesian ML?

### 2.1 Natural Fit for Hierarchical Structure

Relational Database = Hierarchical Structure:
```
Database
├── Table A (FK: customer)
│   ├── Column A1 (age)
│   │   ├── Value range [20-30]
│   │   └── Value range [30-40]
│   └── Column A2 (income)
├── Table B (FK: product)
│   └── ...
```

Bayesian Hierarchical Model encodes this naturally:
```
# FK-level prior
α_FK ~ Normal(0, σ_FK)

# Column-level prior (nested)
β_col|FK ~ Normal(α_FK, σ_col)

# Value-level prior (nested)
γ_val|col ~ Normal(β_col, σ_val)
```

### 2.2 Credible Intervals on Interventions

Frequentist: "Intervention reduces uncertainty by 12%"
Bayesian: "Intervention reduces uncertainty by 12% [95% CI: 9%, 15%]"

The credible interval tells us:
- **Confidence in the estimate**: Wide CI = need more data
- **Risk assessment**: Even worst-case (9%) might be worth it
- **Prioritization**: Compare overlapping CIs to rank interventions

### 2.3 Posterior Predictive for "What-If" Analysis

```python
# Current state
p(y | X, data)  # posterior predictive

# After intervention on FK_i
p(y | do(X_FK_i = x'), data)  # interventional posterior

# Effect distribution
p(Δ | data) = p(uncertainty_before - uncertainty_after | data)
```

---

## 3. Framework Design

### 3.1 Three-Level Hierarchy

```
┌─────────────────────────────────────────────────────────┐
│                    DATABASE                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │  FK: A   │  │  FK: B   │  │  FK: C   │   Level 1    │
│  │  (35%)   │  │  (28%)   │  │  (22%)   │   (Table)    │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘              │
│       │             │             │                     │
│  ┌────┴────┐   ┌────┴────┐   ┌────┴────┐              │
│  │col1 col2│   │col1 col2│   │col1 col2│   Level 2    │
│  │45%  30% │   │60%  25% │   │50%  35% │   (Column)   │
│  └────┬────┘   └─────────┘   └─────────┘              │
│       │                                                 │
│  ┌────┴────────────┐                                   │
│  │ [0,10]: 40%     │                                   │
│  │ [10,20]: 35%    │   Level 3 (Value Range)          │
│  │ [20,30]: 25%    │                                   │
│  └─────────────────┘                                   │
└─────────────────────────────────────────────────────────┘
```

### 3.2 Intervention Types

| Level | Intervention Type | Example |
|-------|------------------|---------|
| FK (Table) | Improve data pipeline | Fix ETL errors from source system |
| Column | Impute/correct | Fill missing values, fix outliers |
| Value Range | Collect more data | Get more samples in sparse regions |

### 3.3 Effect Estimation

For each node, compute:

```python
def estimate_intervention_effect(node, intervention_type):
    """
    Returns: posterior distribution p(Δ_uncertainty | data)
    """
    # 1. Current uncertainty
    U_current = get_epistemic_uncertainty(X, model)

    # 2. Simulate intervention (multiple samples for Bayesian)
    effects = []
    for _ in range(n_posterior_samples):
        X_intervened = apply_intervention(X, node, intervention_type)
        U_after = get_epistemic_uncertainty(X_intervened, model)
        effects.append(U_current - U_after)

    # 3. Return posterior over effect
    return {
        'mean': np.mean(effects),
        'std': np.std(effects),
        'ci_lower': np.percentile(effects, 2.5),
        'ci_upper': np.percentile(effects, 97.5)
    }
```

---

## 4. Bayesian Model Specification

### 4.1 Hierarchical Prior Structure

```python
import pyro
import pyro.distributions as dist

def hierarchical_uncertainty_model(X, fk_structure):
    """
    Hierarchical Bayesian model for uncertainty decomposition.

    X: features grouped by FK
    fk_structure: {fk_name: [column_names]}
    """
    # Hyperpriors
    sigma_fk = pyro.sample("sigma_fk", dist.HalfNormal(1.0))
    sigma_col = pyro.sample("sigma_col", dist.HalfNormal(0.5))
    sigma_val = pyro.sample("sigma_val", dist.HalfNormal(0.25))

    uncertainty_contributions = {}

    for fk_name, columns in fk_structure.items():
        # Level 1: FK effect
        alpha_fk = pyro.sample(f"alpha_{fk_name}",
                               dist.Normal(0, sigma_fk))

        for col in columns:
            # Level 2: Column effect (nested in FK)
            beta_col = pyro.sample(f"beta_{fk_name}_{col}",
                                   dist.Normal(alpha_fk, sigma_col))

            # Level 3: Value range effects (nested in column)
            for value_range in get_value_ranges(X[col]):
                gamma_val = pyro.sample(
                    f"gamma_{fk_name}_{col}_{value_range}",
                    dist.Normal(beta_col, sigma_val)
                )
                uncertainty_contributions[(fk_name, col, value_range)] = gamma_val

    return uncertainty_contributions
```

### 4.2 Intervention Effect Model

```python
def intervention_effect_model(X, node, intervention_type, base_model):
    """
    Model for intervention effect with uncertainty.

    Returns posterior: p(Δ | intervention, data)
    """
    # Prior on effect size (skeptical prior - centered at 0)
    effect_mean = pyro.sample("effect_mean", dist.Normal(0, 0.1))
    effect_std = pyro.sample("effect_std", dist.HalfNormal(0.05))

    # Simulate intervention
    with pyro.plate("simulations", n_sims):
        X_intervened = simulate_intervention(X, node, intervention_type)

        # Get uncertainty before/after
        U_before = base_model.get_uncertainty(X)
        U_after = base_model.get_uncertainty(X_intervened)

        # Observed effect
        delta = U_before - U_after

        # Likelihood
        pyro.sample("observed_effect",
                    dist.Normal(effect_mean, effect_std),
                    obs=delta)

    return effect_mean, effect_std
```

---

## 5. Algorithm

### 5.1 Full Pipeline

```
Input:
  - Dataset X with FK structure
  - Trained UQ model (ensemble, MC Dropout, etc.)

Output:
  - Hierarchical intervention recommendations with credible intervals

Algorithm:
1. DECOMPOSE: Compute uncertainty attribution at each level
   - Level 1: FK attribution (permutation-based)
   - Level 2: Column attribution within top FKs
   - Level 3: Value range attribution within top columns

2. SIMULATE: For each high-attribution node
   - Define intervention (impute, correct, collect more)
   - Simulate n_samples of intervened data
   - Compute uncertainty change for each sample

3. INFER: Get posterior over intervention effects
   - Fit Bayesian model to simulated effects
   - Extract posterior mean and credible intervals

4. RANK: Prioritize interventions
   - By expected effect (posterior mean)
   - By confidence (CI width)
   - By cost-effectiveness (effect / estimated cost)

5. REPORT: Generate actionable recommendations
   - "Change X from A to B → expected -12% uncertainty [CI: -15%, -9%]"
```

### 5.2 Intervention Simulation Methods

```python
def simulate_intervention(X, node, intervention_type):
    """Simulate different types of data interventions."""

    X_new = X.copy()

    if intervention_type == "impute_mean":
        # Replace with column mean (removes variance)
        X_new[node.column] = X[node.column].mean()

    elif intervention_type == "reduce_variance":
        # Shrink toward mean by factor
        mean = X[node.column].mean()
        X_new[node.column] = mean + 0.5 * (X[node.column] - mean)

    elif intervention_type == "remove_outliers":
        # Clip to [Q1-1.5*IQR, Q3+1.5*IQR]
        Q1, Q3 = X[node.column].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        X_new[node.column] = X[node.column].clip(Q1-1.5*IQR, Q3+1.5*IQR)

    elif intervention_type == "add_samples":
        # Simulate having more data in sparse region
        # (reduce uncertainty by bootstrap-like resampling)
        mask = node.value_range.contains(X[node.column])
        X_new = pd.concat([X, X[mask].sample(frac=0.5, replace=True)])

    return X_new
```

---

## 6. Expected Output

### 6.1 Hierarchical Report

```
═══════════════════════════════════════════════════════════════════
HIERARCHICAL BAYESIAN INTERVENTION ANALYSIS
Domain: SALT (Manufacturing ERP)
═══════════════════════════════════════════════════════════════════

LEVEL 1: FK TABLE INTERVENTIONS
───────────────────────────────────────────────────────────────────
FK Table      │ Current │ Intervention          │ Expected Effect
              │ Contrib │                       │ [95% CI]
───────────────────────────────────────────────────────────────────
PLANT         │  37.2%  │ Improve data quality  │ -12.3% [-15.1%, -9.5%]
ITEM          │  28.5%  │ Improve data quality  │  -8.7% [-11.2%, -6.2%]
CUSTOMER      │  22.1%  │ Improve data quality  │  -6.4% [-8.8%, -4.0%]
───────────────────────────────────────────────────────────────────

LEVEL 2: COLUMN INTERVENTIONS (within PLANT)
───────────────────────────────────────────────────────────────────
Column        │ Current │ Intervention          │ Expected Effect
              │ Contrib │                       │ [95% CI]
───────────────────────────────────────────────────────────────────
capacity      │  45.3%  │ Impute missing        │ -5.2% [-6.8%, -3.6%]
efficiency    │  31.2%  │ Fix measurement error │ -3.8% [-5.1%, -2.5%]
age           │  15.8%  │ Update stale data     │ -1.9% [-2.8%, -1.0%]
───────────────────────────────────────────────────────────────────

LEVEL 3: VALUE RANGE INTERVENTIONS (within PLANT.capacity)
───────────────────────────────────────────────────────────────────
Value Range   │ Current │ Intervention          │ Expected Effect
              │ Contrib │                       │ [95% CI]
───────────────────────────────────────────────────────────────────
[0, 100]      │  52.1%  │ Collect more data     │ -3.1% [-4.5%, -1.7%]
[100, 500]    │  31.4%  │ Verify outliers       │ -1.8% [-2.9%, -0.7%]
[500+]        │  16.5%  │ (low priority)        │ -0.8% [-1.4%, -0.2%]
───────────────────────────────────────────────────────────────────

TOP 3 RECOMMENDED INTERVENTIONS (by expected effect):
═══════════════════════════════════════════════════════════════════
Rank │ Target                    │ Action                │ Effect
═══════════════════════════════════════════════════════════════════
  1  │ PLANT (all)               │ Data quality audit    │ -12.3%
  2  │ ITEM (all)                │ Data quality audit    │  -8.7%
  3  │ PLANT.capacity            │ Impute missing values │  -5.2%
═══════════════════════════════════════════════════════════════════

CONFIDENCE ASSESSMENT:
  - High confidence: PLANT, ITEM interventions (narrow CIs)
  - Medium confidence: CUSTOMER interventions
  - Needs more data: PLANT.capacity [0,100] range
```

### 6.2 Visual Output (for paper)

```
          Intervention Effect with 95% Credible Intervals

PLANT          ████████████████████████░░░░  -12.3% [-15.1, -9.5]
ITEM           ████████████████░░░░░░░░░░░░   -8.7% [-11.2, -6.2]
CUSTOMER       ████████████░░░░░░░░░░░░░░░░   -6.4% [-8.8, -4.0]
PLANT.capacity ██████████░░░░░░░░░░░░░░░░░░   -5.2% [-6.8, -3.6]
PLANT.effic    ███████░░░░░░░░░░░░░░░░░░░░░   -3.8% [-5.1, -2.5]
               ──────────────────────────────────────────────────
               -20%     -15%     -10%      -5%       0%
                        Uncertainty Reduction →
```

---

## 7. Theoretical Justification

### 7.1 Connection to Causal Inference

Our "intervention" is analogous to Pearl's do-operator:

```
Standard:     P(Y | X)           # observational
Intervention: P(Y | do(X_i = x)) # interventional
```

In our context:
- Y = prediction uncertainty
- X_i = data at node i
- do(X_i = x) = "improve data quality at node i"

### 7.2 Uncertainty Decomposition Theorem

**Theorem**: For a hierarchical data structure with FK groupings {G_1, ..., G_k}:

```
Var[f(X)] = Σ_i Var[E[f(X) | X_{-G_i}]] + residual
```

where X_{-G_i} denotes all features except those in group G_i.

This justifies FK-level attribution as a valid decomposition of epistemic uncertainty.

### 7.3 Why Credible Intervals Matter

Frequentist CI: "If we repeated this experiment, 95% of CIs would contain the true value"
Bayesian CI: "Given the data, there's 95% probability the true effect is in this interval"

For **decision-making** (which intervention to prioritize), Bayesian CI is more intuitive and actionable.

---

## 8. Implementation Plan

### Phase 1: Core Framework
- [ ] Implement intervention simulation methods
- [ ] Implement effect estimation with bootstrap CI
- [ ] Test on SALT data

### Phase 2: Bayesian Extension
- [ ] Implement hierarchical Pyro model
- [ ] Add proper posterior inference
- [ ] Generate credible intervals

### Phase 3: Validation
- [ ] Compare with ground-truth interventions (synthetic data)
- [ ] Test across multiple domains (F1, H&M, Stack)
- [ ] Validate CI coverage

### Phase 4: Paper
- [ ] Formalize theorems
- [ ] Generate figures
- [ ] Write NeurIPS submission

---

## 9. Paper Positioning for NeurIPS Bayesian ML

**Title**: "Hierarchical Bayesian Intervention Analysis for Structured Uncertainty in Relational Data"

**Abstract** (draft):
> We present a framework for decomposing epistemic uncertainty in machine learning models along the natural hierarchical structure of relational databases. Unlike existing uncertainty quantification methods that provide aggregate uncertainty estimates, our approach identifies *where* in the data hierarchy uncertainty originates and *how much* targeted interventions would reduce it. Using a hierarchical Bayesian model, we provide credible intervals on intervention effects, enabling principled decision-making about data quality improvements. Experiments on 6 real-world domains show that our method consistently identifies actionable root causes with well-calibrated uncertainty estimates.

**Key Contributions**:
1. **Structured Decomposition**: First method to decompose epistemic uncertainty along FK hierarchy
2. **Intervention Effects**: Not just "what's wrong" but "what to change and by how much"
3. **Credible Intervals**: Bayesian uncertainty over the uncertainty reduction estimates
4. **Practical Algorithm**: Works with any UQ backbone (ensemble, MC Dropout, BNN)

---

*This document defines the research direction for v4_structured_uncertainty.*
