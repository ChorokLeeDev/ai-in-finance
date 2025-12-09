# Hierarchical Bayesian Models - The Field

**Created**: 2025-12-09
**Purpose**: Background on hierarchical Bayesian models and our theoretical opportunity

---

## 1. The Founding Work

### Gelman & Hill (2006)
"Data Analysis Using Regression and Multilevel/Hierarchical Models" - The bible of hierarchical modeling.

**The key insight - Three approaches:**

```
Complete pooling:    All groups share one parameter (ignores differences)
No pooling:          Each group has own parameter (ignores similarities)
Partial pooling:     Hierarchical - groups share prior, borrow strength ← THE WINNER
```

---

## 2. Why It Became a Field

Real data is ALWAYS hierarchical:

| Domain | Hierarchy |
|--------|-----------|
| Education | Students → Classrooms → Schools → Districts |
| Medicine | Patients → Doctors → Hospitals → Regions |
| Finance | Transactions → Customers → Segments → Markets |
| Sports | Plays → Games → Seasons → Teams |
| **Our case** | Rows → Value Ranges → Columns → FK Tables |

---

## 3. Key Concepts

### 3.1 Shrinkage

Small sample groups get "pulled" toward global mean. Large sample groups stay close to own estimate.

```
Example:
- FK with 10 samples:   α_k ≈ 0.7 × global_mean + 0.3 × own_estimate
- FK with 1000 samples: α_k ≈ 0.1 × global_mean + 0.9 × own_estimate
```

**Why this matters for us**: Sparse FKs (few samples) get regularized, avoiding overconfident wrong estimates.

### 3.2 Borrowing Strength

```
Estimating σ_FK (how variable are FK importances?)
  ↓
Uses data from ALL FKs
  ↓
Even sparse FKs contribute to learning σ_FK
  ↓
σ_FK then helps estimate sparse FK's importance
```

**Why this matters for us**: Information flows between FKs through the shared hyperprior.

### 3.3 The Bias-Variance Tradeoff

```
No pooling:     Unbiased but high variance (overfit to small samples)
Full pooling:   Biased but low variance (underfit, ignores differences)
Hierarchical:   Optimal tradeoff - small bias, reduced variance
```

---

## 4. Famous Results We Build On

### 4.1 Stein's Paradox (1961)

> Estimating 3+ means simultaneously? The MLE is INADMISSIBLE.
> Shrinkage estimator always beats MLE in total MSE.

**Implication**: Hierarchical models are provably better than estimating each FK independently.

### 4.2 Empirical Bayes (Robbins, 1956)

> Don't need full Bayes - estimate hyperparameters from data.
> James-Stein estimator is empirical Bayes.

**Implication**: Even without full MCMC, we can get shrinkage benefits.

### 4.3 Efron & Morris (1975)

> Derived the optimal shrinkage factor:
> B_k = σ²_within / (σ²_within + σ²_between)

**Implication**: There's a closed-form optimal shrinkage we can compute.

---

## 5. The Core Formulas

### Hierarchical Normal Model

```
Prior:      α_k ~ N(μ, τ²)      # FK importance
Likelihood: y_ki ~ N(α_k, σ²)   # Observations for FK k
```

### Posterior Mean (Shrinkage Estimator)

```
E[α_k | data] = B_k · ȳ_k + (1 - B_k) · μ_pooled

where:
  B_k = τ² / (τ² + σ²/n_k)        # shrinkage factor (0 to 1)
  ȳ_k = sample mean for FK k
  μ_pooled = global mean across all FKs
```

**Interpretation**:
- B_k → 1 when n_k large (trust own data)
- B_k → 0 when n_k small (trust global mean)

### Posterior Variance

```
Var(α_k | data) = B_k · σ²/n_k = τ² · σ²/n_k / (τ² + σ²/n_k)
```

**This is our starting point for deriving intervention bounds!**

---

## 6. What's NOT Done (Our Opportunity)

| Existing Work | Missing (Our Contribution) |
|---------------|---------------------------|
| Shrinkage for parameter estimation | Shrinkage for **uncertainty attribution** |
| Bounds on posterior variance | Bounds on **intervention effect** (Δn → ΔVar) |
| General hierarchical models | Exploiting **FK relational structure** specifically |
| Theory for prediction | Theory for **"where to collect data"** |
| Posterior variance formula | **ROI formula** for data collection |

### The Gap We Fill

**Existing**: "Here's the posterior variance formula for hierarchical models"

**Our contribution**: "Here's how much that variance **decreases** when you collect Δn samples, and how the **FK structure** gives additional benefits"

---

## 7. Key Papers to Read

### Foundational

1. **Gelman & Hill (2006)** - "Data Analysis Using Regression and Multilevel/Hierarchical Models"
   - The bible, start here

2. **Efron & Morris (1975)** - "Data Analysis Using Stein's Estimator and its Generalizations"
   - Original shrinkage bounds, closed-form formulas

3. **Stein (1956)** - "Inadmissibility of the usual estimator for the mean of a multivariate normal distribution"
   - The paradox that started it all

### Prior Specification

4. **Gelman (2006)** - "Prior distributions for variance parameters in hierarchical models"
   - How to set priors on σ_FK (we use HalfNormal)

5. **Polson & Scott (2012)** - "On the half-Cauchy prior for a global scale parameter"
   - Why HalfNormal/HalfCauchy for variance parameters

### Modern Treatment

6. **Scott & Berger (2010)** - "Bayes and empirical-Bayes multiplicity adjustment in the variable-selection problem"
   - Modern shrinkage theory

7. **Van der Pas et al. (2014)** - "The horseshoe estimator: Posterior concentration around sparse vectors"
   - Sparse estimation in hierarchical models

---

## 8. The Path to Our Theoretical Contribution

### Step 1: Formalize Our Model

```
# Hyperprior
σ_FK ~ HalfNormal(1)

# FK-level importance
α_k ~ Normal(0, σ_FK)    for k = 1, ..., K

# Observed importance scores (from permutation)
y_k ~ Normal(α_k, σ_obs)
```

### Step 2: Derive Posterior Variance

Using conjugate normal-normal results:

```
Var(α_k | data, σ_FK) = σ_FK² · σ_obs² / (σ_FK² · n_k + σ_obs²)
```

### Step 3: Derive Intervention Effect Bound

**Key theorem to prove**:

> Collecting Δn additional samples for FK k reduces posterior variance by:
>
> Var(α_k | n_k + Δn) / Var(α_k | n_k) ≤ n_k / (n_k + Δn) + hierarchy_correction
>
> where hierarchy_correction quantifies the benefit of shared σ_FK

### Step 4: Show FK Structure Benefit

Compare:
- Independent estimation: Each FK estimated separately
- Hierarchical estimation: FKs share σ_FK hyperprior

**Prove**: Hierarchical gives tighter variance, especially for sparse FKs.

### Step 5: Derive ROI Formula

The practical output - a formula practitioners can use:

```
Expected uncertainty reduction = f(n_current, Δn, K, σ_FK, σ_obs)
```

No need to run experiments - just plug in the numbers.

---

## 9. Why This Is Publishable

### Theoretical Contribution
- First bounds on uncertainty **attribution** under hierarchical structure
- Quantified benefit of FK hierarchy for **intervention planning**
- Closed-form **ROI formula** for data collection

### Practical Contribution
- Risk managers can estimate effect **before** collecting data
- Optimal allocation of data collection budget across FKs
- Connects Bayesian theory to actionable decisions

### Novelty Check
- Not just applying existing bounds (we exploit FK structure)
- Not just empirical (we prove guarantees)
- Not just theory (we validate on real data)

---

## 10. Next Steps

1. [ ] Write down formal model specification
2. [ ] Derive single-FK posterior variance (known result, verify)
3. [ ] Extend to hierarchical case with learned σ_FK
4. [ ] Derive the intervention effect bound (new contribution)
5. [ ] Prove FK structure gives tighter bound than independent
6. [ ] Validate bounds match our empirical results (F1: 84.5%, SALT: 34.8%)

---

*This document establishes the theoretical foundation for v5.*
*Next: Formalize model and derive bounds.*
