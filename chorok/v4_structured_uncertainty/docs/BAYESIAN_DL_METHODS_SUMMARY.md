# Bayesian Deep Learning Methods - Summary from AI701 Lectures

**Source**: AI701 Lecture PDFs (bdl.pdf, uncertainty.pdf)
**Created**: 2025-12-09

---

## 1. Deep Ensemble (Lakshminarayanan et al., 2017)

**The simplest yet most powerful method.**

```
p(y|x) ≈ (1/M) Σ p(y|x; θ_m)
```

- Train same model M times with different random seeds
- M = 5~10 is sufficient
- Requires M× training and M× parameters
- **Even better than most recent techniques**
- Can explore diverse modes in function space

**Key insight from PDF**: "Deep ensemble might be a good solution but requires heavy computation."

### Relevance to Our Work
- We use LightGBM ensemble (similar principle)
- Our hierarchical Bayesian model is complementary - provides structure

---

## 2. SWAG (Stochastic Weight Averaging Gaussian)

**Maddox et al., 2019**

Based on observation: SGD with constant learning rate defines a Markov chain targeting the true posterior.

**Algorithm**:
1. Train with cyclic learning rates
2. Collect weight snapshots at cycle ends
3. Compute mean: θ_SWA = (1/T) Σ θ_i
4. Compute covariance: Σ = diagonal + low-rank

**Prediction**:
```
θ^(s) ~ N(θ_SWA, Σ)
p(y*|x*) ≈ (1/S) Σ p(y*|x*, θ^(s))
```

### Relevance to Our Work
- Alternative to full ensemble (more memory efficient)
- Could be used as backbone UQ method
- Not implemented yet in our framework

---

## 3. BatchEnsemble (Wen, Tran, Ba, 2020)

**Less expensive ensemble via rank-1 perturbations.**

```
W_m = W ⊙ r_m s_m^T,  m = 1, ..., M
```

- One shared weight matrix W
- M rank-1 perturbations (r_m, s_m)
- Much cheaper than full ensemble
- Train by dividing mini-batch into M sub-batches

### Rank-1 BNN (Dusenberry et al., 2020)
Bayesian version: put priors on r and s, optimize ELBO.

### Relevance to Our Work
- Efficient alternative for neural network ensembles
- Could reduce computational cost for FK attribution

---

## 4. MC Dropout Limitations

**Critical finding from PDF**:

> "MC dropout or variational inference are **suboptimal**, as they **underestimate the posterior variances**."

This confirms our experimental finding that MC Dropout gives different results than proper Bayesian methods.

### Implication for Our Work
- MC Dropout is practical but theoretically limited
- Deep Ensemble or proper VI preferred for accurate uncertainty

---

## 5. Priors on Functions (Functional BNN)

**Problem with weight-space priors**:
- Weights have no physical meaning
- Hard to specify meaningful priors
- Posteriors are high-dimensional and multi-modal
- Weight-function relationship is complicated

**Solution**: Place prior on functions directly.

- **Variational Implicit Processes** (Ma et al., 2019)
- **Functional Variational BNN** (Sun et al., 2019)

### Relevance to Our Work
- Our hierarchical prior is on **importance scores**, not weights
- This is more interpretable than weight-space priors
- FK structure provides meaningful prior information

---

## 6. Neural Processes (Garnelo et al., 2018)

**Meta-learning version of implicit processes.**

- Neural network analogy of Gaussian processes
- Data-driven way of learning stochastic processes
- Learns uncertainties from data

### Architecture
```
μ_z, σ²_z = f_enc(X, Y)
z ~ N(μ_z, σ²_z)
μ*, σ²* = f_dec(z, x*)
```

### Relevance to Our Work
- Could be interesting for multi-domain uncertainty learning
- Not directly applicable to FK attribution problem

---

## 7. Deep Bayesian Active Learning

**Key insight**: Most uncertain samples are most informative.

**Acquisition functions**:
1. **Predictive variance**: Var(y|x, L)
2. **Variation ratio**: Portion of models disagreeing with majority
3. **Maximum entropy**: H[y|x, L]
4. **BALD**: Mutual information I[y, θ|x, L]

**Important finding from PDF**:
> "Snapshot ensemble (deep ensemble from single training run with cyclical learning rate) is more effective for active learning."

### Relevance to Our Work
- FK Attribution can guide **which data to collect**
- High-uncertainty FKs = candidates for data collection
- Connects to "actionable recommendations"

---

## 8. Key Takeaways for Our Paper

### What Works Best (according to PDF)
1. **Deep Ensemble**: Simplest, most powerful, empirically best
2. **SWAG**: Good memory/performance tradeoff
3. **Snapshot Ensemble**: Efficient alternative

### What's Suboptimal
1. **MC Dropout**: Underestimates posterior variance
2. **Mean-field VI**: Also underestimates variance

### Our Positioning
| Method | PDF Assessment | Our Implementation |
|--------|---------------|-------------------|
| Deep Ensemble | Best | ✓ LightGBM Ensemble |
| MC Dropout | Suboptimal | ✓ Implemented (comparison) |
| SWAG | Good tradeoff | ✗ Not implemented |
| Hierarchical Bayesian | Novel | ✓ Our contribution |

### Novel Contribution
The PDF methods focus on **model uncertainty**.
Our work adds **structured decomposition** - where does uncertainty come from?

```
PDF Methods: "How uncertain is the prediction?"
Our Method:  "Which FK/column/value causes the uncertainty?"
             + Bayesian credible intervals on the attribution
```

---

## 9. References from PDF

- Lakshminarayanan et al. (2017) - Deep Ensembles
- Maddox et al. (2019) - SWAG
- Wen et al. (2020) - BatchEnsemble
- Dusenberry et al. (2020) - Rank-1 BNN
- Gal & Ghahramani (2016) - MC Dropout (referenced implicitly)
- Sun et al. (2019) - Functional Variational BNN
- Garnelo et al. (2018) - Neural Processes

---

*This document summarizes Bayesian DL methods from AI701 lectures and their relevance to our Hierarchical Bayesian Intervention Analysis project.*
