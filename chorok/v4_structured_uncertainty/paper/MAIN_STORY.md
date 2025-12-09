# v4: Structured Decomposition of Epistemic Uncertainty

## Core Question

> When an ML model is uncertain, **which data source** is responsible?

## Key Insight

Relational database schema (FK structure) reflects the **data generating process**.
Epistemic uncertainty can be decomposed along this structure.

## Theoretical Claim

Under Error Propagation conditions:

$$\sigma^2_{FK_i} \propto I(Y; X_{FK_i} | X_{-FK_i})$$

FK-level uncertainty attribution = conditional mutual information decomposition

## Method-Agnostic Hypothesis

If this is a **fundamental property** of epistemic uncertainty (not an artifact of a specific UQ method), then FK attribution should work with:

- [x] LightGBM Ensemble
- [x] MC Dropout MLP
- [ ] Bayesian Neural Network (BNN)
- [ ] Deep Ensemble (proper)
- [ ] Gaussian Process (optional)

## Practical Value: Risk Management

Not just "where is uncertainty" but "what to fix to reduce it":

```
Diagnose: ITEM table (32.7%)
  → Drill: SHIPPINGPOINT column (80.9%)
    → Focus: values [0, 6] (51.3%)
      → Action: Improve these → 46% uncertainty reduction
```

## Venue Target

NeurIPS 2026 - Bayesian ML / Uncertainty Quantification

## Feasibility Experiments (Priority)

1. **BNN Feasibility**: Does FK attribution work with BNN?
2. **Method Comparison**: Same ranking across all UQ methods?
3. **Theoretical Validation**: Does attribution correlate with MI?

---

*Created: 2025-12-09*
*Status: Feasibility testing*
