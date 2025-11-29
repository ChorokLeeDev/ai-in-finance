# RelUQ: One-Pager for Advisor Meeting

**Date:** 2025-11-30
**Student:** [Name]
**Target:** NeurIPS 2025 (May deadline)

---

## 1. Research Question

> **When ML models are uncertain, WHERE does that uncertainty come from?**

More specifically: Given a prediction with high uncertainty, which **data source** (table/process) in a relational database is responsible?

---

## 2. Why This Matters (Motivation)

### The Problem in Practice
- Enterprise ML models train on data from **multiple joined tables** (customers, products, transactions, etc.)
- When predictions are uncertain, practitioners ask: *"What should I fix?"*
- Existing methods (SHAP, permutation importance) answer at **feature level**:
  - "Feature `customer_age` contributes 4.2% to uncertainty"
  - **Not actionable**: Which team owns `customer_age`? What process created it?

### The Gap
| What We Have | What We Need |
|--------------|--------------|
| Feature-level attribution | **Process-level** attribution |
| "Which variable?" | "Which data pipeline?" |
| Unstable across runs | Stable, reproducible |
| Interpretable | **Actionable** |

---

## 3. Our Solution: RelUQ

### Key Insight
**Foreign Key (FK) relationships in databases already encode data provenance.**

- Features from `driver` table → DRIVER FK group → owned by driver data team
- Features from `circuit` table → CIRCUIT FK group → owned by circuit data team

### The Framework
```
Input:  Relational DB + Trained Ensemble
Output: "DRIVER data causes 29% of prediction uncertainty"

Method:
1. Group features by FK origin (schema-defined)
2. Measure uncertainty via ensemble variance
3. Attribute to FK groups via permutation
```

---

## 4. What We've Done (Completed Experiments)

| Experiment | Status | Key Result |
|------------|--------|------------|
| Multi-domain validation (F1, Stack, Amazon) | ✅ Done | Stability ρ ≥ 0.85 across all |
| Baseline comparison | ✅ Done | FK matches correlation clustering, beats random |
| Ablation studies (K, P, n, subsample) | ✅ Done | Robust to hyperparameters |
| Theoretical justification | ✅ Done | Variance reduction proof |

### Key Findings
1. **Random grouping fails** (ρ = -0.40): Grouping structure matters
2. **FK = Correlation** (ρ = 0.93): Schema knowledge matches data-driven methods
3. **Only FK is actionable**: "DRIVER 29%" → check driver data team

---

## 5. Counterfactual Analysis (NEW - Just Implemented)

### The Right Question
- ❌ Wrong: "What values minimize uncertainty?" (creates out-of-distribution inputs)
- ✅ Right: "If we REDUCE NOISE in an FK, how much does uncertainty drop?"

### Method: Noise Sensitivity Analysis
```
1. Add noise to each FK group (5%, 10%, 20%, 50%)
2. Measure uncertainty increase
3. Inverse = reduction potential if noise is removed
```

### Results (rel-f1, just ran)
| FK Group | Attribution | Noise Sensitivity | Priority |
|----------|-------------|-------------------|----------|
| DRIVER | 29% | +287% when noised | 🥇 HIGH |
| CIRCUIT | 19% | +197% | 🥈 MEDIUM |
| PERFORMANCE | 22% | +175% | 🥉 |
| RACE | 19% | +157% | |
| CONSTRUCTOR | 11% | +84% | |

### Interpretation
- **DRIVER is most sensitive**: Adding 10% noise → +213% uncertainty
- **Actionable**: "Audit DRIVER data collection for noise sources"
- **Validates attribution**: Top FK by attribution = top FK by sensitivity

---

## 6. The Core Contribution (What We CAN Claim)

### Strong Claims (Supported by Experiments)
1. **FK grouping provides stable uncertainty attribution** (proven empirically)
2. **Schema-defined hierarchy enables drill-down** (FK → Feature → Entity)
3. **FK groups map to business processes** (actionable by design)

### Weak/Unsupported Claims (Need More Work)
1. ~~Intervention simulation predicts actual improvements~~ (not tested)
2. ~~Optimization finds minimal-risk configuration~~ (not implemented)

---

## 7. Path Forward

### Option A: Submit with Current Scope
- **Contribution**: Stable, actionable uncertainty attribution via FK grouping
- **Remove**: Intervention simulation claims (or move to "Future Work")
- **Risk**: Reviewers may ask "so what?" without actionability demonstration

### Option B: Implement Intervention Simulation (2-3 weeks)
- Create reference sets from low-uncertainty samples
- Measure predicted vs. actual uncertainty reduction
- Report calibration (predicted ↔ actual correlation)
- **Strengthens**: Actionability claim with empirical support

### Recommendation
**Option B** - The intervention experiment is straightforward to implement and directly validates our "actionability" claim, which is our main differentiator from correlation clustering.

---

## 8. Questions for Advisor

1. Is the current scope (stable attribution only) sufficient for NeurIPS, or do we need intervention experiments?
2. The theoretical analysis assumes high within-group correlation. Should we measure this empirically?
3. Should we target a different venue (ICML, KDD) if timeline is tight?

---

## Appendix: File Locations

- Paper draft: `paper/main.tex`
- Figures: `paper/figures/` (7 figures, all generated)
- Experiments: `experiments/ablation/`, `experiments/baselines/`
- Core code: `chorok/v3_fk_risk_attribution/`

---

*Generated: 2025-11-30*
