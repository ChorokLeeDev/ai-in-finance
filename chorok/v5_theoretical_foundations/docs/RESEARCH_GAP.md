# Research Gap: Shrinkage Behavior Under Temporal Distribution Shift

**Created**: 2025-12-09
**Status**: Gap Identified and Verified

---

## 1. The Gap in One Sentence

> **Nobody has studied how shrinkage behavior changes under temporal distribution shift, or when borrowing from historical data becomes harmful.**

---

## 2. Why This Gap Exists

### 2.1 Shrinkage is "Under the Hood"

Practitioners observe **symptoms** (accuracy drops), not **mechanisms** (shrinkage changed).

```
What practitioners see:
  "Model accuracy dropped from 85% to 72% after COVID"
  → Action: Retrain model

What's actually happening:
  "Shrinkage factor that was optimal pre-COVID now pulls
   estimates toward outdated patterns, causing errors"
  → Better action: Adjust shrinkage, use temporal neighborhoods
```

### 2.2 Community Silos

| Community | Studies | Thinks About Shrinkage? |
|-----------|---------|------------------------|
| ML Engineering | Model accuracy, deployment | No - black box |
| Deep Learning | Architectures, optimization | No - no hierarchy |
| Classical Statistics | Shrinkage theory | Yes - but static setting |
| Bayesian Statistics | Hierarchical models | Yes - but assumes exchangeability |
| MLOps | Distribution shift detection | No - accuracy-focused |

**The gap is at the intersection that no community owns.**

### 2.3 Shrinkage Requires Hierarchical Structure

Most distribution shift research focuses on neural networks, which don't have explicit shrinkage. Shrinkage research focuses on hierarchical/mixed models, which assume static data.

---

## 3. What Exists (Literature Review)

### 3.1 Shrinkage Theory (Static)

| Paper | Contribution | Temporal Shift? |
|-------|--------------|-----------------|
| Stein (1956) | MLE inadmissibility | No |
| James-Stein (1961) | Shrinkage estimator | No |
| Efron & Morris (1971-75) | Empirical Bayes, tyranny of majority | No |
| Gelman & Hill (2006) | Hierarchical models textbook | No |

**Limitation**: All assume data is exchangeable / from same distribution.

### 3.2 Distribution Shift (No Shrinkage Focus)

| Paper | Contribution | Shrinkage? |
|-------|--------------|------------|
| Covariate shift literature | Detection, adaptation | No |
| Concept drift literature | Online learning | No |
| COVID ML papers (2020-23) | Performance degradation | No |
| Domain adaptation | Transfer learning | No |

**Limitation**: Focus on model accuracy, not shrinkage mechanism.

### 3.3 Recent Relevant Work (Close But Not There)

| Paper | Year | What They Do | What They Don't Do |
|-------|------|--------------|-------------------|
| Local Empirical Bayes | 2024 | Neighborhood-based shrinkage | Not temporal neighborhoods |
| Individual Shrinkage | 2023 | Individual vs aggregate accuracy | Not temporal shift |
| Misspecification-robust Shrinkage | 2025 | Robustness to model error | Not temporal shift specifically |
| Exchangeable GPs for Epidemics | 2024 | Temporal Bayesian models | Not shrinkage behavior analysis |

**These papers are close but don't address the core question.**

---

## 4. The Research Questions

### Primary Question

> **When data undergoes temporal distribution shift, how does optimal shrinkage change, and when does borrowing from historical data become harmful?**

### Sub-Questions

1. **Measurement**: How do we measure shrinkage factor change across a temporal boundary?

2. **Detection**: Can we detect when shrinkage is becoming harmful before accuracy drops?

3. **Adaptation**: Should we use "temporal neighborhoods" - only borrow from recent data?

4. **Theory**: Under what conditions does pre-shift shrinkage hurt post-shift estimation?

---

## 5. Why This Matters

### 5.1 Explains WHY Models Break (Not Just THAT They Broke)

```
Current: "Accuracy dropped, retrain"
Better:  "Shrinkage from old data is harmful, adjust borrowing"
```

### 5.2 Guides Targeted Intervention

| Diagnosis | Intervention |
|-----------|--------------|
| "Accuracy dropped" | Retrain on all data (may include harmful old data) |
| "Shrinkage from pre-shift data is harmful" | Reduce shrinkage, use recent data only, temporal neighborhoods |

### 5.3 Enables Predictive Monitoring

```
REACTIVE (current):
  1. Deploy model
  2. Wait for accuracy to drop
  3. Retrain

PREDICTIVE (shrinkage-aware):
  1. Monitor shrinkage factor stability
  2. Detect when borrowing becomes unstable
  3. Adapt BEFORE accuracy crashes
```

### 5.4 Connects Theory to Practice

Shrinkage theory (60 years old) + Distribution shift (modern ML concern) = Actionable insights for practitioners.

---

## 6. Is This Statistics or ML?

### Answer: It Bridges Both

```
CLASSICAL STATISTICS              BAYESIAN ML                 DISTRIBUTION SHIFT
(1956-1990s)                     (1990s-2020s)               (2010s-now)
     │                                │                            │
     │  Shrinkage theory              │  Hierarchical models       │  Shift detection
     │  James-Stein                   │  MCMC, VI                  │  Monitoring
     │  Efron-Morris                  │  Gelman, Stan              │  Retraining
     │                                │                            │
     │  ASSUMES: Static data          │  ASSUMES: Exchangeability  │  IGNORES: Shrinkage
     │                                │                            │
     └────────────────────────────────┴────────────────────────────┘
                                      │
                                      ▼
                              ┌───────────────────┐
                              │   THE GAP (US)    │
                              │                   │
                              │  Shrinkage under  │
                              │  temporal shift   │
                              └───────────────────┘
```

### The Math is the Same

**Frequentist shrinkage (James-Stein)**:
```
θ̂_shrunk = B × θ̂_own + (1-B) × θ̂_global
B = n / (n + λ)
```

**Bayesian shrinkage (Hierarchical model)**:
```
Posterior mean = B × likelihood_mean + (1-B) × prior_mean
B = n / (n + λ)  ← SAME FORMULA
```

**Our question applies to both**: When λ should change due to temporal shift.

---

## 7. Why SALT Dataset is Perfect

```
SALT Dataset Structure:

Training data: ... ─────────────┬─────────────── ...
                                │
                           Feb 2020
                          COVID SHIFT
                                │
Test data:     ... ─────────────┴─────────────── ...
                                │
                           Jul 2020

NATURAL EXPERIMENT:
- Pre-COVID: Hierarchical model learns shrinkage factors
- Post-COVID: Those factors may be WRONG
- We can measure: Did shrinkage help or hurt after shift?
```

---

## 8. Proposed Contribution

### Empirical

1. Measure shrinkage factors before/after COVID on SALT
2. Show when borrowing from pre-COVID data hurts
3. Demonstrate temporal neighborhood approach

### Theoretical

1. Formalize "temporal exchangeability violation"
2. Derive conditions for when pre-shift shrinkage hurts
3. Connect to Local EB framework (extend to temporal)

### Practical

1. Detection criteria for harmful shrinkage
2. Guidelines for practitioners: when to stop borrowing from history
3. Monitoring approach for shrinkage stability

---

## 9. Key References

### Must Cite (They're Close But We Extend)

1. **Local Empirical Bayes (2024)** - arXiv:2511.21282
   - They: Spatial/experimental neighborhoods
   - We: Temporal neighborhoods

2. **Individual Shrinkage (2023)** - arXiv:2308.01596
   - They: Individual vs aggregate accuracy
   - We: How this changes under temporal shift

3. **Misspecification-robust Shrinkage (2025)** - arXiv:2502.03693
   - They: General robustness
   - We: Temporal misspecification specifically

### Background

4. Efron & Morris (1971, 1973, 1975) - Shrinkage foundations
5. Gelman & Hill (2006) - Hierarchical models
6. COVID distribution shift papers (2020-2023)

---

## 10. Summary

| Aspect | Status |
|--------|--------|
| Gap verified? | **Yes** - no papers on shrinkage behavior under temporal shift |
| Unique angle? | **Yes** - intersection of shrinkage theory + distribution shift |
| Data available? | **Yes** - SALT dataset with COVID boundary |
| Connects to recent work? | **Yes** - extends Local EB, Individual Shrinkage |
| Practical value? | **Yes** - explains WHY models break, guides intervention |
| Novel? | **Yes** - mechanism (shrinkage) vs symptom (accuracy) |

---

*This document establishes the research gap for v5.*
*Key insight: Everyone studies accuracy under shift. Nobody studies shrinkage under shift.*
