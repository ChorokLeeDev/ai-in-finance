# RelUQ: FK-Level Uncertainty Decomposition for Relational ML
## Paper Outline (NeurIPS 2026)

---

## Title Options

1. **"Beyond Feature Importance: FK-Level Uncertainty Decomposition for Data-Centric Relational Learning"**
2. "Where to Invest in Data: Uncertainty Decomposition for Relational Databases"
3. "FK-Level Epistemic Uncertainty: A Diagnostic Framework for Relational ML"

**Recommended**: Option 1 (emphasizes novelty over SHAP/feature importance)

---

## Core Claim (1 sentence)

> We propose FK-level uncertainty decomposition, a diagnostic framework that identifies which foreign key relationships in relational data contribute to model uncertainty, enabling targeted data investment decisions.

---

## Method Summary (2 sentences)

> Our method measures each FK's contribution to ensemble variance via permutation, decomposing total epistemic uncertainty into FK-level components. By combining importance (accuracy impact) with stability (uncertainty contribution), we provide a 2D framework that distinguishes "noisy signals" (important but uncertain—collect more data) from "stable signals" (important and certain—data sufficient).

---

## Key Results (3 bullets)

1. **4 domains, 11 tasks validated**: F1 racing, clinical trials, ERP systems, and online classifieds—framework provides actionable insights in 100% of tasks

2. **Clear data investment targets identified**:
   - OUTCOME_ANALYSES (+17%, +153% imp) → standardize analysis methods
   - SEARCHSTREAM (+10%, +147% imp) → collect more user behavior data
   - SALESDOCUMENT (+12%, +55% imp) → enrich sales order metadata

3. **Stable signals confirmed**: 6 FK-task combinations show >50% uncertainty reduction (model is confident, no additional data needed)

---

## Paper Structure

### 1. Introduction (1.5 pages)

**Hook**: "When building ML models on relational databases, practitioners face a critical resource allocation question: which tables should we invest in for better data quality?"

**Gap**: Feature importance (SHAP) tells us which features matter, but not whether more data would help. Uncertainty quantification tells us overall confidence, but not which FK relationships drive uncertainty.

**Our contribution**: FK-level uncertainty decomposition—answers "where should I collect more data?" for relational ML.

**[NO FIGURE in intro]**

---

### 2. Related Work (1 page)

- **Feature importance**: SHAP, permutation importance (feature-level, not FK-level)
- **Uncertainty quantification**: Ensembles, MC dropout, conformal prediction (model-level, not FK-level)
- **Data-centric AI**: Data quality, active learning, data valuation (no relational structure awareness)
- **Relational deep learning**: RelBench, GNNs for relational data (no uncertainty decomposition)

**Position**: First to combine UQ with relational structure for targeted data investment.

---

### 3. Method (2 pages)

#### 3.1 Problem Formulation

- Relational database with entity table E and FK tables {F₁, F₂, ..., Fₖ}
- Task: predict target y from entity features + FK-aggregated features
- Goal: identify which FK contributes most to epistemic uncertainty

#### 3.2 FK-Level Uncertainty Decomposition

```
Algorithm: FK Uncertainty Contribution
1. Train ensemble of models M₁, ..., Mₙ
2. Compute base uncertainty: U_base = Var(predictions across models)
3. For each FK_i:
   a. Permute FK_i's features (break the relationship)
   b. Compute permuted uncertainty: U_perm
   c. Contribution_i = (U_base - U_perm) / U_base × 100%
4. Return {(FK_i, Contribution_i)}
```

**Interpretation**:
- Positive contribution → FK adds uncertainty (noisy signal)
- Negative contribution → FK reduces uncertainty (stable signal)

#### 3.3 Importance × Stability Framework

**[HERO FIGURE HERE - Figure 1]**

```
                    High Importance
                         │
      🟡 Noisy Signal    │    🟢 Stable Signal
      (Collect data)     │    (Data sufficient)
   ──────────────────────┼──────────────────────
      🔴 Pure Noise      │    ⚪ Irrelevant
      (Remove FK)        │    (Ignore)
                         │
                    Low Importance
```

**X-axis**: FK Importance (% accuracy change when permuted)
**Y-axis**: FK Uncertainty Contribution (% variance change when permuted)

---

### 4. Experiments (2.5 pages)

#### 4.1 Datasets and Tasks

| Domain | Dataset | Tasks | Entity |
|--------|---------|-------|--------|
| Racing | rel-f1 | driver-position, driver-dnf, driver-top3 | drivers |
| Clinical | rel-trial | study-outcome, study-adverse, site-success | studies |
| ERP | rel-salt | item-plant, item-shippoint, sales-payterms | sales items |
| Classifieds | rel-avito | ad-ctr, user-clicks | ads/users |

**Total**: 4 domains, 11 tasks, 45 FK-task combinations

#### 4.2 Results

**[Table 1: Summary of actionable insights across all tasks]**

| Task | Most Noisy FK | Uncertainty | Importance | Action |
|------|---------------|-------------|------------|--------|
| study-outcome | OUTCOME_ANALYSES | +17% | +153% | Standardize methods |
| ad-ctr | SEARCHSTREAM | +10% | +148% | User behavior data |
| sales-payterms | SALESDOCUMENT | +12% | +55% | Sales metadata |
| driver-position | STANDINGS | +11% | +2% | Low priority |
| driver-top3 | All stable | <-50% | - | Data sufficient |
| ... | ... | ... | ... | ... |

**[Figure 2: 2D scatter plot across all domains]**
- Shows clear clustering of noisy vs stable FKs
- Demonstrates framework works across diverse domains

#### 4.3 Multi-Seed Robustness

- All experiments run with 5 seeds (42-46)
- Results show consistent FK rankings across seeds
- Standard deviations reported for all measurements

---

### 5. Discussion (1 page)

#### 5.1 When to Collect More Data

**Rule**: If FK has high importance (>10%) AND positive uncertainty contribution (>5%), invest in that FK's data.

**Examples**:
- OUTCOME_ANALYSES: +17% uncertainty, +153% importance → HIGH PRIORITY
- STANDINGS: +11% uncertainty, +2% importance → LOW PRIORITY (not important enough)

#### 5.2 Task-Specificity as a Feature

The same FK can be noisy for one task and stable for another. This is valuable information:
- SALESDOCUMENT: +12% for sales-payterms, -66% for item-plant
- RESULTS: +8% for driver-dnf, -316% for driver-top3

**Implication**: Task-specific data investment strategies are necessary.

#### 5.3 Limitations

1. **Requires ensemble training**: Computational overhead for uncertainty estimation
2. **Permutation-based**: May miss interactions between FKs
3. **No causal claims**: Observational decomposition only

---

### 6. Conclusion (0.5 pages)

- FK-level uncertainty decomposition provides actionable data investment guidance
- Framework validated across 4 domains, 11 tasks
- Future work: causal validation via actual data collection experiments

---

## Figures Summary

| Figure | Content | Location |
|--------|---------|----------|
| **Figure 1** | 2D Importance × Stability framework (hero figure) | Section 3.3 |
| **Figure 2** | 2D scatter plot with all 45 FK points | Section 4.2 |
| Table 1 | Actionable insights summary | Section 4.2 |
| Table 2 | Dataset/task overview | Section 4.1 |

---

## Supplementary Material

- Full FK contributions for all 11 tasks (Appendix A)
- Multi-seed raw results (Appendix B)
- Code and reproducibility (GitHub link)

---

## Timeline to NeurIPS 2026

| Week | Task |
|------|------|
| 1 | ✅ 4-domain validation complete |
| 1 | ✅ 2D visualization created |
| 1 | ✅ Paper outline complete |
| 2-4 | Write introduction + related work |
| 5-8 | Write method + experiments |
| 9-12 | Internal review + revision |
| 13-16 | Additional experiments if needed |
| 17-20 | Final polish + submission |

**Deadline**: ~May 2026

---

## Reviewer Anticipation

| Concern | Response |
|---------|----------|
| "Just grouped permutation importance" | We measure UNCERTAINTY, not accuracy. Plus 2D framework provides new insight. |
| "Limited novelty" | First to connect UQ with relational structure for data investment. |
| "No real data collection experiment" | Acknowledged as limitation; would strengthen paper if done. |
| "Results are task-specific" | Task-specificity IS the insight—no universal FK categories. |

---

## Success Probability

**Current estimate**: 60-70% NeurIPS acceptance

**Factors**:
- ✅ Novel framework (FK-level UQ)
- ✅ 4 domains, 11 tasks (strong validation)
- ✅ Clear actionable guidance
- ✅ Honest about limitations
- ⚠️ Incremental (not paradigm-shifting)
- ⚠️ No real data collection validation (yet)

**Backup**: KDD 2026 (85% acceptance if NeurIPS rejected)

---

*Outline created: 2025-12-23*
*Status: Ready for writing*
