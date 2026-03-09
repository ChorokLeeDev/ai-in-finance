# R20 — Senior Area Chair Final Verdict

**Paper:** "Diagnosing Conformal Prediction Failures Under Distribution Shift: A COVID-19 Case Study"
**Venue:** UAI 2026
**Round:** R20 (Final)

---

## Verdict: ACCEPT

**Recommendation:** Accept. No further revision required before submission.

---

## Summary Assessment

This paper proposes SHAP concentration — the fraction of feature importance concentrated in the top feature — as a pre-deployment diagnostic for conformal prediction vulnerability under distribution shift. Using COVID-19 as a natural experiment across 8 supply chain tasks plus 8 external multiclass datasets (16 tasks, 9 domains), the authors demonstrate a strong Spearman correlation ($\rho = 0.853$, $p < 0.001$) between SHAP concentration and coverage degradation severity. The paper includes a formal score-inflation theorem, systematic comparison against shift detectors (MMD, C2ST, PSI) that detect shift uniformly but cannot predict severity, model-class sensitivity analysis explaining why the diagnostic is specific to gradient-boosted models, and an exploratory decision framework.

---

## Evaluation Against UAI Standards

### Novelty: Strong
The core idea — using feature importance *structure* (not magnitude) to predict conformal failure *severity* (not just shift presence) before deployment — is genuinely novel. The distinction between "shift detection" and "failure severity prediction" is well-articulated and the paper demonstrates clearly that standard shift detectors fail at this task ($\rho \leq 0.19$). The connection between SHAP concentration and the APS scoring mechanism via Theorem 1 provides mechanistic grounding.

### Technical Soundness: Satisfactory
- **Statistical protocol:** 50-seed ensemble with paired Wilcoxon tests, bootstrap CIs (percentile method noted), LOO stability analysis, ICC for pseudo-replication assessment. The power analysis is honest: $n = 8$ within-SALT is labeled exploratory; $n = 16$ cross-domain is the confirmatory endpoint. The Holm-Bonferroni correction for multiple concentration metrics is applied and the boundary result ($p = 0.050$) is transparently reported.
- **Theorem 1:** The idealized additive model assumption (A1) is an acknowledged approximation (footnote clarifies probability-space vs. log-odds gap). The theorem provides directional intuition verified empirically rather than a tight guarantee — appropriate for this setting. Conservative bounds are verified on all 5 applicable tasks.
- **Partial correlation:** The class cardinality confound is properly addressed. Within SALT ($n = 8$), partial correlations are non-significant as expected at low power. The $n = 16$ partial correlation ($\rho_{\text{partial}} = 0.771$, $p = 0.0008$ for concentration; $\rho_{\text{partial}} = -0.010$ for $\log K$) resolves this convincingly.
- **Mixed-effects:** The KR-correction caveat for 8 clusters is correctly noted ($p \approx 0.01$--$0.03$ vs. Wald $p = 0.0006$); the CI excluding zero is the primary evidence.

### Experimental Design: Good
- **SALT:** All 8 classification tasks included (no cherry-picking), identical temporal split, 50 seeds.
- **External:** 9 datasets spanning 4 documented-shift and 4 null-shift controls (correctly categorized as DS/NC in Table 7), plus Stack Overflow excluded from the primary endpoint with clear justification ($K = 3$ ceiling effect from APS mechanics). The $K \geq 4$ inclusion criterion is grounded in APS structure, not post-hoc.
- **Model sensitivity:** LightGBM, CatBoost, XGBoost, RF, MLP — the gradient-boosting scope is identified as a structural finding, not a limitation. The "why gradient-boosted models?" paragraph correctly grounds this in the bagging dilution mechanism (RF) vs. sequential reinforcement (GBT), without overclaiming TreeSHAP exactness as the sole explanation.
- **Placebo test:** 6-143x lower degradation pre-COVID confirms the diagnostic is informative under genuine shift.
- **Calibration split:** Deterministic first-half/second-half documented in Appendix A.

### Clarity and Presentation: Good
- Contributions condensed to 3 (from earlier 6), appropriately scoped.
- Limitations are stated rather than hidden: threshold is exploratory ($n = 8$), retraining effect is single-seed with Holm-corrected $p = 0.11$, protective-factor rule is from $n = 1$ observation, KDDCup99 is an acknowledged false negative.
- Abstract is dense but accurate; all claims are quantified and hedged appropriately.
- Code repository provided.

### Significance: Moderate-to-High
The paper addresses a practical gap: practitioners deploying conformal prediction need to know *which* models will fail, not just *whether* shift exists. The diagnostic is computed pre-deployment on validation data with zero additional cost (TreeSHAP is already standard). The scope restriction to gradient-boosted models is honest but also practically relevant — LightGBM/XGBoost/CatBoost dominate tabular ML in production.

---

## Remaining Observations (Non-blocking)

1. **s-group sub-nominal coverage (83.6%):** Noted in the table footnote as expected at $K = 459$ per Ding et al. (2023). Adequate.

2. **Validation coverage range (83.6%--99.9%):** Correctly reported; the 83.6% lower bound (s-group) is acknowledged as sub-nominal. The Miller et al. contrast (ID accuracy predicts OOD accuracy for images, but not for conformal coverage in tabular settings) is well-placed in Related Work.

3. **PSI footnote position:** Correctly placed at first mention in Contribution 2, with threshold interpretation.

4. **gibbs2025conditional:** Journal name complete (JRSS-B, vol. 87, pp. 1100--1126, 2025). Correct.

5. **lundberg2018consistent:** Present in bibliography, cited appropriately in the "Why gradient-boosted models?" paragraph alongside lundberg2020local.

6. **Code repo link:** Appears in both Acknowledgements and Appendix A. Sufficient.

7. **APS expanded in abstract:** "Adaptive Prediction Sets (APS)" — confirmed present.

8. **Minor observation (not blocking):** The abstract runs long (~220 words). UAI does not enforce an abstract word limit, so this is stylistic only. The density is justified given the number of quantitative claims.

---

## Checklist

| Criterion | Status |
|---|---|
| Page limit (8 main body + unlimited refs/appendix) | PASS |
| Contributions clearly stated | PASS (3 contributions) |
| Claims supported by evidence | PASS |
| Limitations acknowledged | PASS |
| Statistical protocol sound | PASS |
| Related work adequate | PASS (35 references, key CP/shift/interpretability covered) |
| Reproducibility | PASS (code repo, hyperparameters, seeds, split protocol) |
| No fabricated citations | PASS (R12 corrections verified: kasa2023, kasa2025, gibbs2025) |
| Prior R18/R19 fixes incorporated | PASS (all items confirmed in text) |

---

## Final Statement

The paper makes a clear, well-scoped contribution to the conformal prediction literature: a pre-deployment diagnostic that predicts failure *severity* rather than merely detecting shift presence. The statistical analysis is thorough, limitations are honest, and the scope (gradient-boosted classifiers) is identified as a structural finding with mechanistic explanation rather than hidden as a weakness. The $n = 16$ cross-domain correlation ($\rho = 0.853$, $p < 0.001$) provides confirmatory evidence beyond the exploratory SALT analysis.

**Decision: Accept. Submit without further revision.**
