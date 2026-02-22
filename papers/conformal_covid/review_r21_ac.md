# R21 -- Senior Area Chair Final Verdict

**Paper:** "Diagnosing Conformal Prediction Failures Under Distribution Shift: A COVID-19 Case Study"
**Venue:** UAI 2026 (42nd Conference on Uncertainty in Artificial Intelligence)
**Round:** R21

---

## Verdict: ACCEPT

**Overall Recommendation:** Accept. The paper is ready for submission.

---

## Scores

| Criterion | Score (1--10) | Justification |
|---|---|---|
| **Novelty** | 7 | The core idea -- using feature importance *structure* (concentration) to predict conformal failure *severity* pre-deployment -- is genuinely new. The shift-detection-vs-severity-prediction distinction is well-articulated. Not a methodological breakthrough (no new algorithm), but a practically valuable diagnostic insight with theoretical grounding. |
| **Technical Soundness** | 7 | Statistical protocol is thorough: 50-seed ensemble, paired Wilcoxon, bootstrap CIs, ICC for pseudo-replication, LOO stability, Holm-Bonferroni correction, power analysis. Theorem 1 operates under an idealized additive model (A1 is acknowledged as an approximation from probability-space vs log-odds), but the directional claim is validated empirically. The n=16 partial correlation resolving the class-cardinality confound (rho_partial = 0.771, p = 0.0008) is convincing. The KR-correction caveat for the mixed-effects model with 8 clusters is correctly noted. |
| **Significance** | 7 | Addresses a genuine practitioner gap: shift detectors (MMD, C2ST) tell you *whether* shift exists but not *which* models will fail. GBT models dominate tabular production ML, so the scope restriction is practically relevant rather than limiting. The diagnostic is zero-cost (TreeSHAP is standard). |
| **Presentation** | 7 | Dense but well-organized. Contributions condensed to 3 (appropriate). Limitations are stated honestly: threshold is exploratory (n=8), retraining is single-seed with Holm-corrected p=0.11, protective-factor rule from n=1. Abstract is long (~220 words) but UAI has no enforced limit and the density is justified. |

**Aggregate: 7.0 / 10** -- solid accept for UAI.

---

## 1. Page Limit Compliance

**PASS.** The compiled PDF shows 8 pages of main body content (Sections 1--8, including abstract, contributions, acknowledgements). References begin on page 8 and continue onto pages 9--10. The appendix (supplementary material) starts on page 11. This complies with UAI 2026's hard 8-page main body limit.

## 2. Novelty Assessment

The SHAP concentration diagnostic is novel in the conformal prediction literature. Prior work falls into two categories: (a) shift detection methods (MMD, C2ST, PSI) that detect shift uniformly but cannot predict severity, and (b) adaptive methods (ACI, RAPS, ECP/EACP) that adapt at deployment time. No prior work uses feature importance *structure* as a pre-deployment severity predictor.

Key differentiators:
- Kasa and Taylor (2023) empirically characterize CP degradation across architectures but do not propose a pre-deployment diagnostic.
- Gardner et al. (2023, TableShift) benchmark tabular shift but do not connect feature importance to CP failure.
- Garg et al. (2022) predict accuracy degradation using unlabeled test data -- SHAP concentration requires no test observations.
- Kasa et al. (2025, ECP/EACP) adapt at deployment time; SHAP concentration is computed pre-deployment.

The novelty is practical-diagnostic rather than algorithmic, which is appropriate for an applied contribution. The connection to Theorem 1 (monotone score inflation under concentrated feature dependence) provides mechanistic grounding beyond pure empirical observation.

**Assessment: Adequately novel for UAI.**

## 3. Statistical Rigor

The statistical analysis is thorough and appropriately hedged:

**Strengths:**
- 50-seed ensemble with paired Wilcoxon tests for within-task significance (all p <= 0.005).
- Bootstrap CIs with percentile method explicitly noted; BCa mentioned but not used (justified at n=8).
- ICC analysis (0.675, CI [0.47, 0.90]) properly addresses pseudo-replication concern; effective n_eff = 11.7 exceeds 8 tasks.
- Power analysis is honest: n=8 within-SALT labeled "exploratory" (power ~0.76); n=16 cross-domain labeled "confirmatory" (power >0.99).
- Holm-Bonferroni correction applied for 5 concentration metrics; boundary result (adjusted p=0.050) transparently reported.
- n=16 partial correlation resolves class-cardinality confound: rho_partial(conc.) = 0.771, p = 0.0008 vs rho_partial(log K) = -0.010, p = 0.97.
- Mixed-effects model with KR-correction caveat for 8 clusters (Wald p=0.0006 anti-conservative; estimated KR p ~ 0.01-0.03; CI [0.70, 2.58] excludes zero regardless).

**No statistical concerns remain.** The primary result (rho = 0.853, p < 0.001, n=16) is well-supported with appropriate robustness checks.

## 4. Experimental Design

**Sound.** The design has several strengths:

- **No cherry-picking:** All 8 SALT classification tasks included. The K >= 4 multiclass inclusion criterion is grounded in APS mechanics (binary prediction sets have a structural ceiling), not post-hoc.
- **External validation:** 9 datasets across 9 non-supply-chain domains, with 4 documented-shift (Covertype, Gas Sensor, KDDCup99, PAMAP2) and 4 null-shift controls (Shuttle, Avila, Pendigits, Satimage). The controls are informative: the framework correctly predicts robust coverage under null shift.
- **Seed protocol:** 50 seeds for SALT, 10 seeds for external. Adequate for the claims made.
- **Placebo test:** Pre-COVID temporal split shows 6--143x lower degradation, confirming the diagnostic is informative under genuine event-driven shift.
- **Model sensitivity:** LightGBM, CatBoost, XGBoost, RF, MLP -- the GBT-specificity is identified as a structural finding with mechanistic explanation (bagging dilution vs sequential reinforcement), not hidden as a limitation.
- **Calibration split:** Deterministic first-half/second-half documented in Appendix A. The non-random split preserves temporal order; the same split is applied uniformly across all tasks.

**Key acknowledged limitation:** KDDCup99 is a false negative (C=21.1%, drop=15.9 pp, seed-dependent). The paper handles this honestly: it is flagged as the "principal intermediate-regime false negative" and motivates the seed-stability protocol in the decision framework.

## 5. Theorem 1

**Adequately stated.** Assumptions (A1)--(A3) are explicit:
- (A1) Additive decomposition in probability space -- the footnote correctly notes that TreeExplainer operates in log-odds space, making this an idealization. The theorem provides directional intuition, not a tight guarantee.
- (A2) Concentrated misclassification -- epsilon < 1/K, a reasonable formalization of "the top feature shifts OOD."
- (A3) Residual exchangeability -- non-shifted features maintain distributional equivalence.

The four-part result (pointwise bound, expected inflation, monotone vulnerability, coverage degradation) is clean. Conservative bounds verified on all 5 applicable tasks. The proof sketch is adequate; full derivation in the appendix.

The key honest caveat -- probability-space additivity vs log-odds SHAP computation -- is prominently footnoted rather than buried. The empirical validation (rho=0.853) provides the primary evidence; the theorem provides mechanistic scaffolding.

## 6. Scope Clarity: GBT-Specificity

**Presented as a structural finding, not hidden.** Section 3.6 ("Why gradient-boosted models?") devotes a full paragraph to explaining the mechanism: sequential boosting reinforces single-feature dependence while bagging (RF) dilutes it and MLP-SHAP approximates noisily. The model-sensitivity table (Appendix L) shows the gradient: LGB rho=0.833 > CatBoost 0.667 > XGBoost 0.548 > MLP 0.43 > RF 0.30.

The mixed-effects analysis across 3 GBT models (beta_1 = 1.64, CI excluding zero) and the cross-model non-transferability observation (LGB concentration does not predict CatBoost drop: rho=0.07) are both clearly reported. The conclusion explicitly states the diagnostic "requires prospective validation beyond boosting models."

This is handled appropriately -- the scope is presented as the natural boundary of the finding rather than concealed.

## 7. Related Work Coverage

**Comprehensive.** The paper cites 31 references spanning:
- Core CP: Vovk (2005), Shafer & Vovk (2008), Romano et al. (2019, 2020), Lei et al. (2018), Angelopoulos & Bates (2023)
- CP under shift: Tibshirani et al. (2019), Podkopaev & Ramdas (2021), Barber et al. (2023), Kasa & Taylor (2023), Gibbs et al. (2025)
- Adaptive methods: Gibbs & Candes (2021, 2024), Zaffran et al. (2022), Feldman et al. (2023), Bhatnagar et al. (2023), Angelopoulos et al. (2024), Kasa et al. (2025)
- Shift detection/benchmarks: Koh et al. (2021, WILDS), Malinin et al. (2021, Shifts), Gulrajani & Lopez-Paz (2021), Gardner et al. (2023, TableShift), Gretton et al. (2012), Lopez-Paz & Oquab (2017)
- Class-conditional: Ding et al. (2023), Cauchois et al. (2021)
- Interpretability: Lundberg & Lee (2017), Lundberg et al. (2018, 2020)
- OOD generalization: Miller et al. (2021), Garg et al. (2022)

**No glaring omissions.** The CP-under-shift literature is well-covered. One could argue for citing Bates et al. (2021, "Distribution-Free, Risk-Controlling Prediction Sets") or Cauchois et al. (2024) on conditional coverage, but these are marginal additions that would not change the paper's positioning.

**BibTeX quality:** All 31 entries appear correctly formatted. Key corrections from earlier rounds (kasa2023 authors, kasa2025 UAI PMLR entry, gibbs2025 JRSS-B journal name, lundberg2018consistent) are present and verified. The lundberg2018consistent entry remains an arXiv preprint (arXiv:1802.03888) rather than citing the published NeurIPS 2018 version -- this is a minor bibliographic imprecision but does not affect the paper's technical content.

## 8. Reproducibility

**PASS.** The paper provides:
- Code repository URL (https://github.com/ChorokLeeDev/conformal-covid) in both Acknowledgements and Appendix A.
- Full LightGBM hyperparameters (Appendix A.1): objective, boosting, num_leaves, learning_rate, feature_fraction, bagging_fraction, bagging_freq, num_boost_round, early_stopping_rounds, seed range.
- Conformal prediction setup: APS, alpha=0.1, deterministic first-half/second-half calibration split, quantile formula.
- SHAP computation: TreeExplainer, 10K subsample, mean absolute aggregation.
- External dataset protocols (Table 7): train/test sizes, split mechanisms, shift types for all 9 external datasets.
- Computational resources: standard CPU, 3-4 hours wall-clock.

## 9. Contributions Delivered

| Stated Contribution | Delivered? | Evidence |
|---|---|---|
| Pre-deployment diagnostic (SHAP concentration) | Yes | rho=0.853, p<0.001, n=16; LOO stability; cross-domain transfer; model-sensitivity analysis |
| Formal theory + comparative evidence | Yes | Theorem 1 with 4-part result; conservative bounds verified 5/5; MMD/C2ST/PSI comparison (rho<=0.19) |
| Operational framework (exploratory) | Yes | 3-step protocol; threshold sensitivity analysis; protective-factor check; retraining analysis (appropriately hedged) |

All three contributions are delivered with appropriate scope qualifiers.

## 10. Abstract Accuracy

The abstract is accurate and complete. Every quantitative claim (rho=0.853, p<0.001, n=16, bootstrap CI [0.50,0.96], Kendall tau=0.667, coverage drops 0.1%-77.1%, all paired p<=0.005, 50 seeds, rho<=0.19 for shift detectors, Covertype C=49.8% with 81.8pp drop, +19pp retraining with p=0.036 unadjusted and Holm p=0.11) matches the main text and tables. The abstract correctly scopes the diagnostic to "gradient-boosted models" and labels the decision framework as "exploratory."

The abstract is dense (~220 words) but UAI does not enforce an abstract word limit.

---

## Blocking Issues

**None.**

---

## Minor Suggestions (Non-blocking)

1. **lundberg2018consistent BibTeX entry:** Still listed as "arXiv preprint arXiv:1802.03888" rather than the published NeurIPS 2018 workshop/proceedings version. Cosmetic only -- does not affect technical content.

2. **Abstract length:** At ~220 words the abstract is longer than typical UAI submissions. Consider whether the Stack Overflow exclusion detail and the retraining effect could be moved to the introduction to tighten the abstract. Stylistic preference only.

3. **KDDCup99 as a known failure mode:** The paper acknowledges this false negative honestly, but a brief sentence in the conclusion about what *class* of scenarios the diagnostic misses (intermediate-concentration, high-variance outcomes) could help practitioners calibrate expectations. Already partially addressed in the decision framework's seed-stability protocol.

---

## Comparison with R20 Review

The R20 AC review reached the same ACCEPT verdict. No new issues have emerged in R21. The paper has been stable since R19, with R19/R20 confirming all prior fixes were incorporated. The current R21 reading confirms:

- All R12 BibTeX corrections (kasa2023, kasa2025, gibbs2025, adebayo2018 removal) are present.
- All R18/R19 fixes (PSI footnote, TreeSHAP exactness claim, lundberg2018consistent citation, gibbs JRSS-B journal name) are incorporated.
- No textual regressions or introduced errors detected.

---

## Final Statement

This paper makes a clear, well-scoped contribution: a pre-deployment diagnostic that predicts conformal prediction failure *severity* rather than merely detecting shift presence. The statistical analysis is rigorous with appropriate robustness checks, limitations are honestly stated, and the scope (gradient-boosted classifiers) is presented as a structural finding with mechanistic explanation. The n=16 cross-domain correlation (rho=0.853, p<0.001) provides confirmatory evidence beyond the exploratory SALT analysis. The paper is technically sound, practically relevant, and ready for submission.

**Decision: ACCEPT. Submit to UAI 2026.**
