===============================================================
  TECHNICAL REVIEW REPORT
  UAI 2026: "Diagnosing Conformal Prediction Failures Under
  Distribution Shift: A COVID-19 Case Study" | 2026-02-20
===============================================================

## EXECUTIVE SUMMARY

Three independent agents reviewed this paper's literature positioning, numerical accuracy, and scientific contribution. The paper presents a novel, well-validated pre-deployment diagnostic (SHAP concentration) for conformal prediction failures, with rho=0.853 across 16 multiclass tasks in 9 domains. Core claims are numerically verified. Remaining issues are presentation-level inconsistencies and citation hygiene -- none threaten scientific validity.

## KEY FINDINGS

### Strengths
- Pre-deployment framing is genuinely novel; no prior CP work predicts *which* models fail before test data arrives -- InsightExtractor, LiteratureAgent
- Core statistics verified against raw JSON: correlation, p-values, CIs, bootstrap intervals all match -- MethodCritic
- Theorem 1 (score inflation bound) verified on 5/5 applicable tasks; conservative bound holds -- MethodCritic, InsightExtractor
- Shift detectors (MMD, C2ST, PSI) shown to have full detection power but zero discriminative power for severity (rho <= 0.19) -- a clean, important negative result -- InsightExtractor
- 50-seed ensemble with explicit seed range, exact hyperparameters, public datasets: reproducibility score 7/10 -- MethodCritic

### Concerns
- "11 additional datasets spanning 10 domains" does not match Table 6 (9 datasets visible) -- severity: MED -- MethodCritic (M3)
- Gulrajani citation mischaracterizes the DomainBed finding as "temporal shift" -- severity: MED -- LiteratureAgent
- Single-seed rho=0.909 (n=11) displayed prominently; honest multi-seed value is rho=0.818 -- severity: MED -- MethodCritic (M4)
- Class counts in Table 1 (s-group: 462, s-payterms: 135) differ from 50-seed JSON (459, 137) -- severity: LOW -- MethodCritic (M1)
- NeurIPS BibTeX entry-type inconsistency (@article vs @inproceedings) across 3 entries -- severity: LOW -- LiteratureAgent
- Missing citations: Gibbs & Candes 2024 JMLR, Bhatnagar et al. 2023 ICML -- severity: LOW -- LiteratureAgent
- i-shippoint labeled "ROB*" in Table 1 but "At-risk*" in Table 6 -- severity: LOW -- MethodCritic (M5)

### Critical Issues
None identified by any agent.

## CROSS-AGENT INSIGHTS

All three agents independently converge on the same assessment: the science is sound and the contribution is novel, but presentation-level inconsistencies (dataset counting, single-seed vs multi-seed display, class counts) create unnecessary attack surface for reviewers. The LiteratureAgent and InsightExtractor agree that the pre-deployment framing is the paper's strongest differentiator. MethodCritic's numerical verification confirms the statistical claims hold up under scrutiny.

## REQUIRED ACTIONS (before submission)

1. Reconcile "11 additional datasets / 10 domains" with Table 6 showing 9 -- either list all 11 explicitly or correct the count (MethodCritic M3)
2. Fix Gulrajani citation: change "robust methods often fail under temporal shift" to accurately describe the DomainBed/ERM finding (LiteratureAgent)
3. Replace single-seed rho=0.909 with multi-seed rho=0.818 in the stratified correlation table, or remove the n=11 single-seed row entirely (MethodCritic M4)

## SUGGESTED IMPROVEMENTS (non-blocking)

- Standardize NeurIPS BibTeX entries to @inproceedings (LiteratureAgent)
- Add Gibbs & Candes 2024 JMLR and Bhatnagar et al. 2023 ICML citations (LiteratureAgent)
- Fix class count discrepancies in Table 1 vs JSON files (MethodCritic M1)
- Add PSI footnote noting no canonical academic citation (LiteratureAgent)
- Change "default LightGBM settings" to list actual hyperparameters without "default" (MethodCritic B6)
- Reconcile Table 1 vs Table 6 labeling for i-shippoint (MethodCritic M5)

## TOP 3 ANTICIPATED REVIEWER QUESTIONS

**Q1: "The 40% threshold is post-hoc. How do you justify applying it to external data?"**
Suggested response: The threshold is explicitly labeled as exploratory (derived on n=8 SALT tasks). The cross-domain validation (9 external datasets, 9 domains) is a held-out test of the rank correlation, not the threshold. The rho=0.853 across n=16 tasks is a non-parametric rank measure that does not depend on the threshold value. The threshold is offered as a practical starting point, not a universal constant.

**Q2: "The diagnostic works for boosting (rho=0.55-0.83) but not RF (0.30) or MLP (0.43). Is this just a boosting artifact?"**
Suggested response: Model specificity is a feature, not a bug. The diagnostic is explicitly model-specific -- it measures how a *particular model's* learned dependence structure interacts with shift. RF smooths probability surfaces via bagging, reducing concentration's predictive power. MLP failures can arise from global sensitivity patterns not captured by top-1 SHAP. The paper discloses these limitations and frames SHAP concentration as a diagnostic for the dominant deployed model class (gradient-boosted trees), not a universal law.

**Q3: "Stack Overflow exclusion from n=16 weakens rho from 0.853 to 0.654 if included. Isn't this cherry-picking?"**
Suggested response: Stack Overflow has K=3 classes with a ceiling effect (near-binary structure). Its exclusion is principled and documented in Appendix D with the sensitivity analysis. The n=17 rho=0.654 remains significant. The paper's primary endpoint (n=16 multiclass, >=4 classes) is pre-specified, not adjusted after seeing results.

===============================================================
  VERDICT: CONDITIONAL RECOMMEND (SUBMIT AFTER 3 FIXES)
  Reason: Science is sound and contribution is novel, but three
  presentation inconsistencies (dataset count, Gulrajani citation,
  single-seed rho display) create avoidable reviewer objections.
  All three fixes require < 30 minutes of editing.
===============================================================
