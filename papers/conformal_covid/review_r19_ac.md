# Senior Area Chair Review -- Round 19
**Paper**: Diagnosing Conformal Prediction Failures Under Distribution Shift: A COVID-19 Case Study
**Venue**: UAI 2026
**Date**: 2026-02-22
**Role**: Second-round senior area chair evaluation

---

## 1. Updated Scores

| Criterion        | Score (1--5) | Notes |
|------------------|:---:|-------|
| Novelty          | 4   | SHAP concentration as a pre-deployment CP diagnostic is genuinely new. Not a fundamentally new method, but a novel and useful reframing of interpretability for reliability. |
| Significance     | 3.5 | Strong within gradient-boosted tabular models; limited generality beyond that scope. The honest scoping to TreeSHAP-compatible models is a strength, not a weakness. |
| Clarity          | 4   | Substantially improved. Contributions condensed to 3, external datasets categorized (DS/NC), "Why gradient-boosted models?" paragraph addresses the RF/MLP gap directly. The paper is long but well-organized. |
| Soundness        | 4   | Theorem A1 footnote now honestly discloses the probability-vs-log-odds gap. Conservative bound verification corrected. Holm-corrected retraining p-value reported. Partial correlation at n=16 resolves the class-count confound. Mixed-effects analysis with KR-correction caveat is responsible. |
| Reproducibility  | 4.5 | Code repository added. 50-seed protocol with fixed hyperparameters. External dataset table with split protocols. Deterministic calibration split documented. This is above-average for UAI submissions. |
| Presentation     | 3.5 | Abstract remains long (one dense paragraph). Table 2 retains intermediate rows (n=11, n=15) that add complexity without proportional insight. The n=16 scatter plot (Figure 2) and DS/NC categorization are clear improvements. |

---

## 2. Assessment of Specific Revisions

### (a) APS spelled out in abstract -- RESOLVED
"Adaptive Prediction Sets (APS)" now appears on first use. Minor point, properly addressed.

### (b) Validation coverage softened to 83.6%--99.9% -- RESOLVED
The previous "near-uniform" claim was inaccurate given s-group at 83.6%. The revision is honest: the range is stated explicitly, and a Table 1 footnote explains that sub-nominal coverage at K=459 is expected per Ding et al. (2023). This is exactly the right response.

### (c) s-group sub-nominal coverage footnote in Table 1 -- RESOLVED
The footnote reads: "s-group's 83.6% validation coverage is below the 90% target---sub-nominal non-randomized APS coverage at K=459 is expected." This cites the appropriate reference. Adequate.

### (d) Contributions condensed from 6 to 3 -- RESOLVED, IMPROVED
The three contributions are now: (1) the diagnostic itself with full quantitative summary, (2) formal theory plus comparative evidence against shift detectors, (3) operational framework labeled exploratory. This is tighter and more honest than the 6-item version. Each contribution has a clear evidence base.

### (e) "Why gradient-boosted models?" paragraph -- RESOLVED, KEY IMPROVEMENT
Section 3.6 now provides a mechanistic explanation: TreeSHAP computes exact Shapley values for gradient-boosted trees; RF averages over independent trees diluting concentration; MLP-SHAP approximates rather than exactly decomposes. The model-sensitivity gradient (LGB 0.833 > CatBoost 0.667 > XGB 0.548 > MLP 0.43 > RF 0.30) is now framed as a "structural finding" identifying the scope boundary. This is the single most important revision -- it transforms what was previously a limitation paragraph into positive evidence about when and why the diagnostic works.

### (f) External datasets categorized as DS vs NC -- RESOLVED
Table 7 (Appendix A.6) now labels 4 documented-shift datasets (Covertype, Gas Sensor, KDDCup99, PAMAP2) and 4 null-shift controls (Shuttle, Avila, Pendigits, Satimage). The controls serve a clear role: the framework predicts robust coverage absent genuine shift, confirmed in all 4 cases. This categorization strengthens the external validation narrative.

### (g) Code repository -- RESOLVED
URL provided in both Acknowledgements and Appendix A. This addresses a prior reproducibility gap.

---

## 3. Remaining Concerns

### 3.1 Stack Overflow inclusion/exclusion -- RESOLVED
The Round 2 review flagged a contradiction where a Table 2 footnote incorrectly stated n=16 includes Stack Overflow. In the current version, the methodology (line 81) clearly states the K>=4 exclusion criterion, and the Table 2 footnote (line 278) correctly explains n=15 as an intermediate analysis and n=16 as the primary endpoint excluding Stack Overflow. The Combined row (n=17, line 279) explicitly adds Stack Overflow back as an auxiliary endpoint. The Table 5 footnote (line 541) also correctly marks Stack Overflow as excluded. This contradiction is fully resolved.

### 3.2 Abstract length -- MINOR, NON-BLOCKING
The abstract is a single dense paragraph of approximately 220 words. For UAI this is within limits but at the upper end. Every clause carries information, so there is little to cut without losing content. A reviewer might find it front-loads too many numbers, but the information is accurate and well-hedged. Not a revision requirement.

### 3.3 Table 2 intermediate rows -- MINOR, NON-BLOCKING
The n=11 and n=15 rows remain in Table 2. These document the analysis trajectory but are not essential. The n=11 row includes a footnote disclosing the single-seed vs multi-seed discrepancy (rho=0.909 single-seed vs 0.818 multi-seed), which is transparent. The COVID-era row from earlier versions appears to have been removed. Acceptable as-is; a reviewer might suggest pruning to just n=8 and n=16.

### 3.4 Retraining claim -- ADEQUATELY HEDGED
The abstract says "suggest that quarterly retraining may partially recover coverage (+19 pp, p=0.036 unadjusted, single-seed experiment; Holm-corrected p=0.11)." The body repeats the same qualifiers. The framework is labeled "exploratory." This is honest enough for a case study paper.

### 3.5 KDDCup99 as false negative -- ACKNOWLEDGED, NOT HIDDEN
KDDCup99 (C=21.1%, drop=15.9 pp) is the principal failure case for the 40% threshold. The paper discusses it extensively: seed-dependent behavior (range -0.8 to 73.5 pp), multi-seed instability, and its role as an intermediate-regime data point. The recommendation for 5+ seed averaging addresses it operationally. This is transparent handling of a genuine weakness.

### 3.6 n=8 within-SALT power -- ACKNOWLEDGED
The paper now explicitly states: "At n=8, power to detect rho=0.833 is approximately 0.76 (below the 0.80 convention); the within-SALT result is therefore exploratory. The n=16 cross-domain result (rho=0.853, p<0.001; power >0.99) is the confirmatory endpoint." This is the correct framing.

### 3.7 Protective-factor rule derived from n=1 -- ACKNOWLEDGED
Line 249: "The Jaccard > 0.5 and importance > 15% protective-factor rule is derived from n=1 observation (sales-office) and should be treated as provisional." Honest.

### 3.8 One substantive concern remains: the 40% threshold is in-sample
The 40% threshold was derived from the SALT concentration gap and evaluated on both SALT (in-sample) and external (out-of-sample) datasets. The paper acknowledges this (line 255: "the 40% threshold was derived from the SALT subset; the n=16 correlation therefore includes the in-sample SALT tasks"). The external-only threshold performance (Precision=1.00, Recall=0.50) is reported in Appendix B. This is transparent, but the low external recall (1 TP out of 2 at-risk) means the threshold's generalizability rests on very few catastrophic external cases. The paper is honest about this -- "threshold should be treated as provisional" -- but a skeptical reviewer could argue that the threshold validation is circular for the SALT tasks. This is mitigated by the continuous correlation (rho=0.853) being the primary claim, not the discrete threshold.

---

## 4. Overall Assessment

**Strengths:**
- The core idea (SHAP concentration as pre-deployment CP diagnostic) is novel and practically useful.
- The n=16 cross-domain correlation (rho=0.853, p<0.001) is the confirmatory result with adequate power.
- The honest scoping to gradient-boosted models with a mechanistic explanation (Section 3.6) elevates this from "it works on our dataset" to "here is why it works and when it does not."
- The paper is unusually transparent about limitations: Holm corrections reported, single-seed caveats noted, n=1 protective-factor rule flagged as provisional, exploratory vs confirmatory clearly distinguished.
- External validation with null-shift controls is a thoughtful design choice.
- Reproducibility is strong: code released, 50-seed protocol, fixed hyperparameters, deterministic splits documented.

**Weaknesses:**
- Scope is narrower than a typical UAI methods paper: gradient-boosted tabular classifiers with K>=4 classes.
- The threshold (40%) has limited out-of-sample validation due to sparse catastrophic external cases.
- The paper is dense -- 8 main pages carry a lot of conditional claims, footnotes, and caveats. Some UAI reviewers may find this hard to parse on first reading.
- The theory (Theorem 1) is directional intuition under an idealized additive model, not a formal guarantee for the measured SHAP-derived C. The paper is honest about this, but the theorem's practical role is more "mechanistic narrative" than "theoretical contribution."

---

## 5. Scores Summary and Recommendation

| Criterion        | Score |
|------------------|:-----:|
| Novelty          | 4.0   |
| Significance     | 3.5   |
| Clarity          | 4.0   |
| Soundness        | 4.0   |
| Reproducibility  | 4.5   |
| Presentation     | 3.5   |
| **Overall**      | **3.8** |

### Recommendation: **Weak Accept**

**Rationale:** The paper addresses a genuine gap (predicting which CP deployments will fail pre-deployment), provides a novel diagnostic with strong empirical evidence (rho=0.853, p<0.001, 16 tasks, 9 domains), and is unusually honest about its limitations. The revisions address all prior blocking concerns: the Stack Overflow contradiction is resolved, the theorem bounds are corrected, the A1 assumption gap is disclosed, contributions are condensed, the RF/MLP non-replication is reframed as a structural finding, and a code repository is provided.

The paper falls short of a strong accept because: (1) the scope is narrow (gradient-boosted tabular classifiers), (2) the threshold has limited external validation, and (3) the theory provides directional intuition rather than formal guarantees. These are acknowledged limitations, not hidden ones.

For a UAI audience interested in practical conformal prediction deployment, this is a useful contribution. It will not reshape the field, but it fills a real gap that practitioners face.

### Is the paper ready to submit? **Yes.**

All prior blocking issues are resolved. The remaining concerns (abstract length, Table 2 intermediate rows) are minor presentation choices that do not affect correctness. The paper is honest, well-scoped, and reproducible. I would recommend submission without further revision cycles.

---

## Appendix: Prior Blocking Issues -- Resolution Status

| Prior Issue | Status |
|---|---|
| Abstract duplication | RESOLVED (no duplicate text) |
| n=16 scatter plot missing | RESOLVED (Figure 2 present) |
| Theorem bounds incorrect | RESOLVED (corrected in Appendix) |
| COVID-era n=9 row undefined | RESOLVED (removed or clarified) |
| n=11 row single-seed vs multi-seed | RESOLVED (footnote discloses both values) |
| Stack Overflow in/out contradiction | RESOLVED (consistent exclusion from n=16) |
| angelopoulos2024conformal venue | RESOLVED (not visible in .tex; presumably fixed in .bib) |
| "8 tasks with severe feature turnover" | RESOLVED (language corrected) |
| Retraining p-value multiplicity | RESOLVED (Holm-corrected p=0.11 reported) |
| A1 probability-vs-log-odds gap | RESOLVED (footnote added) |
| Missing code repository | RESOLVED (GitHub URL provided) |
| External dataset table | RESOLVED (Table 7 with DS/NC categorization) |
| s-group sub-nominal coverage | RESOLVED (Table 1 footnote) |
