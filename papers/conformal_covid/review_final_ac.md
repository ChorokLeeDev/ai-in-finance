# UAI 2026 Area Chair Review

**Paper:** "Diagnosing Conformal Prediction Failures Under Distribution Shift: A COVID-19 Case Study"

**Recommendation: Weak Accept**

**Confidence: 4/5 (high)**

---

## Summary

The paper proposes SHAP concentration --- the fraction of total feature importance concentrated in the top feature --- as a pre-deployment diagnostic for predicting conformal prediction failure severity under distribution shift. Using 8 supply chain classification tasks experiencing COVID-19-induced temporal shift, plus 8 external multiclass datasets, the authors show that SHAP concentration correlates with coverage degradation (Spearman rho=0.853, p<0.001, n=16) while standard shift detectors (MMD, C2ST) detect shift uniformly but cannot distinguish catastrophic from robust outcomes. A formal score-inflation theorem under an additive decomposition model provides theoretical grounding.

---

## 1. Novelty and Significance of Contribution

**Strengths:**
- The core idea --- that feature importance concentration predicts conformal failure severity before deployment --- is genuinely novel. Prior work has focused on detecting shift or adapting to it; predicting *which models will fail* from structural properties is a new angle.
- The "shift detection is not severity prediction" insight (Section 5.3) is well-demonstrated and practically important. The rho <= 0.19 for MMD/C2ST vs rho=0.853 for SHAP concentration is a clean separation.
- The observation that catastrophic tasks become *more confident* (decreasing entropy) rather than more uncertain is counterintuitive and valuable (Appendix E).

**Weaknesses:**
- The diagnostic is specific to gradient-boosted models. RF (rho=0.30) and MLP (rho=0.43) do not replicate. This substantially limits the scope of the contribution --- the paper's title and abstract should perhaps more prominently signal this restriction.
- The 40% threshold is derived from n=8 SALT tasks (in-sample) and applied to external datasets. While the cross-domain correlation holds, the threshold itself remains exploratory. The practical decision framework (Section 7) is therefore preliminary.
- The contribution is closer to an empirical observation than a general methodology. The question of *why* top-1 concentration specifically (and not HHI or entropy) is diagnostic remains somewhat ad hoc, explained post-hoc by the sales-office counterexample.

**Assessment:** Moderately novel. The pre-deployment diagnostic framing is new and useful, but the restriction to boosting models and the small task-level sample size temper significance.

---

## 2. Clarity of Exposition and Logical Flow

**Strengths:**
- The paper is exceptionally well-organized. The flow from problem (Section 1) to mechanism (Section 4) to evidence (Section 5) to application (Section 7) is logical and easy to follow.
- Statistical qualifications are thorough: bootstrap CIs, Holm corrections, power analyses, jackknife stability, ICC analyses. The authors are unusually transparent about the limitations of small-n inference.
- The abstract accurately summarizes the paper's claims and evidence, with appropriate hedging ("exploratory rule," "suggestive effect").

**Weaknesses:**
- The contributions list in the introduction (6 items) is too long for a paper of this scope. Items 3-6 could be condensed. Having 6 "contributions" dilutes the message; 3 would be more impactful.
- Section 2 (Related Work) is dense and could benefit from a more structured comparison table rather than running text paragraphs.
- The interplay between the n=8 within-SALT and n=16 cross-domain results is sometimes confusing. The paper would benefit from consistently foregrounding the n=16 result as primary (which it does in some places but not others).

**Assessment:** Above average clarity. Minor structural improvements possible.

---

## 3. Strength of Empirical Evidence

**Strengths:**
- 50-seed protocol with paired Wilcoxon tests is rigorous for the within-SALT analysis.
- External validation across 9 non-supply-chain datasets with 10 seeds each is a genuine out-of-sample test.
- The ICC analysis (Appendix F) properly addresses pseudo-replication concerns.
- Multiple baselines (native FI, ensemble disagreement, MMD, C2ST, PSI, entropy, ECE) are compared.
- Placebo test (pre-COVID split) strengthens the COVID-specific claim.

**Weaknesses:**
- **Core concern: n=16 is still small.** The primary endpoint has only 16 data points for the Spearman correlation. While statistically significant, the bootstrap CI [0.50, 0.96] is wide. A single additional outlier could substantially change the result.
- **Selection of external datasets.** The 9 external datasets are all UCI benchmarks with varying shift mechanisms (some documented, some just random splits labeled as "shift"). Five UCI datasets use pre-defined or random splits where the "shift" is not a documented real-world distributional change. This weakens the external validation --- the paper acknowledges this ("shift mechanisms for 5 UCI pre-defined-split datasets are not separately documented") but could be more explicit about how this affects interpretability.
- **Calibration split.** The deterministic first-half/second-half split (not random) is unusual and potentially problematic if validation data has temporal ordering. The paper notes this preserves temporal order but does not discuss whether this introduces systematic bias.
- **Retraining analysis.** Single-seed, Holm-corrected p=0.11. This is essentially inconclusive and should not be presented as a contribution or framework element.
- **KDDCup99 false negative.** The framework misclassifies the one intermediate-regime external dataset. With only 2 catastrophic external cases (Covertype, arguably KDDCup99), the external threshold validation rests on very few positive examples.
- **Partial correlation at n=16.** The partial correlation controlling for log(K) at n=16 (rho_partial=0.771, p=0.0008) is the strongest evidence for concentration over class count. However, this is still a 16-point partial correlation with 13 degrees of freedom. The claim is directionally convincing but the precision is limited.

**Assessment:** Evidence is carefully collected and honestly reported. The main limitation is the small effective sample size (n=16 tasks), which is inherent to the case-study design.

---

## 4. Reproducibility Concerns

**Strengths:**
- Appendix A provides full hyperparameters, seed ranges, software versions, and computational resources.
- SHAP computation details (subsample size, aggregation method) are specified.
- The SALT dataset is publicly available via RelBench.
- External dataset protocols are documented in Table 7.

**Concerns:**
- **Code availability.** No code repository is mentioned. For a paper proposing a diagnostic tool, a reference implementation would substantially increase impact.
- **SHAP stability.** The paper reports CV < 1% for SALT tasks (50 seeds) but KDDCup99 shows C=21.1 +/- 7.5%. The instability in external datasets undermines the "compute C and threshold at 40%" simplicity.
- **External dataset preprocessing.** Some details are missing: how was Stack Overflow reduced to 3 classes? What specific attack categories were used for KDDCup99? How exactly was the Covertype geographic split implemented?

**Assessment:** Adequate for reproducibility of the core SALT experiments. External experiments need more detail.

---

## 5. Abstract/Introduction/Conclusion Consistency

The abstract, introduction, and conclusion are internally consistent and accurately represent the paper's findings. Specific checks:

- Abstract claims rho=0.853, p<0.001, n=16 --- matches Section 5.3 and Table 3. **Consistent.**
- Abstract claims "all paired p <= 0.005" --- matches Table 1. **Consistent.**
- Abstract hedges retraining as "unadjusted ... single-seed experiment; Holm-corrected p=0.11" --- appropriately qualified. **Consistent.**
- Conclusion accurately frames SHAP concentration as a "pre-deployment signal" and notes the threshold is "exploratory." **Consistent.**
- The abstract is dense (207 words by my estimate; UAI typically allows ~200). Consider trimming.

One minor inconsistency: the abstract says "8 supply chain tasks" and "16 multiclass tasks in 9 domains," but the introduction says "8 classification tasks." Since all SALT tasks are multiclass (K >= 13), this is not a real inconsistency, but the phrasing shifts between "classification" and "multiclass" without explanation.

---

## 6. Writing Issues

### Undefined or Late-Defined Terms
- **APS** is first used in the abstract ("APS conformity-score bounds") before being defined. Spell out "Adaptive Prediction Sets" on first use in the abstract.
- **RAPS** appears in the abstract ("RAPS expansion" is noted in the memory file, but in the abstract text "RAPS" is not spelled out). Actually, RAPS does not appear in the abstract --- only in Section 5.2, where it is properly cited. This is fine.
- **"Cat" column** in Table 1 is defined in the footnote but not in the table header.

### Unclear or Imprecise Claims
- Line 41: "practitioners lack tools to anticipate *which* gradient-boosted models will fail" --- but the paper's own evidence shows this works only for gradient-boosted models. The framing implies a gap the paper only partially fills for one model family.
- Section 4 (Theory): The theorem's Assumption (A1) is explicitly called an "idealization" that does not hold for TreeExplainer (which operates in log-odds space). The footnote acknowledges this, but the disconnect between the formal theorem and the empirical metric should be discussed more prominently. The theorem provides "directional intuition" --- this is honest but means the theory is illustrative rather than explanatory.
- Table 1 footnote: "classified by median" for high-variance tasks --- but the threshold framework uses mean. This dual-labeling scheme (median for Table 1, mean for threshold evaluation) is confusing.

### Minor Issues
- The \ie and \eg macros may not produce correct spacing in all LaTeX environments. Consider using `i.e.,\` and `e.g.,\` directly.
- Table 3: The "Multiclass (4 dom.)" and "Multiclass (8 dom.)" rows are intermediate analyses whose inclusion adds clutter without insight. The primary result (9 dom., n=16) suffices.
- Equation (3): The notation $s(x, y^*)$ sums probabilities "from the most likely class down to and including y*" --- but the equation shows $1 - \sum$, which is the complement. The verbal description and equation are equivalent but could confuse readers unfamiliar with APS.

---

## 7. UAI Formatting Compliance

- **Page limit:** UAI 2026 allows 8 pages for main body. The main body runs from Section 1 through Section 8 (Conclusion), approximately 8 pages including figures and tables. The paper appears compliant, but I cannot verify exact page count from the .tex source alone.
- **Document class:** Uses `uai2026`, which is correct.
- **Author information:** Included (not anonymized). This suggests camera-ready or non-anonymous submission. If this is for initial review, author info should be hidden.
- **References:** Appear on the main-body pages (via `\bibsection` redefinition). This is standard for UAI.
- **Appendix:** 9 pages of supplementary material, clearly separated with `\appendix`. Compliant.
- **Double-column to single-column transition:** The appendix switches to `\onecolumn`. This is acceptable for supplementary material.

**Potential issue:** The `\author` and `\affil` commands are present, suggesting de-anonymized submission. If this is for initial double-blind review, this would violate the anonymity requirement.

---

## Detailed Scores

| Criterion | Score (1-5) | Notes |
|-----------|-------------|-------|
| Novelty | 3.5 | New angle (pre-deployment diagnostic), but narrow scope (boosting only) |
| Significance | 3.5 | Practically useful for a specific model family; limited generalizability |
| Clarity | 4.0 | Well-written with appropriate qualifications; minor structural issues |
| Soundness | 3.5 | Correlational evidence is honest; theorem assumptions do not match practice |
| Reproducibility | 3.0 | Good detail for SALT; external experiments need more specification; no code |
| Presentation | 3.5 | Dense but organized; could streamline contributions and intermediate results |

---

## Overall Assessment

This paper addresses a genuinely useful question: given a deployed conformal predictor, can we anticipate failure severity before observing test data? The proposed diagnostic (SHAP concentration) is simple, interpretable, and shows strong empirical correlation with coverage degradation across 16 multiclass tasks. The statistical analysis is unusually thorough for this type of empirical study, with appropriate power analyses, multiple-comparison corrections, and honest reporting of limitations.

The main weaknesses are: (1) the diagnostic works only for gradient-boosted models, substantially limiting scope; (2) the effective sample size (n=16 tasks) is small, and the correlation could shift with additional datasets; (3) the formal theorem relies on an assumption (additive decomposition in probability space) that does not match the empirical metric (SHAP in log-odds space), making the theory more illustrative than explanatory; and (4) the decision framework is exploratory, with the retraining component essentially inconclusive.

The paper is above the acceptance threshold for UAI because it identifies a real and underexplored problem, proposes a simple and actionable diagnostic, and provides careful empirical evidence with honest qualifications. The restriction to boosting models is a real limitation but does not invalidate the contribution --- gradient-boosted models are the dominant tabular ML paradigm.

---

## Actionable Suggestions for Revision

1. **Foreground the boosting-specific scope** in the title or abstract. Currently "gradient-boosted models" appears in line 1 of the abstract but could be missed. Consider: "...as a pre-deployment diagnostic for conformal prediction vulnerability *in gradient-boosted classifiers*."

2. **Reduce the contributions list** from 6 to 3. Merge items 1+3 (diagnostic + quantification), items 4+5 (framework + external validation), and keep item 2 (theory) and item 6 (shift detection). Six contributions for an empirical case study oversells.

3. **Address the theory-practice gap** more prominently. The footnote in Theorem 1 is honest but buried. A brief paragraph after the theorem explicitly discussing "the theorem establishes intuition under idealized conditions; the empirical validation confirms the directional prediction but the additive probability-space model does not hold exactly" would strengthen the presentation.

4. **Provide a code repository.** For a paper proposing a diagnostic tool, practitioners need a reference implementation.

5. **Clarify external dataset shift mechanisms.** For the 5 UCI datasets with unspecified shift, either (a) document what the shift is, or (b) explicitly label them as "random-split baselines expected to show no shift" and analyze them separately.

6. **Remove or clearly demote the retraining analysis.** Holm-corrected p=0.11 from a single seed is inconclusive. Presenting it as part of the decision framework overstates the evidence.

7. **Simplify Table 3.** Remove intermediate rows (4 dom., 8 dom.) and keep only SALT (n=8), Multiclass primary (n=16), and Combined (n=17). The progressive buildup adds complexity without insight.

8. **Reconcile mean vs median labeling.** Use one consistent scheme throughout. If the framework uses mean-based at-risk labels, Table 1 should too.

9. **Fix APS definition in abstract.** Spell out "Adaptive Prediction Sets (APS)" on first use.

10. **Consider the anonymity requirement.** If submitting for double-blind review, remove author/affiliation information.

---

## Questions for Authors

1. Have you tested the diagnostic on datasets with *known* covariate shift magnitude (e.g., synthetic shifts of controlled intensity)? This would help disentangle concentration from shift severity.

2. The theorem assumes additive decomposition in probability space, but SHAP operates in log-odds space. Can you provide bounds under the log-odds model, even if looser?

3. For the 5 UCI random-split datasets, what is the expected concentration-drop relationship if there is no genuine shift? This would help calibrate the baseline expectation.

4. KDDCup99 shows C=21.1 +/- 7.5% across seeds. If the top feature identity changes across seeds, is the concentration metric itself well-defined for this task?

5. The paper focuses on top-1 concentration. Is there a theoretical reason (beyond the empirical sales-office observation) why a single dominant feature creates qualitatively different vulnerability than two features with similar importance?
