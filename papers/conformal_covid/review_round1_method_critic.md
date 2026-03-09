# MethodCritic Analysis Report

**Paper**: "Diagnosing Conformal Prediction Failures Under Distribution Shift: A COVID-19 Case Study"
**Venue**: UAI 2026 submission
**File**: `/Users/i767700/Github/ai-in-finance/papers/conformal_covid/uai_2026/main.tex`
**Review date**: 2026-02-20

---

## Executive Summary

This paper proposes SHAP concentration (top-1 feature importance fraction) as a pre-deployment diagnostic for conformal prediction vulnerability under distribution shift. The empirical contribution is substantial and well-structured, with extensive robustness analyses (50-seed ensemble, ICC, bootstrap, leave-one-out, multi-model sensitivity, external validation). However, three issues demand attention before acceptance: (1) the abstract contains a duplicated paragraph that will cause immediate desk-rejection at any venue; (2) the Theorem 1 numerical verification in Appendix B (Section S7) reports bound values (0.785, 0.841, 0.990, 0.806, 0.825) that are mathematically irreconcilable with the stated formula under the stated assumptions (eps=0, h_bar=1/K); and (3) the retraining claim (+19pp, p=0.04) reported in the abstract is not adjusted for the 3-task multiple comparison family, which would render it non-significant (Holm-adjusted p=0.12).

---

## Fatal Issues [RED]

### F1. Abstract contains duplicated paragraph

**Issue**: Lines 44-46 of `main.tex` contain the abstract body printed twice. The text from "seeds), we show that..." through the end of the abstract appears in full on line 45, then again with minor wording variation on line 46. The second version differs only in the final clause ("consider quarterly retraining" vs. "suggest that quarterly retraining may partially recover coverage").

**Evidence**: Direct inspection of `/Users/i767700/Github/ai-in-finance/papers/conformal_covid/uai_2026/main.tex`, lines 39-47.

**Methodological Consequence**: This will cause the compiled PDF to render a double-length abstract with visibly duplicated text. Any reviewer or area chair will flag this as a submission error, and many venues (including UAI) would desk-reject for formatting violations.

**Required Fix**: Delete line 46 entirely, keeping only the first version of the abstract body.

### F2. Theorem 1 numerical verification uses values inconsistent with stated formula

**Issue**: Appendix S7 claims "Using eps=0 and h_bar=1/K, Theorem 1(ii) yields lower bounds consistent with all five applicable right-shift tasks: s-shipcond (0.785 vs. observed 0.98), s-payterms (0.841 vs. 0.92), s-group (0.990 vs. 1.00), i-plant (0.806 vs. 0.86), and s-incoterms (0.825 vs. 0.87)."

However, computing the bound from Equation (5) with eps=0 and h_bar=1/K:

```
E[s_test] >= C(1-(K-1)*0) + (1-C)(1-(K-1)/K) = C + (1-C)/K
```

yields:

| Task | K | C | Correct bound | Paper claims |
|------|---|---|---------------|--------------|
| s-shipcond | 45 | 0.507 | 0.518 | 0.785 |
| s-payterms | 135 | 0.542 | 0.545 | 0.841 |
| s-group | 462 | 0.473 | 0.474 | 0.990 |
| i-plant | 35 | 0.239 | 0.261 | 0.806 |
| s-incoterms | 13 | 0.237 | 0.296 | 0.825 |

The correct bounds are much weaker (lower) than claimed. They are still valid as lower bounds (all correct values < observed values), so the theorem statement itself is not invalidated. But the numerical verification reports incorrect numbers.

Back-computing the implied h_bar from the claimed bounds yields h_bar values all *less* than 1/K (e.g., 0.0099 vs. 1/45 = 0.0222 for s-shipcond), which contradicts the sufficient condition for monotonicity (h_bar >= 1/K) stated in the theorem.

**Evidence**: Independent recomputation from the stated formula. Verified algebraically: f(C) = C + (1-C)/K = C(1-1/K) + 1/K, giving 0.507(1-1/45) + 1/45 = 0.496 + 0.022 = 0.518.

**Methodological Consequence**: The theorem's formal content is sound (proof logic verified), but the numerical verification is wrong. This undermines the empirical grounding of the theoretical contribution. A reviewer checking the arithmetic will flag this as either a coding error or a formula transcription error.

**Required Fix**: Recompute the bounds using the correct formula and update the appendix. If the code uses a different formula than what is stated, reconcile the theorem statement with the actual computation.

---

## Major Issues [ORANGE]

### M1. Retraining p-value not adjusted for multiple comparisons

**Issue**: The abstract and Section 5.4 report "+19pp recovery, p=0.04" for retraining on sales-shipcond. However, retraining was evaluated on 3 severe tasks (s-shipcond, s-payterms, s-group), constituting a natural comparison family. Under Holm-Bonferroni correction for 3 tests, the adjusted p-value is 3 x 0.04 = 0.12, which is not significant at alpha=0.05.

**Evidence**: Section 5.4 explicitly reports results for all 3 tasks. The retraining analysis is framed as testing whether retraining helps vulnerable tasks -- the comparison family is the set of vulnerable tasks tested.

**Methodological Consequence**: The abstract's "+19pp, p=0.04" claim is the most actionable statement in the paper (it motivates quarterly retraining), yet it does not survive standard multiplicity correction. This is exactly the type of selective reporting that erodes confidence.

**Required Fix**: Either (a) report the Holm-adjusted p=0.12 in the abstract and text, or (b) frame this as an exploratory, task-specific observation rather than a generalizable claim. The paper already uses "exploratory" language elsewhere -- apply it here too.

### M2. Assumption A1 (additive probability decomposition) mismatches SHAP semantics

**Issue**: Theorem 1's Assumption A1 posits p_hat(y|x) = C * g(y|x_1) + (1-C) * h(y|x_{-1}), an additive decomposition in *probability space*. However, SHAP values (TreeExplainer) decompose the model's output in *log-odds* or *margin space* for tree-based models, not in probability space. The additivity of SHAP in output space does not imply additivity in probability space due to the softmax/sigmoid nonlinearity.

**Evidence**: Lundberg & Lee (2017) define SHAP as an additive feature attribution on f(x), not on p(y|x). For multiclass LightGBM, SHAP decomposes the raw leaf values, which pass through softmax to become probabilities. The concentration C is computed from SHAP values (output space), but used in A1 as if it parameterizes a probability-space decomposition.

**Methodological Consequence**: The gap between what SHAP measures and what A1 assumes means the theorem's conclusion (monotone vulnerability in C) holds under an idealized model that does not exactly match the empirical setup. The strong empirical correlation suggests the approximation is reasonable in practice, but the gap should be disclosed.

**Required Fix**: Add a remark after Assumption A1 noting that SHAP decomposition operates in output space while A1 operates in probability space, and that the assumption is therefore an approximation motivated by the empirical success of TreeExplainer for capturing local feature contributions.

### M3. Partial correlation non-significance undermines uniqueness claim

**Issue**: The partial Spearman correlation of SHAP concentration with coverage drop, controlling for log(num_classes), is rho_partial = 0.629, p = 0.131 (Table 4, n=8). Likewise, log(num_classes) controlling for concentration is rho_partial = 0.334, p = 0.464. Neither partial correlation is significant. The paper states this correctly but the implications are underplayed: at n=8, the data cannot distinguish whether concentration, class cardinality, or their combination drives the effect.

**Evidence**: Table 4, `/Users/i767700/Github/ai-in-finance/papers/conformal_covid/results/icc_and_partial.json`.

**Methodological Consequence**: The paper correctly appeals to cross-domain evidence to address this (Covertype has only 7 classes but fails catastrophically). However, the within-SALT analysis -- which is the paper's primary testbed -- cannot disentangle concentration from cardinality. The cross-domain evidence resolving this comes from a single catastrophic external case (Covertype), which is thin.

**Required Fix**: State more prominently in Section 5.2 that within-SALT, the evidence cannot disentangle concentration from cardinality due to power limitations at n=8, and that the cross-domain resolution relies primarily on the Covertype datapoint.

### M4. Holm-Bonferroni for concentration metric selection is exactly at boundary

**Issue**: Appendix S5 reports testing 5 concentration metrics (top-1, top-2, top-3, HHI, entropy) and notes the Holm-Bonferroni adjusted p for top-1 is 5 x 0.010 = 0.050. The paper argues this is conservative because top-1 was the "a priori natural choice." However, no pre-registration or dated record supports this claim of a priori selection.

**Evidence**: Appendix S5, line 593: "the adjusted p-value for top-1 concentration is 5 x 0.010 = 0.050, at the conventional significance boundary."

**Methodological Consequence**: At exactly p=0.050 (the conventional boundary), the result is marginal. Whether a reviewer accepts the "a priori" argument depends on trust, which anonymous review inherently lacks. The argument that top-1 is natural is reasonable but not verifiable.

**Required Fix**: Either pre-register the analysis plan, or acknowledge that the adjusted p is at the conventional boundary and that readers should weight the cross-domain replication (n=16, p<0.001) as the more reliable endpoint.

---

## Moderate Issues [YELLOW]

### Y1. Class count discrepancies between Table 1 and data files

**Issue**: Table 1 reports s-group=462 classes, s-payterms=135, i-shippoint=70. The `icc_and_partial.json` data file records 459, 137, and 69 respectively. These small discrepancies suggest the Table 1 values and the analysis code may use different data processing stages.

**Evidence**: Cross-referencing Table 1 (lines 204-211) with `/Users/i767700/Github/ai-in-finance/papers/conformal_covid/results/icc_and_partial.json` lines 228-235.

**Required Fix**: Reconcile class counts to a single authoritative source and ensure all analyses use the same values.

### Y2. No prediction set sizes reported for standard APS

**Issue**: Coverage alone is uninformative without prediction set sizes. For high-cardinality tasks (s-group: 462 classes), validation coverage of 83.6% could be achieved by prediction sets containing hundreds of classes, which would be practically useless. Set sizes are reported for ACI and RAPS (Appendix S8) but not for the primary APS results in Table 1.

**Evidence**: Table 1 reports coverage and drops. The RAPS appendix mentions mean set size 232/462 for s-group, suggesting very large sets.

**Required Fix**: Add a column for mean set size to Table 1, or report set sizes in an appendix table, so readers can assess practical utility.

### Y3. COVID-era n=9 row in Table 3 is unexplained

**Issue**: Table 3 (Stratified Correlation Analysis) includes "COVID-era n=9, rho=0.883, p=0.002" but the composition of this group is never explained. The 8 SALT multiclass tasks are all COVID-era. Adding one task to reach n=9 presumably includes a binary SALT task, but this contradicts the binary ceiling effect that motivates excluding binary tasks elsewhere.

**Evidence**: Table 3, line 271. No corresponding explanation in text.

**Required Fix**: Either remove this row or explain what the 9th task is and why it is included in a group that mixes binary and multiclass tasks.

### Y4. Mixed-effects model degrees of freedom

**Issue**: The mixed-effects model with 3 boosting models (n=24, 8 tasks as random intercepts) reports beta_1=1.64, p=0.0006. With only 8 groups (tasks), the effective degrees of freedom for the fixed effect under Satterthwaite or Kenward-Roger approximation may be much less than 22 (the naive n-2). The p-value's validity depends on the df approximation method used, which is not stated.

**Evidence**: `/Users/i767700/Github/ai-in-finance/papers/conformal_covid/results/mixed_effects_analysis.json`: n_obs=24, n_groups=8.

**Required Fix**: Report the df approximation method and the resulting degrees of freedom. If using Wald-type z-tests (as some Python implementations do by default), note this as anti-conservative.

### Y5. External validation dominated by a single catastrophic case

**Issue**: Among 9 external domains, only Covertype is deterministically catastrophic (10/10 seeds). KDDCup99 is the only other at-risk case but is seed-dependent and borderline (mean drop = 15.9pp vs. the 15pp threshold). All other external datasets are robustly non-failing. This means the positive predictive value of the diagnostic in external domains rests essentially on a single datapoint.

**Evidence**: Table S3 (`manuscript_claims_snapshot.json`): only Covertype has both C>40% and drop>15% among external datasets.

**Required Fix**: Acknowledge more explicitly in Section 6 that external catastrophic evidence is concentrated in Covertype, and that prospective deployment studies with additional high-concentration failing tasks are needed to validate the threshold.

### Y6. Stale cross_domain_statistics.json file

**Issue**: The file `/Users/i767700/Github/ai-in-finance/papers/conformal_covid/results/cross_domain_statistics.json` reports "combined_n11" with rho=0.6909, which is labeled as "STALE" in `verified_n11_multiclass.json`. This stale file could cause confusion if any analysis scripts reference it.

**Evidence**: `verified_n11_multiclass.json` line 108: "This file is STALE - does not include external validation datasets."

**Required Fix**: Delete or clearly deprecate the stale file.

---

## Minor Issues [BLUE]

1. **Table 1 note**: "SEV = Severe (>50% drop), ROB = Robust (<20% drop)" -- the gap between 20% and 50% is undefined. i-shippoint (18.5%) and s-payterms (77.1%) are cleanly in ROB/SEV, but what about hypothetical tasks in the 20-50% range?

2. **Eq 3 notation**: The condition "p_hat(y|x) < p_hat(y*|x)" in the subscript of the summation should more precisely handle ties (multiple classes with equal probability). The paper notes using non-randomized APS but the tie-breaking in the formula is not explicit.

3. **LOO stability reporting**: The paper states "rho in [0.75, 0.96]" for leave-one-out at n=8, with "2 of 8 jackknife samples" losing significance. But the text also says "p=0.052" for those 2 -- this is for n=7 (after removing one). Worth noting that 2/8 non-significant jackknife samples at n=7 is not concerning given the power loss.

4. **Figure 2 caption**: References "8 severe-shift tasks (dark)" and "3 binary/cross-domain tasks (lighter)." This refers to the n=11 configuration, but the primary endpoint is n=16. Consider updating the figure or adding a n=16 version.

5. **Jaccard formula (Eq 1)**: Uses A_train and A_test for unique value sets, but Section 3.4 says "we substitute validation data for test data as a proxy." The formula as written uses test data. Consider using A_val in the formula or clarifying the distinction.

6. **s-group validation coverage 83.6% is below the 90% target**: This suggests calibration issues for this task even pre-shift. The paper does not comment on why validation coverage is sub-nominal for some tasks.

7. **Bootstrap seed**: The n=16 bootstrap uses seed=42 (`manuscript_claims_snapshot.json`). A single bootstrap seed is standard but worth noting the CI could vary with different seeds.

8. **Appendix S6 (Table S5)**: Reports "item-incoterms omitted (3-seed pilot only)" -- inconsistent with the 50-seed protocol used for all other SALT tasks. Either run 50 seeds for this task or explain the discrepancy.

9. **The n=11 -> n=16 escalation**: Table 3 shows five different correlation subsets (n=8, 9, 11, 15, 16, 19). While transparency is good, reporting this many subsets without a clear primary endpoint creates a garden-of-forking-paths concern. The paper designates n=16 as primary, which is appropriate, but the progression suggests the sample was expanded until a desired result was achieved. Pre-registration would resolve this.

---

## Code Execution Results

No standalone analysis script was executed end-to-end because the experimental pipeline requires the SALT dataset (cached at ~/.cache/relbench) and substantial compute time (~3-4 hours for 50-seed suite). However, the following verifications were performed:

1. **Spearman correlations**: Independently recomputed from the `manuscript_claims_snapshot.json` data using scipy.stats.spearmanr. All claimed correlations match:
   - n=8 SALT: rho=0.833, p=0.0102 (paper: 0.833, 0.010). MATCH.
   - n=11: rho=0.909, p=0.000106 (paper: 0.909, <0.001). MATCH.
   - n=16: rho=0.853, p=0.000027 (paper: 0.853, <0.001). MATCH.
   - n=16 Kendall tau=0.667, p=0.000135 (paper: 0.667, <0.001). MATCH.

2. **Threshold classification at 40%**: Recomputed TP=5, FP=1, FN=1, TN=9 on n=16. Precision=0.83, Recall=0.83. MATCH.

3. **Theorem 1 bounds**: Recomputed from Eq (5) with eps=0, h_bar=1/K. Results DO NOT MATCH paper's claimed values (see F2 above).

4. **Holm-Bonferroni**: Confirmed 5 x 0.010 = 0.050. MATCH with paper's claim of "at the conventional significance boundary."

5. **Data files**: Cross-checked `icc_and_partial.json`, `mixed_effects_analysis.json`, `pooled_meta_analysis.json`, `manuscript_claims_snapshot.json` against paper claims. All match except: (a) class counts (Y1), (b) theorem bounds (F2).

---

## Reproducibility Score

**6/10**

Justification:
- (+) Result JSON files are extensive and well-organized under `papers/conformal_covid/results/`
- (+) Analysis code exists under `papers/conformal_covid/code/` (~60 scripts)
- (+) 50-seed ensemble design with clear seed range (42-91)
- (+) Software versions reported (Python 3.9, LightGBM 3.3, SHAP 0.41)
- (+) Data source is public (RelBench SALT dataset)
- (-) No requirements.txt or environment specification file
- (-) No single entry-point script to reproduce all results end-to-end
- (-) Stale intermediate files (cross_domain_statistics.json marked STALE)
- (-) No pre-registration or timestamped analysis plan
- (-) Random seeds specified but no verification that results are deterministic across platforms
- (-) External dataset preprocessing details scattered across individual scripts

---

## Recommended Actions (Priority Order)

1. **[CRITICAL]** Delete the duplicated abstract paragraph (line 46 of main.tex). This is a 30-second fix that prevents desk-rejection.

2. **[CRITICAL]** Recompute Theorem 1 bound verification in Appendix S7. Either fix the code that generated the bounds or correct the formula mapping. Report the correct (weaker) bounds, which are still valid.

3. **[HIGH]** Apply Holm-Bonferroni correction to the retraining p-value (3-task family) or reframe as exploratory. Update the abstract accordingly.

4. **[HIGH]** Add a remark about the Assumption A1 gap (SHAP operates in output space, not probability space).

5. **[MEDIUM]** Reconcile class counts between Table 1 and data files.

6. **[MEDIUM]** Report prediction set sizes for standard APS in Table 1 or a companion table.

7. **[MEDIUM]** Explain or remove the COVID-era n=9 row in Table 3.

8. **[MEDIUM]** Report mixed-effects df approximation method and resulting degrees of freedom.

9. **[MEDIUM]** Strengthen language about external catastrophic evidence being concentrated in Covertype.

10. **[LOW]** Clean up stale result files. Add a reproducibility script or Makefile.

---

## Verdict

**MAJOR REVISION REQUIRED**

The paper presents a genuinely useful diagnostic (SHAP concentration) with extensive empirical support. The core correlations are verified and correct. However, the duplicated abstract is a fatal formatting error, the theorem numerical verification contains mathematical errors, and the retraining p-value requires multiplicity correction. None of these issues invalidate the paper's central contribution, but all three must be fixed before the paper can be accepted at a top venue. The theoretical contribution (Theorem 1) is logically sound but the gap between SHAP semantics (output space) and Assumption A1 (probability space) needs disclosure. After these revisions, the paper would be a solid contribution to UAI.
