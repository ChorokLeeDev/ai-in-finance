# MethodCritic Review -- Round 21

**Paper**: "Diagnosing Conformal Prediction Failures Under Distribution Shift: A COVID-19 Case Study"
**File**: `/Users/i767700/Github/ai-in-finance/papers/conformal_covid/uai_2026/main.tex`
**Scope**: Full fresh read, all six dimensions, focused on remaining issues after 20 rounds.

---

## Executive Summary

After 20 rounds of revision, this paper is in strong shape. The statistical claims are well-hedged, the theorem's idealization is explicitly disclosed, and the key limitations (n=8 exploratory, threshold provisional, model-specific scope) are front-and-center. I find no fatal or major issues. What remains are moderate presentation issues, one subtle logical gap in the theorem application, and several minor items that a careful UAI reviewer might flag. The paper is publishable in its current form; the items below would strengthen it further.

---

## Fatal Issues (Red)

None.

---

## Major Issues (Orange)

None.

---

## Moderate Issues (Yellow)

### M1. Theorem 1 applies to K>=3 but verification uses K=45, 137, 459, 35, 13 -- the K=13 tasks are borderline

**Lines 144, 619.** Theorem 1 states K >= 3. The conservative bound verification (Appendix G, line 619) uses epsilon=0, h_bar=1/K, and reports bounds for 5 tasks. Two of these have K=13 (s-incoterms, i-incoterms). However, the paper's own methodology section (line 81) excludes K<=3 tasks from the primary endpoint because "binary APS prediction sets have a structural ceiling that blocks the concentration mechanism." The theorem formally requires only K >= 3, but the empirical motivation suggests the mechanism needs K >= 4 to produce meaningful variation. The verified tasks include K=13 and K=35 which are fine, but the disconnect between the theorem's K>=3 assumption and the endpoint's K>=4 criterion deserves a one-sentence note. Currently a reader could ask: if K=3 is excluded empirically, why does the theorem permit it?

**Suggested fix**: Add a parenthetical after "K >= 3" in Theorem 1 noting that the practical mechanism requires K >= 4 (as stated in Section 3.1), and K >= 3 is the mathematical minimum for the bound to be non-trivial.

### M2. Deterministic calibration split introduces a systematic confound that is acknowledged but not tested

**Lines 398-401.** The calibration split is deterministic (first-half/second-half), preserving temporal order within the validation window. The paper states this "does not affect the concentration--drop correlation, as the same split is applied uniformly across all tasks." This is a plausibility argument, not evidence. If COVID onset severity increases over the Feb-Jul 2020 window (which is likely), the calibration half (first half = early COVID) and evaluation half (second half = later COVID) will differ systematically. This could inflate or deflate coverage estimates relative to a random split, and since all tasks share the same temporal ordering, the direction of bias is consistent -- it would not be expected to change the *ranking* of tasks. However, this should be stated more precisely: the argument is that a constant systematic bias applied uniformly across tasks preserves rank correlations. The current phrasing ("does not affect") is too strong.

Meanwhile, the RAPS experiments (line 688) used random splits, creating a known protocol inconsistency. The paper acknowledges this but the inconsistency means RAPS coverage values in Table A6 are not comparable to APS values in Table 1, which could confuse readers who try to compare across tables.

**Suggested fix**: Soften "does not affect" to "is not expected to affect rank-order correlations, since the same deterministic split is applied uniformly to all tasks." Also consider a brief sensitivity note: was the main correlation recomputed with a single random split to verify robustness?

### M3. Mixed-effects model cluster count caveat could be more specific about the consequence

**Lines 718.** The paper states: "With only 8 task-level clusters, Wald inference is anti-conservative (random-intercept variance is downward-biased at <20 clusters); a Kenward-Roger df correction would yield estimated p approximately 0.01--0.03." This is a good caveat. However, the estimated KR range "0.01--0.03" appears to be an approximation without actual computation. If this was not computed with actual KR correction (e.g., via lmerTest in R or similar), the range should be flagged as "estimated" rather than stated as if computed. A UAI reviewer with mixed-effects expertise will ask how this range was obtained.

**Suggested fix**: Either run the KR correction and report exact p, or change to "a KR correction would likely increase the p-value by an order of magnitude, but the effect size CI [0.70, 2.58] excludes zero regardless."

### M4. The "protective factor" rule is derived from n=1 and the paper says so, but it is still used in threshold evaluation

**Lines 249, 498.** The protective-factor rule (Jaccard > 0.5 and importance > 15% on secondary feature) is described as "provisional until validated on additional false-positive cases" (line 249). However, in Appendix D (line 498), this rule is applied to reclassify s-office from FP to TN, boosting precision from 0.83 to 1.00. The threshold sensitivity analysis (Table A4) is "Step 2 only, no protective-factor check" -- which is the correct conservative approach. But the decision framework (Section 6, line 359) includes this protective-factor check as Step 1. A UAI reviewer will note: the framework's headline accuracy depends partly on an n=1-derived rule.

The paper already flags this (line 249), but the separation between the "Step 2 only" table and the framework that includes the protective-factor check could be made clearer, since a reader might take the framework at face value without noting the n=1 provenance.

**Suggested fix**: In Section 6 (line 359), add "(n=1; provisional)" after the protective-factor criterion for clarity at the point of use, not just at the point of derivation.

---

## Minor Issues (Blue)

### m1. Abstract length
The abstract is 198 words (estimated). UAI 2026 has no strict abstract word limit, but the abstract is dense and information-heavy. A reviewer might find it hard to parse on first read. The parenthetical statistical details (rho, p, tau, CI, all within one sentence around line 44-45) are thorough but could be streamlined.

### m2. "Fixed hyperparameters" claim (line 395)
The appendix describes hyperparameters as "fixed" and the text says "No task-specific tuning was performed." However, learning_rate=0.05, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5 are NOT LightGBM defaults (defaults: lr=0.1, feature_fraction=1.0, bagging_fraction=1.0, bagging_freq=0). The paper says "fixed" which is accurate (the same non-default values are used for all tasks), but a reviewer might wonder why these specific values were chosen. This was flagged in prior rounds and the wording was changed from "default" to "fixed," which resolves the factual error. No further action needed unless a reviewer specifically asks.

### m3. Contribution numbering discrepancy
Line 54-62 lists three contributions. Contribution 1 mentions "(Section 3.6)" for the model-sensitivity analysis, which appears in the main text. However, the detailed model-class sensitivity results (Table A7, mixed-effects) are in Appendix H, not Section 3.6 directly. Section 3.6 (line 119) provides the conceptual argument; Appendix H provides the data. This cross-referencing is adequate but a reviewer might look for the numbers in Section 3.6 and not find them.

### m4. Table 3 column alignment
Table 3 (line 260) uses `\resizebox{\columnwidth}` which may cause font-size inconsistency with other tables. Minor formatting concern.

### m5. i-incoterms omitted from baseline comparison
Line 578 states "item-incoterms omitted (3-seed pilot only)" from the entropy/ECE comparison table. With 7 of 8 tasks shown, this is a gap that should ideally be filled. If the 3-seed data exists, it could be reported with a caveat about limited seeds.

### m6. Placebo test asymmetric training window
Line 352 acknowledges "despite the asymmetric training window" but does not specify what the asymmetry is. The placebo uses train 2018, validate 2019-H1, test 2019-H2, which gives roughly 1 year of training data vs the COVID split's pre-Feb 2020 training period (which could be 2+ years depending on dataset start date). This duration difference could partially explain smaller placebo drops even without COVID. The 6-143x magnitude contrast is convincing enough that this is unlikely to be a sole explanation, but a reviewer could raise it.

### m7. Stack Overflow citation
Line 435 cites Stack Overflow as `\citep{dua2017uci}` (UCI repository). Stack Overflow is not a UCI dataset -- it's from Kaggle or similar sources. If the data was obtained via UCI, this is fine; if not, the citation is incorrect.

### m8. Seed range inconsistency: 50 seeds (SALT) vs 10 seeds (external)
The primary correlation n=16 mixes 50-seed means (SALT) and 10-seed means (external). This is documented (line 613) and defensible, but the different seed counts mean the external point estimates have wider confidence bands. KDDCup99's drop of 15.9 +/- 21.4 pp (10 seeds) has a CI of roughly [-7, 39], spanning both robust and catastrophic. With 50 seeds, this range would narrow, potentially changing the point estimate and its classification. The paper acknowledges KDDCup99's variability but the mixed seed protocol could be noted as a factor.

---

## Dimension Checklist

### Dimension 1: Internal Validity
- Causal claims appropriately hedged as "associated with" rather than "causes" (line 135).
- The key confound (class cardinality) is extensively addressed with partial correlation at n=16 (line 302).
- No uncontrolled confounders that would invalidate the rank correlation.
- **Score: Adequate.**

### Dimension 2: External Validity
- 9 external datasets across diverse domains. Generalization is demonstrated, not merely claimed.
- WEIRD concern is not applicable (datasets, not human subjects).
- Supply chain specificity is acknowledged; external validation addresses this.
- **Score: Good.**

### Dimension 3: Statistical Rigor
- Power analysis explicitly provided (line 253). Exploratory/confirmatory distinction clearly stated.
- Multiple comparisons addressed via Holm-Bonferroni (line 611).
- Effect sizes and CIs reported throughout.
- Bootstrap CIs use percentile method at small n=8 (line 245); BCa alternative noted but not used.
- The Holm-corrected retraining p=0.11 is honestly reported (line 344).
- **Score: Good.** One residual concern: the n=16 primary endpoint includes 8 in-sample SALT tasks on which the threshold was derived. The paper acknowledges this (line 255) but the headline rho=0.853 is partially in-sample. The truly out-of-sample external-only subset shows Prec=1.00, Rec=0.50 -- which is the honest transfer metric.

### Dimension 4: Measurement Quality
- SHAP concentration is well-defined (Eq 2) with stability analysis (CV < 1%, line 245).
- Coverage metric is standard (APS with stated quantile formula).
- Jaccard similarity well-defined (Eq 1).
- No ceiling/floor effects unacknowledged (Stack Overflow ceiling explicitly discussed).
- **Score: Good.**

### Dimension 5: Reproducibility
- Code repository stated (line 375, 391).
- Hyperparameters fully specified.
- Seeds specified (42-91).
- Software versions specified (Python 3.9, LightGBM 3.3, SHAP 0.41).
- Calibration split protocol specified.
- No code provided in the paper to execute.
- **Score: Good** (pending verification that the GitHub repo exists and contains working code).

### Dimension 6: Logical Soundness
- Conclusions appropriately scoped to gradient-boosted models.
- "Exploratory" framework label used correctly.
- Retraining recommendation hedged with Holm-corrected p.
- Alternative explanations (class cardinality, shift detection) systematically addressed.
- The one area where interpretation slightly outruns evidence is the claim that "SHAP concentration is diagnostic where gradient boosting's sequential structure produces genuine single-feature dependence" (line 131) -- this is a mechanistic interpretation of a correlational finding. The RF/MLP comparisons are consistent with this story but do not prove the mechanism. The language is careful enough ("identifies a boundary") that this is defensible.
- **Score: Good.**

---

## Code Execution Results

No analysis code was provided within the paper files for execution. The paper references a GitHub repository at `https://github.com/ChorokLeeDev/conformal-covid`. I did not verify whether this repository exists or contains executable code.

---

## Reproducibility Score

**7/10.** All methodological details are specified. Software versions provided. Seeds specified. Calibration protocol documented. The score is not higher because: (1) I cannot verify the GitHub repo contains working code, (2) external dataset preprocessing scripts are not described in sufficient detail for independent replication (e.g., Covertype wilderness area column indices, Gas Sensor batch file parsing), (3) SHAP subsample of 10K is mentioned but the subsampling method (random? first 10K?) is not specified.

---

## Recommended Actions (Priority Order)

1. **[M2]** Soften the claim that deterministic calibration split "does not affect" the correlation to "is not expected to affect rank-order correlations."
2. **[M1]** Add a parenthetical clarifying why Theorem 1 uses K>=3 while the endpoint requires K>=4.
3. **[M4]** Add "(n=1; provisional)" to the protective-factor criterion in Section 6 where it is used.
4. **[M3]** Clarify whether the KR p-value range was actually computed or estimated.
5. **[m7]** Verify Stack Overflow data source citation.
6. **[m6]** Specify the placebo training window duration for completeness.

---

## Verdict

**MINOR REVISION REQUIRED** -- but only barely. The four moderate issues are presentation/hedging refinements, not methodological flaws. A UAI reviewer panel could accept this paper as-is with high probability; addressing M1-M4 would close the remaining attack surfaces. The statistical work is thorough, the limitations are honestly stated, and the scope claims are appropriately bounded. This is a well-executed empirical contribution with useful practical guidance for practitioners deploying conformal prediction under potential shift.
