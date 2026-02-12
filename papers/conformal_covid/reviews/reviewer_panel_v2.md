# UAI 2026 Simulated Review Panel (v2)
## "Diagnosing Conformal Prediction Failures Under Distribution Shift: A COVID-19 Case Study"

**Date**: 2026-02-11
**Method**: 4 simulated reviewers with distinct expertise profiles
**Paper version**: Revised (post-v1 review round, incorporating cross-domain validation, binary ceiling effect, and various editorial fixes)

---

## Panel Summary

| Reviewer | Role | Soundness | Significance | Novelty | Clarity | Score | Recommendation |
|----------|------|-----------|-------------|---------|---------|-------|----------------|
| R1 | Conformal Prediction Expert | 2.5/4 | 3/4 | 2/4 | 3.5/4 | **5/10** | Borderline |
| R2 | ML Practitioner | 2/4 | 2.5/4 | 2.5/4 | 3/4 | **5/10** | Borderline |
| R3 | Statistician | 2/4 | 2/4 | 2/4 | 3/4 | **4/10** | Weak Reject |
| R4 | Causal/Experimental Design | 2.5/4 | 2.5/4 | 2.5/4 | 3.5/4 | **5/10** | Borderline |
| **Mean** | | **2.25** | **2.5** | **2.25** | **3.25** | **4.75** | **Borderline** |

**Delta from v1**: Mean score improved from 4.5 to 4.75. Two reviewers moved from Weak Reject to Borderline (R1, R2). The cross-domain validation and honest handling of the binary ceiling effect were positively received. However, several v1 weaknesses remain unaddressed, and the revision introduced new concerns.

---

## Tracking: Previous Weaknesses Addressed vs. Remaining

### ADDRESSED (fully or substantially)

| v1 Issue | Status | How Addressed |
|----------|--------|---------------|
| Cross-domain validation doesn't test SHAP concentration | **FIXED** | Table 5 now includes SHAP concentration for 11 tasks across 3 domains with combined rho=0.691, p=0.019 |
| "Quasi-natural experiment" framing | **FIXED** | Renamed to "case study" throughout |
| APS formula incorrect | **FIXED** | APS conformity score definition corrected in Section 4 |
| Stochastic dominance text direction | **FIXED** | Eq. 3 now correctly states the dominance direction |
| Abstract Jaccard claim | **FIXED** | Removed |
| Section 4 overclaimed as "theoretical grounding" | **PARTIALLY FIXED** | Now labeled "Intuition" in title, "heuristic" in text, "mechanistic intuition" in contributions. Still listed as contribution #2. |
| Threshold circularity | **PARTIALLY FIXED** | Data separation protocol added (Section 3.1). Threshold described as "exploratory" in framework section. But F1 metrics still reported as if they validate the threshold. |
| LOO analysis | **IMPROVED** | 6/8 significant reported; 2 jackknife samples at p=0.052 disclosed honestly |
| SHAP stability | **FIXED** | Bootstrap CV <1% reported |
| Retraining language | **PARTIALLY FIXED** | "+19 pp" more honest than "restores" but abstract still says "restores" |

### REMAINING (not addressed or insufficiently addressed)

| v1 Issue | Status | Severity |
|----------|--------|----------|
| n=8 from single database as core evidence | **UNCHANGED** | CRITICAL -- cross-domain adds only 3 binary tasks with near-zero drops |
| Pseudo-replication not quantified (ICC) | **UNCHANGED** | HIGH -- "effective n as low as 5" still asserted without computation |
| Missing baselines (ensemble disagreement, native FI, MMD, PSI) | **UNCHANGED** | HIGH -- no alternative pre-deployment diagnostics compared |
| Theory section still listed as contribution | **PARTIALLY REMAINING** | MODERATE -- demoted in language but still contribution #2 |
| Single model class (LightGBM) | **UNCHANGED** | MODERATE |
| Retraining tested on 1 task only | **UNCHANGED** | MODERATE |
| No conformity score CDF plots | **UNCHANGED** | MODERATE |
| Train-validation Jaccard proxy never validated | **UNCHANGED** | MODERATE |
| Protective-factor heuristic from 1 example | **UNCHANGED** | LOW-MODERATE |

---

## Reviewer 1: Conformal Prediction Expert

**Score: 5/10 (Borderline)**
**Confidence: 4/5**

### Summary

The revised paper proposes SHAP concentration as a pre-deployment diagnostic for conformal prediction vulnerability and now includes a cross-domain analysis across 11 tasks in 3 domains. The binary ceiling effect discovery is a genuine contribution to understanding APS behavior. However, the core evidence for the diagnostic's utility still rests on 8 multiclass tasks from a single database, the theoretical section remains informal despite being listed as a contribution, and several important CP-specific technical issues are unaddressed.

### Strengths

- **Binary ceiling effect is a real insight.** The observation that binary APS prediction sets ({0}, {1}, {0,1}) are structurally protected against the coverage degradation mechanism identified here is technically correct and practically informative. The Mann-Whitney test (p=0.024) confirms the systematic difference. This deserves more prominence.

- **Data separation protocol is clearly stated.** Section 3.1 now makes explicit which data is used for what. This is a significant improvement over v1.

- **Honest reporting of LOO instability.** Disclosing that 2/8 LOO samples lose significance (p=0.052) is commendable. Many papers would hide this.

- **Entropy paradox remains the most interesting finding.** Catastrophic tasks show *decreasing* entropy (confident wrong predictions) while collapsing in coverage. This is counter-intuitive and genuinely useful for practitioners monitoring deployed models.

- **ACI analysis is thorough.** The Utility classification (Useful/Marginal/Vacuous) adds practical value. The observation that ACI makes s-group predictions vacuous (82% of 462 classes) is a strong argument for pre-deployment diagnosis over post-hoc repair.

- **Permutation p-values reported.** For the cross-domain correlation, both scipy and permutation p-values are provided (0.019 and 0.023 respectively). This addresses v1 concerns about the p-value computation method.

### Weaknesses (ordered by severity)

1. **APS formulation still imprecise (TECHNICAL).** The paper defines the APS conformity score as "the cumulative probability mass from the most likely class until the true label is included" (Section 4). This omits the randomization step that makes APS valid (Romano et al., 2020, Section 3.1). The standard APS score includes a uniform random variable U to break ties: s(x,y) = sum_{j=1}^{k-1} f_pi_j(x) + U * f_pi_k(x) where pi_k = y. Without randomization, the coverage guarantee is conservative rather than exact. This matters for the stochastic dominance argument because the randomization affects the score distribution. The paper should either include the full formulation or explicitly state they use the non-randomized version and note the conservativeness.

2. **Stochastic dominance is claimed but never verified.** The paper claims stochastic dominance (Eq. 3) and says "Empirical verification confirms the predicted stochastic dominance for catastrophic tasks" (Section 5.2), but no empirical verification is shown anywhere in the paper -- no CDF plots, no KS tests, no quantile comparisons. This is a factual claim without evidence. Either show the CDFs or remove the empirical verification claim.

3. **The "nearly three orders of magnitude" claim is misleading.** Coverage drops range from 0.1% to 77.1%. The ratio 77.1/0.1 = 771x. But 0.1% is essentially zero drop (within noise for a 90% target). Expressing this as "nearly three orders of magnitude" treats a noise-level measurement as if it were a precise quantity. A more honest framing: "from negligible to catastrophic" or "from <1% to >70%." The ratio is dominated by the denominator being essentially zero.

4. **Cross-domain validation is weaker than presented.** The combined rho=0.691 with bootstrap CI [0.08, 0.97] spans from "negligible" to "near-perfect." At n=11, the CI is too wide to be informative. More critically, the 3 cross-domain tasks are ALL binary, meaning they contribute no information about the SHAP concentration - coverage drop relationship for the failure mode the paper identifies. The effective cross-domain validation for the multiclass diagnostic is n=0. The COVID-era subset (rho=0.883, n=9) is stronger but adds only 1 binary task to the 8 SALT tasks.

5. **Conformal prediction literature gaps.** Several directly relevant works are missing:
   - Cauchois et al. (2024), "Robust and Agnostic Learning of Conditional Distributional Treatment Effects" -- discusses conformity score robustness under shift
   - Angelopoulos et al. (2024), "Conformal Risk Control" -- broader framework for controlling risk under exchangeability
   - Feldman et al. (2023), "Achieving Risk Control in Online Learning Settings" -- online CP that subsumes ACI
   - The Barber et al. (2023) result on coverage degradation bounds beyond exchangeability is cited but not quantitatively applied. Could the authors use Theorem 2 of Barber et al. to derive expected coverage drop given a quantified exchangeability violation?

6. **RAPS not considered.** The paper uses APS exclusively. Regularized APS (RAPS, Angelopoulos et al., 2021) is designed to reduce prediction set sizes. Would RAPS change the vulnerability pattern? If concentrated importance leads to "confidently wrong" predictions, RAPS's regularization might actually make things worse. This is worth discussing.

7. **Calibration split affects results but is not analyzed.** The 50/50 random split of validation into calibration and evaluation introduces variance. Different calibration sets could yield different thresholds. The paper reports seed-level variation but doesn't analyze how the calibration/evaluation split ratio affects results. What happens with 70/30 or 30/70 splits?

### Questions for Authors

1. Is the APS implementation randomized or non-randomized? What software library is used?
2. Can you provide KS test statistics for calibration vs. test conformity score distributions for at least 2 tasks (1 catastrophic, 1 robust)?
3. The Barber et al. (2023) bound on coverage under non-exchangeability is P(Y_{n+1} in C(X_{n+1})) >= 1 - alpha - TV(P_{cal}, P_{test}). Have you estimated the total variation distance for each task?
4. How does the diagnostic perform with RAPS instead of APS?
5. For the COVID-era subset (n=9, rho=0.883), removing the single binary task (study-outcome) gives the original n=8 result. What is the incremental evidence from study-outcome?

### Suggestions

1. Add conformity score CDF plots (calibration vs. test) for 1 catastrophic + 1 robust task as a figure. This is the single most impactful addition for establishing the stochastic dominance mechanism.
2. Either include the full randomized APS formula or explicitly note you use the conservative non-randomized version.
3. Replace "nearly three orders of magnitude" with "from negligible (<1%) to catastrophic (>70%)."
4. Discuss RAPS as an alternative conformal method.
5. Cite and use the Barber et al. (2023) coverage degradation bound quantitatively.

---

## Reviewer 2: ML Practitioner

**Score: 5/10 (Borderline)**
**Confidence: 4/5**

### Summary

The paper proposes a practical framework for predicting which conformal prediction deployments will fail under distribution shift. The idea is appealing -- practitioners need pre-deployment warnings. The revision adds cross-domain validation and the binary ceiling effect, both useful. However, the paper still lacks comparison against obvious alternative diagnostics, tests only one model class, and the cross-domain "validation" does not actually validate the diagnostic for the failure mode it claims to detect.

### Strengths

- **Clear problem statement.** The question "which models will fail?" is practically more useful than "how much will models fail on average?" Every ML team I have worked with would want this.

- **Binary ceiling effect is practically useful.** Knowing that binary APS is structurally protected saves practitioners from unnecessary monitoring overhead.

- **ACI utility classification.** The Useful/Marginal/Vacuous framing in Table 4 is exactly how a deployment team would evaluate ACI. The observation that ACI makes s-group predictions vacuous (82% of classes) is a strong practical argument.

- **SHAP stability (CV <1%).** This addresses a real concern about whether the diagnostic itself is reliable. Good.

- **The entropy paradox has deployment implications.** If entropy-based monitoring is standard practice (it is at many companies), knowing it gives false reassurance on the worst failures is actionable.

- **Placebo test.** The 6-140x COVID vs. pre-COVID ratio is convincing that COVID is a genuine outlier shift.

### Weaknesses (ordered by severity)

1. **Zero comparison against alternative pre-deployment diagnostics (CRITICAL).** This is the most glaring omission, and it was the #3 consensus issue in v1. The paper acknowledges this in Section 5.1: "We do not compare against alternative pre-deployment diagnostics (ensemble disagreement, native feature importance concentration, distribution shift statistics)." But the paper has 50 seeds of trained models -- ensemble disagreement is trivially computable. The paper uses LightGBM -- native `feature_importance(importance_type='gain')` gives a zero-cost concentration metric. The paper computes Jaccard -- Population Stability Index (PSI) is a direct generalization. Without these baselines, we cannot assess whether SHAP concentration adds value over simpler, cheaper alternatives. This is not about being exhaustive; it is about testing the obvious competitors that are already available from the existing experimental setup.

2. **Only LightGBM tested.** SHAP concentration is a model-dependent quantity. A random forest, XGBoost, or neural network trained on the same data could have entirely different concentration profiles. The paper's framework prescribes computing SHAP concentration as a diagnostic, but if the diagnostic changes with the model class, its utility is severely limited. Testing with even one additional model (e.g., RandomForest, which uses the identical SHAP TreeExplainer pipeline) would substantially strengthen the claim.

3. **Cross-domain "validation" adds no multiclass evidence.** All 3 cross-domain tasks are binary. The paper correctly identifies that binary APS has a ceiling effect. But this means the cross-domain analysis validates the ceiling effect observation, not the concentration diagnostic. The headline claim "rho=0.691, p=0.019 across 3 domains" obscures the fact that the concentration-coverage relationship is tested only within SALT. A practitioner reading this paper would believe it has been validated across domains; it has not, for the specific failure mode it identifies.

4. **The framework requires knowing the shift type in advance.** Step 2 checks concentration. Step 3 checks Jaccard of secondary features. But Jaccard requires knowing which features will shift at deployment time. The paper uses validation data as a proxy, but never validates whether train-validation Jaccard correlates with train-test Jaccard. In supply chain, COVID hadn't fully manifested during the validation period (Feb-Jul 2020) -- the test period (after Jul 2020) may have different shift characteristics. The framework's pre-deployment claim rests on the assumption that validation shift previews test shift, but this is never tested.

5. **Computational cost of SHAP not discussed.** The paper mentions "Standard CPU (8 cores, 8GB RAM). Wall-clock: ~3-4 hours for full 8-task, 50-seed suite." But how much of that is SHAP? If SHAP computation takes 80% of the time, and native feature importance gives similar results, the diagnostic is overengineered. TreeExplainer's exact computation for large ensembles can be very expensive at scale (millions of instances, thousands of features). The paper uses 10K subsamples for SHAP -- what is the sensitivity to subsample size?

6. **No deep learning models.** The paper focuses on LightGBM, which has well-defined feature importance via TreeExplainer. For neural networks, SHAP requires approximate methods (KernelSHAP, DeepSHAP) that are slower, noisier, and conceptually different. The framework's applicability to the deep learning ecosystem (where conformal prediction is increasingly deployed) is unclear.

7. **Retraining analysis is thin.** Tested on 1 of 3 severe tasks. Quarterly retraining achieves 41.1% coverage against a 90% target -- the model is still failing. Monthly retraining has a minimum of 0.6% (worse than no retraining). This suggests the retraining recommendation is unreliable. What does the retraining frequency look like for s-group and s-payterms?

8. **Missing: what should practitioners do when the diagnostic triggers?** The framework says "implement quarterly retraining" but quarterly retraining doesn't fix the problem (41.1% << 90%). A practitioner needs guidance beyond "retrain more often." Options like: switch to a model with less concentrated importance, engineer more stable features, expand the training window, use different conformal methods -- none are discussed.

### Questions for Authors

1. What is the Spearman correlation between LightGBM native feature importance concentration (top-1 gain / total gain) and coverage drop? This is computable with zero additional cost.
2. What is the Spearman correlation between ensemble disagreement (std of validation coverage across 50 seeds) and coverage drop?
3. How sensitive is SHAP concentration to the 10K subsample? What if you use 1K or 50K?
4. What is the wall-clock time for SHAP computation alone (excluding model training)?
5. Does the diagnostic apply to CQR (regression conformal) settings, or only APS (classification)?
6. What should a practitioner do when SHAP concentration is >40% and retraining doesn't help?

### Suggestions

1. Add a baseline comparison table: SHAP concentration vs. native FI concentration vs. ensemble disagreement vs. Jaccard alone. Rank by Spearman rho with coverage drop. This is the single most important addition.
2. Test with RandomForest (same TreeExplainer pipeline, ~2 hours of compute).
3. Discuss CQR (regression) generalization explicitly -- is there an analog to the binary ceiling effect?
4. Add a "what to do when retraining fails" discussion.
5. Report SHAP computation time separately from model training time.

---

## Reviewer 3: Statistician

**Score: 4/10 (Weak Reject)**
**Confidence: 5/5**

### Summary

The paper presents a rank correlation between SHAP concentration and APS coverage drop under COVID-19 distribution shift. The revision addresses several v1 issues (honest LOO reporting, data separation protocol, binary ceiling effect), but the core statistical concerns remain: n=8 for the primary claim with pseudo-replication, no multiplicity correction, uninformative confidence intervals, and the "nearly three orders of magnitude" claim based on a ratio with a noise-level denominator. The cross-domain extension to n=11 adds only binary tasks that do not test the primary hypothesis.

### Strengths

- **Paired Wilcoxon tests are appropriate.** Using seed-level pairing to test val vs. test coverage is the right approach given the experimental design.

- **Bootstrap CIs reported.** While wide, reporting them is good practice.

- **LOO stability analysis is well-executed.** Reporting all 8 (or 11) jackknife samples with their p-values is thorough. The honest disclosure that 2/8 lose significance is commendable.

- **Binary ceiling effect is statistically sound.** The Mann-Whitney test (U=22, p=0.024) for binary vs. multiclass drop difference is appropriate. The structural argument (binary APS sets are {0}, {1}, or {0,1}) is correct.

- **SHAP stability analysis is proper.** Bootstrap resampling of validation data to assess concentration stability (CV <1%) is the right approach.

### Weaknesses (ordered by severity)

1. **The effective sample size for the primary correlation is n=8, not n=50 (CRITICAL).** The paper's central claim (rho=0.833, p=0.010) is a Spearman correlation across 8 tasks. The 50 seeds provide replicated measurements within each task but do not increase the correlation sample size. This is acknowledged in the limitations but the abstract and introduction present the 50-seed figure prominently without emphasizing that the correlation is at n=8. At n=8, Spearman's exact distribution has very limited resolution (only 40,320 permutations). The bootstrap CI [0.29, 1.00] confirms this: a CI spanning 0.71 units is not informative. For UAI, a correlation claim at n=8 with CI [0.29, 1.00] is simply too uncertain to be a primary contribution.

2. **Pseudo-replication is acknowledged but never quantified (HIGH).** The paper states "Pseudo-replication means effective sample sizes for paired tests are smaller than 50" and "significance survives at effective n as low as 5." But no ICC, design effect, or effective sample size is actually computed. This is an assertion without evidence. The structure is: 50 seeds share identical training data, test data, and feature engineering; they vary only in LightGBM random seed and calibration/evaluation split. The intraclass correlation (ICC) across seeds within tasks would quantify this dependence. For s-office (std=0.000), the ICC is essentially 1.0 (all seeds give the same answer); the effective n for that task's paired test is close to 1. For s-group (std=0.323 on a mean of 0.124), seeds vary dramatically -- but is this real variation or aliasing of the discrete prediction problem? The paper must compute ICC and report honest effective sample sizes.

3. **Multiple testing not addressed (HIGH).** The paper tests:
   - 8 paired Wilcoxon tests (one per task, val vs. test coverage)
   - Spearman correlation at n=8
   - Spearman correlation at n=9 (COVID-era)
   - Spearman correlation at n=11 (combined)
   - Mann-Whitney test for binary vs. multiclass drops
   - Wilcoxon test for retraining (quarterly vs. none)
   - 5 alternative concentration metrics (top-2, top-3, HHI, entropy, C1) tested
   - 8 LOO jackknife tests
   - Threshold sensitivity across 7 values (25-55%)

   That is approximately 30+ statistical tests with no multiplicity correction. The p=0.010 for the primary correlation would become p=0.30 under a conservative Bonferroni correction across 30 tests. Even restricting to the 3 correlation tests and the 5 metric alternatives gives 8 tests, yielding Bonferroni-adjusted p=0.080 (no longer significant at alpha=0.05). The paper should at minimum apply Holm-Bonferroni to the family of correlation tests and report adjusted p-values.

4. **"Nearly three orders of magnitude" is statistically meaningless (HIGH).** The ratio 77.1/0.1 = 771 is driven by the denominator (0.1% drop for s-office). But s-office has near-perfect coverage (99.9% val, 99.9% test). The 0.1% "drop" is within measurement error -- with 50 seeds, the 95% CI is [0.0, 0.1], meaning the true drop could be exactly zero. Dividing by a quantity consistent with zero produces an arbitrarily large ratio. The claim "nearly three orders of magnitude" appears 3 times in the paper (abstract, Section 1, Figure 2 caption) and is misleading. A statistically honest characterization: "Coverage drops range from negligible (0.1%, consistent with zero) to catastrophic (77.1%)."

5. **The cross-domain extension is statistically weak (MODERATE-HIGH).** Adding 3 binary tasks (all with drops in [-1.3%, 2.9%]) to 8 multiclass tasks (drops in [0.1%, 77.1%]) creates a composite distribution with a structural confound: task type (binary vs. multiclass) perfectly predicts drop magnitude. The combined rho=0.691 is attenuated because binary tasks cluster at zero drop regardless of concentration. This is not cross-domain validation of the diagnostic; it is contamination of a within-domain signal by a structural boundary effect. The paper acknowledges this, but the abstract and introduction present "rho=0.691, p=0.019 across 3 domains (n=11)" as if it were independent evidence.

6. **The threshold analysis is in-sample (MODERATE).** Despite the data-separation protocol, the threshold sensitivity analysis (Table 8) evaluates thresholds against labels defined by the same data used to motivate the thresholds. The 40% threshold was chosen because there is a "natural gap" in the concentration distribution between 29% and 43%. But with n=8, any distribution will show gaps. The F1=0.75 (at 40%) and F1=0.86 (at 45%) are in-sample performance metrics that will not generalize. The cross-domain transfer test (F1=0.80 at 40%, n=11) is slightly better because 3 tasks are out-of-sample, but 2 of those 3 are correctly classified simply because they are below any reasonable threshold (20.8%, 36.8%) -- not because the threshold was well-calibrated.

7. **Confidence interval construction may be inappropriate (MODERATE).** The paper reports "95% CIs from t-distribution" for coverage values. Coverage is bounded [0, 1]. The t-distribution can produce CIs outside this range (e.g., s-group: test coverage CI [3.1, 21.7] is fine, but for tasks near 0 or 1, the CI may hit the boundary). A logit-transformed or Wilson interval would be more appropriate for bounded proportions.

8. **The "all p <= 0.005" claim obscures heterogeneity (LOW-MODERATE).** Seven tasks have p < 0.001 and one (i-shippoint) has p = 0.005. The latter has extreme variance (CV >50%). The blanket "all p <= 0.005" claim treats an uncertain result as equivalent to overwhelming evidence. Reporting individual p-values for all 8 tasks (as in Table 1) is fine; the summary claim should note the heterogeneity.

### Questions for Authors

1. What is the ICC (intraclass correlation coefficient) for test coverage across 50 seeds within each task? Please provide a table with per-task ICC and the implied effective n for each paired test.
2. What is the Bonferroni-adjusted or Holm-Bonferroni-adjusted p-value for the primary Spearman correlation (rho=0.833), accounting for the 5 concentration metrics tested and the 3 correlation subsets (n=8, n=9, n=11)?
3. For s-office (0.1% drop), what is the 95% CI for the drop using an appropriate bounded-proportion CI method (e.g., Wilson interval)?
4. The permutation p-value for n=8 Spearman is 0.0152 (from cross_domain_statistics.json) but the paper reports p=0.010 (scipy). Which is correct? Are you using the exact permutation distribution or the asymptotic approximation?
5. For the LOO analysis, which 2 tasks when removed cause significance loss? Is it always the same 2 tasks (s-payterms and s-shipcond, the two with concentration closest to the removed task)? If so, this suggests the correlation is driven by a binary separation rather than a monotonic trend.
6. Have you computed Cook's distance (or analogous leverage measure for rank correlations) for each of the 8 data points?

### Suggestions

1. Compute and report ICC per task. This is the single most important statistical improvement. Present a table: Task | Mean | Std | ICC | Effective n | Original p | Adjusted p.
2. Apply Holm-Bonferroni correction to the family of Spearman tests. Report both raw and adjusted p-values.
3. Replace "nearly three orders of magnitude" with "from negligible to catastrophic" throughout.
4. Use Wilson or Clopper-Pearson intervals for coverage proportions instead of t-distribution CIs.
5. Report exact permutation p-values (not asymptotic) for the n=8 Spearman correlation. At n=8, exact computation is trivial (8! = 40,320 permutations).
6. Explicitly name the 2 LOO tasks that lose significance and explain why (e.g., are they adjacent in concentration?).

---

## Reviewer 4: Causal/Experimental Design Expert

**Score: 5/10 (Borderline)**
**Confidence: 4/5**

### Summary

The revised paper reframes the study as an "observational case study" rather than a "quasi-natural experiment," which is more honest. The cross-domain extension and binary ceiling effect are genuine additions. However, the fundamental design limitations persist: the primary evidence comes from 8 correlated observations within a single database, the threshold is in-sample, and multiple confounders remain unaddressed. The paper has improved from v1 but not enough to change my assessment from borderline.

### Strengths

- **"Case study" framing is appropriate.** This is a significant improvement over the v1 "quasi-natural experiment" language. The paper now correctly positions itself as an observational study with exploratory findings.

- **Data separation protocol.** The explicit statement that "SHAP concentration and the 40% threshold are computed exclusively on validation-time data" and "Task-level severity labels (SEV/ROB) used for threshold evaluation are defined by observed test coverage drops; the threshold sensitivity analysis is therefore post-hoc validation, not prospective prediction" is exactly the kind of honesty that builds trust.

- **Binary ceiling effect is a substantive finding.** This is not just a limitation disclosure -- it identifies a structural property of binary APS that is independently useful. The Mann-Whitney test is appropriate.

- **Placebo test remains well-designed.** The pre-COVID baseline (2018/2019-H1/2019-H2) with 6-140x ratio confirms COVID as an outlier. This is the cleanest causal evidence in the paper.

- **Limitation section is comprehensive and honest.** Eight numbered limitations covering single database, threshold exploratory nature, binary ceiling, pseudo-replication, missing baselines, single model class, preliminary heuristics, and informal theory. This is unusually transparent.

- **COVID-era subset analysis.** The n=9 COVID-era analysis (rho=0.883, p=0.002) with all 9 LOO samples significant is the strongest result in the paper. It suggests the diagnostic may be specifically useful for event-driven shifts.

### Weaknesses (ordered by severity)

1. **The 8 SALT tasks are not independent observations (CRITICAL).** This was the #1 consensus issue in v1 and remains unresolved. The 8 tasks share:
   - The same underlying database (identical rows, overlapping features)
   - The same temporal split (Feb 2020 / Jul 2020)
   - The same exogenous shock (COVID-19)
   - The same model class (LightGBM with default hyperparameters)
   - The same conformal method (APS with alpha=0.1)
   - Overlapping feature sets (e.g., SALESDOC appears in multiple tasks)

   These shared structures induce correlations between the 8 observations that are not captured by the Spearman test. The effective degrees of freedom may be as low as 3-4 (roughly: SEV tasks with ID features, ROB tasks with entity features, and the 2 edge cases s-office and i-shippoint). A hierarchical model or clustered bootstrap would be more appropriate than treating tasks as independent.

2. **Confounders between concentration and coverage drop (HIGH).** The paper argues that SHAP concentration *predicts* coverage drop. But there are plausible confounders:
   - **Class cardinality**: The 3 SEV tasks have 45, 135, 462 classes; the 4 cleanly ROB tasks have 13, 13, 25, 35 classes. High cardinality mechanically allows larger coverage drops (more classes to exclude) and may independently cause higher concentration (more classes means sparser signal, often concentrated on fewer features). Spearman rho between num_classes and coverage_drop should be reported.
   - **Feature type**: SEV tasks use SALESDOC (transaction ID); ROB tasks use stable entities. The confound is: tasks with ID-type features have both high concentration AND low Jaccard AND high drops. SHAP concentration may be a proxy for "uses an ID feature" rather than a generalizable metric.
   - **Sample size per class**: High-cardinality tasks have fewer samples per class, making LightGBM less stable and more likely to concentrate on a single predictive feature. This creates a confound between class imbalance and SHAP concentration.

   The paper needs to test whether concentration adds predictive power *beyond* class cardinality and feature type. A partial correlation controlling for log(num_classes) would help.

3. **The cross-domain tasks do not control for confounders (HIGH).** The 3 cross-domain tasks are:
   - study-outcome (binary, COVID shift, 20.8% concentration, -1.3% drop)
   - driver-dnf (binary, NO shift, 48.1% concentration, 2.9% drop)
   - driver-top3 (binary, NO shift, 36.8% concentration, 1.2% drop)

   Two of 3 cross-domain tasks have NO distribution shift by design (rel-f1, 2005-2010). Of course they show near-zero drop -- there is no shift to cause a drop. This is a *negative control*, not a *validation*. Including negative controls (no shift) alongside positive controls (severe shift) in the same correlation inflates rho because you are correlating a mix of "shift absent" and "shift present" conditions. The proper test is: among tasks that experience similar shift severity, does concentration predict differential failure? The cross-domain data cannot answer this because no cross-domain multiclass task with severe shift is available.

4. **Garden of forked paths is still not addressed (MODERATE-HIGH).** The paper tested 5 concentration metrics (top-1, top-2, top-3, HHI, entropy). Only top-1 was significant. This is a 5-way multiple test. Additionally, the paper tested 3 correlation subsets (n=8, n=9, n=11) and 7 threshold values. The p=0.010 should be adjusted. R3 from v1 raised this; the paper adds the comment "all alternatives are non-significant (p>0.10)" but does not apply multiplicity correction to the selected metric.

5. **Threshold circularity is only partially resolved (MODERATE).** The paper now states the threshold is "exploratory" and "derived from n=8 multiclass tasks." But the cross-domain transfer test ("Applying the 40% threshold without re-tuning to the full n=11 cross-domain set yields Recall=1.0, F1=0.80") treats this as validation. Looking at the data: all 3 new tasks have near-zero drops (binary ceiling). At ANY threshold, the recall for severe tasks is trivially 1.0 because all severe tasks are SALT tasks that informed the threshold. The only interesting question is false positives, and 1 of 3 new tasks (driver-dnf, 48.1% concentration) is flagged as vulnerable despite being robust (no shift). This is not validation; it is classification of negative controls.

6. **Selection bias in task inclusion is still unexplained (MODERATE).** Are all SALT classification tasks included? The paper uses 8 tasks but does not state whether these are ALL available classification tasks in rel-salt or a subset. If any tasks were excluded (e.g., due to data quality, computational issues, or inconvenient results), this must be disclosed. Looking at the task names (s-shipcond, s-group, s-payterms, i-plant, i-shippoint, s-incoterms, i-incoterms, s-office), these appear to be all SALT classification tasks based on the RelBench documentation, but the paper should confirm explicitly.

7. **Retraining claim is still overstated (LOW-MODERATE).** The abstract says "Quarterly retraining restores vulnerable task coverage by +19 pp (p=0.04)." But 41.1% coverage << 90% target. "Restores" implies returning to acceptable levels. "Partially mitigates" is more accurate. Additionally, this is tested on 1 of 3 severe tasks.

8. **Protective-factor heuristic is based on 1 example (LOW).** The thresholds (Jaccard > 0.5, importance > 15%) are derived from sales-office. N=1 is not validation. The paper should either remove specific thresholds or label them as "illustrative."

### Questions for Authors

1. What is the Spearman correlation between num_classes (or log(num_classes)) and coverage drop? If significant, what is the partial correlation of concentration with drop, controlling for num_classes?
2. Are all SALT classification tasks included? If any were excluded, why?
3. For the cross-domain analysis, if you restrict to tasks with actual distribution shift (n=9 COVID-era), what is the rho? [I see this is reported as 0.883 -- but note that 8/9 are SALT tasks.]
4. Can you compute a permutation-based p-value that respects the hierarchical structure (bootstrap entire tasks, not individual seeds)?
5. What is the correlation between Jaccard alone (of the top-SHAP feature) and coverage drop? If Jaccard of the top feature predicts failure equally well, SHAP concentration reduces to a proxy for "identifies which feature to check Jaccard for."
6. Is there a task where SHAP concentration is low (<30%) but the top feature has low Jaccard (<0.05)? This would be the clean test case for whether concentration matters beyond feature overlap.

### Suggestions

1. Report partial Spearman correlation controlling for log(num_classes). If concentration remains significant after this control, it substantially strengthens the causal argument.
2. State explicitly: "All 8 classification tasks from the rel-salt benchmark are included; no tasks were excluded."
3. Add a "Confounders" subsection in Discussion that addresses class cardinality, feature type, and sample size per class.
4. Replace "restores" with "partially mitigates" in the abstract.
5. Consider a hierarchical bootstrap (resample tasks, not seeds) for the n=8 correlation CI.
6. For the protective-factor heuristic, replace specific thresholds with qualitative guidance: "Check whether high-importance secondary features have stable train-validation overlap."

---

## Consensus Issues (v2)

### Issue 1: No comparison against alternative pre-deployment diagnostics (4/4 agree, UNCHANGED from v1)

This was the #3 consensus issue in v1 and remains completely unaddressed. The paper has 50 seeds of LightGBM models -- ensemble disagreement (validation coverage variance across seeds) is trivially computable. LightGBM provides native `feature_importance(importance_type='gain')` -- concentration from this is zero-cost. These are the obvious baselines that any reviewer would expect. The paper's Section 5.1 now explicitly acknowledges this gap and promises "future work," but this is not acceptable for a paper whose central contribution is a specific diagnostic metric.

**Minimum action**: Compute (a) ensemble disagreement (std of val coverage across 50 seeds) and (b) native LightGBM feature importance concentration as Spearman correlates with coverage drop. Report in a comparison table.

### Issue 2: Effective sample size never quantified (3/4 agree, UNCHANGED from v1)

The pseudo-replication concern was raised by R1, R2, R4 in v1. The paper added a limitation acknowledging it but did not compute ICC or effective n. The assertion "significance survives at effective n as low as 5" appears without computation.

**Minimum action**: Compute ICC per task from 50-seed data. Report effective n. Recalculate p-values at effective n for the task with the weakest signal (i-shippoint, p=0.005).

### Issue 3: n=8 primary evidence from single database (4/4 agree, MARGINALLY IMPROVED)

The cross-domain extension adds 3 binary tasks, raising to n=11. But the binary tasks are structurally different (ceiling effect) and 2/3 have no shift. The effective new evidence for the multiclass diagnostic is n=0 cross-domain tasks. The COVID-era subset (n=9) adds 1 binary task.

**Status**: Acknowledged by authors; the binary ceiling effect is itself an interesting finding. But the fundamental limitation of n=8 multiclass observations from 1 database persists. Adding multiclass tasks from other domains (e.g., WILDS benchmarks) would be transformative.

### Issue 4: Theory still listed as contribution (3/4 agree, PARTIALLY IMPROVED)

The section is now titled "Intuition" and uses "heuristic" language. But it is still listed as contribution #2: "Mechanistic intuition: We provide a heuristic argument..." This is an improvement over v1 ("theoretical grounding") but R1 and R3 still object to calling it a contribution when no empirical verification (CDF plots, KS tests) is provided for the stochastic dominance claim.

**Minimum action**: Either add CDF plots verifying stochastic dominance, or reframe contribution #2 as part of contribution #1 (the diagnostic finding includes intuition for why it works).

---

## What Would Flip Each Reviewer?

### R1 (Conformal Prediction Expert) -> 6/10 Weak Accept

Needs **all 3**:
1. Add conformity score CDF plots (cal vs. test) for 1 catastrophic + 1 robust task with KS test statistics
2. Fix APS formulation (include or explicitly exclude randomization) and cite Barber et al. (2023) bound quantitatively
3. Add at least 1 baseline comparison (ensemble disagreement OR native FI concentration)

### R2 (ML Practitioner) -> 6/10 Weak Accept

Needs **2 of 4**:
1. Baseline comparison table: SHAP concentration vs. native FI concentration vs. ensemble disagreement
2. Test with 1 additional model class (RandomForest is minimal effort)
3. Add at least 2 multiclass cross-domain tasks with genuine shift
4. Provide practical guidance for when retraining fails

### R3 (Statistician) -> 5/10 Borderline (very hard to flip further without more data)

Needs **all 3**:
1. Compute and report ICC per task with effective n
2. Apply Holm-Bonferroni correction to the family of correlation and metric-selection tests
3. Replace "nearly three orders of magnitude" with a statistically defensible characterization

Even with all 3, the n=8 limitation and [0.29, 1.00] CI make this fundamentally underpowered for the central claim. Would need n>=15 multiclass tasks to reach Borderline Accept.

### R4 (Causal/Experimental Design) -> 6/10 Weak Accept

Needs **2 of 3**:
1. Report partial correlation controlling for log(num_classes) -- if concentration remains significant, this substantially addresses the confounding concern
2. Confirm all SALT classification tasks included (no selection bias) + add class cardinality as a covariate
3. Replace "restores" with "partially mitigates" and add confounders discussion

---

## New Issues Introduced by Revision

These concerns were not present in v1 but arise from the revision's additions:

1. **Inconsistency between paper and data.** The paper states "permutation p=0.023" for the combined correlation. The JSON data shows permutation_p=0.0227, scipy_p=0.0186. The paper text says "Spearman rho=0.691, p=0.019." It is unclear whether p=0.019 is scipy (0.0186 rounded) or something else. The permutation p (0.023) is more conservative and should be the primary report at n=11.

2. **The COVID-era subset (n=9) is almost identical to SALT-only (n=8).** Adding study-outcome (binary, -1.3% drop, 20.8% concentration) barely changes the correlation (0.833 -> 0.883). Presenting this as a separate analysis with its own p-value (p=0.002) creates an appearance of independent evidence that does not exist. The marginal information content of the 9th task is near zero.

3. **The abstract mentions "rho=0.691, p=0.019 across 3 domains" before "rho=0.833, p=0.010 within supply chain."** This presentation order suggests the cross-domain result is the weaker of two findings, but a reader encountering the abstract first will take the cross-domain number as the primary result. Since the cross-domain correlation is attenuated by the binary ceiling effect (a confound, not evidence), the stronger within-domain result should be primary.

4. **Table 5 includes rel-f1 (pre-COVID 2005-2010) under "Cross-Domain Validation."** But rel-f1 has NO distribution shift by design. Including tasks without shift in a study of shift vulnerability is like including healthy controls in a disease severity correlation -- it inflates the correlation by adding extreme-value observations that conform to the null hypothesis. The shift column correctly labels this as "None," but the correlation computation treats these as evidence for the diagnostic.

5. **The "COVID-era subset" framing is ambiguous.** The paper defines COVID-era as "8 SALT + study-outcome" (n=9). But the SALT test period (after Jul 2020) may capture post-COVID normalization rather than peak COVID disruption. The clinical trial test period (after Jan 2021) captures vaccine rollout. These are different COVID eras with different shift characteristics. Lumping them as "COVID-era" obscures this heterogeneity.

---

## Overall Assessment

The paper has improved meaningfully from v1. The "case study" framing, data separation protocol, binary ceiling effect, cross-domain extension, and honest limitation reporting are all positive changes. The entropy paradox and ACI utility analysis remain genuinely useful findings.

However, the revision has not addressed the three most impactful weaknesses:
1. **No baseline comparisons** (the single easiest improvement with the highest potential impact)
2. **No ICC/effective sample size quantification** (the single most important statistical improvement)
3. **No multiclass cross-domain evidence** (the hardest to address but most transformative)

The paper sits at the boundary between "well-executed exploratory analysis" and "UAI contribution." The core idea (pre-deployment diagnostic via feature importance structure) is sound and practically important. But the evidence base is too narrow (n=8 multiclass from 1 database) and the evaluation is incomplete (no baselines, no ICC, no CDF verification) for a top venue.

**Recommendation**: Borderline overall. With the baseline comparisons (improvement #1 above, ~half day of work) and ICC computation (~2 hours), the paper would move to Weak Accept for 3 of 4 reviewers. The multiclass cross-domain gap is harder to address but the binary ceiling effect finding partially compensates by explaining why the gap matters.

**Estimated score after addressing minimum viable improvements**: 5.5-6.0 (Borderline to Weak Accept)
