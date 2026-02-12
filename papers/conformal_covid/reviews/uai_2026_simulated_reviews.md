# UAI 2026 Simulated Review Panel
## "Diagnosing Conformal Prediction Failures Under Distribution Shift: A COVID-19 Case Study"

**Date**: 2026-02-10
**Method**: 4 parallel Opus agents with distinct reviewer personas
**Paper**: `papers/conformal_covid/uai_2026/main.tex`

---

## Panel Summary

| Reviewer | Role | Soundness | Significance | Novelty | Clarity | Score | Recommendation |
|----------|------|-----------|-------------|---------|---------|-------|----------------|
| R1 | Theory/Methods | 2/4 | 3/4 | 2/4 | 3/4 | **5/10** | Weak Reject |
| R2 | Empirical/ML | 2/4 | 2/4 | 3/4 | 3/4 | **4/10** | Weak Reject |
| R3 | Applications/Practice | 3/4 | 2/4 | 2/4 | 3/4 | **5/10** | Borderline |
| R4 | Causal/Statistical (AC) | 2/4 | 2/4 | 3/4 | 3/4 | **4/10** | Weak Reject |
| **Mean** | | **2.25** | **2.25** | **2.5** | **3.0** | **4.5** | **Weak Reject** |

---

## Consensus Issues (4/4 agree)

### 1. n=8 from a single database is the core weakness

Every reviewer flags this as the paper's fundamental limitation.

- **R1**: "The paper's central claim is about SHAP concentration as a diagnostic, but it is validated on exactly one scenario."
- **R2**: "The entire paper rests on a rank correlation with n=8 data points. At this sample size, the bootstrap 95% CI is [0.29, 1.00], which spans from 'weak' to 'perfect' correlation -- this is not informative."
- **R3**: "The primary results are on 8 tasks from one database (SALT). A practitioner in NLP, computer vision, or finance would struggle to extrapolate."
- **R4**: "These are 8 views of the same dataset under the same shock, not 8 independent experiments. A Spearman correlation at n=8 with shared structure is far less informative than n=8 truly independent experiments."

**Action**: Multi-domain validation is the #1 priority. R4 specifies: "at least 3 different domains, each with multiple tasks, using the same threshold without re-tuning."

### 2. Section 4 "theoretical grounding" is informal and shouldn't be called a contribution

- **R1**: "The stochastic dominance claim (Eq. 3) is stated as a displayed equation but never proven. For UAI, this section should either be formalized as a proper proposition with assumptions and proof, or honestly labeled as 'heuristic motivation.'"
- **R2**: "The 'Proposition (informal)' is labeled as such, but the paper then repeatedly cites this as 'theoretical grounding' in the abstract and conclusion. An informal argument is not a theorem."
- **R3**: "This is fine as motivation but should not be listed as a 'contribution.'"
- **R4**: "If this were removed entirely and the paper framed as purely empirical, it would be stronger -- the informal theory creates expectations of rigor that are not met."

**Action**: Either formalize under explicit assumptions with proof, or demote to "heuristic motivation" and remove from contributions list.

### 3. Missing baselines

- **R1**: Ensemble disagreement from 50 seeds (obvious pre-deployment baseline already computed), feature distribution statistics (PSI, KL), calibration score statistics
- **R2**: MMD, domain classifier, feature drift monitors (PSI, KS), prediction disagreement, calibration curve analysis
- **R3**: Native LightGBM `feature_importance(importance_type='gain')` (zero-cost), permutation importance, simple ID-feature check
- **R4**: (Implicitly agrees via "what would change my mind")

**Action**: At minimum, compute (a) ensemble disagreement and (b) native feature importance concentration. Both are trivially available from existing data.

### 4. Threshold circularity / in-sample validation

- **R1**: "The protective-factor heuristic is justified by exactly one example (sales-office). This is not a validated rule; it is an ad-hoc fix for a single false positive."
- **R2**: "With n=8, any partition of 8 numbers will show 'gaps.' The paper selects 40% but the better-performing threshold is 45%."
- **R3**: "A practitioner deploying conformal prediction on medical imaging or NLP would have no basis for using 40%."
- **R4**: "The labels were assigned after observing the test data, and the threshold was chosen to separate the observed distribution. This is textbook overfitting to your own labels."

**Action**: Either hold out tasks for prospective validation, or drop the F1 claim and present the threshold as exploratory.

### 5. Cross-domain "validation" doesn't validate SHAP concentration

- **R1**: "The cross-domain experiments do not validate the concentration metric at all -- they validate feature overlap, which is a different (and already known) predictor."
- **R2**: "The moderate-shift group shows no signal (p=0.368), and the method is tested on one model class only."
- **R3**: "The cross-domain validation is too thin to support the general claims implicit in the framework."
- **R4**: "This section validates Jaccard, not the paper's main contribution."

**Action**: Compute SHAP concentration for cross-domain tasks and test correlation, or rename the section honestly.

---

## Majority Issues (3/4 agree)

### 6. Pseudo-replication not adequately quantified (R1, R2, R4)

The 50 seeds share identical training/test data. The claim "significance survives effective n as low as 5" is asserted without computation.

- **R2**: "Provides no formal analysis of what the effective sample size actually is (e.g., via intraclass correlation or cluster-robust inference)."
- **R4**: "The paper should formally discuss what the effective degrees of freedom are under this dependence structure."

**Action**: Compute ICC across 50 seeds per task. Report honest effective n.

### 7. "Quarterly retraining restores coverage" is misleading (R2, R3, R4)

- **R3**: "41.1% mean coverage still fails the 90% target by a wide margin. Quarterly retraining is not 'restoring' coverage -- it is improving it from catastrophic to merely bad."
- **R4**: "Tested on only 1 task. The difference between quarterly (41.1%) and bi-annual (27.0%) is not directly compared."
- **R2**: "The other two severe tasks (s-group, s-payterms) are not tested."

**Action**: Temper language ("partially mitigates" not "restores"), test on all 3 severe tasks, or caveat as single-task finding.

### 8. Jaccard equation inconsistency (R1, R2, R3)

Eq. 1 defines train-test Jaccard but the framework (line 384) uses train-validation Jaccard. The proxy is never validated.

**Action**: Add a separate definition for validation Jaccard, or validate the proxy correlation.

---

## Split Opinions (judgment calls)

| Issue | Positive View | Negative View |
|-------|--------------|---------------|
| **Enough for UAI?** | R3: "genuine practical need, admirable rigor" | R2: "well-executed pilot study, not a UAI paper" |
| **Entropy paradox value** | R1, R3, R4: "genuinely interesting, deserves more prominence" | R2: acknowledges but doesn't weight heavily |
| **ACI analysis** | R3, R4: "practically very useful, important negative result" | R1, R2: useful but secondary |
| **Honesty as strength** | All 4 praise the limitations section | R4: "honesty alone doesn't fix the evidence gap" |
| **Novelty** | R2, R4 give 3/4: "creative reframing of SHAP" | R1, R3 give 2/4: "intuitive combination of existing tools" |

---

## What Would Flip Each Reviewer?

### R1 (Theory/Methods) -> Weak Accept
1. Formalize the stochastic dominance theory under explicit assumptions
2. Add ensemble disagreement as pre-deployment baseline
3. Validate on 1+ additional dataset with severe shift

### R2 (Empirical/ML) -> Weak Accept
1. Validate on 2-3 additional datasets with genuine distribution shift
2. Test multiple model classes (random forest, neural net)
3. Compare against MMD, domain classifiers, feature drift baselines
4. Quantify effective sample size under pseudo-replication

### R3 (Applications/Practice) -> Weak Accept
1. Compare against native feature importance concentration (zero-cost baseline)
2. Provide guidance for threshold calibration in new domains
3. Discuss adaptation to streaming settings

### R4 (Causal/Statistical) -> Weak Accept (needs 2 of 4)
1. 3 independent domains, same threshold, no re-tuning
2. Formal proof or rigorous simulation for stochastic dominance
3. Pre-registration or prospective validation of 40% threshold
4. Report effective degrees of freedom via hierarchical model

---

## Minimum Viable Revision (prioritized)

### Cheap (editorial + existing data)
1. Demote Section 4 to "heuristic motivation", remove from contributions
2. Compute ensemble disagreement from 50-seed data as baseline
3. Compute native feature importance concentration as zero-cost baseline
4. Compute ICC to report effective sample size
5. Fix Jaccard notation inconsistency (Eq. 1 vs. framework)
6. Temper retraining language ("partially mitigates" not "restores")
7. Report which 2 LOO tasks lose significance + Cook's distances

### Moderate effort
8. Compute SHAP concentration for cross-domain tasks
9. Drop "quasi-natural experiment" framing (use "observational case study")
10. Present threshold as exploratory, drop F1 as performance metric

### Significant effort (but highest impact)
11. Validate on 1-2 more shift datasets (WILDS benchmarks?)
12. Test with random forest or XGBoost (same SHAP pipeline)
13. Add conformity score CDF plots (cal vs. test) for 1 catastrophic + 1 robust task

---

## Individual Reviews

### R1: Theory/Methods (Score: 5/10, Weak Reject, Confidence: 4/5)

**Strengths:**
1. Practically important and well-motivated question
2. Honest and transparent reporting
3. Entropy paradox finding is genuinely interesting
4. Placebo test is well-designed
5. Comprehensive experimental coverage
6. Clear data separation protocol

**Weaknesses:**
1. Theoretical grounding (Section 4) is informal plausibility argument, not theorem. Stochastic dominance (Eq. 3) never proven. APS definition imprecise (omits randomization). "Diversification protects" is hand-waving.
2. Spearman rho at n=8: unclear if exact permutation or asymptotic p-value. Bootstrap CIs unreliable at n=8 due to ties in resampled ranks.
3. Pseudo-replication more severe than acknowledged. 50 seeds are repeated measurements, not independent replications.
4. "Exchangeability breaking" is technically imprecise — exchangeability is trivially violated by temporal split. The real question is *magnitude* of degradation.
5. Single dataset, model class, shift type. Cross-domain validation doesn't test SHAP concentration.
6. 40% threshold is fragile and in-sample. Protective-factor heuristic based on 1 example.
7. Missing comparison with ensemble disagreement, feature distribution statistics.

**Key Questions:**
- Is p=0.010 from exact permutation or asymptotic approximation?
- Can you provide a worked example verifying stochastic dominance?
- Can you construct a counterexample where uniform importance still fails?
- Why not use ensemble disagreement as baseline?
- How well does train-validation Jaccard approximate train-test Jaccard?
- SHAP computed on COVID-period validation data — does it already encode shift information?

---

### R2: Empirical/ML (Score: 4/10, Weak Reject, Confidence: 4/5)

**Strengths:**
1. Well-motivated pre-deployment vs. post-hoc distinction
2. Honest limitations section
3. Informative 50-seed experimental design
4. Placebo test well-designed
5. ACI analysis thorough with Utility framing
6. SHAP stability (CV < 1%) reported

**Weaknesses:**
1. n=8 insufficient for central claim. Bootstrap CI [0.29, 1.00] uninformative. 2/8 LOO samples lose significance.
2. Pseudo-replication: no ICC or effective sample size computed.
3. Single model class (LightGBM) — SHAP concentration is model-dependent.
4. Single dataset with shared structure — 8 views of same shift, not 8 experiments.
5. Missing baselines: MMD, domain classifier, PSI, KS test, ensemble disagreement.
6. Moderate shift group (n=4, p=0.368) weakens story. Diagnostic requires knowing shift severity in advance — circular.
7. Retraining on single task not generalizable.
8. Placebo test seed count inconsistency.
9. Missing figures: conformity score CDFs for stochastic dominance claim.
10. Theory is informal. Eq. 1 train-test vs. framework train-validation inconsistency.
11. 40% threshold not well justified (45% gives better F1).

**Key Questions:**
- How to validate without already knowing which tasks fail?
- ICC across 50 seeds?
- Results with random forest or neural network?
- MMD comparison?
- Why not retraining on all 3 severe tasks?
- How to determine shift severity before deployment?
- Why 40% over 45%?
- Show actual conformity score distributions?

---

### R3: Applications/Practice (Score: 5/10, Borderline, Confidence: 4/5)

**Strengths:**
1. Clear practical framing of real deployment problem
2. "Confidently wrong" observation genuinely useful for ML monitoring
3. Honest limitations section
4. Statistical rigor exceeds minimum requirements
5. ACI analysis practically informative
6. Placebo test strengthens causal interpretation
7. Reproducibility details thorough

**Weaknesses:**
1. 40% threshold not transferable to new domains. 45% gives better F1.
2. SHAP cost vs. simpler diagnostics (native feature importance, ID check) unclear.
3. "Quarterly retraining" ignores modern drift-triggered retraining. Still only 41.1% coverage.
4. Framework is batch-only, silent on streaming/online settings.
5. Missing ML monitoring tool comparison (Evidently.ai, NannyML, WhyLabs, etc.).
6. Useful/Marginal/Vacuous thresholds (40%/60%) arbitrary and not task-dependent.
7. Supply chain domain narrow; cross-domain validation weak.
8. Theory section informal, shouldn't be a listed contribution.
9. Train-validation Jaccard proxy untested.
10. No sensitivity to model class.

**Key Questions:**
- Native feature importance vs. SHAP concentration comparison?
- What to do when retraining still fails (41.1% << 90%)?
- How to calibrate threshold for new domains?
- Train-validation vs. train-test Jaccard correlation?
- SHAP concentration for cross-domain clinical trial tasks?
- Conformal calibration coverage as simpler baseline?
- Results with RAPS or THR instead of APS?

---

### R4: Causal/Statistical — Area Chair (Score: 4/10, Weak Reject, Confidence: 4/5)

**Strengths:**
1. Clear and important problem statement
2. Honest reporting of limitations
3. Entropy paradox genuinely interesting
4. Placebo test well-designed
5. ACI analysis thorough and practically useful
6. Protective-factor concept shows thoughtful domain knowledge

**Weaknesses:**
1. Effective sample size problem fundamental. 8 tasks share database, split, model, pipeline, overlapping features. Need hierarchical modeling or clustered SEs.
2. "Quasi-natural experiment" framing misleading. No control group, no random assignment, confounded with time. Should be "observational case study."
3. SEV/ROB classification circular. Labels defined by test outcomes, threshold "validated" against them = textbook overfitting.
4. Garden of forked paths: 5 concentration metrics x 7 thresholds x 2 stratifications tested. Only top-1 significant. No multiplicity correction.
5. Theory too informal for UAI contribution.
6. LOO reveals fragility: which 2 tasks drive significance? Cook's distances needed.
7. Cross-domain validation validates Jaccard, not SHAP concentration.
8. Retraining analysis: monthly worse than no-retraining (min 0.6%), quarterly still fails 90% target. "Restores coverage" misleading.
9. Selection bias: are all SALT classification tasks included?
10. Jaccard measures value-set overlap, not distributional similarity.

**Key Questions:**
- How many total SALT classification tasks? Any excluded?
- Which 2 LOO tasks lose significance? Cook's distances?
- Actual p-values for top-2, top-3, HHI, entropy?
- Formal proof or empirical CDF verification of stochastic dominance?
- Train-validation vs. train-test Jaccard correlation?
- ICC of coverage across seeds within tasks?
- SHAP concentration for cross-domain tasks?
- Protective-factor thresholds (Jaccard>0.5, importance>15%) also post-hoc?

**What would flip to Weak Accept (any 2 of 4):**
1. 3 independent domains, same threshold, no re-tuning
2. Formal proof or rigorous simulation for stochastic dominance
3. Pre-registration or prospective validation of threshold
4. Effective degrees of freedom via hierarchical model
