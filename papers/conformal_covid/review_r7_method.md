# MethodCritic Analysis Report -- Round 7

**Paper**: Diagnosing Conformal Prediction Failures Under Distribution Shift: A COVID-19 Case Study
**File**: `/Users/i767700/Github/ai-in-finance/papers/conformal_covid/uai_2026/main.tex`
**Date**: 2026-02-20

---

## Executive Summary

The paper is methodologically mature after six revision rounds. The primary correlation claim (rho=0.853, n=16, p<0.001) is supported by data, the theorem bounds verify numerically, and the multi-seed design is rigorous. However, three issues from R6 remain incompletely resolved: (1) the "default LightGBM settings" claim is **factually false** -- 4 of 5 listed hyperparameters deviate from LightGBM defaults; (2) class counts in Table 1 still disagree with the 50-seed summary JSON; and (3) the dual classification scheme (SEV/ROB vs At-risk) creates a labeling contradiction for i-shippoint. Additionally, I identify a new moderate issue: sub-nominal validation coverage in multiple external datasets indicates calibration instability that is not discussed.

---

## Prior R6 Issue Verification

### M1: Table 1 class counts vs JSON -- CONFIRMED, STILL PRESENT

**Table 1** reports: s-group = 462, s-payterms = 135, i-shippoint = 70.
**`ensemble_50seeds_summary.json`** reports: s-group = 459, s-payterms = 137, i-shippoint = 69.

Per-seed RAPS data confirms class counts vary across seeds (s-group: 455--465; s-payterms: 133--138; i-shippoint: 69--74). The mode across 10 RAPS seeds matches Table 1 (462, 135, 70), but the 50-seed summary JSON uses a different seed's count. Neither source is inherently wrong -- class counts genuinely vary per calibration/train split -- but the paper never explains which seed or aggregation method produces the reported counts, and the two canonical data files disagree.

**Severity**: MINOR (the variation is real and does not affect results; simply needs a footnote or consistent source)

### M3: Binary datasets -- RESOLVED

The abstract and Section 3.1 now correctly state "9 additional datasets" / "9 non-supply-chain domains" / n=16 multiclass primary endpoint. Stack Overflow is properly excluded from the multiclass set. No unnamed binary datasets remain.

### M4: Single-seed vs multi-seed rho -- ADEQUATELY HANDLED

Table 3 footnote dagger discloses: "Single-seed external values; multi-seed-consistent value at n=11 is rho=0.818, p=0.002." Appendix C.7 provides full explanation of KDDCup99's role in the reduction. The primary endpoint uses consistent multi-seed means. This is transparent and well-handled.

### M5: i-shippoint ROB* vs At-risk* -- CONFIRMED, STILL PRESENT

Table 1 classifies i-shippoint as **ROB*** (threshold: <20% drop; mean drop = 18.5%).
Table 6 classifies i-shippoint as **At-risk*** (threshold: >15pp drop; same mean drop = 18.5%).

The same task receives opposite labels in different tables because two incompatible threshold schemes coexist:
- Table 1: SEV (>50%), ROB (<20%), with i-shippoint at 18.5% -> ROB*
- Table 6 / threshold sensitivity: At-risk (>15pp), with i-shippoint at 18.5pp -> At-risk*

A reader encountering both tables will be confused about whether i-shippoint is considered a success or failure case. The asterisk footnotes partially mitigate this but the contradiction is structural.

**Severity**: MODERATE -- affects reader comprehension and Table 1/Table 6 consistency

### B6: "Default LightGBM settings" -- CONFIRMED FALSE

**Section 3.2 (line 102)**: "Train LightGBM classifier with default hyperparameters"
**Appendix A.1 (line 415)**: "All models trained with default LightGBM settings: ... learning_rate=0.05, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5"

**Actual LightGBM defaults** (verified against LightGBM documentation):
| Parameter | Paper/Code | LightGBM Default |
|-----------|-----------|-----------------|
| learning_rate | 0.05 | **0.1** |
| num_leaves | 31 | 31 |
| feature_fraction | 0.8 | **1.0** |
| bagging_fraction | 0.8 | **1.0** |
| bagging_freq | 5 | **0** |

Four of five listed parameters are non-default. The claim "default LightGBM settings" is **factually incorrect** and appears in both the main text and appendix. This is misleading because:
1. A reader trusting "defaults" would not attempt to reproduce with these parameters
2. The regularization choices (subsampling, reduced learning rate) could affect concentration estimates by changing which features the model relies on
3. "No task-specific tuning" is separately stated and remains true, but "default settings" is a different claim

**Severity**: MAJOR -- affects reproducibility and is a factual error

---

## Fresh Findings (New in R7)

### N1: Sub-Nominal Validation Coverage in External Datasets

Several external datasets show validation coverage substantially below the 90% target across multiple seeds:

**Gas Sensor**: 4/10 seeds have val coverage < 90% (range: 87.97--98.31%), mean 92.14%
**Avila**: 5/10 seeds have val coverage < 90% (range: 87.34--92.71%), mean 89.90%

Sub-nominal validation coverage means the conformal predictor is not achieving its guarantee even *before* shift. When the baseline is already impaired, coverage "drops" become harder to interpret: a task showing val=88% and test=88% has zero drop but was never properly calibrated.

The paper does not discuss validation coverage quality for external datasets, focusing exclusively on the drop metric. For Gas Sensor (negative drop = test > val), the "robust" classification is partly an artifact of under-calibration at validation time -- had calibration been tighter, the gap could be different in either direction.

This does not invalidate the primary correlation (which uses drops consistently), but it weakens the interpretive claim about Gas Sensor and Avila being "correctly unflagged as robust."

**Severity**: MODERATE -- interpretive gap for external validation claims

### N2: Covertype / Satimage Domain Overlap

The paper claims "9 non-supply-chain domains" for 9 external datasets. However:
- **Covertype**: Forest cover type classification from cartographic/remote sensing variables
- **Satimage**: Landsat satellite imagery soil classification

Both are geospatial remote sensing classification tasks. Counting them as separate domains inflates the claimed domain diversity. At minimum this is arguable; a conservative count would be 8 external domains. This does not affect the n=16 task count or the correlation computation (which operates on tasks, not domains), but the "9 domains" framing overstates generalizability breadth.

**Severity**: MINOR -- presentational rather than substantive

### N3: Retraining p-value in Abstract

The abstract states: "suggest that quarterly retraining may partially recover coverage when vulnerable (+19 pp, p = 0.04 (unadjusted))."

Section 5.4 properly discloses that Holm correction over 3 tasks yields p=0.12 (non-significant). However, the abstract reports only the unadjusted p-value. While "(unadjusted)" is technically present, the abstract is where most readers form their first impression. Reporting an unadjusted p-value in the abstract while the corrected value is non-significant in the body is a transparency concern.

**Severity**: MODERATE -- the corrected p-value should appear alongside the unadjusted one, or the abstract should note non-significance after correction

### N4: KDDCup99 Actual Category Inconsistency Across JSONs

`kddcup99_validation.json` (single-seed): actual_category = "robust" (drop = -0.83%)
`external_multiseed_validation.json`: actual_category = "severe" (mean drop = 15.85%)

The same dataset is labeled "robust" in one file and "severe" in another. While the paper correctly uses the multi-seed mean, the single-seed JSON remains in the results directory with a contradictory label. This is a housekeeping issue but could confuse anyone auditing the data artifacts.

**Severity**: MINOR -- data artifact inconsistency, not paper text issue

### N5: Shuttle Concentration Instability Not Fully Discussed

Shuttle's per-seed concentration ranges from 19.97% to 46.61% (std = 8.29%), with the top feature changing across 6 different features across 10 seeds (A1, A3, A5, A6, A7, A8, A9). This is the most extreme concentration instability of any dataset. The seed stability protocol (Section 7) discusses KDDCup99 (std=7.51%) but Shuttle (std=8.29%) is more unstable and is not mentioned by name. Shuttle's robustness is correct (drop < 1% regardless), but the instability is relevant to the seed stability protocol discussion.

**Severity**: MINOR

### N6: RAPS Table Reporting Uses 10-Seed but 50-Seed Available

Table 8 (RAPS appendix, line 646) reports "10-seed means +/- std" and notes differences from 50-seed Table 1. The RAPS experiment has 10 seeds but the primary experiment has 50. This asymmetry means RAPS comparisons have less statistical power and wider CIs. The paper acknowledges the difference ("APS drops differ from 50-seed Table 1") but does not explain why only 10 seeds were used for RAPS when 50 were computationally feasible for APS.

**Severity**: MINOR

### N7: s-group Median vs Mean Disconnect

Appendix B (line 460): s-group has mean test coverage 12.4% but median 0.5%. Most models catastrophically fail, but a few outlier seeds inflate the mean. The paper correctly flags this with asterisks and CV>50% disclosure, but the practical implication is stronger than acknowledged: for 40+ of 50 seeds, test coverage is near-zero, making the "mean drop = 71.2%" an understatement of typical behavior. This is disclosed adequately but worth noting.

**Severity**: MINOR -- properly disclosed

---

## Dimension-by-Dimension Assessment

### Dimension 1: Internal Validity

Strength: The 50-seed paired design provides strong within-task causal evidence for coverage degradation. The placebo test (6--143x ratio) supports that COVID is the driver, not routine drift. ICC analysis confirms task independence.

Weakness: Observational design -- cannot rule out that SHAP concentration proxies for some other task property. Partial correlation (controlling log class count) is non-significant at n=8. The paper acknowledges this as "associative evidence, not causal identification."

**Assessment**: Adequate for the claims being made (associative, not causal).

### Dimension 2: External Validity

Strength: 9 external datasets across multiple domains, consistent LightGBM+APS pipeline with no tuning.

Weakness: External catastrophic evidence is concentrated in a single dataset (Covertype). KDDCup99 is an intermediate case. The remaining 7 external datasets are all robust (low concentration, low drop), providing only one-directional evidence. The paper correctly states "external catastrophic evidence is concentrated in Covertype" but this is a fundamental limitation: the diagnostic's ability to predict catastrophic failure is validated by exactly 1 external instance.

**Assessment**: External validation demonstrates robustness prediction better than catastrophic prediction. The honest disclosure mitigates but does not resolve this.

### Dimension 3: Statistical Rigor

Strengths:
- 50-seed design with paired tests
- Bootstrap CIs on correlations
- Multiple comparison correction for metric selection (Holm)
- ICC and effective sample size computation
- Leave-one-out stability analysis
- Kendall tau alongside Spearman rho

Weaknesses:
- The n=16 primary endpoint mixes 50-seed SALT means with 10-seed external means (different precision)
- Retraining p=0.04 unadjusted is non-significant after Holm correction (p=0.12)
- Partial correlations are non-significant at n=8 (acknowledged)

**Assessment**: Above-average statistical rigor for an applied ML paper.

### Dimension 4: Measurement Quality

SHAP concentration is well-defined (Eq. 2), computed on validation data only, and shown to be stable within SALT tasks (CV < 5%, bootstrap CI within +/-1pp). However, external datasets show higher instability (Shuttle std=8.29%, KDDCup99 std=7.51%), which is partially addressed by the seed stability protocol but not fully characterized.

The coverage drop metric is straightforward but the dual threshold scheme (SEV/ROB vs At-risk) creates measurement confusion.

**Assessment**: Generally adequate; dual threshold scheme is the main gap.

### Dimension 5: Reproducibility

Strengths:
- Code available in `/papers/conformal_covid/code/`
- JSON result files for all experiments
- Fixed seed ranges documented (42--91 for SALT, 42--51 for external)
- Software versions specified (Python 3.9, LightGBM 3.3, SHAP 0.41)

Weaknesses:
- **"Default LightGBM settings" is false** -- actual hyperparameters are documented in Appendix A.1 but mislabeled
- Class counts vary per seed; paper does not specify which seed defines the Table 1 values
- `kddcup99_validation.json` single-seed result contradicts multi-seed categorization

**Assessment**: Reproducibility is high despite the mislabeling issue, because Appendix A.1 lists the actual parameter values even while calling them "default."

### Dimension 6: Logical and Interpretive Soundness

The core argument -- that SHAP concentration captures model-specific vulnerability to feature shift -- is well-supported within the SALT supply chain domain. The theory (Theorem 1) provides a plausible mechanism under stated assumptions, with the (A1) approximation clearly disclosed.

The paper is appropriately cautious about causal claims and threshold generalizability. Limitations are honestly stated in Section 8. The binary ceiling effect is a genuine structural insight.

The main interpretive gap is the conflation of "robust" labeling across Tables 1 and 6 for i-shippoint, which undermines the clarity of the classification framework.

**Assessment**: Sound, with appropriate caveats.

---

## Severity Classification Summary

### Fatal Issues -- NONE

### Major Issues

**B6 (persistent from R6): "Default LightGBM settings" is factually false.**
- Evidence: Code uses learning_rate=0.05, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5; LightGBM defaults are 0.1, 1.0, 1.0, 0 respectively.
- Consequence: Readers attempting reproduction with actual defaults will get different results. Misrepresents the experimental setup.
- Fix: Replace "default LightGBM settings" with "fixed LightGBM hyperparameters" or "common LightGBM settings" in both Section 3.2 (line 102) and Appendix A.1 (line 415). The actual values in the appendix are correct and sufficient.

### Moderate Issues

**M5 (persistent from R6): Dual classification scheme produces contradictory labels.**
- Evidence: i-shippoint is ROB* in Table 1 (drop < 20%), At-risk* in Table 6 (drop > 15pp).
- Consequence: Reader confusion about whether the framework succeeds or fails for this task.
- Fix: Either (a) unify to a single threshold scheme across all tables, or (b) add a clarifying footnote to Table 1 stating "Under the At-risk criterion (>15pp) used in the decision framework, i-shippoint is classified At-risk*."

**N1: Sub-nominal validation coverage in external datasets.**
- Evidence: Avila has 5/10 seeds below 90% target (mean 89.9%); Gas Sensor has 4/10 seeds below 90%.
- Consequence: "Robust" external classification is partially an artifact of under-calibration.
- Fix: Add a sentence in Section 5.5 or Appendix D noting that several external datasets exhibit sub-nominal validation coverage and that "robust" classification should be interpreted relative to their own baselines, not the theoretical guarantee.

**N3: Abstract reports unadjusted retraining p-value while corrected value is non-significant.**
- Evidence: Abstract: "p = 0.04 (unadjusted)"; Section 5.4: Holm-corrected p = 0.12.
- Consequence: Overstates retraining evidence in the abstract.
- Fix: Append "(Holm-corrected p = 0.12)" to the abstract retraining claim, or remove the p-value from the abstract entirely and say "suggestive but non-significant after multiple comparison correction."

### Minor Issues

**M1**: Table 1 class counts (462, 135, 70) disagree with 50-seed JSON (459, 137, 69). Add footnote noting per-seed variation.

**N2**: Covertype and Satimage are both remote sensing domains. Consider saying "8--9 domains" or noting the overlap.

**N4**: `kddcup99_validation.json` labels KDDCup99 "robust" (single-seed drop = -0.83%) while multi-seed data shows "severe" (mean drop = 15.85%). Clean up or annotate the single-seed JSON.

**N5**: Shuttle (concentration std = 8.29%) is more unstable than KDDCup99 but not mentioned in the seed stability protocol discussion.

**N6**: RAPS comparison uses 10 seeds vs 50 for APS without justification.

---

## Code Execution Results

No full pipeline execution was performed (SALT dataset requires relbench installation and data download). Verification was limited to:
1. **Theorem bounds**: Independently recomputed E[s_test] >= C + (1-C)/K for all 5 tasks. All match paper's claimed bounds to 3 decimal places. PASS.
2. **Hyperparameter audit**: Code consistently uses learning_rate=0.05, feature_fraction=0.8, bagging_fraction=0.8 across all scripts. Code is internally consistent but does not use LightGBM defaults. FAIL on "default" claim.
3. **RAPS i-shippoint worsening**: Recomputed from seed-level data: APS mean drop = 9.3pp, RAPS mean drop = 20.5pp, difference = 11.2pp. Matches paper claim. PASS.
4. **Class count variation**: Verified from RAPS seed-level data that class counts vary across seeds (s-group: 455--465, s-payterms: 133--138). Both Table 1 and JSON values are within-range. PASS with caveat.

---

## Reproducibility Score

**8/10**

Justification: Code is available, hyperparameters are documented (even if mislabeled), seed ranges are fixed, JSON result files exist for all experiments, and software versions are specified. The main deductions are: (1) "default settings" mislabeling could cause failed reproduction attempts (-1), and (2) ambiguous class count source (-0.5), partially offset by the otherwise excellent documentation (+0.5).

---

## Recommended Actions (Priority Order)

1. **Replace "default LightGBM settings/hyperparameters" with "fixed LightGBM hyperparameters"** in Section 3.2 (line 102) and Appendix A.1 (line 415). The actual values in the appendix are correct and need no change -- only the label "default" must be removed.

2. **Add a cross-reference footnote to Table 1** for i-shippoint: "Under the At-risk criterion (drop > 15pp) used in the decision framework (Table 6), i-shippoint is classified At-risk*."

3. **Add "(Holm-corrected p = 0.12)" to the abstract retraining claim** or rephrase as "suggestive evidence" without the p-value.

4. **Add a sentence about sub-nominal validation coverage** in external datasets to Section 5.5 or Table 6 footnotes.

5. **Add footnote to Table 1** stating that class counts are from a single canonical seed and vary across the 50-seed ensemble (range X--Y).

6. *(Optional)* Note Covertype/Satimage domain overlap or soften "9 domains" to "8--9 domains."

---

## Verdict

**MINOR REVISION REQUIRED**

The paper has one major issue (factually false "default settings" claim) that is a straightforward text fix. The moderate issues (dual threshold confusion, sub-nominal validation coverage, abstract p-value) are presentational rather than substantive. No fatal flaws remain. The core methodology, statistical analysis, and theoretical results are sound. After fixing the "default" label and adding the recommended clarifications, the paper meets the standard for acceptance.
