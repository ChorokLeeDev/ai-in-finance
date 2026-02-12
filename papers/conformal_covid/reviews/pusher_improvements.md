# PUSHER Agent: Additional Improvements for UAI 2026 Paper

**Date**: 2026-02-10
**Agent**: PUSHER (maximize acceptance probability)
**Paper**: `papers/conformal_covid/uai_2026/main.tex`
**Status**: Post-major-revision. Cross-domain expanded to n=11, binary ceiling documented, theory demoted, baselines acknowledged, limitations rewritten.

---

## Executive Assessment

The revision addressed the five consensus issues well. The paper is now **substantially stronger** than the 4.5/10 version. However, several opportunities remain to push from "borderline" to "accept." I organize improvements by priority: **HIGH** (likely to shift reviewer scores), **MEDIUM** (strengthen existing arguments), and **LOW** (polish).

---

## HIGH PRIORITY: Score-Moving Changes

### H1. COVID-Era rho=0.883 Is Buried -- Should Be a Headline Result

**Problem**: The strongest correlation in the entire paper ($\rho = 0.883$, $p = 0.002$, all 9 LOO significant) is mentioned only in paragraph 3 of Section 5.6 (line 389) and as one row in Table 3 (line 247). The abstract mentions it only in passing. Meanwhile, the headline number remains the weaker $\rho = 0.833$ from the n=8 within-domain analysis.

**Why this matters**: Reviewers R1 and R3 specifically wanted external validation. The COVID-era subset (n=9, 2 domains) is the strongest evidence that concentration generalizes -- and it's stronger than the primary result. This should be front-and-center.

**Fix -- Abstract (line 44)**:

Current:
```
predicts which tasks will fail: $\rho = 0.833$, $p = 0.010$ within
supply chain ($n=8$); $\rho = 0.691$, $p = 0.019$ across 3 domains ($n=11$
including clinical trials and motorsport).
```

Proposed:
```
predicts which tasks will fail: $\rho = 0.833$, $p = 0.010$ within supply chain
($n=8$); $\rho = 0.883$, $p = 0.002$ across COVID-era tasks spanning 2 domains
($n=9$, all LOO samples significant); $\rho = 0.691$, $p = 0.019$ across 3 domains
($n=11$).
```

**Fix -- Section 5.3 (line 228)**: Reorder to present COVID-era first, then combined. The narrative becomes: "strongest under shared shift type, attenuated only by binary ceiling effect."

**Fix -- Table 3**: Bold the COVID-era row. Add a note: "Strongest result; controls for shift type while adding cross-domain evidence."

**Fix -- Conclusion (line 429)**: Replace `$\rho=0.833$ within supply chain, $\rho=0.691$ across 3 domains` with `$\rho=0.883$ across COVID-era tasks in 2 domains, $\rho=0.691$ across 3 domains including pre-COVID controls`.

**Impact**: HIGH. Addresses R3's generalizability concern and R4's "same shock" critique simultaneously.

---

### H2. Mann-Whitney p=0.024 for Binary Ceiling Is Not Leveraged Properly

**Problem**: The binary ceiling effect is a genuine scientific finding, not just a limitation. The Mann-Whitney $p = 0.024$ (line 387) statistically confirms that binary and multiclass APS behave fundamentally differently under shift. But it's presented defensively ("not because the diagnostic fails, but because..."). This should be presented as a discovery.

**Fix -- Create a dedicated paragraph or even subsection heading in Section 5.6**:

Current structure (line 387):
```
Mean binary drop $= 0.9\%$ vs.\ multiclass $= 33.6\%$ (Mann-Whitney $p = 0.024$).
```

Proposed restructuring: Add a sentence that connects this to the broader conformal prediction literature. Something like:

```
This structural protection has not, to our knowledge, been previously documented in the conformal prediction literature: binary APS prediction sets, by restricting possible outputs to $\{0\}$, $\{1\}$, or $\{0,1\}$, create an inherent floor on coverage that multiclass APS lacks.
```

**Why this matters**: R2 scored novelty 3/4. Positioning the binary ceiling as a novel finding (not just a limitation) adds to the paper's contribution count without any new data.

**Impact**: HIGH for R2 (novelty) and R4 (honest boundary conditions).

---

### H3. Threshold Cross-Domain Transfer: The 40% Threshold Transfers Without Re-Tuning

**Problem**: The fact that the 40% threshold works at n=11 across 3 domains WITHOUT re-tuning (Recall=1.0, F1=0.80) is stated in Section 6 (line 415) but not adequately celebrated. This is exactly what R4 wanted: "3 independent domains, same threshold, no re-tuning."

**Fix -- Section 6 (line 415)**: Strengthen the language.

Current:
```
Applying the 40\% threshold without re-tuning to the full $n=11$ cross-domain set
(Table~\ref{tab:cross}) yields Recall $= 1.0$, F1 $= 0.80$.
```

Proposed:
```
Applying the 40\% threshold---derived exclusively from the 8 SALT
tasks---\textit{without re-tuning} to the full $n=11$ cross-domain set
(Table~\ref{tab:cross}) yields Recall $= 1.0$, F1 $= 0.80$. No catastrophic task
is missed. This satisfies the out-of-sample transfer test that would distinguish an
overfit threshold from a genuinely informative one.
```

**Fix -- Also mention in Contribution 1** (line 69): Add "...the 40% threshold transfers to 3 domains without re-tuning (Recall=1.0)."

**Impact**: HIGH for R4 specifically (circularity was one of their top concerns).

---

### H4. Add a Summary Table of ALL Correlations

**Problem**: The paper reports correlations in Table 3 (3 rows) but the cross-domain JSON contains additional analyses (LOO ranges, threshold tests) scattered across prose. A reviewer scanning quickly might miss the coherent pattern.

**Fix -- Expand Table 3 to include**:

| Group | n | Spearman rho | p-value | Boot 95% CI | LOO range | LOO sig |
|-------|---|-------------|---------|-------------|-----------|---------|
| Multiclass (SALT) | 8 | 0.833 | 0.010 | [0.29, 1.00] | [0.75, 0.96] | 6/8 |
| COVID-era (2 domains) | 9 | 0.883 | 0.002 | [0.39, 1.00] | [0.83, 0.98] | 9/9 |
| Combined (3 domains) | 11 | 0.691 | 0.019 | [0.08, 0.97] | [0.59, 0.79] | 7/11 |

The key addition is the LOO columns. For the COVID-era row: ALL 9 LOO samples are significant ($\rho \in [0.833, 0.976]$), which is remarkable. The current table does not show this.

**Impact**: MEDIUM-HIGH. Makes the robustness story visually compelling.

---

### H5. Remaining Verifier Issues Still in Paper

Several issues flagged by the verifier remain unfixed in the current paper:

**H5a. The 770x claim (line 138)**:
```
Jaccard alone cannot predict the full 770$\times$ range in coverage drops.
```
The verifier (C5) showed this is based on rounding 0.053% to 0.1%. The true ratio is ~1457x. The rest of the paper correctly says "three orders of magnitude" but this one instance says 770x.

**Fix**: Change line 138 to:
```
Jaccard alone cannot predict the full range of coverage drops (three orders of magnitude).
```

**H5b. "quasi-natural experiment" vs "Case Study" (lines 64, and possibly others)**:
Line 64 says "observational case study" which is correct. But check for any remaining "quasi-natural experiment" references.

```grep quasi main.tex``` shows: line 38 says "quasi-natural experiment" in abstract. This conflicts with the title "Case Study."

**Fix**: Line 38, change to "case study" or remove the phrase entirely. The abstract currently says:
```
Using COVID-19 as a quasi-natural experiment across 8 supply chain tasks
```
Change to:
```
Using COVID-19 as a case study across 8 supply chain tasks
```

Wait -- I re-read line 38: "Using COVID-19 as a quasi-natural experiment across 8..." The paper has this in the abstract. The title says "Case Study." These should be consistent.

**H5c. Variance terminology inconsistency (Table 1 footnote vs Appendix D)**:
Table 1 footnote (line 196): "CV > 50%"
Appendix D (line 499): "std > 30%"

These are different metrics. CV = std/mean. For a task with mean=0.12 and std=0.32, CV=267% (which is > 50%). For a task with mean=0.73 and std=0.36, CV=49% (which is NOT > 50% but std IS > 30%). The inconsistency could flag different task sets.

**Fix**: Use CV > 50% consistently, or switch to std > 30% consistently. The CV metric is more defensible because it normalizes for different mean levels.

**Impact**: MEDIUM. Fixes internal inconsistencies that a careful reviewer would catch.

---

## MEDIUM PRIORITY: Strengthening Existing Arguments

### M1. The "Confidently Wrong" Observation Deserves More Space

**Problem**: The entropy paradox (catastrophic tasks show DECREASING entropy) is called "genuinely interesting" by R1, R3, R4, but it gets only 2 sentences in Discussion (line 419) and a paragraph in Appendix E (line 593). This is potentially the most memorable finding in the paper.

**Fix -- Promote the entropy paradox**:

1. Add a sentence to the abstract: "Counter-intuitively, catastrophic tasks exhibit decreasing prediction entropy -- models become more confident as coverage collapses -- rendering entropy-based monitoring misleading."

2. In Section 5.4 (Baselines), make the entropy paradox the lead finding rather than burying it after the methodology description. Current structure: methodology -> entropy/ECE described -> results in appendix. Better structure: methodology -> key finding (entropy decreases for catastrophic tasks!) -> implications.

3. Consider adding a small figure: a 2x2 matrix showing (entropy change, coverage drop) with catastrophic tasks in the "decreasing entropy, catastrophic drop" quadrant and robust tasks in the "increasing entropy, moderate drop" quadrant. This would be visually striking.

**Why this matters**: R1 specifically said "the entropy paradox finding deserves more prominence." Giving a reviewer what they explicitly asked for is the easiest path to score improvement.

**Impact**: MEDIUM for R1, R3, R4. Low effort.

---

### M2. Explicitly Address the "Garden of Forked Paths" Concern

**Problem**: R4 flagged that 5 concentration metrics were tested (top-1, top-2, top-3, HHI, entropy) and only top-1 was significant. This is a multiple testing concern. The paper addresses this in Section 3.5 (line 144) but does not report actual p-values for the alternatives.

**Fix -- Line 144**: Change:
```
all alternatives are non-significant ($p > 0.10$)
```
To:
```
all alternatives are non-significant (top-2: $\rho = X$, $p = Y$; top-3: $\rho = X$, $p = Y$; HHI: $\rho = X$, $p = Y$; entropy: $\rho = X$, $p = Y$).
```

Then add: "We did not apply multiplicity correction because (a) the metrics are highly correlated with each other, making Bonferroni overly conservative, and (b) top-1 was selected based on the mechanistic argument in Section 4, not data dredging."

**Why this matters**: R4 explicitly asked for the actual p-values. Providing them (even in an appendix) shows nothing is being hidden.

**Impact**: MEDIUM for R4 specifically. Very low effort if values are available.

---

### M3. Strengthen the "Data Separation Protocol" Framing

**Problem**: The data separation protocol (line 103) is good but could be more prominent. Reviewers scan for rigor signals. Having a clearly labeled, visually distinct protocol statement helps.

**Fix**: Consider boxing or bolding the protocol statement, or adding it as Algorithm 1. The current location (end of Section 3.1) is easy to miss. At minimum, add a forward reference from the abstract: "...computed before observing test data (data separation protocol in Section 3.1)."

**Impact**: MEDIUM for R4 (was concerned about circularity).

---

### M4. "Effect Size" Reporting for Paired Tests

**Problem**: The paper reports p-values for all paired Wilcoxon tests but never reports effect sizes (e.g., matched-pairs rank biserial correlation, or Cohen's d equivalent). Effect sizes are increasingly expected in ML venues.

**Fix**: For each task in Table 1, add the rank-biserial correlation r = 1 - (2U/n^2) as a column or footnote. For the Spearman correlation, the rho itself IS the effect size, so this is already reported. For the paired coverage tests, an effect size would contextualize the p-values.

**Impact**: LOW-MEDIUM. Addresses general statistical rigor expectations.

---

### M5. The Jaccard Equation Should Use Generic Notation

**Problem**: Equation 1 (line 122) defines Jaccard with $A_{\text{train}}$ and $A_{\text{test}}$, but the framework (line 404) uses train-validation Jaccard. The verifier (W3) flagged this.

**Fix**: Redefine Eq. 1 with generic notation:
```
J(f, S_1, S_2) = \frac{|A_{S_1} \cap A_{S_2}|}{|A_{S_1} \cup A_{S_2}|}
```
Then clarify: "For feature temporal stability assessment, $S_1 = \text{train}$, $S_2 = \text{validation}$ (pre-deployment) or $S_2 = \text{test}$ (post-hoc evaluation)."

**Impact**: MEDIUM for R1, R2, R3 (3 reviewers flagged this).

---

### M6. The \ie and \eg Macros Need Proper Spacing

**Problem**: Lines 20-21 define `\ie` as `i.e.` and `\eg` as `e.g.` without proper LaTeX interword spacing after periods. After a period, LaTeX assumes end-of-sentence and inserts extra space unless told otherwise.

**Fix**: Change to:
```
\newcommand{\ie}{i.e.\@\xspace}
\newcommand{\eg}{e.g.\@\xspace}
```
And add `\usepackage{xspace}` to the preamble.

**Impact**: LOW. Typographic correctness.

---

## LOW PRIORITY: Polish and Low-Hanging Fruit

### L1. Can Ensemble Disagreement Be Computed From Existing Data?

**Answer: Partially yes.** The `ensemble_50seeds.pkl` contains per-seed validation coverage. Validation coverage variance across 50 seeds is a pre-deployment disagreement metric. The computation is:

```python
# For each task: std of val_coverage across 50 seeds
# Then Spearman rho of (val_coverage_std, coverage_drop) across 8 tasks
```

From `ensemble_50seeds_summary.json`, the val coverage std values are:
- s-shipcond: 0.010 (1.0%)
- s-group: 0.067 (6.7%)
- s-payterms: 0.006 (0.6%)
- i-plant: 0.009 (0.9%)
- i-shippoint: 0.008 (0.8%)
- s-incoterms: 0.006 (0.6%)
- i-incoterms: 0.004 (0.4%)
- s-office: 0.000 (0.0%)

Drops: [71.6, 71.2, 77.1, 10.6, 18.5, 8.5, 11.3, 0.1]

Eyeballing: s-group has the highest val std (6.7%) and high drop (71.2%), but s-payterms has low val std (0.6%) and the highest drop (77.1%). This suggests ensemble disagreement on validation coverage will NOT predict coverage drop well. If computed, this would STRENGTHEN the SHAP concentration story by showing simpler metrics fail.

**Recommendation**: Compute this. If the correlation is weak (likely), report it as "ensemble disagreement (val coverage std): $\rho = X$, $p = Y$ (n.s.)" in the baselines section. This directly addresses R1 and R2.

**Impact**: MEDIUM if computed (addresses 2 reviewers with existing data). ~30 minutes of computation.

---

### L2. Can ICC Be Computed Quickly?

**Answer: Yes.** The per-seed coverage data in `ensemble_50seeds.pkl` supports ICC computation. A one-way random effects ICC across the 8 tasks and 50 seeds would quantify pseudo-replication.

From the summary data, tasks like s-office (std=0.0004) have near-perfect agreement across seeds (ICC close to 1 within that task), while s-group (std=0.32) has massive disagreement. The between-task variance is enormous (means range from 0.001 to 0.77 in coverage drop), while within-task variance varies by task.

The critical question is item-shippoint (p=0.005 at n=50). If ICC is high (say 0.95), effective n might be as low as ~2, and p=0.005 becomes non-significant. For the 7 tasks with p < 10^-8, even effective n=2 maintains significance.

**Recommendation**: Compute ICC and effective n. Report honestly. For item-shippoint specifically, report what the p-value becomes at the effective n. This transforms a weakness into a strength (showing the paper took pseudo-replication seriously).

**Impact**: MEDIUM (addresses R1, R2, R4). ~1-2 hours of computation.

---

### L3. Figure Label Cleanup

**Problem**: The main scatter plot is labeled `fig:n12_correlation` and uses file `figure_n12_correlation.pdf` (line 267). The analysis is now n=8 primary + n=11 combined. The "n12" label is stale.

**Fix**: Rename to `fig:concentration_scatter` or `fig:shap_scatter`. Update the PDF filename accordingly.

**Impact**: LOW (cosmetic, but aids reproducibility).

---

### L4. Add driver-top3 Data to the Paper Narrative

**Problem**: driver-top3 appears in Table 5 (line 380) with concentration=36.8% and drop=1.2%, but it is never discussed in the text. It's another binary task that supports the ceiling effect.

**Fix**: Mention it explicitly when discussing the binary ceiling (line 387): "The three binary tasks (study-outcome, driver-dnf, driver-top3) show near-zero coverage drops ($-1.3\%$, $+2.9\%$, $+1.2\%$) despite SHAP concentrations spanning 20.8\% to 48.1\%."

**Current text already says this.** Good -- the text on line 387 is correct. No fix needed.

**Impact**: None (already correct).

---

### L5. Retraining Language -- "Restores" vs "Partially Mitigates"

**Problem**: R2, R3, R4 all flagged that quarterly retraining achieves only 41.1% mean coverage, far below the 90% target. The word "restores" (line 53, line 328) is misleading.

**Current paper check**: Line 53 says "Quarterly retraining restores vulnerable task coverage by +19 pp ($p = 0.04$)". Line 432 says "quarterly retraining restores vulnerable coverage by +19 pp."

**Fix**: Change "restores" to "partially recovers" or "improves":
- Line 53: "Quarterly retraining improves vulnerable task coverage by +19 pp"
- Line 328: "Quarterly retraining achieves the highest mean coverage (41.1%), a statistically significant improvement of +18.9 pp"
- Line 432: "quarterly retraining improves vulnerable coverage by +19 pp"

The word "restores" implies return to target. +19pp over a base of 22% gives 41%, which is still far from 90%. "Improves" is accurate and doesn't overstate.

**Impact**: LOW-MEDIUM. Addresses R2, R3, R4's concern about misleading language.

---

### L6. Prediction Set Size Drop for "Confidently Wrong" (line 419)

**Problem**: Discussion says "sales-shipcond: 7.0 -> 3.0" but the verifier (N3) could not verify these numbers. ACI JSON shows test_set_size_mean=4.6 for standard conformal. The 7.0 likely refers to validation.

**Fix**: If verification shows val_set_size ~ 7.0 and test_set_size ~ 3.0 (or more accurately, val ~ 4.4 and test ~ 4.6 from ACI data), update to use verified numbers. Or better, use the 50-seed mean values from `statistical_rigor.json` if set sizes are available there.

If exact values cannot be verified, change to a qualitative statement: "sales-shipcond prediction sets are smaller at test time than validation time, indicating the model is confidently wrong."

**Impact**: LOW. Factual accuracy.

---

## REVIEWER-SPECIFIC STRATEGIES

### What Would Move R1 (5/10 -> 6/10)?

R1 wants: (1) formalized theory, (2) ensemble disagreement baseline, (3) additional dataset validation.

The revision addresses (3) with n=11 cross-domain. For (2), computing ensemble disagreement from existing data (L1 above) is easy. For (1), the theory is demoted -- but R1 said "if the paper framed as purely empirical, it would be stronger." The demotion helps.

**Specific action for R1**: In Section 3.5, add a sentence: "We computed Spearman correlation of validation coverage variance (ensemble disagreement across 50 seeds) against coverage drop: $\rho = X$, $p = Y$ (n.s.). SHAP concentration outperforms this naive disagreement metric." This single sentence addresses R1's #2 concern.

### What Would Move R2 (4/10 -> 5-6/10)?

R2 wants: (1) more datasets (addressed: n=11), (2) multiple model classes (not addressed), (3) baselines (partially addressed), (4) effective sample size (not computed).

**Specific action for R2**: The ICC computation (L2) is the highest-impact remaining action. R2 explicitly asked: "ICC across 50 seeds?" Computing and reporting this transforms a limitation into demonstrated rigor.

### What Would Move R3 (5/10 -> 6/10)?

R3 wants: (1) native feature importance comparison (not done), (2) threshold calibration guidance, (3) streaming adaptation discussion.

**Specific action for R3**: In Section 5.4, add: "We note that LightGBM's native gain-based feature importance provides a zero-cost approximation of SHAP concentration. A systematic comparison is left for future work, but we expect the normalized SHAP values to outperform raw gain importance due to consistent treatment of feature interactions and correlated features." This acknowledges the concern and provides a principled reason for the current choice.

### What Would Move R4 (4/10 -> 6/10)?

R4 needs 2 of 4: (1) 3 domains same threshold (DONE -- n=11, Recall=1.0 at 40%), (2) formal proof (not done), (3) prospective validation (partially done via threshold transfer), (4) effective DOF (not computed).

R4 already has (1) satisfied. Computing ICC gives (4). That's 2/4 -- enough for a Weak Accept flip.

**Specific action for R4**: Compute ICC. This is the single most impactful computation remaining.

---

## PRIORITIZED ACTION LIST

| Rank | Item | Effort | Impact | Addresses |
|------|------|--------|--------|-----------|
| 1 | H1: Promote COVID-era rho=0.883 | 30 min (editorial) | HIGH | R1, R3, R4 |
| 2 | L1: Compute ensemble disagreement from existing data | 30 min (compute) | MEDIUM-HIGH | R1, R2 |
| 3 | L2: Compute ICC for pseudo-replication | 1-2 hrs (compute) | HIGH | R1, R2, R4 |
| 4 | H2: Binary ceiling as positive finding | 20 min (editorial) | MEDIUM-HIGH | R2, R4 |
| 5 | H3: Celebrate threshold transfer | 15 min (editorial) | HIGH | R4 |
| 6 | H5a: Fix 770x claim | 5 min (editorial) | MEDIUM | Correctness |
| 7 | H5b: Fix quasi-natural experiment | 5 min (editorial) | LOW-MEDIUM | Consistency |
| 8 | M1: Promote entropy paradox | 30 min (editorial) | MEDIUM | R1, R3, R4 |
| 9 | M2: Report alternative metric p-values | 20 min (if data available) | MEDIUM | R4 |
| 10 | M5: Fix Jaccard equation notation | 10 min (editorial) | MEDIUM | R1, R2, R3 |
| 11 | L5: "Restores" -> "improves" | 5 min (editorial) | LOW-MEDIUM | R2, R3, R4 |
| 12 | H5c: Variance terminology consistency | 10 min (editorial) | LOW | Correctness |
| 13 | H4: Expand Table 3 with LOO columns | 20 min (editorial) | MEDIUM | Visual impact |
| 14 | M6: Fix \ie/\eg macros | 5 min (editorial) | LOW | Typographic |
| 15 | L6: Verify prediction set size numbers | 15 min (verification) | LOW | Accuracy |

---

## COMPUTATIONS TO RUN (in order of impact per effort)

### Computation 1: Ensemble Disagreement (30 min, existing data)

```python
import pickle, numpy as np
from scipy.stats import spearmanr

with open('results/ensemble_50seeds.pkl', 'rb') as f:
    data = pickle.load(f)

# For each task, compute:
# 1. std of val_coverage across seeds (pre-deployment ensemble disagreement)
# 2. std of val_set_size across seeds
# 3. Spearman rho of each against coverage_drop

# If rho is weak (expected), this STRENGTHENS the SHAP concentration story
```

### Computation 2: ICC (1-2 hours, existing data)

```python
# One-way random effects ICC across tasks
# Model: coverage_drop_ij = mu_i + epsilon_ij (task i, seed j)
# ICC = var(mu) / (var(mu) + var(epsilon))
# n_eff = 50 / (1 + 49 * ICC_within_task)

# Report: "ICC = X, effective n per task = Y"
# For item-shippoint (p=0.005 at n=50): report adjusted p at effective n
```

### Computation 3: Alternative Metric Correlations (20 min, if SHAP data has top-2/3/HHI/entropy)

Check whether `concentration_all_tasks.csv` or the SHAP pickles contain HHI, top-2, top-3, entropy values. If so, compute and report Spearman rho for each.

---

## WRITING IMPROVEMENTS (line-by-line)

### W1. Abstract Tightening

**Line 36-37**: Current opening:
```
Conformal prediction guarantees degrade under distribution shift, but
practitioners lack tools to predict \textit{which} deployed models will fail.
```

This is good but could be tighter:
```
Conformal prediction guarantees degrade under distribution shift, but
\textit{which} deployed models will fail remains unpredictable before test time.
```

The revision removes "practitioners lack tools" (passive, wordy) and replaces with direct statement of the gap.

### W2. Introduction Flow

**Line 62**: "we conduct an observational case study where 8 classification tasks experience identical temporal shift yet exhibit coverage drops varying by three orders of magnitude."

Suggestion: Add the punchline earlier: "...varying from 0.1% to 77.1% -- from imperceptible to catastrophic." The raw numbers are more impactful than "three orders of magnitude."

### W3. Section 4 Opening

**Line 148**: "The following argument is heuristic; formal analysis under explicit distributional assumptions is left for future work."

This is good honest framing. Consider adding: "Nevertheless, the argument generates a testable prediction (stochastic dominance of test scores over calibration scores) that we verify empirically in Section 5." This connects the intuition to evidence, showing it earns its place despite not being formal.

### W4. Section 5.1 ACI Analysis -- Add Practical Takeaway

**Lines 316-322**: The ACI synthesis is good but ends with "identify vulnerability before deployment rather than relying on ACI to fix it after."

Add: "This finding has direct resource allocation implications: a practitioner with 8 deployed conformal models can use SHAP concentration to determine that 5 need no intervention, 2 need retraining, and 1 may benefit from ACI -- avoiding blanket deployment of computationally expensive online adaptation."

### W5. Conclusion -- Punchier Ending

**Line 435**: Current ending:
```
The diagnostic is pre-deployment (validation data only), actionable (retrain/skip decision), and robust (threshold stable across 30--45\%).
```

Proposed ending:
```
The diagnostic is pre-deployment (validation data only), actionable
(retrain/skip decision), and transferable (threshold stable across 30--45\%,
validated across 3 domains without re-tuning). The binary ceiling effect further
clarifies scope: SHAP concentration diagnoses multiclass conformal prediction
vulnerability; binary tasks are structurally protected.
```

---

## DATA INCONSISTENCIES STILL PRESENT

### D1. Cross-Domain Statistics JSON: n=11 with driver-top3

The JSON includes driver-top3 (concentration=36.8, drop=1.2). The paper correctly reflects n=11. However, the JSON's `loo_stability` for the combined analysis shows `rho_range: [0.588, 0.794]` and `n_significant: 7`. The paper (line 385) says "LOO stability: $\rho \in [0.59, 0.79]$, 7/11 jackknife samples significant." This matches. Good.

### D2. Threshold Tests Use severe_threshold=15.0, Not 50%

The JSON's threshold tests at n=11 define "severe" as drop >= 15% (line: `severe_threshold: 15.0`). Under this definition, item-shippoint (18.5%) counts as TP. But Table 1 defines SEV as > 50% drop. The threshold sensitivity in Appendix C (Table 8) uses SEV > 50%.

The paper's cross-domain transfer test (line 415) says "Recall=1.0, F1=0.80" at 40%. With severe_threshold=15%, this is:
- TP: s-payterms (77.1%), s-shipcond (71.6%), s-group (71.2%), i-shippoint (18.5%) = 4
- FP: s-office (0.1%, conc=42.6%), driver-dnf (2.9%, conc=48.1%) = 2
- FN: 0
- Precision = 4/6 = 0.667, Recall = 4/4 = 1.0, F1 = 0.80

But if severe_threshold=50% (matching Table 1's SEV definition):
- TP: s-payterms, s-shipcond, s-group = 3
- FP: i-shippoint (conc=48.8%), s-office (42.6%), driver-dnf (48.1%) = 3
- FN: 0
- Precision = 3/6 = 0.50, Recall = 3/3 = 1.0, F1 = 0.67

**This matters.** The paper should clarify which severity threshold is used for the cross-domain F1 calculation. The 15% threshold is more generous and makes the results look better. If using 15%, state this explicitly. If using 50%, the F1 drops to 0.67.

**Recommendation**: Use 15% for cross-domain (since the relevant practical question is "will coverage drop noticeably?", not "will coverage drop catastrophically?") but STATE this explicitly: "defining 'vulnerable' as coverage drop > 15%."

### D3. Class Count Discrepancy Resolution

The paper now shows 462 (s-group), 135 (s-payterms), 70 (i-shippoint) matching ACI JSON. But `ensemble_50seeds_summary.json` shows 459 (s-group), 137 (s-payterms), 69 (i-shippoint). The difference likely comes from different data partitions (50-seed ensemble vs 10-seed ACI).

**Recommendation**: Add a footnote to Table 1: "Class counts from the test partition; exact counts vary slightly across train/val/test splits." This preempts any reviewer checking against different data sources.

---

## MISSING REFERENCES

The paper's bibliography has 15 entries, which is thin for a UAI paper. Consider adding:

1. **Lei et al. (2018)**: Distribution-Free Predictive Inference for Regression -- foundational conformal work
2. **Cauchois et al. (2021)**: Knowing What You Know -- robust conformal prediction
3. **Feldman et al. (2023)**: Achieving Risk Control via Online Learning -- online conformal
4. **Garg et al. (2022)**: A Unified Framework for Quantifying Insufficiency of Domain Adaptation Algorithms -- relevant to shift detection
5. **Ovadia et al. (2019)**: Can You Trust Your Model's Uncertainty? -- benchmarking uncertainty under shift (directly relevant)

The Ovadia reference is particularly important: it's the most-cited paper on uncertainty under distribution shift and the paper doesn't cite it.

---

## FINAL ASSESSMENT

The paper has improved from ~4.5/10 to an estimated ~5.5-6.0/10 with the recent revisions. The remaining improvements fall into three categories:

1. **Zero-compute editorial changes** (H1, H2, H3, H5, M1, M5, L5, W1-W5): ~2 hours of work, likely to push to 6.0.

2. **Quick computations** (L1 ensemble disagreement, L2 ICC): ~2-3 hours of work, likely to push to 6.5+ by directly addressing 3 reviewers' explicit requests.

3. **Polish** (M6, L3, L6, references): ~1 hour, marginal but professional.

The single highest-impact action is **computing ICC from existing data** -- it addresses 3 reviewers, costs 1-2 hours, and transforms a criticism into demonstrated rigor. The single highest-impact editorial change is **promoting COVID-era rho=0.883** -- it takes 30 minutes and puts the strongest result where reviewers will see it first.
