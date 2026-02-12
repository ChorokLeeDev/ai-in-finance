# Verifier Audit Report: UAI 2026 Paper

**Paper**: Diagnosing Conformal Prediction Failures Under Distribution Shift: A COVID-19 Case Study
**Date**: 2026-02-10
**Scope**: Every number, claim, and cross-reference checked against data files

---

## Summary

| Severity | Count |
|----------|-------|
| CRITICAL | 6 |
| WARNING  | 12 |
| NOTE     | 8 |

---

## CRITICAL Findings

### [SEVERITY: CRITICAL] C1. Cross-Domain Table 5: Classification Rows Use Stale/Incorrect Data
- **Location**: `main.tex` lines 357-361 (Table 5, "Clinical Trials -- Classification" rows)
- **Issue**: Three classification rows report data that does not match any available data file, and two of the three tasks (study-adverse, site-success) are **regression tasks**, not classification tasks.
  - `study-outcome`: Paper says Val=100.0%, Test=100.0%, Drop=0.0% (single seed). APS pickle (50 seeds) says Val=88.3%, Test=89.6%, Drop=-1.3%.
  - `study-adverse`: Paper says Val=88.6%, Test=25.5%, Drop=63.1% (classification). But this is a **regression** task; CQR JSON (5 seeds) says Val=91.9%, Test=88.5%, Drop=3.5%.
  - `site-success`: Paper says Val=94.8%, Test=42.8%, Drop=52.0% (classification). But this is a **regression** task; CQR JSON (5 seeds) says Val=99.5%, Test=99.5%, Drop=0.0%.
- **Fix**: Remove the classification rows for study-adverse and site-success (they are regression tasks). Update study-outcome to use 50-seed APS data: Val=88.3%, Test=89.6%, Drop=-1.3%.
- **Evidence**: `results/conformal/aps_rel-trial_study-outcome.pkl` (val=0.883, test=0.896), `results/cqr_rel-trial_study-adverse.json` (Val=91.9%, Test=88.5%), `results/cqr_rel-trial_site-success.json` (Val=99.5%, Test=99.5%).

---

### [SEVERITY: CRITICAL] C2. Table 3 Reports n=12 Combined Correlation That No Longer Matches Any Data File
- **Location**: `main.tex` line 250 (Table 3 footnote)
- **Issue**: Paper footnote says "Combined n=12: rho=0.676, p=0.016 (omitted: conflates two mechanisms)." The actual cross-domain analysis (`cross_domain_statistics.json`) has n=10 combined: rho=0.745, p=0.013. The n=12 figure is not reproducible from any current data file. If the n=12 included study-adverse and site-success as classification tasks, that data is incorrect (see C1).
- **Fix**: Update to "Combined n=10: rho=0.745, p=0.013" or remove entirely since the paper already correctly focuses on n=8 SALT-only.
- **Evidence**: `cross_domain_statistics.json` combined section: n=10, rho=0.7455, p=0.0133.

---

### [SEVERITY: CRITICAL] C3. "Moderate Shift" rho=0.632 (n=4) Cannot Be Reproduced
- **Location**: `main.tex` line 232 (Section 5.3) and line 243 (Table 3)
- **Issue**: The paper claims "4 additional tasks from clinical trials and motorsports with moderate feature stability (Jaccard 0.13-0.86) show no concentration effect (rho=0.632, p=0.368, n.s.)". This value cannot be reproduced from any combination of the available cross-domain data:
  - Using APS drops for study-outcome (-1.3%) and driver-dnf (2.9%) plus CQR drops for study-adverse (3.5%) and site-success (0.0%): rho=-0.200, p=0.800.
  - Using the stale classification drops from Table 5 (0.0%, 63.1%, 52.0%) plus driver-dnf (2.9%): rho=-0.400, p=0.600.
  - No combination of 4 tasks with available SHAP concentrations yields rho=0.632.
- **Fix**: Recompute using consistent methodology (either all APS or all CQR), or remove this claim if the data source is unknown. The Jaccard range is also wrong (actual: 0.10-0.95, not 0.13-0.86).
- **Evidence**: Concentrations: study-outcome=20.8%, study-adverse=17.0%, site-success=34.4%, driver-dnf=48.1%. No combination yields rho=0.632.

---

### [SEVERITY: CRITICAL] C4. Class Count Mismatches Between Paper Table 1 and ACI Data
- **Location**: `main.tex` lines 179, 180, 182 (Table 1, "Cl" column)
- **Issue**: Three tasks have inconsistent class counts:
  - `s-group`: Paper says 459, ACI JSON says 462.
  - `s-payterms`: Paper says 137, ACI JSON says 135.
  - `i-shippoint`: Paper says 69, ACI JSON says 70.
- **Fix**: Determine the authoritative source (likely the ACI JSON which is computed from actual data) and make Table 1 consistent with it. If both are from different data partitions, explain this.
- **Evidence**: `aci_all_tasks_summary.json`: sales-group num_classes=462, sales-payterms num_classes=135, item-shippoint num_classes=70.

---

### [SEVERITY: CRITICAL] C5. 770x Range Claim Based on Rounding Artifact
- **Location**: `main.tex` lines 62, 134, 256, 406 ("nearly three orders of magnitude (770x)")
- **Issue**: The exact coverage drops are 0.053% (sales-office) to 77.1% (sales-payterms), giving a ratio of ~1457x, not 770x. The 770x figure comes from rounding 0.053% up to 0.1% in Table 1, then dividing 77.1/0.1=771. Table 1 reports the sales-office drop as "0.1 [0.0, 0.1]" but the actual 95% CI is [0.039%, 0.067%], so the true mean is 0.05%, not 0.1%.
- **Fix**: Either (a) report "0.05%" in Table 1 and update to ~1500x ("more than three orders"), or (b) keep 0.1% rounding but change to "nearly three orders (771x based on rounded values)" and acknowledge the rounding, or (c) simply say "three orders of magnitude" without a specific multiplier.
- **Evidence**: `statistical_rigor.json` sales-office coverage_drop mean=0.000529 (=0.053%), CI=[0.000388, 0.000670] (=[0.039%, 0.067%]).

---

### [SEVERITY: CRITICAL] C6. Binary Ceiling Effect Not Discussed in Paper
- **Location**: Entire paper -- missing content
- **Issue**: The cross-domain analysis reveals a binary APS ceiling effect: binary classification tasks cluster near zero coverage drop regardless of SHAP concentration (mean=0.8%, range=[-1.3%, 2.9%]), while multiclass tasks show meaningful variation (mean=33.6%, range=[0.05%, 77.1%]). Mann-Whitney p=0.044. This means:
  1. The n=10 combined correlation (rho=0.745) is inflated by mixing two structurally different populations.
  2. The SHAP concentration diagnostic may not apply to binary tasks at all.
  3. driver-dnf has high concentration (48.1%) but near-zero drop (2.9%) -- this is not a "protective factor" effect but a binary ceiling effect.
- **Fix**: Add a paragraph in Discussion or Limitations acknowledging the binary ceiling effect. Qualify the cross-domain validation accordingly.
- **Evidence**: `cross_domain_statistics.json` binary_ceiling section: binary_drop_mean=0.8, multiclass_drop_mean=33.6, mannwhitney_p=0.0444.

---

## WARNING Findings

### [SEVERITY: WARNING] W1. Category Label Inconsistency Across Code and Paper
- **Location**: `concentration_all_tasks.csv`, `compute_cross_domain_statistics.py`, `main.tex` Table 1
- **Issue**: The paper uses a two-category system (SEV >50% drop, ROB <20% drop), the CSV uses three categories (Catastrophic, Severe, Robust), and the cross-domain threshold test uses a different binary split (severe >= 15% drop). Specific conflicts:
  - `item-plant` (drop=10.6%): CSV="Severe", Paper="ROB", cross-domain="not severe"
  - `item-shippoint` (drop=18.5%): CSV="Severe", Paper="ROB*", cross-domain="severe"
  - The threshold test in `cross_domain_statistics.json` uses 15% as the severe cutoff, but the paper's Table 1 defines SEV as >50%.
- **Fix**: Standardize categories. The paper's two-tier system (SEV/ROB) is cleaner for the n=8 SALT analysis, but the cross-domain threshold tests use 15% which reclassifies item-shippoint as "severe."
- **Evidence**: `compute_cross_domain_statistics.py` line 233: `severe_threshold=15.0`. Paper Table 1 footnote: "SEV = Severe (>50% drop), ROB = Robust (<20% drop)".

---

### [SEVERITY: WARNING] W2. Abstract ACI Range "5--82%" is Misleading
- **Location**: `main.tex` lines 53, 410
- **Issue**: The abstract and conclusion state ACI "inflates prediction sets to 5--82% of classes." The 5% lower bound is from sales-office, which is a robust task that does NOT need ACI (its standard coverage is already 99.9%). For the three severe tasks that actually need ACI, the range is 30--82%. Including 5% from a task that doesn't benefit from (and is slightly harmed by) ACI is misleading.
- **Fix**: Change to "30--82% of classes for vulnerable tasks" or "5--82% across all tasks (30--82% for those needing intervention)."
- **Evidence**: ACI JSON: sales-office size_per_classes=0.048 (5%), sales-shipcond=0.30 (30%), sales-payterms=0.40 (40%), sales-group=0.82 (82%).

---

### [SEVERITY: WARNING] W3. Equation 1 (Jaccard) Definition Uses Train/Test but Framework Uses Train/Validation
- **Location**: `main.tex` line 119 (Eq. 1) vs line 384 (Section 6 framework)
- **Issue**: Equation 1 defines Jaccard as overlap between A_train and A_test. But the Decision Framework (Section 6) says to check "secondary features with train-validation Jaccard > 0.5." The paper itself notes this is intentional ("validation overlap serves as a pre-deployment proxy") but the equation should match the actual usage for the pre-deployment diagnostic.
- **Fix**: Either (a) add a second equation for the pre-deployment train-validation Jaccard, or (b) redefine Eq. 1 generically as J(f) = |A_S1 intersect A_S2| / |A_S1 union A_S2| and explain that S1=train, S2=validation for pre-deployment and S2=test for post-hoc evaluation.
- **Evidence**: Line 119: "A_train and A_test". Line 384: "train-validation Jaccard > 0.5".

---

### [SEVERITY: WARNING] W4. Exploratory Validation Jaccard Range Claim is Wrong
- **Location**: `main.tex` line 232
- **Issue**: Paper says "moderate feature stability (Jaccard 0.13--0.86)". Actual Jaccard values: study-outcome=0.53, driver-dnf=0.10, study-adverse=0.87, site-success=0.95. The actual range is 0.10--0.95, not 0.13--0.86.
- **Fix**: Update to "Jaccard 0.10--0.95" or verify which 4 tasks were used and their actual Jaccard values.
- **Evidence**: APS pickle driver-dnf jaccard_mean=0.10, CQR JSON study-adverse jaccard=0.872, site-success jaccard=0.954.

---

### [SEVERITY: WARNING] W5. Cross-Domain Statistics Not Yet Incorporated Into Paper
- **Location**: Entire paper
- **Issue**: The `cross_domain_statistics.json` contains new results that should be incorporated:
  - COVID-era n=9 analysis: rho=0.883, p=0.002 (strongest result)
  - Combined n=10 analysis: rho=0.745, p=0.013
  - Binary ceiling effect: Mann-Whitney p=0.044
  - These new results are not mentioned in the paper body.
- **Fix**: Add a paragraph in Section 5.6 (Cross-Domain Validation) discussing the combined correlation and the binary ceiling effect. Consider reporting the COVID-era rho=0.883 as a secondary result.
- **Evidence**: `cross_domain_statistics.json` covid_era: rho=0.8833, p=0.0016; binary_ceiling: mannwhitney_p=0.0444.

---

### [SEVERITY: WARNING] W6. Table 5 Mixes Single-Seed and Multi-Seed Results Without Clear Indication
- **Location**: `main.tex` lines 356-368 (Table 5)
- **Issue**: The classification section header says "single seed" but the regression section says "50 seeds". However:
  - study-outcome APS was actually run with 50 seeds (not single seed).
  - The CQR results were run with 5 seeds (not 50 seeds as the header implies).
- **Fix**: Correct the seed counts: study-outcome should say 50 seeds, CQR regression should say 5 seeds. The table header for regression says "50 seeds" but the actual CQR JSON files show num_seeds=5.
- **Evidence**: `aps_rel-trial_study-outcome.pkl`: num_seeds=50. `cqr_rel-trial_study-adverse.json`: num_seeds=5. `cqr_rel-f1_driver-position.json`: num_seeds=5.

---

### [SEVERITY: WARNING] W7. Bootstrap CI Inconsistency Between Data Files
- **Location**: `statistical_rigor.json` vs `cross_domain_statistics.json`
- **Issue**: For the same SALT-only n=8 analysis:
  - `statistical_rigor.json`: bootstrap_95ci = [0.29, 1.0]
  - `cross_domain_statistics.json` (salt_only): bootstrap_95ci = [0.3, 1.0]
  - Paper uses [0.29, 1.00]
- **Fix**: Minor rounding difference (likely different code paths or rounding precision). Ensure both files use the same computation. The paper's [0.29, 1.00] matches statistical_rigor.json.
- **Evidence**: Direct comparison of JSON files.

---

### [SEVERITY: WARNING] W8. sales-office Coverage Drop Rounding Creates 20x Error in Ratio
- **Location**: `main.tex` Table 1 line 185 and multiple locations citing 770x
- **Issue**: Table 1 reports sales-office drop as "0.1 [0.0, 0.1]" but the actual mean is 0.053% with CI [0.039%, 0.067%]. Rounding 0.053% to 0.1% nearly doubles the reported minimum, which then halves the reported range ratio from ~1500x to ~770x.
- **Fix**: Report "0.05 [0.04, 0.07]" in Table 1, or add a decimal place for this task only. Alternatively, use "<0.1%" to avoid implying the mean equals 0.1%.
- **Evidence**: `statistical_rigor.json` sales-office: mean=0.000529 (0.053%), ci_lo=0.000388 (0.039%), ci_hi=0.000670 (0.067%).

---

### [SEVERITY: WARNING] W9. Threshold Sensitivity Table Uses Inconsistent TP Counts Between SALT and Combined
- **Location**: `main.tex` Table 8 (Appendix threshold sensitivity) vs `cross_domain_statistics.json`
- **Issue**: Paper Table 8 has TP=3 at threshold 30-45%. This uses the SALT-only SEV definition (>50% drop, so 3 SEV tasks). But `cross_domain_statistics.json` combined threshold tests show TP=4 at threshold 30-45% with severe_threshold=15%. These use different severity cutoffs (50% vs 15%), making the threshold tests non-comparable.
- **Fix**: Make explicit which severity threshold is used in each context. The paper's threshold table should state "SEV defined as >50% coverage drop."
- **Evidence**: Paper Table 8: TP=3. `cross_domain_statistics.json` combined threshold 40: TP=4 (because item-shippoint at 18.5% is "severe" under the 15% cutoff).

---

### [SEVERITY: WARNING] W10. Paper Does Not Explain Why ACI Uses gamma=0.01 When Data Has Multiple Gammas
- **Location**: `main.tex` line 279 (Section 5.1)
- **Issue**: The ACI JSON contains results for gamma=0.001, 0.01, and 0.05. The paper reports only gamma=0.01 but the ACI table caption says "gamma=0.01, alpha=0.1". No justification for choosing gamma=0.01 over the alternatives. For severe tasks, gamma=0.001 gives *better* coverage (e.g., sales-shipcond: 80.9% vs 80.5% at gamma=0.01) while gamma=0.05 gives worse.
- **Fix**: Add a sentence justifying the gamma=0.01 choice (e.g., "We report gamma=0.01 following the original recommendation of Gibbs and Candes (2021); results for gamma in {0.001, 0.05} are qualitatively similar").
- **Evidence**: ACI JSON shows 3 gamma values per task, paper uses only one.

---

### [SEVERITY: WARNING] W11. Paper Claims "all paired p < 0.005" but ACI Table Shows p=0.005 for s-payterms
- **Location**: `main.tex` line 42 (abstract), line 166, line 406 (conclusion)
- **Issue**: The abstract and results section claim "all paired p < 0.005" for coverage drops. The paper's Table 1 shows i-shippoint with p=0.005. If this is exactly 0.005 (not rounded), then the claim "< 0.005" is incorrect. The actual p-value from the data is 0.00498, which is marginally below 0.005.
- **Fix**: Change to "all paired p <= 0.005" or verify the exact p-value. The actual value is 0.00498, which rounds to 0.005 and is technically < 0.005 at full precision. This is borderline -- consider "p < 0.01" for safety.
- **Evidence**: `statistical_rigor.json` item-shippoint paired_wilcoxon_p = 0.00498.

---

### [SEVERITY: WARNING] W12. item-incoterms ACI Results Are 3-Seed Pilot But Used Alongside 10-Seed Results
- **Location**: `main.tex` lines 284, 298 (Table 4)
- **Issue**: item-incoterms uses 3 seeds while all other tasks use 10 seeds. The delta of +1.2pp is reported as "n.s." without a p-value. With only 3 seeds, statistical testing is essentially impossible (Wilcoxon requires n>=6 for p<0.05). The table correctly marks this with a dagger but the abstract counts it as one of "8 tasks."
- **Fix**: The current handling is honest (marked with dagger), but consider adding an explicit note that no statistical test is possible with n=3. The paper should acknowledge this more prominently in the text.
- **Evidence**: `aci_item-incoterms_3seeds_pilot.json`: seeds=[42,43,44], standard_mean=0.889, aci_mean=0.900.

---

## NOTE Findings

### [SEVERITY: NOTE] N1. CQR Regression Table Header Says "50 seeds" But Data Has 5 Seeds
- **Location**: `main.tex` line 363 (Table 5 regression section header)
- **Issue**: The header says "Regression -- CQR (rel-trial / rel-f1, 50 seeds)" but the actual CQR JSON files all have num_seeds=5. This is likely a copy-paste error from the main SALT results (which do use 50 seeds).
- **Fix**: Change "50 seeds" to "5 seeds" in the regression section header.
- **Evidence**: `cqr_rel-trial_study-adverse.json`: num_seeds=5. `cqr_rel-trial_site-success.json`: num_seeds=5. `cqr_rel-f1_driver-position.json`: num_seeds=5.

---

### [SEVERITY: NOTE] N2. Figure 2 Caption Says "n12" But Analysis is n=8
- **Location**: `main.tex` line 262 (Figure 2 label: fig:n12_correlation)
- **Issue**: The figure is named "figure_n12_correlation.pdf" and the label is "fig:n12_correlation", suggesting it was created when n=12 was the main result. The current paper focuses on n=8 SALT-only. The figure caption correctly says "8 severe-shift tasks" and mentions "4 moderate-shift tasks" as lighter points, but the filename is confusing for reproducibility.
- **Fix**: Rename the figure file and label to match the current analysis focus. This is cosmetic but aids clarity.
- **Evidence**: File: `results/figure_n12_correlation.pdf`, Label: `fig:n12_correlation`.

---

### [SEVERITY: NOTE] N3. "Confidently Wrong" Section Uses Single-Seed Numbers
- **Location**: `main.tex` line 397 (Discussion)
- **Issue**: The Discussion says "sales-shipcond: 7.0 -> 3.0" for prediction set sizes. These appear to be from a single seed or early run, not the 50-seed ensemble. The ACI JSON (10 seeds) shows standard test_set_size_mean=4.6, not 7.0. The 7.0 number likely refers to validation set size.
- **Fix**: Verify the source of "7.0 -> 3.0" and use 50-seed means if available.
- **Evidence**: `aci_all_tasks_summary.json` sales-shipcond standard test_set_size_mean=4.6.

---

### [SEVERITY: NOTE] N4. SHAP Concentration Values in Code Match CSV
- **Location**: `compute_cross_domain_statistics.py` lines 36-44, `concentration_all_tasks.csv`
- **Issue**: Verification passed. The hardcoded SALT_CONCENTRATION values in the code match the CSV concentrations to 1 decimal place:
  - sales-group: code=47.3, CSV=47.30 -- OK
  - sales-payterms: code=54.2, CSV=54.18 -- OK (rounded)
  - sales-shipcond: code=50.7, CSV=50.70 -- OK
  - All other tasks match similarly.
- **Fix**: No fix needed. Values are consistent.
- **Evidence**: Direct comparison of `compute_cross_domain_statistics.py` SALT_CONCENTRATION dict vs `concentration_all_tasks.csv`.

---

### [SEVERITY: NOTE] N5. Paper Mentions "quasi-natural experiment" but Title Says "Case Study"
- **Location**: `main.tex` lines 62, 405
- **Issue**: The title says "Case Study" (correctly changed from "Natural Experiment" per revision log), but the body still uses "quasi-natural experiment" in two places (lines 62, 405). These should be consistent.
- **Fix**: Either change body text to "case study" or keep "quasi-natural experiment" in the body (acceptable since the title distinguishes it). Minor style issue.
- **Evidence**: Line 23: title says "Case Study". Lines 62, 405: text says "quasi-natural experiment."

---

### [SEVERITY: NOTE] N6. Table 1 Note Says "CV > 50%" for High Variance But Text Says "std > 30%"
- **Location**: `main.tex` line 192 (Table 1 footnote) vs line 476 (Appendix D)
- **Issue**: Table 1 footnote says "high model variance (CV > 50%)" while Appendix D says "extreme model variance (std > 30%)". These are different metrics (CV = coefficient of variation vs raw standard deviation).
- **Fix**: Use consistent terminology. CV > 50% and std > 30% may select different task sets depending on the mean. Clarify which metric is used for the asterisk in Table 1.
- **Evidence**: Table 1 footnote: "CV > 50%". Appendix D: "std > 30%".

---

### [SEVERITY: NOTE] N7. ACI Table Reports "Std Cov." from 10-Seed Subset But Main Table Uses 50-Seed
- **Location**: `main.tex` lines 289-301 (Table 4) vs lines 178-185 (Table 1)
- **Issue**: Correctly acknowledged in Table 4 footnote ("Standard coverage uses 10-seed subset, differs slightly from 50-seed Table 1"). Some values differ notably:
  - sales-group: Table 1 test=12.4%, Table 4 Std=10.4%
  - item-shippoint: Table 1 test=72.7%, Table 4 Std=82.1%
  The discrepancies are properly noted but readers may find it confusing.
- **Fix**: Consider adding 50-seed values in parentheses or a footnote for each differing value.
- **Evidence**: `statistical_rigor.json` sales-group test_coverage mean=0.124 (12.4%). `aci_all_tasks_summary.json` sales-group standard test_coverage_mean=0.104 (10.4%).

---

### [SEVERITY: NOTE] N8. Paper Uses \ie and \eg Macros Without Proper Spacing
- **Location**: `main.tex` lines 20-21 (macro definitions)
- **Issue**: The macros `\ie` and `\eg` are defined as plain text "i.e." and "e.g." without trailing spacing or proper italic formatting. When used in text (e.g., line 397: "\eg, sales-shipcond"), the lack of proper spacing after the period may cause typographic issues in some LaTeX compilers.
- **Fix**: Define as `\newcommand{\ie}{i.e.\@}` and `\newcommand{\eg}{e.g.\@}` with `\@` to ensure proper sentence spacing, or use the `xspace` package.
- **Evidence**: Lines 20-21: `\newcommand{\ie}{i.e.}` and `\newcommand{\eg}{e.g.}`.

---

## Cross-Reference Summary

### Numbers Verified as Correct
- Spearman rho=0.833, p=0.010 (SALT n=8) -- matches both JSON files
- Bootstrap 95% CI [0.29, 1.00] -- matches statistical_rigor.json
- LOO range [0.75, 0.96] -- matches (0.964 rounds to 0.96)
- LOO: 6/8 significant, 2 at p=0.052 -- matches
- All 8 Table 1 coverage values (val, test, drop, CIs) -- match statistical_rigor.json
- All 8 Table 1 p-values -- match (all < 0.005)
- All 7 ACI Table 4 rows -- match aci_all_tasks_summary.json
- All 7 entropy/ECE values (Table 7) -- match aci_all_tasks_summary.json
- SHAP concentration values (Table 9) -- match CSV
- Retraining table values -- not independently verifiable (no separate JSON)
- Placebo table values -- not independently verifiable (no separate JSON)

### Numbers That Need Correction
1. Cross-domain classification rows (Table 5): completely wrong
2. n=12 combined rho=0.676: stale, should be n=10, rho=0.745
3. n=4 moderate rho=0.632: cannot be reproduced
4. Class counts: 3 tasks differ between Table 1 and ACI JSON
5. 770x ratio: depends on rounding 0.053% to 0.1%
6. CQR seeds: Table 5 says 50, actual is 5
7. Prediction set sizes "7.0 -> 3.0": unverifiable, likely stale
8. Jaccard range "0.13-0.86": actual is 0.10-0.95
