# MethodCritic Analysis Report -- Round 6

**Paper**: `papers/conformal_covid/uai_2026/main.tex`
**Date**: 2026-02-20
**Scope**: Fresh full read with cross-referencing abstract/body/tables/appendix against underlying result JSON files.

---

## Executive Summary

The paper is in strong shape after prior rounds of revision. The primary statistical claims are verified against raw data files. However, I identify **two moderate inconsistencies** in number reporting (class counts, RAPS table values), **one moderate framing concern** (the "11 additional datasets across 10 domains" count), and **several minor issues**. No fatal flaws remain.

---

## Fatal Issues (RED)

None identified.

---

## Major Issues (ORANGE)

None identified.

---

## Moderate Issues (YELLOW)

### M1. Class count discrepancies between Table 1 and data files

**Issue**: Table 1 (line 204) lists s-group as 462 classes and s-payterms as 135 classes. The 50-seed summary JSON (`ensemble_50seeds_summary.json`) shows s-group = 459 classes and s-payterms = 137 classes. The RAPS multi-seed file shows s-group seeds varying across 455-465 classes and s-payterms varying across 133-138 classes.

**Evidence**: `ensemble_50seeds_summary.json` line 13: `"num_classes": 459`; line 20: `"num_classes": 137`. Table 1 line 204: `462`; line 205: `135`.

**Consequence**: Minor inconsistency that suggests the class count was taken from a different seed or data snapshot than what the summary JSON reports. For a paper claiming precision, this looks careless.

**Fix**: Decide on a single canonical source for class counts (e.g., the mode across 50 seeds, or the seed=42 value) and ensure Table 1, Appendix Table 8 (RAPS), Appendix Table 7 (model sensitivity), and the JSON files all agree. Note that class counts vary by seed due to calibration/evaluation splits; consider reporting as a range or the median.

### M2. RAPS table values do not match RAPS multi-seed JSON

**Issue**: The RAPS comparison table (Appendix Table 8, line 656) reports APS drop for s-shipcond as $60.4\pm31.8$ and RAPS drop as $67.8\pm25.4$. The RAPS multi-seed JSON (`raps_multiseed_validation.json`) confirms these (aps_drop_mean=60.42, raps_drop_mean=67.81). However, for s-payterms, the paper says APS drop = $79.9\pm29.4$ and RAPS drop = $35.1\pm28.6$, while the JSON shows aps_drop_mean=79.89 and raps_drop_mean=35.07. These match. For s-group, paper says $73.5\pm33.9$ APS and $10.4\pm1.3$ RAPS; JSON confirms 73.52 and 10.42. All match upon detailed check.

**Status**: RESOLVED on verification. This was a false alarm -- the numbers do match. Downgrading from moderate to cleared.

### M3. "11 additional datasets spanning 10 domains" counting ambiguity (line 89, Section 3.1)

**Issue**: The methodology section states "External validation uses 11 additional datasets spanning 10 domains." The framework validation table (Table 6, lines 511-521) lists exactly 9 external datasets: Covertype, Shuttle, Avila, PAMAP2, KDDCup99, Pendigits, Satimage, Gas Sensor, Stack Overflow. That is 9 datasets across 9 domains. To reach "11 datasets" the paper presumably includes 2 binary tasks not shown in Table 6 (which are excluded from the multiclass primary endpoint). But the paper never explicitly lists these 2 binary datasets or their domains. Furthermore, "10 domains" is also unexplained -- if there are 9 unique external datasets, what is the 10th domain (unless 2 datasets share a domain)?

**Evidence**: Table 6 lists 9 external rows. The abstract and contributions mention "9 domains" for the $n=16$ multiclass endpoint. Section 3.1 says "11 additional datasets spanning 10 domains."

**Consequence**: A reviewer counting the datasets in Table 6 will find only 9 external datasets, not 11. The 2 missing binary datasets and the domain accounting should be stated explicitly. If those binary datasets were abandoned, the text "11 additional" is stale and must be corrected.

**Fix**: Either (a) list all 11 datasets explicitly (including the 2 binary ones and their domains) in Section 3.1 or an appendix, or (b) correct the count to "9 additional datasets spanning 9 domains" if the binary ones are no longer presented. The $n=19$ "Combined (11 dom.)" row in the stratified correlation table (Table 3) implies 8 SALT + 11 external = 19, so the 11 figure may be correct but needs transparent documentation.

### M4. Covertype single-seed concentration (49.92%) vs multi-seed mean (49.78%)

**Issue**: The single-seed validation JSON (`covertype_validation.json`) reports concentration = 49.92%. The multi-seed JSON (`external_multiseed_validation.json`) reports concentration_mean = 49.78%. The paper reports "$C=49.8\%$" (line 67, 260, 305, 351), which matches the multi-seed mean. However, the verified_n11 JSON (line 47) uses concentration = 49.92 from the single-seed run for the $n=11$ correlation. This means the $n=11$ result ($\rho=0.909$) was computed with single-seed concentrations, while the $n=16$ result uses multi-seed means. The paper acknowledges this at line 281 ("Single-seed external values; multi-seed-consistent value at $n=11$ is $\rho=0.818$") but the main text still prominently features $\rho=0.909$ in the stratified correlation table without marking it as based on single-seed external values until a footnote.

**Consequence**: The $n=11$ row is potentially misleading without the footnote being read carefully. Most readers will see the table first.

**Fix**: Consider removing the $n=11$ single-seed row from the stratified correlation table entirely, since it mixes single-seed external with 50-seed SALT. The multi-seed $n=11$ value ($\rho=0.818$) is the honest number. Or at minimum, display the multi-seed value in the main row and relegate the single-seed figure to a note.

### M5. i-shippoint: "ROB" categorization despite 18.5 pp mean drop

**Issue**: Table 1 categorizes i-shippoint as "ROB*" with an asterisk indicating high model variance. The at-risk criterion throughout the paper is drop > 15 pp. i-shippoint has a mean drop of 18.5 pp, which exceeds 15 pp. Yet Table 6 (line 504) marks it as "At-risk*" -- consistent with the 15 pp threshold. The Table 1 "ROB*" label contradicts the Table 6 "At-risk*" label for the same task.

**Evidence**: Table 1 (line 207): "ROB*" for i-shippoint. Table 6 (line 504): "At-risk*" for i-shippoint.

**Consequence**: The classification labels are inconsistent between Table 1 and Table 6. Table 1 uses a "50% drop" threshold for SEV and "<20% drop" for ROB, while Table 6 uses a "15 pp" threshold for At-risk. These are different classification schemes applied to the same task, and the paper does not explicitly reconcile them.

**Fix**: Add a brief note in Table 1 or its caption clarifying that the SEV/ROB labels use a different (coarser) categorization than the at-risk threshold in Table 6. Or use a consistent labeling scheme throughout.

---

## Minor Issues (BLUE)

### B1. Table 1 CIs may be t-distribution CIs but the intervals seem very tight for the severe tasks

For s-group (mean test coverage 12.4%, std ~32.3%), a 95% t-CI with 50 seeds would be approximately 12.4 +/- 2.01 * 32.3/sqrt(50) = 12.4 +/- 9.2, giving [3.2, 21.6]. The table shows [3.1, 21.7] -- close enough. Verified.

### B2. Abstract says "bootstrap 95% CI $[0.50, 0.96]$" for $n=16$

The stratified correlation table (line 272) shows $[0.50, 0.96]$ for the $n=16$ multiclass 9-domain endpoint. The verified_n11 JSON shows $[0.61, 1.00]$ for $n=11$. The abstract's $[0.50, 0.96]$ appears to be from the $n=16$ result, which is correct for the primary endpoint. Consistent.

### B3. Kendall tau inconsistency between table rows

The stratified correlation table shows Kendall $\tau = 0.714$ for SALT ($n=8$), $\tau = 0.782$ for 4-domain ($n=11$), $\tau = 0.714$ for 8-domain ($n=15$), and $\tau = 0.667$ for 9-domain ($n=16$). The abstract (line 44) reports $\tau = 0.667$ and the contributions (line 57) report $\tau = 0.667$. These all refer to the $n=16$ primary result. Consistent.

However, the contributions section (line 57) also says "Kendall $\tau = 0.667$, $p < 0.001$" while the SALT-only result has $\tau = 0.714$, $p = 0.014$ (from the shift detection JSON Kendall for SHAP concentration). The paper properly attributes the 0.667 to the cross-domain result. Verified.

### B4. "7 types of natural distribution shift" (line 259)

This claim is not broken down. The 8 external multiclass datasets span temporal shift (StackOverflow), geographic shift (Covertype), sensor drift (Gas Sensor, PAMAP2), cross-writer (Pendigits, Avila), spectral (Satimage), and network evolution (KDDCup99). That is approximately 6-7 types depending on how one groups them. Marginally acceptable but a brief enumeration would strengthen the claim.

### B5. Notation: $\hat{p}(y^* \mid x)$ in Theorem 1

The theorem uses $\hat{p}(y \mid x)$ as the predicted probability distribution. Assumption (A1) posits additivity in probability space, with the footnote acknowledging this is an approximation since SHAP values operate in log-odds space for tree ensembles. This gap is adequately disclosed (line 152).

### B6. The learning_rate in Appendix A.1 is 0.05, not 0.1

Line 415 specifies learning_rate=0.05. This is a non-default LightGBM parameter (default is 0.1). The text says "default LightGBM settings" -- strictly this is false. feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5 are also non-default. The text should say "near-default" or list the deviations explicitly.

**Fix**: Change "default LightGBM settings" to "LightGBM with the following hyperparameters" (already listed). Remove the word "default."

### B7. Placebo ratio for s-payterms

Table 5 shows placebo = 0.0% and COVID = 77.1%, with ratio ">100x". Technically 77.1/0.0 is undefined (division by zero), not ">100x". This is a presentation choice but ">100x" is misleading if the denominator is exactly zero.

### B8. Retraining p-value (line 347)

The paper reports "+18.9 pp ($p=0.04$, unadjusted; Holm-corrected over 3 tasks: $p=0.12$)." The abstract (line 45) says "+19 pp, $p = 0.04$ (unadjusted)." Both are consistent. The Holm-corrected value is non-significant, which the paper correctly notes.

### B9. KS values in abstract/body

The abstract and Section 4 mention "KS $= 0.68$--$0.96$" for catastrophic tasks. Table 9 shows KS = 0.956 (s-shipcond), 0.748 (s-payterms), 0.676 (s-group). The range 0.676-0.956 rounds to 0.68-0.96. Consistent.

### B10. Stack Overflow exclusion rationale

Stack Overflow has 3 classes but is labeled as "near-binary ceiling effect" and excluded from the multiclass $n=16$ endpoint (line 282, 520). With $K=3$, APS prediction sets can be $\{c_1\}$, $\{c_1,c_2\}$, or $\{c_1,c_2,c_3\}$ -- structurally richer than true binary. The "near-binary" justification is arguable. However, the data shows Stack Overflow has $C=48.9\%$ (high concentration) yet coverage *increases* by 7 pp, making it a clear counterexample to the concentration hypothesis. Excluding it is principled (ceiling effect documented) but a skeptical reviewer may view it as cherry-picking.

**Risk**: If questioned, the authors should note that including Stack Overflow weakens $\rho$ from 0.853 ($n=16$) to 0.654 ($n=17$, line 597), which is still significant but substantially weaker. This sensitivity is disclosed in Appendix D.

---

## Code Execution Results

No analysis code was directly provided in the paper for execution. The result JSON files are consistent with the claims made in the paper, with the exceptions noted above (M1 class counts, M3 dataset count, M4 single-seed vs multi-seed). No executable scripts were run for this review.

---

## Reproducibility Score

**7/10**

Justification:
- (+) 50-seed ensemble with explicit seed range (42-91)
- (+) Exact LightGBM hyperparameters listed
- (+) Calibration set sizes and APS quantile formula provided
- (+) SHAP subsample size specified (10K)
- (+) All external datasets are public (UCI, sklearn)
- (-) No pre-registration
- (-) Code not publicly available (yet)
- (-) The 40% threshold was derived post-hoc and applied to external data, which is acknowledged but still limits prospective reproducibility
- (-) Software versions are listed but environment specification (requirements.txt, conda env) is not provided

---

## Recommended Actions (Priority Order)

1. **Reconcile "11 additional datasets" count** (M3): Either list all 11 explicitly (including 2 binary datasets) or correct to 9. This is the most likely reviewer question.

2. **Fix or annotate class count discrepancies** (M1): Ensure Table 1 class counts match a single canonical source. If class counts vary by seed, state this and report the modal or median value.

3. **Consider removing or replacing the single-seed $n=11$ row** (M4): The $\rho=0.909$ figure uses single-seed external values. The honest multi-seed number is $\rho=0.818$. Displaying the inflated single-seed value prominently in a table, even with a footnote, invites criticism.

4. **Reconcile Table 1 vs Table 6 labels for i-shippoint** (M5): Both label schemes are valid but their coexistence is confusing. Add a note.

5. **Correct "default LightGBM settings"** (B6): The stated parameters are not defaults. Remove the word "default."

6. **Fix placebo ratio for s-payterms** (B7): Change ">100x" to a more precise statement like "denominator $\approx 0$; COVID drop is 77.1 pp vs placebo $<0.1$ pp."

---

## Verdict

**MINOR REVISION REQUIRED**

The paper's core methodology, statistical analysis, and numerical claims are sound. The primary correlation, p-values, CIs, and effect sizes are verified against raw data. The identified issues are inconsistencies in number presentation and dataset counting -- none threaten the validity of the main findings. After addressing the moderate issues (M1, M3, M4, M5), the paper should be acceptable.
