# Consistency Check Report: UAI 2026 Paper

**File**: `papers/conformal_covid/uai_2026/main.tex`
**Date**: 2026-02-10
**Checker**: Consistency Checker Agent

---

## Summary

Found **10 inconsistencies** (3 significant, 4 moderate, 3 minor). No orphaned cross-references were found. The old values (rho=0.632, rho=0.676, n=12, "theoretical grounding", "5--82%", "p < 0.005", study-adverse, site-success) have been successfully purged.

---

## SIGNIFICANT ISSUES

### [INCONSISTENCY] Sales-group class count: 462 in tables vs 459 in text
- **Location A**: Line 183 (Table 2) -- `s-group & 462`
- **Location B**: Line 368 (Table 6) -- `s-group & 462`
- **Location C**: Line 316 -- `For sales-group (459 classes), ACI expands sets to 82\%`
- **Location D**: Line 320 -- `s-group: 459 classes -> 82% Vacuous`
- **Data source**: ACI JSON (`aci_all_tasks_summary.json`) says `"num_classes": 462`
- **Fix**: Lines 316 and 320 should say "462 classes" instead of "459 classes". The tables and JSON agree on 462.

### [INCONSISTENCY] Figure caption references "4 moderate-shift tasks" -- stale from old version
- **Location**: Line 267 -- `\caption{...Lighter points: 4 moderate-shift tasks (different mechanism). The 40\% threshold (dashed) is set on validation data only.}\label{fig:n12_correlation}`
- **Problem**: The current paper has 3 cross-domain tasks (study-outcome, driver-dnf, driver-top3), not 4 "moderate-shift" tasks. The figure caption is a leftover from a previous version that had a "moderate shift" subgroup. The concept of "moderate-shift tasks" has been replaced by "binary ceiling" / "cross-domain" throughout the paper.
- **Fix**: Update caption to reference 3 cross-domain/binary tasks, e.g., `Lighter points: 3 binary/cross-domain tasks (ceiling effect; Section~\ref{sec:cross_domain}).`

### [INCONSISTENCY] Cross-domain F1 claims may not match manual calculation
- **Location A**: Line 415 -- `Applying the 40\% threshold without re-tuning to the full $n=11$ cross-domain set (Table~\ref{tab:cross}) yields Recall $= 1.0$, F1 $= 0.80$.`
- **Location B**: Line 415 -- `The 45\% threshold achieves F1 $= 0.89$ at $n=11$.`
- **Problem**: At n=11 with the 40% threshold (Step 2 only):
  - Above 40%: s-payterms (54.2%), s-shipcond (50.7%), i-shippoint (48.8%), driver-dnf (48.1%), s-group (47.3%), s-office (42.6%) = 6 tasks flagged
  - True SEV: s-payterms, s-shipcond, s-group = 3 tasks
  - TP=3, FP=3, FN=0 => Precision=0.50, Recall=1.0, F1=0.667
  - At 45%: Above 45%: 5 tasks (removes s-office). TP=3, FP=2, FN=0 => Precision=0.60, Recall=1.0, F1=0.75
  - The claimed F1=0.80 (at 40%) and F1=0.89 (at 45%) do not match these calculations under Step 2 alone. Even with Step 3 (protective-factor removing s-office), F1 at 40% = 0.75 (not 0.80).
- **Fix**: Recompute the n=11 F1 values. The confusion may arise from (a) including the protective-factor Step 3 in the count, or (b) a different definition of "true positive" for cross-domain tasks. Verify and correct the F1 numbers, or add a clarifying note about which steps are included.

---

## MODERATE ISSUES

### [INCONSISTENCY] "770x" remains in one location while "three orders of magnitude" used elsewhere
- **Location A**: Line 138 -- `Jaccard alone cannot predict the full 770$\times$ range in coverage drops.`
- **Location B**: Line 64 -- `coverage drops varying by three orders of magnitude`
- **Location C**: Line 261 -- `coverage drops differ by three orders of magnitude`
- **Location D**: Line 428 -- `Coverage drops vary by three orders of magnitude`
- **Problem**: Line 138 still uses the old "770x" phrasing. The MEMORY notes say the change was "770x -> three orders of magnitude" but line 138 was missed. Additionally, 770x is technically less than three orders of magnitude (which would be 1000x), so "nearly three orders" would be more precise.
- **Fix**: Change line 138 to use "three orders of magnitude" (or "nearly three orders of magnitude") for consistency. Consider whether "three orders of magnitude" should be "nearly three orders" everywhere for accuracy.

### [INCONSISTENCY] Abstract rounds coverage drop range to "0% to 77%" while body says "0.1% to 77.1%"
- **Location A**: Line 42 (abstract) -- `coverage drops ranging from 0\% to 77\%`
- **Location B**: Line 73 (contributions) -- `Coverage drops across 8 tasks range from 0.1\% to 77.1\%`
- **Location C**: Line 170 (results) -- `drops range from 0.1\% to 77.1\%`
- **Problem**: The abstract says "0%" which implies no drop at all for the most robust task, but the actual drop is 0.1% (statistically significant). This is misleading rounding.
- **Fix**: Change abstract to "0.1\% to 77\%" or keep "0\% to 77\%" but acknowledge this is approximate. The body text correctly says "0.1% to 77.1%".

### [INCONSISTENCY] "quasi-natural experiment" in abstract vs "case study" everywhere else
- **Location A**: Line 40 (abstract) -- `Using COVID-19 as a quasi-natural experiment across 8`
- **Location B**: Line 23 (title) -- `A COVID-19 Case Study`
- **Location C**: Line 64 (introduction) -- `we conduct an observational case study`
- **Location D**: Line 427 (conclusion) -- `Using COVID-19 as a case study`
- **Problem**: The abstract uses "quasi-natural experiment" while the title and the rest of the paper consistently use "case study." According to MEMORY notes, the title change was from "Natural Experiment" to "Case Study." The abstract's "quasi-natural experiment" is a hybrid that doesn't match either.
- **Fix**: Change abstract (line 40) to "case study" for consistency with the title and rest of the paper. Alternatively, keep "quasi-natural experiment" as the experimental design descriptor (it describes the research design, while "case study" is the paper framing), but then add "quasi-natural experiment" to the introduction as well.

### [INCONSISTENCY] Figure filename still references "n12" but paper now uses n=11
- **Location**: Line 266 -- `\includegraphics[width=\linewidth]{results/figure_n12_correlation.pdf}`
- **Problem**: The filename `figure_n12_correlation.pdf` references the old n=12 combined analysis. The paper now uses n=11. While the filename doesn't appear in the typeset paper, it could cause confusion during review if supplementary materials are inspected.
- **Fix**: Not critical for submission (filename is not visible in PDF), but consider renaming to `figure_n11_correlation.pdf` for internal consistency. The actual figure content should also be verified to show 11 points, not 12.

---

## MINOR ISSUES

### [INCONSISTENCY] Appendix baselines text calls i-plant and i-shippoint "moderate-severity" while table labels them "ROB"
- **Location A**: Line 593 -- `the two moderate-severity tasks (i-plant, i-shippoint; coverage drops of 10--19\%)`
- **Location B**: Line 196 (Table 2 footnote) -- `ROB = Robust (<20% drop)`
- **Problem**: These tasks are labeled ROB in all tables but described as "moderate-severity" in the appendix prose. They are within the ROB range (<20% drop) per the paper's own definition.
- **Fix**: Change "moderate-severity" to "moderate-drop robust tasks" or just "robust tasks with non-trivial drops (10--19\%)" to align with the ROB label used elsewhere.

### [INCONSISTENCY] Retraining improvement: "+19 pp" in 3 places vs "+18.9 pp" in body
- **Location A**: Lines 53, 75, 432 -- `+19~pp ($p=0.04$)`
- **Location B**: Line 328 -- `+18.9~pp over no retraining (Wilcoxon $p=0.04$)`
- **Problem**: The abstract, contributions, and conclusion round to +19 pp while the body gives the precise +18.9 pp. This is standard rounding and not technically wrong, but creates a minor discrepancy.
- **Fix**: Acceptable as-is (abstract rounding is standard), but could harmonize by using "+19 pp" everywhere or "~19 pp" in the abstract.

### [INCONSISTENCY] Jaccard equation uses train-test but framework uses train-validation
- **Location A**: Lines 121-124 -- Eq. 1 defines Jaccard using $A_{\text{train}}$ and $A_{\text{test}}$
- **Location B**: Line 404 -- `secondary features with train-validation Jaccard > 0.5`
- **Location C**: Line 559 -- `train-validation Jaccard=0.61`
- **Problem**: The equation defines Jaccard between train and test sets, but the framework and appendix use "train-validation Jaccard." This is intentional (the framework uses validation as a pre-deployment proxy), but the equation doesn't generalize the notation.
- **Fix**: Either (a) generalize the equation to use generic set notation ($A_1, A_2$) with explanation that it can be applied to any two data splits, or (b) add a sentence noting that the framework applies Eq. 1 with validation data substituted for test data as a pre-deployment proxy.

---

## VERIFIED CLEAN (No Issues Found)

The following items from the checklist were verified as consistent:

1. **"theoretical grounding" completely removed** -- Section 4 title is "Intuition" (line 146); no occurrences of "theoretical grounding" found.
2. **"5--82%" completely removed** -- ACI range is consistently "30--82%" in abstract (line 55) and conclusion (line 433).
3. **"p < 0.005" completely removed** -- All occurrences use "$p \leq 0.005$" (lines 42, 73, 170, 176, 428).
4. **Old rho values removed** -- No occurrences of 0.632 or 0.676 found.
5. **Old n=12 removed** -- No occurrences of "n=12" in paper text (only in figure filename).
6. **study-adverse and site-success not present** -- Confirmed absent from the paper.
7. **Rho values consistent**: $\rho=0.833$ (SALT, n=8), $\rho=0.691$ (combined, n=11), $\rho=0.883$ (COVID-era, n=9) -- all appear correctly in abstract, Section 5.3, Table 3, and conclusion.
8. **Cross-domain table (Table 6) class counts match Table 2**: s-payterms=135, s-shipcond=45, s-group=462, i-shippoint=70, s-office=25, i-incoterms=13, i-plant=35, s-incoterms=13.
9. **Binary ceiling effect discussed consistently** in abstract (lines 46-47), contribution 5 (lines 77), Section 5.3 (line 387), limitations (line 423), and conclusion (line 431).
10. **All \ref{} cross-references have matching \label{} definitions** -- no orphaned references found.
11. **ACI table values match JSON data** -- All coverage, delta, p-value, and size/classes values verified against `aci_all_tasks_summary.json`.
12. **Table 2 values match `statistical_rigor.json`** -- All 8 tasks' val coverage, test coverage, drop, and CIs verified.
13. **LOO stability ranges consistent** -- [0.75, 0.96] for SALT (lines 73, 230, 254); [0.59, 0.79] for combined (line 385).
14. **Bootstrap CIs consistent** -- [0.29, 1.00] for SALT (lines 230, 245); [0.08, 0.97] for combined (lines 247, 385); [0.39, 1.00] for COVID-era (lines 246, 389).

---

## PRIORITY ORDER FOR FIXES

1. **CRITICAL**: Sales-group class count 459 -> 462 (lines 316, 320)
2. **CRITICAL**: Figure 2 caption stale "4 moderate-shift tasks" -> "3 binary/cross-domain tasks" (line 267)
3. **IMPORTANT**: Verify and correct cross-domain F1 claims at n=11 (line 415)
4. **MODERATE**: Replace remaining "770x" with "three orders of magnitude" (line 138)
5. **MODERATE**: Harmonize "quasi-natural experiment" in abstract with "case study" elsewhere (line 40)
6. **LOW**: Fix abstract rounding "0%" -> "0.1%" (line 42)
7. **LOW**: Rename figure file from n12 to n11 (line 266)
8. **LOW**: Fix "moderate-severity" label for ROB tasks in appendix (line 593)
