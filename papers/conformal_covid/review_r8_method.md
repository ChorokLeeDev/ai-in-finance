# MethodCritic Analysis Report -- Round 8

**Paper**: `papers/conformal_covid/uai_2026/main.tex`
**Date**: 2026-02-20
**Scope**: Full methodological review, number verification, remaining R7 issues

---

## Executive Summary

The paper is in strong shape. The primary correlation (rho=0.853, n=16) is numerically verified, the threshold sensitivity table checks out, and the statistical apparatus is well-constructed. One stale dataset/domain count remains in Section 5.5 (a carryover text error). The five R7 moderate issues (N1, N2, N4, N5, N6) remain unaddressed in the manuscript text, though none individually threaten the core claims. One new minor numerical rounding issue is noted.

---

## Fatal Issues [RED]

None.

---

## Major Issues [ORANGE]

None.

---

## Moderate Issues [YELLOW]

### M1. Stale dataset/domain count in Section 5.5 (PERSISTS from prior rounds)

**Issue**: Section 5.5 (line 351) reads "External validation extends to 11 additional datasets across 10 non-supply-chain domains." This contradicts Section 3.1 (line 89), which correctly states "9 additional datasets spanning 9 non-supply-chain domains." The abstract (line 45), Contribution 6 (line 67), Section 4.3 (line 259), Figure 2 caption (line 294), and the framework discussion (line 378) all correctly use "9 domains." Only Section 5.5 retains the stale count.

**Evidence**: Table 6 (framework validation) lists exactly 9 external datasets: Covertype, Shuttle, Avila, PAMAP2, KDDCup99, Pendigits, Satimage, Gas Sensor, Stack Overflow. The "11 datasets / 10 domains" figure appears to be a leftover from a prior revision that included binary tasks or additional datasets in the count before exclusion.

**Consequence**: A careful reader comparing Section 3.1 with Section 5.5 will notice the discrepancy and question data provenance.

**Fix**: Change line 351 from "11 additional datasets across 10 non-supply-chain domains" to "9 additional datasets across 9 non-supply-chain domains."

---

### M2. Sub-nominal validation coverage in Avila and Gas Sensor not discussed (R7-N1, PERSISTS)

**Issue**: Avila has 5/10 seeds with validation coverage below the 90% target (min 87.3%, mean 89.9%). Gas Sensor has 4/10 seeds below 90% (min 88.0%, mean 92.1%). Both datasets have small calibration sets (Avila n_cal ~ 1,043; Gas Sensor n_cal ~ 1,954), which explains the variance, but the sub-nominal behavior is not mentioned.

**Evidence**: From `external_phase2_validation.json` (Avila) and `external_multiseed_validation.json` (Gas Sensor), verified computationally. Seeds with val coverage below 90%: Avila seeds 44 (88.1%), 47 (89.4%), 48 (88.2%), 50 (87.3%), 51 (89.9%); Gas Sensor seeds 44 (88.4%), 45 (88.4%), 47 (88.5%), 50 (88.0%).

**Consequence**: Sub-nominal validation coverage means the conformal guarantee is not met even before shift is applied. A reviewer could argue these datasets do not provide clean "robust" baselines since the calibration is already failing. This is primarily a calibration-set-size issue (non-randomized APS conservativeness is insufficient at small n_cal), but it should be disclosed.

**Fix**: Add a sentence in Section 5.5 or Appendix A.2: "Avila and Gas Sensor have small calibration sets (n_cal ~ 1,000--2,000), leading to sub-nominal validation coverage for some seeds (5/10 and 4/10 below 90%, respectively); both remain robust under shift despite this calibration noise."

---

### M3. Covertype and Satimage domain overlap (R7-N2, PERSISTS)

**Issue**: Covertype (ecological remote sensing) and Satimage (Landsat satellite image classification) are both from the UCI repository and both involve remote-sensing/image-derived features. They are counted as 2 separate domains in the "9 domains" claim. A strict domain taxonomy might count them as 1 (remote sensing), making the count "8 external domains" and reducing apparent diversity.

**Evidence**: Covertype source is "UCI / sklearn" with domain "Ecological remote sensing." Satimage source is "Landsat" with domain "satellite image classification." Both involve geospatial pixel/spectral feature classification.

**Consequence**: The "9 domains" claim slightly overstates domain diversity. Since neither is a catastrophic case alongside Covertype (Satimage is robustly non-catastrophic with C=9.0%), this does not affect the correlation, but it weakens the domain-independence argument.

**Fix**: Either (a) merge into a single "remote sensing" domain and report "8 external domains", or (b) add a footnote: "Covertype and Satimage both involve remote-sensing features but differ in target variable (forest type vs. land use) and feature space (topographic vs. spectral)."

---

### M4. RAPS 10-seed vs. main 50-seed asymmetry (R7-N6, PERSISTS)

**Issue**: Table 7 (RAPS) uses 10 seeds while Table 1 uses 50 seeds. The table footnote mentions this, but the main text in Section 5.2 does not. The 10-seed APS baselines in Table 7 diverge substantially from the 50-seed Table 1 values for high-variance tasks: s-shipcond differs by 11.2 pp (60.4% vs. 71.6%), i-shippoint by 9.2 pp (9.3% vs. 18.5%).

**Evidence**: Computed from `raps_multiseed_validation.json`. The within-table APS-vs-RAPS comparison is valid (same seed set), but readers who cross-reference Table 7 APS drops with Table 1 will see discrepancies.

**Consequence**: Without main-text acknowledgment, a reader may misinterpret the APS column in Table 7 as inconsistent with Table 1, undermining trust.

**Fix**: Add to Section 5.2: "The APS baselines in Table 7 differ from Table 1 because Table 7 uses 10 seeds (sufficient for APS-RAPS paired comparison) vs. 50 in Table 1; high-variance tasks (s-shipcond, i-shippoint) show the largest differences."

---

## Minor Issues [BLUE]

### B1. KDDCup99 JSON label inconsistency (R7-N4, PERSISTS)

The single-seed `kddcup99_validation.json` records `actual_category: robust` with drop = -0.83 pp. The multi-seed mean drop is 15.85 pp, making it "at-risk" under the >15 pp criterion. The paper correctly treats it as the FN case in Table 6, so this is purely an artifact-level inconsistency that does not affect the manuscript text.

**Fix**: Update `kddcup99_validation.json` to note that this is a single-seed result and the multi-seed classification differs.

### B2. Shuttle concentration instability not discussed (R7-N5, PERSISTS)

Shuttle has extreme concentration instability across seeds (range 19.97--46.61%, std=8.29%, 7 different top features across 10 seeds). Seed 50 has C=46.61% which exceeds the 40% threshold, yet the per-seed classification is "9/10 correct" (noted in Table 6). The instability mechanism (top feature identity changes across seeds) is described for KDDCup99 in the seed stability protocol (Section 6) but Shuttle is not mentioned despite exhibiting even more extreme instability.

**Fix**: Add Shuttle as a second example in the seed stability protocol paragraph: "Similarly, Shuttle (mean $C=30.7\pm8.3\%$) has 7 different top features across 10 seeds, with one seed exceeding 40\%; the multi-seed mean correctly classifies it as robust."

### B3. Rho rounding: 0.853 vs. 0.8529

The paper reports rho=0.853 throughout. Computed value using paper's own table values is 0.8529, which rounds to 0.853. This is correct to 3 significant figures, but the intermediate precision discrepancy is traceable to rounding of input concentration/drop values (e.g., pendigits C=14.5 vs. 14.45 in JSON). No action needed; this is within acceptable rounding.

### B4. Section 5.5 cites `\citep{dua2017uci}` for all external datasets

Not all external datasets are from UCI. PAMAP2 is from UCI, but Stack Overflow is from the relational benchmark (relbench). This is a minor citation scope issue.

### B5. KDDCup99 mean drop: 15.85 vs. 15.9

The abstract and Section 5.5 use "$15.9\pm21.4$ pp" while the JSON records 15.85. This is acceptable rounding. The std is 21.35 in JSON vs. 21.4 in text, also acceptable. However, Table 6 footnote says "15.9 pp" -- all consistent.

---

## Code Execution Results

All numerical claims verified computationally:

| Claim | Paper Value | Computed Value | Match |
|-------|------------|----------------|-------|
| n=16 Spearman rho | 0.853 | 0.8529 | Yes (rounding) |
| n=16 Kendall tau | 0.667 | 0.6667 | Yes |
| n=8 SALT rho | 0.833 | 0.833 | Yes |
| n=11 multi-seed rho | 0.818 | 0.818 | Yes |
| KDDCup99 mean drop | 15.9 pp | 15.85 pp | Yes (rounding) |
| Covertype mean drop | 81.8 pp | 81.80 pp | Yes |
| SALT drop range | 0.1--77.1% | 0.1--77.1% | Yes |
| Threshold 40% precision | 0.83 | 0.83 | Yes |
| Threshold 40% recall | 0.83 | 0.83 | Yes |
| Threshold 40% TP/FP/FN/TN | 5/1/1/9 | 5/1/1/9 | Yes |
| Mixed-effects beta_1 (3 boosting) | 1.64 | 1.64 (JSON) | Yes |
| RAPS i-shippoint worsening | 11.2 pp | 11.2 pp | Yes |

Avila sub-nominal: 5/10 seeds < 90% (confirmed).
Gas Sensor sub-nominal: 4/10 seeds < 90% (confirmed).
Shuttle seed 50 concentration: 46.61% > 40% threshold (confirmed).

---

## Reproducibility Score

**8/10**

Justification: Seeds specified (42--91), software versions given (Python 3.9, LightGBM 3.3, SHAP 0.41), calibration sizes reported, hyperparameters fixed and documented. JSON result files are comprehensive with per-seed breakdowns. Deductions: (1) analysis code is not provided in the submission or linked to a repository; (2) no environment specification file (requirements.txt / conda env); (3) random 50/50 calibration split does not specify whether the split is stratified.

---

## Recommended Actions (Priority Order)

1. **Fix Section 5.5 stale count** (M1): Change "11 additional datasets across 10 non-supply-chain domains" to "9 additional datasets across 9 non-supply-chain domains." Single-line edit.

2. **Add sub-nominal coverage disclosure** (M2): One sentence in Section 5.5 or Appendix A.2 noting Avila (5/10 seeds) and Gas Sensor (4/10 seeds) have some sub-nominal validation coverage due to small calibration sets.

3. **Address Covertype/Satimage domain overlap** (M3): Footnote or parenthetical clarifying that both are remote-sensing adjacent but differ in feature space and target.

4. **Add RAPS seed-count note to main text** (M4): One sentence in Section 5.2 explaining the 10-vs-50 seed difference.

5. **Mention Shuttle instability** (B2): Brief addition to seed stability protocol paragraph.

6. **Update KDDCup99 JSON label** (B1): Artifact cleanup, no manuscript change needed.

---

## Remaining R7 Issues Status

| R7 ID | Description | Status |
|-------|-------------|--------|
| N1 | Sub-nominal val coverage (Avila, Gas Sensor) | **Not fixed** -- M2 above |
| N2 | Covertype/Satimage domain overlap | **Not fixed** -- M3 above |
| N4 | KDDCup99 JSON label inconsistency | **Not fixed** -- B1 above |
| N5 | Shuttle instability undiscussed | **Not fixed** -- B2 above |
| N6 | RAPS 10 vs 50 seed asymmetry | **Not fixed** -- M4 above |

---

## New Issues Found This Round

| ID | Severity | Description |
|----|----------|-------------|
| M1 | YELLOW | Section 5.5 stale "11 datasets / 10 domains" count |
| B3 | BLUE | Rho rounding precision (acceptable) |
| B4 | BLUE | UCI citation scope for non-UCI datasets |

Note: M1 is distinct from the previously fixed dataset count issue (which was in Section 3.1 and the abstract). This instance in Section 5.5 was missed in prior rounds.

---

## Verdict

**MINOR REVISION REQUIRED**

The paper has no fatal or major issues. The M1 stale count in Section 5.5 is the only error that would catch a reviewer's eye as a clear inconsistency with the rest of the paper. The remaining moderate issues (M2--M4) are disclosure improvements rather than errors. All numerical claims are verified. The paper is at accept-level quality contingent on fixing the Section 5.5 count discrepancy.
