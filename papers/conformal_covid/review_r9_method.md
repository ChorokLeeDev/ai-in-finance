# MethodCritic Analysis Report -- Round 9

**Paper**: `/Users/i767700/Github/ai-in-finance/papers/conformal_covid/uai_2026/main.tex`
**Date**: 2026-02-20
**Focus**: Numerical consistency, disclosure gaps, remaining open items from R8

---

## Executive Summary

The paper is in strong shape after 8 rounds of revision. No fatal issues remain. This review identifies one moderate numerical error (Stack Overflow concentration value in Table 5 contradicts source data by ~6.6x), two minor disclosure gaps (sub-nominal validation coverage, seed-count asymmetry), and two transparency notes. The primary claims and the n=16 primary endpoint are unaffected by these issues.

---

## Verified Claims

The following items from R8 were checked and are now correctly stated:

1. **Section 5.5 count**: "9 additional datasets across 9 non-supply-chain domains" -- VERIFIED in abstract (line 45), Section 3.1 (line 89), and Section 5.5 (line 351). The count is consistent across all three locations.

2. **n=16 multiclass primary endpoint**: 8 SALT + 8 external multiclass (excluding Stack Overflow) = 16. Arithmetic is correct.

3. **Threshold sensitivity table** (Table 4): TP=5, FP=1, FN=1 at 40% -- consistent with Table 5 framework validation (4 SALT at-risk + Covertype = 5 TP; s-office = 1 FP; KDDCup99 = 1 FN; i-shippoint is at-risk but correctly flagged).

4. **LightGBM hyperparameters** (Appendix A.1): Described as "fixed" settings, not "default" -- accurate and avoids the common confusion with actual LightGBM defaults (lr=0.1, ff=1.0, bf=1.0).

5. **External seed determinism counts**: "6 deterministic, 1 near-deterministic" (line 67) -- verified: Avila 10/10, PAMAP2 10/10, Pendigits 10/10, Satimage 10/10, Gas Sensor 10/10 = 5 deterministic robust + Covertype 10/10 deterministic catastrophic = 6 deterministic total; Shuttle 9/10 = 1 near-deterministic. Correct.

---

## Issues Found

### Moderate Issues

**M1: Stack Overflow concentration value in Table 5 is incorrect.**

- **Issue**: Table 5 (line 520) reports Stack Overflow concentration as 7.4%. The source data (`/Users/i767700/Github/ai-in-finance/papers/conformal_covid/results/external_multiseed_validation.json`, line 522-523) shows concentration_mean = 48.89%, concentration_std = 0.91%. The JSON also shows predicted_from_mean = "vulnerable", contradicting the table's "ROB" Step 2 classification.
- **Evidence**: 7.4% vs 48.89% is a ~6.6x discrepancy. Every single seed in the JSON shows concentration in the 47-51% range. The value 7.4% does not appear anywhere in the seed-level results.
- **Consequence**: A reader checking the table would conclude Stack Overflow has very low concentration (7.4%), which would make the "near-binary ceiling effect" exclusion seem arbitrary. The actual high concentration (48.89%) combined with robust coverage (-7.01 pp drop, i.e., coverage *improves*) is precisely what makes it the strongest evidence for the binary ceiling claim -- a high-concentration task that is nevertheless protected.
- **Impact on primary results**: None. Stack Overflow is excluded from the n=16 primary endpoint. But the table is factually wrong.
- **Fix**: Change "7.4" to "48.9" in Table 5, line 520. Change "ROB" in the Step 2 column to "VULN" (since 48.9% > 40% threshold). Update the Table 5 footnote (line 530) to note that Stack Overflow is a VULN prediction with robust outcome, providing direct evidence that the binary ceiling effect overrides the concentration signal. Update the in-sample F1 note if needed (this adds a second FP to the external set when Stack Overflow is included, but it is excluded from the primary endpoint so the reported F1 is unaffected).

**M2: "9 non-supply-chain domains" claim -- domain overlap between Avila/Pendigits and Covertype/Satimage.**

- **Issue**: The paper claims 9 external datasets span "9 non-supply-chain domains." However: (a) Avila Bible (handwriting analysis of a historical manuscript) and Pendigits (handwritten digit recognition) are both handwriting/pen-stroke recognition domains; (b) Covertype (forest cover type from cartographic variables including elevation and satellite-derived features) and Satimage (satellite image multispectral classification) are both remote sensing/geospatial classification domains.
- **Evidence**: The Pendigits validation JSON explicitly labels its domain as "handwriting/HCI." Avila's primary features are calligraphic measurements. Both involve pen-stroke or handwriting-derived features. Similarly, Covertype and Satimage both classify land/terrain types from remotely sensed features.
- **Consequence**: If these pairs are collapsed to single domains, the count would be 7 external domains (not 9), and the total would be 8 domains (not 9). This does not affect n=16 (which counts tasks, not domains) or the correlation, but it inflates the apparent diversity of the validation set.
- **Fix**: Either (a) change "9 non-supply-chain domains" to "9 non-supply-chain datasets" throughout (abstract, Section 3.1, Section 5.5), which is strictly correct; or (b) provide brief justification for why these are distinct domains (e.g., Avila uses calligraphic style features while Pendigits uses spatial coordinates; Covertype uses cartographic/topographic variables while Satimage uses multispectral pixel values). Option (a) is safer and requires only word substitution.

### Minor Issues

**N1: Sub-nominal validation coverage not discussed.**

- **Issue**: Multiple external datasets show validation coverage below the 90% target for a substantial fraction of seeds:
  - Avila: 5/10 seeds below 90% (range 87.3-92.7%; mean 89.9%)
  - Gas Sensor: 4/10 seeds below 90% (range 88.0-98.3%; mean 92.1%)
  - Pendigits: 5/10 seeds below 90% (range 88.8-93.1%; mean 90.1%)
  - Satimage: 2/10 seeds below 90% (range 88.2-93.3%; mean 90.6%)
  - SALT s-group: mean val coverage = 83.6% (well below 90%, all 50 seeds presumably below nominal)
- **Evidence**: All from source JSON files and Table 1.
- **Consequence**: Sub-nominal validation coverage means the conformal predictor is already miscalibrated before shift occurs. For s-group, the reported 71.2 pp "drop" is measured from an 83.6% baseline, not from 90%. The actual deviation from nominal is 83.6% to 12.4% = 71.2 pp, but the "expected" drop from nominal (90% to 12.4% = 77.6 pp) would be larger. This is not a validity threat to the correlation analysis (which uses observed val-test differences), but it warrants a sentence acknowledging that several tasks/datasets show baseline miscalibration.
- **Fix**: Add a brief note in Section 3.2 or Appendix A.2: "We note that validation coverage varies across seeds and datasets; some seeds show sub-nominal coverage (e.g., s-group mean 83.6%), reflecting finite-sample calibration variability especially for high-cardinality tasks. Coverage drops are computed as observed val-test differences, so this does not bias the diagnostic comparison."

**N5: Shuttle single-seed anomaly not characterized.**

- **Issue**: Shuttle is "9/10 seeds" deterministic, with seed 50 showing concentration = 46.61% (vs. mean 30.66%) -- well above the 40% threshold. The paper notes "9/10 seeds" but does not discuss what drives this one anomalous seed.
- **Evidence**: `/Users/i767700/Github/ai-in-finance/papers/conformal_covid/results/external_phase2_validation.json`, seed 50: C=46.61%, top_feature = "A6". Other seeds have different top features (A9, A8, A1, A3, A5, A7) and lower concentrations (19.97-38.64%). Notably, the top feature identity changes across seeds (7 different features across 10 seeds), consistent with a flat importance distribution where random variation determines the top feature.
- **Consequence**: The Shuttle case actually strengthens the seed stability protocol recommendation (Section 6) -- it shows exactly why multi-seed averaging is needed. But the unstated detail (top feature instability) would help readers understand.
- **Fix**: Optional. Could add a parenthetical in the Table 5 footnote or Section 6: "Shuttle (9/10 seeds, C=30.7 +/- 8.3%): the one anomalous seed (C=46.6%) reflects top-feature identity instability across seeds, consistent with a flat importance distribution."

**N6: Seed count asymmetry mentioned only in table footnotes.**

- **Issue**: SALT uses 50 seeds (primary), RAPS uses 10 seeds (Table 7), external datasets use 10 seeds (Table 5). The asymmetry is noted in Table 7's footnote ("APS drops differ from 50-seed Table 1 due to smaller seed range") but not in the main text body.
- **Evidence**: Lines 109 (50 seeds), 645 (10-seed RAPS), Table 5 header (10/10 seeds).
- **Consequence**: A reader of the main text may not realize that precision of external drop estimates (10 seeds) is lower than SALT drop estimates (50 seeds), which affects the reliability of the n=16 correlation differently for different points. KDDCup99's high variance (std=21.4 pp on 10 seeds) is the clearest consequence.
- **Fix**: Add one sentence in Section 3.3 or Section 5.5: "External datasets use 10 seeds (vs. 50 for SALT) due to computational constraints; drop estimates are consequently less precise for external datasets, as reflected in the wider KDDCup99 confidence band."

### Informational Notes (no action required)

**I1: Covertype validation coverage is borderline (89.98% mean, with seeds at 89.84-90.11%).** This is essentially nominal given finite-sample correction, and the coverage drop to 8.19% is unambiguous regardless.

**I2: The paper correctly avoids calling the LightGBM settings "default."** Previous memory notes that lr=0.05, ff=0.8, bf=0.8 are NOT LightGBM defaults. The paper's "fixed hyperparameters" language is accurate.

---

## Severity Summary

| ID | Severity | Description | Primary result impact |
|----|----------|-------------|----------------------|
| M1 | MODERATE | Stack Overflow C=7.4% in Table 5 contradicts source data (48.9%) | None (excluded from n=16) |
| M2 | MODERATE | "9 domains" claim -- Avila/Pendigits and Covertype/Satimage overlap | None (n=16 counts tasks) |
| N1 | MINOR | Sub-nominal val coverage (Avila, Gas Sensor, Pendigits, s-group) | None |
| N5 | MINOR | Shuttle 1-seed anomaly (C=46.6%, top feature instability) | None |
| N6 | MINOR | 50-seed vs 10-seed asymmetry not in main text | None |

---

## Recommended Actions (Priority Order)

1. **Fix Stack Overflow concentration in Table 5**: Change 7.4% to 48.9%, Step 2 from ROB to VULN. This is a factual correction. (~2 min edit)

2. **Resolve "9 domains" claim**: Change to "9 datasets" or add domain-distinction justification. (~5 min edit)

3. **Add sentence on sub-nominal validation coverage**: Brief acknowledgment in Section 3.2 or Appendix A.2. (~5 min edit)

4. **Add sentence on seed count asymmetry**: One sentence in Section 3.3 or 5.5. (~2 min edit)

5. **Optionally characterize Shuttle anomaly**: Parenthetical in Table 5 footnote. (~2 min edit)

---

## Verdict

**MINOR REVISION REQUIRED**

The paper's primary claims, statistical analyses, and conclusions are sound. The Stack Overflow table error (M1) is the most urgent fix -- it is a clear factual mistake that any careful reviewer would catch, even though it does not affect the primary endpoint. The domain-overlap issue (M2) is a framing choice rather than a validity threat. All other items are minor disclosure improvements. None of the issues identified affect the n=16 correlation, the theorem verification, or the core conclusions.
