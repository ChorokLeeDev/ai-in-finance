# Statistical and Mathematical Correctness Review
## main_icaif.tex — Detailed Findings

**Reviewer Role:** Statistics Referee
**Date:** 2026-03-01
**Document:** Structural Decay of Cross-Factor Predictability (ICAIF submission)

---

## 1. BONFERRONI CORRECTIONS

### Issue 1.1: In-sample 0.01/30 Bonferroni threshold
**Line 186:** `In-sample: Bonferroni $\alpha_{\text{fam}} = 0.01$ across 30 directed pairs ($\alpha/30 = 0.00033$).`

**Calculation:** 0.01 ÷ 30 = 0.000333...
**Reported:** 0.00033
**Status:** ✓ CORRECT (rounded to 5 significant figures; acceptable precision given context)

---

### Issue 1.2: OOS regional 0.05/12 Bonferroni threshold
**Line 188:** `OOS: corrected per regime ($\alpha/3 = 0.0167$) or per region-by-regime combination ($\alpha/12 = 0.0042$ for 4 regions $\times$ 3 regimes).`

**Calculation:** 0.05 ÷ 12 = 0.004166...
**Reported:** 0.0042
**Status:** ✓ CORRECT (rounded to 2 significant figures; acceptable)

---

### Issue 1.3: Bonferroni survival claims (International Table 5)
**Lines 553–556:** International replication claims Asia-Pacific ex Japan and Developed ex-US produce "Crisis-regime OOS effects surviving Bonferroni ($\alpha/12 = 0.0042$)."

**Table 4 p-values:**
- Dev. ex-US Crisis OOS: HAC-$p = <0.001$ ✓ survives 0.0042
- Asia-Pac. Crisis OOS: HAC-$p = <0.001$ ✓ survives 0.0042

**Status:** ✓ CORRECT — both p-values are reported as `<0.001`, which is clearly below the 0.0042 threshold.

---

## 2. SAMPLE SIZE RECONCILIATION

### Issue 2.1: Normal regime split (pre/post-2008)
**Line 273–275:**
```
Pre-2008 Normal (n = 3,140)
Post-2008 Normal (n = 1,557)
Sum: 3,140 + 1,557 = 4,697 (26 fewer than Table 1's Normal total of 4,723)
```

**Table 1 (Regime Summary):** Normal = 4,723 days
**Footnote explanation:** "26 fewer than Table~\ref{tab:regimes}'s Normal total of $4{,}723$, due to regime-boundary exclusion at lag~1."

**Arithmetic:** 4,723 - 26 = 4,697 ✓ CORRECT
**Status:** ✓ ADEQUATE EXPLANATION — lag-1 boundary exclusion is a standard practice in time-series analysis; the 26-day difference (~0.55%) is negligible and expected.

---

### Issue 2.2: Frozen OOS sample sizes
**Table 3 (Frozen OOS, 2013–2024):**
```
Normal: n = 724
Elevated: n = 953
Crisis: n = 1,119
Total: 724 + 953 + 1,119 = 2,796
```

**Expected:** 2013–2024 is 12 trading years ≈ 3,000 trading days.
**Status:** ✓ PLAUSIBLE — The OOS period spans ~2,796 days, reasonable for ~12 trading years (accounting for weekends, holidays, partial years).

---

### Issue 2.3: Neural/Four-Model diagnostic sample sizes
**Table 2 (Four-Model Diagnostic, labeled "Neural"):**
```
Normal: n = 4,496
Elevated: n = 2,792
Crisis: n = 1,017
```

**vs. Table 1 regime totals:**
```
Normal: 4,723
Elevated: 3,023
Crisis: 1,071
```

**Caption explanation (lines 351–353):** "Sample sizes reflect lag-9 input window and train/validation split ($n_{\text{eff}} < n_{\text{regime}}$)."

**Status:** ⚠ MEDIUM — The explanation is present but vague.
- Normal: 4,723 → 4,496 is a 227-day loss (4.8%), plausible for lag-9 window + train/validation split.
- Elevated: 3,023 → 2,792 is a 231-day loss (7.6%), slightly higher but acceptable.
- Crisis: 1,071 → 1,017 is a 54-day loss (5.0%), consistent.

**Recommendation:** The explanation could specify:
  - Lag window reduces by ⌊9⌋ = 9 observations (per regime)
  - Train/validation split ratio (e.g., 70/30) further reduces effective sample
  - Expected reduction ≈ 5–10% is reasonable for these parameters

The explanation exists but is terse. This is acceptable but could be clearer.

---

## 3. DELTA-F EFFECT SIZE CALCULATION

### Issue 3.1: MOM→SMB ΔF reporting
**Line 50 (abstract) and Line 535 (main text):**
```
"MOM→SMB achieves near-perfect OOS replication (ΔF = 0.1%)"
Lines 533–535: "in-sample Normal F = 130.7, in-sample Crisis F = 29.8,
frozen OOS Normal F = 130.6 (ΔF = 0.1%)"
```

**Calculation:**
```
ΔF = (|130.7 - 130.6| / 130.7) × 100
   = (0.1 / 130.7) × 100
   = 0.0764...%
   ≈ 0.076%
```

**Reported:** 0.1%
**Status:** ⚠ MEDIUM — Technically a rounding to 1 significant figure, but the exact value is 0.076%, which rounds to 0.08% (2 sig figs) rather than 0.1%.

**Recommendation:** Revise to "ΔF = 0.08%" or "ΔF ≈ 0.07%" for precision. Alternatively, report as "ΔF < 0.1%" if conservative rounding is intentional.

---

## 4. P-VALUE SANITY CHECKS

### Issue 4.1: Normal regime HML→SMB
**Line 251:** p-value = **8.75 × 10⁻⁹**

**From Table 2 (Four-Model Diagnostic, Normal regime):**
- Linear MSE improvement: 0.86% (marked **)
- This suggests a small effect size with high significance

**Status:** ⚠ MEDIUM — The p-value is extraordinarily small (8.75 × 10⁻⁹) for an effect size (ΔR² = 2.06%, line 271) that is modest. This is mathematically possible but warrants explanation:
  - Large sample size (n = 3,140 pre-2008, line 271)
  - Granger F-test on lag coefficients, not R² directly
  - With n = 3,140, even a tiny standardized effect can yield minuscule p-values

**Verification (Chow test):** Line 284 reports F(3, n-6) = 9.68, p = 2.29 × 10⁻⁶. The denominator degrees of freedom (n-6) is appropriate for a 3-lag model. This is internally consistent.

**Conclusion:** ✓ PLAUSIBLE — The extreme p-value is unusual but justified by the large sample size. The ΔR² = 2.06% is genuine but modest; statistical significance ≠ practical significance.

---

### Issue 4.2: Quandt-Andrews sup-F break
**Line 279:** "supremum F = 21.2, p = 1.23 × 10⁻¹³"

**Status:** ⚠ MEDIUM — A p-value of 10⁻¹³ from a supremum test is extremely small and depends heavily on:
  - Sample size (8,817 total days, line 154)
  - Number of possible break points tested (reduces effective degrees of freedom due to testing the distribution across many break dates)

The Quandt-Andrews test p-value computation is standard but must account for the multiplicity implicit in testing all possible break dates. The reported p-value **assumes correct asymptotic distribution**. No reference is provided to the exact critical values or bootstrap procedure used.

**Recommendation:** Specify whether p-value is from:
  1. Asymptotic distribution (Andrews 1993)
  2. Bootstrap with n resamples (and n value)
  3. Simulation under null

---

### Issue 4.3: OOS Elevated result robustness
**Line 484:** OOS Elevated HAC-$p$ = **0.043** (survives α/3 = 0.0167? NO)

**Expected threshold for 3-regime correction:** α/3 = 0.05/3 = 0.0167
**Reported p-value:** 0.043
**Status:** ✗ FAILS 3-regime Bonferroni correction (0.043 > 0.0167)

**Lines 497–498 acknowledge:** "does not survive 3-regime Bonferroni ($\alpha/3 = 0.0167$; HAC $p = 0.043$)"

**Conclusion:** ✓ CORRECTLY ACKNOWLEDGED — Authors explicitly state this does not survive correction. Transparency is good.

---

## 5. TRANSFER ENTROPY Z-SCORES

### Issue 5.1: TE z-score magnitudes (Normal regime)
**Table 3 (Transfer Entropy):**
```
Normal: HML→SMB z = 2.45**, SMB→HML z = 5.37***
```

**Status:** ✓ PLAUSIBLE — z-scores of 2.45 and 5.37 correspond to p-values:
- z = 2.45 → p = 0.007 ✓ matches "p < 0.01" in caption
- z = 5.37 → p < 10⁻⁶ ✓ matches "p < 10⁻⁶" in caption

**Relationship verified:** Both z-scores and p-values are internally consistent.

---

## 6. QUANTILE GRANGER COEFFICIENTS

### Issue 6.1: Table 4 (Quantile Granger)
**Line 430:** SMB→HML: β₀.₉₅ = **0.212** (marked bold)

**Context (lines 446–453):**
```
"SMB→HML operates through tail dependence (β̂₀.₉₅ = 0.212, 8× the median)"
```

**Calculation check:** 0.212 ÷ (-0.026) ≈ -8.15 (actually, |0.212| / |-0.026| ≈ 8.15)

**Status:** ⚠ MEDIUM — The coefficient at τ = 0.95 (0.212) flips sign from the median (-0.026). This is a large tail effect and is plausible for small-cap factor returns, but:
  1. The Wald test p-value is reported as **0.001**, which is strong evidence of coefficient heterogeneity across quantiles.
  2. The interpretation as "8× the median" is slightly misleading: the ratio is 0.212 / 0.026 ≈ 8.15 in absolute value, but the **sign difference** (positive vs. negative) is arguably more important than the magnitude ratio.

**Recommendation:** Clarify that the relationship reverses sign in the right tail (median: negative, 95th percentile: positive), not just increases in magnitude.

---

## 7. DEGREES OF FREEDOM IN CHOW TEST

### Issue 7.1: F(3, n-6) specification
**Line 284:** "Chow test at January 2008 confirms continued decay ($F(3,n{-}6) = 9.68$, $p = 2.29 \times 10^{-6}$)"

**Specification:** F(3, n-6)

**Standard Chow test form:** F(k, n-2k) where k = number of regressors

**Context:** Granger regression with 3 lags of HML → SMB suggests:
- Regressor count: k = 1 (HML lag terms) + 1 (intercept) = 2 for the restricted model
- Full model: either 3 lags (k = 3) or with controls (k = 6?)

**Analysis:**
- If k = 3: F(3, n-6) is CORRECT (numerator = 3 restricted coefficients, denominator = n - 2×3)
- Line 284 states "3-lag model" (or similar) is used; F(3, n-6) fits a model with 3 HML lags

**Status:** ✓ CORRECT — The degrees of freedom match a 3-lag Chow test:
- Numerator df = 3 (testing 3 HML lag coefficients)
- Denominator df = n - 6 (n observations minus 2×3 coefficients per regime)

---

## 8. CONFIDENCE INTERVAL COMPUTATION

### Issue 8.1: Post-2008 Normal coefficient CI
**Line 288:** "Post-2008 coefficient: $\hat{\beta} = 0.012$, 95\% CI $[-0.049, 0.073]$"

**Standard formula:** β ± 1.96 × SE

**Expected properties:**
- CI should be centered on β = 0.012
- Upper bound: 0.012 + 1.96 × SE = 0.073 → SE = 0.0311
- Lower bound: 0.012 - 1.96 × SE = -0.049 → SE = 0.0311

**Check:**
```
CI width = 0.073 - (-0.049) = 0.122
Half-width = 0.061
SE = 0.061 / 1.96 ≈ 0.0311
```

**Symmetry:** (0.012 + (-0.049)) / 2 = -0.0185 (NOT 0.012)
**Issue:** The CI is **asymmetric** around the point estimate!

**Recalculation:**
```
Point estimate: β = 0.012
Lower: -0.049 → z-score = (0.012 - (-0.049)) / SE = 0.061 / SE
Upper: 0.073 → z-score = (0.073 - 0.012) / SE = 0.061 / SE
```

Both margins are 0.061 units from the center, so SE = 0.0311 is consistent.

**Status:** ✓ CORRECT — Despite appearance of asymmetry in visual representation, the CI is correctly computed as β ± 1.96 × 0.0311. The apparent asymmetry is an artifact of the point estimate (0.012) not being exactly centered; this is normal for Wald CIs when the point estimate is near boundary.

---

## 9. DIVISION BY ZERO / IMPOSSIBLE VALUES

### Issue 9.1: Regime statistics
**Table 1:**
```
Normal: Days = 4,723 (53.6%), P(z_t = z_{t-1}) = 0.994
Elevated: Days = 3,023 (34.3%), P(z_t = z_{t-1}) = 0.991
Crisis: Days = 1,071 (12.1%), P(z_t = z_{t-1}) = 0.993
```

**Checks:**
- Proportions: 53.6% + 34.3% + 12.1% = 100.0% ✓
- Transition probabilities all ∈ (0, 1) ✓
- Expected days: 8,817 × 0.536 ≈ 4,726 ✓ (matches 4,723 within rounding)

**Status:** ✓ ALL VALID — No division by zero, no impossible values.

---

### Issue 9.2: Student-t degrees of freedom
**Line 171:** "$\hat{\nu}_{\text{Normal}} = 6.2$, $\hat{\nu}_{\text{Elevated}} = 3.9$, $\hat{\nu}_{\text{Crisis}} = 5.5$"

**Constraints:** ν > 0 (required for Student-t distribution)
**Range:** ν ∈ [3.9, 6.2], all well-defined
**Interpretation:**
- ν < ∞ indicates heavy tails (empirically justified)
- ν > 2 ensures finite variance ✓
- ν > 4 ensures finite excess kurtosis ✓

**Status:** ✓ PLAUSIBLE — All parameters are well-defined and consistent with financial returns data.

---

## 10. TRANSFER ENTROPY PERMUTATION TEST

### Issue 10.1: TE permutation procedure
**Lines 148 & 396:** "Frenzel--Pompe kNN for directed information flow" (Table 3 caption: "$k = 5$, 200 permutations")

**Status:** ✓ STANDARD — The procedure (kNN-based TE with permutation test) is standard practice. The sample size (200 permutations) is adequate for exploratory analysis but underpowered for extreme p-values < 0.005.

**Recommendation:** Given that TE for SMB→HML achieves p < 10⁻⁶, consider whether 200 permutations sufficiently capture the null distribution at extreme tails. Increasing to 1,000 permutations would strengthen confidence in p-values < 0.01.

---

## 11. ADDITIONAL ISSUE: HAC BANDWIDTH SENSITIVITY (Table 5)

### Issue 11.1: OOS Elevated result crosses α = 0.05
**Table 5 (HAC Bandwidth Sensitivity):**
```
B = 1: p = 0.041 ✓ < 0.05
B = 2: p = 0.043 ✓ < 0.05 [primary]
B = 4: p = 0.048 ✓ < 0.05
B = 6: p = 0.056 ✗ > 0.05 [NW default]
```

**Issue:** The primary result (HAC p = 0.043, line 484) depends on bandwidth selection. At the Newey-West default (B = 6), the result **loses significance at α = 0.05**.

**Lines 501–502 acknowledge:** "is sensitive to bandwidth (Table~\ref{tab:bandwidth}: $p$ crosses 0.05 at NW default)"

**Status:** ✓ CORRECTLY ACKNOWLEDGED — Authors explicitly flag the bandwidth sensitivity and default choice. This increases uncertainty around the OOS Elevated result, which aligns with the "exploratory Tier 3" designation.

---

## 12. PERMUTATION TEST (Circularity check)

### Issue 12.1: OOS permutation test
**Line 505:** "The permutation test ($p = 0.022$, 50,000 shuffles) demonstrates that the OOS signal is not a circularity artifact"

**Interpretation:** Shuffling regime labels within sample while preserving temporal structure yields p = 0.022, suggesting the OOS Elevated signal is real (not a labeling artifact).

**Status:** ⚠ MEDIUM — The permutation test correctly rules out one type of circularity (regime-label dependence) but does NOT address:
  1. **Regime prevalence drift** (acknowledged lines 495–496: Elevated share goes from 13.7% train to 33.7% test)
  2. **Bonferroni correction** (acknowledged line 497: fails α/30 and α/3 thresholds)
  3. **Bootstrap reweighting sensitivity** (acknowledged lines 499–500: p = 0.153 when reweighted to training prevalence)

**Conclusion:** The permutation test is valuable but insufficient alone. The authors properly acknowledge its limitations.

---

## SUMMARY TABLE: ISSUES BY SEVERITY

| Item | Issue | Severity | Status |
|------|-------|----------|--------|
| 1.1 | Bonferroni 0.01/30 = 0.00033 | LOW | ✓ Correct |
| 1.2 | Bonferroni 0.05/12 = 0.0042 | LOW | ✓ Correct |
| 1.3 | Intl. table Bonferroni survivals | LOW | ✓ Correct |
| 2.1 | Normal regime split (3140+1557) | MEDIUM | ✓ Explained |
| 2.2 | OOS sample sizes | LOW | ✓ Plausible |
| 2.3 | Neural table n values | MEDIUM | ✓ Explained (could be clearer) |
| 3.1 | ΔF = 0.076% reported as 0.1% | MEDIUM | ⚠ Should be 0.08% |
| 4.1 | p = 8.75×10⁻⁹ for ΔR² = 2% | MEDIUM | ✓ Plausible (large n) |
| 4.2 | Quandt-Andrews sup-F p-value | MEDIUM | ⚠ Missing methodology detail |
| 4.3 | OOS Elevated p = 0.043 vs. α/3 = 0.0167 | MEDIUM | ✓ Correctly acknowledged |
| 6.1 | Quantile sign flip (β median vs. tail) | MEDIUM | ⚠ Could highlight sign reversal |
| 7.1 | Chow F(3, n-6) degrees of freedom | LOW | ✓ Correct |
| 8.1 | Post-2008 CI [-0.049, 0.073] | LOW | ✓ Correct |
| 9.1 | Regime proportions & transitions | LOW | ✓ Valid |
| 9.2 | Student-t ν parameters | LOW | ✓ Valid |
| 11.1 | HAC bandwidth sensitivity | MEDIUM | ✓ Acknowledged |
| 12.1 | Permutation test circularity check | MEDIUM | ✓ Acknowledged limitations |

---

## CRITICAL ISSUES: NONE

No CRITICAL (wrong) errors detected.

---

## MEDIUM ISSUES REQUIRING ATTENTION

### 1. **ΔF Rounding (Line 50, 535)**
- **Current:** ΔF = 0.1%
- **Correct:** ΔF = 0.076% ≈ 0.08%
- **Recommendation:** Revise to 0.08% or state "ΔF ≈ 0.07%"

### 2. **Neural Table Sample Size Explanation (Lines 351–353)**
- **Current:** Vague caption about "lag-9 input window and train/validation split"
- **Recommendation:** Specify expected loss quantitatively:
  ```
  "Sample sizes reflect lag-9 window (−9 per regime) and 70/30 train/validation
  split, resulting in 5–10% effective-sample reduction per regime."
  ```

### 3. **Quantile Granger Sign Reversal (Line 430, Table 4)**
- **Current:** "operates through tail dependence (β̂₀.₉₅ = 0.212, 8× the median)"
- **Issue:** Sign reversal (median = −0.026, tail = +0.212) is as important as magnitude
- **Recommendation:**
  ```
  "operates through tail-dependent reversal: negative in median returns
  (β̂₀.₅ = −0.026) but strongly positive in upper tail (β̂₀.₉₅ = 0.212),
  confirmed by Wald p = 0.001"
  ```

### 4. **Quandt-Andrews p-value Methodology (Line 279)**
- **Current:** "supremum F = 21.2, p = 1.23 × 10⁻¹³" (no reference to computation method)
- **Recommendation:** Add footnote specifying:
  ```
  "Following Andrews (1993); p-value computed via asymptotic distribution
  of sup-F statistic under continuous break alternative."
  ```

---

## LOW ISSUES (Notation/Presentation)

1. **Table 3 caption ("Neural"):** Rename to "Four-Model Diagnostic" for consistency with text (line 349).
2. **Regime boundary lag-1 exclusion:** While explained in footnote, consider moving to main text for clarity.

---

## FINAL ASSESSMENT

**Overall Statistical Rigor:** GOOD

- **Strengths:**
  - Bonferroni corrections properly computed
  - Sample sizes reconciled with transparent explanations
  - Confidence intervals correctly computed
  - No impossible values or division-by-zero errors
  - Limitations explicitly acknowledged (Tier 3 exploratory status, bandwidth sensitivity, permutation test limitations)

- **Weaknesses:**
  - ΔF = 0.1% should be 0.08% (minor rounding)
  - Some vague explanations (e.g., neural table sample sizes)
  - Quantile reversal could be highlighted more clearly
  - Quandt-Andrews methodology not fully specified

**Recommendation:** PUBLISH WITH MINOR REVISIONS addressing items 1–4 above.

---

**Report compiled by:** Statistics Referee
**Date:** 2026-03-01
