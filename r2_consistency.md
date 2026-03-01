# CONSISTENCY REVIEW: Internal Contradictions in main_icaif.tex

**Reviewer Role:** Hostile consistency auditor
**Date:** March 1, 2026
**Status:** COMPREHENSIVE AUDIT COMPLETE

---

## Executive Summary

Systematic review of all 10 consistency categories identified **THREE CRITICAL issues** and **ONE MEDIUM issue** affecting the paper's credibility and internal coherence.

---

## ISSUES FOUND

### ISSUE 1: QUANTILE GRANGER SAMPLE SIZE CONTRADICTION [CRITICAL]

**Location:**
- **Line 420** (Table 4 caption): `Quantile Granger: Normal Regime ($n = 2{,}485$, pre-2008 Normal subsample after lag exclusion)`
- **Lines 271-275** (Results text): Pre-2008 Normal sample with "26 fewer than Table~\ref{tab:regimes}'s Normal total of $4{,}723$, due to regime-boundary exclusion at lag~1" → yields n = 4,697 regime-assigned observations

**Contradiction:**
Table 4 claims the quantile Granger sample is n = 2,485 for "pre-2008 Normal regime." However:
- Line 271 states: "Pre-2008 Normal ($n = 3{,}140$)" for the univariate Granger test on the same subsample
- The multivariate quantile Granger should have an even larger effective sample, not 45% smaller (2,485 vs 3,140)
- This 655-observation discrepancy (~21% reduction) is unexplained and inconsistent with the lag-exclusion rationale (which would affect both tests identically)

**Implications:**
- Quantile Granger statistics (Wald p-values) may be misleading if n is overstated or understated
- The tail-dependence mechanism (Wald p = 0.001, β̂₀.₉₅ = 0.212) relies on this n
- Readers cannot verify statistical power

**Explanation Needed:**
Why does univariate Granger yield n=3,140 but quantile Granger yields n=2,485 when both test identical data with identical lag structures?

**Rating:** **CRITICAL**

**Proposed Fix:**
Either (1) explain the 655-observation gap in Table 4 caption with specific exclusion criterion, or (2) if n=2,485 is correct, revise line 271 to match, or (3) provide the quantile-Granger effective sample derivation. Recommend adding footnote: "Quantile Granger uses [specific exclusion] reducing n from 3,140 to 2,485."

---

### ISSUE 2: OOS FROZEN SAMPLE SIZE INCONSISTENCY [CRITICAL]

**Location:**
- **Lines 158-159** (Data section): "Under percentage units, the frozen OOS yields $n = 953$ Elevated-regime days; decimal units yield $n = 836$ (agreement 86.3%)."
- **Line 484** (Table 3, OOS results): "Elevated & 953 & \textbf{0.003}"
- **Abstract, line 47**: "Elevated-regime signal ($F$-$p = 0.003$)"

**Contradiction:**
The paper reports finding an OOS signal in the Elevated regime (n=953, F-p=0.003) as the primary OOS result. However:
- Lines 158-159 reveal that the reported n=953 is **only under percentage-unit scaling**
- Under decimal-unit scaling (equally valid), n=836 with 86.3% agreement—implying 8.7% of regime classifications are discordant
- Agreement of 86.3% is **below 90%** and suggests non-negligible scale-dependent regime-reassignment
- No replication of OOS test under decimal units shown

**Implications:**
- The claimed OOS signal (the paper's secondary evidence tier) may not be robust to scaling conventions
- This violates the stated principle (line 160) that "the primary contribution (in-sample finding, structural break, VIX validation) is scale-invariant"
- Yet the OOS Elevated signal is presented as confirmatory evidence without scale robustness testing
- The 14 observations with discordant assignments (953-836=117 net difference, not simple counting but overlap) could drive the signal

**Explanation Needed:**
Why is the decimal-unit OOS Granger test (n=836) not reported? If F-p remains ~0.003, robustness is confirmed; if not, selectivity is confirmed.

**Rating:** **CRITICAL**

**Proposed Fix:**
Add Table: "OOS Elevated HML→SMB Under Decimal-Unit Scaling (n=836): [F, HAC-p, Bootstrap-p, ΔR²]." If test cannot be shown for decimal units, acknowledge this as a limitation and downgrade OOS signal from "secondary validation" to "exploratory artifact of scaling convention." Alternatively, justify percentage-unit convention over decimal units *a priori* (economics-based).

---

### ISSUE 3: ASYMMETRIC NOMENCLATURE FOR STRUCTURAL BREAK vs. DECAY [MEDIUM]

**Location:**
- **Title (line 22):** "Structural **Decay** of Cross-Factor Predictability"
- **Abstract line 31:** "Cross-factor predictive relationships can **structurally break** down"
- **Line 88 (Introduction):** "This paper documents **structural decay** of cross-factor predictability"
- **Lines 91-92:** "with a **structural break** at June 1998...and null predictability post-2008"
- **Lines 278-290 (Results):** "The **Quandt-Andrews sup-$F$ identifies June 1998 as the primary break**...Together, the evidence supports **gradual erosion**"

**Contradiction:**
The terms **structural break**, **structural decay**, and **gradual erosion** are used inconsistently for the same phenomenon, creating semantic confusion:
- "Structural break" suggests a discontinuous event (June 1998, Quandt-Andrews sup-F test)
- "Structural decay" and "gradual erosion" suggest continuous decline
- The Chow test at Jan 2008 (line 284) further complicates the narrative: is this a *second* break, confirmation of continued decay, or evidence of multiple regimes?

**Implications:**
- Readers cannot distinguish between (a) one sharp break at June 1998 followed by stability, (b) continuous decay from June 1998 onward, or (c) multiple breaks (1998 + 2008)
- The coefficient evidence (β shifts from -0.189 pre-GFC to +0.010 post-GFC, line 285-286) suggests a 2008 break, not 1998
- The post-2008 coefficient statement (line 287: β=0.012, CI [-0.049, 0.073]) indicates null dynamics post-2008, consistent with a break around 2008, not 1998

**Explanation Needed:**
Was June 1998 the primary break or merely the earliest detected by sup-F? Does the evidence support one break, two breaks, or gradual decay?

**Rating:** **MEDIUM**

**Proposed Fix:**
1. Adopt a consistent terminology: recommend "**Two-break structural decay**: primary break June 1998 (sup-F), secondary break January 2008 (Chow test)."
2. Revise title to: "Two-Break Structural Decay of Cross-Factor Predictability: Regime-Conditional Granger Analysis..."
3. In results (lines 288-290), explicitly state: "The evidence supports **two structural breaks** (June 1998, January 2008), not gradual smooth decay."
4. Clarify: are coefficients/dynamics identical in the 1998-2008 interval, or does the Elevated regime show decay during this window?

---

### ISSUE 4: CONFLICTING CLAIMS ON NONLINEARITY (LOW-MEDIUM SEVERITY)

**Location:**
- **Lines 370-374** (primary fit, seed 28): "finds no nonlinear improvement for forward HML$\to$SMB under the primary fit (all $p > 0.13$)...**LSTM attention concentrates 68.2% on lag~1 in Normal**"
- **Lines 377-382** (seed 42, Cluster 5, sensitivity fit): "Under an alternative fit (Cluster~5, seed~42, highest-LL achieving 90\% GFC detection, $\Delta\text{BIC} = 218$), **RF shows significant nonlinear improvement** ($p = 0.010$ Elevated, $p = 0.005$ Crisis). The ``purely linear'' characterization is **fit-dependent**"
- **Line 711** (Scope/Limitations): "The ``purely linear'' characterization is fit-dependent (seed~42: RF $p = 0.010$ Elevated)."

**Contradiction:**
The paper claims:
1. Primary finding (lines 370-374): Normal-regime HML→SMB is **linear** (all nonlinear p > 0.13)
2. But then admits (lines 377-382): Seed 42 shows **nonlinear effects exist** (p=0.010 Elevated, p=0.005 Crisis)
3. Yet this is framed as "sensitivity," not a *contradiction* of the primary result

**The Issue:**
- Table 1 explicitly uses **seed 28, Cluster 1 (BIC-optimal)** for the primary result
- Cluster 5 (seed 42) is the "**economically valid**" alternative fit (line 632), achieving 90% GFC detection vs. 0% for Cluster 1
- If Cluster 5 shows strong nonlinear effects (RF p=0.010, 0.005) and the BIC gap is only ΔBIC=218, why privilege the purely-linear result (seed 28)?
- The paper claims (lines 95-102) Tiers 1-2 are "primary" and "confirmatory" but does not explicitly address: **If Cluster 5 is economically motivated and shows nonlinearity, should the "primary contribution" be updated?**

**Implications:**
- The published primary claim (linear forward, nonlinear reverse) may be model-selection driven, not robust
- Practitioners choosing seed 42 would get opposite conclusions about linearity
- Table 1 (tab:optima) shows all 7 clusters have IS Normal p < 10^-7, so fit-swapping does not destroy the core Granger finding—but the **linearity characterization reverses**

**Explanation Needed:**
Why is BIC-optimality (ΔBIC=0 for Cluster 1, ΔBIC=218 for Cluster 5) the decisive criterion for "primary" when GFC detection (0% vs. 90%) might be more economically relevant?

**Rating:** **LOW-MEDIUM** (The core Granger result is robust; only the linearity characterization is fit-dependent. This is disclosed but somewhat buried.)

**Proposed Fix:**
Revise lines 370-382 to emphasize: "The **forward HML→SMB relationship is robustly Granger-significant across all fits (Table 1). However, whether this relationship is linear or nonlinear depends on HMM model selection: seed 28 (BIC-optimal) detects no nonlinear improvement (p > 0.13); seed 42 (economically valid, 90% GFC detection) shows RF nonlinearity (p=0.010 Elevated). This suggests nonlinearity is either absent or present but model-dependent; we recommend future work uses ensemble HMM specifications.**"

---

## SUMMARY TABLE OF ISSUES

| Issue | Dimension | Lines | Severity | Type |
|-------|-----------|-------|----------|------|
| 1 | Sample sizes (n values) | 271, 420 | CRITICAL | Quantile Granger n=2,485 vs. Granger n=3,140 unexplained |
| 2 | Sample sizes (OOS) | 158-159, 484, Abstract 47 | CRITICAL | OOS result only shown under percentage-unit scaling; decimal units (n=836) not reported |
| 3 | Terminology consistency | Title 22, Abstract 31, Lines 88, 278-290 | MEDIUM | "Break" vs. "decay" vs. "erosion" inconsistently applied; unclear if one or two breaks |
| 4 | Nonlinearity characterization | Lines 370-382, 711 | LOW-MEDIUM | Linearity claim fit-dependent; seed 42 contradicts seed 28 on RF nonlinearity |

---

## PASSED CONSISTENCY CHECKS

### ✓ Statistic Consistency (p-values, F-stats)

All reported p-values are **consistent across locations**:
- **8.75 × 10⁻⁹**: Lines 36, 105, 251, 492, 730 — **identical in abstract, intro, table, results, conclusion**
- **1.23 × 10⁻¹³**: Lines 40, 92, 279, 731 — **identical (June 1998 break)**
- **2.29 × 10⁻⁶**: Lines 284, 732 — **identical (January 2008 Chow test)**
- **0.001** (Wald p for SMB→HML tail dependence): Lines 45, 744 — **identical**

✓ **PASS: Core statistics are consistently reported.**

---

### ✓ Bonferroni Thresholds

Bonferroni-corrected thresholds are applied **correctly and consistently**:
- **α/30 = 0.00033** (30 pairs, in-sample): Lines 186, 244, 475, 497 — **consistent**
- **α/3 = 0.0167** (3 regimes, OOS): Lines 187, 498 — **consistent**
- **α/12 = 0.0042** (4 regions × 3 regimes, international): Lines 188, 555 — **consistent**

✓ **PASS: All Bonferroni thresholds applied uniformly.**

---

### ✓ OOS Labeled as Exploratory Consistently

The paper **uniformly** designates OOS results as "exploratory" or "Tier 3":
- Abstract (line 46): "exploratory Elevated-regime signal"
- Lines 99: "(3)~\emph{exploratory} (HML$\to$SMB frozen OOS, honestly fragile)"
- Line 508: "Tier~3 \emph{exploratory only}"
- Line 586: "frozen OOS signal is exploratory"
- Line 750: "HML$\to$SMB frozen OOS is exploratory"

✓ **PASS: OOS never upgraded to primary/confirmatory; always exploratory/Tier 3.**

---

### ✓ "Diagnostic Not Tradable" Maintained

The paper consistently states the contribution is **diagnostic, not alpha-generative**:
- Line 117: "the contribution is diagnostic, not tradable alpha"
- Line 659: "do not generate trading profits (Sharpe ratio $= -0.07$)"
- Line 664: "the regime-conditional framework thus excels at informing practitioners \emph{when} to revisit historically calibrated cross-factor covariance structures—a diagnostic task"
- Line 719-720: "findings are diagnostic (supporting model recalibration during regime shifts) rather than alpha-generative"

✓ **PASS: Disclaimer maintained consistently.**

---

### ✓ Seed 28 vs. Seed 42 Clearly Distinguished

- **Seed 28** (BIC-optimal, primary): Lines 174, 243, 351, 645 — **labeled as "primary fit"**
- **Seed 42** (economically valid, Cluster 5): Lines 377-382, 711 — **labeled as "alternative fit," "sensitivity caveat," explicitly distinguished from primary**

✓ **PASS: Seeds clearly segregated into primary (28) vs. sensitivity (42) categories.**

---

### ✓ ΔR² Consistency

Effect size ΔR² is **consistently reported**:
- Pre-2008 Normal: **2.06%** (lines 271, 730) — **consistent across text and conclusion**
- Overall pre-GFC: **≈2%** (lines 116, 657) — **consistent order-of-magnitude**
- Post-2008: **<0.01%** (line 272) — **null, consistent with narrative**

✓ **PASS: Effect sizes internally consistent.**

---

### ✓ 16-Year Null Consistency

Post-2008 consistency with zero is **uniformly stated as 16 years**:
- Lines 41, 288, 733 — **all cite "16 years" in span 1990-2024 through 2008-2024**

✓ **PASS: Temporal claim consistent.**

---

### ✓ Cross-References and Citations

Spot-check of `\ref{}` and `\cite{}` commands:
- `\ref{tab:main}` (line 262): ✓ resolves to Table 1 (line 241)
- `\ref{tab:optima}` (lines 179, 328, 629): ✓ resolves to Table 5 (line 635)
- `\cite{andrews1991heteroskedasticity}` (line 185): ✓ standard Andrews HAC reference
- `\ref{alg:protocol}` (line 760): ✓ resolves to Algorithm 1 (line 139)

✓ **PASS: All spot-checked references resolve correctly.**

---

### ✓ Tier 1/2/3 Labeling Consistency

Tier classification is applied uniformly:
- **Tier 1** (primary): In-sample Normal-regime break, VIX validation
- **Tier 2** (confirmatory): MOM→SMB OOS, international results
- **Tier 3** (exploratory): HML→SMB frozen OOS

All instances correctly use these labels (lines 95-101, 99, 508, 586).

✓ **PASS: Tiers consistently applied.**

---

### ✓ Crisis Regime Detection Consistency

GFC detection rates in Table 5 (tab:optima):
- Cluster 1 (primary, seed 28): **0% GFC**, ΔBIC=0
- Cluster 5 (economic, seed 42): **90% GFC**, ΔBIC=218

These are cited consistently at lines 176-179 and Table 5.

✓ **PASS: GFC detection rates match throughout.**

---

## UNRESOLVED MINOR ISSUES (INFORMATIONAL)

### 1. Coefficient Sign Reversal (Not a Contradiction, but Noteworthy)

**Lines 285-286:** "β̂ shifts from **-0.189** (pre-GFC) to **+0.010** (post-GFC)"

This sign flip is **genuinely interesting** but the paper does not theorize why SMB should be **negatively** predicted by HML pre-crisis and **null** post-crisis, rather than positively predicted throughout. The economic mechanism discussion (lines 667-675) hypothesizes deleveraging but does not explain the negative coefficient. This is a gap, not an inconsistency, but could confuse readers.

**Recommendation:** Add sentence clarifying: "The negative pre-GFC coefficient suggests that HML stress *reduces* subsequent SMB returns (potentially due to margin constraints on value managers triggering size-premium unwinding); post-GFC, this channel collapses."

---

### 2. HAC Bandwidth Reporting

Table 2 (tab:bandwidth, lines 513-527) shows OOS Elevated p-value crosses 0.05 at B=6 (NW default). The text (line 501) notes "HAC $p$ crosses 0.05 at NW default" but does not state whether the primary results (lines 498, 504) use B=2 (Andrews auto) or NW default.

**Recommendation:** Explicitly state in main results: "Primary OOS results use Andrews auto-bandwidth (B=2), yielding p=0.043; under NW default (B=6), p=0.056, crossing 0.05."

---

## SUMMARY OF RECOMMENDATIONS

### Tier 1: MUST FIX (Credibility Risk)
1. **Quantile Granger n**: Explain the 655-observation gap between Granger n=3,140 and Quantile Granger n=2,485, or provide corrected statistics.
2. **OOS Sample Size Robustness**: Report OOS Granger results under decimal-unit scaling (n=836) to verify scale-invariance claim.

### Tier 2: SHOULD FIX (Clarity Risk)
3. **Structural Break Nomenclature**: Revise to consistently use "two-break structural decay" (June 1998 + January 2008) or clarify the single-break interpretation with evidence.
4. **Nonlinearity Characterization**: Emphasize that linearity claim is seed-28–dependent; seed 42 contradicts it.

### Tier 3: NICE TO FIX (Pedagogical)
5. Add coefficient sign-flip interpretation (negative pre-GFC, null post-GFC).
6. Explicitly report primary HAC bandwidth choice vs. sensitivity.

---

## CONCLUSION

The paper is **internally coherent on core statistics** (p-values, Bonferroni thresholds, effect sizes) but contains **two critical sample-size inconsistencies** that undermine reader ability to verify claims. The **medium-severity terminology issue** conflates "breaks" and "decay" without clear resolution. The **low-severity nonlinearity issue** is disclosed but somewhat buried.

**Overall Assessment:** Paper passes 7/10 consistency checks cleanly. The 3 issues (CRITICAL ×2, MEDIUM ×1) are correctible with transparent additions and do not invalidate the core primary finding (Tier 1: in-sample Normal-regime break), but they do affect secondary evidence and mechanistic claims.

---

**File created:** /sessions/sweet-dazzling-heisenberg/mnt/causal_regimes/r2_consistency.md
