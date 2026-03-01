# Consistency Check Report: main_icaif.tex

## Summary
**FAIL** — Found 1 significant inconsistency and 1 notation concern that should be resolved.

---

## Check 1: Repeated Numbers Match
**PASS** ✓

All critical numbers that appear multiple times are consistent:
- $p = 8.75 \times 10^{-9}$ (Normal regime HML→SMB): Lines 36, 105, 251, 492, 730 — all match
- $p = 1.23 \times 10^{-13}$ (structural break June 1998): Lines 40, 92, 279, 731 — all match
- $p = 2.29 \times 10^{-6}$ (Chow test): Lines 284, 732 — all match
- CI $[-0.049, 0.073]$ (post-2008): Lines 106, 288, 734 — all match
- $n = 953$ (OOS Elevated regime): Lines 158, 484 — all match
- $\Delta F = 0.1\%$ (MOM→SMB replication): Lines 50, 535, 752 — all match
- $\Delta R^2 = 2.06\%$ (pre-2008 Normal): Lines 271, 730 — all match
- 7 local-optima clusters: Lines 38, 114, 327, 629, 734 — all match
- 19/30 regime-heterogeneous pairs (63%): Lines 106-107, 603 — all match
- 16 years consistent with zero: Lines 41, 288, 733 — all match
- $F = 21.2$ (Quandt-Andrews): Line 279 only (no duplicates)

---

## Check 2: P-values in Abstract vs. Table
**FAIL** ✗

**CRITICAL INCONSISTENCY FOUND:**

**Abstract (line 36):**
- HML→SMB Normal: "$p = 8.75 \times 10^{-9}$" ✓ (matches Table 1, line 251)

**Abstract (lines 40, 52):**
- Structural break: "$p = 1.23 \times 10^{-13}$" ✓ (matches Results section, line 279)

**Abstract (line 47):**
- OOS Elevated-regime signal: "$F$-$p = 0.003$"
- Table (line 484): "$F$-$p = 0.003$" ✓ **MATCH**
- But Table (line 484) also shows: "HAC-$p = 0.043$" (not mentioned in abstract)
- Abstract correctly caveats this as "exploratory" and "does not survive Bonferroni correction"

---

## Check 3: International P-value Notation Inconsistency
**FAIL** ✗

**NOTATION MISMATCH BETWEEN TEXT AND TABLE:**

Line 553 (text): "Asia-Pacific ex Japan (Crisis OOS $F = 39.39$, **$p < 0.0001$**)"
- Table line 574: "Crisis OOS & 39.39 & **$<$0.001**"
- **Inconsistent:** $< 0.0001$ ≠ $< 0.001$

Line 554 (text): "Developed ex US (Crisis OOS $F = 15.85$, **$p = 0.0001$**)"
- Table line 571: "Crisis OOS & 15.85 & **$<$0.001**"
- **Inconsistent:** $= 0.0001$ ≠ $< 0.001$; also different notation (= vs. <)

**Resolution needed:** Text uses $< 0.0001$ and $= 0.0001$; Table uses $< 0.001$ for both.
Pick one notation convention and apply consistently.

---

## Check 4: F-Statistics Text vs. Table
**PASS** ✓

- Quandt-Andrews: $F = 21.2$ (line 279) — no table to cross-check, but used only once
- Chow test: $F(3,n-6) = 9.68$ (line 284) — no table conflict
- VIX Normal: $F = 18.6$ (line 304) — text only, no conflict
- VIX post-2008: $F = 0.13$ (line 305) — text only, no conflict
- MOM→SMB in-sample Normal: $F = 130.7$ (line 533) — text only, no conflict
- MOM→SMB in-sample Crisis: $F = 29.8$ (line 534) — text only, no conflict
- MOM→SMB OOS Normal: $F = 130.6$ (line 534) — text only, no conflict
- International: $F = 39.39$, $F = 15.85$ — match tables (lines 571, 574)

All F-statistics in text that can be cross-checked with tables are consistent.

---

## Check 5: Sample Sizes Across Tables
**PASS** ✓

**Table 1 (Regime Summary, line 215):**
- Normal: 4,723 days

**Table 2 (Neural Models, line 354):**
- Normal: $n = 4,496$ (with lag-9 input and train/val split) — caption explains reduction

**Table 4 (OOS, line 477):**
- Normal: $n = 724$ (2013–2024, frozen OOS)
- Elevated: $n = 953$ ✓ (matches line 158)
- Crisis: $n = 1,119$

Sample size reductions are explained (lag exclusion, train/val splits, time period). No contradiction.

---

## Check 6: Tier Labels Assignment Consistency
**PASS** ✓

- **Tier 1 (Primary):** In-sample Normal-regime structural break, VIX validated, robust across specs
  - Correctly labeled throughout: lines 96, 100, 585, 714

- **Tier 2 (Confirmatory):** MOM→SMB OOS replication, international results
  - Correctly labeled: lines 98, 100, 585–586

- **Tier 3 (Exploratory):** HML→SMB frozen OOS
  - Correctly labeled: lines 99–100, 161, 382, 466, 508, 586–587, 714, 721, 750–751

No tier misassignments detected. Hierarchical distinction is maintained throughout.

---

## Check 7: "Exploratory" Language Consistency for OOS HML→SMB
**PASS** ✓

Every mention of the OOS HML→SMB result is consistently framed as "exploratory":
- Line 46: "exploratory"
- Line 99: "\emph{exploratory}"
- Line 161: "exploratory OOS result"
- Line 382: "exploratory"
- Line 466: "(Exploratory)" [subheading]
- Line 508: "Tier~3 \emph{exploratory only}"
- Line 586: "exploratory (Tier~3)"
- Line 721: "exploratory OOS signal (Tier~3)"
- Line 750: "exploratory (regime-redistributed, Bonferroni-nonsignificant)"

All consistent. "Exploratory" language is applied uniformly; no mixed signals.

---

## Check 8: Regime Names Used Consistently
**PASS** ✓

Three regime labels used consistently throughout:
- **Normal:** Lines 35, 96, 176, 221, 251, etc. — always "Normal"
- **Elevated:** Lines 47, 158, 222, 253, etc. — always "Elevated"
- **Crisis:** Lines 172, 223, 255, etc. — always "Crisis"

No switching between "Crisis," "High," "Stress," or other variants. Regime names are stable.

---

## Check 9: Bonferroni Thresholds Consistent with Methodology
**PASS** ✓

**Methodology definition (line 186):**
- In-sample: $\alpha_{\text{fam}} = 0.01$ across 30 directed pairs → $\alpha/30 = 0.00033$

**Results section:**
- Line 244 (Table caption): "Bonferroni threshold: $p < 0.00033$" ✓
- Line 258 (Table footnote): "Below 1% but not Bonferroni-significant" (refers to $\alpha = 0.01$, consistent)
- Line 475 (OOS Table caption): "30-pair Bonferroni ($\alpha/30 = 0.00033$)" ✓
- Line 497: "does not survive 30-pair Bonferroni ($\alpha/30 = 0.00033$)" ✓
- Line 498: "does not survive 3-regime Bonferroni ($\alpha/3 = 0.0167$)" ✓ (for OOS subset)
- Line 555: OOS international: "Bonferroni ($\alpha/12 = 0.0042$, correcting for 4 regions $\times$ 3 regimes)" ✓

All Bonferroni thresholds correctly derived and applied.

---

## Check 10: "Structural Decay" (Title) vs. "Structural Break" (Results) Coherence
**PASS** ✓

**Title (line 22):** "Structural Decay of Cross-Factor Predictability"

**Body text:**
- "Structural decay" appears: lines 88, 104, 160, 584 (noun, phenomenon)
- "Structural break" appears: lines 39, 92, 96, 160, 277, 305, etc. (technical finding: June 1998)

**Interpretation:**
"Structural decay" = broad phenomenon of deterioration over time
"Structural break" = specific statistical breakpoint (June 1998)

This distinction is coherent: the structural break (June 1998) is evidence for and a marker of the broader structural decay. Not contradictory; complementary terminology. ✓

---

## FINAL VERDICT

**OVERALL: FAIL**

**Issues Found:**

1. **Critical:** International p-value notation inconsistency (lines 553–554 vs. table 571, 574)
   - Text: "$p < 0.0001$" and "$p = 0.0001$"
   - Table: "$p < 0.001$"
   - **Action:** Standardize notation. Clarify if actual p-values are $< 0.0001$, $= 0.0001$, or $< 0.001$.

2. **Minor:** Notation convention inconsistency in abstract/results
   - Some use "=", others use "<" for small p-values
   - **Action:** Adopt consistent notation across all p-value reporting.

**All other checks pass.** No contradictions in:
- Repeated numbers (check 1)
- Main p-values (check 2)
- F-statistics (check 4)
- Sample sizes (check 5)
- Tier assignments (check 6)
- Exploratory language (check 7)
- Regime names (check 8)
- Bonferroni thresholds (check 9)
- Structural decay vs. break (check 10)

---

## Recommended Actions

1. Resolve lines 553–554 notation. Either:
   - Change text to "$p < 0.001$" for both (matches table), OR
   - Change table to "$p < 0.0001$" and "$p = 0.0001$" (matches text)

2. Adopt a single p-value notation convention (e.g., always use < for sub-threshold, = for exact values).

3. Verify actual numerical p-values for Developed ex-US and Asia-Pac in the underlying analysis to ensure table/text alignment.
