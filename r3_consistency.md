# Internal Consistency Review: ICAIF 2026 Submission
## Hostile Academic Review

---

## CRITICAL ISSUES

### 1. **Quandt-Andrews $p$-value Discrepancy**
**Severity: CRITICAL**

- **Abstract** (line 40): States Quandt-Andrews sup-$F$ identifies June 1998 with "$p = 1.23 \times 10^{-13}$"
- **Main text** (line 281): Same claim: "supremum $F = 21.2$, $p = 1.23 \times 10^{-13}$"
- **Conclusion** (line 738): Repeats "$p = 1.23 \times 10^{-13}$"

**Issue**: This p-value is *extraordinarily* small (essentially zero at machine precision). For a test with ~8,817 trading days of data, a sup-F statistic of 21.2 would typically yield $p \approx 10^{-5}$ to $10^{-6}$, not $10^{-13}$. This is either:
- A transcription error (likely off by 5-8 orders of magnitude), or
- An undisclosed multiple-comparison correction or Monte Carlo simulation detail is being applied

**Files affected**: Abstract, Results (§The Structural Break), Conclusion
**Recommendation**: Verify the actual Quandt-Andrews $p$-value. If correct, explain the calculation method. If incorrect, fix throughout (3 instances).

---

### 2. **Sample Size Mismatch in Quantile Granger Table**
**Severity: CRITICAL**

- **Table 4 caption** (lines 424-426): States "$n = 2{,}485$; pre-2008 Normal subsample after lag-9 exclusion and quantile-boundary trimming at $\tau \in \{0.05, 0.95\}$"
- **Granger in-sample findings** (lines 273-274):
  - Pre-2008 Normal: $n = 3{,}140$
  - Post-2008 Normal: $n = 1{,}557$
  - Sum: $4{,}697$ total Normal observations after lag exclusion

**Issue**: Table 4 reports $n = 2{,}485$ for "pre-2008 Normal subsample," but this does not match:
- The stated pre-2008 Normal $n = 3{,}140$ from the main analysis
- Expected subsample size with additional lag-9 and quantile-boundary trimming

A loss from 3,140 to 2,485 is only 17.5%. The footnote justifies lag-1 boundary loss (26 observations), but the additional 629-observation gap from lag-9 + quantile trimming is not explained.

**Files affected**: Table 4 (Quantile Granger results)
**Recommendation**: Either (a) report the lag-9 sample size separately, or (b) explain why quantile trimming removes 629 observations.

---

### 3. **LSTM Permutation Test Underpowered but Not Disclosed in Main Results**
**Severity: CRITICAL**

- **Section on Complexity Characterization** (lines 329-334): Claims LSTM results show no nonlinear improvement
- **Methodology** (lines 149-150): Algorithm specifies permutation tests for complexity diagnostic
- **Table 3 footnote** (line 354): *Buried in table caption*: "200 shuffles for RF/MLP, 100 for LSTM"
- **Limitations** (lines 729-731): *Only in Discussion*: "The LSTM permutation test uses 100 shuffles (vs. 200 for RF/MLP), adequate for a null result but underpowered to detect small nonlinear effects; future work should increase to ≥500."

**Issue**:
1. The claim that "no nonlinear improvement" (line 373) is based on LSTM with $n_{\text{perm}} = 100$ shuffles, which the authors themselves admit is "underpowered to detect small nonlinear effects"
2. The **main results section makes the claim without qualification**; the limitation is buried in Discussion
3. This is particularly problematic because the linear--nonlinear boundary is a stated contribution (lines 109-112)

**Files affected**: Results section (complexity subsection), Table 3, Discussion (Limitations)
**Recommendation**:
- Move the underpowered caveat to the Results section alongside Table 3
- Either re-run LSTM with 200 shuffles (matching RF/MLP) or downgrade the "no nonlinear improvement" claim to "inconclusive for LSTM"

---

## MEDIUM ISSUES

### 4. **Bonferroni Correction Inconsistency Across Sections**
**Severity: MEDIUM**

- **Methodology** (line 188): "Bonferroni $\alpha_{\text{fam}} = 0.01$ across 30 directed pairs ($\alpha/30 = 0.00033$)"
- **OOS section** (line 189): "corrected per regime ($\alpha/3 = 0.0167$) or per region-by-regime combination ($\alpha/12 = 0.0042$)"
- **Frozen OOS results** (line 504): "does not survive 30-pair Bonferroni ($\alpha/30 = 0.00033$)"
- **Same paragraph** (line 505): "does not survive 3-regime Bonferroni ($\alpha/3 = 0.0167$; HAC $p = 0.043$)"

**Issue**: The paper applies *three different Bonferroni thresholds* to the same data:
1. $\alpha/30 = 0.00033$ (30 pairs × 1 regime)
2. $\alpha/3 = 0.0167$ (1 pair × 3 regimes)
3. $\alpha/12 = 0.0042$ (4 regions × 3 regimes, lines 190, 562)

The OOS Elevated result ($p = 0.003$ unadjusted, $p = 0.043$ HAC) survives threshold #3 ($\alpha/12 = 0.0042$) but not #1 or #2. The paper claims it "does not survive" Bonferroni but then reports it is significant under $\alpha/12$, creating ambiguity.

**Files affected**: Lines 188-191, 504-505, 562-563
**Recommendation**:
- Clearly state the hierarchy of correction applied to OOS results (should be $\alpha/12$ for region×regime, $\alpha/30$ as a sensitivity check)
- Table 5 (International Replication) applies $\alpha/12$ and correctly identifies 2/4 regions surviving this threshold; this should be explicitly cross-referenced in the OOS section

---

### 5. **Inconsistent Sample Size Notation in Table 3 (Neural Models)**
**Severity: MEDIUM**

- **Table 3 caption** (lines 351-356): "Sample sizes reflect lag-9 input window and train/validation split ($n_{\text{eff}} < n_{\text{regime}}$)"
- **Table 3 numbers**:
  - Normal: $n = 4{,}496$ (but regime total is 4,723)
  - Elevated: $n = 2{,}792$ (but regime total is 3,023)
  - Crisis: $n = 1{,}017$ (but regime total is 1,071)

**Issue**: The losses are:
- Normal: $4{,}723 - 4{,}496 = 227$ (4.8%)
- Elevated: $3{,}023 - 2{,}792 = 231$ (7.6%)
- Crisis: $1{,}071 - 1{,}017 = 54$ (5.0%)

These are slightly larger than the lag-1 boundary effect (footnote 275-277 reports 26 days). Lag-9 would lose more days, but the table doesn't explain this discrepancy. The caption says "lag-9 input window" but doesn't explain if additional train/validation split removes data (e.g., 80/20 split would halve sample, but the losses are much smaller).

**Files affected**: Table 3
**Recommendation**: Clarify whether the effective sample includes validation-set exclusion or only the training set; if so, report both.

---

### 6. **Transfer Entropy Terminology Shift**
**Severity: MEDIUM**

- **Abstract** (line 43): "Transfer entropy additionally reveals a stronger nonlinear reverse channel SMB→HML ($z = 5.37$ vs. forward $z = 2.45$)"
- **Results** (line 415): "Transfer entropy (Table~\ref{tab:te}) reveals the reverse channel SMB→HML is substantially stronger in Normal ($z = 5.37$ vs. forward $z = 2.45$); both collapse in Crisis."
- **Discussion/Conclusion** (line 749): "Transfer entropy + quantile Granger reveal a directional asymmetry (linear forward, nonlinear reverse via tail dependence, Wald $p = 0.001$)"

**Issue**:
- The Abstract calls TE findings "nonlinear"
- The Results section presents TE as evidence of a "reverse channel" without claiming nonlinearity
- The Conclusion lumps TE findings with quantile Granger (tail dependence) to support "nonlinear reverse"

However, transfer entropy measures *information flow* (mutual information), not nonlinearity per se. The reverse SMB→HML signal in TE is strong, but TE doesn't distinguish between linear and nonlinear information flow. The nonlinearity claim rests entirely on the quantile Granger result (Wald $p = 0.001$, line 435: $\hat{\beta}_{0.95} = 0.212$ vs. median $= -0.026$), not on TE itself.

**Files affected**: Abstract, Results (Complexity subsection), Conclusion
**Recommendation**:
- Clarify in Abstract that TE reveals information-theoretic asymmetry, not necessarily nonlinearity
- Emphasize that the *nonlinear* mechanism (tail dependence) is identified by quantile Granger, not TE alone

---

### 7. **Discrepancy in Elevated Regime OOS Sample Size**
**Severity: MEDIUM**

- **Frozen OOS section** (lines 160-161): "Under percentage units, the frozen OOS yields $n = 953$ Elevated-regime days; decimal units yield $n = 836$ (agreement 86.3%)."
- **Table 6** (line 491): OOS Elevated reports $n = 953$

**Consistency**: ✓ The value matches

**New Issue Identified**:
- **Earlier in text** (lines 500-503): "post-GFC markets spend more time in higher-volatility states---the frozen classifier assigns formerly Normal observations to Elevated (Elevated share doubles from 13.7\% training to 33.7\% test)."
- **Calculation check**: Test period is 2013--2024 (roughly 12 years = ~3,000 trading days)
  - 33.7% of ~3,000 ≈ 1,011 days
  - But $n = 953$ is reported in Table 6

**Issue**: The prevalence ratio (33.7% of test data) implies ~1,011 Elevated days, not 953. Either:
1. The test period is slightly shorter than 3,000 days, or
2. The 33.7% prevalence figure is inaccurate

The discrepancy is ~5.8%, which could reflect missing data or a slightly shorter OOS period.

**Files affected**: Lines 500-503, Table 6
**Recommendation**: Clarify the OOS period length (should be stated explicitly, e.g., "2013--2024, 3,019 trading days") and verify the 33.7% figure against actual counts.

---

### 8. **MOM→SMB Validation Claims vs. Tier Assignments**
**Severity: MEDIUM**

- **Tier assignment** (lines 99-102): MOM$\to$SMB OOS replication is listed as **Tier 2 (confirmatory)**
- **MOM→SMB section** (lines 537-554):
  - "frozen OOS Normal $F = 130.6$ ($p < 10^{-28}$)---near-perfect replication ($\Delta F < 0.1\%$)"
  - **However**, the in-sample Normal is also extraordinarily strong: $F = 130.7$ ($p < 10^{-28}$)
  - The text claims this "proves the protocol detects genuine OOS confirmation for sufficiently strong signals"

**Issue**:
- If in-sample $F = 130.7$ and OOS $F = 130.6$ are nearly identical, the OOS result is *expected*, not surprising
- A finding this strong suggests potential **look-ahead bias or data leakage** (e.g., has the test period data influenced the regime classification even though the HMM is "frozen"?)
- Alternatively, the phenomenon is so robust that in-sample ≈ OOS is expected; but this should be discussed

The paper treats MOM→SMB as strong confirmatory evidence, but doesn't address why it is so perfectly replicated (perhaps too perfectly?). The conclusion (line 759) claims "textbook replication ($\Delta F < 0.1\%$)" as validation, but this level of replication is suspicious if based on a frozen model that has any distributional mismatch between training and test.

**Files affected**: Lines 99, 537-554, 759
**Recommendation**:
- Investigate whether frozen regime classification suffers distributional shift (e.g., test Elevated/Crisis prevalence against training)
- Acknowledge that MOM→SMB's perfect OOS replication may reflect an exceptionally robust linear relationship, not protocol validity per se
- Consider running a permutation test on MOM→SMB to rule out overfitting

---

### 9. **Inconsistent Lag Specification**
**Severity: MEDIUM**

- **Table 3 (Complexity, Figure~\ref{fig:complexity} caption, line 322)**: "Granger $p$-values for HML$\to$SMB across lags 1--15 by regime"
- **Table 3** (complexity diagnostic): Uses lag-9 input window implicitly (line 356: "lag-9 input window")
- **Granger results** (main Table 2): All results report lag-1 Granger tests
- **Discussion of robustness** (line 326): "lags 1--15"

**Issue**:
- The Granger $p$-values in Table 2 are lag-1 only
- Figure 3 (lag sensitivity) shows lags 1--15 are all significant in Normal
- The complexity diagnostic (Table 3, neural models) uses a lag-9 input window for LSTM/MLP, but the Granger test itself still reports lag-1 coefficients

There is no explicit claim of inconsistency, but the paper could be clearer about whether:
1. Main results are lag-1 only, or
2. Results are robust across lags 1--15 (shown in Figure 3 implicitly)

**Files affected**: Tables 2, 3; Figures in lag sensitivity
**Recommendation**:
- Explicitly state "lag-1 Granger tests reported in Table 2; Figure 3 confirms significance across lags 1--15"
- Clarify that complexity diagnostic uses lag-9 for neural models to match the information window

---

## LOW ISSUES

### 10. **Forward Reference to Table 5 Before Introduction**
**Severity: LOW**

- **Introduction, line 181**: "see Table~\ref{tab:optima}"
- **Table 5 label** (line 642): "Table 5: Local Optima..."
- **But line 211**: Forward reference in methodology is to "Table~\ref{tab:regimes}"

**Issue**: In line 181, the text references Table 5 (Local Optima) before Table 1 (Regime Summary) has been shown. The table ordering is:
1. Table 1: Regime Summary (line 214)
2. Table 2: Granger Results (line 243)
3. Table 3: Complexity Diagnostic (line 350)
4. Table 4: Quantile Granger (line 422)
5. Table 5: Frozen OOS (line 479)
6. Table 6: HAC Bandwidth (line 518)
7. Table 7: International Replication (line 567)
8. Table 8: Regime-Heterogeneous Pairs (line 615)
9. Table 9: Local Optima (line 642)

The forward reference in line 181 to "see Table~\ref{tab:optima}" appears in Student-$t$ HMM description and references Table 9, which appears 13 pages later. This is acceptable formatting (forward references are common), but slightly unusual.

**Files affected**: Line 181
**Recommendation**: No action needed; this is a minor formatting preference. Alternatively, move the Local Optima table earlier if robustness across 7 clusters is a primary contribution.

---

### 11. **Missing Explicit Statement of OOS Period in Table 6**
**Severity: LOW**

- **Table 6 caption** (line 520): "HAC Bandwidth Sensitivity: OOS Elevated HML$\to$SMB"
- **Table 6 itself** reports HAC $p$-values for bandwidths $B \in \{1, 2, 4, 6, 10\}$

**Issue**: The table header does not explicitly state the OOS period (2013--2024) or that this is from Table 5's Elevated regime. A reader might assume this is in-sample.

**Files affected**: Table 6
**Recommendation**: Update caption to "HAC Bandwidth Sensitivity: OOS Elevated HML$\to$SMB (2013--2024)" or reference Table 5 explicitly.

---

### 12. **Permutation Test Details Inconsistent**
**Severity: LOW**

- **Circularity mitigation** (line 198): "Permutation test: 50,000 label shuffles within regime ($p = 0.022$)"
- **Complexity diagnostic** (Table 3, line 354): "200 shuffles for RF/MLP, 100 for LSTM"

**Issue**: Two different permutation tests are described:
1. The regime-label shuffling (50,000 shuffles, line 198, reported as $p = 0.022$)
2. The complexity diagnostic shuffles (100-200 shuffles per Table 3)

These serve different purposes (testing for circularity vs. testing nonlinear improvement), so inconsistency is acceptable. However, the paper could be clearer about which test is which.

**Files affected**: Lines 198, 354
**Recommendation**: No action strictly required, but could clarify "permutation test for circularity" vs. "permutation test for nonlinear improvement" in the text.

---

### 13. **Minor Notation Inconsistency: Regime Labels**
**Severity: LOW**

- **Main text and tables**: Regimes labeled "Normal," "Elevated," "Crisis"
- **Some figure captions** (e.g., line 236): Uses same labels
- **Conclusion** (lines 740-741): Refers to "all 7 HMM local-optima clusters" but doesn't specify if these are always labeled consistently

**Issue**: The regime names are economic labels assigned *after* estimation. The BIC-optimal fit assigns 0% of GFC to "Crisis," so "Crisis" is a statistical cluster, not an economic one. Alternative fits assign 90% of GFC to "Crisis." The paper acknowledges this (lines 178-180) but doesn't always make clear which fit is being discussed.

**Files affected**: Throughout
**Recommendation**:
- Always specify "BIC-optimal fit (seed 28)" when reporting results
- When discussing sensitivity fits, explicitly name the cluster (e.g., "Cluster 5, 90% GFC detection")
- This is largely done correctly, but a few instances could be clearer (e.g., line 741 doesn't specify which regime definition)

---

### 14. **Vague Reference to "Best Case" HAC Robustness**
**Severity: LOW**

- **HAC robustness footnote** (lines 268-272):
  - Lists HAC $p$-values across bandwidths and kernels
  - States "All 90 kernel--bandwidth combinations yield $p < 10^{-7}$"
  - But the main text says "worst case at Quadratic Spectral $B = 30$" yields $p = 8.8 \times 10^{-8}$

**Issue**:
- If worst case is $8.8 \times 10^{-8}$, then all 90 combinations yield $p < 10^{-7}$ is correct
- However, saying "range: $[3.2 \times 10^{-9}, 8.8 \times 10^{-8}]$" and "worst case at Quadratic Spectral $B = 30$" is slightly misleading—it suggests uncertainty, when in fact the entire range is $< 10^{-7}$
- This is minor; the claim is correct but could be stated more directly

**Files affected**: Lines 266-272
**Recommendation**: State more directly: "All 90 HAC specifications yield $p < 10^{-7}$, range $[3.2 \times 10^{-9}, 8.8 \times 10^{-8}]$, confirming robust significance."

---

### 15. **Tier Terminology Used Inconsistently**
**Severity: LOW**

- **Introduction, evidence hierarchy** (lines 96-102): Clearly defines Tiers 1-3
- **Frozen OOS subsection** (line 475): "Tier 3 (exploratory)"
- **Discussion** (line 593): "frozen OOS signal is exploratory (Tier 3)"
- **Conclusion** (line 757-758): "OOS evidence: The HML$\to$SMB frozen OOS is exploratory (regime-redistributed, Bonferroni-nonsignificant, bootstrap $p = 0.153$)"

**Consistency**: ✓ Mostly consistent, Tier 3 is always labeled "exploratory"

**Minor note**: Tier 1 and 2 are not always explicitly invoked in later sections (e.g., the conclusion doesn't say "Primary finding [Tier 1]"). This is acceptable but could be slightly clearer.

**Files affected**: Conclusion (lines 735-762)
**Recommendation**: Optional: Add brief tier labels to conclusion subsections (e.g., "[Tier 1]" after line 735) for clarity.

---

## SUMMARY

| Severity | Count | Issues |
|----------|-------|--------|
| **CRITICAL** | 3 | Quandt-Andrews $p$-value (discrepancy), Quantile sample size, LSTM permutation underpowered |
| **MEDIUM** | 6 | Bonferroni threshold inconsistency, Table 3 sample size, TE terminology, OOS prevalence, MOM→SMB perfect replication, Lag specification |
| **LOW** | 6 | Forward reference, Missing OOS period label, Permutation test naming, Regime label notation, HAC phrasing, Tier labeling |

---

## VERDICT

**CONVERGED WITH RESERVATIONS**

The paper is internally consistent on its major claims, but three critical issues require resolution:

1. **Quandt-Andrews $p$-value** must be verified and corrected (likely a transcription error)
2. **Quantile sample size** ($n = 2{,}485$) must be explained or corrected
3. **LSTM permutation test** caveat must be moved to Results section

Medium-severity issues should be addressed via revision, particularly the Bonferroni threshold clarification (OOS section is currently ambiguous) and the perfect MOM→SMB replication (warrants discussion of potential look-ahead bias or confirmation bias).

If these three critical items are corrected, the paper demonstrates strong internal consistency.

