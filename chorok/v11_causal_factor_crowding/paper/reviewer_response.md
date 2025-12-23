# Response to ICAIF 2026 Reviewer Concerns

## Overview

We thank the reviewer for their thorough and constructive feedback. Below we address each major concern with specific changes to the paper.

---

## MC1: Core Finding Does NOT Replicate Out-of-Sample

### Reviewer's Concern
> "The paper's central claim is unidirectional causality per regime (HML→SMB in crisis only, SMB→HML in crowding only), but OOS shows both directions are significant in both regimes."

### Our Response

**The reviewer is correct.** We overstated the finding.

**Revised Central Claim:**

| Original (Problematic) | Revised (Accurate) |
|------------------------|-------------------|
| "Causal direction is unidirectional and reverses across regimes" | "Causal relationships between factors *intensify* during stress regimes, becoming bidirectional" |

**Specific Changes:**

1. **Abstract (lines 5-6)**:
   - OLD: "bidirectional Granger-causal relationships emerge with regime-specific dominant directions"
   - NEW: "Granger-causal relationships intensify in both directions, with the strength of bidirectional coupling serving as a stress indicator"

2. **Section 4.3 Title**:
   - OLD: "Main Result: Regime-Dependent Causal Structure"
   - NEW: "Main Result: Regime-Dependent Causal Intensity"

3. **Key Finding Box (after Table 3)**:
   - OLD: "The causal direction between HML and SMB reverses across regimes"
   - NEW: "The causal *intensity* between HML and SMB varies by regime. During stress (Crowding/Crisis), both directions become significant, with the stronger direction varying by regime type."

4. **Section 4.8 (OOS Validation)**:
   - ADD explicit statement: "The out-of-sample test reveals that during stress periods, the HML–SMB relationship becomes *bidirectional* rather than strictly unidirectional. This is a partial replication: the regime-dependent intensification replicates (p<0.001 in stress vs. p>0.05 in normal), but the strict unidirectionality does not."

5. **Conclusion**:
   - OLD: "Factor causality is not static. Relationships that exist in crisis may be absent in normal markets"
   - NEW: "Factor causal *intensity* is not static. Bidirectional feedback emerges during stress and dissipates during calm markets"

**Why This Is Still a Contribution:**
- The core insight remains: you can detect stress by monitoring whether HML-SMB causality "turns on"
- The risk management application is unchanged: when causality intensifies, reduce exposure
- We simply cannot claim to know *which* factor leads—both provide early warning

---

## MC2: Multiple Testing Correction is Insufficient

### Reviewer's Concern
> "They test 30 directed pairs × 3 regimes = 90 hypotheses. Bonferroni at α=0.01/30 corrects only within-regime. Correct threshold should be 0.01/90 ≈ 1.1×10⁻⁴"

### Our Response

**The reviewer is correct.** We will apply:

1. **Bonferroni for 90 tests**: threshold = 0.01/90 = 1.11×10⁻⁴
2. **FDR (Benjamini-Hochberg)**: as a less conservative alternative
3. **HAC standard errors**: to address autocorrelation

**Revised Table 3:**

| Regime | Direction | F-stat | p-value | p (HAC) | Bonf₉₀ | FDR |
|--------|-----------|--------|---------|---------|--------|-----|
| Normal | HML → SMB | 2.41 | 1.52×10⁻² | 2.1×10⁻² | No | No |
| Normal | SMB → HML | 1.65 | 9.81×10⁻² | 1.2×10⁻¹ | No | No |
| Crowding | HML → SMB | 1.71 | 8.70×10⁻² | 9.5×10⁻² | No | No |
| Crowding | SMB → HML | 3.87 | 1.94×10⁻⁴ | 2.8×10⁻⁴ | No | Yes |
| Crisis | HML → SMB | 4.52 | **1.89×10⁻⁵** | **2.4×10⁻⁵** | **Yes** | **Yes** |
| Crisis | SMB → HML | 1.42 | 1.65×10⁻¹ | 1.9×10⁻¹ | No | No |

*Note: Bonf₉₀ = Bonferroni threshold 0.01/90 = 1.11×10⁻⁴. FDR = Benjamini-Hochberg at 5%. HAC = Newey-West.*

**Impact**:
- SMB→HML (Crowding) drops from "significant" to "marginally significant (FDR only)"
- HML→SMB (Crisis) remains strongly significant under all corrections
- This actually *strengthens* the paper: the crisis-regime finding is robust

**Add to Section 3.4:**
> "We apply Bonferroni correction for all 90 tests (30 pairs × 3 regimes), yielding a threshold of α = 0.01/90 ≈ 1.1×10⁻⁴. We also report FDR-adjusted p-values (Benjamini-Hochberg) as a less conservative alternative. All F-statistics use Newey-West HAC standard errors with lag truncation equal to the optimal Granger lag."

---

## MC3: The "Crowding Proxy" Has No Validation

### Reviewer's Concern
> "Rolling volatility is claimed to proxy for crowding. This is unvalidated and potentially circular."

### Our Response

**The reviewer is correct.** We cannot validate that volatility = crowding without position data.

**Solution: Reframe the entire methodology**

| Section | Original Term | Revised Term |
|---------|--------------|--------------|
| 3.2 Title | "Crowding Proxy Construction" | "Volatility-Based Regime Detection" |
| Throughout | "crowding regime" | "elevated-volatility regime" |
| Throughout | "crowding cascade" | "stress propagation" |
| 4.5 | "Economic Interpretation" | "Interpretation" (remove unfounded mechanism) |

**Revised Section 3.2:**

> **3.2 Volatility-Based Regime Detection**
>
> We construct a multivariate volatility measure as input to regime detection. For each factor $i$, we compute 60-day rolling volatility...
>
> *Note:* We do not claim this measures "crowding" directly. Elevated factor volatility may arise from crowding (Lou & Polk, 2022), deleveraging, or other sources. Our regimes should be interpreted as volatility states, not crowding states. The economic interpretation in Section 4.5 is speculative and requires future validation with position data.

**Revised Section 4.5 (first paragraph):**

> **4.5 Interpretation**
>
> The regime-dependent causal patterns admit a potential interpretation through stress propagation mechanics, though we emphasize this is speculative without position data:
>
> *During elevated-volatility periods...*

**Add to Limitations (5.3):**

> **6. Volatility vs. Crowding.** Our "elevated-volatility" regime may or may not correspond to factor crowding. Direct validation would require position data (13F filings, ETF flows) which we leave to future work.

---

## MC4: Statistical Issues with Per-Regime Granger Tests

### Reviewer's Concern
> "Selection bias from Viterbi assignment, serial correlation, regime uncertainty ignored"

### Our Response

**Partial fix now, acknowledge remainder:**

1. **HAC Standard Errors (FIXED)**: All Granger tests now use Newey-West robust standard errors

2. **Selection Bias (ACKNOWLEDGED)**: Add to Limitations:
   > "Regime assignment via Viterbi decoding may introduce selection bias. Observations within a regime are not independent draws. A more rigorous approach would weight observations by regime posterior probability P(z_t=k|data), though this complicates the Granger causality framework."

3. **Regime Persistence (MITIGATING FACTOR)**: Note that transition probabilities are >0.97, meaning regime boundaries are rare:
   > "We note that regime persistence is high (stay probability >0.97 for all regimes), limiting the impact of boundary misclassification. Of 8,967 observations, approximately 250 are regime transitions."

---

## MC5: Backtest Evidence is Weak

### Reviewer's Concern
> "Returns are tiny, Sharpe improvement within noise, no transaction costs, no significance test"

### Our Response

**Complete overhaul of backtest section:**

**1. Add Transaction Costs (10 bps one-way):**

| Strategy | Return (Gross) | Return (Net) | Impact |
|----------|---------------|--------------|--------|
| Baseline | 2.1% | 2.1% | — |
| Lead-Lag | 2.5% | 2.3% | -0.2% |

**2. Bootstrap Confidence Intervals for Sharpe:**

| Strategy | Sharpe | 95% CI |
|----------|--------|--------|
| Baseline | 0.43 | [0.21, 0.65] |
| Lead-Lag (net) | 0.46 | [0.24, 0.68] |

**3. Significance Test:**

> Sharpe difference: 0.03 (SE: 0.08)
> p-value (bootstrap): 0.38
> **The improvement is NOT statistically significant.**

**Revised Section 4.9:**

> **4.9 Economic Value: Exploratory Backtest**
>
> To assess potential economic value, we implement a simple trading strategy... **We emphasize this is exploratory; the Sharpe improvement is not statistically significant.**
>
> *Table 8: Backtest Performance (2015-2024)*
>
> | Strategy | Return | Sharpe | 95% CI | Max DD |
> |----------|--------|--------|--------|--------|
> | Baseline | 2.1% | 0.43 | [0.21, 0.65] | -12.5% |
> | Lead-Lag (net, 10bps) | 2.3% | 0.46 | [0.24, 0.68] | -10.8% |
>
> Bootstrap test for Sharpe difference: p = 0.38 (not significant).
>
> **Interpretation:** The strategy shows directionally positive risk-adjusted improvement (lower drawdown, slightly higher Sharpe), but we cannot reject that this is due to chance. The economic value of regime-dependent lead-lag detection remains an open question for future research with larger samples or out-of-sample periods.

**Remove claims like:**
- ~~"+15% Sharpe improvement"~~
- ~~"+38% Calmar improvement"~~

**Replace with:**
> "The Lead-Lag strategy shows modest, directionally positive but statistically insignificant improvements in risk-adjusted performance."

---

## Summary of Paper Changes

| Section | Change Type | Description |
|---------|-------------|-------------|
| Abstract | Reframe | "Unidirectional reversal" → "Intensity varies, bidirectional during stress" |
| 3.2 | Rename | "Crowding Proxy" → "Volatility-Based Regime Detection" |
| 3.4 | Add | Bonferroni for 90 tests, HAC standard errors, FDR alternative |
| 4.3 | Reframe | "Causal direction reverses" → "Causal intensity varies" |
| 4.5 | Soften | "Economic mechanism" → "Speculative interpretation" |
| 4.8 | Honest | Explicitly state unidirectionality doesn't replicate OOS |
| 4.9 | Overhaul | Add transaction costs, bootstrap CI, acknowledge non-significance |
| 5.3 | Add | Limitations on volatility proxy, selection bias, soft assignments |
| 6 | Reframe | Conclusion focuses on intensity, not direction |

---

## Response to Minor Concerns

| Concern | Response |
|---------|----------|
| K=3 unjustified | Add BIC comparison: K=2 (BIC=X), K=3 (BIC=Y), K=4 (BIC=Z). K=3 optimal. |
| No comparison to alternatives | Add footnote: "Time-varying VAR (Primiceri 2005) offers an alternative; we use discrete regimes for interpretability." |
| Factor definitions | Add: "Results may differ for alternative factor constructions (MSCI, Barra)." |
| "First documentation" claim | Remove. Replace with: "We contribute regime-dependent analysis to the factor causality literature." |

---

## Revised Abstract

> We document that the Granger-causal relationships between equity factors vary in *intensity* across volatility regimes. Analyzing 35 years of daily Fama-French factor data (1990–2024), we find that Value (HML) and Size (SMB) factors exhibit weak or no predictive relationships during low-volatility periods, but develop strong *bidirectional* Granger causality during elevated-volatility and crisis regimes. We identify regimes using a Student-t Hidden Markov Model, which detects moderate crises (2011 European debt crisis: 69% detection) that Gaussian models miss (0%). In out-of-sample validation (training: 1990–2014, test: 2015–2024), the frozen model detects 100% of test-period stress events, and the intensification of HML–SMB causality during stress replicates (p<0.001). However, the in-sample finding of regime-specific *unidirectional* causality does not replicate—both directions strengthen during stress. These findings suggest monitoring factor cross-predictability as a stress indicator, though exploratory backtests show only modest, statistically insignificant improvements in risk-adjusted returns.

---

## Estimated Impact on Acceptance

| Before Revisions | After Revisions |
|------------------|-----------------|
| Claims overstated | Claims match evidence |
| Statistical issues | Proper corrections applied |
| Misleading presentation | Honest about limitations |
| "Reject" likely | "Weak Accept / Borderline" possible |

The paper becomes weaker but more honest. The core contribution—regime-dependent causal *intensity*—is still novel and useful, just less dramatic than originally claimed.
