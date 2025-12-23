# Detecting Market Stress via Factor Causality Emergence

## Abstract (Revised)

Standard Gaussian regime-switching models fail to detect moderate market crises, classifying events like the 2011 European debt crisis as "normal" because their severity falls below thresholds calibrated to extreme tail events. We show that Student-t Hidden Markov Models, by bounding the influence of outliers, detect such moderate crises with 69% accuracy versus 0% for Gaussian alternatives. Applying this improved regime detection to Fama-French factor data (1990–2024), we document that Granger-causal relationships between factors intensify during detected stress regimes: Value (HML) and Size (SMB) factors exhibit minimal cross-predictability during calm periods but develop significant bidirectional causality during elevated-volatility regimes (p < 0.001). In out-of-sample validation, a model trained on 1990–2014 detects 100% of stress events in 2015–2024, and the causal intensification pattern replicates. The emergence of factor cross-predictability may serve as a stress indicator, though translating this into economic value remains an open challenge.

---

## 1 Introduction

### 1.1 The Regime Detection Problem

Regime-switching models are foundational tools in financial risk management, enabling practitioners to adjust portfolio exposures based on detected market conditions (Hamilton, 1989; Ang & Bekaert, 2002). However, a critical limitation of standard Gaussian Hidden Markov Models (HMMs) has received insufficient attention: **they systematically fail to detect moderate crises**.

Consider the 2011 European debt crisis. Factor volatility during this period reached 63% of 2008 Global Financial Crisis levels—severe enough to warrant defensive positioning, but not extreme by historical standards. A Gaussian HMM, calibrated to the full 1990–2024 sample, classifies **zero percent** of August–October 2011 as "crisis." The reason is mathematical: Gaussian likelihoods are unbounded, so regime thresholds are dominated by the most extreme historical observations. Any event falling short of 2008 severity is classified as normal.

This is not merely an academic concern. Practitioners relying on Gaussian regime models would have maintained normal-regime positioning through a period that saw:
- S&P 500 drawdown of 19%
- VIX spike to 48
- Significant factor dislocation

**Our first contribution** is demonstrating that Student-t HMMs resolve this problem. The bounded likelihood ratio of heavy-tailed distributions allows moderate deviations to shift posterior probability toward stress regimes. Using identical data and identical regime structure (K=3), a Student-t HMM detects 69% of the 2011 crisis period—versus 0% for Gaussian.

### 1.2 Application: Regime-Dependent Factor Causality

Accurate regime detection enables downstream analyses that would otherwise be corrupted by regime misclassification. We demonstrate this with an application to **factor causality**.

It is well-established that factor correlations increase during market stress (Ang & Chen, 2002). But correlation is symmetric—it cannot distinguish whether Value stress predicts Size stress, or vice versa. Granger causality can identify directional predictability, but standard full-sample tests average across regimes, potentially masking regime-specific patterns.

Using our Student-t regime classifications, we test for Granger causality *within* each regime. We find:

**Finding 1: Causal intensity varies by regime.** During low-volatility regimes, HML and SMB exhibit no significant cross-predictability (p > 0.1). During elevated-volatility regimes, both directions become significant (p < 0.01), with the HML → SMB relationship showing the strongest signal during crisis periods (p = 1.89×10⁻⁵, surviving Bonferroni correction for 90 tests).

**Finding 2: The pattern replicates out-of-sample.** Training on 1990–2014 and testing on 2015–2024, the frozen Student-t HMM detects 100% of test-period stress events. The intensification of HML–SMB causality during detected stress regimes replicates (p < 0.001 in stress vs. p > 0.1 in calm).

**Finding 3: The pattern generalizes across factor pairs.** The stress-induced causality intensification is not unique to HML–SMB. We observe similar patterns for MOM–MKT (momentum predicts market during stress), and RMW–HML (profitability predicts value during stress), suggesting a general phenomenon of factor interconnection during market turbulence.

### 1.3 What We Do Not Claim

We emphasize several limitations upfront:

1. **Not structural causality.** Granger causality establishes predictability, not intervention effects. An unobserved common factor (e.g., liquidity) could generate these patterns.

2. **Not unidirectional.** Our original hypothesis was that causal *direction* would reverse across regimes (Size → Value in buildup, Value → Size in unwind). The out-of-sample evidence shows *bidirectional* intensification instead. The honest finding is about causal *intensity*, not direction.

3. **Not yet economically exploitable.** Exploratory backtests show modest, statistically insignificant improvements in risk-adjusted returns. The practical value of these findings for portfolio management remains an open question.

### 1.4 Contributions

1. **Methodological:** We demonstrate that Student-t HMMs detect moderate financial crises that Gaussian HMMs miss entirely, with practical implications for regime-based risk management (Section 3.3, 4.2).

2. **Empirical:** We document that Granger-causal relationships between Fama-French factors intensify during stress regimes, with the emergence of cross-predictability serving as a potential stress indicator (Section 4.3–4.4).

3. **Validation:** We provide rigorous out-of-sample validation showing that both regime detection and causal intensification generalize to unseen data (Section 4.8).

### 1.5 Paper Organization

Section 2 reviews related work on regime-switching models and causal discovery in finance. Section 3 describes our methodology: data, Student-t HMM specification, and per-regime Granger causality testing. Section 4 presents results, including regime characteristics, the Gaussian vs. Student-t comparison, causal network analysis, and out-of-sample validation. Section 5 discusses implications and limitations. Section 6 concludes.

---

## Key Differences from Original Introduction

| Aspect | Original | Revised |
|--------|----------|---------|
| **Lead** | 2007 quant meltdown, crowding narrative | Regime detection failure (2011 crisis) |
| **Main contribution** | Causal direction reversal | Student-t methodology + causal intensity |
| **Tone** | "We discover/establish" | "We demonstrate/document" |
| **Facts framing** | "Direction reverses" | "Intensity varies" |
| **Limitations** | End of paper | Upfront in intro (1.3) |
| **Economic claims** | "Direct implications" | "Open question" |

---

## Title Options

| Option | Emphasis |
|--------|----------|
| **Detecting Market Stress via Factor Causality Emergence** | Methodology + application |
| **Student-t Hidden Markov Models for Financial Regime Detection** | Pure methodology |
| **When Factors Become Entangled: Regime-Dependent Cross-Predictability** | Empirical finding |
| **Regime-Dependent Factor Causality: Evidence from Student-t HMMs** | Balanced |

**Recommendation:** "Detecting Market Stress via Factor Causality Emergence" — positions the paper as a *tool* paper with an interesting application, rather than a *discovery* paper with overclaims.
