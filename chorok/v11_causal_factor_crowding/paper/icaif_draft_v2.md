# Regime-Dependent Causal Structure in Factor Returns: Early Warning Through Crowding Spillover

## Abstract

Factor investing strategies are increasingly susceptible to crowding risk, where correlated positions amplify drawdowns during market stress. While existing research documents factor correlations and regime-switching behavior independently, the **causal structure** between factors—and how it changes across market regimes—remains unexplored. Using 35 years of Fama-French factor data (1990-2025), we discover that causal relationships between factors are **regime-dependent**: Value (HML) Granger-causes Size (SMB) only during Crisis regimes (p = 1.89e-05, 9-day lag), while the reverse direction—Size causes Value—emerges only during Crowding regimes (p = 1.94e-04, 3-day lag). This directional asymmetry provides a novel early warning mechanism: the emergence of specific causal links signals regime transitions before they fully materialize. Our Student-t Hidden Markov Model detects these regime shifts 2 months before the 2008 Lehman collapse, enabling proactive risk management.

**Keywords:** Factor Crowding, Causal Discovery, Regime Switching, Heavy-Tailed Distributions, Risk Management

---

## 1. Introduction

The 2007 Quant Meltdown and 2008 Financial Crisis demonstrated that factor crowding can trigger cascading losses across seemingly diversified portfolios. When multiple investors hold similar factor exposures, forced liquidation by one fund creates price pressure that affects all others—a phenomenon that standard correlation-based risk models fail to anticipate.

**The Missing Piece: Causal Direction**

Existing research establishes that:
1. Factor correlations increase during market stress (Ang & Chen, 2002)
2. Returns exhibit regime-switching behavior (Hamilton, 1989)
3. Factor crowding amplifies drawdowns (Stein, 2009)

However, a critical question remains unanswered: **Does the causal structure between factors change across regimes?**

Correlation tells us factors move together, but not which factor drives which. If Value (HML) crowding *causes* Size (SMB) crowding with a predictable lag, this provides actionable early warning. If this causal link only exists during specific regimes, detecting regime transitions becomes essential for risk management.

### Our Contributions

1. **Novel Finding: Regime-Dependent Causality**
   - HML → SMB causal link exists *only* in Crisis regime (p = 1.89e-05)
   - SMB → HML causal link exists *only* in Crowding regime (p = 1.94e-04)
   - Normal regime shows no significant causal link between these factors

2. **Mechanistic Explanation**
   - Crowding regime: Size stocks become crowded first, pulling Value positions
   - Crisis regime: Value positions unwind, cascading to Size factor
   - The *direction* of causality indicates the *source* of crowding pressure

3. **Early Warning System**
   - Emergence of causal links signals regime transition
   - Student-t HMM detects regime shifts 2 months before Lehman collapse
   - Provides actionable lead time for portfolio de-risking

4. **Methodological Contribution**
   - Student-t emissions capture heavy-tailed crisis behavior
   - Per-regime Granger causality reveals hidden causal dynamics
   - Framework applicable to other factor systems and asset classes

---

## 2. Related Work

### 2.1 Factor Crowding

Stein (2009) documents how crowded trades amplify losses during liquidation events. Lou & Polk (2022) measure crowding through return comovement. Hua & Sun (2024) study crowding dynamics but do not examine causal structure.

**Gap:** Existing work measures crowding *levels* but not causal *spillover* between factors.

### 2.2 Regime-Switching Models

Hamilton (1989) introduced Markov-switching models for business cycles. Guidolin & Timmermann (2007) apply regime models to asset allocation. Ang & Bekaert (2002) document regime-dependent correlations.

**Gap:** Regime models focus on return distributions, not causal structure.

### 2.3 Causal Discovery in Finance

Howard et al. (2025) construct causal networks among individual stocks. CausalStock (NeurIPS 2024) discovers temporal causality for stock prediction. FANTOM (2025) performs regime-switching causal discovery.

**Gap:** No prior work examines regime-dependent causality at the *factor* level, specifically for crowding spillover.

### 2.4 Our Position

We fill the intersection of three literatures:

| Aspect | Prior Work | Our Contribution |
|--------|------------|------------------|
| Factor relationships | Correlation-based | **Causal direction** |
| Regime analysis | Distribution changes | **Causal structure changes** |
| Crowding risk | Level measurement | **Spillover prediction** |

---

## 3. Method

### 3.1 Problem Setup

Given factor return time series $\mathbf{X}_t \in \mathbb{R}^d$ (d = 6 Fama-French factors), we seek to:
1. Identify latent market regimes $z_t \in \{1, ..., K\}$
2. Estimate regime-specific causal adjacency matrices $\mathbf{W}^{(k)}$
3. Detect regime transitions that signal causal structure changes

### 3.2 Crowding Proxy Construction

We construct a crowding proxy from factor returns using rolling volatility:

$$X_{t,i} = \sqrt{\frac{1}{\tau} \sum_{s=t-\tau}^{t} (R_{s,i} - \bar{R}_i)^2}$$

where $\tau = 60$ days. High volatility indicates crowded positions being unwound (forced selling creates volatility).

### 3.3 Student-t Hidden Markov Model

We model regime-switching with heavy-tailed emissions:

**Transition:** $P(z_t = k | z_{t-1} = j) = A_{jk}$

**Emission:** $\mathbf{X}_t | z_t = k \sim \text{MVT}_d(\boldsymbol{\mu}^{(k)}, \boldsymbol{\Sigma}^{(k)}, \nu^{(k)})$

where $\nu^{(k)}$ controls tail heaviness. Lower $\nu$ (heavier tails) characterizes crisis regimes.

**Why Student-t?** Financial returns exhibit kurtosis >> 3. Gaussian HMMs underestimate extreme event probabilities, missing moderate crises like the 2011 EU Debt Crisis. Student-t's bounded log-likelihood ratio enables detection of moderate-severity regime shifts.

### 3.4 Per-Regime Granger Causality

For each regime $k$, we test pairwise Granger causality:

$$H_0: \text{Factor } i \text{ does not Granger-cause Factor } j$$

We use F-tests across lags 1-15 days, selecting the lag with minimum p-value. An edge $i \rightarrow j$ is included in $\mathbf{W}^{(k)}$ if $p < 0.01$ (Bonferroni-corrected for 30 pairs).

### 3.5 Early Warning Mechanism

The emergence of regime-specific causal links signals transitions:

1. **Normal → Crowding:** SMB → HML link appears
2. **Crowding → Crisis:** HML → SMB link appears
3. **Crisis → Normal:** Both links disappear

Monitoring these causal links provides early warning before regime transitions complete.

---

## 4. Experiments

### 4.1 Data

- **Source:** Fama-French 6 factors (MKT, SMB, HML, RMW, CMA, MOM)
- **Period:** January 1990 – December 2024 (35 years)
- **Frequency:** Daily (8,967 observations after rolling window)
- **Crowding proxy:** 60-day rolling volatility, z-scored

### 4.2 Regime Detection Results

Our Student-t HMM identifies three distinct regimes:

| Regime | Days | % of Sample | Avg Volatility | ν (tail) |
|--------|------|-------------|----------------|----------|
| Normal | 3,310 | 37% | Low | ~15 |
| Crowding | 4,490 | 50% | Medium | ~8 |
| Crisis | 1,167 | 13% | High | ~4 |

**Crisis periods detected:**
- 2000-2002 Dot-com crash
- 2008 Financial Crisis (detected July 2008, 2 months before Lehman)
- 2011 EU Debt Crisis
- 2020 COVID-19 crash (detected March 9, 1 week before market bottom)

### 4.3 Key Result: Regime-Dependent Causal Structure

**Table 1: HML ↔ SMB Causal Relationship by Regime**

| Regime | HML → SMB | SMB → HML |
|--------|-----------|-----------|
| Normal | p = 0.015 (NS) | p = 0.098 (NS) |
| Crowding | p = 0.087 (NS) | **p = 1.94e-04** (lag=3) |
| Crisis | **p = 1.89e-05** (lag=9) | p = 0.165 (NS) |

NS = Not significant at α = 0.01

**Key Finding:** The causal direction *reverses* between regimes:
- Crowding: Size drives Value (SMB → HML)
- Crisis: Value drives Size (HML → SMB)

This is **not** captured by correlation analysis, which would show both factors moving together regardless of direction.

### 4.4 Full Causal Structure Comparison

**Table 2: Significant Causal Edges by Regime (p < 0.01)**

| Regime | Total Edges | Unique Edges | Key Patterns |
|--------|-------------|--------------|--------------|
| Normal | 15 | 2 | MKT drives all factors |
| Crowding | 19 | 5 | SMB → HML, bidirectional links |
| Crisis | 18 | 4 | HML → SMB, feedback loops |

**Unique to Crisis regime:**
- HML → SMB (Value crowding causes Size crowding)
- RMW → MKT (Profitability affects market)
- RMW → CMA (Quality-Investment link)

**Unique to Crowding regime:**
- SMB → HML (Size crowding causes Value crowding)
- RMW → SMB, RMW → HML (Profitability spillover)

### 4.5 Economic Interpretation

**Crowding Regime (SMB → HML):**
- Small-cap stocks become crowded (popular factor tilt)
- Crowding pressure spills to Value stocks (many small-caps are also value)
- 3-day lag suggests rapid transmission through portfolio rebalancing

**Crisis Regime (HML → SMB):**
- Value positions unwind during stress (flight from risky "cheap" stocks)
- Forced selling cascades to Size factor (overlapping holdings)
- 9-day lag reflects slower institutional unwinding

### 4.6 Comparison with Gaussian HMM

Student-t HMM detects regimes that Gaussian HMM misses:

| Crisis | Student-t Detection | Gaussian Detection |
|--------|---------------------|-------------------|
| 2008 Financial | 96% | 96% |
| 2011 EU Debt | **69%** | **0%** |
| 2020 COVID | 85% | 81% |

The 2011 EU Debt Crisis, with volatility at 63% of 2008 levels, falls below Gaussian's threshold. Because regime detection is prerequisite for per-regime causality, **Gaussian HMM would miss the regime-dependent causal structure entirely for moderate crises.**

### 4.7 Early Warning Performance

**Table 3: Lead Time Before Peak Crisis**

| Event | Regime Shift Detected | Peak Crisis | Lead Time |
|-------|----------------------|-------------|-----------|
| Lehman (Sep 15, 2008) | Jul 16, 2008 | Sep 2008 | **2 months** |
| EU Crisis (Aug 2011) | Aug 1, 2011 | Aug 2011 | **Same week** |
| COVID (Mar 16, 2020) | Mar 9, 2020 | Mar 2020 | **1 week** |

---

## 5. Discussion

### 5.1 Why Causal Direction Matters

Correlation tells us HML and SMB move together during stress. But:
- If HML causes SMB: Monitor Value crowding, hedge Size exposure
- If SMB causes HML: Monitor Size crowding, hedge Value exposure

The regime-dependent direction provides **specific, actionable guidance**.

### 5.2 Mechanism: Crowding Cascade

```
Normal → Crowding:
  Small-cap strategies crowd → SMB volatility ↑
  → Spills to overlapping Value positions (SMB → HML)

Crowding → Crisis:
  Value positions forced to liquidate → HML volatility ↑
  → Cascades to Small-cap (shared holdings) (HML → SMB)
```

### 5.3 Practical Implications

1. **Risk Management:** Monitor causal link emergence as regime indicator
2. **Portfolio Construction:** Reduce exposure to "destination" factor when causal link appears
3. **Timing:** 3-9 day lags provide actionable lead time

### 5.4 Limitations

1. **Crowding proxy:** Rolling volatility is a proxy; direct crowding data would improve precision
2. **Stationarity:** Causal structure may evolve over decades (we assume regime-stationarity)
3. **Granger limitations:** Granger causality implies predictive, not necessarily structural causality

### 5.5 Future Work

1. Incorporate ETF flow data for direct crowding measurement
2. Extend to multi-asset (equity, fixed income, commodities)
3. Build real-time monitoring system for causal link detection

---

## 6. Conclusion

We present the first evidence that causal relationships between factors are regime-dependent. Value (HML) Granger-causes Size (SMB) only during Crisis regimes, while the reverse direction emerges only during Crowding regimes. This finding has immediate practical implications: the emergence of specific causal links provides early warning of regime transitions, with 2-month lead time before the 2008 Lehman collapse.

Our results demonstrate that factor risk management must move beyond correlation to consider directional causality—and that this causal structure is not static but regime-dependent. As factor investing continues to grow, understanding these crowding spillover dynamics becomes essential for avoiding the next cascade.

---

## References

Ang, A., & Bekaert, G. (2002). International asset allocation with regime shifts. *Review of Financial Studies*, 15(4), 1137-1187.

Ang, A., & Chen, J. (2002). Asymmetric correlations of equity portfolios. *Journal of Financial Economics*, 63(3), 443-494.

FANTOM (2025). Causal Discovery in Multi-Regime Time Series. *arXiv:2506.17065*.

Guidolin, M., & Timmermann, A. (2007). Asset allocation under multivariate regime switching. *Journal of Economic Dynamics and Control*, 31(11), 3503-3544.

Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. *Econometrica*, 57(2), 357-384.

Howard, C., et al. (2025). Causal Network Representations in Factor Investing. *Working Paper*.

Hua, J., & Sun, W. (2024). Dynamics of Factor Crowding. *SSRN Working Paper*.

Liu, C., & Rubin, D. B. (1995). ML estimation of the t distribution using EM and its extensions, ECM and ECME. *Statistica Sinica*, 5(1), 19-39.

Lou, D., & Polk, C. (2022). Comomentum: Inferring arbitrage activity from return correlations. *Review of Financial Studies*, 35(7), 3272-3302.

Stein, J. C. (2009). Sophisticated investors and market efficiency. *Journal of Finance*, 64(4), 1517-1548.

---

## Appendix

### A. Implementation Details

- **Student-t HMM:** 3 regimes, 100 EM iterations, convergence threshold 1e-4
- **Granger causality:** F-test, lags 1-15, significance α = 0.01
- **Bonferroni correction:** Applied for 30 pairwise tests

### B. Robustness Checks

1. **Different lag ranges (5-20):** Results stable
2. **Different significance thresholds (0.05, 0.01, 0.001):** Key finding (HML↔SMB regime-dependency) robust
3. **Subsample analysis (pre/post 2008):** Pattern consistent

### C. Full Causal Structure Tables

[Per-regime adjacency matrices available in supplementary material]
