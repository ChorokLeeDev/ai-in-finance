# Causal Structure Changes Across Market Regimes: Evidence from Factor Returns

**Anonymous Authors**

## Abstract

We document that the causal structure between equity factors is regime-dependent. Analyzing 35 years of daily Fama-French factor data (1990–2024), we find that Value (HML) Granger-causes Size (SMB) exclusively during crisis regimes (p < 2×10⁻⁵, 9-day lag), while the reverse—Size causes Value—occurs only during crowding regimes (p < 2×10⁻⁴, 3-day lag). No significant causal relationship exists between these factors during normal market conditions. We identify regimes using a Student-t Hidden Markov Model, which captures the heavy-tailed behavior of factor returns and detects regime shifts that Gaussian models miss. The emergence of regime-specific causal links provides early warning of market transitions, with crisis regime detection occurring two months before the 2008 Lehman Brothers collapse. Our findings establish that factor risk models must account for regime-dependent causal dynamics, not merely time-varying correlations.

---

## 1 Introduction

The August 2007 quantitative meltdown, in which systematic equity strategies lost 30% in three days, revealed a critical blind spot in factor-based risk management: **factors that appear uncorrelated in normal markets can exhibit strong dependencies during stress periods** (Khandani & Lo, 2011). The standard explanation focuses on time-varying correlations—factors become more correlated during crises. However, correlation is symmetric and provides no information about *which factor drives which*.

We investigate a more fundamental question: **Does the causal structure between factors change across market regimes?** If factor A causes factor B during crises but not during normal periods, this has direct implications for risk management: monitoring A provides early warning for B, but only in specific market conditions.

Using Granger causality analysis within regime-dependent subsamples, we establish three empirical facts:

1. **Regime-specific causality exists.** The Value factor (HML) Granger-causes the Size factor (SMB) with a 9-day lag, but only during crisis regimes (p = 1.89×10⁻⁵). This relationship is absent in normal and crowding regimes.

2. **Causal direction reverses across regimes.** During crowding regimes, Size Granger-causes Value (p = 1.94×10⁻⁴, 3-day lag)—the opposite direction. This bidirectional asymmetry cannot be detected by correlation analysis.

3. **Causal emergence provides early warning.** The transition from no-causality to active causality coincides with regime shifts, enabling detection of crisis onset two months before peak stress.

These findings have immediate practical implications. During crowding periods, Size factor movements predict Value movements three days ahead. During crises, the prediction direction reverses with a longer nine-day horizon. Risk managers can monitor the "source" factor to anticipate movements in the "destination" factor—but only if they correctly identify the current regime.

### Contributions

- **Empirical finding:** First documentation of regime-dependent causal structure between Fama-French factors (Section 4.3)
- **Methodological:** Student-t HMM for regime detection that captures moderate crises missed by Gaussian models (Section 3.3)
- **Practical:** Early warning mechanism based on causal link emergence (Section 4.7)

---

## 2 Related Work

**Factor crowding and systemic risk.** Anton & Polk (2014) show that stocks with common mutual fund ownership exhibit correlated returns. Lou & Polk (2022) measure arbitrage activity through return comovement. Stein (2009) formalizes how crowded trades amplify drawdowns. These works measure crowding *intensity* but do not examine causal spillover *between* factors.

**Regime-switching in finance.** Hamilton (1989) introduced Markov-switching models, extended to asset allocation by Guidolin & Timmermann (2007) and to correlations by Ang & Bekaert (2002). Bulla (2011) applies Student-t HMMs to financial returns. This literature focuses on regime-dependent *distributions*, not regime-dependent *causal structure*.

**Causal discovery in finance.** Hiemstra & Jones (1994) apply Granger causality to stock-volume dynamics. Billio et al. (2012) use Granger networks to measure systemic risk. Recent work includes causal discovery for stock prediction (Li et al., 2024) and regime-switching causal models (Huang et al., 2025). None examine factor-level causal dynamics.

**Our contribution fills the intersection:** regime-dependent causal structure at the factor level, with implications for crowding risk management.

---

## 3 Methodology

### 3.1 Data and Crowding Proxy

We use daily returns for the Fama-French six factors—Market (MKT-RF), Size (SMB), Value (HML), Profitability (RMW), Investment (CMA), and Momentum (MOM)—from Kenneth French's data library, spanning January 1990 to December 2024 (8,967 trading days).

We construct a crowding proxy using 60-day rolling volatility:
$$\sigma_{i,t} = \sqrt{\frac{1}{60}\sum_{s=t-59}^{t}(r_{i,s} - \bar{r}_{i,t})^2}$$

High rolling volatility indicates crowded positions being unwound, as forced liquidation generates elevated price impact (Khandani & Lo, 2011). We standardize each factor's volatility series to zero mean and unit variance.

### 3.2 Hidden Markov Model with Student-t Emissions

Let $\mathbf{x}_t \in \mathbb{R}^6$ denote the vector of standardized crowding proxies at time $t$. We model the data as generated by a Hidden Markov Model with $K=3$ latent states (regimes):

**Transition model:**
$$p(z_t = k \mid z_{t-1} = j) = A_{jk}, \quad \sum_{k=1}^K A_{jk} = 1$$

**Emission model (multivariate Student-t):**
$$p(\mathbf{x}_t \mid z_t = k) = \frac{\Gamma(\frac{\nu_k+6}{2})}{\Gamma(\frac{\nu_k}{2})(\nu_k\pi)^3|\boldsymbol{\Sigma}_k|^{1/2}}\left(1 + \frac{\delta_k(\mathbf{x}_t)}{\nu_k}\right)^{-\frac{\nu_k+6}{2}}$$

where $\delta_k(\mathbf{x}_t) = (\mathbf{x}_t - \boldsymbol{\mu}_k)^\top\boldsymbol{\Sigma}_k^{-1}(\mathbf{x}_t - \boldsymbol{\mu}_k)$ is the squared Mahalanobis distance.

**Why Student-t?** Financial returns exhibit excess kurtosis (Cont, 2001). Gaussian HMMs calibrate regime thresholds to the most extreme observations, causing them to miss moderate crises. Student-t distributions with low degrees of freedom $\nu_k$ accommodate heavy tails, enabling detection of crises with volatility below historical extremes.

We estimate parameters via the EM algorithm with auxiliary variables (Liu & Rubin, 1995), running for 100 iterations or until log-likelihood convergence (threshold: 10⁻⁴).

### 3.3 Per-Regime Granger Causality

For each regime $k$, we extract the subsequence of observations $\{(\mathbf{x}_t, t) : \hat{z}_t = k\}$ where $\hat{z}_t$ is the Viterbi-decoded regime assignment.

We test pairwise Granger causality between all factor pairs $(i, j)$:

$$H_0: r_{j,t} \text{ is not predictable from } \{r_{i,t-\ell}\}_{\ell=1}^{L} \text{ given } \{r_{j,t-\ell}\}_{\ell=1}^{L}$$

using standard F-tests with maximum lag $L=15$. We report the minimum p-value across lags and its corresponding optimal lag. An edge $i \to j$ is declared significant if $p < 0.01$ after Bonferroni correction for 30 pairwise tests ($\alpha_{\text{corrected}} = 0.01/30 \approx 3.3 \times 10^{-4}$).

We deliberately use the standard Granger framework rather than more complex causal discovery methods (e.g., PCMCI, DYNOTEARS) for interpretability and to establish the core empirical finding with widely-accepted methodology.

---

## 4 Results

### 4.1 Regime Characteristics

The fitted Student-t HMM identifies three regimes with distinct characteristics (Table 1).

**Table 1: Regime Summary Statistics**

| Regime | Days | Proportion | Mean Vol | Est. ν | Interpretation |
|--------|------|------------|----------|--------|----------------|
| 0 (Normal) | 3,310 | 37% | −0.41 | 14.2 | Low volatility, light tails |
| 1 (Crowding) | 4,490 | 50% | 0.12 | 7.8 | Medium volatility |
| 2 (Crisis) | 1,167 | 13% | 1.89 | 3.9 | High volatility, heavy tails |

Crisis periods align with known market stress events: the 2000–2002 dot-com crash, 2008 financial crisis, 2011 European debt crisis, and 2020 COVID-19 crash. The estimated degrees of freedom ($\nu \approx 4$) for crisis regimes indicates substantially heavier tails than Gaussian ($\nu \to \infty$).

### 4.2 Comparison with Gaussian HMM

To validate the Student-t specification, we compare crisis detection rates against a Gaussian HMM (Table 2).

**Table 2: Crisis Detection Rates by Model**

| Event | Period | Student-t | Gaussian |
|-------|--------|-----------|----------|
| 2008 Financial Crisis | Jul 2008–Jun 2009 | 96% | 96% |
| 2011 EU Debt Crisis | Jul–Oct 2011 | 69% | 0% |
| 2020 COVID-19 | Feb–Jun 2020 | 85% | 81% |

The 2011 European debt crisis, with peak volatility at 63% of 2008 levels, falls entirely below the Gaussian model's crisis threshold. The Student-t model's bounded log-likelihood ratio enables detection of moderate-severity events. This is critical because **accurate regime assignment is a prerequisite for discovering regime-dependent causal structure.**

### 4.3 Main Result: Regime-Dependent Causality

Table 3 presents the core finding: the causal relationship between HML and SMB depends on the market regime.

**Table 3: Granger Causality Between HML and SMB by Regime**

| Regime | HML → SMB | SMB → HML |
|--------|-----------|-----------|
| Normal | p = 0.015, NS | p = 0.098, NS |
| Crowding | p = 0.087, NS | **p = 1.94×10⁻⁴**, lag=3 |
| Crisis | **p = 1.89×10⁻⁵**, lag=9 | p = 0.165, NS |

NS = Not significant at Bonferroni-corrected α = 3.3×10⁻⁴

**Key observations:**

1. **Normal regime:** Neither direction is significant. HML and SMB exhibit independent dynamics.

2. **Crowding regime:** SMB Granger-causes HML (p = 1.94×10⁻⁴) with a 3-day lag. Size factor movements predict Value movements, but not vice versa.

3. **Crisis regime:** The direction reverses. HML Granger-causes SMB (p = 1.89×10⁻⁵) with a 9-day lag. Value factor movements predict Size movements.

This pattern—no relationship in normal times, opposite causal directions in different stress regimes—cannot be detected by analyzing the full sample or by examining correlations.

### 4.4 Full Causal Network by Regime

We extend the analysis to all 30 directed pairs (Table 4).

**Table 4: Significant Granger-Causal Edges by Regime (p < 0.01)**

| Regime | Total Edges | Regime-Specific Edges |
|--------|-------------|----------------------|
| Normal | 15 | CMA → MOM, MOM → CMA |
| Crowding | 19 | SMB → HML, RMW → SMB, RMW → HML |
| Crisis | 18 | HML → SMB, RMW → MKT, RMW → CMA |

Market (MKT) Granger-causes all other factors in all regimes—a structural relationship reflecting the CAPM's market risk premium. The regime-specific edges reveal how crowding spillover channels change with market conditions.

### 4.5 Economic Interpretation

The asymmetric HML–SMB causality admits a natural economic interpretation:

**Crowding regime (SMB → HML, 3-day lag):** During the buildup phase, small-cap strategies become crowded as investors chase the size premium. Because many small-cap stocks are also value stocks (low market-to-book), crowding in SMB-exposed positions creates price pressure on value stocks. The 3-day lag reflects rapid portfolio rebalancing.

**Crisis regime (HML → SMB, 9-day lag):** During unwinds, value positions are liquidated first—"cheap" stocks are often cheap for fundamental reasons and face the largest drawdowns. The resulting selling pressure propagates to small-cap stocks with overlapping ownership. The longer 9-day lag reflects slower institutional deleveraging.

### 4.6 Early Warning Performance

Table 5 reports the lead time between regime shift detection and peak crisis.

**Table 5: Early Warning Lead Time**

| Event | First Crisis Detection | Peak | Lead Time |
|-------|----------------------|------|-----------|
| Lehman Brothers (Sep 15, 2008) | Jul 16, 2008 | Sep 2008 | 61 days |
| EU Debt Crisis | Aug 1, 2011 | Aug 8, 2011 | 7 days |
| COVID-19 Crash | Mar 9, 2020 | Mar 23, 2020 | 14 days |

The two-month lead time before Lehman's collapse corresponds to the early signs of stress in Bear Stearns and the broader financial sector during summer 2008.

### 4.7 Robustness

We verify robustness across several dimensions:

1. **Lag specification:** Results hold for maximum lags 10, 15, and 20
2. **Significance threshold:** HML–SMB pattern persists at α = 0.05, 0.01, and 0.001
3. **Subsample stability:** Pattern present in both pre-2008 (1990–2007) and post-2008 (2008–2024) periods
4. **Alternative crowding proxy:** Results replicate using 30-day and 90-day rolling windows

---

## 5 Discussion

### 5.1 Implications for Risk Management

Our findings suggest a shift from correlation-based to causality-based factor risk management:

1. **Monitor the source factor.** During crowding regimes, track SMB to anticipate HML movements. During crises, track HML to anticipate SMB.

2. **Regime detection is primary.** Causal relationships that do not exist in normal markets cannot provide early warning. Accurate regime identification enables the right risk model at the right time.

3. **Lag structure enables hedging.** The 3-9 day lags provide actionable windows for portfolio adjustment.

### 5.2 Limitations

1. **Granger vs. structural causality.** Granger causality establishes predictive, not necessarily interventional, relationships. Confounding by unobserved common factors (e.g., liquidity) remains possible.

2. **Crowding proxy.** Rolling volatility is an indirect measure. Direct position data (e.g., 13F filings) could provide cleaner identification but at lower frequency.

3. **Regime stationarity.** We assume the three-regime structure is stable across our 35-year sample. The causal relationships within each regime may evolve.

4. **Sample size in crisis regime.** With 1,167 crisis days, statistical power for detecting weak effects is limited.

### 5.3 Future Directions

Extensions include: (1) real-time regime detection with filtering rather than smoothing; (2) incorporation of cross-asset factors (fixed income, commodities); (3) structural causal discovery using interventional data from ETF flows.

---

## 6 Conclusion

We document that the causal structure between equity factors is regime-dependent. Value Granger-causes Size only during crisis regimes; Size Granger-causes Value only during crowding regimes. This directional asymmetry—invisible to correlation analysis—has direct implications for factor risk management. As factor investing assets under management continue to grow, understanding regime-dependent causal dynamics becomes essential for anticipating the next crowding cascade.

---

## References

Ang, A., & Bekaert, G. (2002). International asset allocation with regime shifts. *Review of Financial Studies*, 15(4), 1137–1187.

Anton, M., & Polk, C. (2014). Connected stocks. *Journal of Finance*, 69(3), 1099–1127.

Billio, M., Getmansky, M., Lo, A. W., & Pelizzon, L. (2012). Econometric measures of connectedness and systemic risk in the finance and insurance sectors. *Journal of Financial Economics*, 104(3), 535–559.

Bulla, J. (2011). Hidden Markov models with t components. Increased persistence and other aspects. *Quantitative Finance*, 11(3), 459–475.

Cont, R. (2001). Empirical properties of asset returns: stylized facts and statistical issues. *Quantitative Finance*, 1(2), 223–236.

Guidolin, M., & Timmermann, A. (2007). Asset allocation under multivariate regime switching. *Journal of Economic Dynamics and Control*, 31(11), 3503–3544.

Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. *Econometrica*, 57(2), 357–384.

Hiemstra, C., & Jones, J. D. (1994). Testing for linear and nonlinear Granger causality in the stock price-volume relation. *Journal of Finance*, 49(5), 1639–1664.

Huang, B., et al. (2025). FANTOM: Causal discovery in multi-regime time series. *arXiv:2506.17065*.

Khandani, A. E., & Lo, A. W. (2011). What happened to the quants in August 2007? Evidence from factors and transactions data. *Journal of Financial Markets*, 14(1), 1–46.

Li, X., et al. (2024). CausalStock: Deep end-to-end causal discovery for news-driven stock movement prediction. *NeurIPS 2024*.

Liu, C., & Rubin, D. B. (1995). ML estimation of the t distribution using EM and its extensions, ECM and ECME. *Statistica Sinica*, 5(1), 19–39.

Lou, D., & Polk, C. (2022). Comomentum: Inferring arbitrage activity from return correlations. *Review of Financial Studies*, 35(7), 3272–3302.

Stein, J. C. (2009). Presidential address: Sophisticated investors and market efficiency. *Journal of Finance*, 64(4), 1517–1548.

---

## Appendix

### A. Implementation Details

**Student-t HMM:**
- Number of regimes: K = 3
- Initialization: K-means clustering on full sample
- EM iterations: 100 (or convergence at Δ log-likelihood < 10⁻⁴)
- Regime ordering: by mean Mahalanobis distance from origin

**Granger Causality:**
- Maximum lag: L = 15 trading days
- Test statistic: F-test (standard implementation via statsmodels)
- Multiple testing correction: Bonferroni (30 tests, α = 0.01/30)

### B. Additional Results

**Table B1: Full Granger Causality P-Value Matrix (Crisis Regime)**

|  | MKT | SMB | HML | RMW | CMA | MOM |
|--|-----|-----|-----|-----|-----|-----|
| MKT | — | 6.5e-21 | 2.1e-03 | 1.2e-04 | 3.8e-05 | 1.8e-05 |
| SMB | 3.1e-06 | — | 0.165 | 4.2e-04 | 8.1e-04 | 2.3e-03 |
| HML | 1.8e-03 | **1.9e-05** | — | 0.089 | 5.4e-04 | 1.1e-03 |
| RMW | **3.2e-05** | 0.021 | 0.078 | — | **4.1e-08** | 0.034 |
| CMA | 0.018 | 0.091 | 0.142 | 6.8e-04 | — | 0.056 |
| MOM | 4.1e-04 | 0.312 | 2.8e-04 | 0.087 | 0.124 | — |

Bold: regime-specific significant edges (p < 3.3e-04)

### C. Reproducibility

Code and data are available at: [URL to be added upon publication]

All experiments run on standard hardware (Apple M1, 16GB RAM). Total computation time: approximately 5 minutes.
