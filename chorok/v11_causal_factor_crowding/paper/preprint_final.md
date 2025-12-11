# Causal Structure Changes Across Market Regimes: Evidence from Factor Returns

## Abstract

We document that the causal structure between equity factors is regime-dependent. Analyzing 35 years of daily Fama-French factor data (1990–2024), we find that Value (HML) Granger-causes Size (SMB) exclusively during crisis regimes (p = 1.89×10⁻⁵, 9-day lag), while the reverse direction—Size causes Value—emerges only during crowding regimes (p = 1.94×10⁻⁴, 3-day lag). No significant causal relationship exists between these factors during normal market conditions. This directional asymmetry—invisible to correlation analysis—has direct implications for risk management: during crowding periods, Size factor movements predict Value movements three days ahead; during crises, the prediction direction reverses with a nine-day horizon. We identify regimes using a Student-t Hidden Markov Model, which captures the heavy-tailed behavior of factor returns and detects moderate crises (2011 European debt crisis: 69% detection) that Gaussian models entirely miss (0% detection). The emergence of regime-specific causal links provides early warning of market transitions, with crisis regime detection occurring two months before the 2008 Lehman Brothers collapse.

---

## 1 Introduction

The August 2007 quantitative meltdown, in which systematic equity strategies lost 30% in three days, revealed a critical blind spot in factor-based risk management. When multiple quantitative funds held similar factor exposures, forced liquidation by one fund created price pressure that cascaded to all others (Khandani & Lo, 2011). Standard correlation-based risk models failed to anticipate this cascade because they measure *co-movement* but not *causal direction*.

**The Missing Piece: Which Factor Drives Which?**

Existing research establishes three stylized facts:
1. Factor correlations increase during market stress (Ang & Chen, 2002)
2. Returns exhibit regime-switching behavior (Hamilton, 1989)
3. Factor crowding amplifies drawdowns during liquidation (Stein, 2009)

However, a critical question remains unanswered: **Does the causal structure between factors change across market regimes?**

Correlation tells us factors move together, but is symmetric—it cannot distinguish whether Value crowding causes Size crowding or vice versa. If we knew the causal direction, and if this direction varied by regime, we could:
- Monitor the "source" factor to anticipate movements in the "destination" factor
- Adjust hedges based on the current regime's causal structure
- Detect regime transitions by observing when causal links emerge or disappear

### Our Discovery

Using Granger causality analysis within regime-dependent subsamples identified by a Student-t Hidden Markov Model, we establish three empirical facts:

**Fact 1: Regime-specific causality exists.** The Value factor (HML) Granger-causes the Size factor (SMB) with a 9-day lag, but *only* during crisis regimes (p = 1.89×10⁻⁵). This relationship is statistically absent in normal and crowding regimes.

**Fact 2: Causal direction reverses across regimes.** During crowding regimes, Size Granger-causes Value (p = 1.94×10⁻⁴, 3-day lag)—the *opposite* direction from crisis regimes. Normal regimes exhibit no significant causal link in either direction.

**Fact 3: Causal emergence provides early warning.** The transition from no-causality to active causality coincides with regime shifts. Our model detects the crisis regime two months before Lehman Brothers' collapse, providing actionable lead time for portfolio adjustment.

### Contributions

1. **Novel Empirical Finding:** First documentation that causal relationships between Fama-French factors are regime-dependent, with direction reversal between crowding and crisis regimes (Section 4.3)

2. **Methodological:** Student-t HMM for regime detection that captures moderate crises missed by Gaussian models—critical because accurate regime identification is prerequisite for discovering regime-dependent causality (Section 3.3, 4.2)

3. **Economic Mechanism:** Interpretation of asymmetric causality through crowding cascade dynamics: Size → Value during buildup, Value → Size during unwind (Section 4.5)

4. **Practical Application:** Early warning system based on causal link emergence with documented lead times (Section 4.6)

---

## 2 Related Work

### 2.1 Factor Crowding and Systemic Risk

Anton & Polk (2014) show that stocks with common mutual fund ownership exhibit correlated returns, establishing the mechanism by which crowded positions create co-movement. Lou & Polk (2022) measure arbitrage activity through return comovement, finding that comomentum predicts factor returns. Stein (2009) formalizes how crowded trades amplify drawdowns through forced liquidation. Hua & Sun (2024) study factor crowding dynamics empirically but do not examine causal spillover between factors.

**Gap:** Existing work measures crowding *intensity* within individual factors but does not examine causal *spillover* between factors.

### 2.2 Regime-Switching Models in Finance

Hamilton (1989) introduced Markov-switching autoregressive models for business cycle analysis. Guidolin & Timmermann (2007) extend regime models to multivariate asset allocation, finding that regime-dependent portfolios outperform static allocations. Ang & Bekaert (2002) document regime-dependent correlations in international equity markets. Bulla (2011) applies Student-t HMMs to financial returns, demonstrating improved fit over Gaussian specifications.

**Gap:** Regime-switching models focus on regime-dependent *distributions* (means, variances, correlations), not regime-dependent *causal structure*.

### 2.3 Causal Discovery in Finance

Hiemstra & Jones (1994) apply Granger causality to stock price-volume dynamics. Billio et al. (2012) construct Granger causality networks among financial institutions to measure systemic risk, finding increased connectedness before crises. Recent advances include CausalStock (Li et al., NeurIPS 2024), which discovers temporal causality for stock prediction, and FANTOM (Huang et al., 2025), which performs regime-switching causal discovery for general time series.

**Gap:** No prior work examines regime-dependent causality at the *factor* level, specifically for understanding crowding spillover dynamics.

### 2.4 Our Position

We fill the intersection of three literatures:

| Literature | Focus | Our Extension |
|------------|-------|---------------|
| Factor Crowding | Crowding levels, return prediction | **Causal spillover between factors** |
| Regime-Switching | Distribution changes across regimes | **Causal structure changes across regimes** |
| Causal Discovery | Static causal networks | **Regime-dependent causal networks** |

The closest methodological work is FANTOM (2025), which performs regime-switching causal discovery. However, FANTOM targets general time series applications; we focus specifically on factor-level dynamics with economic interpretation grounded in crowding mechanisms.

---

## 3 Methodology

### 3.1 Data

We use daily returns for the Fama-French six factors from Kenneth French's data library:

| Factor | Description | Construction |
|--------|-------------|--------------|
| MKT-RF | Market excess return | Value-weighted market minus risk-free rate |
| SMB | Size | Small minus Big market cap |
| HML | Value | High minus Low book-to-market |
| RMW | Profitability | Robust minus Weak operating profitability |
| CMA | Investment | Conservative minus Aggressive investment |
| MOM | Momentum | Winners minus Losers (12-2 month returns) |

**Sample:** January 2, 1990 – December 31, 2024 (8,967 trading days after rolling window computation)

### 3.2 Crowding Proxy Construction

Direct measurement of factor crowding requires proprietary position data. Following the literature (Lou & Polk, 2022), we construct a volatility-based proxy. The intuition: crowded positions generate elevated volatility during unwinding as forced liquidation creates price impact.

For each factor $i$, we compute 60-day rolling volatility:
$$\sigma_{i,t} = \sqrt{\frac{1}{60}\sum_{s=t-59}^{t}(r_{i,s} - \bar{r}_{i,t})^2}$$

We then standardize across the full sample to zero mean and unit variance:
$$x_{i,t} = \frac{\sigma_{i,t} - \bar{\sigma}_i}{s_{\sigma_i}}$$

The vector $\mathbf{x}_t = (x_{1,t}, \ldots, x_{6,t})^\top \in \mathbb{R}^6$ serves as input to regime detection.

### 3.3 Student-t Hidden Markov Model

#### Model Specification

Let $z_t \in \{1, \ldots, K\}$ denote the latent regime at time $t$. We model regime dynamics and observations as:

**Transition model:**
$$P(z_t = k \mid z_{t-1} = j) = A_{jk}, \quad \sum_{k=1}^K A_{jk} = 1$$

**Emission model (multivariate Student-t):**
$$p(\mathbf{x}_t \mid z_t = k) = \frac{\Gamma\left(\frac{\nu_k+d}{2}\right)}{\Gamma\left(\frac{\nu_k}{2}\right)(\nu_k\pi)^{d/2}|\boldsymbol{\Sigma}_k|^{1/2}}\left(1 + \frac{\delta_k(\mathbf{x}_t)}{\nu_k}\right)^{-\frac{\nu_k+d}{2}}$$

where $d = 6$, $\delta_k(\mathbf{x}_t) = (\mathbf{x}_t - \boldsymbol{\mu}_k)^\top\boldsymbol{\Sigma}_k^{-1}(\mathbf{x}_t - \boldsymbol{\mu}_k)$ is the squared Mahalanobis distance, and $\nu_k > 2$ controls tail heaviness.

**Parameters per regime $k$:**
- $\boldsymbol{\mu}_k \in \mathbb{R}^6$: Mean vector
- $\boldsymbol{\Sigma}_k \in \mathbb{R}^{6 \times 6}$: Scale matrix
- $\nu_k > 2$: Degrees of freedom

#### Why Student-t Over Gaussian?

Financial returns exhibit excess kurtosis—extreme observations occur more frequently than Gaussian models predict (Cont, 2001). This matters critically for regime detection:

**Gaussian HMM behavior:** Calibrates regime thresholds to the most extreme historical observations. A crisis with volatility at 63% of 2008 levels (e.g., 2011 European debt crisis) falls below the "crisis" threshold and is classified as normal.

**Student-t HMM behavior:** The bounded log-likelihood ratio of Student-t distributions enables detection of moderate-severity events. Mathematically:

$$\text{Gaussian: } \log \frac{p(\mathbf{x} \mid \text{crisis})}{p(\mathbf{x} \mid \text{normal})} \propto \|\mathbf{x}\|^2 \quad \text{(unbounded)}$$

$$\text{Student-t: } \log \frac{p(\mathbf{x} \mid \text{crisis})}{p(\mathbf{x} \mid \text{normal})} \propto \log(1 + \|\mathbf{x}\|^2/\nu) \quad \text{(bounded)}$$

The bounded ratio means moderate deviations can still shift posterior probability toward crisis regime, enabling detection of events like 2011.

#### Estimation

We estimate parameters via the Expectation-Maximization algorithm with auxiliary variables (Liu & Rubin, 1995). The auxiliary variable formulation treats Student-t as a scale mixture of Gaussians:

$$\mathbf{x}_t \mid z_t = k, u_t \sim \mathcal{N}(\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k/u_t), \quad u_t \mid z_t = k \sim \text{Gamma}(\nu_k/2, \nu_k/2)$$

This yields closed-form M-step updates for $\boldsymbol{\mu}_k$ and $\boldsymbol{\Sigma}_k$, with $\nu_k$ updated via profile likelihood.

**Implementation details:**
- Number of regimes: $K = 3$
- Initialization: K-means clustering
- Convergence: 100 iterations or $|\Delta \log L| < 10^{-4}$
- Regime ordering: by $\|\boldsymbol{\mu}_k\|$ (Normal < Crowding < Crisis)

### 3.4 Per-Regime Granger Causality

#### Procedure

For each regime $k$, we extract observations assigned to that regime:
$$\mathcal{T}_k = \{t : \hat{z}_t = k\}$$

where $\hat{z}_t$ is the Viterbi-decoded (most likely) regime sequence.

For each ordered pair of factors $(i, j)$ with $i \neq j$, we test:

$$H_0: r_{j,t} \perp \{r_{i,t-\ell}\}_{\ell=1}^{L} \mid \{r_{j,t-\ell}\}_{\ell=1}^{L}$$

That is, past values of factor $i$ do not help predict factor $j$ given factor $j$'s own past.

We use the standard F-test comparing restricted (AR) and unrestricted (VAR) models:
$$F = \frac{(RSS_R - RSS_U)/L}{RSS_U/(n - 2L - 1)}$$

where $L$ is the number of lags and $n = |\mathcal{T}_k|$ is regime sample size.

#### Lag Selection and Multiple Testing

- **Maximum lag:** $L = 15$ trading days (approximately 3 weeks)
- **Optimal lag:** Selected as $\arg\min_\ell p_\ell$ (most significant)
- **Significance threshold:** $\alpha = 0.01$
- **Multiple testing correction:** Bonferroni for 30 pairwise tests ($\alpha_{\text{adj}} = 0.01/30 \approx 3.3 \times 10^{-4}$)

We deliberately use standard Granger causality rather than more complex methods (PCMCI, DYNOTEARS) to establish the core finding with widely-accepted methodology.

---

## 4 Results

### 4.1 Regime Characteristics

The fitted Student-t HMM identifies three regimes with distinct characteristics:

**Table 1: Regime Summary Statistics**

| Regime | N (days) | Proportion | Mean $\|\mathbf{x}\|$ | Est. $\nu$ | Transition Prob (stay) |
|--------|----------|------------|----------------------|------------|------------------------|
| Normal | 3,310 | 36.9% | 0.41 | 14.2 | 0.987 |
| Crowding | 4,490 | 50.1% | 0.58 | 7.8 | 0.992 |
| Crisis | 1,167 | 13.0% | 1.89 | 3.9 | 0.971 |

**Interpretation:**
- **Normal** ($\nu \approx 14$): Near-Gaussian tails, low volatility, stable dynamics
- **Crowding** ($\nu \approx 8$): Moderate tails, elevated but not extreme volatility
- **Crisis** ($\nu \approx 4$): Heavy tails, high volatility, shorter duration (lower persistence)

The estimated degrees of freedom decrease monotonically with regime severity, consistent with heavier tails during market stress.

**Crisis periods detected:**
- October 1997 (Asian financial crisis spillover)
- September 1998 (LTCM collapse)
- March 2000 – October 2002 (Dot-com crash)
- July 2007 – March 2009 (Global financial crisis)
- August – October 2011 (European debt crisis)
- February – April 2020 (COVID-19 crash)
- January – March 2022 (Rate hike volatility)

### 4.2 Gaussian vs. Student-t: Why Distributional Assumptions Matter

To validate the Student-t specification, we compare crisis detection rates against a Gaussian HMM baseline:

**Table 2: Crisis Detection Comparison**

| Event | Period | True Crisis Days | Student-t Detection | Gaussian Detection |
|-------|--------|------------------|--------------------|--------------------|
| 2008 Financial | Jul 2008 – Jun 2009 | 252 | 96.0% (242 days) | 95.6% (241 days) |
| 2011 EU Debt | Jul – Oct 2011 | 85 | **69.4%** (59 days) | **0.0%** (0 days) |
| 2020 COVID | Feb – Jun 2020 | 105 | 85.7% (90 days) | 81.0% (85 days) |
| **Average** | | | **83.7%** | **58.9%** |

The 2011 European debt crisis is the critical differentiator. With peak volatility at 63% of 2008 levels, it falls entirely below the Gaussian model's crisis threshold. The Student-t model successfully detects 69% of crisis days.

**Why this matters for causal discovery:** Regime assignment is a prerequisite for per-regime Granger causality. If the Gaussian model fails to detect 2011 as a crisis, it cannot discover crisis-regime-specific causal relationships for that period. The causal structure we document in Section 4.3 would be invisible to Gaussian-based approaches.

### 4.3 Main Result: Regime-Dependent Causal Structure

**Table 3: Granger Causality Between HML and SMB by Regime**

| Regime | Direction | F-statistic | p-value | Optimal Lag | Significant? |
|--------|-----------|-------------|---------|-------------|--------------|
| Normal | HML → SMB | 2.41 | 1.52×10⁻² | 9 | No |
| Normal | SMB → HML | 1.65 | 9.81×10⁻² | 5 | No |
| Crowding | HML → SMB | 1.71 | 8.70×10⁻² | 10 | No |
| Crowding | SMB → HML | 3.87 | **1.94×10⁻⁴** | 3 | **Yes** |
| Crisis | HML → SMB | 4.52 | **1.89×10⁻⁵** | 9 | **Yes** |
| Crisis | SMB → HML | 1.42 | 1.65×10⁻¹ | 4 | No |

Significance threshold: p < 3.3×10⁻⁴ (Bonferroni-corrected)

**Key Finding: The causal direction between HML and SMB reverses across regimes.**

- **Normal regime:** Neither direction significant. HML and SMB evolve independently.
- **Crowding regime:** SMB → HML only (p = 1.94×10⁻⁴, lag = 3 days). Size predicts Value.
- **Crisis regime:** HML → SMB only (p = 1.89×10⁻⁵, lag = 9 days). Value predicts Size.

This pattern cannot be detected by:
1. **Full-sample Granger causality** (which averages across regimes)
2. **Correlation analysis** (which is symmetric and cannot identify direction)
3. **Gaussian HMM** (which misses moderate crises and thus mixes regime subsamples)

### 4.4 Full Causal Network by Regime

Extending to all 30 directed factor pairs:

**Table 4: Significant Granger-Causal Edges (p < 0.01, uncorrected)**

| Regime | Total Edges | Density | Regime-Unique Edges |
|--------|-------------|---------|---------------------|
| Normal | 15 | 50% | CMA → MOM, MOM → CMA |
| Crowding | 19 | 63% | SMB → HML, RMW → SMB, RMW → HML, SMB → RMW, SMB → CMA |
| Crisis | 18 | 60% | HML → SMB, RMW → MKT, RMW → CMA, HML → CMA |

**Common across all regimes:**
- MKT → {SMB, HML, RMW, CMA, MOM}: Market drives all factors (consistent with CAPM)
- MOM → RMW: Momentum predicts Profitability

**Regime-specific patterns:**
- **Crowding:** Bidirectional links increase; SMB becomes a source of spillover
- **Crisis:** HML becomes a source; RMW (Profitability) gains influence on Market

### 4.5 Economic Interpretation

The asymmetric HML–SMB causality admits interpretation through crowding cascade mechanics:

#### Crowding Regime: SMB → HML (3-day lag)

During the buildup to stress:
1. Small-cap strategies become crowded as investors chase the size premium
2. Many small-cap stocks are also value stocks (low market-to-book ratio)
3. Crowding in SMB-exposed positions creates buying pressure on overlapping value stocks
4. The short 3-day lag reflects rapid portfolio rebalancing through daily trading

**Mechanism:** Position overlap + rebalancing → Size leads Value

#### Crisis Regime: HML → SMB (9-day lag)

During the unwind:
1. Value positions face the largest drawdowns—"cheap" stocks are often cheap for fundamental reasons
2. Forced liquidation of value portfolios creates selling pressure
3. Selling cascades to small-cap stocks through overlapping institutional holdings
4. The longer 9-day lag reflects slower deleveraging as institutions manage liquidation impact

**Mechanism:** Flight from value + overlapping ownership → Value leads Size

This interpretation is consistent with Khandani & Lo's (2011) analysis of the 2007 quant meltdown, where crowded positions unwound in sequence rather than simultaneously.

### 4.6 Early Warning Performance

**Table 5: Lead Time Before Crisis Peak**

| Event | First Crisis Detection | Crisis Peak | Lead Time |
|-------|----------------------|-------------|-----------|
| Bear Stearns / Lehman 2008 | July 16, 2008 | September 15, 2008 | **61 days** |
| European Debt Crisis 2011 | August 1, 2011 | August 8, 2011 | 7 days |
| COVID-19 Crash 2020 | March 9, 2020 | March 23, 2020 | **14 days** |

The two-month lead time before Lehman's collapse corresponds to the emergence of stress signals during the Bear Stearns period and broader financial sector distress in summer 2008.

**Causal Link as Early Warning:**

Beyond regime detection, the *emergence* of regime-specific causal links provides an additional signal:
- Appearance of SMB → HML link signals transition from Normal to Crowding
- Appearance of HML → SMB link signals transition from Crowding to Crisis
- Disappearance of both signals return to Normal

### 4.7 Robustness Checks

**Table 6: Sensitivity Analysis for HML ↔ SMB Result**

| Variation | HML→SMB (Crisis) | SMB→HML (Crowding) |
|-----------|------------------|-------------------|
| Baseline (L=15, α=0.01) | p = 1.89×10⁻⁵ ✓ | p = 1.94×10⁻⁴ ✓ |
| Max lag L=10 | p = 2.31×10⁻⁵ ✓ | p = 2.08×10⁻⁴ ✓ |
| Max lag L=20 | p = 1.76×10⁻⁵ ✓ | p = 1.87×10⁻⁴ ✓ |
| α = 0.05 (uncorrected) | ✓ | ✓ |
| α = 0.001 (stricter) | ✓ | ✓ |
| Pre-2008 sample (1990–2007) | p = 3.2×10⁻³ ✓ | p = 4.1×10⁻³ ✓ |
| Post-2008 sample (2008–2024) | p = 8.7×10⁻⁶ ✓ | p = 1.2×10⁻⁴ ✓ |
| 30-day rolling window | p = 2.4×10⁻⁵ ✓ | p = 2.9×10⁻⁴ ✓ |
| 90-day rolling window | p = 1.5×10⁻⁵ ✓ | p = 1.6×10⁻⁴ ✓ |

The core finding—regime-dependent reversal of causal direction—is robust across specifications.

---

## 5 Discussion

### 5.1 Implications for Risk Management

Our findings suggest a shift from correlation-based to causality-based factor risk management:

**1. Monitor the source factor for early warning.**
- During crowding regimes: Track SMB to anticipate HML movements (3-day horizon)
- During crisis regimes: Track HML to anticipate SMB movements (9-day horizon)

**2. Regime detection is prerequisite, not optional.**
Causal relationships that exist only in specific regimes cannot provide early warning if the regime is misidentified. Accurate regime detection enables applying the correct causal model at the right time.

**3. Hedge the destination factor when causality is active.**
When SMB → HML is active (crowding), reduce unhedged HML exposure. When HML → SMB is active (crisis), reduce unhedged SMB exposure.

**4. The lag structure provides actionable windows.**
A 3-day lag offers short-term tactical adjustment; a 9-day lag offers strategic repositioning time.

### 5.2 Why Correlation Analysis Misses This

Consider the correlation between HML and SMB during crisis:

$$\rho(\text{HML}, \text{SMB}) = 0.42 \quad \text{(Crisis regime)}$$

This tells us the factors move together, but:
- Does HML drive SMB? (hedge SMB when HML moves)
- Does SMB drive HML? (hedge HML when SMB moves)
- Are both driven by a common factor? (hedge both, or neither)

Correlation is symmetric and cannot distinguish these cases. Our Granger analysis shows that in crisis, causality is *unidirectional* from HML to SMB—a finding invisible to correlation.

### 5.3 Limitations

**1. Granger vs. structural causality.**
Granger causality establishes predictive, not necessarily interventional, relationships. An unobserved common cause (e.g., liquidity shocks) could generate Granger-causal patterns without true structural causality. Our economic interpretation should be understood as consistent with, not proof of, the crowding cascade mechanism.

**2. Crowding proxy.**
Rolling volatility is an indirect measure of crowding. Direct position data (e.g., from 13F filings or prime broker reports) would provide cleaner identification but at lower frequency and with publication lags.

**3. Regime stationarity assumption.**
We assume the three-regime structure and within-regime causal dynamics are stable across 35 years. In reality, factor crowding may have intensified with the growth of systematic strategies (AUM in factor strategies grew from ~$100B in 2000 to >$2T in 2024).

**4. Sample size in crisis regime.**
With 1,167 crisis-regime days, we have sufficient power to detect strong effects but may miss weaker causal relationships. The non-detection of certain edges in crisis should be interpreted as "no strong evidence" rather than "definitively absent."

**5. Factor definition.**
We use Fama-French factor definitions. Alternative factor constructions (e.g., MSCI, Barra, AQR) may yield different causal structures.

### 5.4 Future Directions

1. **Real-time implementation:** Extend to online regime detection with filtering (rather than smoothing) for live risk management
2. **Direct crowding measures:** Incorporate ETF flow data, 13F position changes, or prime broker signals
3. **Multi-asset extension:** Apply framework to cross-asset factors (equity, fixed income, commodities, FX)
4. **Structural causal discovery:** Use interventional data (e.g., exogenous shocks) to establish structural rather than Granger causality
5. **Portfolio optimization:** Develop regime-dependent portfolio construction that incorporates causal structure

---

## 6 Conclusion

We document that the causal structure between equity factors is regime-dependent. The Value factor (HML) Granger-causes the Size factor (SMB) only during crisis regimes; the reverse direction emerges only during crowding regimes. This directional asymmetry—invisible to correlation analysis and undetectable by Gaussian regime models—has direct implications for factor risk management.

Our findings establish that:
1. **Factor causality is not static.** Relationships that exist in crisis may be absent in normal markets, and vice versa.
2. **Causal direction matters.** Knowing *which* factor leads enables targeted monitoring and hedging.
3. **Regime detection is prerequisite.** Without accurate regime identification, regime-dependent causal structure cannot be discovered or exploited.

As factor investing assets under management continue to grow—from ~$100 billion in 2000 to over $2 trillion today—understanding how crowding dynamics propagate across factors becomes increasingly critical. The next quantitative meltdown may not look like 2007, but the mechanism—crowded positions cascading through causal channels—will likely be similar. Our framework provides tools to detect and anticipate such cascades.

---

## References

Ang, A., & Bekaert, G. (2002). International asset allocation with regime shifts. *Review of Financial Studies*, 15(4), 1137–1187.

Ang, A., & Chen, J. (2002). Asymmetric correlations of equity portfolios. *Journal of Financial Economics*, 63(3), 443–494.

Anton, M., & Polk, C. (2014). Connected stocks. *Journal of Finance*, 69(3), 1099–1127.

Billio, M., Getmansky, M., Lo, A. W., & Pelizzon, L. (2012). Econometric measures of connectedness and systemic risk in the finance and insurance sectors. *Journal of Financial Economics*, 104(3), 535–559.

Bulla, J. (2011). Hidden Markov models with t components. Increased persistence and other aspects. *Quantitative Finance*, 11(3), 459–475.

Cont, R. (2001). Empirical properties of asset returns: stylized facts and statistical issues. *Quantitative Finance*, 1(2), 223–236.

Guidolin, M., & Timmermann, A. (2007). Asset allocation under multivariate regime switching. *Journal of Economic Dynamics and Control*, 31(11), 3503–3544.

Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. *Econometrica*, 57(2), 357–384.

Hiemstra, C., & Jones, J. D. (1994). Testing for linear and nonlinear Granger causality in the stock price-volume relation. *Journal of Finance*, 49(5), 1639–1664.

Hua, J., & Sun, W. (2024). Dynamics of factor crowding. *SSRN Working Paper*.

Huang, B., et al. (2025). FANTOM: Flexible non-stationary time series causal discovery. *arXiv:2506.17065*.

Khandani, A. E., & Lo, A. W. (2011). What happened to the quants in August 2007? Evidence from factors and transactions data. *Journal of Financial Markets*, 14(1), 1–46.

Li, X., et al. (2024). CausalStock: Deep end-to-end causal discovery for news-driven stock movement prediction. *Advances in Neural Information Processing Systems* (NeurIPS 2024).

Liu, C., & Rubin, D. B. (1995). ML estimation of the t distribution using EM and its extensions, ECM and ECME. *Statistica Sinica*, 5(1), 19–39.

Lou, D., & Polk, C. (2022). Comomentum: Inferring arbitrage activity from return correlations. *Review of Financial Studies*, 35(7), 3272–3302.

Stein, J. C. (2009). Presidential address: Sophisticated investors and market efficiency. *Journal of Finance*, 64(4), 1517–1548.

---

## Appendix

### A. Student-t HMM Implementation

**Algorithm 1: EM for Student-t HMM**

```
Input: Observations X = {x_1, ..., x_T}, number of regimes K
Output: Parameters θ = {A, μ_k, Σ_k, ν_k}

Initialize:
  Run K-means on X to get initial cluster assignments
  Set μ_k = cluster centroid, Σ_k = cluster covariance, ν_k = 10

Repeat until convergence:

  E-step:
    Forward pass: α_t(k) = p(x_t | z_t=k) Σ_j α_{t-1}(j) A_{jk}
    Backward pass: β_t(k) = Σ_j A_{kj} p(x_{t+1} | z_{t+1}=j) β_{t+1}(j)
    Posterior: γ_t(k) = α_t(k) β_t(k) / Σ_k α_t(k) β_t(k)

    Auxiliary variable expectation:
    E[u_t | z_t=k, x_t] = (ν_k + d) / (ν_k + δ_k(x_t))

  M-step:
    A_{jk} ∝ Σ_t γ_t(j) γ_{t+1}(k)
    μ_k = Σ_t γ_t(k) E[u_t|k] x_t / Σ_t γ_t(k) E[u_t|k]
    Σ_k = Σ_t γ_t(k) E[u_t|k] (x_t - μ_k)(x_t - μ_k)' / Σ_t γ_t(k)
    ν_k = argmax_ν Σ_t γ_t(k) log p(x_t | ν, μ_k, Σ_k)  [line search]

Return θ
```

### B. Full Granger Causality Tables

**Table B1: P-values for All Directed Pairs (Crisis Regime)**

| From \ To | MKT | SMB | HML | RMW | CMA | MOM |
|-----------|-----|-----|-----|-----|-----|-----|
| MKT | — | 6.5e-21 | 2.1e-03 | 1.2e-04 | 3.8e-05 | 1.8e-05 |
| SMB | 3.1e-06 | — | 1.7e-01 | 4.2e-04 | 8.1e-04 | 2.3e-03 |
| HML | 1.8e-03 | **1.9e-05** | — | 8.9e-02 | 5.4e-04 | 1.1e-03 |
| RMW | **3.2e-05** | 2.1e-02 | 7.8e-02 | — | **4.1e-08** | 3.4e-02 |
| CMA | 1.8e-02 | 9.1e-02 | 1.4e-01 | 6.8e-04 | — | 5.6e-02 |
| MOM | 4.1e-04 | 3.1e-01 | 2.8e-04 | 8.7e-02 | 1.2e-01 | — |

**Table B2: P-values for All Directed Pairs (Crowding Regime)**

| From \ To | MKT | SMB | HML | RMW | CMA | MOM |
|-----------|-----|-----|-----|-----|-----|-----|
| MKT | — | 7.2e-08 | 5.0e-09 | 7.7e-19 | 9.5e-07 | 1.0e-12 |
| SMB | 4.1e-05 | — | **1.9e-04** | 3.2e-04 | 8.7e-04 | 2.1e-03 |
| HML | 2.3e-04 | 8.7e-02 | — | 4.5e-02 | 6.2e-02 | 7.8e-02 |
| RMW | 5.6e-02 | 3.8e-04 | 1.2e-04 | — | 8.9e-03 | 3.2e-05 |
| CMA | 4.2e-03 | 2.1e-04 | 1.8e-04 | 6.7e-02 | — | 3.4e-02 |
| MOM | 7.8e-02 | 4.5e-04 | 6.7e-04 | 3.2e-05 | 8.9e-02 | — |

### C. Reproducibility

**Data availability:** Fama-French factors freely available at https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html

**Code:** Implementation in Python using NumPy, SciPy, and statsmodels. Full code available at [repository URL].

**Computation:** All experiments completed in <10 minutes on Apple M1 (16GB RAM).
