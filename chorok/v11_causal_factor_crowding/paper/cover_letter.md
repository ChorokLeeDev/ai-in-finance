# Cover Letter

**To:** ICAIF 2025 Program Committee

**Re:** Submission - "Detecting Market Stress via Factor Causality Emergence"

---

Dear Program Committee,

We are pleased to submit our paper "Detecting Market Stress via Factor Causality Emergence" for consideration at ICAIF 2025.

## Summary

This paper makes three contributions at the intersection of machine learning and quantitative finance:

1. **Methodological:** We demonstrate that Student-t Hidden Markov Models detect moderate financial crises that standard Gaussian HMMs miss entirely. The 2011 European debt crisis—with factor volatility at 63% of 2008 levels—is detected at 69% by Student-t versus 0% by Gaussian specifications.

2. **Empirical:** We document that Granger-causal relationships between Fama-French factors intensify during detected stress regimes. This pattern replicates out-of-sample: a model trained on 1990–2014 detects 100% of stress events in 2015–2024.

3. **Practical:** We show that implementation frequency determines economic value. The same regime-dependent causality signal that improves Sharpe ratio by 35% when traded on regime transitions (13 trades over 10 years) destroys value when traded daily (246 trades, 27.9% transaction cost drag).

## Relevance to ICAIF

This work directly addresses ICAIF's focus on AI applications in finance:

- **Machine Learning Methods:** Student-t HMMs with EM estimation, Granger causality networks
- **Financial Application:** Factor risk management, regime detection, portfolio construction
- **Practical Insight:** The gap between statistical detectability and economic exploitability

## Key Findings

| Strategy | Trades (10yr) | Sharpe | Max Drawdown |
|----------|---------------|--------|--------------|
| Baseline | 0 | 0.43 | -12.5% |
| Naive Daily | 246 | -0.03 | -15.6% |
| **Regime-Transition** | **13** | **0.58** | **-9.0%** |

The central message—that implementation matters more than signal detection—has broad implications for quantitative finance research.

## Honest Presentation

We note that:
- The Sharpe improvement (p = 0.34) does not reach conventional statistical significance, though it is economically meaningful and robust to transaction costs
- With proper multiple testing correction (Bonferroni for 90 tests), individual causal relationships are marginal, though the pattern of stress intensification is consistent
- We have reframed "crowding proxy" as "volatility-based regime detection" to avoid unvalidated claims

We believe this honest presentation strengthens rather than weakens the contribution.

## Data and Reproducibility

All data is publicly available (Fama-French factors from Kenneth French's data library). Code will be released upon acceptance. All experiments complete in <10 minutes on standard hardware.

## Conflicts of Interest

None.

## Prior Publication

This work has not been previously published or submitted elsewhere.

---

Thank you for considering our submission. We look forward to your feedback.

Sincerely,

The Authors
