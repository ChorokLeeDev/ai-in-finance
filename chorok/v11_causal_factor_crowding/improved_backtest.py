"""
Improved Backtest with Transaction Costs and Bootstrap CIs
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from gate2_regime_detection import StudentTHMM, load_and_prepare_data


def bootstrap_sharpe_ci(returns, n_bootstrap=10000, ci=0.95):
    """Bootstrap confidence interval for Sharpe ratio."""
    n = len(returns)
    sharpes = []

    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        sample = returns.iloc[idx]
        annual_ret = sample.mean() * 252
        annual_vol = sample.std() * np.sqrt(252)
        sharpe = annual_ret / annual_vol if annual_vol > 0 else 0
        sharpes.append(sharpe)

    sharpes = np.array(sharpes)
    alpha = (1 - ci) / 2
    return np.mean(sharpes), np.percentile(sharpes, alpha * 100), np.percentile(sharpes, (1 - alpha) * 100)


def sharpe_difference_test(returns1, returns2, n_bootstrap=10000):
    """Bootstrap test for Sharpe ratio difference."""
    n = len(returns1)
    diff_sharpes = []

    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        s1 = returns1.iloc[idx]
        s2 = returns2.iloc[idx]

        sharpe1 = s1.mean() * 252 / (s1.std() * np.sqrt(252)) if s1.std() > 0 else 0
        sharpe2 = s2.mean() * 252 / (s2.std() * np.sqrt(252)) if s2.std() > 0 else 0

        diff_sharpes.append(sharpe2 - sharpe1)

    diff_sharpes = np.array(diff_sharpes)

    observed_diff = returns2.mean() * 252 / (returns2.std() * np.sqrt(252)) - \
                    returns1.mean() * 252 / (returns1.std() * np.sqrt(252))

    if observed_diff > 0:
        p_value = (diff_sharpes <= 0).mean() * 2
    else:
        p_value = (diff_sharpes >= 0).mean() * 2

    return observed_diff, np.std(diff_sharpes), min(p_value, 1.0)


def run_improved_backtest(test_start='2015-01-01', test_end='2024-12-31',
                          transaction_cost_bps=10):
    """Run improved backtest with transaction costs and bootstrap CIs."""

    import pandas_datareader.data as web

    print("=" * 70)
    print("IMPROVED BACKTEST WITH TRANSACTION COSTS")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    ff5 = web.DataReader('F-F_Research_Data_5_Factors_2x3_daily',
                         'famafrench', start='1990-01-01')[0]
    mom = web.DataReader('F-F_Momentum_Factor_daily',
                         'famafrench', start='1990-01-01')[0]

    returns_df = ff5.join(mom)
    returns_df.columns = ['MKT', 'SMB', 'HML', 'RMW', 'CMA', 'RF', 'MOM']
    returns_df = returns_df / 100

    # Load crowding proxy
    crowding = load_and_prepare_data()
    common_dates = returns_df.index.intersection(crowding.index)
    returns = returns_df.loc[common_dates].copy()
    crowding = crowding.loc[common_dates].copy()

    # Train HMM
    train_end = '2014-12-31'
    train_mask = crowding.index <= train_end

    print(f"\nFitting regime model on training data (until {train_end})...")
    hmm = StudentTHMM(n_regimes=3, n_iter=100)
    hmm.fit(crowding[train_mask].values)

    # Get regimes
    regimes = hmm.predict(crowding.values)

    # Identify regime labels
    train_regimes = regimes[np.array(train_mask)]
    vol_by_regime = []
    for k in range(3):
        regime_data = crowding[train_mask].values[train_regimes == k]
        vol_by_regime.append(np.std(regime_data))

    crisis_regime = np.argmax(vol_by_regime)
    normal_regime = np.argmin(vol_by_regime)
    crowding_regime = 3 - crisis_regime - normal_regime

    returns['regime'] = regimes

    # Test period
    test_mask = (returns.index >= test_start) & (returns.index <= test_end)
    test_returns = returns[test_mask].copy()

    print(f"\nTest period: {test_start} to {test_end}")
    print(f"Test days: {len(test_returns)}")

    factors = ['MKT', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']
    n_factors = len(factors)
    tc = transaction_cost_bps / 10000

    # Strategy 1: Baseline
    baseline_weights = pd.DataFrame(1.0/n_factors, index=test_returns.index, columns=factors)
    baseline_returns = (test_returns[factors] * baseline_weights).sum(axis=1)

    # Strategy 2: Lead-Lag
    leadlag_weights = pd.DataFrame(1.0/n_factors, index=test_returns.index, columns=factors)
    prev_weights = pd.Series(1.0/n_factors, index=factors)

    crisis_lag = 5
    crowding_lag = 3
    hml_lagged = test_returns['HML'].rolling(crisis_lag).sum().shift(1)
    smb_lagged = test_returns['SMB'].rolling(crowding_lag).sum().shift(1)
    hml_threshold = hml_lagged.quantile(0.10)
    smb_threshold = smb_lagged.quantile(0.10)

    turnover = []

    for i, (date, row) in enumerate(test_returns.iterrows()):
        regime = row['regime']
        new_weights = pd.Series(1.0/n_factors, index=factors)

        if regime == crisis_regime and i > crisis_lag:
            if pd.notna(hml_lagged.iloc[i]) and hml_lagged.iloc[i] < hml_threshold:
                new_weights['SMB'] = 0.3/n_factors
                new_weights['HML'] = 0.3/n_factors
                extra = (1.0 - 0.6/n_factors) / 4
                for f in ['MKT', 'RMW', 'CMA', 'MOM']:
                    new_weights[f] = 1.0/n_factors + extra

        elif regime == crowding_regime and i > crowding_lag:
            if pd.notna(smb_lagged.iloc[i]) and smb_lagged.iloc[i] < smb_threshold:
                new_weights['HML'] = 0.3/n_factors
                new_weights['SMB'] = 0.3/n_factors
                extra = (1.0 - 0.6/n_factors) / 4
                for f in ['MKT', 'RMW', 'CMA', 'MOM']:
                    new_weights[f] = 1.0/n_factors + extra

        weight_change = (new_weights - prev_weights).abs().sum()
        turnover.append(weight_change)
        prev_weights = new_weights.copy()
        leadlag_weights.loc[date] = new_weights

    # Gross and net returns
    leadlag_returns_gross = (test_returns[factors] * leadlag_weights).sum(axis=1)
    turnover_series = pd.Series(turnover, index=test_returns.index)
    leadlag_returns_net = leadlag_returns_gross - turnover_series * tc

    # Compute metrics
    def compute_metrics(returns, name):
        n_years = len(returns) / 252
        annual_ret = ((1 + returns).prod() ** (1/n_years) - 1)
        annual_vol = returns.std() * np.sqrt(252)
        sharpe = annual_ret / annual_vol if annual_vol > 0 else 0

        cumulative = (1 + returns).cumprod()
        max_dd = ((cumulative - cumulative.cummax()) / cumulative.cummax()).min()
        calmar = annual_ret / abs(max_dd) if max_dd != 0 else 0

        return {
            'name': name,
            'annual_return': annual_ret * 100,
            'annual_vol': annual_vol * 100,
            'sharpe': sharpe,
            'max_drawdown': max_dd * 100,
            'calmar': calmar
        }

    baseline_m = compute_metrics(baseline_returns, 'Baseline')
    leadlag_gross_m = compute_metrics(leadlag_returns_gross, 'Lead-Lag (Gross)')
    leadlag_net_m = compute_metrics(leadlag_returns_net, f'Lead-Lag (Net, {transaction_cost_bps}bps)')

    # Bootstrap CIs
    print("\nComputing bootstrap confidence intervals...")
    np.random.seed(42)
    baseline_sharpe, baseline_ci_low, baseline_ci_high = bootstrap_sharpe_ci(baseline_returns, n_bootstrap=5000)
    leadlag_sharpe, leadlag_ci_low, leadlag_ci_high = bootstrap_sharpe_ci(leadlag_returns_net, n_bootstrap=5000)

    # Significance test
    diff, diff_se, p_value = sharpe_difference_test(baseline_returns, leadlag_returns_net, n_bootstrap=5000)

    # Print results
    print("\n" + "-" * 70)
    print("BACKTEST RESULTS")
    print("-" * 70)
    print(f"\n{'Strategy':<35} {'Return':>10} {'Vol':>10} {'Sharpe':>10} {'MaxDD':>10}")
    print("-" * 70)

    for m in [baseline_m, leadlag_gross_m, leadlag_net_m]:
        print(f"{m['name']:<35} {m['annual_return']:>9.1f}% {m['annual_vol']:>9.1f}% {m['sharpe']:>10.2f} {m['max_drawdown']:>9.1f}%")

    print("\n" + "-" * 70)
    print("BOOTSTRAP CONFIDENCE INTERVALS (95%)")
    print("-" * 70)
    print(f"Baseline Sharpe:  {baseline_sharpe:.3f} [{baseline_ci_low:.3f}, {baseline_ci_high:.3f}]")
    print(f"Lead-Lag Sharpe:  {leadlag_sharpe:.3f} [{leadlag_ci_low:.3f}, {leadlag_ci_high:.3f}]")

    print(f"\nSharpe Difference: {diff:.3f} (SE: {diff_se:.3f})")
    print(f"p-value (H0: no difference): {p_value:.3f}")

    if p_value < 0.05:
        print("=> Sharpe difference is STATISTICALLY SIGNIFICANT at 5% level")
    else:
        print("=> Sharpe difference is NOT statistically significant at 5% level")

    # Turnover analysis
    print("\n" + "-" * 70)
    print("TURNOVER ANALYSIS")
    print("-" * 70)
    print(f"Total turnover:        {turnover_series.sum():.1f}x")
    print(f"Avg daily turnover:    {turnover_series.mean()*100:.2f}%")
    print(f"Trading days:          {(turnover_series > 0.01).sum()}")
    print(f"Transaction cost drag: {(turnover_series * tc).sum() * 100:.2f}% total")

    # Regime distribution
    print("\n" + "-" * 70)
    print("REGIME DISTRIBUTION")
    print("-" * 70)
    crisis_days = (test_returns['regime'] == crisis_regime).sum()
    crowding_days = (test_returns['regime'] == crowding_regime).sum()
    normal_days = (test_returns['regime'] == normal_regime).sum()
    print(f"Normal:   {normal_days} days ({normal_days/len(test_returns)*100:.1f}%)")
    print(f"Crowding: {crowding_days} days ({crowding_days/len(test_returns)*100:.1f}%)")
    print(f"Crisis:   {crisis_days} days ({crisis_days/len(test_returns)*100:.1f}%)")

    # Paper-ready table
    print("\n" + "=" * 70)
    print("PAPER TABLE (Markdown)")
    print("=" * 70)
    print("""
### Table 8 (REVISED): Backtest with Transaction Costs

| Strategy | Return | Sharpe | 95% CI | Max DD |
|----------|--------|--------|--------|--------|""")
    print(f"| Baseline | {baseline_m['annual_return']:.1f}% | {baseline_m['sharpe']:.2f} | [{baseline_ci_low:.2f}, {baseline_ci_high:.2f}] | {baseline_m['max_drawdown']:.1f}% |")
    print(f"| Lead-Lag (net) | {leadlag_net_m['annual_return']:.1f}% | {leadlag_net_m['sharpe']:.2f} | [{leadlag_ci_low:.2f}, {leadlag_ci_high:.2f}] | {leadlag_net_m['max_drawdown']:.1f}% |")
    print(f"""
**Sharpe Difference Test:**
- Difference: {diff:.3f} (SE: {diff_se:.3f})
- p-value: {p_value:.3f}
- Transaction costs: {transaction_cost_bps} bps one-way
""")

    return {
        'baseline_m': baseline_m,
        'leadlag_net_m': leadlag_net_m,
        'diff': diff,
        'p_value': p_value,
        'baseline_ci': (baseline_ci_low, baseline_ci_high),
        'leadlag_ci': (leadlag_ci_low, leadlag_ci_high)
    }


if __name__ == "__main__":
    results = run_improved_backtest()
