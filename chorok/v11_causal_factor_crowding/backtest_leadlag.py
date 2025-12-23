"""
Simple Backtest: P&L Impact of Lead-Lag Relationships

Strategy:
- Baseline: Equal-weight factor portfolio (long all factors)
- Signal: Use regime-dependent lead-lag to reduce exposure to "destination" factor
  when "source" factor shows stress signal

Key insight from OOS validation:
- Crisis: HML→SMB (both directions significant, HML leads)
- Crowding: SMB→HML (both directions significant, SMB leads)
- Normal: Independent (no action)

This backtest demonstrates the ECONOMIC VALUE of regime-dependent lead-lag discovery.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from gate2_regime_detection import StudentTHMM, load_and_prepare_data


def load_factor_returns():
    """Load raw factor returns (not the crowding proxy)."""
    import pandas_datareader.data as web

    print("Loading Fama-French factor returns...")

    # Get Fama-French 5 factors
    ff5 = web.DataReader('F-F_Research_Data_5_Factors_2x3_daily',
                         'famafrench', start='1990-01-01')[0]

    # Get Momentum factor
    mom = web.DataReader('F-F_Momentum_Factor_daily',
                         'famafrench', start='1990-01-01')[0]

    # Combine
    df = ff5.join(mom)
    df.columns = ['MKT', 'SMB', 'HML', 'RMW', 'CMA', 'RF', 'MOM']

    # Convert from percentage to decimal
    df = df / 100

    print(f"Loaded {len(df)} days of returns")

    return df


def compute_signal(returns, factor, window=5, threshold=1.5):
    """
    Compute stress signal for a factor.

    Signal = 1 if factor had significant negative return in recent window
    (below -threshold * rolling std)
    """
    rolling_mean = returns[factor].rolling(60).mean()
    rolling_std = returns[factor].rolling(60).std()

    # Z-score of recent return
    recent_return = returns[factor].rolling(window).sum()
    z_score = (recent_return - rolling_mean * window) / (rolling_std * np.sqrt(window))

    # Stress signal: large negative z-score
    signal = (z_score < -threshold).astype(int)

    return signal


def run_backtest(test_start='2015-01-01', test_end='2024-12-31'):
    """
    Run the lead-lag backtest.

    Strategies:
    1. Baseline: Equal-weight all factors
    2. Lead-Lag: Reduce destination factor when source shows stress
    """
    print("=" * 70)
    print("LEAD-LAG BACKTEST")
    print("=" * 70)

    # =========================================================================
    # Step 1: Load data and fit regime model
    # =========================================================================
    print("\n[1] Loading data...")

    # Load returns
    returns_df = load_factor_returns()

    # Load crowding proxy for regime detection
    crowding = load_and_prepare_data()

    # Align dates
    common_dates = returns_df.index.intersection(crowding.index)
    returns = returns_df.loc[common_dates].copy()
    crowding = crowding.loc[common_dates].copy()

    # Train/test split for regime model
    train_end = '2014-12-31'
    train_mask = crowding.index <= train_end

    print(f"\n[2] Fitting regime model on training data (until {train_end})...")

    # Fit HMM on training data only
    hmm = StudentTHMM(n_regimes=3, n_iter=100)
    hmm.fit(crowding[train_mask].values)

    # Get regimes for full period
    regimes = hmm.predict(crowding.values)

    # Identify regime labels
    train_regimes = regimes[train_mask]
    vol_by_regime = []
    for k in range(3):
        regime_data = crowding[train_mask].values[train_regimes == k]
        vol = np.std(regime_data)
        vol_by_regime.append(vol)

    crisis_regime = np.argmax(vol_by_regime)
    normal_regime = np.argmin(vol_by_regime)
    crowding_regime = 3 - crisis_regime - normal_regime

    regime_names = {
        normal_regime: 'Normal',
        crowding_regime: 'Crowding',
        crisis_regime: 'Crisis'
    }

    # Add regime to returns dataframe
    returns['regime'] = regimes

    print(f"  Regimes identified: {regime_names}")

    # =========================================================================
    # Step 2: Define strategies
    # =========================================================================
    print("\n[3] Running backtest strategies...")

    # Filter to test period
    test_mask = (returns.index >= test_start) & (returns.index <= test_end)
    test_returns = returns[test_mask].copy()

    print(f"  Test period: {test_start} to {test_end}")
    print(f"  Test days: {len(test_returns)}")

    # Factor list (excluding RF)
    factors = ['MKT', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']

    # Compute stress signals
    hml_stress = compute_signal(test_returns, 'HML', window=3, threshold=1.5)
    smb_stress = compute_signal(test_returns, 'SMB', window=3, threshold=1.5)

    # =========================================================================
    # Strategy 1: Baseline (Equal Weight)
    # =========================================================================
    baseline_weights = pd.DataFrame(index=test_returns.index, columns=factors)
    baseline_weights[:] = 1.0 / len(factors)

    baseline_returns = (test_returns[factors] * baseline_weights).sum(axis=1)

    # =========================================================================
    # Strategy 2: Lead-Lag (Regime-Aware) - Improved
    # Uses cumulative lag-period return of source to predict destination
    # =========================================================================
    leadlag_weights = pd.DataFrame(index=test_returns.index, columns=factors)
    leadlag_weights[:] = 1.0 / len(factors)  # Start with equal weight

    # Key lags from OOS validation:
    # Crisis: HML->SMB with ~2-9 day lag
    # Crowding: SMB->HML with ~3-7 day lag
    crisis_lag = 5  # Look at past 5 days of HML
    crowding_lag = 3  # Look at past 3 days of SMB

    # Compute lagged cumulative returns
    hml_lagged = test_returns['HML'].rolling(crisis_lag).sum().shift(1)
    smb_lagged = test_returns['SMB'].rolling(crowding_lag).sum().shift(1)

    # Thresholds (negative = stress)
    hml_threshold = hml_lagged.quantile(0.10)  # Bottom 10%
    smb_threshold = smb_lagged.quantile(0.10)  # Bottom 10%

    # Apply lead-lag rules
    for i, (date, row) in enumerate(test_returns.iterrows()):
        regime = row['regime']
        base_weight = 1.0 / len(factors)

        if regime == crisis_regime and i > crisis_lag:
            # Crisis: HML leads SMB
            # If HML had bad recent performance, reduce SMB
            if hml_lagged.iloc[i] < hml_threshold:
                leadlag_weights.loc[date, 'SMB'] = base_weight * 0.3  # Reduce to 30%
                leadlag_weights.loc[date, 'HML'] = base_weight * 0.3  # Also reduce HML
                # Redistribute to safer factors
                extra = base_weight * 1.4 / 4
                for f in ['MKT', 'RMW', 'CMA', 'MOM']:
                    leadlag_weights.loc[date, f] = base_weight + extra

        elif regime == crowding_regime and i > crowding_lag:
            # Crowding: SMB leads HML
            # If SMB had bad recent performance, reduce HML
            if smb_lagged.iloc[i] < smb_threshold:
                leadlag_weights.loc[date, 'HML'] = base_weight * 0.3  # Reduce to 30%
                leadlag_weights.loc[date, 'SMB'] = base_weight * 0.3  # Also reduce SMB
                # Redistribute
                extra = base_weight * 1.4 / 4
                for f in ['MKT', 'RMW', 'CMA', 'MOM']:
                    leadlag_weights.loc[date, f] = base_weight + extra

        # Normal regime: keep equal weight (no action)

    leadlag_returns = (test_returns[factors] * leadlag_weights).sum(axis=1)

    # =========================================================================
    # Strategy 3: Always Defensive (reduce HML and SMB always)
    # =========================================================================
    defensive_weights = pd.DataFrame(index=test_returns.index, columns=factors)
    for f in factors:
        if f in ['HML', 'SMB']:
            defensive_weights[f] = 0.5 / len(factors)
        else:
            defensive_weights[f] = 1.25 / len(factors)

    defensive_returns = (test_returns[factors] * defensive_weights).sum(axis=1)

    # =========================================================================
    # Step 3: Compute performance metrics
    # =========================================================================
    print("\n[4] Computing performance metrics...")

    def compute_metrics(returns, name):
        """Compute key performance metrics."""
        cumulative = (1 + returns).cumprod()
        total_return = cumulative.iloc[-1] - 1

        # Annualized metrics (252 trading days)
        n_years = len(returns) / 252
        annual_return = (1 + total_return) ** (1 / n_years) - 1
        annual_vol = returns.std() * np.sqrt(252)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0

        # Drawdown
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # Calmar ratio
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        return {
            'name': name,
            'total_return': total_return * 100,
            'annual_return': annual_return * 100,
            'annual_vol': annual_vol * 100,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown * 100,
            'calmar': calmar
        }

    metrics = [
        compute_metrics(baseline_returns, 'Baseline (Equal Weight)'),
        compute_metrics(leadlag_returns, 'Lead-Lag Strategy'),
        compute_metrics(defensive_returns, 'Always Defensive'),
    ]

    # =========================================================================
    # Step 4: Print results
    # =========================================================================
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)

    print(f"\nTest Period: {test_start} to {test_returns.index[-1].strftime('%Y-%m-%d')}")
    print(f"Trading Days: {len(test_returns)}")

    print("\n" + "-" * 70)
    print(f"{'Strategy':<25} {'Return':>10} {'Vol':>10} {'Sharpe':>10} {'MaxDD':>10} {'Calmar':>10}")
    print("-" * 70)

    for m in metrics:
        print(f"{m['name']:<25} {m['annual_return']:>9.1f}% {m['annual_vol']:>9.1f}% {m['sharpe']:>10.2f} {m['max_drawdown']:>9.1f}% {m['calmar']:>10.2f}")

    # =========================================================================
    # Step 5: Improvement analysis
    # =========================================================================
    print("\n" + "-" * 70)
    print("IMPROVEMENT vs BASELINE")
    print("-" * 70)

    baseline_m = metrics[0]
    for m in metrics[1:]:
        sharpe_imp = (m['sharpe'] - baseline_m['sharpe']) / baseline_m['sharpe'] * 100 if baseline_m['sharpe'] != 0 else 0
        dd_imp = (baseline_m['max_drawdown'] - m['max_drawdown']) / abs(baseline_m['max_drawdown']) * 100 if baseline_m['max_drawdown'] != 0 else 0

        print(f"\n{m['name']}:")
        print(f"  Sharpe improvement: {sharpe_imp:+.1f}%")
        print(f"  Drawdown reduction: {dd_imp:+.1f}%")

    # =========================================================================
    # Step 6: Signal analysis
    # =========================================================================
    print("\n" + "-" * 70)
    print("SIGNAL ANALYSIS")
    print("-" * 70)

    # Count action days
    n_crisis_actions = ((leadlag_weights['SMB'] < 1.0/len(factors)) & (test_returns['regime'] == crisis_regime)).sum()
    n_crowding_actions = ((leadlag_weights['HML'] < 1.0/len(factors)) & (test_returns['regime'] == crowding_regime)).sum()

    # Count signals by regime
    crisis_days = (test_returns['regime'] == crisis_regime).sum()
    crowding_days = (test_returns['regime'] == crowding_regime).sum()
    normal_days = (test_returns['regime'] == normal_regime).sum()

    print(f"  Crisis regime days: {crisis_days} ({crisis_days/len(test_returns)*100:.1f}%)")
    print(f"  Crowding regime days: {crowding_days} ({crowding_days/len(test_returns)*100:.1f}%)")
    print(f"  Normal regime days: {normal_days} ({normal_days/len(test_returns)*100:.1f}%)")
    print(f"  Actions in Crisis: {n_crisis_actions} days")
    print(f"  Actions in Crowding: {n_crowding_actions} days")

    # =========================================================================
    # Return data for plotting
    # =========================================================================
    return {
        'test_returns': test_returns,
        'baseline_returns': baseline_returns,
        'leadlag_returns': leadlag_returns,
        'defensive_returns': defensive_returns,
        'metrics': metrics,
        'regime_names': regime_names,
        'crisis_regime': crisis_regime,
        'crowding_regime': crowding_regime,
    }


def create_backtest_figure(results, save_path=None):
    """Create backtest performance figure."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # =========================================================================
    # Panel A: Cumulative Returns
    # =========================================================================
    ax = axes[0, 0]

    baseline_cum = (1 + results['baseline_returns']).cumprod()
    leadlag_cum = (1 + results['leadlag_returns']).cumprod()
    defensive_cum = (1 + results['defensive_returns']).cumprod()

    ax.plot(baseline_cum.index, baseline_cum.values, 'b-', label='Baseline (Equal Weight)', linewidth=1.5)
    ax.plot(leadlag_cum.index, leadlag_cum.values, 'g-', label='Lead-Lag Strategy', linewidth=1.5)
    ax.plot(defensive_cum.index, defensive_cum.values, 'gray', linestyle='--', label='Always Defensive', linewidth=1, alpha=0.7)

    ax.set_ylabel('Cumulative Return')
    ax.set_title('(A) Cumulative Performance', fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # =========================================================================
    # Panel B: Drawdown
    # =========================================================================
    ax = axes[0, 1]

    def compute_drawdown(returns):
        cum = (1 + returns).cumprod()
        rolling_max = cum.cummax()
        dd = (cum - rolling_max) / rolling_max * 100
        return dd.astype(float)  # Ensure float type

    baseline_dd = compute_drawdown(results['baseline_returns'])
    leadlag_dd = compute_drawdown(results['leadlag_returns'])

    ax.fill_between(baseline_dd.index, baseline_dd.values.astype(float), 0, alpha=0.3, color='blue', label='Baseline')
    ax.fill_between(leadlag_dd.index, leadlag_dd.values.astype(float), 0, alpha=0.3, color='green', label='Lead-Lag')

    ax.set_ylabel('Drawdown (%)')
    ax.set_title('(B) Drawdown Comparison', fontweight='bold')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)

    # =========================================================================
    # Panel C: Performance Metrics Bar Chart
    # =========================================================================
    ax = axes[1, 0]

    strategies = ['Baseline', 'Lead-Lag']
    metrics_to_plot = ['sharpe', 'calmar']
    x = np.arange(len(metrics_to_plot))
    width = 0.35

    baseline_vals = [results['metrics'][0]['sharpe'], results['metrics'][0]['calmar']]
    leadlag_vals = [results['metrics'][1]['sharpe'], results['metrics'][1]['calmar']]

    bars1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline', color='#2196F3', alpha=0.8)
    bars2 = ax.bar(x + width/2, leadlag_vals, width, label='Lead-Lag', color='#4CAF50', alpha=0.8)

    # Add improvement labels
    for i, (b, l) in enumerate(zip(baseline_vals, leadlag_vals)):
        if b != 0:
            imp = (l - b) / abs(b) * 100
            ax.annotate(f'{imp:+.0f}%', xy=(x[i] + width/2, l),
                       xytext=(0, 3), textcoords='offset points',
                       ha='center', fontsize=9, color='#2E7D32', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(['Sharpe Ratio', 'Calmar Ratio'])
    ax.set_title('(C) Risk-Adjusted Performance', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # =========================================================================
    # Panel D: Regime Distribution and Action Frequency
    # =========================================================================
    ax = axes[1, 1]

    test_returns = results['test_returns']
    regimes = test_returns['regime']

    # Count by regime
    regime_counts = {}
    for k, name in results['regime_names'].items():
        regime_counts[name] = (regimes == k).sum()

    names = list(regime_counts.keys())
    counts = list(regime_counts.values())
    colors = ['#4CAF50', '#FFC107', '#F44336']

    bars = ax.bar(names, counts, color=colors, alpha=0.8, edgecolor='black')

    # Add percentage labels
    total = sum(counts)
    for bar, count in zip(bars, counts):
        pct = count / total * 100
        ax.annotate(f'{pct:.0f}%',
                   xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   xytext=(0, 3), textcoords='offset points',
                   ha='center', fontsize=10, fontweight='bold')

    ax.set_ylabel('Number of Days')
    ax.set_title('(D) Regime Distribution (Test Period)', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # =========================================================================
    # Final adjustments
    # =========================================================================
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"\nFigure saved to: {save_path}")

    plt.close()


def print_paper_table(results):
    """Print markdown table for paper."""

    print("\n" + "=" * 70)
    print("PAPER-READY TABLE (Markdown)")
    print("=" * 70)

    print("""
### Table: Backtest Performance (2015-2024)

| Strategy | Annual Return | Volatility | Sharpe | Max Drawdown | Calmar |
|----------|--------------|------------|--------|--------------|--------|""")

    for m in results['metrics']:
        print(f"| {m['name']} | {m['annual_return']:.1f}% | {m['annual_vol']:.1f}% | {m['sharpe']:.2f} | {m['max_drawdown']:.1f}% | {m['calmar']:.2f} |")

    # Improvement row
    baseline = results['metrics'][0]
    leadlag = results['metrics'][1]

    sharpe_imp = (leadlag['sharpe'] - baseline['sharpe']) / baseline['sharpe'] * 100 if baseline['sharpe'] != 0 else 0
    dd_imp = (baseline['max_drawdown'] - leadlag['max_drawdown']) / abs(baseline['max_drawdown']) * 100

    print(f"""
**Improvement (Lead-Lag vs Baseline):**
- Sharpe Ratio: {sharpe_imp:+.1f}%
- Drawdown Reduction: {dd_imp:+.1f}%

*Note: Lead-Lag strategy uses regime-dependent signals to reduce exposure to destination
factors when source factors show stress. Baseline is equal-weight buy-and-hold.*
""")


if __name__ == "__main__":
    # Run backtest
    results = run_backtest(test_start='2015-01-01', test_end='2024-12-31')

    # Create figure
    create_backtest_figure(
        results,
        save_path='/Users/i767700/Github/ai-in-finance/chorok/v11_causal_factor_crowding/fig_backtest.png'
    )

    # Print paper table
    print_paper_table(results)

    print("\n" + "=" * 70)
    print("BACKTEST COMPLETE")
    print("=" * 70)
