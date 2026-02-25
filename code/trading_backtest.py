"""
Trading Strategy Backtest
=========================

Simple trading strategy based on regime-dependent causal structure:
- During Crisis: HML predicts SMB (9-day lag) → Trade SMB based on HML signal
- During Crowding: SMB predicts HML (3-day lag) → Trade HML based on SMB signal

Compare against buy-and-hold benchmark.
"""

import numpy as np
import pandas as pd
import urllib.request
import zipfile
import io
from scipy.stats import f as f_dist
import warnings
warnings.filterwarnings('ignore')


def download_ff_data():
    """Download Fama-French 5 factors daily data."""
    url = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip'

    with urllib.request.urlopen(url, timeout=30) as response:
        data = response.read()

    with zipfile.ZipFile(io.BytesIO(data)) as z:
        csv_name = z.namelist()[0]
        with z.open(csv_name) as f:
            df = pd.read_csv(f, skiprows=3)

    df.columns = df.columns.str.strip()
    df = df.rename(columns={df.columns[0]: 'Date'})
    df = df[df['Date'].astype(str).str.match(r'^\d{8}$')]
    df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')

    for col in ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df.dropna().set_index('Date')


def compute_rolling_volatility(df, window=60):
    """Compute rolling volatility for regime detection."""
    factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
    vol_df = pd.DataFrame(index=df.index)

    for factor in factors:
        vol_df[f'{factor}_vol'] = df[factor].rolling(window).std()

    return vol_df.dropna()


def classify_regimes_rolling(df, vol_df, lookback=252*5):
    """
    Classify regimes using rolling lookback window for quantile thresholds.
    This ensures out-of-sample validity.
    """
    regimes = pd.Series(index=vol_df.index, dtype=str)
    avg_vol = vol_df.mean(axis=1)

    for i in range(lookback, len(avg_vol)):
        # Use past data only for thresholds
        past_vol = avg_vol.iloc[i-lookback:i]
        q25 = past_vol.quantile(0.25)
        q75 = past_vol.quantile(0.75)

        current_vol = avg_vol.iloc[i]

        if current_vol <= q25:
            regimes.iloc[i] = 'Normal'
        elif current_vol >= q75:
            regimes.iloc[i] = 'Crisis'
        else:
            regimes.iloc[i] = 'Crowding'

    return regimes.dropna()


def generate_signals(df, regimes, crisis_lag=9, crowding_lag=3):
    """
    Generate trading signals based on causal structure.

    During Crisis: Use HML to predict SMB → Generate SMB signal
    During Crowding: Use SMB to predict HML → Generate HML signal

    Signal logic: positive predictor → go long destination factor
    """
    signals = pd.DataFrame(index=regimes.index)
    signals['SMB_signal'] = 0.0  # -1, 0, or 1
    signals['HML_signal'] = 0.0

    # Align data
    df_aligned = df.loc[regimes.index]

    for i in range(max(crisis_lag, crowding_lag), len(regimes)):
        date = regimes.index[i]
        regime = regimes.iloc[i]

        if regime == 'Crisis':
            # HML predicts SMB with 9-day lag
            # Use average HML over past crisis_lag days
            hml_avg = df_aligned['HML'].iloc[i-crisis_lag:i].mean()
            signals.loc[date, 'SMB_signal'] = 1.0 if hml_avg > 0 else -1.0

        elif regime == 'Crowding':
            # SMB predicts HML with 3-day lag
            smb_avg = df_aligned['SMB'].iloc[i-crowding_lag:i].mean()
            signals.loc[date, 'HML_signal'] = 1.0 if smb_avg > 0 else -1.0

    return signals


def backtest_strategy(df, signals, regimes, transaction_cost=0.001):
    """
    Backtest the trading strategy.

    Returns daily P&L and cumulative returns.
    """
    # Align all data
    common_idx = signals.index
    df_aligned = df.loc[common_idx]
    regimes_aligned = regimes.loc[common_idx]

    # Calculate daily returns
    results = pd.DataFrame(index=common_idx)
    results['regime'] = regimes_aligned

    # Strategy returns (apply signal from previous day)
    results['SMB_return'] = df_aligned['SMB'] / 100  # Convert from percentage
    results['HML_return'] = df_aligned['HML'] / 100

    # Shift signals to avoid look-ahead bias
    results['SMB_signal'] = signals['SMB_signal'].shift(1)
    results['HML_signal'] = signals['HML_signal'].shift(1)

    # Strategy daily returns
    results['SMB_strategy'] = results['SMB_signal'] * results['SMB_return']
    results['HML_strategy'] = results['HML_signal'] * results['HML_return']

    # Combined strategy return
    results['strategy_return'] = results['SMB_strategy'] + results['HML_strategy']

    # Transaction costs (charge when signal changes)
    results['SMB_trade'] = (results['SMB_signal'].diff().abs() > 0).astype(float)
    results['HML_trade'] = (results['HML_signal'].diff().abs() > 0).astype(float)
    results['tc'] = (results['SMB_trade'] + results['HML_trade']) * transaction_cost

    results['strategy_return_net'] = results['strategy_return'] - results['tc']

    # Benchmark: equal-weight SMB + HML
    results['benchmark_return'] = (results['SMB_return'] + results['HML_return']) / 2

    # Cumulative returns
    results['strategy_cum'] = (1 + results['strategy_return_net']).cumprod()
    results['benchmark_cum'] = (1 + results['benchmark_return']).cumprod()

    return results.dropna()


def calculate_metrics(results):
    """Calculate performance metrics."""
    metrics = {}

    # Strategy metrics
    strat_returns = results['strategy_return_net']
    metrics['strategy_total_return'] = (results['strategy_cum'].iloc[-1] - 1) * 100
    metrics['strategy_annual_return'] = strat_returns.mean() * 252 * 100
    metrics['strategy_annual_vol'] = strat_returns.std() * np.sqrt(252) * 100
    metrics['strategy_sharpe'] = (strat_returns.mean() / strat_returns.std()) * np.sqrt(252) if strat_returns.std() > 0 else 0
    metrics['strategy_max_dd'] = (results['strategy_cum'] / results['strategy_cum'].cummax() - 1).min() * 100

    # Benchmark metrics
    bench_returns = results['benchmark_return']
    metrics['benchmark_total_return'] = (results['benchmark_cum'].iloc[-1] - 1) * 100
    metrics['benchmark_annual_return'] = bench_returns.mean() * 252 * 100
    metrics['benchmark_annual_vol'] = bench_returns.std() * np.sqrt(252) * 100
    metrics['benchmark_sharpe'] = (bench_returns.mean() / bench_returns.std()) * np.sqrt(252) if bench_returns.std() > 0 else 0
    metrics['benchmark_max_dd'] = (results['benchmark_cum'] / results['benchmark_cum'].cummax() - 1).min() * 100

    # Trade statistics
    metrics['n_trades'] = (results['SMB_trade'].sum() + results['HML_trade'].sum())
    metrics['n_days'] = len(results)
    metrics['turnover_annual'] = metrics['n_trades'] / metrics['n_days'] * 252

    return metrics


def run_backtest():
    """Run the full backtest."""
    print("=" * 70)
    print("TRADING STRATEGY BACKTEST")
    print("Based on Regime-Dependent Causal Structure")
    print("=" * 70)

    # Load data
    print("\n1. Loading data...")
    df = download_ff_data()
    vol_df = compute_rolling_volatility(df)

    # Classify regimes (using rolling lookback for OOS validity)
    print("2. Detecting regimes (5-year rolling lookback)...")
    regimes = classify_regimes_rolling(df, vol_df, lookback=252*5)

    print(f"   Period: {regimes.index[0].date()} to {regimes.index[-1].date()}")
    print(f"   Total days: {len(regimes)}")

    regime_counts = regimes.value_counts()
    for regime in ['Normal', 'Crowding', 'Crisis']:
        if regime in regime_counts:
            pct = regime_counts[regime] / len(regimes) * 100
            print(f"   {regime}: {regime_counts[regime]} days ({pct:.1f}%)")

    # Generate signals
    print("\n3. Generating trading signals...")
    signals = generate_signals(df, regimes, crisis_lag=9, crowding_lag=3)

    smb_active = (signals['SMB_signal'] != 0).sum()
    hml_active = (signals['HML_signal'] != 0).sum()
    print(f"   SMB signal active: {smb_active} days ({smb_active/len(signals)*100:.1f}%)")
    print(f"   HML signal active: {hml_active} days ({hml_active/len(signals)*100:.1f}%)")

    # Backtest
    print("\n4. Running backtest (10bp transaction cost)...")
    results = backtest_strategy(df, signals, regimes, transaction_cost=0.001)

    # Calculate metrics
    metrics = calculate_metrics(results)

    # Print results
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)

    print(f"\n{'Metric':<30} {'Strategy':<15} {'Benchmark':<15}")
    print("-" * 60)
    print(f"{'Total Return (%)':<30} {metrics['strategy_total_return']:<15.2f} {metrics['benchmark_total_return']:<15.2f}")
    print(f"{'Annual Return (%)':<30} {metrics['strategy_annual_return']:<15.2f} {metrics['benchmark_annual_return']:<15.2f}")
    print(f"{'Annual Volatility (%)':<30} {metrics['strategy_annual_vol']:<15.2f} {metrics['benchmark_annual_vol']:<15.2f}")
    print(f"{'Sharpe Ratio':<30} {metrics['strategy_sharpe']:<15.2f} {metrics['benchmark_sharpe']:<15.2f}")
    print(f"{'Max Drawdown (%)':<30} {metrics['strategy_max_dd']:<15.2f} {metrics['benchmark_max_dd']:<15.2f}")

    print(f"\n{'Trade Statistics':<30}")
    print("-" * 60)
    print(f"{'Total Trades':<30} {metrics['n_trades']:.0f}")
    print(f"{'Annual Turnover':<30} {metrics['turnover_annual']:.1f}")

    # Performance by regime
    print("\n" + "=" * 70)
    print("PERFORMANCE BY REGIME")
    print("=" * 70)

    for regime in ['Normal', 'Crowding', 'Crisis']:
        regime_mask = results['regime'] == regime
        if regime_mask.sum() > 0:
            regime_results = results[regime_mask]
            strat_ret = regime_results['strategy_return_net'].mean() * 252 * 100
            bench_ret = regime_results['benchmark_return'].mean() * 252 * 100
            n_days = regime_mask.sum()
            print(f"\n{regime} ({n_days} days):")
            print(f"  Strategy annualized return: {strat_ret:.2f}%")
            print(f"  Benchmark annualized return: {bench_ret:.2f}%")
            print(f"  Excess return: {strat_ret - bench_ret:.2f}%")

    return results, metrics


if __name__ == "__main__":
    results, metrics = run_backtest()
