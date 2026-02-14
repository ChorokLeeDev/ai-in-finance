"""
Crisis Trading Backtest (Tab:trading) using selected HMM fit regimes.
Matches paper definition (main_icaif.tex:641-652):
  - During Crisis regime only, go long SMB when 9-day cumulative HML > 0,
    short SMB when < 0. Flat in Normal/Elevated.
  - Benchmark: Buy-and-hold SMB
  - Period: 1995-2024
  - No transaction costs
Output: results/trading_selected.json
"""

import numpy as np
import pandas as pd
import json
import urllib.request
import zipfile
import io
from datetime import datetime

RESULTS_DIR = '/Users/i767700/Github/ai-in-finance/papers/causal_regimes/results'


def load_ff_data():
    url5 = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip'
    with urllib.request.urlopen(url5, timeout=60) as response:
        data = response.read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        with z.open(z.namelist()[0]) as f:
            df5 = pd.read_csv(f, skiprows=3)
    df5.columns = df5.columns.str.strip()
    df5 = df5.rename(columns={df5.columns[0]: 'Date'})
    df5 = df5[df5['Date'].astype(str).str.match(r'^\d{8}$')]
    df5['Date'] = pd.to_datetime(df5['Date'], format='%Y%m%d')
    for col in ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']:
        df5[col] = pd.to_numeric(df5[col], errors='coerce')
    df5 = df5.set_index('Date').dropna()
    return df5.loc['1990-01-01':'2024-12-31']


def main():
    print("=" * 70)
    print("CRISIS TRADING BACKTEST (SELECTED FIT)")
    print("=" * 70, flush=True)

    ff = load_ff_data()
    regime_df = pd.read_csv(f"{RESULTS_DIR}/selected_fit_regimes.csv")
    regime_df['date'] = pd.to_datetime(regime_df['date'])
    regime_df = regime_df.set_index('date')

    common = ff.index.intersection(regime_df.index).sort_values()
    df = ff.loc[common].copy()
    df['regime'] = regime_df.loc[common, 'regime_label']

    # 9-day cumulative HML
    df['hml_cumul_9'] = df['HML'].rolling(9).sum()

    # Strategy: Crisis only, long SMB if hml_cumul_9 > 0, short if < 0
    df['signal'] = 0.0
    crisis_mask = df['regime'] == 'Crisis'
    df.loc[crisis_mask & (df['hml_cumul_9'] > 0), 'signal'] = 1.0
    df.loc[crisis_mask & (df['hml_cumul_9'] < 0), 'signal'] = -1.0
    # Shift signal by 1 to avoid look-ahead
    df['signal'] = df['signal'].shift(1)

    # Filter to 1995-2024
    df = df.loc['1995-01-01':'2024-12-31'].dropna(subset=['signal', 'hml_cumul_9'])

    # Strategy returns
    df['strategy_ret'] = df['signal'] * df['SMB'] / 100  # FF data in %
    df['benchmark_ret'] = df['SMB'] / 100

    # Cumulative
    df['strategy_cum'] = (1 + df['strategy_ret']).cumprod()
    df['benchmark_cum'] = (1 + df['benchmark_ret']).cumprod()

    n_years = len(df) / 252

    # Strategy metrics
    strat_ann_ret = (df['strategy_cum'].iloc[-1] ** (1 / n_years) - 1) * 100
    strat_vol = df['strategy_ret'].std() * np.sqrt(252) * 100
    strat_sharpe = (df['strategy_ret'].mean() / df['strategy_ret'].std() * np.sqrt(252)) if df['strategy_ret'].std() > 0 else 0
    strat_cum_max = df['strategy_cum'].cummax()
    strat_dd = ((df['strategy_cum'] - strat_cum_max) / strat_cum_max).min() * 100

    # Benchmark metrics
    bench_ann_ret = (df['benchmark_cum'].iloc[-1] ** (1 / n_years) - 1) * 100
    bench_vol = df['benchmark_ret'].std() * np.sqrt(252) * 100
    bench_sharpe = (df['benchmark_ret'].mean() / df['benchmark_ret'].std() * np.sqrt(252)) if df['benchmark_ret'].std() > 0 else 0
    bench_cum_max = df['benchmark_cum'].cummax()
    bench_dd = ((df['benchmark_cum'] - bench_cum_max) / bench_cum_max).min() * 100

    print(f"\n  Strategy: ann_ret={strat_ann_ret:.1f}%, sharpe={strat_sharpe:.2f}, max_dd={strat_dd:.1f}%")
    print(f"  Benchmark: ann_ret={bench_ann_ret:.1f}%, sharpe={bench_sharpe:.2f}, max_dd={bench_dd:.1f}%")

    results = {
        'metadata': {
            'description': 'Crisis trading backtest: long/short SMB based on HML signal in Crisis regime',
            'period': '1995-2024',
            'signal': '9-day cumulative HML, Crisis regime only',
            'benchmark': 'Buy-and-hold SMB',
            'transaction_costs': 'Not modeled',
            'timestamp': datetime.now().isoformat(),
        },
        'strategy': {
            'annual_return_pct': round(strat_ann_ret, 1),
            'sharpe_ratio': round(strat_sharpe, 2),
            'max_drawdown_pct': round(strat_dd, 1),
            'annual_volatility_pct': round(strat_vol, 1),
        },
        'benchmark': {
            'annual_return_pct': round(bench_ann_ret, 1),
            'sharpe_ratio': round(bench_sharpe, 2),
            'max_drawdown_pct': round(bench_dd, 1),
            'annual_volatility_pct': round(bench_vol, 1),
        },
        'n_days': len(df),
        'n_crisis_days': int((df['regime'] == 'Crisis').sum()),
    }

    out_path = f"{RESULTS_DIR}/trading_selected.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == '__main__':
    main()
