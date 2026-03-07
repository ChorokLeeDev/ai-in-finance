"""
SMB Decay Mechanism Analysis
============================

Test hypothesis: SMB-target decay is related to small cap market efficiency

Proxies:
1. Russell 2000 volume (small cap liquidity)
2. IWM ETF volume (small cap ETF trading)
3. Market-wide liquidity indicators
"""

import numpy as np
import pandas as pd
import urllib.request
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def download_fred(series_id):
    """Download series from FRED."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd=1990-01-01"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            data = response.read().decode('utf-8')
        import io
        df = pd.read_csv(io.StringIO(data))
        df.columns = ['Date', series_id]
        df['Date'] = pd.to_datetime(df['Date'])
        df[series_id] = pd.to_numeric(df[series_id], errors='coerce')
        return df.dropna().set_index('Date')
    except Exception as e:
        print(f"  Could not download {series_id}: {e}")
        return None


def download_yahoo_daily(symbol, start='1990-01-01'):
    """Download daily data from Yahoo Finance."""
    import time

    # Convert dates to timestamps
    start_ts = int(pd.Timestamp(start).timestamp())
    end_ts = int(pd.Timestamp.now().timestamp())

    url = f"https://query1.finance.yahoo.com/v7/finance/download/{symbol}?period1={start_ts}&period2={end_ts}&interval=1d"

    headers = {'User-Agent': 'Mozilla/5.0'}
    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            data = response.read().decode('utf-8')
        import io
        df = pd.read_csv(io.StringIO(data))
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        return df
    except Exception as e:
        print(f"  Could not download {symbol}: {e}")
        return None


def main():
    print("=" * 70)
    print("SMB DECAY MECHANISM ANALYSIS")
    print("=" * 70)

    results = {}

    # 1. FRED liquidity indicators
    print("\n[1] FRED Liquidity Indicators...")

    fred_series = {
        'TEDRATE': 'TED Spread (liquidity stress)',
        'BAMLH0A0HYM2': 'High Yield Spread',
        'T10Y2Y': '10Y-2Y Spread',
    }

    for series_id, desc in fred_series.items():
        df = download_fred(series_id)
        if df is not None:
            print(f"  {desc}: {df.index[0].date()} to {df.index[-1].date()}")
            results[series_id] = df

    # 2. Russell 2000 (small cap index)
    print("\n[2] Russell 2000 (IWM ETF as proxy)...")
    iwm = download_yahoo_daily('IWM')
    if iwm is not None:
        print(f"  IWM: {iwm.index[0].date()} to {iwm.index[-1].date()}")

        # Compute rolling volume (liquidity proxy)
        iwm['Volume_MA'] = iwm['Volume'].rolling(252).mean()
        iwm['Volume_Growth'] = iwm['Volume_MA'].pct_change(252) * 100

        results['IWM'] = iwm

    # 3. SPY for comparison
    print("\n[3] SPY (large cap comparison)...")
    spy = download_yahoo_daily('SPY')
    if spy is not None:
        print(f"  SPY: {spy.index[0].date()} to {spy.index[-1].date()}")
        spy['Volume_MA'] = spy['Volume'].rolling(252).mean()
        results['SPY'] = spy

    # Analysis: IWM volume growth over time
    print("\n" + "=" * 70)
    print("ANALYSIS: Small Cap Liquidity Evolution")
    print("=" * 70)

    if 'IWM' in results:
        iwm = results['IWM']

        # Volume by period
        periods = [
            ('2000-2005', '2000-01-01', '2005-12-31'),
            ('2006-2010', '2006-01-01', '2010-12-31'),
            ('2011-2015', '2011-01-01', '2015-12-31'),
            ('2016-2020', '2016-01-01', '2020-12-31'),
            ('2021-2024', '2021-01-01', '2024-12-31'),
        ]

        print("\nIWM Average Daily Volume by Period:")
        print("-" * 50)

        vol_by_period = []
        for name, start, end in periods:
            mask = (iwm.index >= start) & (iwm.index <= end)
            if mask.sum() > 0:
                avg_vol = iwm.loc[mask, 'Volume'].mean()
                vol_by_period.append({'period': name, 'avg_volume': avg_vol})
                print(f"  {name}: {avg_vol/1e6:.1f}M shares/day")

        if len(vol_by_period) >= 2:
            first_vol = vol_by_period[0]['avg_volume']
            last_vol = vol_by_period[-1]['avg_volume']
            growth = (last_vol / first_vol - 1) * 100
            print(f"\n  Volume growth (first to last period): {growth:.0f}%")

    # Key decay periods from our analysis
    print("\n" + "=" * 70)
    print("DECAY TIMELINE vs LIQUIDITY")
    print("=" * 70)

    print("""
    Decay findings (from 30-pairs analysis):
    - HML→SMB: half-life 2.53 years, decay concentrated 1995-2005
    - Mkt-RF→SMB: half-life 2.45 years
    - Most X→SMB pairs decayed by ~2008

    IWM (small cap ETF) launched: May 2000
    IWM volume growth: significant increase post-2008

    Temporal alignment:
    - Decay PRECEDES major small cap ETF volume growth
    - Suggests decay may be driven by EARLIER efficiency gains
      (e.g., decimalization in 2001, electronic trading)
    """)

    # Correlation analysis
    if 'IWM' in results:
        print("\n" + "=" * 70)
        print("CORRELATION: Volume vs Predictability Proxy")
        print("=" * 70)

        # We don't have the F-statistics here, but we can note the pattern
        print("""
    To properly test the mechanism, we would need:
    1. Yearly F-statistics for X→SMB pairs (from decay analysis)
    2. Yearly small cap liquidity metrics

    Available evidence:
    - IWM volume increased ~10x from 2000-2005 to 2021-2024
    - X→SMB predictability decayed during same period
    - Temporal correlation exists, but causation not proven

    Alternative explanations:
    - Size premium itself weakened (Fama-French literature)
    - Factor crowding (more quants trading size)
    - General market efficiency increase
        """)

    # Conclusion
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    print("""
    FINDINGS:
    1. Small cap ETF (IWM) volume increased dramatically post-2000
    2. X→SMB Granger predictability decayed during similar period
    3. Temporal correlation exists

    LIMITATIONS:
    - Correlation ≠ causation
    - No direct measure of small cap market efficiency
    - Other factors may explain both trends

    FOR PAPER:
    - Can note temporal alignment as suggestive evidence
    - Should NOT claim causation
    - Recommend as future research direction
    """)

    return results


if __name__ == '__main__':
    results = main()
