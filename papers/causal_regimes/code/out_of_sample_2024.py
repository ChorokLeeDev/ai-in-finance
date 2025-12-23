"""
Out-of-Sample Validation: 2024 Data
====================================

Test if the regime-dependent causal structure discovered in 1990-2023
holds in 2024 data.

Key question: Does HML→SMB in crisis, SMB→HML in crowding still hold?
"""

import numpy as np
import pandas as pd
import urllib.request
import zipfile
import io
from scipy import stats
from scipy.stats import pearsonr
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

    # Convert to float
    for col in ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df.dropna()


def compute_rolling_volatility(df, window=60):
    """Compute rolling volatility for regime detection."""
    factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
    vol_df = pd.DataFrame(index=df.index)

    for factor in factors:
        vol_df[f'{factor}_vol'] = df[factor].rolling(window).std()

    return vol_df.dropna()


def classify_regimes_simple(vol_df):
    """
    Simple regime classification based on average volatility.
    - Crisis: volatility > 75th percentile
    - Normal: volatility < 25th percentile
    - Crowding: in between
    """
    avg_vol = vol_df.mean(axis=1)

    q25 = avg_vol.quantile(0.25)
    q75 = avg_vol.quantile(0.75)

    regimes = pd.Series(index=vol_df.index, dtype=str)
    regimes[avg_vol <= q25] = 'Normal'
    regimes[avg_vol >= q75] = 'Crisis'
    regimes[(avg_vol > q25) & (avg_vol < q75)] = 'Crowding'

    return regimes


def granger_causality_test(x, y, max_lag=10):
    """
    Test if x Granger-causes y.
    Returns best lag and p-value.
    """
    from scipy.stats import f as f_dist

    n = len(x)
    best_p = 1.0
    best_lag = 1

    for lag in range(1, max_lag + 1):
        if n - lag < lag * 2 + 10:
            continue

        # Restricted model: y ~ y_lagged
        y_curr = y[lag:]
        y_lagged = np.column_stack([y[lag-i-1:-i-1] for i in range(lag)])

        # Unrestricted model: y ~ y_lagged + x_lagged
        x_lagged = np.column_stack([x[lag-i-1:-i-1] for i in range(lag)])

        # Fit models
        X_r = np.column_stack([np.ones(len(y_curr)), y_lagged])
        X_u = np.column_stack([np.ones(len(y_curr)), y_lagged, x_lagged])

        try:
            beta_r = np.linalg.lstsq(X_r, y_curr, rcond=None)[0]
            beta_u = np.linalg.lstsq(X_u, y_curr, rcond=None)[0]

            rss_r = np.sum((y_curr - X_r @ beta_r) ** 2)
            rss_u = np.sum((y_curr - X_u @ beta_u) ** 2)

            # F-test
            df1 = lag
            df2 = len(y_curr) - 2 * lag - 1

            if df2 > 0 and rss_u > 0:
                f_stat = ((rss_r - rss_u) / df1) / (rss_u / df2)
                p_value = 1 - f_dist.cdf(f_stat, df1, df2)

                if p_value < best_p:
                    best_p = p_value
                    best_lag = lag
        except:
            continue

    return best_lag, best_p


def run_validation():
    """Run out-of-sample validation on 2024 data."""
    print("=" * 60)
    print("OUT-OF-SAMPLE VALIDATION: 2024 DATA")
    print("=" * 60)

    # Download data
    print("\n1. Downloading Fama-French data...")
    df = download_ff_data()

    # Split data
    df_train = df[df['Date'] < '2024-01-01']
    df_test = df[df['Date'] >= '2024-01-01']

    print(f"   Training: {df_train['Date'].min().date()} to {df_train['Date'].max().date()} ({len(df_train)} days)")
    print(f"   Test (2024): {df_test['Date'].min().date()} to {df_test['Date'].max().date()} ({len(df_test)} days)")

    # Compute volatility and regimes for 2024
    print("\n2. Computing regimes for 2024...")

    # Use full history for volatility thresholds
    vol_full = compute_rolling_volatility(df)
    vol_2024 = vol_full.loc[df_test.index.intersection(vol_full.index)]

    # Classify using historical quantiles
    avg_vol_full = vol_full.mean(axis=1)
    q25 = avg_vol_full.quantile(0.25)
    q75 = avg_vol_full.quantile(0.75)

    avg_vol_2024 = vol_2024.mean(axis=1)
    regimes_2024 = pd.Series(index=vol_2024.index, dtype=str)
    regimes_2024[avg_vol_2024 <= q25] = 'Normal'
    regimes_2024[avg_vol_2024 >= q75] = 'Crisis'
    regimes_2024[(avg_vol_2024 > q25) & (avg_vol_2024 < q75)] = 'Crowding'

    print(f"   Regime distribution in 2024:")
    for regime in ['Normal', 'Crowding', 'Crisis']:
        count = (regimes_2024 == regime).sum()
        pct = count / len(regimes_2024) * 100
        print(f"     {regime}: {count} days ({pct:.1f}%)")

    # Test Granger causality in each regime
    print("\n3. Testing Granger causality in 2024 regimes...")

    df_2024_aligned = df.loc[regimes_2024.index]

    results = {}
    for regime in ['Normal', 'Crowding', 'Crisis']:
        regime_mask = regimes_2024 == regime
        n_days = regime_mask.sum()

        if n_days < 30:
            print(f"\n   {regime} regime: Insufficient data ({n_days} days)")
            results[regime] = {'n_days': n_days, 'hml_to_smb': None, 'smb_to_hml': None}
            continue

        regime_data = df_2024_aligned[regime_mask]
        smb = regime_data['SMB'].values
        hml = regime_data['HML'].values

        # Test HML → SMB
        lag1, p1 = granger_causality_test(hml, smb, max_lag=10)

        # Test SMB → HML
        lag2, p2 = granger_causality_test(smb, hml, max_lag=10)

        print(f"\n   {regime} regime ({n_days} days):")
        print(f"     HML → SMB: lag={lag1}, p={p1:.4f} {'***' if p1 < 0.01 else '**' if p1 < 0.05 else '*' if p1 < 0.1 else ''}")
        print(f"     SMB → HML: lag={lag2}, p={p2:.4f} {'***' if p2 < 0.01 else '**' if p2 < 0.05 else '*' if p2 < 0.1 else ''}")

        results[regime] = {
            'n_days': n_days,
            'hml_to_smb': {'lag': lag1, 'p_value': p1},
            'smb_to_hml': {'lag': lag2, 'p_value': p2}
        }

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    print("\nExpected pattern (from 1990-2023 training):")
    print("  - Normal: No significant causality in either direction")
    print("  - Crowding: SMB → HML (Size causes Value)")
    print("  - Crisis: HML → SMB (Value causes Size)")

    print("\n2024 Results:")

    matches = 0
    total = 0

    for regime, res in results.items():
        if res['hml_to_smb'] is None:
            print(f"  {regime}: Insufficient data")
            continue

        hml_sig = res['hml_to_smb']['p_value'] < 0.05
        smb_sig = res['smb_to_hml']['p_value'] < 0.05

        if regime == 'Normal':
            expected = "No causality"
            match = not hml_sig and not smb_sig
        elif regime == 'Crowding':
            expected = "SMB → HML"
            match = smb_sig and not hml_sig
        else:  # Crisis
            expected = "HML → SMB"
            match = hml_sig and not smb_sig

        actual = []
        if hml_sig:
            actual.append("HML→SMB")
        if smb_sig:
            actual.append("SMB→HML")
        if not actual:
            actual = ["No causality"]

        status = "MATCH" if match else "PARTIAL" if (hml_sig or smb_sig) else "NO MATCH"
        print(f"  {regime}: Expected {expected}, Got {', '.join(actual)} [{status}]")

        total += 1
        if match:
            matches += 1

    print(f"\nOverall: {matches}/{total} regime patterns matched")

    return results


if __name__ == "__main__":
    results = run_validation()
