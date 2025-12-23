"""
Rolling Out-of-Sample Validation
================================

Test if the regime-dependent causal structure generalizes by:
1. Training regime model on expanding window
2. Testing causality on next year's data
3. Tracking pattern consistency over time

This addresses the limitation that 2024 single-year test showed 0/2 patterns.
"""

import numpy as np
import pandas as pd
import urllib.request
import zipfile
import io
from scipy import stats
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


def classify_regimes(vol_df, q25, q75):
    """
    Classify regimes based on given quantile thresholds.
    """
    avg_vol = vol_df.mean(axis=1)

    regimes = pd.Series(index=vol_df.index, dtype=str)
    regimes[avg_vol <= q25] = 'Normal'
    regimes[avg_vol >= q75] = 'Crisis'
    regimes[(avg_vol > q25) & (avg_vol < q75)] = 'Crowding'

    return regimes


def granger_test(x, y, max_lag=10):
    """Test if x Granger-causes y."""
    n = len(x)
    best_p = 1.0
    best_lag = 1

    for lag in range(1, max_lag + 1):
        if n - lag < lag * 2 + 10:
            continue

        y_curr = y[lag:]
        y_lagged = np.column_stack([y[lag-i-1:-i-1] for i in range(lag)])
        x_lagged = np.column_stack([x[lag-i-1:-i-1] for i in range(lag)])

        X_r = np.column_stack([np.ones(len(y_curr)), y_lagged])
        X_u = np.column_stack([np.ones(len(y_curr)), y_lagged, x_lagged])

        try:
            beta_r = np.linalg.lstsq(X_r, y_curr, rcond=None)[0]
            beta_u = np.linalg.lstsq(X_u, y_curr, rcond=None)[0]

            rss_r = np.sum((y_curr - X_r @ beta_r) ** 2)
            rss_u = np.sum((y_curr - X_u @ beta_u) ** 2)

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


def run_rolling_validation():
    """Run rolling out-of-sample validation."""
    print("=" * 70)
    print("ROLLING OUT-OF-SAMPLE VALIDATION")
    print("=" * 70)

    # Download data
    print("\n1. Loading data...")
    df = download_ff_data()
    vol_df = compute_rolling_volatility(df)

    # Define test years (excluding first few for training)
    test_years = list(range(2000, 2025))

    results = []

    print("\n2. Running rolling tests...")
    print("-" * 70)
    print(f"{'Year':<6} {'Train Period':<20} {'Crisis':<12} {'Crowding':<12} {'Pattern'}")
    print("-" * 70)

    for test_year in test_years:
        # Training: all data up to test_year
        train_mask = vol_df.index.year < test_year
        test_mask = vol_df.index.year == test_year

        if train_mask.sum() < 252 or test_mask.sum() < 50:
            continue

        # Compute thresholds from training data
        vol_train = vol_df[train_mask]
        avg_vol_train = vol_train.mean(axis=1)
        q25 = avg_vol_train.quantile(0.25)
        q75 = avg_vol_train.quantile(0.75)

        # Classify test year regimes using training thresholds
        vol_test = vol_df[test_mask]
        regimes_test = classify_regimes(vol_test, q25, q75)

        # Get aligned factor data
        df_test = df.loc[regimes_test.index]

        # Test Granger causality in each regime
        year_results = {'year': test_year, 'train_end': test_year - 1}

        for regime in ['Crisis', 'Crowding']:
            regime_mask = regimes_test == regime
            n_days = regime_mask.sum()

            if n_days < 30:
                year_results[f'{regime.lower()}_hml_smb'] = None
                year_results[f'{regime.lower()}_smb_hml'] = None
                year_results[f'{regime.lower()}_days'] = n_days
                continue

            regime_data = df_test[regime_mask]
            smb = regime_data['SMB'].values
            hml = regime_data['HML'].values

            _, p1 = granger_test(hml, smb, max_lag=10)
            _, p2 = granger_test(smb, hml, max_lag=10)

            year_results[f'{regime.lower()}_hml_smb'] = p1
            year_results[f'{regime.lower()}_smb_hml'] = p2
            year_results[f'{regime.lower()}_days'] = n_days

        # Determine pattern match
        crisis_match = (year_results.get('crisis_hml_smb') is not None and
                       year_results['crisis_hml_smb'] < 0.05 and
                       (year_results.get('crisis_smb_hml') is None or
                        year_results['crisis_smb_hml'] >= 0.05))

        crowding_match = (year_results.get('crowding_smb_hml') is not None and
                         year_results['crowding_smb_hml'] < 0.05 and
                         (year_results.get('crowding_hml_smb') is None or
                          year_results['crowding_hml_smb'] >= 0.05))

        year_results['crisis_match'] = crisis_match
        year_results['crowding_match'] = crowding_match

        # Print row
        crisis_str = f"{year_results.get('crisis_days', 0)}d" if year_results.get('crisis_hml_smb') is not None else "N/A"
        crowding_str = f"{year_results.get('crowding_days', 0)}d" if year_results.get('crowding_smb_hml') is not None else "N/A"

        pattern = []
        if crisis_match:
            pattern.append("HML→SMB")
        if crowding_match:
            pattern.append("SMB→HML")
        pattern_str = ", ".join(pattern) if pattern else "-"

        print(f"{test_year:<6} 1990-{test_year-1:<14} {crisis_str:<12} {crowding_str:<12} {pattern_str}")

        results.append(year_results)

    # Summary statistics
    print("\n" + "=" * 70)
    print("ROLLING VALIDATION SUMMARY")
    print("=" * 70)

    df_results = pd.DataFrame(results)

    valid_crisis = df_results[df_results['crisis_hml_smb'].notna()]
    valid_crowding = df_results[df_results['crowding_smb_hml'].notna()]

    print(f"\nCrisis regime (expected: HML → SMB):")
    print(f"  Years with sufficient crisis days: {len(valid_crisis)}")
    print(f"  Years showing expected pattern: {valid_crisis['crisis_match'].sum()}")
    print(f"  Hit rate: {valid_crisis['crisis_match'].mean()*100:.1f}%")

    print(f"\nCrowding regime (expected: SMB → HML):")
    print(f"  Years with sufficient crowding days: {len(valid_crowding)}")
    print(f"  Years showing expected pattern: {valid_crowding['crowding_match'].sum()}")
    print(f"  Hit rate: {valid_crowding['crowding_match'].mean()*100:.1f}%")

    # Overall pattern consistency
    both_valid = df_results[(df_results['crisis_hml_smb'].notna()) | (df_results['crowding_smb_hml'].notna())]
    total_matches = both_valid['crisis_match'].sum() + both_valid['crowding_match'].sum()
    total_tests = len(valid_crisis) + len(valid_crowding)

    print(f"\nOverall: {total_matches}/{total_tests} regime-year combinations match expected pattern")
    print(f"         ({total_matches/total_tests*100:.1f}% hit rate)")

    return df_results


if __name__ == "__main__":
    results = run_rolling_validation()
