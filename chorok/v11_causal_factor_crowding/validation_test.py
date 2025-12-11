"""
V10 Validation Test: Causal Relationships in Factor Crowding

Goal: Show that causal relationships exist between factor crowding levels
      before building the full causal discovery system.

Tests:
1. Granger Causality Test
2. Cross-Correlation Analysis
3. VAR Impulse Response

Data: Fama-French 5 Factors + Momentum (daily)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. DATA LOADING: Fama-French Factors
# ============================================================

def download_ff_factors():
    """
    Download Fama-French 5 Factors + Momentum from Kenneth French's website
    """
    import pandas_datareader.data as web

    # Fama-French 5 Factors (Daily)
    ff5 = web.DataReader('F-F_Research_Data_5_Factors_2x3_daily', 'famafrench', start='1990-01-01')[0]

    # Momentum Factor (Daily)
    mom = web.DataReader('F-F_Momentum_Factor_daily', 'famafrench', start='1990-01-01')[0]

    # Combine
    factors = ff5.join(mom, how='inner')
    factors.columns = ['MKT', 'SMB', 'HML', 'RMW', 'CMA', 'RF', 'MOM']

    # Remove risk-free rate from market
    factors['MKT'] = factors['MKT'] - factors['RF']
    factors = factors.drop('RF', axis=1)

    print(f"Loaded {len(factors)} days of factor data")
    print(f"Factors: {list(factors.columns)}")
    print(f"Date range: {factors.index[0]} to {factors.index[-1]}")

    return factors


def load_ff_factors_from_csv(filepath=None):
    """
    Alternative: Load from local CSV if pandas_datareader fails
    Download manually from: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
    """
    if filepath:
        factors = pd.read_csv(filepath, index_col=0, parse_dates=True)
    else:
        # Try pandas_datareader
        try:
            factors = download_ff_factors()
        except Exception as e:
            print(f"Error downloading: {e}")
            print("Please download manually from Kenneth French website")
            return None
    return factors


# ============================================================
# 2. CROWDING PROXY CONSTRUCTION
# ============================================================

def compute_crowding_proxy(factor_returns, window=60):
    """
    Construct crowding proxy for each factor based on:
    1. Rolling volatility (higher vol = more crowding unwinding)
    2. Rolling autocorrelation (lower autocorr = more crowded/mean-reverting)
    3. Cross-factor correlation (higher = more systematic crowding)

    This is a simplified proxy. Real crowding uses:
    - Short interest spread
    - Valuation spread
    - Fund flow data
    """
    crowding = pd.DataFrame(index=factor_returns.index)

    for col in factor_returns.columns:
        # Rolling volatility (normalized)
        vol = factor_returns[col].rolling(window).std()
        vol_norm = (vol - vol.rolling(252).mean()) / vol.rolling(252).std()

        # Rolling autocorrelation (lower = more crowded)
        autocorr = factor_returns[col].rolling(window).apply(
            lambda x: x.autocorr(lag=1) if len(x) > 1 else np.nan
        )
        autocorr_norm = -(autocorr - autocorr.rolling(252).mean()) / autocorr.rolling(252).std()

        # Combine (equal weight for simplicity)
        crowding[col] = 0.5 * vol_norm.fillna(0) + 0.5 * autocorr_norm.fillna(0)

    # Add cross-factor correlation as overall crowding indicator
    rolling_corr = factor_returns.rolling(window).corr()
    # Mean pairwise correlation (excluding diagonal)

    return crowding.dropna()


def compute_simple_crowding(factor_returns, window=60):
    """
    Even simpler crowding proxy: just rolling volatility
    Higher volatility = crowding unwinding risk
    """
    crowding = pd.DataFrame(index=factor_returns.index)

    for col in factor_returns.columns:
        vol = factor_returns[col].rolling(window).std()
        # Z-score normalization
        crowding[col] = (vol - vol.rolling(252).mean()) / vol.rolling(252).std()

    return crowding.dropna()


# ============================================================
# 3. VALIDATION TEST 1: GRANGER CAUSALITY
# ============================================================

def test_granger_causality(crowding_data, maxlag=20):
    """
    Test if one factor's crowding Granger-causes another's

    H0: X does NOT Granger-cause Y
    H1: X Granger-causes Y

    Returns matrix of p-values
    """
    from statsmodels.tsa.stattools import grangercausalitytests

    factors = crowding_data.columns.tolist()
    n = len(factors)

    # Results matrix: p_values[i,j] = p-value for "factor i causes factor j"
    p_values = pd.DataFrame(np.nan, index=factors, columns=factors)
    best_lags = pd.DataFrame(np.nan, index=factors, columns=factors)

    print("=" * 60)
    print("GRANGER CAUSALITY TEST RESULTS")
    print("=" * 60)

    for cause in factors:
        for effect in factors:
            if cause == effect:
                continue

            # Granger test requires [effect, cause] order
            data = crowding_data[[effect, cause]].dropna()

            if len(data) < maxlag + 10:
                continue

            try:
                result = grangercausalitytests(data, maxlag=maxlag, verbose=False)

                # Find minimum p-value across all lags
                min_p = 1.0
                best_lag = 0
                for lag in range(1, maxlag + 1):
                    # Use F-test p-value
                    p = result[lag][0]['ssr_ftest'][1]
                    if p < min_p:
                        min_p = p
                        best_lag = lag

                p_values.loc[cause, effect] = min_p
                best_lags.loc[cause, effect] = best_lag

                if min_p < 0.05:
                    print(f"* {cause} → {effect}: p={min_p:.4f} at lag={best_lag} days")

            except Exception as e:
                pass

    print("\n" + "=" * 60)
    print(f"Significant relationships (p < 0.05): {(p_values < 0.05).sum().sum()}")
    print("=" * 60)

    return p_values, best_lags


# ============================================================
# 4. VALIDATION TEST 2: CROSS-CORRELATION
# ============================================================

def analyze_cross_correlation(crowding_data, max_lag=30):
    """
    Analyze cross-correlation at different lags
    Peak at positive lag means first variable leads second
    """
    from scipy import stats

    factors = crowding_data.columns.tolist()

    print("=" * 60)
    print("CROSS-CORRELATION ANALYSIS")
    print("=" * 60)

    results = []

    for i, f1 in enumerate(factors):
        for j, f2 in enumerate(factors):
            if i >= j:
                continue

            # Compute cross-correlation at different lags
            correlations = []
            lags = range(-max_lag, max_lag + 1)

            for lag in lags:
                if lag < 0:
                    corr = crowding_data[f1].iloc[-lag:].corr(
                        crowding_data[f2].iloc[:lag]
                    )
                elif lag > 0:
                    corr = crowding_data[f1].iloc[:-lag].corr(
                        crowding_data[f2].iloc[lag:]
                    )
                else:
                    corr = crowding_data[f1].corr(crowding_data[f2])
                correlations.append(corr)

            # Find peak
            peak_idx = np.argmax(np.abs(correlations))
            peak_lag = list(lags)[peak_idx]
            peak_corr = correlations[peak_idx]

            results.append({
                'factor1': f1,
                'factor2': f2,
                'peak_lag': peak_lag,
                'peak_corr': peak_corr
            })

            if abs(peak_corr) > 0.3 and peak_lag != 0:
                leader = f1 if peak_lag > 0 else f2
                follower = f2 if peak_lag > 0 else f1
                print(f"* {leader} leads {follower} by {abs(peak_lag)} days (corr={peak_corr:.3f})")

    return pd.DataFrame(results)


# ============================================================
# 5. VALIDATION TEST 3: VAR IMPULSE RESPONSE
# ============================================================

def analyze_var_impulse_response(crowding_data, maxlags=10, periods=30):
    """
    Fit VAR model and analyze impulse response functions
    Shows how a shock to one factor propagates to others
    """
    from statsmodels.tsa.api import VAR

    print("=" * 60)
    print("VAR IMPULSE RESPONSE ANALYSIS")
    print("=" * 60)

    # Fit VAR
    model = VAR(crowding_data)

    # Select optimal lag using AIC
    lag_order = model.select_order(maxlags=maxlags)
    print(f"Optimal lag order (AIC): {lag_order.aic}")

    # Fit with optimal lag
    results = model.fit(lag_order.aic)

    # Compute impulse response
    irf = results.irf(periods=periods)

    # Find significant responses
    print("\nSignificant impulse responses:")
    factors = crowding_data.columns.tolist()

    significant_pairs = []
    for i, shock in enumerate(factors):
        for j, response in enumerate(factors):
            if i == j:
                continue

            # Get cumulative response
            cum_response = irf.cum_effects[periods-1, j, i]

            # Check if significant (simple heuristic: > 0.1 std)
            if abs(cum_response) > 0.1:
                significant_pairs.append({
                    'shock': shock,
                    'response': response,
                    'cumulative_effect': cum_response
                })
                print(f"* Shock to {shock} → {response}: cumulative effect = {cum_response:.3f}")

    return results, irf, pd.DataFrame(significant_pairs)


# ============================================================
# 6. MAIN: RUN ALL VALIDATION TESTS
# ============================================================

def run_validation_tests():
    """
    Run all validation tests to check if causal relationships exist
    """
    print("=" * 60)
    print("V10 VALIDATION: CAUSAL FACTOR CROWDING")
    print("=" * 60)
    print()

    # 1. Load data
    print("Loading Fama-French factor data...")
    try:
        factors = download_ff_factors()
    except Exception as e:
        print(f"Error: {e}")
        print("\nCreating synthetic data for demonstration...")
        factors = create_synthetic_factors()

    print()

    # 2. Compute crowding proxy
    print("Computing crowding proxy...")
    crowding = compute_simple_crowding(factors, window=60)
    print(f"Crowding data shape: {crowding.shape}")
    print()

    # 3. Granger Causality Test
    p_values, best_lags = test_granger_causality(crowding, maxlag=20)
    print()

    # 4. Cross-Correlation Analysis
    xcorr_results = analyze_cross_correlation(crowding, max_lag=30)
    print()

    # 5. VAR Impulse Response
    try:
        var_results, irf, sig_pairs = analyze_var_impulse_response(crowding, maxlags=10)
    except Exception as e:
        print(f"VAR analysis error: {e}")
        var_results, irf, sig_pairs = None, None, None

    print()
    print("=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    n_granger = (p_values < 0.05).sum().sum() if p_values is not None else 0
    n_xcorr = len(xcorr_results[xcorr_results['peak_lag'].abs() > 5]) if xcorr_results is not None else 0
    n_var = len(sig_pairs) if sig_pairs is not None else 0

    print(f"Granger causal relationships found: {n_granger}")
    print(f"Lead-lag relationships (>5 days): {n_xcorr}")
    print(f"Significant VAR impulse responses: {n_var}")

    if n_granger > 0 or n_xcorr > 0:
        print("\n>>> VALIDATION PASSED: Causal structure exists!")
        print(">>> Proceed to full causal discovery (DYNOTEARS/PCMCI)")
    else:
        print("\n>>> VALIDATION FAILED: No clear causal structure")
        print(">>> Review crowding proxy construction")

    return {
        'factors': factors,
        'crowding': crowding,
        'granger_pvalues': p_values,
        'granger_lags': best_lags,
        'xcorr': xcorr_results,
        'var_results': var_results,
        'irf': irf
    }


def create_synthetic_factors():
    """
    Create synthetic factor data for testing when real data unavailable
    """
    np.random.seed(42)
    n = 5000  # ~20 years of daily data

    dates = pd.date_range('2000-01-01', periods=n, freq='B')

    # Create factors with some causal structure
    # MOM leads HML with lag 5
    # SMB leads CMA with lag 10

    mom = np.random.randn(n) * 0.01
    smb = np.random.randn(n) * 0.008

    # HML partially caused by MOM (lagged)
    hml = np.zeros(n)
    for t in range(5, n):
        hml[t] = 0.3 * mom[t-5] + 0.7 * np.random.randn() * 0.009

    # CMA partially caused by SMB (lagged)
    cma = np.zeros(n)
    for t in range(10, n):
        cma[t] = 0.25 * smb[t-10] + 0.75 * np.random.randn() * 0.007

    # RMW independent
    rmw = np.random.randn(n) * 0.006

    # MKT affected by all (contemporaneous)
    mkt = 0.1 * mom + 0.1 * smb + 0.1 * hml + np.random.randn(n) * 0.012

    factors = pd.DataFrame({
        'MKT': mkt,
        'SMB': smb,
        'HML': hml,
        'RMW': rmw,
        'CMA': cma,
        'MOM': mom
    }, index=dates)

    print("Created synthetic factor data with embedded causal structure:")
    print("  - MOM → HML (lag 5)")
    print("  - SMB → CMA (lag 10)")

    return factors


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    results = run_validation_tests()
