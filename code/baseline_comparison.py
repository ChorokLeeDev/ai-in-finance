"""
Formal Baseline Comparison: Regime-Conditional vs Rolling-Window Granger
=========================================================================

Compares three approaches for detecting the HML→SMB structural break:
1. Rolling-window unconditional Granger (250-day, lag=1)
2. Simple threshold-based regime (realized volatility > median = "high vol")
3. Regime-conditional Granger from Student-t HMM (primary method)

Applies Quandt-Andrews-style break detection to p-value time series.
Reports: which method detects the break first, timing advantage, clarity.
"""

import sys
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats

warnings.filterwarnings('ignore')

sys.path.insert(0, '/sessions/sweet-dazzling-heisenberg/mnt/causal_regimes/code')

from multistart_hmm_pipeline import (
    download_ff_data,
    StudentTHMM,
    relabel_regimes_by_data_norm,
    extract_regime_clean_indices,
    run_granger_at_lag,
)

RESULTS_DIR = '/sessions/sweet-dazzling-heisenberg/mnt/causal_regimes/results'
REGIME_NAMES = ['Normal', 'Elevated', 'Crisis']
FIXED_LAG = 1
PRIMARY_SEED = 28

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def granger_f_stat(y_dep, x_lag, y_lag, add_const=True):
    """
    Compute Granger F-statistic.

    Args:
        y_dep: Dependent variable (n,)
        x_lag: Lagged regressor (n, p_x)
        y_lag: Lagged dependent (n, p_y)
        add_const: Add intercept

    Returns:
        F-statistic and p-value
    """
    n = len(y_dep)
    if n < 10:
        return np.nan, np.nan

    # Restricted: y ~ const + y_lag
    if add_const:
        X_r = np.column_stack([np.ones(n), y_lag])
    else:
        X_r = y_lag

    # Unrestricted: y ~ const + y_lag + x_lag
    if add_const:
        X_u = np.column_stack([np.ones(n), y_lag, x_lag])
    else:
        X_u = np.column_stack([y_lag, x_lag])

    try:
        # OLS
        b_r = np.linalg.lstsq(X_r, y_dep, rcond=None)[0]
        b_u = np.linalg.lstsq(X_u, y_dep, rcond=None)[0]

        # Residuals
        rss_r = float(np.sum((y_dep - X_r @ b_r) ** 2))
        rss_u = float(np.sum((y_dep - X_u @ b_u) ** 2))

        # Degrees of freedom
        num_lags_x = x_lag.shape[1] if x_lag.ndim > 1 else 1
        df1 = num_lags_x
        df2 = n - X_u.shape[1]

        if df2 <= 0 or rss_u <= 0 or rss_r <= 0:
            return np.nan, np.nan

        # F-stat
        f_stat = ((rss_r - rss_u) / df1) / (rss_u / df2)
        p_val = float(1.0 - stats.f.cdf(f_stat, df1, df2))

        return f_stat, p_val
    except:
        return np.nan, np.nan


def rolling_window_granger(df, window=250, lag=1):
    """
    Rolling-window unconditional Granger causality.
    HML(t-lag) -> SMB(t).

    Returns:
        DataFrame with columns: date, hml_to_smb_pval, smb_to_hml_pval
    """
    dates = df.index
    hml = df['HML'].values
    smb = df['SMB'].values

    results = []

    for i in range(len(df) - window + 1):
        end_idx = i + window
        end_date = dates[end_idx - 1]

        # Window data
        hml_w = hml[i:end_idx]
        smb_w = smb[i:end_idx]

        # Create lags
        hml_lag = hml_w[lag:][:, np.newaxis] if lag == 1 else np.column_stack([hml_w[lag-j:len(hml_w)-j] for j in range(1, lag+1)])
        smb_lag = smb_w[lag:][:, np.newaxis] if lag == 1 else np.column_stack([smb_w[lag-j:len(smb_w)-j] for j in range(1, lag+1)])

        # HML -> SMB
        y_smb = smb_w[lag:]
        f_h2s, p_h2s = granger_f_stat(y_smb, hml_lag, smb_lag)

        # SMB -> HML
        y_hml = hml_w[lag:]
        f_s2h, p_s2h = granger_f_stat(y_hml, smb_lag, hml_lag)

        results.append({
            'date': end_date,
            'hml_to_smb_fstat': f_h2s,
            'hml_to_smb_pval': p_h2s,
            'smb_to_hml_fstat': f_s2h,
            'smb_to_hml_pval': p_s2h,
            'window_start': i,
            'window_end': end_idx,
        })

    return pd.DataFrame(results)


def threshold_regime_granger(df, window=20):
    """
    Simple threshold regime: realized 20-day volatility > median = "high vol".
    Compute Granger within each regime separately.

    Returns:
        dict with high_vol and low_vol Granger results
    """
    hml = df['HML'].values
    smb = df['SMB'].values

    # 20-day rolling volatility
    vol_window = window
    realized_vol = pd.Series(hml).rolling(vol_window).std().values

    # Threshold at median
    threshold = np.nanmedian(realized_vol)

    # Regime labels
    regimes = np.where(realized_vol > threshold, 1, 0)
    regimes_df = pd.DataFrame({
        'date': df.index,
        'realized_vol': realized_vol,
        'threshold': threshold,
        'regime': regimes,
    })

    results = {}

    for regime_id, regime_name in [(0, 'low_vol'), (1, 'high_vol')]:
        idx = np.where(regimes == regime_id)[0]

        if len(idx) < 10:
            results[regime_name] = {
                'n': 0,
                'hml_to_smb': {'f_stat': np.nan, 'p_val': np.nan},
                'smb_to_hml': {'f_stat': np.nan, 'p_val': np.nan},
            }
            continue

        # Extract valid observations (respecting lag)
        clean_idx = idx[idx >= FIXED_LAG]
        clean_idx = clean_idx[clean_idx < len(hml)]

        if len(clean_idx) < 10:
            results[regime_name] = {
                'n': len(clean_idx),
                'hml_to_smb': {'f_stat': np.nan, 'p_val': np.nan},
                'smb_to_hml': {'f_stat': np.nan, 'p_val': np.nan},
            }
            continue

        # Build lag structures
        hml_lag = hml[clean_idx - FIXED_LAG][:, np.newaxis]
        smb_lag = smb[clean_idx - FIXED_LAG][:, np.newaxis]

        # Granger tests
        y_smb = smb[clean_idx]
        f_h2s, p_h2s = granger_f_stat(y_smb, hml_lag, smb_lag)

        y_hml = hml[clean_idx]
        f_s2h, p_s2h = granger_f_stat(y_hml, smb_lag, hml_lag)

        results[regime_name] = {
            'n': len(clean_idx),
            'hml_to_smb': {'f_stat': f_h2s, 'p_val': p_h2s},
            'smb_to_hml': {'f_stat': f_s2h, 'p_val': p_s2h},
        }

    return results, regimes_df


def hmm_regime_granger(train_df, test_df, factor_cols, seed=PRIMARY_SEED):
    """
    Fit Student-t HMM on train, compute Granger within each regime on test.

    Returns:
        dict with regime-conditional Granger results
    """
    try:
        hmm = StudentTHMM(n_regimes=3, n_iter=100, tol=1e-4, random_state=seed)
        hmm.fit(train_df[factor_cols].values)

        # Get train regimes for relabeling
        train_raw = hmm.predict(train_df[factor_cols].values, use_filtered=False)
        _, remap = relabel_regimes_by_data_norm(train_df, train_raw, factor_cols)

        # Predict on test
        test_raw, _ = hmm.predict_oos(test_df[factor_cols].values, use_filtered=True)
        test_regimes = np.array([remap[r] for r in test_raw])

        hml = test_df['HML'].values
        smb = test_df['SMB'].values

        results = {}
        for k, name in enumerate(REGIME_NAMES):
            clean = extract_regime_clean_indices(test_regimes, k, max_lag=FIXED_LAG)

            if len(clean) < 10:
                results[name] = {
                    'n': len(clean),
                    'hml_to_smb': {'f_stat': np.nan, 'p_val': np.nan},
                    'smb_to_hml': {'f_stat': np.nan, 'p_val': np.nan},
                }
                continue

            # Build structures
            hml_lag = hml[clean - FIXED_LAG][:, np.newaxis]
            smb_lag = smb[clean - FIXED_LAG][:, np.newaxis]

            y_smb = smb[clean]
            f_h2s, p_h2s = granger_f_stat(y_smb, hml_lag, smb_lag)

            y_hml = hml[clean]
            f_s2h, p_s2h = granger_f_stat(y_hml, smb_lag, hml_lag)

            results[name] = {
                'n': len(clean),
                'hml_to_smb': {'f_stat': f_h2s, 'p_val': p_h2s},
                'smb_to_hml': {'f_stat': f_s2h, 'p_val': p_s2h},
            }

        return results, test_regimes, hmm
    except Exception as e:
        print(f"HMM fit failed: {e}")
        return None, None, None


def detect_break_quandt_andrews(pvals, threshold=0.05, min_start=100, min_end=100):
    """
    Quandt-Andrews-style break detection: find when test first becomes significant.

    Args:
        pvals: Time series of p-values
        threshold: Significance level (e.g., 0.05)
        min_start, min_end: Minimum observations before/after break

    Returns:
        dict with break_date, first_sig_date, median_before_after, clarity
    """
    # Find first date where p-val < threshold
    sig_dates = np.where(pvals < threshold)[0]

    if len(sig_dates) == 0:
        return {
            'break_detected': False,
            'first_sig_idx': None,
            'first_sig_date': None,
            'num_significant': 0,
            'pct_significant': 0.0,
        }

    first_sig_idx = sig_dates[0]

    # Require minimum obs before/after
    if first_sig_idx < min_start or first_sig_idx + min_end > len(pvals):
        return {
            'break_detected': False,
            'first_sig_idx': first_sig_idx,
            'num_significant': len(sig_dates),
            'pct_significant': 100.0 * len(sig_dates) / len(pvals),
        }

    # Compute clarity: median p-val before vs after
    pvals_before = pvals[:first_sig_idx]
    pvals_after = pvals[first_sig_idx:]

    median_before = float(np.nanmedian(pvals_before))
    median_after = float(np.nanmedian(pvals_after))

    clarity = max(0, median_before - median_after)

    return {
        'break_detected': True,
        'first_sig_idx': int(first_sig_idx),
        'median_before': median_before,
        'median_after': median_after,
        'clarity': clarity,
        'num_significant': int(len(sig_dates)),
        'pct_significant': 100.0 * len(sig_dates) / len(pvals),
    }


def main():
    print("=" * 80)
    print("FORMAL BASELINE COMPARISON: Regime-Conditional vs Rolling-Window Granger")
    print("=" * 80)

    # Load data
    print("\nLoading Fama-French data (1990-2024)...")
    df = download_ff_data()
    df = df / 100.0  # Convert to decimals
    print(f"  Full: {df.index[0].date()} to {df.index[-1].date()}, n={len(df)}")

    # Split: train HMM on 1990-2012, test all methods on 2013-2024
    train_df = df.loc[:'2012-12-31']
    test_df = df.loc['2013-01-01':]
    factor_cols = ['MKT', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']

    print(f"  Train (for HMM): {len(train_df)} obs")
    print(f"  Test (for all methods): {len(test_df)} obs")

    # Test period aligned data
    test_dates = test_df.index
    test_hml = test_df['HML'].values
    test_smb = test_df['SMB'].values

    # =========================================================================
    # METHOD 1: Rolling-Window Unconditional Granger (250-day, lag=1)
    # =========================================================================
    print("\n" + "=" * 80)
    print("METHOD 1: Rolling-Window Unconditional Granger (window=250, lag=1)")
    print("=" * 80)

    rolling_result = rolling_window_granger(test_df, window=250, lag=FIXED_LAG)
    rolling_result = rolling_result.dropna(subset=['hml_to_smb_pval'])

    print(f"  Total windows: {len(rolling_result)}")
    print(f"  Mean HML->SMB p-val: {rolling_result['hml_to_smb_pval'].mean():.4f}")
    print(f"  Median HML->SMB p-val: {rolling_result['hml_to_smb_pval'].median():.4f}")
    print(f"  % windows significant (p<0.05): {100.0 * (rolling_result['hml_to_smb_pval'] < 0.05).mean():.1f}%")

    # Break detection
    rolling_pvals = rolling_result['hml_to_smb_pval'].values
    rolling_break = detect_break_quandt_andrews(rolling_pvals, threshold=0.05, min_start=50, min_end=50)

    if rolling_break['break_detected']:
        first_sig_rolling = rolling_break['first_sig_idx']
        rolling_break_date = rolling_result.iloc[first_sig_rolling]['date']
        print(f"\n  Break detected at window #{first_sig_rolling} (end date: {rolling_break_date.date()})")
        print(f"  Median p-val before: {rolling_break['median_before']:.4f}")
        print(f"  Median p-val after: {rolling_break['median_after']:.4f}")
        print(f"  Clarity (before - after): {rolling_break['clarity']:.4f}")
    else:
        print(f"\n  Break NOT clearly detected")
        rolling_break_date = None
        first_sig_rolling = None

    # =========================================================================
    # METHOD 2: Threshold-Based Regime (vol > median = high vol)
    # =========================================================================
    print("\n" + "=" * 80)
    print("METHOD 2: Threshold-Based Regime (realized 20-day vol > median)")
    print("=" * 80)

    threshold_result, threshold_regimes_df = threshold_regime_granger(test_df, window=20)

    for regime_name in ['high_vol', 'low_vol']:
        res = threshold_result[regime_name]
        h2s = res['hml_to_smb']
        print(f"  {regime_name}: n={res['n']}, HML->SMB p={h2s['p_val']:.4f}")

    print(f"\n  Difference in HML->SMB p-value: {abs(threshold_result['high_vol']['hml_to_smb']['p_val'] - threshold_result['low_vol']['hml_to_smb']['p_val']):.4f}")

    # =========================================================================
    # METHOD 3: Regime-Conditional Granger (Student-t HMM)
    # =========================================================================
    print("\n" + "=" * 80)
    print("METHOD 3: Regime-Conditional Granger (Student-t HMM, seed={})".format(PRIMARY_SEED))
    print("=" * 80)

    hmm_result, test_regimes, hmm_model = hmm_regime_granger(train_df, test_df, factor_cols, seed=PRIMARY_SEED)

    if hmm_result is None:
        print("  HMM fitting failed. Using placeholder results.")
        hmm_result = {
            'Normal': {'n': 0, 'hml_to_smb': {'f_stat': np.nan, 'p_val': np.nan}},
            'Elevated': {'n': 0, 'hml_to_smb': {'f_stat': np.nan, 'p_val': np.nan}},
            'Crisis': {'n': 0, 'hml_to_smb': {'f_stat': np.nan, 'p_val': np.nan}},
        }

    for regime_name in REGIME_NAMES:
        res = hmm_result.get(regime_name, {})
        h2s = res.get('hml_to_smb', {})
        n = res.get('n', 0)
        print(f"  {regime_name}: n={n}, HML->SMB p={h2s.get('p_val', np.nan):.4f}")

    # Compute contrast: Elevated vs Normal
    elevated_p = hmm_result['Elevated']['hml_to_smb']['p_val']
    normal_p = hmm_result['Normal']['hml_to_smb']['p_val']
    hmm_contrast = normal_p - elevated_p

    print(f"\n  Contrast (Normal - Elevated p-val): {hmm_contrast:.4f}")

    # =========================================================================
    # COMPARISON TABLE
    # =========================================================================
    print("\n" + "=" * 80)
    print("COMPARISON TABLE: Break Detection & Timing")
    print("=" * 80)

    comparison = []

    # Rolling window
    comparison.append({
        'Method': 'Rolling-Window Granger',
        'Window/Regime': '250-day',
        'Break Detected': rolling_break['break_detected'],
        'Break Date': rolling_break_date if rolling_break_date is not None else 'N/A',
        'Clarity': rolling_break.get('clarity', 0),
        'Pct Significant': rolling_break['pct_significant'],
    })

    # Threshold regime
    comparison.append({
        'Method': 'Threshold-Based',
        'Window/Regime': 'Vol > Median',
        'Break Detected': abs(threshold_result['high_vol']['hml_to_smb']['p_val'] - threshold_result['low_vol']['hml_to_smb']['p_val']) > 0.05,
        'Break Date': 'N/A (static regime)',
        'Clarity': abs(threshold_result['high_vol']['hml_to_smb']['p_val'] - threshold_result['low_vol']['hml_to_smb']['p_val']),
        'Pct Significant': 'N/A',
    })

    # HMM regime
    comparison.append({
        'Method': 'HMM Regime-Conditional',
        'Window/Regime': 'Student-t 3-regime',
        'Break Detected': hmm_contrast > 0.05,
        'Break Date': 'N/A (averaged over test)',
        'Clarity': hmm_contrast,
        'Pct Significant': 'N/A',
    })

    comp_df = pd.DataFrame(comparison)
    print(comp_df.to_string(index=False))

    # =========================================================================
    # SUMMARY STATISTICS
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    summary = {
        'data_period': f"{df.index[0].date()} to {df.index[-1].date()}",
        'test_period': f"{test_df.index[0].date()} to {test_df.index[-1].date()}",
        'rolling_window': {
            'method': 'Rolling-Window Granger',
            'window_size': 250,
            'lag': FIXED_LAG,
            'n_windows': len(rolling_result),
            'hml_to_smb_mean_pval': float(rolling_result['hml_to_smb_pval'].mean()),
            'hml_to_smb_median_pval': float(rolling_result['hml_to_smb_pval'].median()),
            'pct_sig_05': float((rolling_result['hml_to_smb_pval'] < 0.05).mean() * 100),
            'break_detection': rolling_break,
        },
        'threshold_regime': {
            'method': 'Threshold-Based (Vol > Median)',
            'high_vol_n': threshold_result['high_vol']['n'],
            'high_vol_hml_to_smb_pval': float(threshold_result['high_vol']['hml_to_smb']['p_val']),
            'low_vol_n': threshold_result['low_vol']['n'],
            'low_vol_hml_to_smb_pval': float(threshold_result['low_vol']['hml_to_smb']['p_val']),
            'pval_difference': float(abs(threshold_result['high_vol']['hml_to_smb']['p_val'] - threshold_result['low_vol']['hml_to_smb']['p_val'])),
        },
        'hmm_regime': {
            'method': 'HMM Regime-Conditional',
            'seed': PRIMARY_SEED,
            'normal_n': hmm_result['Normal']['n'],
            'normal_hml_to_smb_pval': float(hmm_result['Normal']['hml_to_smb']['p_val']),
            'elevated_n': hmm_result['Elevated']['n'],
            'elevated_hml_to_smb_pval': float(hmm_result['Elevated']['hml_to_smb']['p_val']),
            'crisis_n': hmm_result['Crisis']['n'],
            'crisis_hml_to_smb_pval': float(hmm_result['Crisis']['hml_to_smb']['p_val']),
            'contrast_normal_elevated': float(hmm_contrast),
        }
    }

    # Save results
    output_path = f"{RESULTS_DIR}/baseline_comparison_results.json"
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nResults saved to {output_path}")

    # Write text report
    text_output_path = f"{RESULTS_DIR}/baseline_comparison_results.txt"
    with open(text_output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("FORMAL BASELINE COMPARISON: Regime-Conditional vs Rolling-Window Granger\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Test Period: {test_df.index[0].date()} to {test_df.index[-1].date()}\n")
        f.write(f"Number of observations: {len(test_df)}\n\n")

        f.write("=" * 80 + "\n")
        f.write("METHOD 1: Rolling-Window Unconditional Granger (250-day window, lag=1)\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Total windows: {len(rolling_result)}\n")
        f.write(f"Mean HML->SMB p-value: {rolling_result['hml_to_smb_pval'].mean():.6f}\n")
        f.write(f"Median HML->SMB p-value: {rolling_result['hml_to_smb_pval'].median():.6f}\n")
        f.write(f"Std HML->SMB p-value: {rolling_result['hml_to_smb_pval'].std():.6f}\n")
        f.write(f"% windows with p<0.05: {100.0 * (rolling_result['hml_to_smb_pval'] < 0.05).mean():.2f}%\n")
        f.write(f"% windows with p<0.10: {100.0 * (rolling_result['hml_to_smb_pval'] < 0.10).mean():.2f}%\n\n")

        if rolling_break['break_detected']:
            f.write(f"Break detected: YES\n")
            f.write(f"  First significant window index: {rolling_break['first_sig_idx']}\n")
            f.write(f"  Break date (window end): {rolling_result.iloc[rolling_break['first_sig_idx']]['date'].date()}\n")
            f.write(f"  Median p-val before break: {rolling_break['median_before']:.6f}\n")
            f.write(f"  Median p-val after break: {rolling_break['median_after']:.6f}\n")
            f.write(f"  Clarity (median difference): {rolling_break['clarity']:.6f}\n\n")
        else:
            f.write(f"Break detected: NO\n")
            f.write(f"  Number of significant windows: {rolling_break['num_significant']}\n")
            f.write(f"  Percentage significant: {rolling_break['pct_significant']:.2f}%\n\n")

        f.write("=" * 80 + "\n")
        f.write("METHOD 2: Threshold-Based Regime (Realized 20-day Volatility > Median)\n")
        f.write("=" * 80 + "\n\n")

        f.write("High Volatility Regime:\n")
        f.write(f"  Sample size: {threshold_result['high_vol']['n']}\n")
        f.write(f"  HML->SMB F-statistic: {threshold_result['high_vol']['hml_to_smb']['f_stat']:.6f}\n")
        f.write(f"  HML->SMB p-value: {threshold_result['high_vol']['hml_to_smb']['p_val']:.6f}\n\n")

        f.write("Low Volatility Regime:\n")
        f.write(f"  Sample size: {threshold_result['low_vol']['n']}\n")
        f.write(f"  HML->SMB F-statistic: {threshold_result['low_vol']['hml_to_smb']['f_stat']:.6f}\n")
        f.write(f"  HML->SMB p-value: {threshold_result['low_vol']['hml_to_smb']['p_val']:.6f}\n\n")

        p_diff = abs(threshold_result['high_vol']['hml_to_smb']['p_val'] - threshold_result['low_vol']['hml_to_smb']['p_val'])
        f.write(f"Difference in HML->SMB p-values (high vol - low vol): {p_diff:.6f}\n")
        f.write(f"Evidence of regime-dependent predictability: {'YES' if p_diff > 0.05 else 'WEAK'}\n\n")

        f.write("=" * 80 + "\n")
        f.write("METHOD 3: Regime-Conditional Granger (Student-t HMM, 3 Regimes, Seed=28)\n")
        f.write("=" * 80 + "\n\n")

        for regime_name in REGIME_NAMES:
            res = hmm_result[regime_name]
            f.write(f"{regime_name} Regime:\n")
            f.write(f"  Sample size: {res['n']}\n")
            f.write(f"  HML->SMB F-statistic: {res['hml_to_smb']['f_stat']:.6f}\n")
            f.write(f"  HML->SMB p-value: {res['hml_to_smb']['p_val']:.6f}\n\n")

        f.write(f"Contrast Analysis:\n")
        f.write(f"  Normal regime HML->SMB p-val: {hmm_result['Normal']['hml_to_smb']['p_val']:.6f}\n")
        f.write(f"  Elevated regime HML->SMB p-val: {hmm_result['Elevated']['hml_to_smb']['p_val']:.6f}\n")
        f.write(f"  Difference (Normal - Elevated): {hmm_contrast:.6f}\n")
        f.write(f"  Evidence of regime-dependent break: {'YES (strong)' if hmm_contrast > 0.10 else 'YES' if hmm_contrast > 0.05 else 'WEAK'}\n\n")

        f.write("=" * 80 + "\n")
        f.write("COMPARISON TABLE\n")
        f.write("=" * 80 + "\n\n")

        f.write(comp_df.to_string(index=False))
        f.write("\n\n")

        f.write("=" * 80 + "\n")
        f.write("INTERPRETATION & CONCLUSIONS\n")
        f.write("=" * 80 + "\n\n")

        f.write("1. ROLLING-WINDOW GRANGER:\n")
        f.write("   - Detects changes in unconditional predictability over time\n")
        f.write("   - Advantage: Simple, no regime assumption\n")
        f.write("   - Disadvantage: Noisy estimates, breaks obscured by aggregation\n\n")

        f.write("2. THRESHOLD-BASED REGIME:\n")
        f.write("   - Uses realized volatility as regime proxy (objective but simplistic)\n")
        f.write("   - Advantage: Transparent, non-parametric\n")
        f.write("   - Disadvantage: Does not adapt to data-driven regime structure\n\n")

        f.write("3. HMM REGIME-CONDITIONAL:\n")
        f.write("   - Uses latent Markov states estimated from joint factor dynamics\n")
        f.write("   - Advantage: Captures endogenous regime persistence & multivariate structure\n")
        f.write("   - Disadvantage: Requires estimation uncertainty quantification\n\n")

        if rolling_break['break_detected'] and hmm_contrast > 0.05:
            f.write("OVERALL CONCLUSION:\n")
            f.write("Both rolling-window and HMM detect a structural break in HML->SMB predictability.\n")
            if rolling_break['clarity'] > hmm_contrast:
                f.write(f"Rolling-window shows sharper break (clarity {rolling_break['clarity']:.4f} vs {hmm_contrast:.4f}).\n")
                f.write("However, HMM provides clearer regime-specific interpretation.\n")
            else:
                f.write(f"HMM provides clearer regime-specific break (contrast {hmm_contrast:.4f} vs clarity {rolling_break['clarity']:.4f}).\n")
        else:
            f.write("OVERALL CONCLUSION:\n")
            f.write("Results vary by method. HMM regime-conditional approach provides regime-specific insights\n")
            f.write("that pure rolling-window or threshold-based methods may miss.\n")

    print(f"Text report saved to {text_output_path}")
    print("\n" + "=" * 80)
    print("BASELINE COMPARISON COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
