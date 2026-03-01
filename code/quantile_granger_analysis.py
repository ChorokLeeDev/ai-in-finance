#!/usr/bin/env python3
"""
Quantile Granger Causality Analysis
====================================

Tests whether the reverse causal asymmetry (SMB→HML stronger than HML→SMB)
is driven by tail dependence. Runs quantile regression at [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
and tests for coefficient equality across quantiles (Wald test).

Hypothesis: SMB→HML shows larger coefficients at tails (0.05, 0.95) vs median (0.50),
explaining why Transfer Entropy (which captures nonlinear/tail effects) detects it,
while linear Granger causality (median-focused) does not.
"""

import sys
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import f as f_dist

warnings.filterwarnings('ignore')

# Try to import statsmodels for quantile regression
try:
    import statsmodels.api as sm
    from statsmodels.regression.quantile_regression import QuantReg
except ImportError:
    print("WARNING: statsmodels not available, attempting pip install...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--break-system-packages", "statsmodels"])
    import statsmodels.api as sm
    from statsmodels.regression.quantile_regression import QuantReg

import urllib.request
import zipfile
import io

RESULTS_DIR = '/sessions/sweet-dazzling-heisenberg/mnt/causal_regimes/results'
CODE_DIR = '/sessions/sweet-dazzling-heisenberg/mnt/causal_regimes/code'

# Add code dir to path for importing multistart_hmm_pipeline
sys.path.insert(0, CODE_DIR)

try:
    from multistart_hmm_pipeline import download_ff_data, StudentTHMM, relabel_regimes_by_data_norm
except ImportError:
    print("Could not import multistart_hmm_pipeline. Using alternative data loading...")
    download_ff_data = None


def load_ff_data_direct():
    """
    Direct Fama-French data download (standalone version).
    """
    print("Downloading Fama-French 5 factors (daily)...")
    url5 = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip'

    with urllib.request.urlopen(url5, timeout=60) as response:
        data = response.read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        csv_name = z.namelist()[0]
        with z.open(csv_name) as f:
            df5 = pd.read_csv(f, skiprows=3)

    df5.columns = df5.columns.str.strip()
    df5 = df5.rename(columns={df5.columns[0]: 'Date'})
    df5 = df5[df5['Date'].astype(str).str.match(r'^\d{8}$')]
    df5['Date'] = pd.to_datetime(df5['Date'], format='%Y%m%d')
    for col in ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']:
        df5[col] = pd.to_numeric(df5[col], errors='coerce')
    df5 = df5.set_index('Date').dropna()

    print("Downloading Momentum factor (daily)...")
    url_mom = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip'
    with urllib.request.urlopen(url_mom, timeout=60) as response:
        data = response.read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        csv_name = z.namelist()[0]
        with z.open(csv_name) as f:
            mom = pd.read_csv(f, skiprows=13)

    mom.columns = mom.columns.str.strip()
    mom = mom.rename(columns={mom.columns[0]: 'Date'})
    mom = mom[mom['Date'].astype(str).str.match(r'^\d{8}$')]
    mom['Date'] = pd.to_datetime(mom['Date'], format='%Y%m%d')
    mom = mom.rename(columns={mom.columns[1]: 'MOM'})
    mom['MOM'] = pd.to_numeric(mom['MOM'], errors='coerce')
    mom = mom.set_index('Date').dropna()

    df = df5.join(mom[['MOM']], how='inner')
    df = df.rename(columns={'Mkt-RF': 'MKT'})
    df = df.drop('RF', axis=1, errors='ignore')
    df = df.loc['1990-01-01':'2024-12-31']

    print(f"Loaded {len(df)} trading days ({df.index[0].date()} to {df.index[-1].date()})")
    return df


def load_canonical_regimes():
    """Load canonical regime assignments from JSON."""
    path = f"{RESULTS_DIR}/canonical_regimes.json"
    print(f"Loading canonical regimes from {path}...")
    with open(path) as f:
        data = json.load(f)

    assignments = data['assignments']
    regime_df = pd.DataFrame(assignments)
    regime_df['date'] = pd.to_datetime(regime_df['date'])
    regime_df = regime_df.set_index('date').sort_index()

    regime_map = {'Normal': 0, 'Elevated': 1, 'Crisis': 2}
    regime_df['regime_id'] = regime_df['regime_name'].map(regime_map)

    print(f"Loaded {len(regime_df)} regime assignments")
    for name in ['Normal', 'Elevated', 'Crisis']:
        n = (regime_df['regime_name'] == name).sum()
        print(f"  {name}: {n} days")

    return regime_df


def extract_regime_clean_indices(regimes, regime_id, max_lag=1):
    """
    Extract indices where observation and all lags are in the specified regime.
    """
    valid_indices = []
    for t in range(max_lag, len(regimes)):
        all_same = True
        for lag in range(0, max_lag + 1):
            if regimes[t - lag] != regime_id:
                all_same = False
                break
        if all_same:
            valid_indices.append(t)
    return np.array(valid_indices, dtype=int)


def quantile_regression(y, X, quantiles, regime_name='Unknown'):
    """
    Fit quantile regression of y on X at specified quantiles.
    Returns dict with results for each quantile.
    """
    # Ensure X includes constant
    if X.shape[1] == 1:
        X = sm.add_constant(X)

    results = {}

    for q in quantiles:
        try:
            qreg = QuantReg(y, X)
            res = qreg.fit(q=q, p_tol=1e-5)

            # Extract coefficient for the lagged predictor (not the constant)
            coeff = float(res.params[1])  # Second parameter (first is constant)
            std_err = float(res.bse[1]) if len(res.bse) > 1 else np.nan
            t_stat = coeff / std_err if std_err > 0 else np.nan
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(y) - X.shape[1])) if not np.isnan(t_stat) else np.nan

            results[q] = {
                'quantile': q,
                'n_obs': len(y),
                'coefficient': coeff,
                'std_error': std_err,
                't_statistic': t_stat,
                'p_value': p_value,
                'lower_ci': coeff - 1.96 * std_err,
                'upper_ci': coeff + 1.96 * std_err,
            }
        except Exception as e:
            print(f"    WARNING: Quantile regression failed at q={q}: {str(e)}")
            results[q] = {
                'quantile': q,
                'n_obs': len(y),
                'coefficient': np.nan,
                'std_error': np.nan,
                't_statistic': np.nan,
                'p_value': np.nan,
                'lower_ci': np.nan,
                'upper_ci': np.nan,
            }

    return results


def wald_test_equality(coeff_dict, regime_name='Unknown'):
    """
    Test whether coefficients are equal across quantiles.
    Uses Wald test: H0: b(0.05) = b(0.25) = b(0.50) = b(0.75) = b(0.95)
    """
    quantiles = sorted([q for q in coeff_dict.keys()])
    coeffs = np.array([coeff_dict[q]['coefficient'] for q in quantiles])
    ses = np.array([coeff_dict[q]['std_error'] for q in quantiles])

    # Check for NaNs
    if np.isnan(coeffs).any() or np.isnan(ses).any():
        return {
            'n_quantiles': len(quantiles),
            'wald_stat': np.nan,
            'df': len(quantiles) - 1,
            'p_value': np.nan,
            'coeff_min': np.nanmin(coeffs),
            'coeff_max': np.nanmax(coeffs),
            'coeff_range': np.nanmax(coeffs) - np.nanmin(coeffs),
            'note': 'NaN values present'
        }

    # Simple Wald test: assume asymptotic normality
    # Test statistic = sum of squared standardized differences from mean
    mean_coeff = np.mean(coeffs)
    wald_stat = 0.0

    for i, (q, coeff, se) in enumerate(zip(quantiles, coeffs, ses)):
        if se > 0:
            wald_stat += ((coeff - mean_coeff) / se) ** 2

    df = len(quantiles) - 1
    p_value = 1.0 - stats.chi2.cdf(wald_stat, df) if df > 0 else np.nan

    return {
        'n_quantiles': len(quantiles),
        'quantiles_tested': quantiles,
        'wald_stat': wald_stat,
        'df': df,
        'p_value': p_value,
        'coeff_min': np.min(coeffs),
        'coeff_max': np.max(coeffs),
        'coeff_mean': mean_coeff,
        'coeff_range': np.max(coeffs) - np.min(coeffs),
        'tail_vs_median_ratio': coeffs[quantiles.index(0.95)] / coeffs[quantiles.index(0.50)] if len(quantiles) >= 2 else np.nan,
    }


def run_quantile_analysis(df_factors, df_regimes, regime_name='Normal', regime_id=0, lag=1):
    """
    Run quantile Granger analysis for a single regime.
    Tests both HML→SMB and SMB→HML.
    """
    print(f"\n{'='*80}")
    print(f"REGIME: {regime_name} (ID={regime_id})")
    print(f"{'='*80}")

    # Extract clean indices for this regime
    regimes_array = df_regimes['regime_id'].values
    clean_idx = extract_regime_clean_indices(regimes_array, regime_id, max_lag=lag)

    n_total = (regimes_array == regime_id).sum()
    n_clean = len(clean_idx)

    print(f"Total observations in regime: {n_total}")
    print(f"Clean observations (all lags in regime): {n_clean}")

    if n_clean < 50:
        print(f"WARNING: Too few clean observations ({n_clean}), skipping quantile analysis")
        return None

    # Get factor data aligned with regimes
    common_dates = df_factors.index.intersection(df_regimes.index)
    df_factors = df_factors.loc[common_dates]
    df_regimes = df_regimes.loc[common_dates]

    hml = df_factors['HML'].values
    smb = df_factors['SMB'].values

    # Prepare lagged arrays for clean indices
    y_smb = smb[clean_idx]
    y_hml = hml[clean_idx]
    x_hml_lag = hml[clean_idx - lag].reshape(-1, 1)  # SMB predicted by HML
    x_smb_lag = smb[clean_idx - lag].reshape(-1, 1)  # HML predicted by SMB

    quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

    # HML → SMB: regress SMB on lagged HML
    print(f"\n  Running quantile regression: SMB ~ lagged HML (HML→SMB)")
    hml_to_smb_results = quantile_regression(y_smb, x_hml_lag, quantiles, regime_name)

    # Test for equality across quantiles
    hml_to_smb_wald = wald_test_equality(hml_to_smb_results, regime_name)

    print(f"    Wald test (H0: equal coefficients across quantiles):")
    print(f"      Test stat = {hml_to_smb_wald['wald_stat']:.4f}, df={hml_to_smb_wald['df']}, p={hml_to_smb_wald['p_value']:.4f}")
    print(f"      Coeff range: [{hml_to_smb_wald['coeff_min']:.6f}, {hml_to_smb_wald['coeff_max']:.6f}]")
    print(f"      Tail/Median ratio (Q95/Q50): {hml_to_smb_wald['tail_vs_median_ratio']:.4f}")

    # SMB → HML: regress HML on lagged SMB
    print(f"\n  Running quantile regression: HML ~ lagged SMB (SMB→HML)")
    smb_to_hml_results = quantile_regression(y_hml, x_smb_lag, quantiles, regime_name)

    # Test for equality across quantiles
    smb_to_hml_wald = wald_test_equality(smb_to_hml_results, regime_name)

    print(f"    Wald test (H0: equal coefficients across quantiles):")
    print(f"      Test stat = {smb_to_hml_wald['wald_stat']:.4f}, df={smb_to_hml_wald['df']}, p={smb_to_hml_wald['p_value']:.4f}")
    print(f"      Coeff range: [{smb_to_hml_wald['coeff_min']:.6f}, {smb_to_hml_wald['coeff_max']:.6f}]")
    print(f"      Tail/Median ratio (Q95/Q50): {smb_to_hml_wald['tail_vs_median_ratio']:.4f}")

    return {
        'regime_name': regime_name,
        'regime_id': regime_id,
        'n_total': n_total,
        'n_clean': n_clean,
        'lag': lag,
        'quantiles': quantiles,
        'hml_to_smb': {
            'results': hml_to_smb_results,
            'wald_test': hml_to_smb_wald,
        },
        'smb_to_hml': {
            'results': smb_to_hml_results,
            'wald_test': smb_to_hml_wald,
        }
    }


def main():
    print("\n" + "="*80)
    print("QUANTILE GRANGER CAUSALITY ANALYSIS")
    print("="*80)
    print("Tests tail dependence hypothesis: reverse channel (SMB→HML) operates via tails")
    print("="*80)

    # Load data
    print("\nLoading Fama-French data...")
    df_factors = load_ff_data_direct() if download_ff_data is None else download_ff_data()
    df_factors = df_factors / 100.0  # Convert to decimals

    print("Loading canonical regime assignments...")
    df_regimes = load_canonical_regimes()

    # Align data
    common_dates = df_factors.index.intersection(df_regimes.index)
    df_factors = df_factors.loc[common_dates]
    df_regimes = df_regimes.loc[common_dates]
    print(f"Aligned {len(common_dates)} trading days")

    # Run analysis for each regime
    regime_configs = [
        ('Normal', 0),
        ('Elevated', 1),
        ('Crisis', 2),
    ]

    all_results = {}
    for regime_name, regime_id in regime_configs:
        result = run_quantile_analysis(df_factors, df_regimes, regime_name, regime_id, lag=1)
        if result is not None:
            all_results[regime_name] = result

    # Create summary
    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"{'Regime':<12} {'n_clean':>8} {'HML→SMB':<20} {'SMB→HML':<20}")
    print(f"{'':12} {'':>8} {'Coeff (Q50)':>10} {'Wald p':>10} {'Coeff (Q50)':>10} {'Wald p':>10}")
    print("-" * 80)

    for regime_name in ['Normal', 'Elevated', 'Crisis']:
        if regime_name in all_results:
            r = all_results[regime_name]
            h2s_coeff = r['hml_to_smb']['results'].get(0.50, {}).get('coefficient', np.nan)
            h2s_wald_p = r['hml_to_smb']['wald_test'].get('p_value', np.nan)
            s2h_coeff = r['smb_to_hml']['results'].get(0.50, {}).get('coefficient', np.nan)
            s2h_wald_p = r['smb_to_hml']['wald_test'].get('p_value', np.nan)

            print(f"{regime_name:<12} {r['n_clean']:>8d} "
                  f"{h2s_coeff:>10.6f} {h2s_wald_p:>10.4f} "
                  f"{s2h_coeff:>10.6f} {s2h_wald_p:>10.4f}")

    # Save detailed results
    output = {
        'description': 'Quantile Granger causality analysis: tests tail dependence hypothesis',
        'hypothesis': 'SMB→HML operates via tail dependence (stronger at Q=0.05, 0.95 vs Q=0.50)',
        'method': 'Quantile regression with Wald test for coefficient equality',
        'quantiles': [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95],
        'lag': 1,
        'data_range': f"{df_factors.index[0].date()} to {df_factors.index[-1].date()}",
        'n_total_days': len(df_factors),
        'regimes': all_results,
    }

    out_path = f"{RESULTS_DIR}/quantile_granger_results.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nDetailed results saved to {out_path}")

    # Also save a text summary
    txt_path = f"{RESULTS_DIR}/quantile_granger_results.txt"
    with open(txt_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("QUANTILE GRANGER CAUSALITY ANALYSIS - TEXT SUMMARY\n")
        f.write("="*80 + "\n\n")

        f.write("HYPOTHESIS:\n")
        f.write("The reverse causal channel (SMB→HML) operates primarily through tail dependence,\n")
        f.write("explaining why Transfer Entropy (nonlinear, tail-sensitive) detects it strongly\n")
        f.write("while linear Granger causality (median-focused) does not.\n\n")

        f.write("METHOD:\n")
        f.write("Quantile regression of HML on lagged SMB (and vice versa) at quantiles\n")
        f.write("q = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95].\n")
        f.write("Wald test for coefficient equality across quantiles.\n\n")

        f.write("KEY RESULTS:\n")
        f.write("-" * 80 + "\n\n")

        for regime_name in ['Normal', 'Elevated', 'Crisis']:
            if regime_name not in all_results:
                continue

            r = all_results[regime_name]
            f.write(f"\n{regime_name.upper()} REGIME\n")
            f.write(f"  Clean observations: {r['n_clean']} (total in regime: {r['n_total']})\n\n")

            # HML → SMB
            f.write(f"  Direction: HML → SMB (lagged HML predicts SMB)\n")
            f.write(f"  {'-'*75}\n")
            f.write(f"  {'Quantile':>10} {'Coefficient':>15} {'Std Err':>15} {'t-stat':>12} {'p-value':>12}\n")
            f.write(f"  {'-'*75}\n")

            h2s_res = r['hml_to_smb']['results']
            for q in [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]:
                if q in h2s_res and not np.isnan(h2s_res[q]['coefficient']):
                    c = h2s_res[q]['coefficient']
                    se = h2s_res[q]['std_error']
                    t = h2s_res[q]['t_statistic']
                    p = h2s_res[q]['p_value']
                    f.write(f"  {q:>10.2f} {c:>15.8f} {se:>15.8f} {t:>12.4f} {p:>12.4f}\n")

            h2s_wald = r['hml_to_smb']['wald_test']
            f.write(f"\n  Wald Test (H0: Coefficients equal across quantiles):\n")
            f.write(f"    Test statistic: {h2s_wald['wald_stat']:.4f}\n")
            f.write(f"    Degrees of freedom: {h2s_wald['df']}\n")
            f.write(f"    p-value: {h2s_wald['p_value']:.4f}\n")
            f.write(f"    Coefficient range: [{h2s_wald['coeff_min']:.8f}, {h2s_wald['coeff_max']:.8f}]\n")
            f.write(f"    Tail/Median ratio (Q95/Q50): {h2s_wald['tail_vs_median_ratio']:.4f}\n")

            # SMB → HML
            f.write(f"\n  Direction: SMB → HML (lagged SMB predicts HML)\n")
            f.write(f"  {'-'*75}\n")
            f.write(f"  {'Quantile':>10} {'Coefficient':>15} {'Std Err':>15} {'t-stat':>12} {'p-value':>12}\n")
            f.write(f"  {'-'*75}\n")

            s2h_res = r['smb_to_hml']['results']
            for q in [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]:
                if q in s2h_res and not np.isnan(s2h_res[q]['coefficient']):
                    c = s2h_res[q]['coefficient']
                    se = s2h_res[q]['std_error']
                    t = s2h_res[q]['t_statistic']
                    p = s2h_res[q]['p_value']
                    f.write(f"  {q:>10.2f} {c:>15.8f} {se:>15.8f} {t:>12.4f} {p:>12.4f}\n")

            s2h_wald = r['smb_to_hml']['wald_test']
            f.write(f"\n  Wald Test (H0: Coefficients equal across quantiles):\n")
            f.write(f"    Test statistic: {s2h_wald['wald_stat']:.4f}\n")
            f.write(f"    Degrees of freedom: {s2h_wald['df']}\n")
            f.write(f"    p-value: {s2h_wald['p_value']:.4f}\n")
            f.write(f"    Coefficient range: [{s2h_wald['coeff_min']:.8f}, {s2h_wald['coeff_max']:.8f}]\n")
            f.write(f"    Tail/Median ratio (Q95/Q50): {s2h_wald['tail_vs_median_ratio']:.4f}\n")

            # Interpretation
            f.write(f"\n  INTERPRETATION:\n")
            if h2s_wald['p_value'] < 0.05 and s2h_wald['p_value'] > 0.05:
                f.write(f"    ✓ HML→SMB: Linear (p={h2s_wald['p_value']:.4f}, no tail heterogeneity)\n")
                f.write(f"    ✓ SMB→HML: Nonlinear (p={s2h_wald['p_value']:.4f}, tail heterogeneity present)\n")
                f.write(f"    → Supports tail dependence hypothesis for reverse channel\n")
            elif h2s_wald['p_value'] > 0.05 and s2h_wald['p_value'] < 0.05:
                f.write(f"    ✗ HML→SMB: Nonlinear (p={h2s_wald['p_value']:.4f})\n")
                f.write(f"    ✗ SMB→HML: Linear (p={s2h_wald['p_value']:.4f})\n")
                f.write(f"    → Contradicts hypothesis\n")
            else:
                f.write(f"    ? Both channels show similar patterns (p_HtoS={h2s_wald['p_value']:.4f}, p_StoH={s2h_wald['p_value']:.4f})\n")

            f.write("\n" + "="*80 + "\n")

    print(f"Text summary saved to {txt_path}")

    print("\n" + "="*80)
    print("QUANTILE GRANGER ANALYSIS COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
