"""
HAC Bandwidth Sensitivity Test for OOS Elevated HML->SMB
==========================================================
Tests whether the OOS Elevated Granger result survives wider HAC bandwidths.
Specifically: bandwidth=1 (paper), bandwidth=7 (rule-of-thumb for n=836),
and Andrews AR(1) plug-in (already confirmed = 1 in paper).
"""
import sys, json, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
warnings.filterwarnings('ignore')

sys.path.insert(0, '/Users/i767700/Github/ai-in-finance/papers/causal_regimes/code')
from multistart_hmm_pipeline import (
    download_ff_data,
    StudentTHMM,
    relabel_regimes_by_data_norm,
    extract_regime_clean_indices,
)

RESULTS_DIR = '/Users/i767700/Github/ai-in-finance/papers/causal_regimes/results'
TRAIN_END = '2012-12-31'
TEST_START = '2013-01-01'
PRIMARY_SEED = 28
FIXED_LAG = 1


def granger_hac_bandwidth(y_curr, y_lagged, x_lagged, lag, bandwidth):
    """HAC Wald test with specified bandwidth."""
    n = len(y_curr)
    p = y_lagged.shape[1]
    X_u = np.column_stack([np.ones(n), y_lagged, x_lagged])
    model = sm.OLS(y_curr, X_u)
    result = model.fit(cov_type='HAC', cov_kwds={'maxlags': bandwidth})
    n_params = X_u.shape[1]
    R = np.zeros((p, n_params))
    for i in range(p):
        R[i, 1 + p + i] = 1.0
    r = np.zeros(p)
    Rb = R @ result.params - r
    cov_sub = R @ result.cov_params() @ R.T
    wald_stat = float(Rb @ np.linalg.solve(cov_sub, Rb))
    hac_p = float(1 - stats.chi2.cdf(wald_stat, df=p))
    return wald_stat, hac_p


def run_granger_with_bandwidths(y_all, x_all, clean_indices, lag, bandwidths):
    """Run Granger test with multiple HAC bandwidths."""
    usable = np.array([idx for idx in clean_indices if idx >= lag])
    n = len(usable)
    print(f"  n_obs={n}")
    y_curr = y_all[usable]
    y_lagged = np.column_stack([y_all[usable - i - 1] for i in range(lag)])
    x_lagged = np.column_stack([x_all[usable - i - 1] for i in range(lag)])

    # F-test baseline
    Xr = np.column_stack([np.ones(n), y_lagged])
    Xu = np.column_stack([np.ones(n), y_lagged, x_lagged])
    br = np.linalg.lstsq(Xr, y_curr, rcond=None)[0]
    bu = np.linalg.lstsq(Xu, y_curr, rcond=None)[0]
    rr = float(np.sum((y_curr - Xr @ br) ** 2))
    ru = float(np.sum((y_curr - Xu @ bu) ** 2))
    df1, df2 = lag, n - 2 * lag - 1
    f_stat = ((rr - ru) / df1) / (ru / df2)
    f_p = float(1 - stats.f.cdf(f_stat, df1, df2))
    print(f"  F-test: F={f_stat:.4f}, p={f_p:.4f}")

    results = {}
    for bw in bandwidths:
        ws, hp = granger_hac_bandwidth(y_curr, y_lagged, x_lagged, lag, bw)
        results[bw] = {'wald': ws, 'hac_p': hp}
        print(f"  HAC bandwidth={bw}: Wald={ws:.4f}, p={hp:.4f}")
    return results


def main():
    print("Loading Fama-French data...")
    df = download_ff_data()

    train_df = df[df.index <= TRAIN_END].copy()
    test_df = df[df.index >= TEST_START].copy()
    print(f"Train: {len(train_df)} days, Test: {len(test_df)} days")

    # Load primary seed HMM fit results
    results_path = f"{RESULTS_DIR}/frozen_oos_primary_results.json"
    try:
        with open(results_path) as f:
            saved = json.load(f)
        print(f"Loaded saved OOS results from {results_path}")
        # Extract OOS regime labels for primary seed
        seed_results = saved.get('seed_results', {}).get(str(PRIMARY_SEED), {})
        oos_regimes = seed_results.get('test_regimes_labeled')
        if oos_regimes is None:
            raise ValueError("No OOS regime labels found")
        oos_regimes = np.array(oos_regimes)
    except Exception as e:
        print(f"Could not load saved results ({e}), re-running HMM...")
        # Re-train HMM on train data
        factor_cols = ['MKT','SMB','HML','RMW','CMA','MOM']
        X_train = train_df[factor_cols].values
        hmm = StudentTHMM(n_regimes=3, n_iter=200, random_state=PRIMARY_SEED)
        hmm.fit(X_train)
        train_regimes_raw = hmm.predict(X_train, use_filtered=False)
        _, remap = relabel_regimes_by_data_norm(train_df, train_regimes_raw, factor_cols)
        # Apply to test
        X_test = test_df[factor_cols].values
        test_regimes_raw, _ = hmm.predict_oos(X_test, use_filtered=True)
        oos_regimes = np.array([remap[r] for r in test_regimes_raw])

    # Extract HML and SMB test series
    smb_test = test_df['SMB'].values
    hml_test = test_df['HML'].values

    # Regime 1 = Elevated (0=Normal, 1=Elevated, 2=Crisis)
    ELEVATED_IDX = 1
    elevated_clean = extract_regime_clean_indices(oos_regimes, ELEVATED_IDX, FIXED_LAG)
    print(f"\nOOS Elevated: {np.sum(oos_regimes == ELEVATED_IDX)} days total")
    print(f"OOS Elevated clean (lag-1): {len(elevated_clean)} days")

    # Bandwidths to test: 1 (paper), 4, 7 (rule-of-thumb), 10, 15
    # Rule-of-thumb for n=836: floor(0.75 * n^(1/3)) = floor(0.75 * 9.41) = floor(7.06) = 7
    bws = [1, 4, 7, 10, 15]
    print(f"\n=== HAC Bandwidth Sensitivity for OOS Elevated HML->SMB ===")
    print(f"Rule-of-thumb bandwidth for n~836: floor(0.75 * n^(1/3)) = {int(0.75 * (836 ** (1/3)))}")
    results = run_granger_with_bandwidths(smb_test, hml_test, elevated_clean, FIXED_LAG, bws)

    print("\n=== SUMMARY ===")
    print(f"{'Bandwidth':<12} {'HAC p-value':<15} {'Significant (p<0.05)'}")
    print("-" * 45)
    for bw, r in results.items():
        sig = "YES" if r['hac_p'] < 0.05 else "NO"
        print(f"{bw:<12} {r['hac_p']:.4f}{'':9} {sig}")

    # Save results
    out = {'bandwidths': {str(bw): r for bw, r in results.items()}}
    with open(f"{RESULTS_DIR}/hac_bandwidth_sensitivity.json", 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {RESULTS_DIR}/hac_bandwidth_sensitivity.json")


if __name__ == '__main__':
    main()
