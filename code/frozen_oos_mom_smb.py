"""
Frozen OOS Secondary Analysis: MOM→SMB
========================================
Uses the same HMM fit (seed 28, trained on 1990-2012) as the primary analysis.
Tests Granger causality for MOM→SMB in the Elevated regime.
MOM→SMB ranked #1 in OOS Elevated by F-statistic (F=20.3).

This provides secondary validation that regime-dependent predictive structure
is not specific to HML→SMB.
"""
import sys, json, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

sys.path.insert(0, '/Users/i767700/Github/ai-in-finance/papers/causal_regimes/code')
from multistart_hmm_pipeline import (
    download_ff_data,
    StudentTHMM,
    relabel_regimes_by_data_norm,
    extract_regime_clean_indices,
    run_granger_at_lag,
)

RESULTS_DIR = '/Users/i767700/Github/ai-in-finance/papers/causal_regimes/results'
REGIME_NAMES = ['Normal', 'Elevated', 'Crisis']
PRIMARY_SEED = 28
FIXED_LAG = 1   # in-sample BIC-optimal


def apply_train_remap(test_raw, remap):
    """Apply train-period relabeling order to test raw regime labels."""
    return np.array([remap[r] for r in test_raw])


def granger_f(target, predictor, clean_idx, lag=1):
    """Return F-statistic only (fast, for permutation loop)."""
    usable = clean_idx[clean_idx >= lag]
    n = len(usable)
    if n < 2 * lag + 10:
        return np.nan
    y = target[usable]
    yl = np.column_stack([target[usable - i - 1] for i in range(lag)])
    xl = np.column_stack([predictor[usable - i - 1] for i in range(lag)])
    Xr = np.column_stack([np.ones(n), yl])
    Xu = np.column_stack([np.ones(n), yl, xl])
    br = np.linalg.lstsq(Xr, y, rcond=None)[0]
    bu = np.linalg.lstsq(Xu, y, rcond=None)[0]
    rr = float(np.sum((y - Xr @ br) ** 2))
    ru = float(np.sum((y - Xu @ bu) ** 2))
    df1, df2 = lag, n - 2 * lag - 1
    if df2 <= 0 or ru <= 0:
        return np.nan
    return ((rr - ru) / df1) / (ru / df2)


def run_permutation_test(test_regimes, target, predictor, n_perm=10000, lag=FIXED_LAG, seed=42):
    """
    Permute OOS regime labels (preserving counts) n_perm times.
    Computes Granger F on the permuted 'Elevated' (idx=1) set each time.
    Returns (actual_F, perm_p, perm_95pct, perm_se).
    """
    rng = np.random.default_rng(seed)
    clean_actual = extract_regime_clean_indices(test_regimes, 1, max_lag=lag)
    actual_F = granger_f(target, predictor, clean_actual, lag)
    print(f"  Permutation test: actual F={actual_F:.4f}, n_clean={len(clean_actual)}")

    perm_Fs = []
    for i in range(n_perm):
        perm_reg = rng.permutation(test_regimes)
        clean_p = extract_regime_clean_indices(perm_reg, 1, max_lag=lag)
        F_p = granger_f(target, predictor, clean_p, lag)
        perm_Fs.append(F_p)
        if (i + 1) % 2000 == 0:
            print(f"  Permutation {i+1}/{n_perm}...")

    perm_arr = np.array(perm_Fs)
    valid = ~np.isnan(perm_arr)
    n_valid = valid.sum()
    exceeds = (perm_arr[valid] >= actual_F).sum()
    perm_p = float(exceeds / n_valid) if n_valid > 0 else np.nan
    # Standard error for proportion
    perm_se = float(np.sqrt(perm_p * (1 - perm_p) / n_valid)) if n_valid > 0 else np.nan
    perm_95 = float(np.nanpercentile(perm_arr, 95))
    print(f"  Permutation p={perm_p:.4f} (SE={perm_se:.4f}), actual F={actual_F:.3f} vs 95th pctile={perm_95:.3f}")
    return actual_F, perm_p, perm_95, perm_se


def main():
    print("="*70)
    print("Frozen OOS Secondary Analysis: MOM -> SMB")
    print("="*70)

    print("\nLoading Fama-French data...")
    df = download_ff_data()
    df = df / 100.0  # Convert percentage to decimal
    print(f"  Full: {df.index[0].date()} to {df.index[-1].date()}, n={len(df)}")

    train_df = df.loc[:'2012-12-31']
    test_df  = df.loc['2013-01-01':]
    factor_cols = ['MKT','SMB','HML','RMW','CMA','MOM']
    print(f"  Train: {len(train_df)} | Test: {len(test_df)}")

    # Fit HMM on train data only (same seed as primary)
    print(f"\nFitting Student-t HMM (seed={PRIMARY_SEED}) on 1990-2012...")
    hmm = StudentTHMM(n_regimes=3, n_iter=100, tol=1e-4, random_state=PRIMARY_SEED)
    hmm.fit(train_df[factor_cols].values)
    print(f"  Train LL: {hmm.log_likelihood_:.2f}")

    # Relabeling order determined by TRAIN data only
    train_raw = hmm.predict(train_df[factor_cols].values, use_filtered=False)
    _, remap = relabel_regimes_by_data_norm(train_df, train_raw, factor_cols)
    train_counts = {REGIME_NAMES[k]: int((np.array([remap[r] for r in train_raw])==k).sum())
                    for k in range(3)}
    print(f"  Train regime counts: {train_counts}")

    # Apply same remap to test (no test-data relabeling)
    test_raw, _ = hmm.predict_oos(test_df[factor_cols].values, use_filtered=True)
    test_regimes = apply_train_remap(test_raw, remap)
    test_counts = {REGIME_NAMES[k]: int((test_regimes==k).sum()) for k in range(3)}
    print(f"  Test regime counts: {test_counts}")

    mom = test_df['MOM'].values
    smb = test_df['SMB'].values

    # Run Granger tests for all regimes
    print(f"\n--- MOM -> SMB Granger causality (lag={FIXED_LAG}) ---")
    granger = {}
    for k, name in enumerate(REGIME_NAMES):
        clean = extract_regime_clean_indices(test_regimes, k, max_lag=FIXED_LAG)
        result = run_granger_at_lag(smb, mom, clean, FIXED_LAG)
        granger[name] = {
            'n_clean': len(clean),
            'result': result,
        }
        if result:
            print(f"  {name}: n={len(clean)} "
                  f"F={result['f_stat']:.3f} F-p={result['f_p_value']:.4f} "
                  f"HAC-p={result['hac_p_value']:.4f} dR2={result['delta_r2']:.4f}")
        else:
            print(f"  {name}: n={len(clean)} (insufficient data)")

    # Also compute reverse direction (SMB -> MOM)
    print(f"\n--- SMB -> MOM Granger causality (lag={FIXED_LAG}) ---")
    granger_reverse = {}
    for k, name in enumerate(REGIME_NAMES):
        clean = extract_regime_clean_indices(test_regimes, k, max_lag=FIXED_LAG)
        result = run_granger_at_lag(mom, smb, clean, FIXED_LAG)
        granger_reverse[name] = {
            'n_clean': len(clean),
            'result': result,
        }
        if result:
            print(f"  {name}: n={len(clean)} "
                  f"F={result['f_stat']:.3f} F-p={result['f_p_value']:.4f} "
                  f"HAC-p={result['hac_p_value']:.4f} dR2={result['delta_r2']:.4f}")
        else:
            print(f"  {name}: n={len(clean)} (insufficient data)")

    # Permutation test for Elevated regime (MOM -> SMB)
    print(f"\n--- Permutation test (Elevated, MOM -> SMB, n_perm=10000) ---")
    actual_F, perm_p, perm_95, perm_se = run_permutation_test(
        test_regimes, smb, mom, n_perm=10000, lag=FIXED_LAG, seed=42)

    # Compile results
    elevated_result = granger['Elevated']['result']
    output = {
        'description': (
            'Frozen OOS secondary analysis: MOM->SMB. '
            'HMM trained 1990-2012 (seed 28), tested 2013-2024. '
            f'Lag={FIXED_LAG} fixed from in-sample BIC. '
            'MOM->SMB ranked #1 in OOS Elevated by F-statistic. '
            'This validates that regime-dependent predictive structure is not specific to HML->SMB.'
        ),
        'pair': 'MOM->SMB',
        'fixed_lag': FIXED_LAG,
        'seed': PRIMARY_SEED,
        'train_period': '1990-01-02 to 2012-12-31',
        'test_period': '2013-01-02 to 2024-12-31',
        'train_n': len(train_df),
        'test_n': len(test_df),
        'train_ll': float(hmm.log_likelihood_),
        'train_counts': train_counts,
        'test_counts': test_counts,
        'granger_mom_to_smb': {
            name: {
                'n_clean': granger[name]['n_clean'],
                'f_stat': granger[name]['result']['f_stat'] if granger[name]['result'] else None,
                'f_p_value': granger[name]['result']['f_p_value'] if granger[name]['result'] else None,
                'hac_p_value': granger[name]['result']['hac_p_value'] if granger[name]['result'] else None,
                'delta_r2': granger[name]['result']['delta_r2'] if granger[name]['result'] else None,
            }
            for name in REGIME_NAMES
        },
        'granger_smb_to_mom': {
            name: {
                'n_clean': granger_reverse[name]['n_clean'],
                'f_stat': granger_reverse[name]['result']['f_stat'] if granger_reverse[name]['result'] else None,
                'f_p_value': granger_reverse[name]['result']['f_p_value'] if granger_reverse[name]['result'] else None,
                'hac_p_value': granger_reverse[name]['result']['hac_p_value'] if granger_reverse[name]['result'] else None,
                'delta_r2': granger_reverse[name]['result']['delta_r2'] if granger_reverse[name]['result'] else None,
            }
            for name in REGIME_NAMES
        },
        'permutation_test_elevated': {
            'n_perm': 10000,
            'actual_F': actual_F,
            'perm_p': perm_p,
            'perm_se': perm_se,
            'perm_95pct_F': perm_95,
            'n_clean_elevated': int(granger['Elevated']['n_clean']),
        },
    }

    outpath = f"{RESULTS_DIR}/mom_smb_oos.json"
    with open(outpath, 'w') as fout:
        json.dump(output, fout, indent=2)
    print(f"\n{'='*70}")
    print(f"Saved -> {outpath}")
    print(f"{'='*70}")

    # Summary
    print("\n=== SUMMARY ===")
    print(f"MOM->SMB in OOS Elevated regime:")
    if elevated_result:
        print(f"  F-statistic: {elevated_result['f_stat']:.3f}")
        print(f"  F-p-value:   {elevated_result['f_p_value']:.4f}")
        print(f"  HAC-p-value: {elevated_result['hac_p_value']:.4f}")
        print(f"  Delta-R2:    {elevated_result['delta_r2']:.4f}")
    print(f"  Permutation p: {perm_p:.4f} (SE={perm_se:.4f})")
    print(f"  (10,000 shuffles)")


if __name__ == '__main__':
    main()
