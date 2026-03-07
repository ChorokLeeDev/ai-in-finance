#!/usr/bin/env python3
"""
permutation_semantic_drift.py
==============================
Two analyses that address the method critic's two remaining concerns:

1. PERMUTATION TEST — OOS Elevated HML→SMB Granger
   Permutes OOS regime labels (preserving regime counts) 1000× and records
   the Granger F-statistic on the permuted "Elevated" set. Reports:
     - actual F vs permutation null distribution
     - permutation p-value (fraction permuted F ≥ actual F)
   This bypasses all HAC/asymptotic assumptions: if permutation p < 0.05,
   the result is non-parametrically significant.

2. SEMANTIC DRIFT DIAGNOSTIC
   Compares factor return distributions of:
     (a) Training Normal (1990–2012) vs OOS Elevated (2013–2024)
         → should differ if OOS Elevated ≠ "relabeled Normal"
     (b) Training Elevated (1990–2012) vs OOS Elevated (2013–2024)
         → should be similar if frozen classifier is consistent
   Tests: KS test, Levene variance test, mean difference t-test.

Uses the canonical frozen OOS setup from multistart_hmm_pipeline.py:
  - Student-t HMM, K=3, seed=28, train 1990–2012
  - relabel_regimes_by_data_norm (data-norm-based, not centroid-norm)
  - extract_regime_clean_indices + run_granger_at_lag (boundary-clean, lag=1)

Output: results/permutation_semantic_drift.json + .log
"""

import sys, json, warnings, os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, levene, ttest_ind, kurtosis, skew
warnings.filterwarnings('ignore')

sys.path.insert(0, '/Users/i767700/Github/ai-in-finance/papers/causal_regimes/code')
from multistart_hmm_pipeline import (
    download_ff_data,
    StudentTHMM,
    relabel_regimes_by_data_norm,
    extract_regime_clean_indices,
    run_granger_at_lag,
)

RESULTS_DIR = Path('/Users/i767700/Github/ai-in-finance/papers/causal_regimes/results')
FACTOR_COLS = ['MKT', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']
PRIMARY_SEED = 28
FIXED_LAG    = 1
N_PERM       = 1000
ELEV_IDX     = 1   # Normal=0, Elevated=1, Crisis=2 after relabeling
TRAIN_END    = '2012-12-31'
TEST_START   = '2013-01-01'


# ── Granger F-statistic (raw, for permutation) ────────────────────────────
def granger_f(smb, hml, clean_idx, lag=1):
    """Return F-statistic only (fast, for permutation loop)."""
    usable = clean_idx[clean_idx >= lag]
    n = len(usable)
    if n < 2 * lag + 10:
        return np.nan
    y = smb[usable]
    yl = np.column_stack([smb[usable - i - 1] for i in range(lag)])
    xl = np.column_stack([hml[usable - i - 1] for i in range(lag)])
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


# ── Permutation test ──────────────────────────────────────────────────────
def run_permutation_test(test_regimes, smb_test, hml_test,
                         n_perm=N_PERM, lag=FIXED_LAG, seed=0):
    """
    Permute OOS regime labels (preserving counts) n_perm times.
    For each permutation, compute Granger F on 'Elevated' (idx=1) set.
    Returns (actual_F, perm_Fs, perm_p).
    """
    rng = np.random.default_rng(seed)
    n_test = len(test_regimes)

    # Actual result
    clean_actual = extract_regime_clean_indices(test_regimes, ELEV_IDX, lag)
    actual_F = granger_f(smb_test, hml_test, clean_actual, lag)
    print(f"  Actual F (OOS Elevated, lag={lag}): {actual_F:.4f}  n_clean={len(clean_actual)}")

    # Permutation loop
    perm_Fs = []
    for i in range(n_perm):
        perm_reg = rng.permutation(test_regimes)
        clean_p  = extract_regime_clean_indices(perm_reg, ELEV_IDX, lag)
        F_p      = granger_f(smb_test, hml_test, clean_p, lag)
        perm_Fs.append(F_p)
        if (i + 1) % 200 == 0:
            print(f"  Permutation {i+1}/{n_perm}...")

    perm_arr = np.array(perm_Fs)
    valid    = ~np.isnan(perm_arr)
    perm_p   = float(np.mean(perm_arr[valid] >= actual_F)) if valid.any() else np.nan

    print(f"\n  Permutation p-value: {perm_p:.4f}  "
          f"(actual F={actual_F:.3f} vs perm mean={np.nanmean(perm_arr):.3f}, "
          f"perm 95th pctile={np.nanpercentile(perm_arr,95):.3f})")
    return actual_F, perm_Fs, perm_p


# ── Semantic drift diagnostic ─────────────────────────────────────────────
def distribution_compare(a, b, label_a, label_b, factor):
    """KS, Levene, t-test between two arrays."""
    ks_s, ks_p   = ks_2samp(a, b)
    lev_s, lev_p = levene(a, b)
    t_s,   t_p   = ttest_ind(a, b, equal_var=False)
    print(f"    {factor}: mean {label_a}={np.mean(a):.4f}, {label_b}={np.mean(b):.4f} "
          f"| std {np.std(a):.4f} vs {np.std(b):.4f} "
          f"| KS p={ks_p:.4e} | Levene p={lev_p:.4e} | t-test p={t_p:.4e}")
    return {
        'mean_a': float(np.mean(a)), 'mean_b': float(np.mean(b)),
        'std_a':  float(np.std(a)),  'std_b':  float(np.std(b)),
        'kurt_a': float(kurtosis(a)), 'kurt_b': float(kurtosis(b)),
        'ks_stat': float(ks_s), 'ks_p': float(ks_p),
        'levene_p': float(lev_p),
        'ttest_p': float(t_p),
    }


def run_semantic_drift(train_df, train_regimes, test_df, test_regimes):
    """
    Compare factor return distributions across regimes.
    Key comparison: Training Normal vs OOS Elevated (should differ → no drift)
    Also:           Training Elevated vs OOS Elevated (should be similar → stable)
    """
    results = {}
    NORM_IDX  = 0
    ELEV_IDX_ = 1
    CRIS_IDX  = 2

    train_normal_mask = (train_regimes == NORM_IDX)
    train_elev_mask   = (train_regimes == ELEV_IDX_)
    test_elev_mask    = (test_regimes  == ELEV_IDX_)

    print(f"\n  Train Normal:    n={train_normal_mask.sum()}")
    print(f"  Train Elevated:  n={train_elev_mask.sum()}")
    print(f"  OOS   Elevated:  n={test_elev_mask.sum()}")

    factors = ['HML', 'SMB', 'MKT']
    for factor in factors:
        train_norm_vals = train_df.loc[train_normal_mask, factor].values
        train_elev_vals = train_df.loc[train_elev_mask,  factor].values
        test_elev_vals  = test_df.loc[test_elev_mask,    factor].values

        # Primary: Training Normal vs OOS Elevated
        print(f"\n  [Train Normal vs OOS Elevated] — {factor}")
        r1 = distribution_compare(
            train_norm_vals, test_elev_vals,
            'TrainNorm', 'OOS_Elev', factor
        )

        # Secondary: Training Elevated vs OOS Elevated
        print(f"  [Train Elevated vs OOS Elevated] — {factor}")
        r2 = distribution_compare(
            train_elev_vals, test_elev_vals,
            'TrainElev', 'OOS_Elev', factor
        )

        results[factor] = {
            'train_normal_vs_oos_elevated': r1,
            'train_elevated_vs_oos_elevated': r2,
        }

    return results


# ── Summary interpretation ────────────────────────────────────────────────
def interpret_drift(drift_results):
    """
    Semantic drift occurs if OOS Elevated ≈ Training Normal.
    We check: for HML (the key factor), is Training Normal vs OOS Elevated
    significantly different (p < 0.05 on at least 2 of 3 tests)?
    """
    hml = drift_results.get('HML', {})
    tn_vs_oe = hml.get('train_normal_vs_oos_elevated', {})
    te_vs_oe = hml.get('train_elevated_vs_oos_elevated', {})

    tn_different = sum([
        tn_vs_oe.get('ks_p', 1) < 0.05,
        tn_vs_oe.get('levene_p', 1) < 0.05,
        tn_vs_oe.get('ttest_p', 1) < 0.05,
    ])
    te_similar = sum([
        te_vs_oe.get('ks_p', 0) >= 0.05,
        te_vs_oe.get('levene_p', 0) >= 0.05,
        te_vs_oe.get('ttest_p', 0) >= 0.05,
    ])

    print(f"\n  HML distribution test summary:")
    print(f"    Train Normal vs OOS Elevated: {tn_different}/3 tests p<0.05  "
          f"(higher → less drift)")
    print(f"    Train Elevated vs OOS Elevated: {te_similar}/3 tests p≥0.05  "
          f"(higher → more stable classifier)")

    verdict = "LOW DRIFT" if tn_different >= 2 else "POSSIBLE DRIFT"
    print(f"    Semantic drift verdict: {verdict}")
    return verdict


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────
    df = download_ff_data()
    train_df = df[df.index <= TRAIN_END].copy()
    test_df  = df[df.index >= TEST_START].copy()
    print(f"Train: {len(train_df)} days ({train_df.index[0].date()}–{train_df.index[-1].date()})")
    print(f"Test:  {len(test_df)}  days ({test_df.index[0].date()}–{test_df.index[-1].date()})")

    # ── Fit frozen HMM (train only, seed=28) ─────────────────────────────
    print(f"\nFitting Student-t HMM (K=3, seed={PRIMARY_SEED}) on train 1990–2012...")
    hmm = StudentTHMM(n_regimes=3, n_iter=100, tol=1e-4, random_state=PRIMARY_SEED)
    hmm.fit(train_df[FACTOR_COLS].values)
    print(f"  Train LL: {hmm.log_likelihood_:.2f}")

    # Relabel by data-norm (canonical, not centroid-norm)
    # use_filtered=False for train (smoothed in-sample is fine; no future leakage within train)
    train_raw, _ = hmm.predict_oos(train_df[FACTOR_COLS].values, use_filtered=False)
    train_regimes, remap = relabel_regimes_by_data_norm(train_df, train_raw, FACTOR_COLS)
    train_counts = {k: int((train_regimes == v).sum()) for k, v in
                    {'Normal': 0, 'Elevated': 1, 'Crisis': 2}.items()}
    print(f"  Train regime counts: {train_counts}")

    # Apply same remap to test
    # use_filtered=True: no future info in OOS labels — matches frozen_oos_primary.py
    test_raw, _ = hmm.predict_oos(test_df[FACTOR_COLS].values, use_filtered=True)
    test_regimes = np.array([remap[r] for r in test_raw])
    test_counts = {k: int((test_regimes == v).sum()) for k, v in
                   {'Normal': 0, 'Elevated': 1, 'Crisis': 2}.items()}
    print(f"  Test  regime counts: {test_counts}")

    smb_test = test_df['SMB'].values
    hml_test = test_df['HML'].values

    # Verify OOS Elevated result matches paper (F-p=0.014, n=836)
    clean_elev = extract_regime_clean_indices(test_regimes, ELEV_IDX, FIXED_LAG)
    result_elev = run_granger_at_lag(smb_test, hml_test, clean_elev, FIXED_LAG)
    if result_elev:
        print(f"\n  OOS Elevated HML→SMB: n={result_elev['n_obs']}, "
              f"F-p={result_elev['f_p_value']:.4e}, "
              f"HAC-p={result_elev['hac_p_value']:.4e}  "
              f"(paper: n=836, F-p=0.014, HAC-p=0.041)")
    else:
        print("  WARNING: OOS Elevated result is None — check regime counts")

    # ── 1. Permutation test ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"1. PERMUTATION TEST — OOS Elevated HML→SMB ({N_PERM} permutations)")
    print(f"{'='*60}")
    actual_F, perm_Fs, perm_p = run_permutation_test(
        test_regimes, smb_test, hml_test,
        n_perm=N_PERM, lag=FIXED_LAG, seed=28
    )

    sig_str = ('***' if perm_p < 0.001 else '**' if perm_p < 0.01
               else '*' if perm_p < 0.05 else '†' if perm_p < 0.10 else 'n.s.')
    print(f"\n  RESULT: permutation p = {perm_p:.4f} {sig_str}")

    # ── 2. Semantic drift diagnostic ──────────────────────────────────────
    print(f"\n{'='*60}")
    print("2. SEMANTIC DRIFT DIAGNOSTIC")
    print(f"{'='*60}")
    drift_results = run_semantic_drift(train_df, train_regimes, test_df, test_regimes)
    drift_verdict = interpret_drift(drift_results)

    # ── Save results ──────────────────────────────────────────────────────
    perm_arr = np.array(perm_Fs)
    out = {
        'permutation_test': {
            'n_perm': N_PERM,
            'actual_F': float(actual_F),
            'perm_mean_F': float(np.nanmean(perm_arr)),
            'perm_median_F': float(np.nanmedian(perm_arr)),
            'perm_95pct_F': float(np.nanpercentile(perm_arr, 95)),
            'perm_p': perm_p,
            'n_clean_actual': int(len(clean_elev)),
        },
        'oos_granger_verification': {
            'n_obs': result_elev['n_obs'] if result_elev else None,
            'f_p':   result_elev['f_p_value'] if result_elev else None,
            'hac_p': result_elev['hac_p_value'] if result_elev else None,
        },
        'semantic_drift': drift_results,
        'semantic_drift_verdict': drift_verdict,
        'train_regime_counts': train_counts,
        'test_regime_counts': test_counts,
    }
    out_path = RESULTS_DIR / 'permutation_semantic_drift.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Permutation test (OOS Elevated HML→SMB):")
    print(f"  actual F = {actual_F:.3f}")
    print(f"  perm 95th pctile = {np.nanpercentile(perm_arr, 95):.3f}")
    print(f"  permutation p = {perm_p:.4f} {sig_str}")
    print(f"Semantic drift (HML): {drift_verdict}")
    print("Done.")


if __name__ == '__main__':
    main()
