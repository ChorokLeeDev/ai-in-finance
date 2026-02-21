"""
verify_k3_and_gaps.py
=====================
Two verifications required by reviewers:

1. K=3 BIC check on TRAINING DATA ONLY (1990-2012)
   - Fits Student-t HMM for K=2,3,4 with 10 seeds each on training period
   - Reports BIC; confirms K=3 still selected without look at OOS data

2. Regime gap statistics
   - Loads cached regime labels (selected_fit_regimes.csv)
   - For each regime, computes gap lengths between non-consecutive obs
   - Reports median/max gap and % observations dropped at boundaries (lag=1)
"""

import numpy as np
import pandas as pd
import json
import sys
import os
import urllib.request
import zipfile
import io
import warnings
warnings.filterwarnings('ignore')

from scipy.special import gammaln
from scipy.optimize import minimize_scalar
from scipy.cluster.vq import kmeans2

RESULTS_DIR = '/Users/i767700/Github/ai-in-finance/papers/causal_regimes/results'
CODE_DIR    = '/Users/i767700/Github/ai-in-finance/papers/causal_regimes/code'

# ---------------------------------------------------------------------------
# Reuse StudentTHMM from multistart pipeline
# ---------------------------------------------------------------------------
sys.path.insert(0, CODE_DIR)
from multistart_hmm_pipeline import StudentTHMM, download_ff_data

# ---------------------------------------------------------------------------
# BIC helper
# ---------------------------------------------------------------------------
def compute_bic(log_lik, n_params, n_obs):
    return -2 * log_lik + n_params * np.log(n_obs)

def n_params_student_t_hmm(K, d):
    """Number of free parameters in a K-regime Student-t HMM with d features."""
    # mu: K*d
    # Sigma (symmetric): K * d*(d+1)/2
    # nu: K
    # A (rows sum to 1): K*(K-1)
    # pi (sums to 1): K-1
    return K*d + K*d*(d+1)//2 + K + K*(K-1) + (K-1)

# ---------------------------------------------------------------------------
# Part 1: K=3 verification on training data 1990-2012
# ---------------------------------------------------------------------------
def verify_k3_on_training_data():
    print("=" * 60)
    print("Part 1: K selection on TRAINING DATA ONLY (1990-2012)")
    print("=" * 60)

    df = download_ff_data()
    train = df.loc[:'2012-12-31']
    X_train = train[['MKT','SMB','HML','RMW','CMA','MOM']].values
    T, d = X_train.shape
    print(f"Training observations: {T} days ({train.index[0].date()} to {train.index[-1].date()})")

    seeds = list(range(10))  # 10 seeds per K for speed
    results = {}

    for K in [2, 3, 4]:
        lls = []
        for seed in seeds:
            try:
                hmm = StudentTHMM(n_regimes=K, n_iter=200, tol=1e-5, random_state=seed)
                hmm.fit(X_train)
                if hmm.log_likelihood_ is not None and np.isfinite(hmm.log_likelihood_):
                    lls.append(hmm.log_likelihood_)
            except Exception as e:
                pass

        if not lls:
            print(f"  K={K}: all seeds failed")
            continue

        best_ll = max(lls)
        p = n_params_student_t_hmm(K, d)
        bic = compute_bic(best_ll, p, T)
        aic = -2 * best_ll + 2 * p

        results[K] = {'best_ll': best_ll, 'n_params': p, 'bic': bic, 'aic': aic,
                      'n_seeds_converged': len(lls)}
        print(f"  K={K}: best LL={best_ll:.2f}  n_params={p}  BIC={bic:.2f}  AIC={aic:.2f}  "
              f"(converged: {len(lls)}/{len(seeds)})")

    if results:
        best_k_bic = min(results, key=lambda k: results[k]['bic'])
        best_k_aic = min(results, key=lambda k: results[k]['aic'])
        print(f"\n  => BIC selects K={best_k_bic}  |  AIC selects K={best_k_aic}")
        if best_k_bic == 3:
            print("  => VERIFIED: K=3 is BIC-optimal on training data alone. No look-ahead bias.")
        else:
            print(f"  => WARNING: BIC selects K={best_k_bic} on training data. "
                  f"Paper assumes K=3 — OOS model selection may be biased.")

    # Save
    out = {'training_period': '1990-2012',
           'n_train': int(T),
           'n_features': int(d),
           'seeds_per_k': seeds,
           'results': {str(k): v for k, v in results.items()},
           'bic_selected_k': int(best_k_bic) if results else None}
    with open(os.path.join(RESULTS_DIR, 'k3_verification.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved to results/k3_verification.json")
    return results

# ---------------------------------------------------------------------------
# Part 2: Regime gap statistics from cached labels
# ---------------------------------------------------------------------------
def compute_gap_statistics():
    print("\n" + "=" * 60)
    print("Part 2: Regime gap and boundary exclusion statistics")
    print("=" * 60)

    regimes_path = os.path.join(RESULTS_DIR, 'selected_fit_regimes.csv')
    df = pd.read_csv(regimes_path, index_col=0, parse_dates=True)
    print(f"Loaded {len(df)} rows from selected_fit_regimes.csv")
    print(f"Columns: {list(df.columns)}")

    # Find regime label column
    regime_col = None
    for col in ['regime', 'regime_label', 'smoothed_regime', 'filtered_regime', 'label']:
        if col in df.columns:
            regime_col = col
            break
    if regime_col is None:
        # Try to use the first integer-valued column
        for col in df.columns:
            if df[col].dtype in [np.int64, np.int32, np.float64]:
                regime_col = col
                break
    print(f"Using regime column: '{regime_col}'")

    regime_names = {0: 'Normal', 1: 'Elevated', 2: 'Crisis'}
    stats_out = {}

    for k, name in regime_names.items():
        mask = df[regime_col] == k
        idx = df.index[mask]
        n_obs = len(idx)

        if n_obs < 2:
            print(f"  {name}: only {n_obs} observations, skipping")
            continue

        # Gap lengths in calendar days between consecutive regime obs
        gaps = (idx[1:] - idx[:-1]).days
        # A "gap" > 1 means non-consecutive (another regime intervened)
        non_consec = gaps[gaps > 1]
        n_gaps = len(non_consec)
        med_gap = float(np.median(non_consec)) if len(non_consec) > 0 else 0.0
        max_gap = float(np.max(non_consec)) if len(non_consec) > 0 else 0.0

        # Boundary exclusion for lag=1: drop obs where previous obs is NOT same regime
        # i.e., first obs of each contiguous block
        is_first_of_block = np.concatenate([[True], gaps > 1])
        n_boundary_excluded = int(is_first_of_block.sum())
        pct_excluded = 100 * n_boundary_excluded / n_obs

        print(f"\n  {name} (k={k}): n={n_obs}")
        print(f"    Non-consecutive gaps: {n_gaps}")
        print(f"    Median gap length: {med_gap:.0f} days")
        print(f"    Max gap length: {max_gap:.0f} days")
        print(f"    Boundary exclusions (lag=1): {n_boundary_excluded} ({pct_excluded:.1f}%)")

        stats_out[name] = {
            'n_obs': n_obs,
            'n_non_consecutive_gaps': n_gaps,
            'median_gap_days': med_gap,
            'max_gap_days': max_gap,
            'boundary_exclusions_lag1': n_boundary_excluded,
            'pct_boundary_excluded': pct_excluded
        }

    # Total boundary exclusions
    total_obs = sum(s['n_obs'] for s in stats_out.values())
    total_excl = sum(s['boundary_exclusions_lag1'] for s in stats_out.values())
    print(f"\n  TOTAL: {total_excl}/{total_obs} = {100*total_excl/total_obs:.1f}% excluded at boundaries (lag=1)")
    stats_out['TOTAL'] = {
        'total_obs': total_obs,
        'total_boundary_excluded': total_excl,
        'pct_total_excluded': 100*total_excl/total_obs
    }

    with open(os.path.join(RESULTS_DIR, 'gap_statistics.json'), 'w') as f:
        json.dump(stats_out, f, indent=2)
    print(f"  Saved to results/gap_statistics.json")
    return stats_out


if __name__ == '__main__':
    k_results = verify_k3_on_training_data()
    gap_stats = compute_gap_statistics()
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
