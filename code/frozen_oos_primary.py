"""
Frozen OOS Primary Analysis — fully clean, no circularity
===========================================================
Fixes:
  1. Relabeling uses TRAIN-period ordering only (not test data)
  2. Lag=1 fixed from in-sample BIC optimum (no OOS look-ahead)
  3. Multi-seed (top-5 LL from in-sample) to assess seed sensitivity

HMM trained on 1990-2012, Granger tested on 2013-2024.
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
# Top-5 seeds by LL from in-sample multistart (primary + 4 alternatives)
TOP_SEEDS = [28, 15, 7, 42, 3]   # will be overridden from multistart_hmm_results.json
FIXED_LAG = 1   # in-sample BIC-optimal for all regimes


def load_top_seeds(n=5):
    """Load top-n seeds by log-likelihood from multistart_hmm_results.json."""
    path = f"{RESULTS_DIR}/multistart_hmm_results.json"
    with open(path) as f:
        d = json.load(f)
    summaries = d.get('fit_summaries', [])
    if not summaries:
        return [PRIMARY_SEED]
    # Sort by log_likelihood descending
    sorted_fits = sorted(summaries, key=lambda x: x.get('log_likelihood', -1e9), reverse=True)
    seeds = [s['seed'] for s in sorted_fits[:n] if 'seed' in s]
    if PRIMARY_SEED not in seeds:
        seeds = [PRIMARY_SEED] + seeds[:n-1]
    print(f"  Top-{n} seeds by LL: {seeds}")
    return seeds


def apply_train_remap(test_raw, remap):
    """Apply train-period relabeling order to test raw regime labels."""
    return np.array([remap[r] for r in test_raw])


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


def run_permutation_test(test_regimes, smb, hml, n_perm=1000, lag=FIXED_LAG, seed=28):
    """
    Permute OOS regime labels (preserving counts) n_perm times.
    Computes Granger F on the permuted 'Elevated' (idx=1) set each time.
    Uses the SAME regime assignments as the primary result (no re-fitting).
    Returns (actual_F, perm_p, perm_95pct).
    """
    rng = np.random.default_rng(seed)
    clean_actual = extract_regime_clean_indices(test_regimes, 1, max_lag=lag)
    actual_F = granger_f(smb, hml, clean_actual, lag)
    print(f"  Permutation test: actual F={actual_F:.4f}, n_clean={len(clean_actual)}")

    perm_Fs = []
    for i in range(n_perm):
        perm_reg = rng.permutation(test_regimes)
        clean_p = extract_regime_clean_indices(perm_reg, 1, max_lag=lag)
        F_p = granger_f(smb, hml, clean_p, lag)
        perm_Fs.append(F_p)
        if (i + 1) % 200 == 0:
            print(f"  Permutation {i+1}/{n_perm}...")

    perm_arr = np.array(perm_Fs)
    valid = ~np.isnan(perm_arr)
    perm_p = float(np.mean(perm_arr[valid] >= actual_F)) if valid.any() else np.nan
    perm_95 = float(np.nanpercentile(perm_arr, 95))
    print(f"  Permutation p={perm_p:.4f}  (actual F={actual_F:.3f} vs 95th pctile={perm_95:.3f})")
    return actual_F, perm_p, perm_95


def run_frozen_oos_for_seed(seed, train_df, test_df, factor_cols, lag=FIXED_LAG,
                             run_perm=False, n_perm=1000):
    """Run frozen OOS for a single seed. Returns per-regime Granger results."""
    hmm = StudentTHMM(n_regimes=3, n_iter=100, tol=1e-4, random_state=seed)
    hmm.fit(train_df[factor_cols].values)

    # Relabeling order determined by TRAIN data only
    train_raw = hmm.predict(train_df[factor_cols].values, use_filtered=False)
    _, remap = relabel_regimes_by_data_norm(train_df, train_raw, factor_cols)
    train_counts = {REGIME_NAMES[k]: int((np.array([remap[r] for r in train_raw])==k).sum())
                    for k in range(3)}

    # Apply same remap to test (no test-data relabeling)
    test_raw, _ = hmm.predict_oos(test_df[factor_cols].values, use_filtered=True)  # filtered=True: no future info
    test_regimes = apply_train_remap(test_raw, remap)
    test_counts = {REGIME_NAMES[k]: int((test_regimes==k).sum()) for k in range(3)}

    hml = test_df['HML'].values
    smb = test_df['SMB'].values

    granger = {}
    for k, name in enumerate(REGIME_NAMES):
        clean = extract_regime_clean_indices(test_regimes, k, max_lag=lag)
        h2s = run_granger_at_lag(smb, hml, clean, lag)
        s2h = run_granger_at_lag(hml, smb, clean, lag)
        granger[name] = {
            'n_clean': len(clean),
            'hml_to_smb': h2s,
            'smb_to_hml': s2h,
        }

    result = {
        'seed': seed,
        'train_ll': float(hmm.log_likelihood_),
        'train_counts': train_counts,
        'test_counts': test_counts,
        'granger': granger,
    }

    if run_perm:
        print(f"\n  Running permutation test (n_perm={n_perm}) on exact regime assignments...")
        actual_F, perm_p, perm_95 = run_permutation_test(
            test_regimes, smb, hml, n_perm=n_perm, lag=lag)
        result['permutation_test'] = {
            'n_perm': n_perm,
            'actual_F': actual_F,
            'perm_p': perm_p,
            'perm_95pct_F': perm_95,
            'n_clean_elevated': int(len(extract_regime_clean_indices(test_regimes, 1, lag))),
        }

    return result


def main():
    print("Loading Fama-French data...")
    df = download_ff_data()
    df = df / 100.0
    print(f"  Full: {df.index[0].date()} to {df.index[-1].date()}, n={len(df)}")

    train_df = df.loc[:'2012-12-31']
    test_df  = df.loc['2013-01-01':]
    factor_cols = ['MKT','SMB','HML','RMW','CMA','MOM']
    print(f"  Train: {len(train_df)} | Test: {len(test_df)}")

    # Load top seeds
    top_seeds = load_top_seeds(n=5)

    print(f"\nRunning frozen OOS for {len(top_seeds)} seeds (lag={FIXED_LAG} fixed from in-sample BIC):")
    all_results = []
    for seed in top_seeds:
        print(f"\n  Seed {seed}:")
        is_primary = (seed == PRIMARY_SEED)
        res = run_frozen_oos_for_seed(seed, train_df, test_df, factor_cols, lag=FIXED_LAG,
                                      run_perm=is_primary, n_perm=1000)
        for name in REGIME_NAMES:
            g = res['granger'][name]
            h2s = g.get('hml_to_smb')
            if h2s:
                print(f"    {name}: HML->SMB n={g['n_clean']} "
                      f"F-p={h2s['f_p_value']:.4f} HAC-p={h2s['hac_p_value']:.4f} "
                      f"ΔR²={h2s['delta_r2']:.4f}")
        all_results.append(res)

    # Summary across seeds
    print("\n--- Summary (HML->SMB F-p by regime across seeds) ---")
    for name in REGIME_NAMES:
        ps = []
        for res in all_results:
            h2s = res['granger'][name].get('hml_to_smb')
            if h2s:
                ps.append(h2s['f_p_value'])
        if ps:
            print(f"  {name}: min={min(ps):.4f}  median={sorted(ps)[len(ps)//2]:.4f}  max={max(ps):.4f}")

    # Primary seed result
    primary = next(r for r in all_results if r['seed'] == PRIMARY_SEED)

    output = {
        'description': (
            'Frozen OOS primary: HMM trained 1990-2012, tested 2013-2024. '
            f'Lag={FIXED_LAG} fixed from in-sample BIC. '
            'Relabeling uses train-period ordering. '
            'filtered=True (no future info in OOS). '
            'No circularity.'
        ),
        'fixed_lag': FIXED_LAG,
        'primary_seed': PRIMARY_SEED,
        'seeds_tested': top_seeds,
        'train_period': '1990-01-02 to 2012-12-31',
        'test_period': '2013-01-02 to 2024-12-31',
        'train_n': len(train_df),
        'test_n': len(test_df),
        'primary': primary,
        'all_seeds': all_results,
    }
    outpath = f"{RESULTS_DIR}/frozen_oos_primary.json"
    with open(outpath, 'w') as fout:
        json.dump(output, fout, indent=2)
    print(f"\nSaved → {outpath}")


if __name__ == '__main__':
    main()
