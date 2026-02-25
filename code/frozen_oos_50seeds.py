"""
Frozen OOS 50-seed analysis.
Trains HMM on 1990-2012 for all 50 seeds, tests Granger on 2013-2024.
Reports p-value distribution across all seeds.
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
FIXED_LAG = 1

def apply_train_remap(test_raw, remap):
    return np.array([remap[r] for r in test_raw])

def run_frozen_oos_for_seed(seed, train_df, test_df, factor_cols, lag=FIXED_LAG):
    hmm = StudentTHMM(n_regimes=3, n_iter=100, tol=1e-4, random_state=seed)
    hmm.fit(train_df[factor_cols].values)
    train_ll = float(hmm.log_likelihood_)

    train_raw = hmm.predict(train_df[factor_cols].values, use_filtered=False)
    _, remap = relabel_regimes_by_data_norm(train_df, train_raw, factor_cols)

    test_raw, _ = hmm.predict_oos(test_df[factor_cols].values, use_filtered=True)
    test_regimes = apply_train_remap(test_raw, remap)

    hml = test_df['HML'].values
    smb = test_df['SMB'].values

    results = {}
    for k, name in enumerate(REGIME_NAMES):
        clean = extract_regime_clean_indices(test_regimes, k, max_lag=lag)
        h2s = run_granger_at_lag(smb, hml, clean, lag)
        results[name] = {
            'n_clean': len(clean),
            'hml_to_smb': h2s,
        }
    return {'seed': seed, 'train_ll': train_ll, 'granger': results}


def main():
    print("Loading Fama-French data...")
    df = download_ff_data() / 100.0
    train_df = df.loc[:'2012-12-31']
    test_df  = df.loc['2013-01-01':]
    factor_cols = ['MKT','SMB','HML','RMW','CMA','MOM']
    print(f"  Train: {len(train_df)} | Test: {len(test_df)}")

    # Load all 50 seeds
    with open(f'{RESULTS_DIR}/multistart_hmm_results.json') as f:
        d = json.load(f)
    all_seed_summaries = sorted(d['fit_summaries'],
                                key=lambda x: x.get('log_likelihood', -1e9), reverse=True)
    all_seeds = [s['seed'] for s in all_seed_summaries]
    print(f"Running frozen OOS for {len(all_seeds)} seeds...")

    all_results = []
    for i, seed in enumerate(all_seeds):
        try:
            res = run_frozen_oos_for_seed(seed, train_df, test_df, factor_cols)
            elev = res['granger']['Elevated'].get('hml_to_smb', {})
            f_p = elev.get('f_p_value', None)
            hac_p = elev.get('hac_p_value', None)
            print(f"  [{i+1:2d}/50] Seed {seed:3d}: train_ll={res['train_ll']:.1f}  "
                  f"Elevated F-p={f_p:.4f}  HAC-p={hac_p:.4f}" if f_p is not None
                  else f"  [{i+1:2d}/50] Seed {seed:3d}: FAILED")
            all_results.append(res)
            # Save incrementally
            with open(f'{RESULTS_DIR}/frozen_oos_50seeds.json', 'w') as fout:
                json.dump({'description': '50-seed frozen OOS: trained 1990-2012, tested 2013-2024',
                           'fixed_lag': FIXED_LAG, 'n_seeds': len(all_results),
                           'all_seeds': all_results}, fout, indent=2)
        except Exception as e:
            print(f"  [{i+1:2d}/50] Seed {seed:3d}: ERROR {e}")

    # Summary
    elevated_fps = [(r['seed'], r['train_ll'],
                     r['granger']['Elevated']['hml_to_smb'].get('f_p_value', 1.0),
                     r['granger']['Elevated']['hml_to_smb'].get('hac_p_value', 1.0))
                    for r in all_results
                    if r['granger']['Elevated'].get('hml_to_smb')]

    sig_f = [(s, ll, fp, hp) for s, ll, fp, hp in elevated_fps if fp < 0.05]
    sig_hac = [(s, ll, fp, hp) for s, ll, fp, hp in elevated_fps if hp < 0.05]
    print(f"\n=== SUMMARY ===")
    print(f"Seeds with Elevated F-p < 0.05: {len(sig_f)}/{len(elevated_fps)}")
    print(f"Seeds with Elevated HAC-p < 0.05: {len(sig_hac)}/{len(elevated_fps)}")
    print(f"Elevated F-p distribution:")
    fps = sorted([x[2] for x in elevated_fps])
    print(f"  min={fps[0]:.4f}  p25={fps[len(fps)//4]:.4f}  median={fps[len(fps)//2]:.4f}  p75={fps[3*len(fps)//4]:.4f}  max={fps[-1]:.4f}")

    print(f"\nSaved → {RESULTS_DIR}/frozen_oos_50seeds.json")


if __name__ == '__main__':
    main()
