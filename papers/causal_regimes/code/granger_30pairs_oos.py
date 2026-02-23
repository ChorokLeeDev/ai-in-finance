"""
Granger causality on all 30 directed factor pairs in OOS Elevated regime.
Reports rank of HML→SMB among 30 pairs by F-statistic and p-value.
Addresses 30-pair multiplicity concern via max-statistic rank test.
Convention: percentage-unit (matches n=953 Elevated primary spec).
"""
import sys, json, warnings
import numpy as np
import pandas as pd
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
PRIMARY_SEED = 28
FIXED_LAG = 1
FACTOR_NAMES = ['MKT', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']


def granger_full(y, x_cause, clean_idx, lag=1):
    """Return (F_stat, p_value) for x_cause → y Granger test."""
    usable = clean_idx[clean_idx >= lag]
    n = len(usable)
    if n < 2 * lag + 10:
        return np.nan, np.nan
    t = usable
    yv = y[t]
    yl = np.column_stack([y[t - i - 1] for i in range(lag)])
    xl = np.column_stack([x_cause[t - i - 1] for i in range(lag)])
    Xr = np.column_stack([np.ones(n), yl])
    Xu = np.column_stack([np.ones(n), yl, xl])
    br = np.linalg.lstsq(Xr, yv, rcond=None)[0]
    bu = np.linalg.lstsq(Xu, yv, rcond=None)[0]
    rr = float(np.sum((yv - Xr @ br)**2))
    ru = float(np.sum((yv - Xu @ bu)**2))
    df1 = lag
    df2 = n - 2 * lag - 1
    if df2 <= 0 or ru <= 0:
        return np.nan, np.nan
    F = ((rr - ru) / df1) / (ru / df2)
    p = float(1 - stats.f.cdf(F, df1, df2))
    return float(F), p


def main():
    print("Loading FF data (percentage-unit)...")
    df = download_ff_data()
    # No /100: percentage-unit convention

    train_df = df.loc[:'2012-12-31'].copy()
    test_df  = df.loc['2013-01-01':].copy()

    hmm_cols = FACTOR_NAMES
    print(f"Fitting HMM seed {PRIMARY_SEED} on train data ({len(train_df)} days)...")
    hmm = StudentTHMM(n_regimes=3, n_iter=100, tol=1e-4, random_state=PRIMARY_SEED)
    hmm.fit(train_df[hmm_cols].values)

    train_raw = hmm.predict(train_df[hmm_cols].values, use_filtered=False)
    _, remap = relabel_regimes_by_data_norm(train_df, train_raw, hmm_cols)
    test_raw, _ = hmm.predict_oos(test_df[hmm_cols].values, use_filtered=True)
    test_regimes = np.array([remap[r] for r in test_raw])

    clean_elevated = extract_regime_clean_indices(test_regimes, 1, max_lag=FIXED_LAG)
    print(f"  n_clean Elevated = {len(clean_elevated)}")

    # Extract factor arrays for test period
    factor_data = {name: test_df[name].values for name in FACTOR_NAMES}

    # Run all 30 directed pairs
    results = []
    for cause in FACTOR_NAMES:
        for effect in FACTOR_NAMES:
            if cause == effect:
                continue
            F, p = granger_full(factor_data[effect], factor_data[cause], clean_elevated, lag=FIXED_LAG)
            results.append({
                'cause': cause,
                'effect': effect,
                'pair': f"{cause}->{effect}",
                'F': round(float(F), 4) if not np.isnan(F) else None,
                'p': round(float(p), 4) if not np.isnan(p) else None,
            })

    # Sort by F-statistic descending
    valid_results = [r for r in results if r['F'] is not None]
    valid_results.sort(key=lambda x: x['F'], reverse=True)

    print("\n=== ALL 30 DIRECTED PAIRS (Elevated regime OOS, sorted by F) ===")
    hml_smb_rank_f = None
    hml_smb_rank_p = None
    for i, r in enumerate(valid_results, 1):
        marker = " <-- HML->SMB" if r['cause'] == 'HML' and r['effect'] == 'SMB' else ""
        print(f"  {i:2d}. {r['pair']:12s}  F={r['F']:.4f}  p={r['p']:.4f}{marker}")
        if r['cause'] == 'HML' and r['effect'] == 'SMB':
            hml_smb_rank_f = i

    # Rank by p-value ascending
    valid_results_by_p = sorted(valid_results, key=lambda x: x['p'])
    for i, r in enumerate(valid_results_by_p, 1):
        if r['cause'] == 'HML' and r['effect'] == 'SMB':
            hml_smb_rank_p = i
            break

    n_pairs = len(valid_results)
    max_stat_p_f = hml_smb_rank_f / n_pairs
    max_stat_p_p = hml_smb_rank_p / n_pairs

    # HML->SMB stats
    hml_smb = next(r for r in results if r['cause'] == 'HML' and r['effect'] == 'SMB')

    print(f"\n=== HML->SMB SUMMARY ===")
    print(f"  F = {hml_smb['F']:.4f}, p = {hml_smb['p']:.4f}")
    print(f"  Rank by F: {hml_smb_rank_f}/{n_pairs} (max-statistic p = {max_stat_p_f:.4f})")
    print(f"  Rank by p: {hml_smb_rank_p}/{n_pairs} (max-statistic p = {max_stat_p_p:.4f})")

    output = {
        "description": "Granger causality all 30 directed pairs, OOS Elevated regime, percentage-unit, seed 28",
        "n_clean_elevated": len(clean_elevated),
        "n_pairs": n_pairs,
        "lag": FIXED_LAG,
        "hml_smb": {
            "F": hml_smb['F'],
            "p": hml_smb['p'],
            "rank_by_F": hml_smb_rank_f,
            "rank_by_p": hml_smb_rank_p,
            "max_stat_p_by_F": round(max_stat_p_f, 4),
            "max_stat_p_by_p": round(max_stat_p_p, 4),
        },
        "all_pairs_sorted_by_F": valid_results,
    }
    out_path = f"{RESULTS_DIR}/granger_30pairs_oos.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
