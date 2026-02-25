"""
10,000-permutation test — percentage-unit convention, primary seed 28.
Complements permutation_10k_decimal.py (decimal-unit, p=0.063).
Percentage-unit: raw French data, no /100 conversion → n=953 clean Elevated.
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
)

RESULTS_DIR = '/Users/i767700/Github/ai-in-finance/papers/causal_regimes/results'
PRIMARY_SEED = 28
FIXED_LAG = 1
N_PERM = 10000

def granger_f(smb, hml, clean_idx, lag=1):
    usable = clean_idx[clean_idx >= lag]
    n = len(usable)
    if n < 2 * lag + 10:
        return np.nan
    t = usable
    y = smb[t]
    X_r = smb[t - 1].reshape(-1, 1)
    X_u = np.column_stack([smb[t - 1], hml[t - 1]])
    def ols_rss(X, y):
        X = np.column_stack([np.ones(len(X)), X]) if X.ndim == 1 else np.column_stack([np.ones(len(X)), X])
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        return np.sum((y - X @ b)**2)
    rr = ols_rss(X_r, y)
    ru = ols_rss(X_u, y)
    df1, df2 = 1, n - 3
    if df2 <= 0 or ru <= 0:
        return np.nan
    return ((rr - ru) / df1) / (ru / df2)


def main():
    print("Loading FF data (percentage-unit — no /100 conversion)...")
    df = download_ff_data()
    # No unit conversion: raw percentage values

    train_df = df.loc[:'2012-12-31'].copy()
    test_df  = df.loc['2013-01-01':].copy()

    smb = test_df['SMB'].values
    hml = test_df['HML'].values

    hmm_cols = ['MKT', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']
    print(f"Fitting HMM seed {PRIMARY_SEED} on train data ({len(train_df)} days)...")
    hmm = StudentTHMM(n_regimes=3, n_iter=100, tol=1e-4, random_state=PRIMARY_SEED)
    hmm.fit(train_df[hmm_cols].values)

    train_raw = hmm.predict(train_df[hmm_cols].values, use_filtered=False)
    _, remap = relabel_regimes_by_data_norm(train_df, train_raw, hmm_cols)
    test_raw, _ = hmm.predict_oos(test_df[hmm_cols].values, use_filtered=True)
    test_regimes = np.array([remap[r] for r in test_raw])

    clean_elevated = extract_regime_clean_indices(test_regimes, 1, max_lag=FIXED_LAG)
    actual_F = granger_f(smb, hml, clean_elevated, lag=FIXED_LAG)
    n_clean = len(clean_elevated)
    print(f"  Actual F={actual_F:.4f}, n_clean={n_clean}")

    rng = np.random.default_rng(seed=PRIMARY_SEED)
    perm_Fs = []
    for i in range(N_PERM):
        perm_reg = rng.permutation(test_regimes)
        clean_p = extract_regime_clean_indices(perm_reg, 1, max_lag=FIXED_LAG)
        F_p = granger_f(smb, hml, clean_p, lag=FIXED_LAG)
        perm_Fs.append(F_p)
        if (i + 1) % 1000 == 0:
            current_p = np.mean(np.array(perm_Fs[:i+1]) >= actual_F)
            print(f"  Permutation {i+1}/{N_PERM}: running p={current_p:.4f}")

    perm_arr = np.array(perm_Fs)
    valid = ~np.isnan(perm_arr)
    perm_p = float(np.mean(perm_arr[valid] >= actual_F))
    perm_95 = float(np.percentile(perm_arr[valid], 95))
    mc_se = float(np.sqrt(perm_p * (1 - perm_p) / valid.sum()))

    result = {
        "description": "10,000-permutation OOS Elevated Granger test, percentage-unit convention, seed 28",
        "convention": "percentage_unit",
        "seed": PRIMARY_SEED,
        "n_perm": N_PERM,
        "n_valid_perm": int(valid.sum()),
        "actual_F": float(actual_F),
        "n_clean_elevated": n_clean,
        "perm_p": perm_p,
        "perm_95pct_F": perm_95,
        "mc_se": mc_se,
        "ci_95_lower": round(perm_p - 1.96 * mc_se, 4),
        "ci_95_upper": round(perm_p + 1.96 * mc_se, 4),
    }
    out_path = f"{RESULTS_DIR}/permutation_10k_pctunit.json"
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n=== RESULT ===")
    print(f"p = {perm_p:.4f}  (SE={mc_se:.4f}, 95% CI [{result['ci_95_lower']}, {result['ci_95_upper']}])")
    print(f"Saved to {out_path}")


if __name__ == '__main__':
    main()
