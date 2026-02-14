"""
Transfer Entropy using selected HMM fit regimes.
Adapted from te_canonical.py to read selected_fit_regimes.csv.
Output: results/te_selected.json
"""

import json
import io
import zipfile
import urllib.request
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.special import digamma
from scipy.stats import norm
import warnings
import time
from datetime import datetime

warnings.filterwarnings('ignore')

RESULTS_DIR = '/Users/i767700/Github/ai-in-finance/papers/causal_regimes/results'


def frenzel_pompe_cmi(x, y, z, k=5):
    """CMI(X; Y | Z) via Frenzel & Pompe (2007), kNN with Chebyshev distance."""
    N = len(x)
    if N < k + 5:
        return 0.0
    if x.ndim == 1: x = x.reshape(-1, 1)
    if y.ndim == 1: y = y.reshape(-1, 1)
    if z.ndim == 1: z = z.reshape(-1, 1)
    rng = np.random.RandomState(0)
    x = x + rng.randn(*x.shape) * 1e-10
    y = y + rng.randn(*y.shape) * 1e-10
    z = z + rng.randn(*z.shape) * 1e-10
    xyz = np.hstack([x, y, z])
    xz = np.hstack([x, z])
    yz = np.hstack([y, z])
    tree_xyz = cKDTree(xyz)
    tree_xz = cKDTree(xz)
    tree_yz = cKDTree(yz)
    tree_z = cKDTree(z)
    dists, _ = tree_xyz.query(xyz, k=k + 1, p=np.inf)
    eps = dists[:, -1] + 1e-15
    n_xz = np.array([tree_xz.query_ball_point(xz[i], eps[i], p=np.inf, return_length=True) - 1 for i in range(N)], dtype=float)
    n_yz = np.array([tree_yz.query_ball_point(yz[i], eps[i], p=np.inf, return_length=True) - 1 for i in range(N)], dtype=float)
    n_z = np.array([tree_z.query_ball_point(z[i], eps[i], p=np.inf, return_length=True) - 1 for i in range(N)], dtype=float)
    n_xz = np.maximum(n_xz, 1)
    n_yz = np.maximum(n_yz, 1)
    n_z = np.maximum(n_z, 1)
    return digamma(k) + np.mean(digamma(n_z + 1) - digamma(n_xz + 1) - digamma(n_yz + 1))


def load_ff_data():
    url5 = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip'
    with urllib.request.urlopen(url5, timeout=60) as response:
        data = response.read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        with z.open(z.namelist()[0]) as f:
            df5 = pd.read_csv(f, skiprows=3)
    df5.columns = df5.columns.str.strip()
    df5 = df5.rename(columns={df5.columns[0]: 'Date'})
    df5 = df5[df5['Date'].astype(str).str.match(r'^\d{8}$')]
    df5['Date'] = pd.to_datetime(df5['Date'], format='%Y%m%d')
    for col in ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']:
        df5[col] = pd.to_numeric(df5[col], errors='coerce')
    df5 = df5.set_index('Date').dropna()
    return df5.loc['1990-01-01':'2024-12-31']


def main():
    t0 = time.time()
    print("=" * 70)
    print("TRANSFER ENTROPY (SELECTED FIT): HML <-> SMB")
    print("=" * 70, flush=True)

    ff = load_ff_data()
    regime_df = pd.read_csv(f"{RESULTS_DIR}/selected_fit_regimes.csv")
    regime_df['date'] = pd.to_datetime(regime_df['date'])
    regime_df = regime_df.set_index('date')

    common = ff.index.intersection(regime_df.index).sort_values()
    hml = ff.loc[common, 'HML'].values
    smb = ff.loc[common, 'SMB'].values
    labels = regime_df.loc[common, 'regime_label'].values
    print(f"Aligned: {len(common)} days", flush=True)

    MAX_LAG = 9
    K_NN = 5
    N_PERMS = 200
    results = {}

    for regime_name in ['Normal', 'Elevated', 'Crisis']:
        print(f"\n--- {regime_name} ---", flush=True)
        regime_mask = (labels == regime_name)
        n_total = int(regime_mask.sum())

        valid_t = []
        for t in range(MAX_LAG, len(labels)):
            if all(labels[t - lag] == regime_name for lag in range(MAX_LAG + 1)):
                valid_t.append(t)
        valid_t = np.array(valid_t, dtype=int)
        n_clean = len(valid_t)
        print(f"  n_total={n_total}, n_clean={n_clean}", flush=True)

        if n_clean < MAX_LAG + K_NN + 10:
            results[regime_name] = {'n_total': n_total, 'n_clean': n_clean,
                                     'hml_to_smb': {}, 'smb_to_hml': {}}
            continue

        # TE(HML -> SMB)
        y_hs = smb[valid_t]
        x_hs = np.column_stack([hml[valid_t - l] for l in range(1, MAX_LAG + 1)])
        z_hs = np.column_stack([smb[valid_t - l] for l in range(1, MAX_LAG + 1)])

        print(f"  Computing TE(HML->SMB)...", flush=True)
        te_hs = frenzel_pompe_cmi(y_hs, x_hs, z_hs, k=K_NN)
        rng = np.random.RandomState(42)
        null_hs = np.zeros(N_PERMS)
        for p in range(N_PERMS):
            if (p+1) % 50 == 0: print(f"    Perm {p+1}/{N_PERMS}", flush=True)
            null_hs[p] = frenzel_pompe_cmi(y_hs, x_hs[rng.permutation(len(y_hs))], z_hs, k=K_NN)
        z_hs_score = (te_hs - null_hs.mean()) / null_hs.std() if null_hs.std() > 0 else 0.0
        p_hs = 1.0 - norm.cdf(z_hs_score)
        print(f"  TE(HML->SMB): z={z_hs_score:.3f}, p={p_hs:.4f}", flush=True)

        # TE(SMB -> HML)
        y_sh = hml[valid_t]
        x_sh = np.column_stack([smb[valid_t - l] for l in range(1, MAX_LAG + 1)])
        z_sh = np.column_stack([hml[valid_t - l] for l in range(1, MAX_LAG + 1)])

        print(f"  Computing TE(SMB->HML)...", flush=True)
        te_sh = frenzel_pompe_cmi(y_sh, x_sh, z_sh, k=K_NN)
        rng2 = np.random.RandomState(42)
        null_sh = np.zeros(N_PERMS)
        for p in range(N_PERMS):
            if (p+1) % 50 == 0: print(f"    Perm {p+1}/{N_PERMS}", flush=True)
            null_sh[p] = frenzel_pompe_cmi(y_sh, x_sh[rng2.permutation(len(y_sh))], z_sh, k=K_NN)
        z_sh_score = (te_sh - null_sh.mean()) / null_sh.std() if null_sh.std() > 0 else 0.0
        p_sh = 1.0 - norm.cdf(z_sh_score)
        print(f"  TE(SMB->HML): z={z_sh_score:.3f}, p={p_sh:.4f}", flush=True)

        results[regime_name] = {
            'n_total': n_total, 'n_clean': n_clean,
            'hml_to_smb': {'te': round(float(te_hs), 6), 'z_score': round(float(z_hs_score), 3),
                           'p_value': float(p_hs)},
            'smb_to_hml': {'te': round(float(te_sh), 6), 'z_score': round(float(z_sh_score), 3),
                           'p_value': float(p_sh)},
        }

    output = {
        'metadata': {'description': 'Transfer entropy using selected HMM fit',
                     'k_nn': K_NN, 'max_lag': MAX_LAG, 'n_permutations': N_PERMS,
                     'timestamp': datetime.now().isoformat()},
        'results': results,
    }
    out_path = f"{RESULTS_DIR}/te_selected.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")
    print(f"Total: {time.time()-t0:.0f}s", flush=True)


if __name__ == '__main__':
    main()
