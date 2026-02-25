#!/usr/bin/env python3
"""
Transfer Entropy using Canonical Regime Assignments.

Computes TE(HML->SMB) and TE(SMB->HML) per regime using:
  - Canonical regime assignments from canonical_regimes.json
  - Frenzel-Pompe (2007) kNN CMI estimator, k=5
  - Multi-lag embedding: lags 1-9 jointly (embedding dimension = 9)
  - 200-shuffle permutation test with z-score and normal-approx p-value
  - Boundary handling: all 9 lags must fall within the same regime
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

warnings.filterwarnings('ignore')

RESULTS_DIR = '/Users/i767700/Github/ai-in-finance/papers/causal_regimes/results'


# =============================================================================
# Frenzel-Pompe CMI Estimator
# =============================================================================

def frenzel_pompe_cmi(x, y, z, k=5):
    """
    Conditional mutual information CMI(X; Y | Z) via Frenzel & Pompe (2007).
    Phys. Rev. Lett. 99, 204101.
    Uses kNN with Chebyshev (L-inf) distance.
    """
    N = len(x)
    if N < k + 5:
        return 0.0

    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if z.ndim == 1:
        z = z.reshape(-1, 1)

    # Tiny jitter to break ties
    rng = np.random.RandomState(0)
    jitter_scale = 1e-10
    x = x + rng.randn(*x.shape) * jitter_scale
    y = y + rng.randn(*y.shape) * jitter_scale
    z = z + rng.randn(*z.shape) * jitter_scale

    xyz = np.hstack([x, y, z])
    xz = np.hstack([x, z])
    yz = np.hstack([y, z])

    tree_xyz = cKDTree(xyz)
    tree_xz = cKDTree(xz)
    tree_yz = cKDTree(yz)
    tree_z = cKDTree(z)

    # k-th neighbor distance in joint space (L-inf)
    dists, _ = tree_xyz.query(xyz, k=k + 1, p=np.inf)
    eps = dists[:, -1]
    eps_arr = eps + 1e-15

    # Count neighbors within eps in marginal spaces
    n_xz = np.array([
        tree_xz.query_ball_point(xz[i], eps_arr[i], p=np.inf, return_length=True) - 1
        for i in range(N)
    ], dtype=float)
    n_yz = np.array([
        tree_yz.query_ball_point(yz[i], eps_arr[i], p=np.inf, return_length=True) - 1
        for i in range(N)
    ], dtype=float)
    n_z = np.array([
        tree_z.query_ball_point(z[i], eps_arr[i], p=np.inf, return_length=True) - 1
        for i in range(N)
    ], dtype=float)

    # Floor at 1 to avoid digamma(0)
    n_xz = np.maximum(n_xz, 1)
    n_yz = np.maximum(n_yz, 1)
    n_z = np.maximum(n_z, 1)

    cmi = digamma(k) + np.mean(digamma(n_z + 1) - digamma(n_xz + 1) - digamma(n_yz + 1))
    return cmi


# =============================================================================
# Data Loading
# =============================================================================

def load_ff5_mom_daily():
    """Download FF5 + Momentum daily, filter to 1990-01-02 through 2024-12-31."""
    # FF5
    print("Downloading Fama-French 5 factors daily...")
    url5 = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
    response = urllib.request.urlopen(url5, timeout=60)
    zipdata = zipfile.ZipFile(io.BytesIO(response.read()))
    csv_name = [f for f in zipdata.namelist() if f.endswith('.CSV') or f.endswith('.csv')][0]
    raw = zipdata.read(csv_name).decode('utf-8')
    lines = raw.split('\n')
    header_idx = None
    for i, line in enumerate(lines):
        if 'Mkt-RF' in line:
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("Could not find header row in FF5 CSV")
    data_lines = []
    for i in range(header_idx, len(lines)):
        line = lines[i].strip()
        if line == '':
            break
        data_lines.append(line)
    df5 = pd.read_csv(io.StringIO('\n'.join(data_lines)))
    df5.columns = df5.columns.str.strip()
    date_col = df5.columns[0]
    df5 = df5.rename(columns={date_col: 'date'})
    df5['date'] = pd.to_datetime(df5['date'].astype(str), format='%Y%m%d')
    for col in ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']:
        df5[col] = pd.to_numeric(df5[col], errors='coerce')
    df5 = df5.dropna().set_index('date').sort_index()

    # Momentum
    print("Downloading Momentum factor daily...")
    url_mom = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip"
    response = urllib.request.urlopen(url_mom, timeout=60)
    zipdata = zipfile.ZipFile(io.BytesIO(response.read()))
    csv_name = zipdata.namelist()[0]
    raw = zipdata.read(csv_name).decode('utf-8')
    lines = raw.split('\n')
    header_idx = None
    for i, line in enumerate(lines):
        if 'Mom' in line:
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("Could not find header row in Momentum CSV")
    data_lines = []
    for i in range(header_idx, len(lines)):
        line = lines[i].strip()
        if line == '':
            break
        data_lines.append(line)
    mom = pd.read_csv(io.StringIO('\n'.join(data_lines)))
    mom.columns = mom.columns.str.strip()
    mom = mom.rename(columns={mom.columns[0]: 'date', mom.columns[1]: 'MOM'})
    mom['date'] = pd.to_datetime(mom['date'].astype(str), format='%Y%m%d')
    mom['MOM'] = pd.to_numeric(mom['MOM'], errors='coerce')
    mom = mom.dropna().set_index('date').sort_index()

    # Join
    df = df5.join(mom[['MOM']], how='inner')

    # Filter to 1990-01-02 through 2024-12-31
    df = df.loc['1990-01-02':'2024-12-31']
    print(f"  Loaded {len(df)} trading days ({df.index[0].date()} to {df.index[-1].date()})")
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
    print(f"  Loaded {len(regime_df)} regime assignments")

    for name in ['Normal', 'Elevated', 'Crisis']:
        n = (regime_df['regime_name'] == name).sum()
        print(f"    {name}: {n} days")

    return regime_df


# =============================================================================
# Boundary-clean index extraction
# =============================================================================

def get_clean_indices(regime_labels, regime_name, max_lag=9):
    """
    For each observation at time t (t >= max_lag), check that ALL indices
    t, t-1, ..., t-max_lag fall within the same regime.

    Returns: array of valid original indices t where the constraint holds.
    """
    n = len(regime_labels)
    valid_indices = []
    for t in range(max_lag, n):
        all_same = True
        for lag in range(0, max_lag + 1):
            if regime_labels[t - lag] != regime_name:
                all_same = False
                break
        if all_same:
            valid_indices.append(t)
    return np.array(valid_indices, dtype=int)


# =============================================================================
# MAIN
# =============================================================================

def main():
    t0 = time.time()
    print("=" * 80)
    print("TRANSFER ENTROPY -- CANONICAL REGIMES")
    print("Frenzel-Pompe (2007) CMI, kNN k=5, multi-lag embedding (lags 1-9)")
    print("200-shuffle permutation test, normal-approximation p-values")
    print("=" * 80)

    # 1. Load FF5+MOM data
    ff = load_ff5_mom_daily()

    # 2. Load canonical regimes
    regime_df = load_canonical_regimes()

    # 3. Align dates
    common_dates = ff.index.intersection(regime_df.index)
    common_dates = common_dates.sort_values()
    ff = ff.loc[common_dates]
    regime_df = regime_df.loc[common_dates]
    print(f"\nAligned dates: {len(common_dates)} trading days")
    print(f"  Range: {common_dates[0].date()} to {common_dates[-1].date()}")

    hml = ff['HML'].values
    smb = ff['SMB'].values
    regime_labels = regime_df['regime_name'].values

    MAX_LAG = 9
    K_NN = 5
    N_PERMS = 200

    results = {
        'description': 'Transfer entropy per canonical regime: HML<->SMB',
        'method': 'Frenzel-Pompe (2007) kNN CMI, k=5',
        'embedding': 'Multi-lag, lags 1-9 jointly (dim=9)',
        'permutations': N_PERMS,
        'boundary_rule': 'All 9 lags + current must be in same regime',
        'data_range': f'{common_dates[0].date()} to {common_dates[-1].date()}',
        'n_total_days': int(len(common_dates)),
        'regimes': {}
    }

    regime_names = ['Normal', 'Elevated', 'Crisis']

    for regime_name in regime_names:
        print(f"\n{'=' * 60}")
        print(f"REGIME: {regime_name}")
        print(f"{'=' * 60}")

        n_total = int(np.sum(regime_labels == regime_name))
        print(f"  Total days in regime: {n_total}")

        # Get boundary-clean indices
        valid_t = get_clean_indices(regime_labels, regime_name, max_lag=MAX_LAG)
        n_clean = len(valid_t)
        print(f"  Clean observations (all 9 lags in regime): {n_clean}")

        if n_clean < MAX_LAG + K_NN + 10:
            print(f"  WARNING: Too few clean observations ({n_clean}), skipping regime")
            results['regimes'][regime_name] = {
                'n_total': n_total,
                'n_clean': n_clean,
                'hml_to_smb': {'te_observed': None, 'z_score': None, 'p_value': None},
                'smb_to_hml': {'te_observed': None, 'z_score': None, 'p_value': None}
            }
            continue

        # Build arrays for CMI computation
        # TE(HML -> SMB) = CMI(SMB_t ; [HML_{t-1},...,HML_{t-9}] | [SMB_{t-1},...,SMB_{t-9}])
        y_t_hs = smb[valid_t]
        x_past_hs = np.column_stack([hml[valid_t - l] for l in range(1, MAX_LAG + 1)])
        y_past_hs = np.column_stack([smb[valid_t - l] for l in range(1, MAX_LAG + 1)])

        # TE(SMB -> HML) = CMI(HML_t ; [SMB_{t-1},...,SMB_{t-9}] | [HML_{t-1},...,HML_{t-9}])
        y_t_sh = hml[valid_t]
        x_past_sh = np.column_stack([smb[valid_t - l] for l in range(1, MAX_LAG + 1)])
        y_past_sh = np.column_stack([hml[valid_t - l] for l in range(1, MAX_LAG + 1)])

        # --- TE(HML -> SMB) ---
        print(f"\n  Computing TE(HML -> SMB)...")
        t1 = time.time()

        te_hml_smb = frenzel_pompe_cmi(y_t_hs, x_past_hs, y_past_hs, k=K_NN)

        rng = np.random.RandomState(42)
        null_hml_smb = np.zeros(N_PERMS)
        for p in range(N_PERMS):
            if (p + 1) % 50 == 0:
                print(f"    Permutation {p+1}/{N_PERMS}...")
            perm_idx = rng.permutation(len(y_t_hs))
            null_hml_smb[p] = frenzel_pompe_cmi(y_t_hs, x_past_hs[perm_idx], y_past_hs, k=K_NN)

        mean_null = np.mean(null_hml_smb)
        std_null = np.std(null_hml_smb)
        z_hml_smb = (te_hml_smb - mean_null) / std_null if std_null > 0 else 0.0
        p_hml_smb = 1.0 - norm.cdf(z_hml_smb)

        print(f"    TE(HML->SMB) = {te_hml_smb:.6f}")
        print(f"    z-score = {z_hml_smb:.3f}, p-value = {p_hml_smb:.2e}")
        print(f"    Time: {time.time() - t1:.1f}s")

        # --- TE(SMB -> HML) ---
        print(f"\n  Computing TE(SMB -> HML)...")
        t1 = time.time()

        te_smb_hml = frenzel_pompe_cmi(y_t_sh, x_past_sh, y_past_sh, k=K_NN)

        rng2 = np.random.RandomState(42)
        null_smb_hml = np.zeros(N_PERMS)
        for p in range(N_PERMS):
            if (p + 1) % 50 == 0:
                print(f"    Permutation {p+1}/{N_PERMS}...")
            perm_idx = rng2.permutation(len(y_t_sh))
            null_smb_hml[p] = frenzel_pompe_cmi(y_t_sh, x_past_sh[perm_idx], y_past_sh, k=K_NN)

        mean_null2 = np.mean(null_smb_hml)
        std_null2 = np.std(null_smb_hml)
        z_smb_hml = (te_smb_hml - mean_null2) / std_null2 if std_null2 > 0 else 0.0
        p_smb_hml = 1.0 - norm.cdf(z_smb_hml)

        print(f"    TE(SMB->HML) = {te_smb_hml:.6f}")
        print(f"    z-score = {z_smb_hml:.3f}, p-value = {p_smb_hml:.2e}")
        print(f"    Time: {time.time() - t1:.1f}s")

        results['regimes'][regime_name] = {
            'n_total': n_total,
            'n_clean': n_clean,
            'hml_to_smb': {
                'te_observed': round(float(te_hml_smb), 6),
                'z_score': round(float(z_hml_smb), 3),
                'p_value': float(f"{p_hml_smb:.4e}")
            },
            'smb_to_hml': {
                'te_observed': round(float(te_smb_hml), 6),
                'z_score': round(float(z_smb_hml), 3),
                'p_value': float(f"{p_smb_hml:.4e}")
            }
        }

    # Save results
    out_path = f"{RESULTS_DIR}/te_canonical.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n{'=' * 60}")
    print(f"Results saved to {out_path}")

    elapsed = time.time() - t0
    print(f"Total runtime: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Summary table
    print(f"\n{'=' * 80}")
    print(f"SUMMARY")
    print(f"{'=' * 80}")
    print(f"{'Regime':<12} {'n_total':>8} {'n_clean':>8}  "
          f"{'TE(H->S)':>10} {'z':>7} {'p':>10}  "
          f"{'TE(S->H)':>10} {'z':>7} {'p':>10}")
    print("-" * 100)
    for regime_name in regime_names:
        r = results['regimes'][regime_name]
        hs = r['hml_to_smb']
        sh = r['smb_to_hml']
        if hs['te_observed'] is not None:
            print(f"{regime_name:<12} {r['n_total']:>8} {r['n_clean']:>8}  "
                  f"{hs['te_observed']:>10.6f} {hs['z_score']:>7.3f} {hs['p_value']:>10.2e}  "
                  f"{sh['te_observed']:>10.6f} {sh['z_score']:>7.3f} {sh['p_value']:>10.2e}")
        else:
            print(f"{regime_name:<12} {r['n_total']:>8} {r['n_clean']:>8}  "
                  f"{'N/A':>10} {'N/A':>7} {'N/A':>10}  "
                  f"{'N/A':>10} {'N/A':>7} {'N/A':>10}")


if __name__ == '__main__':
    main()
