"""
FF25 Portfolio Overlap Analysis — Normal Regime, Seed 28 (Primary Fit)
======================================================================

Replicates the FF25 overlap mechanism analysis from ff25_overlap_mechanism.py
but uses the PRIMARY HMM fit (seed=28, full-sample 1990-2024) and tests the
Normal regime (Bonferroni-significant Granger: p=8.75e-9).

Motivation:
  The existing analysis (ff25_overlap_mechanism.py) tests the Crisis regime. The method critic identified this as a structural
  weakness: the mechanism evidence and the primary statistical claim come from
  different model fits. This script runs the spatial gradient test under the
  primary fit, in the Normal regime, which carries the Bonferroni-significant
  Granger result. If rho_s > 0 and significant, one model simultaneously delivers
  all key findings.

Design:
  - HMM: K=3 Student-t, seed=28, full-sample 1990-2024
  - Regime labeling: relabel_regimes_by_data_norm (canonical, data-norm-based)
  - Regime tested: Normal (id=0 after relabeling)
  - Test: HML -> Portfolio Granger F-test, lag=1 (BIC-selected in main analysis)
  - Spatial gradient: Spearman rank(overlap_score, -log10(p)), 10,000 permutations
  - Overlap score: (4 - size_quintile) + BM_quintile, range 0-8

Outputs:
  results/ff25_overlap_normal_seed28.json
"""

import numpy as np
import pandas as pd
import json
import urllib.request
import zipfile
import io
import os
import sys
from scipy import stats
from scipy.special import gammaln
from scipy.optimize import minimize_scalar
from scipy.cluster.vq import kmeans2
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not BASE_DIR or BASE_DIR == '/':
    BASE_DIR = '/Users/i767700/Github/ai-in-finance/papers/causal_regimes'
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Add code dir to path to import canonical functions
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)

from multistart_hmm_pipeline import (
    download_ff_data,
    StudentTHMM,
    relabel_regimes_by_data_norm,
    extract_regime_clean_indices,
)

FACTOR_COLS = ['MKT', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']
PRIMARY_SEED = 28
LAG = 1        # pre-fixed from BIC in main analysis
NORMAL_ID = 0  # after relabeling by data norm
N_PERMS = 10000


# =============================================================================
# FF25 DATA
# =============================================================================

def download_ff25_portfolios():
    """Download FF 25 Size×BM portfolios (daily, value-weighted)."""
    cached = os.path.join(DATA_DIR, '25_Portfolios_5x5_Daily.csv')
    if os.path.exists(cached):
        print(f"  Using cached FF25 data: {cached}")
        with open(cached, 'rb') as f:
            raw_csv = f.read()
    else:
        print("Downloading FF 25 Size×BM portfolios (daily)...")
        url = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/25_Portfolios_5x5_Daily_CSV.zip'
        with urllib.request.urlopen(url, timeout=60) as response:
            raw_data = response.read()
        with zipfile.ZipFile(io.BytesIO(raw_data)) as z:
            csv_name = z.namelist()[0]
            with z.open(csv_name) as f:
                raw_csv = f.read()
        with open(cached, 'wb') as out:
            out.write(raw_csv)

    lines = raw_csv.decode('utf-8', errors='replace').split('\n')

    # Find header line ("SMALL LoBM" or similar)
    header_idx = None
    for i, line in enumerate(lines):
        if 'SMALL' in line and 'BM' in line:
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("Could not find FF25 header line")

    # Find end of first (value-weighted) section
    end_idx = len(lines)
    for i in range(header_idx + 2, len(lines)):
        stripped = lines[i].strip()
        if stripped == '' or 'Equal' in stripped or 'Average' in stripped:
            end_idx = i
            break

    from io import StringIO
    data_text = '\n'.join(lines[header_idx:end_idx])
    df = pd.read_csv(StringIO(data_text))
    df.columns = df.columns.str.strip()
    df = df.rename(columns={df.columns[0]: 'Date'})
    df['Date'] = df['Date'].astype(str).str.strip()
    df = df[df['Date'].str.match(r'^\d{8}$')]
    df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
    for col in df.columns:
        if col != 'Date':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.replace([-99.99, -999], np.nan)
    df = df.set_index('Date').sort_index()
    df = df.loc['1990-01-01':'2024-12-31']
    df = df.dropna(how='all')

    # Map columns to Size/BM labels (order: S1B1..S1B5, S2B1..S2B5, ..., S5B5)
    size_labels = ['S', '2', '3', '4', 'B']
    bm_labels = ['L', '2', '3', '4', 'H']
    portfolio_names = [f"{size_labels[si]}/{bm_labels[bi]}"
                       for si in range(5) for bi in range(5)]

    actual_cols = [c for c in df.columns
                   if 'BM' in c or 'SMALL' in c.upper() or 'BIG' in c.upper() or 'ME' in c.upper()]
    if len(actual_cols) < 25:
        actual_cols = list(df.columns[:25])
    if len(actual_cols) < 25:
        raise ValueError(f"Expected >=25 FF25 columns, got {len(actual_cols)}")

    cols_25 = actual_cols[:25]
    col_map = dict(zip(cols_25, portfolio_names))
    df = df[cols_25].rename(columns=col_map)
    print(f"  FF25: {df.shape[0]} days, {df.shape[1]} portfolios")
    return df, portfolio_names, size_labels, bm_labels


# =============================================================================
# GRANGER F-TEST (fast, single lag, for per-portfolio tests)
# =============================================================================

def granger_f_at_lag(y_all, x_all, clean_idx, lag=1):
    """F-test Granger: x -> y within clean_idx observations."""
    usable = np.array([i for i in clean_idx if i >= lag])
    if len(usable) < 2 * lag + 10:
        return 1.0, 0.0, 0.0, len(usable)

    y_curr = y_all[usable]
    y_lagged = np.column_stack([y_all[usable - i - 1] for i in range(lag)])
    x_lagged = np.column_stack([x_all[usable - i - 1] for i in range(lag)])

    X_r = np.column_stack([np.ones(len(y_curr)), y_lagged])
    X_u = np.column_stack([np.ones(len(y_curr)), y_lagged, x_lagged])

    try:
        beta_r = np.linalg.lstsq(X_r, y_curr, rcond=None)[0]
        beta_u = np.linalg.lstsq(X_u, y_curr, rcond=None)[0]
        rss_r = np.sum((y_curr - X_r @ beta_r) ** 2)
        rss_u = np.sum((y_curr - X_u @ beta_u) ** 2)
        df1, df2 = lag, len(y_curr) - 2 * lag - 1
        if df2 > 0 and rss_u > 0:
            f_stat = ((rss_r - rss_u) / df1) / (rss_u / df2)
            p_val = float(1 - stats.f.cdf(f_stat, df1, df2))
            tss = np.sum((y_curr - y_curr.mean()) ** 2)
            dr2 = float((rss_r - rss_u) / tss) if tss > 0 else 0.0
            return p_val, float(f_stat), dr2, len(usable)
    except Exception:
        pass
    return 1.0, 0.0, 0.0, len(usable)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("FF25 OVERLAP ANALYSIS — NORMAL REGIME, SEED 28 (PRIMARY FIT)")
    print("=" * 70)

    # ---- 1. Load FF5+MOM factor data (canonical) ----
    print("\nLoading FF data...")
    df = download_ff_data()   # returns df with columns MKT, SMB, HML, RMW, CMA, MOM
    print(f"  Factor data: {len(df)} days ({df.index[0].date()} to {df.index[-1].date()})")

    # ---- 2. Fit HMM with seed=28 ----
    print(f"\nFitting Student-t HMM (K=3, seed={PRIMARY_SEED})...")
    X = df[FACTOR_COLS].values
    hmm = StudentTHMM(n_regimes=3, n_iter=200, tol=1e-5, random_state=PRIMARY_SEED)
    hmm.fit(X)
    regimes_raw = hmm.predict(X, use_filtered=False)

    # ---- 3. Relabel by data norm (canonical) ----
    regimes, order = relabel_regimes_by_data_norm(df, regimes_raw, FACTOR_COLS)
    regime_names = {0: 'Normal', 1: 'Elevated', 2: 'Crisis'}
    print(f"  Log-likelihood: {hmm.log_likelihood_:.2f}")
    print(f"  Relabeling order (raw→named): {order}")
    for k in range(3):
        n = int((regimes == k).sum())
        print(f"  Regime {k} ({regime_names[k]}): {n} days ({n/len(regimes)*100:.1f}%)")

    # Verify Normal regime is consistent with primary analysis (expect ~5800+ days full sample)
    n_normal = int((regimes == NORMAL_ID).sum())
    print(f"\n  Normal regime count: {n_normal} (expect ~5800 full-sample primary fit)")

    # ---- 4. Download FF25 portfolios ----
    print("\nLoading FF25 portfolios...")
    df_25, portfolio_names, size_labels, bm_labels = download_ff25_portfolios()

    # Align dates
    df_25 = df_25[~df_25.index.duplicated(keep='first')]
    common_dates = df.index.intersection(df_25.index)
    df_aligned = df.loc[common_dates]
    df_25_aligned = df_25.loc[common_dates]
    regimes_aligned = regimes[df.index.isin(common_dates)]
    print(f"  Common dates: {len(common_dates)} ({common_dates[0].date()} to {common_dates[-1].date()})")

    # ---- 5. Extract clean Normal-regime indices ----
    clean_idx = extract_regime_clean_indices(regimes_aligned, NORMAL_ID, max_lag=LAG)
    print(f"\n  Clean Normal indices (lag={LAG}): {len(clean_idx)}")

    # ---- 6. Overlap scores ----
    # Higher = more dual-factor exposure (small-cap + high-BM)
    # score(si, bi) = (4 - si) + bi  [si: 0=Small..4=Big, bi: 0=Low..4=High]
    overlap_scores = np.array([(4 - si) + bi for si in range(5) for bi in range(5)],
                               dtype=float)

    print("\n  Overlap score matrix (higher = more overlap with SMB+HML long legs):")
    header = "       " + "".join(f"{b:>5}" for b in bm_labels)
    print(f"  {header}")
    for si in range(5):
        row = f"  {size_labels[si]:>5}  " + "".join(
            f"{(4-si)+bi:>5.0f}" for bi in range(5))
        print(row)

    # ---- 7. Per-portfolio Granger: HML -> portfolio, Normal regime ----
    print(f"\n{'='*70}")
    print(f"PER-PORTFOLIO GRANGER TESTS (HML → Portfolio, Normal, lag={LAG})")
    print(f"{'='*70}")

    hml = df_aligned['HML'].values
    p_grid = np.ones((5, 5))
    f_grid = np.zeros((5, 5))
    dr2_grid = np.zeros((5, 5))
    n_grid = np.zeros((5, 5), dtype=int)
    detail = {}

    bonferroni_alpha = 0.05 / 25

    for si in range(5):
        for bi in range(5):
            pname = f"{size_labels[si]}/{bm_labels[bi]}"
            port = df_25_aligned[pname].values
            p, f, dr2, n = granger_f_at_lag(hml, port, clean_idx, lag=LAG)
            p_grid[si, bi] = p
            f_grid[si, bi] = f
            dr2_grid[si, bi] = dr2
            n_grid[si, bi] = n
            detail[pname] = {
                'n_obs': n,
                'p_value': round(p, 6),
                'f_stat': round(f, 4),
                'delta_r2': round(dr2, 6),
                'bonferroni_significant': bool(p < bonferroni_alpha),
                'nominally_significant': bool(p < 0.05),
            }

    # Print p-value grid
    print(f"\n  Granger p-values (HML → Portfolio, Normal regime, lag={LAG}):")
    print(f"  Bonferroni threshold: {bonferroni_alpha:.4f}")
    print(f"  {'':>5}", end='')
    for b in bm_labels:
        print(f"  {b:>8}", end='')
    print()
    for si in range(5):
        print(f"  {size_labels[si]:>5}", end='')
        for bi in range(5):
            p = p_grid[si, bi]
            star = "**" if p < bonferroni_alpha else ("*" if p < 0.05 else "  ")
            print(f"  {p:>6.4f}{star}", end='')
        print()

    # ---- 8. Spatial gradient test ----
    print(f"\n{'='*70}")
    print("SPATIAL GRADIENT TEST")
    print(f"{'='*70}")

    neg_log_p = -np.log10(np.maximum(p_grid.flatten(), 1e-300))

    rho_obs, p_spearman = stats.spearmanr(overlap_scores, neg_log_p)
    tau_obs, p_kendall = stats.kendalltau(overlap_scores, neg_log_p)

    print(f"\n  Observed Spearman rho: {rho_obs:.4f}  (asymptotic p={p_spearman:.4f})")
    print(f"  Observed Kendall tau:  {tau_obs:.4f}  (asymptotic p={p_kendall:.4f})")

    # 10,000 spatial permutations
    np.random.seed(28)
    rho_perms = np.zeros(N_PERMS)
    for i in range(N_PERMS):
        shuffled = neg_log_p.copy()
        np.random.shuffle(shuffled)
        rho_perms[i], _ = stats.spearmanr(overlap_scores, shuffled)

    perm_p = float(np.mean(rho_perms >= rho_obs))
    print(f"\n  Permutation p-value ({N_PERMS:,} shuffles): {perm_p:.4f}")
    print(f"  Perm dist: mean={rho_perms.mean():.4f}, std={rho_perms.std():.4f}")
    print(f"  Observed rho is {(rho_obs-rho_perms.mean())/rho_perms.std():.2f} SD above perm mean")

    if perm_p < 0.05:
        print(f"  >>> SIGNIFICANT spatial gradient confirmed (p < 0.05)")
    elif perm_p < 0.10:
        print(f"  >>> MARGINAL spatial gradient (p < 0.10)")
    else:
        print(f"  >>> Not significant at p < 0.05")

    # ---- 9. Summary stats ----
    n_bonf = sum(1 for si in range(5) for bi in range(5) if p_grid[si, bi] < bonferroni_alpha)
    n_nom  = sum(1 for si in range(5) for bi in range(5) if p_grid[si, bi] < 0.05)

    print(f"\n  Bonferroni-significant portfolios: {n_bonf}/25")
    print(f"  Nominally significant (p<0.05):    {n_nom}/25")

    # ---- 10. Save results ----
    results = {
        'description': 'FF25 Overlap Mechanism — Normal Regime, Seed 28 (Primary Fit)',
        'seed': PRIMARY_SEED,
        'regime_tested': 'Normal',
        'regime_id': NORMAL_ID,
        'lag': LAG,
        'n_clean_normal_obs': int(len(clean_idx)),
        'n_common_dates': int(len(common_dates)),
        'hmm_log_likelihood': float(hmm.log_likelihood_),
        'relabel_order': [int(x) for x in order],
        'regime_counts': {regime_names[k]: int((regimes == k).sum()) for k in range(3)},
        'overlap_score_definition': '(4 - size_quintile) + BM_quintile; range 0-8; S/H=8, B/L=0',
        'spatial_gradient_test': {
            'spearman_rho': round(float(rho_obs), 4),
            'spearman_p_asymptotic': round(float(p_spearman), 4),
            'kendall_tau': round(float(tau_obs), 4),
            'kendall_p_asymptotic': round(float(p_kendall), 4),
            'permutation_p': round(float(perm_p), 4),
            'n_permutations': N_PERMS,
            'significant_at_005': bool(perm_p < 0.05),
            'significant_at_010': bool(perm_p < 0.10),
        },
        'granger_grid': {
            'bonferroni_threshold': round(float(bonferroni_alpha), 4),
            'n_bonferroni_significant': n_bonf,
            'n_nominally_significant': n_nom,
            'per_portfolio': detail,
            'p_value_grid': p_grid.tolist(),
            'f_stat_grid': f_grid.tolist(),
            'neg_log10_p_grid': (-np.log10(np.maximum(p_grid, 1e-300))).tolist(),
        },
    }

    out_path = os.path.join(RESULTS_DIR, 'ff25_overlap_normal_seed28.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved → {out_path}")

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Seed: {PRIMARY_SEED} (primary fit)   Regime: Normal   Lag: {LAG}")
    print(f"  Clean Normal obs: {len(clean_idx)}")
    print(f"  Spearman rho = {rho_obs:.4f}   permutation p = {perm_p:.4f}")
    print(f"  Bonferroni-sig portfolios: {n_bonf}/25   Nominal sig: {n_nom}/25")
    print("  Done.")


if __name__ == '__main__':
    main()
