"""
Neural Granger Causality (Canonical Regimes): HML -> SMB
=========================================================
Uses canonical regime assignments from canonical_regimes.json.
Implements Tank et al. (2022)-style Neural Granger Causality:
  - MLP (64-32 hidden, ReLU) and Random Forest (100 trees)
  - 5-fold expanding window temporal CV (NOT random splits)
  - Permutation test (200 permutations) for significance
  - Boundary handling: all 9 lags must fall within same regime

Compares restricted (SMB lags only) vs unrestricted (SMB + HML lags).
Also computes linear Granger (OLS) for reference.

Output: results/neural_granger_canonical.json
"""

import numpy as np
import pandas as pd
import json
import urllib.request
import zipfile
import io
import sys
import time
import warnings
from datetime import datetime
from scipy.stats import f as f_dist
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

warnings.filterwarnings('ignore')

RESULTS_DIR = "/Users/i767700/Github/ai-in-finance/papers/causal_regimes/results"
MAX_LAG = 9
N_FOLDS = 5
N_PERMUTATIONS = 200
REGIME_NAMES = ["Normal", "Elevated", "Crisis"]


# =============================================================================
# DATA LOADING
# =============================================================================

def load_ff5_daily():
    """Download Fama-French 5 factors daily from Kenneth French's website."""
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
    print(f"Downloading FF5 daily from: {url}", flush=True)

    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    response = urllib.request.urlopen(req, timeout=30)
    zip_data = response.read()

    with zipfile.ZipFile(io.BytesIO(zip_data)) as z:
        csv_name = [n for n in z.namelist() if n.endswith('.CSV') or n.endswith('.csv')][0]
        with z.open(csv_name) as f:
            lines = f.read().decode('utf-8').split('\n')

    # Find header row
    header_idx = None
    for i, line in enumerate(lines):
        if 'Mkt-RF' in line or 'Mkt' in line:
            header_idx = i
            break

    if header_idx is None:
        raise ValueError("Could not find header in FF5 CSV")

    data_rows = []
    for line in lines[header_idx + 1:]:
        parts = line.strip().split(',')
        if len(parts) >= 6:
            try:
                date_str = parts[0].strip()
                if len(date_str) == 8:
                    vals = [float(p.strip()) for p in parts[1:7]]
                    data_rows.append([date_str] + vals)
            except (ValueError, IndexError):
                continue

    cols = ['date', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']
    df = pd.DataFrame(data_rows, columns=cols)
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df = df.set_index('date')
    df = df[(df.index >= '1990-01-02') & (df.index <= '2024-12-31')]
    print(f"  Loaded {len(df)} daily observations ({df.index[0].date()} to {df.index[-1].date()})", flush=True)
    return df


def load_canonical_regimes():
    """Load canonical regime assignments from JSON."""
    path = f"{RESULTS_DIR}/canonical_regimes.json"
    print(f"Loading canonical regimes from: {path}", flush=True)
    with open(path, 'r') as f:
        data = json.load(f)

    assignments = data['assignments']
    regime_df = pd.DataFrame(assignments)
    regime_df['date'] = pd.to_datetime(regime_df['date'])
    regime_df = regime_df.set_index('date')
    print(f"  Loaded {len(regime_df)} regime assignments", flush=True)
    print(f"  Regime counts: {data['metadata']['regime_counts']}", flush=True)
    return regime_df


# =============================================================================
# LAG MATRIX WITH REGIME BOUNDARY HANDLING
# =============================================================================

def build_regime_lag_matrix(smb_full, hml_full, regime_labels_full, regime_name, max_lag=9):
    """
    Build feature matrices for a specific regime with boundary handling.
    
    For an observation at time t, ALL 9 lags (t-1, ..., t-9) must fall within
    the same regime. This excludes observations near regime boundaries.
    
    Args:
        smb_full: full SMB series (numpy array, aligned with regime_labels_full)
        hml_full: full HML series (numpy array, aligned with regime_labels_full)
        regime_labels_full: regime name for each time point
        regime_name: which regime to extract
        max_lag: number of lags
    
    Returns:
        y: target SMB values
        X_r: restricted features (SMB lags only)
        X_u: unrestricted features (SMB + HML lags)
        n_total: total observations in this regime
        n_clean: observations after boundary exclusion
    """
    n = len(smb_full)
    regime_mask = (regime_labels_full == regime_name)
    n_total = int(regime_mask.sum())

    valid_indices = []
    for t in range(max_lag, n):
        if not regime_mask[t]:
            continue
        # Check that ALL lags also belong to this regime
        all_lags_in_regime = True
        for lag in range(1, max_lag + 1):
            if not regime_mask[t - lag]:
                all_lags_in_regime = False
                break
        if all_lags_in_regime:
            valid_indices.append(t)

    valid_indices = np.array(valid_indices)
    n_clean = len(valid_indices)

    if n_clean < 20:
        print(f"    WARNING: Only {n_clean} clean observations for regime {regime_name}", flush=True)
        return None, None, None, n_total, n_clean

    y = smb_full[valid_indices]

    # Build lag features
    X_r_cols = []
    X_u_cols = []
    for lag in range(1, max_lag + 1):
        X_r_cols.append(smb_full[valid_indices - lag])
        X_u_cols.append(smb_full[valid_indices - lag])
    for lag in range(1, max_lag + 1):
        X_u_cols.append(hml_full[valid_indices - lag])

    X_r = np.column_stack(X_r_cols)
    X_u = np.column_stack(X_u_cols)

    return y, X_r, X_u, n_total, n_clean


# =============================================================================
# LINEAR GRANGER TEST (OLS F-test + MSE comparison)
# =============================================================================

def linear_granger_test(y, X_r, X_u, max_lag=9):
    """Standard linear Granger causality via OLS F-test with MSE comparison."""
    n = len(y)

    X_r_i = np.column_stack([np.ones(n), X_r])
    X_u_i = np.column_stack([np.ones(n), X_u])

    beta_r = np.linalg.lstsq(X_r_i, y, rcond=None)[0]
    beta_u = np.linalg.lstsq(X_u_i, y, rcond=None)[0]

    rss_r = np.sum((y - X_r_i @ beta_r) ** 2)
    rss_u = np.sum((y - X_u_i @ beta_u) ** 2)

    df1 = max_lag
    df2 = n - X_u_i.shape[1]

    if df2 > 0 and rss_u > 0:
        f_stat = ((rss_r - rss_u) / df1) / (rss_u / df2)
        p_value = 1 - f_dist.cdf(f_stat, df1, df2)
    else:
        f_stat = 0.0
        p_value = 1.0

    mse_r = rss_r / n
    mse_u = rss_u / n
    mse_improvement = (mse_r - mse_u) / mse_r * 100 if mse_r > 0 else 0.0

    return {
        'f_stat': float(f_stat),
        'p_value': float(p_value),
        'mse_restricted': float(mse_r),
        'mse_unrestricted': float(mse_u),
        'mse_improvement_pct': float(mse_improvement),
        'n_obs': int(n)
    }


# =============================================================================
# EXPANDING WINDOW TEMPORAL CV
# =============================================================================

def expanding_window_cv_indices(n, n_folds=5):
    """
    Generate expanding window temporal CV fold indices.
    
    Fold k uses the first (k+1)/n_folds fraction of data for training
    and the next 1/n_folds fraction for testing. This preserves temporal order.
    
    Specifically for n_folds=5:
      Fold 0: train=[0, n/5),        test=[n/5, 2n/5)
      Fold 1: train=[0, 2n/5),       test=[2n/5, 3n/5)
      Fold 2: train=[0, 3n/5),       test=[3n/5, 4n/5)
      Fold 3: train=[0, 4n/5),       test=[4n/5, n)
      Fold 4: train=[0, 4n/5),       test=[4n/5, n)  (same as fold 3 but train=[0, n))
    
    Actually, standard expanding window with 5 folds and min training size:
      Split data into n_folds+1 = 6 blocks.
      Fold k: train = blocks 0..k, test = block k+1
    """
    block_size = n // (n_folds + 1)
    folds = []
    for k in range(n_folds):
        train_end = (k + 1) * block_size
        test_start = train_end
        test_end = min((k + 2) * block_size, n)
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)
        if len(train_idx) > 0 and len(test_idx) > 0:
            folds.append((train_idx, test_idx))
    return folds


# =============================================================================
# NONLINEAR GRANGER: TEMPORAL CV + PERMUTATION TEST
# =============================================================================

def nonlinear_granger_temporal(y, X_r, X_u, model_class, model_params,
                                model_name, n_folds=5, n_permutations=200,
                                random_state=42):
    """
    Nonlinear Granger causality via expanding-window temporal CV + permutation test.
    
    Args:
        y: target values (temporally ordered)
        X_r: restricted features (SMB lags only)
        X_u: unrestricted features (SMB + HML lags)
        model_class: sklearn model class
        model_params: dict of model hyperparameters
        model_name: string label for printing
        n_folds: number of CV folds
        n_permutations: number of permutations for null distribution
        random_state: random seed
    
    Returns:
        dict with MSE improvement, p-value, etc.
    """
    n = len(y)
    max_lag = X_r.shape[1]  # number of SMB lag columns

    # Generate temporal CV folds
    folds = expanding_window_cv_indices(n, n_folds=n_folds)
    actual_folds = len(folds)

    def compute_temporal_cv_mse(X_data, seed):
        """Compute expanding-window temporal CV MSE."""
        mse_folds = []
        for train_idx, test_idx in folds:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_data[train_idx])
            X_test = scaler.transform(X_data[test_idx])
            y_train = y[train_idx]
            y_test = y[test_idx]

            model = model_class(random_state=seed, **model_params)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            mse_folds.append(mean_squared_error(y_test, pred))
        return np.mean(mse_folds)

    # Restricted model
    print(f"        {model_name} restricted CV MSE...", flush=True)
    mse_r = compute_temporal_cv_mse(X_r, random_state)

    # Unrestricted model
    print(f"        {model_name} unrestricted CV MSE...", flush=True)
    mse_u = compute_temporal_cv_mse(X_u, random_state)

    observed_improvement = (mse_r - mse_u) / mse_r * 100 if mse_r > 0 else 0.0
    print(f"        {model_name} observed MSE improvement: {observed_improvement:.3f}%", flush=True)

    # Permutation test: shuffle HML columns, recompute unrestricted MSE
    print(f"        {model_name} running {n_permutations} permutations...", flush=True)
    rng = np.random.RandomState(random_state)
    null_improvements = []

    hml_col_start = max_lag  # HML columns start after SMB lag columns

    t0 = time.time()
    for perm_i in range(n_permutations):
        if (perm_i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (perm_i + 1) / elapsed
            eta = (n_permutations - perm_i - 1) / rate
            print(f"          Permutation {perm_i + 1}/{n_permutations} "
                  f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)", flush=True)

        X_u_perm = X_u.copy()
        # Shuffle each HML lag column independently
        for col in range(hml_col_start, X_u.shape[1]):
            X_u_perm[:, col] = rng.permutation(X_u_perm[:, col])

        perm_mse = compute_temporal_cv_mse(X_u_perm, random_state)
        perm_imp = (mse_r - perm_mse) / mse_r * 100 if mse_r > 0 else 0.0
        null_improvements.append(perm_imp)

    null_improvements = np.array(null_improvements)
    # p-value: fraction of null improvements >= observed (with continuity correction)
    p_value = (np.sum(null_improvements >= observed_improvement) + 1) / (n_permutations + 1)

    elapsed_total = time.time() - t0
    print(f"        {model_name} permutation test done in {elapsed_total:.0f}s, p={p_value:.4f}", flush=True)

    return {
        'mse_restricted': float(mse_r),
        'mse_unrestricted': float(mse_u),
        'mse_improvement_pct': float(observed_improvement),
        'permutation_p_value': float(p_value),
        'null_improvement_mean': float(np.mean(null_improvements)),
        'null_improvement_std': float(np.std(null_improvements)),
        'n_permutations': n_permutations,
        'n_folds': actual_folds,
        'n_obs': int(n)
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("NEURAL GRANGER CAUSALITY (CANONICAL REGIMES): HML -> SMB")
    print("Expanding-window temporal CV + Permutation test")
    print("=" * 80)
    print(flush=True)

    overall_start = time.time()

    # ---- Load data ----
    ff_df = load_ff5_daily()
    regime_df = load_canonical_regimes()

    # ---- Align data ----
    common_dates = ff_df.index.intersection(regime_df.index)
    common_dates = common_dates.sort_values()
    print(f"\n  Common dates: {len(common_dates)} "
          f"({common_dates[0].date()} to {common_dates[-1].date()})", flush=True)

    ff_aligned = ff_df.loc[common_dates]
    regime_aligned = regime_df.loc[common_dates]

    smb_full = ff_aligned['SMB'].values.astype(np.float64)
    hml_full = ff_aligned['HML'].values.astype(np.float64)
    regime_labels_full = regime_aligned['regime_name'].values

    print(f"\n  Regime distribution in aligned data:", flush=True)
    for rname in REGIME_NAMES:
        count = np.sum(regime_labels_full == rname)
        print(f"    {rname}: {count}", flush=True)

    # ---- Model specifications ----
    mlp_params = {
        'hidden_layer_sizes': (64, 32),
        'activation': 'relu',
        'max_iter': 500,
        'early_stopping': True,
        'validation_fraction': 0.15,
        'n_iter_no_change': 20,
        'learning_rate': 'adaptive',
        'learning_rate_init': 0.001,
    }

    rf_params = {
        'n_estimators': 100,
        'max_depth': None,
        'min_samples_leaf': 5,
        'n_jobs': -1,
    }

    # ---- Per-regime analysis ----
    results = {}

    for regime_name in REGIME_NAMES:
        print(f"\n{'='*70}", flush=True)
        print(f"  REGIME: {regime_name}", flush=True)
        print(f"{'='*70}", flush=True)

        # Build lag matrix with boundary handling
        y, X_r, X_u, n_total, n_clean = build_regime_lag_matrix(
            smb_full, hml_full, regime_labels_full, regime_name, max_lag=MAX_LAG
        )

        print(f"    n_total (regime days): {n_total}", flush=True)
        print(f"    n_clean (after boundary exclusion): {n_clean}", flush=True)

        if y is None:
            print(f"    SKIPPING: too few observations", flush=True)
            results[regime_name] = {
                'n_total': n_total,
                'n_clean': n_clean,
                'status': 'skipped_insufficient_data'
            }
            continue

        print(f"    Boundary exclusion removed {n_total - n_clean} observations "
              f"({(n_total - n_clean) / n_total * 100:.1f}%)", flush=True)

        # --- Linear Granger ---
        print(f"\n    [1/3] Linear Granger (OLS F-test)...", flush=True)
        linear_result = linear_granger_test(y, X_r, X_u, max_lag=MAX_LAG)
        print(f"      F-stat={linear_result['f_stat']:.3f}, "
              f"p={linear_result['p_value']:.4f}, "
              f"MSE improvement={linear_result['mse_improvement_pct']:.3f}%", flush=True)

        # --- Random Forest Granger ---
        print(f"\n    [2/3] Random Forest Granger (100 trees, temporal CV)...", flush=True)
        rf_result = nonlinear_granger_temporal(
            y, X_r, X_u,
            model_class=RandomForestRegressor,
            model_params=rf_params,
            model_name="RF",
            n_folds=N_FOLDS,
            n_permutations=N_PERMUTATIONS,
            random_state=42
        )

        # --- MLP Granger ---
        print(f"\n    [3/3] MLP Granger (64-32 hidden, ReLU, temporal CV)...", flush=True)
        mlp_result = nonlinear_granger_temporal(
            y, X_r, X_u,
            model_class=MLPRegressor,
            model_params=mlp_params,
            model_name="MLP",
            n_folds=N_FOLDS,
            n_permutations=N_PERMUTATIONS,
            random_state=42
        )

        results[regime_name] = {
            'n_total': n_total,
            'n_clean': n_clean,
            'linear': {
                'f_stat': linear_result['f_stat'],
                'p_value': linear_result['p_value'],
                'mse_improvement_pct': linear_result['mse_improvement_pct'],
                'mse_restricted': linear_result['mse_restricted'],
                'mse_unrestricted': linear_result['mse_unrestricted'],
            },
            'rf': {
                'mse_improvement_pct': rf_result['mse_improvement_pct'],
                'p_value': rf_result['permutation_p_value'],
                'mse_restricted': rf_result['mse_restricted'],
                'mse_unrestricted': rf_result['mse_unrestricted'],
                'null_improvement_mean': rf_result['null_improvement_mean'],
                'null_improvement_std': rf_result['null_improvement_std'],
                'n_folds': rf_result['n_folds'],
            },
            'mlp': {
                'mse_improvement_pct': mlp_result['mse_improvement_pct'],
                'p_value': mlp_result['permutation_p_value'],
                'mse_restricted': mlp_result['mse_restricted'],
                'mse_unrestricted': mlp_result['mse_unrestricted'],
                'null_improvement_mean': mlp_result['null_improvement_mean'],
                'null_improvement_std': mlp_result['null_improvement_std'],
                'n_folds': mlp_result['n_folds'],
            },
        }

    # ---- Summary ----
    print(f"\n\n{'='*80}", flush=True)
    print("SUMMARY: Neural Granger Causality HML -> SMB", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"{'Regime':<12} {'n_clean':>7} {'Linear%':>9} {'RF%':>9} {'RF p':>8} {'MLP%':>9} {'MLP p':>8}", flush=True)
    print("-" * 70, flush=True)

    for regime_name in REGIME_NAMES:
        r = results[regime_name]
        if 'status' in r and r['status'] == 'skipped_insufficient_data':
            print(f"{regime_name:<12} {r['n_clean']:>7} {'SKIP':>9} {'SKIP':>9} {'SKIP':>8} {'SKIP':>9} {'SKIP':>8}", flush=True)
            continue
        print(f"{regime_name:<12} {r['n_clean']:>7} "
              f"{r['linear']['mse_improvement_pct']:>8.3f}% "
              f"{r['rf']['mse_improvement_pct']:>8.3f}% "
              f"{r['rf']['p_value']:>8.4f} "
              f"{r['mlp']['mse_improvement_pct']:>8.3f}% "
              f"{r['mlp']['p_value']:>8.4f}", flush=True)

    # ---- Save ----
    output = {
        'metadata': {
            'description': 'Neural Granger causality HML->SMB by canonical regime',
            'method': 'Expanding-window temporal CV + permutation test',
            'max_lag': MAX_LAG,
            'n_folds': N_FOLDS,
            'n_permutations': N_PERMUTATIONS,
            'models': {
                'linear': 'OLS with F-test',
                'rf': 'RandomForest(100 trees, min_samples_leaf=5)',
                'mlp': 'MLPRegressor(64-32, ReLU, early_stopping)',
            },
            'boundary_handling': 'All 9 lags must fall within same regime',
            'date_range': '1990-01-02 to 2024-12-31',
            'timestamp': datetime.now().isoformat(),
        },
        'results': {}
    }

    # Flatten for cleaner output format
    for regime_name in REGIME_NAMES:
        r = results[regime_name]
        if 'status' in r and r['status'] == 'skipped_insufficient_data':
            output['results'][regime_name] = {
                'n_total': r['n_total'],
                'n_clean': r['n_clean'],
                'status': 'skipped_insufficient_data',
            }
        else:
            output['results'][regime_name] = {
                'n_total': r['n_total'],
                'n_clean': r['n_clean'],
                'linear_mse_improvement_pct': r['linear']['mse_improvement_pct'],
                'linear_f_stat': r['linear']['f_stat'],
                'linear_p_value': r['linear']['p_value'],
                'rf_mse_improvement_pct': r['rf']['mse_improvement_pct'],
                'rf_p_value': r['rf']['p_value'],
                'rf_mse_restricted': r['rf']['mse_restricted'],
                'rf_mse_unrestricted': r['rf']['mse_unrestricted'],
                'rf_null_improvement_mean': r['rf']['null_improvement_mean'],
                'rf_null_improvement_std': r['rf']['null_improvement_std'],
                'mlp_mse_improvement_pct': r['mlp']['mse_improvement_pct'],
                'mlp_p_value': r['mlp']['p_value'],
                'mlp_mse_restricted': r['mlp']['mse_restricted'],
                'mlp_mse_unrestricted': r['mlp']['mse_unrestricted'],
                'mlp_null_improvement_mean': r['mlp']['null_improvement_mean'],
                'mlp_null_improvement_std': r['mlp']['null_improvement_std'],
            }

    out_path = f"{RESULTS_DIR}/neural_granger_canonical.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {out_path}", flush=True)

    total_time = time.time() - overall_start
    print(f"\nTotal runtime: {total_time:.0f}s ({total_time/60:.1f} min)", flush=True)


if __name__ == '__main__':
    main()
