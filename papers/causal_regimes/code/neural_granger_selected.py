"""
Neural Granger Causality using selected HMM fit regimes.
Adapted from neural_granger_canonical.py to read selected_fit_regimes.csv.
Output: results/neural_granger_selected.json
"""

import numpy as np
import pandas as pd
import json
import urllib.request
import zipfile
import io
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


def load_ff_data():
    """Download FF5 + MOM daily."""
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
    df5 = df5.rename(columns={'Mkt-RF': 'MKT'}).drop('RF', axis=1, errors='ignore')
    return df5.loc['1990-01-01':'2024-12-31']


def load_selected_regimes():
    """Load regime assignments from selected_fit_regimes.csv."""
    path = f"{RESULTS_DIR}/selected_fit_regimes.csv"
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    print(f"Loaded {len(df)} regime assignments from CSV")
    return df


def build_regime_lag_matrix(smb_full, hml_full, regime_labels_full, regime_name, max_lag=9):
    n = len(smb_full)
    regime_mask = (regime_labels_full == regime_name)
    n_total = int(regime_mask.sum())
    valid_indices = []
    for t in range(max_lag, n):
        if not regime_mask[t]:
            continue
        if all(regime_mask[t - lag] for lag in range(1, max_lag + 1)):
            valid_indices.append(t)
    valid_indices = np.array(valid_indices)
    n_clean = len(valid_indices)
    if n_clean < 20:
        return None, None, None, n_total, n_clean
    y = smb_full[valid_indices]
    X_r_cols = [smb_full[valid_indices - lag] for lag in range(1, max_lag + 1)]
    X_u_cols = X_r_cols + [hml_full[valid_indices - lag] for lag in range(1, max_lag + 1)]
    return y, np.column_stack(X_r_cols), np.column_stack(X_u_cols), n_total, n_clean


def linear_granger_test(y, X_r, X_u, max_lag=9):
    n = len(y)
    X_r_i = np.column_stack([np.ones(n), X_r])
    X_u_i = np.column_stack([np.ones(n), X_u])
    beta_r = np.linalg.lstsq(X_r_i, y, rcond=None)[0]
    beta_u = np.linalg.lstsq(X_u_i, y, rcond=None)[0]
    rss_r = np.sum((y - X_r_i @ beta_r) ** 2)
    rss_u = np.sum((y - X_u_i @ beta_u) ** 2)
    df1, df2 = max_lag, n - X_u_i.shape[1]
    f_stat = ((rss_r - rss_u) / df1) / (rss_u / df2) if df2 > 0 and rss_u > 0 else 0.0
    p_value = 1 - f_dist.cdf(f_stat, df1, df2) if f_stat > 0 else 1.0
    mse_r, mse_u = rss_r / n, rss_u / n
    return {
        'f_stat': float(f_stat), 'p_value': float(p_value),
        'mse_improvement_pct': float((mse_r - mse_u) / mse_r * 100) if mse_r > 0 else 0.0,
    }


def expanding_window_cv_indices(n, n_folds=5):
    block_size = n // (n_folds + 1)
    folds = []
    for k in range(n_folds):
        train_end = (k + 1) * block_size
        test_end = min((k + 2) * block_size, n)
        folds.append((np.arange(0, train_end), np.arange(train_end, test_end)))
    return folds


def nonlinear_granger_temporal(y, X_r, X_u, model_class, model_params, model_name,
                                n_folds=5, n_permutations=200, random_state=42):
    n = len(y)
    max_lag = X_r.shape[1]
    folds = expanding_window_cv_indices(n, n_folds)

    def compute_cv_mse(X_data, seed):
        mse_folds = []
        for train_idx, test_idx in folds:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_data[train_idx])
            X_test = scaler.transform(X_data[test_idx])
            model = model_class(random_state=seed, **model_params)
            model.fit(X_train, y[train_idx])
            mse_folds.append(mean_squared_error(y[test_idx], model.predict(X_test)))
        return np.mean(mse_folds)

    mse_r = compute_cv_mse(X_r, random_state)
    mse_u = compute_cv_mse(X_u, random_state)
    observed = (mse_r - mse_u) / mse_r * 100 if mse_r > 0 else 0.0
    print(f"    {model_name}: observed MSE improvement={observed:.3f}%", flush=True)

    rng = np.random.RandomState(random_state)
    null_imps = []
    t0 = time.time()
    for i in range(n_permutations):
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"      Perm {i+1}/{n_permutations} ({elapsed:.0f}s)", flush=True)
        X_u_perm = X_u.copy()
        for col in range(max_lag, X_u.shape[1]):
            X_u_perm[:, col] = rng.permutation(X_u_perm[:, col])
        perm_mse = compute_cv_mse(X_u_perm, random_state)
        null_imps.append((mse_r - perm_mse) / mse_r * 100 if mse_r > 0 else 0.0)

    null_imps = np.array(null_imps)
    p_value = (np.sum(null_imps >= observed) + 1) / (n_permutations + 1)
    print(f"    {model_name}: p={p_value:.4f} ({time.time()-t0:.0f}s)", flush=True)

    return {
        'mse_improvement_pct': float(observed),
        'p_value': float(p_value),
        'null_mean': float(np.mean(null_imps)),
        'null_std': float(np.std(null_imps)),
    }


def main():
    print("=" * 70)
    print("NEURAL GRANGER CAUSALITY (SELECTED FIT): HML -> SMB")
    print("=" * 70, flush=True)

    ff_df = load_ff_data()
    regime_df = load_selected_regimes()

    common = ff_df.index.intersection(regime_df.index).sort_values()
    smb = ff_df.loc[common, 'SMB'].values.astype(np.float64)
    hml = ff_df.loc[common, 'HML'].values.astype(np.float64)
    labels = regime_df.loc[common, 'regime_label'].values

    mlp_params = {
        'hidden_layer_sizes': (64, 32), 'activation': 'relu', 'max_iter': 500,
        'early_stopping': True, 'validation_fraction': 0.15, 'n_iter_no_change': 20,
        'learning_rate': 'adaptive', 'learning_rate_init': 0.001,
    }
    rf_params = {'n_estimators': 100, 'max_depth': None, 'min_samples_leaf': 5, 'n_jobs': -1}

    results = {}
    for regime_name in REGIME_NAMES:
        print(f"\n--- {regime_name} ---", flush=True)
        y, X_r, X_u, n_total, n_clean = build_regime_lag_matrix(smb, hml, labels, regime_name, MAX_LAG)
        if y is None:
            results[regime_name] = {'n_total': n_total, 'n_clean': n_clean, 'status': 'insufficient'}
            continue

        linear = linear_granger_test(y, X_r, X_u, MAX_LAG)
        rf = nonlinear_granger_temporal(y, X_r, X_u, RandomForestRegressor, rf_params, "RF",
                                         N_FOLDS, N_PERMUTATIONS, 42)
        mlp = nonlinear_granger_temporal(y, X_r, X_u, MLPRegressor, mlp_params, "MLP",
                                          N_FOLDS, N_PERMUTATIONS, 42)
        results[regime_name] = {
            'n_total': n_total, 'n_clean': n_clean,
            'linear_mse_improvement_pct': linear['mse_improvement_pct'],
            'linear_p_value': linear['p_value'],
            'rf_mse_improvement_pct': rf['mse_improvement_pct'],
            'rf_p_value': rf['p_value'],
            'mlp_mse_improvement_pct': mlp['mse_improvement_pct'],
            'mlp_p_value': mlp['p_value'],
        }

    output = {
        'metadata': {
            'description': 'Neural Granger HML->SMB using selected HMM fit',
            'max_lag': MAX_LAG, 'n_folds': N_FOLDS, 'n_permutations': N_PERMUTATIONS,
            'timestamp': datetime.now().isoformat(),
        },
        'results': results,
    }
    out_path = f"{RESULTS_DIR}/neural_granger_selected.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == '__main__':
    main()
