"""
Neural Granger Causality: HML -> SMB by Regime
===============================================
Implements Tank et al. (2022)-style Neural Granger Causality comparison.

For each HMM regime, compares:
  1. Linear Granger (OLS F-test)
  2. MLP Granger (sklearn MLPRegressor)
  3. Random Forest Granger (sklearn RandomForestRegressor)

Tests whether nonlinear methods find stronger causal effects than linear,
especially in the Crisis regime where nonlinear dynamics are expected.

Output: JSON with per-regime, per-method MSE improvement and significance.
"""

import numpy as np
import pandas as pd
import json
import urllib.request
import zipfile
import io
import sys
import warnings
from datetime import datetime
from scipy import stats
from scipy.special import gammaln
from scipy.optimize import minimize_scalar
from scipy.cluster.vq import kmeans2
from scipy.stats import f as f_dist
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error

warnings.filterwarnings('ignore')

# =============================================================================
# StudentTHMM (from critical_fixes_analysis.py)
# =============================================================================

class StudentTHMM:
    """Student-t HMM with both filtered and smoothed probability outputs."""

    def __init__(self, n_regimes=3, n_iter=100, tol=1e-4, random_state=42):
        self.n_regimes = n_regimes
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state
        self.mu = None
        self.Sigma = None
        self.nu = None
        self.A = None
        self.pi = None
        self.gamma = None
        self.alpha = None
        self.log_likelihood_ = None

    def _init_params(self, X):
        np.random.seed(self.random_state)
        T, d = X.shape
        K = self.n_regimes
        centroids, labels = kmeans2(X, K, minit='++')
        norms = np.linalg.norm(centroids, axis=1)
        order = np.argsort(norms)
        centroids = centroids[order]
        new_labels = np.zeros_like(labels)
        for new_k, old_k in enumerate(order):
            new_labels[labels == old_k] = new_k
        labels = new_labels
        self.mu = centroids
        self.Sigma = np.zeros((K, d, d))
        for k in range(K):
            mask = labels == k
            if mask.sum() > d:
                self.Sigma[k] = np.cov(X[mask].T) + 1e-6 * np.eye(d)
            else:
                self.Sigma[k] = np.eye(d)
        self.nu = np.array([15.0, 7.0, 4.0])
        self.A = np.eye(K) * 0.95 + np.ones((K, K)) * 0.05 / K
        self.A = self.A / self.A.sum(axis=1, keepdims=True)
        self.pi = np.ones(K) / K

    def _mvt_logpdf(self, x, mu, Sigma, nu):
        d = len(mu)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        diff = x - mu
        Sigma_inv = np.linalg.inv(Sigma)
        mahal = np.sum(diff @ Sigma_inv * diff, axis=1)
        sign, logdet = np.linalg.slogdet(Sigma)
        logpdf = (
            gammaln((nu + d) / 2) - gammaln(nu / 2)
            - 0.5 * d * np.log(nu * np.pi)
            - 0.5 * logdet
            - 0.5 * (nu + d) * np.log(1 + mahal / nu)
        )
        return logpdf

    def _compute_emission_probs(self, X):
        T, d = X.shape
        K = self.n_regimes
        log_B = np.zeros((T, K))
        for k in range(K):
            log_B[:, k] = self._mvt_logpdf(X, self.mu[k], self.Sigma[k], self.nu[k])
        return log_B

    def _forward(self, log_B):
        T, K = log_B.shape
        log_alpha = np.zeros((T, K))
        log_alpha[0] = np.log(self.pi + 1e-300) + log_B[0]
        log_A = np.log(self.A + 1e-300)
        for t in range(1, T):
            for k in range(K):
                log_alpha[t, k] = (
                    np.logaddexp.reduce(log_alpha[t-1] + log_A[:, k])
                    + log_B[t, k]
                )
        return log_alpha

    def _backward(self, log_B):
        T, K = log_B.shape
        log_beta = np.zeros((T, K))
        log_beta[-1] = 0
        log_A = np.log(self.A + 1e-300)
        for t in range(T - 2, -1, -1):
            for k in range(K):
                log_beta[t, k] = np.logaddexp.reduce(
                    log_A[k, :] + log_B[t+1, :] + log_beta[t+1, :]
                )
        return log_beta

    def _e_step(self, X):
        T, d = X.shape
        K = self.n_regimes
        log_B = self._compute_emission_probs(X)
        log_alpha = self._forward(log_B)
        log_beta = self._backward(log_B)
        log_likelihood = np.logaddexp.reduce(log_alpha[-1])

        log_gamma = log_alpha + log_beta
        log_gamma = log_gamma - np.logaddexp.reduce(log_gamma, axis=1, keepdims=True)
        self.gamma = np.exp(log_gamma)

        log_alpha_norm = log_alpha - np.logaddexp.reduce(log_alpha, axis=1, keepdims=True)
        self.alpha = np.exp(log_alpha_norm)

        log_A = np.log(self.A + 1e-300)
        self.xi = np.zeros((T - 1, K, K))
        for t in range(T - 1):
            for j in range(K):
                for k in range(K):
                    self.xi[t, j, k] = np.exp(
                        log_alpha[t, j] + log_A[j, k] + log_B[t+1, k] + log_beta[t+1, k]
                        - log_likelihood
                    )

        self.u = np.zeros((T, K))
        for k in range(K):
            diff = X - self.mu[k]
            Sigma_inv = np.linalg.inv(self.Sigma[k])
            mahal = np.sum(diff @ Sigma_inv * diff, axis=1)
            self.u[:, k] = (self.nu[k] + d) / (self.nu[k] + mahal)

        return log_likelihood

    def _m_step(self, X):
        T, d = X.shape
        K = self.n_regimes
        self.pi = self.gamma[0] / self.gamma[0].sum()
        for j in range(K):
            for k in range(K):
                self.A[j, k] = self.xi[:, j, k].sum() / self.gamma[:-1, j].sum()
        self.A = self.A / self.A.sum(axis=1, keepdims=True)
        for k in range(K):
            weights = self.gamma[:, k] * self.u[:, k]
            self.mu[k] = (weights[:, None] * X).sum(axis=0) / weights.sum()
        for k in range(K):
            diff = X - self.mu[k]
            weights = self.gamma[:, k] * self.u[:, k]
            weighted_outer = np.zeros((d, d))
            for t in range(T):
                weighted_outer += weights[t] * np.outer(diff[t], diff[t])
            self.Sigma[k] = weighted_outer / self.gamma[:, k].sum()
            self.Sigma[k] += 1e-6 * np.eye(d)
        for k in range(K):
            self._update_nu(X, k)
        self._enforce_ordering()

    def _update_nu(self, X, k):
        T, d = X.shape
        def neg_expected_ll(nu):
            if nu <= 2:
                return 1e10
            diff = X - self.mu[k]
            Sigma_inv = np.linalg.inv(self.Sigma[k])
            mahal = np.sum(diff @ Sigma_inv * diff, axis=1)
            term1 = gammaln((nu + d) / 2) - gammaln(nu / 2)
            term2 = -0.5 * d * np.log(nu)
            term3 = -0.5 * (nu + d) * np.log(1 + mahal / nu)
            ll = self.gamma[:, k] * (term1 + term2 + term3)
            return -ll.sum()
        result = minimize_scalar(neg_expected_ll, bounds=(2.1, 50), method='bounded')
        self.nu[k] = result.x

    def _enforce_ordering(self):
        norms = np.linalg.norm(self.mu, axis=1)
        order = np.argsort(norms)
        if not np.array_equal(order, np.arange(self.n_regimes)):
            self.mu = self.mu[order]
            self.Sigma = self.Sigma[order]
            self.nu = self.nu[order]
            self.A = self.A[order][:, order]
            self.pi = self.pi[order]
            self.gamma = self.gamma[:, order]
            if self.alpha is not None:
                self.alpha = self.alpha[:, order]
            if self.xi is not None:
                self.xi = self.xi[:, order, :][:, :, order]

    def fit(self, X):
        X = np.asarray(X)
        self._init_params(X)
        prev_ll = -np.inf
        for iteration in range(self.n_iter):
            log_likelihood = self._e_step(X)
            self._m_step(X)
            if abs(log_likelihood - prev_ll) < self.tol:
                print(f"  HMM converged at iteration {iteration + 1}")
                break
            prev_ll = log_likelihood
        self.log_likelihood_ = log_likelihood
        return self

    def predict(self, X, use_filtered=False):
        X = np.asarray(X)
        self._e_step(X)
        if use_filtered:
            return np.argmax(self.alpha, axis=1)
        return np.argmax(self.gamma, axis=1)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_ff5_daily():
    """Download Fama-French 5 factors daily from Kenneth French's website."""
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
    print(f"Downloading FF5 daily from: {url}")

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
    df = df[(df.index >= '1990-01-01') & (df.index <= '2024-12-31')]
    print(f"  Loaded {len(df)} daily observations ({df.index[0].date()} to {df.index[-1].date()})")
    return df


# =============================================================================
# LAG MATRIX CONSTRUCTION
# =============================================================================

def build_lag_matrix(smb, hml, max_lag=9):
    """
    Build feature matrices for Granger causality testing.
    
    Returns:
        y: target (SMB at time t)
        X_restricted: lagged SMB only (lags 1..max_lag)
        X_unrestricted: lagged SMB + lagged HML (lags 1..max_lag each)
    """
    n = len(smb)
    y = smb[max_lag:]
    
    X_r_cols = []
    for lag in range(1, max_lag + 1):
        X_r_cols.append(smb[max_lag - lag: n - lag])
    X_r = np.column_stack(X_r_cols)
    
    X_u_cols = list(X_r_cols)
    for lag in range(1, max_lag + 1):
        X_u_cols.append(hml[max_lag - lag: n - lag])
    X_u = np.column_stack(X_u_cols)
    
    return y, X_r, X_u


# =============================================================================
# LINEAR GRANGER TEST (OLS F-test)
# =============================================================================

def linear_granger_test(y, X_r, X_u, max_lag=9):
    """Standard linear Granger causality via OLS F-test."""
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
    
    tss = np.sum((y - y.mean()) ** 2)
    r2_r = 1 - rss_r / tss
    r2_u = 1 - rss_u / tss
    delta_r2 = r2_u - r2_r
    
    mse_r = rss_r / n
    mse_u = rss_u / n
    mse_improvement = (mse_r - mse_u) / mse_r * 100 if mse_r > 0 else 0.0
    
    return {
        'f_stat': float(f_stat),
        'p_value': float(p_value),
        'mse_restricted': float(mse_r),
        'mse_unrestricted': float(mse_u),
        'mse_improvement_pct': float(mse_improvement),
        'delta_r2': float(delta_r2),
        'r2_restricted': float(r2_r),
        'r2_unrestricted': float(r2_u),
        'n_obs': int(n)
    }


# =============================================================================
# NONLINEAR GRANGER: CV-BASED MSE COMPARISON + PERMUTATION TEST
# =============================================================================

def nonlinear_granger_cv(y, X_r, X_u, model_class, model_params, n_folds=5,
                          n_permutations=200, random_state=42):
    """
    Nonlinear Granger causality via cross-validated MSE comparison.
    
    Compares restricted model (only SMB lags) vs unrestricted (SMB + HML lags).
    Uses permutation test: shuffle HML lag columns to compute null distribution.
    """
    n = len(y)
    max_lag = X_r.shape[1]
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    # Pre-compute fold indices to reuse
    fold_indices = list(kf.split(X_r))
    
    def compute_cv_mse(X_data, model_class, model_params, random_state):
        """Compute cross-validated MSE for given features."""
        mse_folds = []
        for train_idx, test_idx in fold_indices:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_data[train_idx])
            X_test = scaler.transform(X_data[test_idx])
            y_train = y[train_idx]
            y_test = y[test_idx]
            
            model = model_class(random_state=random_state, **model_params)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            mse_folds.append(mean_squared_error(y_test, pred))
        return np.mean(mse_folds), np.std(mse_folds)
    
    # Restricted model
    print(f"        Computing restricted CV MSE...", flush=True)
    mse_r_mean, mse_r_std = compute_cv_mse(X_r, model_class, model_params, random_state)
    
    # Unrestricted model
    print(f"        Computing unrestricted CV MSE...", flush=True)
    mse_u_mean, mse_u_std = compute_cv_mse(X_u, model_class, model_params, random_state)
    
    mse_improvement = (mse_r_mean - mse_u_mean) / mse_r_mean * 100 if mse_r_mean > 0 else 0.0
    
    # Permutation test
    print(f"        Running {n_permutations} permutations...", flush=True)
    rng = np.random.RandomState(random_state)
    null_improvements = []
    
    hml_col_start = max_lag
    
    for perm_i in range(n_permutations):
        if (perm_i + 1) % 50 == 0:
            print(f"          Permutation {perm_i + 1}/{n_permutations}...", flush=True)
        
        X_u_perm = X_u.copy()
        for col in range(hml_col_start, X_u.shape[1]):
            X_u_perm[:, col] = rng.permutation(X_u_perm[:, col])
        
        perm_mse, _ = compute_cv_mse(X_u_perm, model_class, model_params, random_state)
        perm_imp = (mse_r_mean - perm_mse) / mse_r_mean * 100 if mse_r_mean > 0 else 0.0
        null_improvements.append(perm_imp)
    
    null_improvements = np.array(null_improvements)
    p_value = (np.sum(null_improvements >= mse_improvement) + 1) / (n_permutations + 1)
    
    # Feature importance via permutation importance on full model
    print(f"        Computing feature importance...", flush=True)
    scaler_full = StandardScaler()
    X_u_scaled = scaler_full.fit_transform(X_u)
    model_full = model_class(random_state=random_state, **model_params)
    model_full.fit(X_u_scaled, y)
    
    perm_imp_result = permutation_importance(
        model_full, X_u_scaled, y, n_repeats=20, random_state=random_state,
        scoring='neg_mean_squared_error'
    )
    
    smb_importance = np.mean(perm_imp_result.importances_mean[:max_lag])
    hml_importance = np.mean(perm_imp_result.importances_mean[max_lag:])
    
    # Per-lag importance for detailed reporting
    per_lag_importance = {}
    for lag in range(1, max_lag + 1):
        per_lag_importance[f'SMB_lag{lag}'] = float(perm_imp_result.importances_mean[lag - 1])
        per_lag_importance[f'HML_lag{lag}'] = float(perm_imp_result.importances_mean[max_lag + lag - 1])
    
    return {
        'mse_restricted': float(mse_r_mean),
        'mse_unrestricted': float(mse_u_mean),
        'mse_improvement_pct': float(mse_improvement),
        'mse_r_std': float(mse_r_std),
        'mse_u_std': float(mse_u_std),
        'permutation_p_value': float(p_value),
        'null_improvement_mean': float(np.mean(null_improvements)),
        'null_improvement_std': float(np.std(null_improvements)),
        'n_permutations': n_permutations,
        'avg_smb_lag_importance': float(smb_importance),
        'avg_hml_lag_importance': float(hml_importance),
        'hml_importance_ratio': float(hml_importance / smb_importance) if smb_importance > 0 else 0.0,
        'per_lag_importance': per_lag_importance,
        'n_obs': int(n),
        'n_folds': n_folds
    }


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():
    print("=" * 80)
    print("NEURAL GRANGER CAUSALITY: HML -> SMB BY REGIME")
    print("Tank et al. (2022) style comparison: Linear vs MLP vs Random Forest")
    print("=" * 80)
    print(flush=True)
    
    # ---- Load data ----
    df = load_ff5_daily()
    
    # ---- Fit HMM ----
    print("\nFitting 3-regime Student-t HMM on [Mkt-RF, SMB, HML, RMW, CMA]...")
    factor_cols = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
    X_hmm = df[factor_cols].values
    
    hmm = StudentTHMM(n_regimes=3, n_iter=100, tol=1e-4, random_state=28)
    hmm.fit(X_hmm)
    
    regimes = hmm.predict(X_hmm)
    df['regime'] = regimes
    
    regime_names = {0: 'Calm', 1: 'Transition', 2: 'Crisis'}
    
    print("\nRegime summary:")
    for k in range(3):
        mask = regimes == k
        n_k = mask.sum()
        vol = df.loc[mask, 'Mkt-RF'].std()
        print(f"  Regime {k} ({regime_names[k]}): {n_k} days ({n_k/len(df)*100:.1f}%), "
              f"Mkt vol={vol:.3f}, nu={hmm.nu[k]:.1f}")
    sys.stdout.flush()
    
    # ---- Setup ----
    max_lag = 9
    n_folds = 5
    
    smb = df['SMB'].values
    hml = df['HML'].values
    
    # MLP params - moderate capacity
    mlp_params = {
        'hidden_layer_sizes': (64, 32),
        'activation': 'relu',
        'max_iter': 500,
        'early_stopping': True,
        'validation_fraction': 0.15,
        'n_iter_no_change': 20,
        'learning_rate': 'adaptive',
        'learning_rate_init': 0.001,
        'alpha': 0.01,
    }
    
    # RF params - fast but effective
    rf_params = {
        'n_estimators': 100,
        'max_depth': 6,
        'min_samples_leaf': 10,
        'max_features': 'sqrt',
        'n_jobs': -1,
    }
    
    results = {
        'metadata': {
            'description': 'Neural Granger Causality: HML -> SMB by regime',
            'method': 'Tank et al. (2022) style comparison',
            'max_lag': max_lag,
            'n_folds': n_folds,
            'date_range': f"{df.index[0].date()} to {df.index[-1].date()}",
            'n_total_obs': len(df),
            'mlp_architecture': str(mlp_params['hidden_layer_sizes']),
            'rf_n_estimators': rf_params['n_estimators'],
            'timestamp': datetime.now().isoformat()
        },
        'regimes': {}
    }
    
    # Define analysis sets: regimes first (smaller, faster), full sample last
    analysis_sets = []
    for k in range(3):
        analysis_sets.append((regime_names[k], regimes == k, 200))
    analysis_sets.append(('Full Sample', np.ones(len(df), dtype=bool), 100))
    
    for name, mask, n_perm in analysis_sets:
        n_regime = mask.sum()
        print(f"\n{'=' * 70}")
        print(f"  {name.upper()} (n={n_regime})")
        print(f"{'=' * 70}")
        sys.stdout.flush()
        
        smb_regime = smb[mask]
        hml_regime = hml[mask]
        
        if n_regime < max_lag + 50:
            print(f"  SKIP: too few observations ({n_regime})")
            results['regimes'][name] = {'skipped': True, 'n_obs': int(n_regime)}
            continue
        
        y, X_r, X_u = build_lag_matrix(smb_regime, hml_regime, max_lag)
        
        # 1. Linear Granger
        print(f"\n  [1] Linear Granger (OLS F-test)...")
        lin_result = linear_granger_test(y, X_r, X_u, max_lag)
        sig_str = "***" if lin_result['p_value'] < 0.001 else "**" if lin_result['p_value'] < 0.01 else "*" if lin_result['p_value'] < 0.05 else "ns"
        print(f"      F={lin_result['f_stat']:.3f}, p={lin_result['p_value']:.6f} {sig_str}")
        print(f"      MSE improvement: {lin_result['mse_improvement_pct']:.4f}%")
        print(f"      Delta-R2: {lin_result['delta_r2']:.6f}")
        sys.stdout.flush()
        
        # 2. MLP Granger
        print(f"\n  [2] MLP Granger (5-fold CV + {n_perm} permutations)...")
        sys.stdout.flush()
        mlp_result = nonlinear_granger_cv(
            y, X_r, X_u, MLPRegressor, mlp_params,
            n_folds=n_folds, n_permutations=n_perm, random_state=42
        )
        sig_str = "***" if mlp_result['permutation_p_value'] < 0.001 else "**" if mlp_result['permutation_p_value'] < 0.01 else "*" if mlp_result['permutation_p_value'] < 0.05 else "ns"
        print(f"      MSE restricted:   {mlp_result['mse_restricted']:.6f}")
        print(f"      MSE unrestricted: {mlp_result['mse_unrestricted']:.6f}")
        print(f"      MSE improvement:  {mlp_result['mse_improvement_pct']:.4f}%")
        print(f"      Permutation p:    {mlp_result['permutation_p_value']:.4f} {sig_str}")
        print(f"      HML importance ratio: {mlp_result['hml_importance_ratio']:.3f}")
        sys.stdout.flush()
        
        # 3. RF Granger
        print(f"\n  [3] Random Forest Granger (5-fold CV + {n_perm} permutations)...")
        sys.stdout.flush()
        rf_result = nonlinear_granger_cv(
            y, X_r, X_u, RandomForestRegressor, rf_params,
            n_folds=n_folds, n_permutations=n_perm, random_state=42
        )
        sig_str = "***" if rf_result['permutation_p_value'] < 0.001 else "**" if rf_result['permutation_p_value'] < 0.01 else "*" if rf_result['permutation_p_value'] < 0.05 else "ns"
        print(f"      MSE restricted:   {rf_result['mse_restricted']:.6f}")
        print(f"      MSE unrestricted: {rf_result['mse_unrestricted']:.6f}")
        print(f"      MSE improvement:  {rf_result['mse_improvement_pct']:.4f}%")
        print(f"      Permutation p:    {rf_result['permutation_p_value']:.4f} {sig_str}")
        print(f"      HML importance ratio: {rf_result['hml_importance_ratio']:.3f}")
        sys.stdout.flush()
        
        # Store
        regime_result = {
            'n_obs': int(n_regime),
            'n_effective': int(len(y)),
            'n_permutations': n_perm,
            'linear_granger': lin_result,
            'mlp_granger': mlp_result,
            'rf_granger': rf_result
        }
        results['regimes'][name] = regime_result
        
        # Save incrementally to prevent data loss
        output_path = '/Users/i767700/Github/ai-in-finance/papers/causal_regimes/results/neural_granger_results.json'
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    # ---- Summary comparison ----
    print("\n" + "=" * 80)
    print("SUMMARY: MSE IMPROVEMENT (%) — HML -> SMB")
    print("=" * 80)
    print(f"\n{'Regime':<15} {'N':>6} | {'Linear':>10} {'MLP':>10} {'RF':>10} | {'Lin p':>10} {'MLP p':>10} {'RF p':>10}")
    print("-" * 95)
    
    for name in ['Calm', 'Transition', 'Crisis', 'Full Sample']:
        r = results['regimes'].get(name, {})
        if r.get('skipped') or not r:
            continue
        
        lin_imp = r['linear_granger']['mse_improvement_pct']
        mlp_imp = r['mlp_granger']['mse_improvement_pct']
        rf_imp = r['rf_granger']['mse_improvement_pct']
        lin_p = r['linear_granger']['p_value']
        mlp_p = r['mlp_granger']['permutation_p_value']
        rf_p = r['rf_granger']['permutation_p_value']
        
        print(f"{name:<15} {r['n_obs']:>6} | {lin_imp:>9.4f}% {mlp_imp:>9.4f}% {rf_imp:>9.4f}% | "
              f"{lin_p:>10.6f} {mlp_p:>10.4f} {rf_p:>10.4f}")
    
    # ---- Key findings ----
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    
    crisis = results['regimes'].get('Crisis', {})
    calm = results['regimes'].get('Calm', {})
    transition = results['regimes'].get('Transition', {})
    
    if crisis and not crisis.get('skipped'):
        for method, method_key, p_key in [
            ('Linear', 'linear_granger', 'p_value'),
            ('MLP', 'mlp_granger', 'permutation_p_value'),
            ('RF', 'rf_granger', 'permutation_p_value')
        ]:
            crisis_imp = crisis[method_key]['mse_improvement_pct']
            crisis_p = crisis[method_key][p_key]
            
            print(f"\n  {method}:")
            print(f"    Crisis MSE improvement: {crisis_imp:.4f}% (p={crisis_p:.4f})")
            
            if calm and not calm.get('skipped'):
                calm_imp = calm[method_key]['mse_improvement_pct']
                ratio = crisis_imp / calm_imp if calm_imp != 0 else float('inf')
                print(f"    Calm MSE improvement:   {calm_imp:.4f}%")
                print(f"    Crisis/Calm ratio:       {ratio:.2f}x")
            
            if method in ['MLP', 'RF']:
                lin_crisis = crisis['linear_granger']['mse_improvement_pct']
                if lin_crisis > 0:
                    nl_ratio = crisis_imp / lin_crisis
                    print(f"    Nonlinear/Linear ratio (Crisis): {nl_ratio:.2f}x")
                    if nl_ratio > 1.2:
                        print(f"    --> NONLINEAR EFFECTS DETECTED: {method} finds {(nl_ratio-1)*100:.0f}% stronger effect")
                    elif nl_ratio > 0.8:
                        print(f"    --> CONSISTENT: {method} confirms linear finding (robust)")
                    else:
                        print(f"    --> LINEAR DOMINATES: nonlinear model finds weaker effect")
    
    # ---- Nonlinearity test: compare MLP/RF improvement across regimes ----
    print("\n" + "-" * 80)
    print("NONLINEARITY DIAGNOSTIC")
    print("-" * 80)
    
    for name in ['Calm', 'Transition', 'Crisis']:
        r = results['regimes'].get(name, {})
        if r.get('skipped') or not r:
            continue
        lin = r['linear_granger']['mse_improvement_pct']
        mlp = r['mlp_granger']['mse_improvement_pct']
        rf = r['rf_granger']['mse_improvement_pct']
        best_nl = max(mlp, rf)
        best_method = 'MLP' if mlp >= rf else 'RF'
        
        if lin > 0 and best_nl > 0:
            nl_advantage = best_nl / lin
        elif best_nl > 0 and lin <= 0:
            nl_advantage = float('inf')
        else:
            nl_advantage = 0.0
        
        print(f"\n  {name}:")
        print(f"    Linear: {lin:.4f}%, MLP: {mlp:.4f}%, RF: {rf:.4f}%")
        if nl_advantage != float('inf'):
            print(f"    Best nonlinear ({best_method}): {nl_advantage:.2f}x linear")
        else:
            print(f"    Best nonlinear ({best_method}): finds effect where linear does not")
    
    print(f"\nResults saved to: {output_path}")
    sys.stdout.flush()
    
    return results


if __name__ == '__main__':
    main()
