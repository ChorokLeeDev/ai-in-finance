"""
International Replication of Regime-Conditional Granger Analysis
==================================================================

This script replicates the HML→SMB Granger predictability analysis
across international Fama-French factor data.

Regions: Developed ex-US, Europe, Japan, Asia Pacific ex Japan
Protocol:
  1. Load daily factor data (1990-2024)
  2. Fit Student-t HMM (K=3) on training period (1990-2012)
  3. Classify 2013-2024 with frozen parameters
  4. Run per-regime Granger test (HML→SMB, lag=1)
  5. Run Quandt-Andrews sup-F structural break test
  6. Run quantile Granger (tau=0.05,0.25,0.50,0.75,0.95) for SMB→HML
"""

import numpy as np
import pandas as pd
import json
import os
import sys
import urllib.request
import zipfile
import io
from datetime import datetime
from scipy import stats
from scipy.special import gammaln
from scipy.optimize import minimize_scalar, minimize
from scipy.cluster.vq import kmeans2
from scipy.stats import f as f_dist, chi2
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

RESULTS_DIR = '/sessions/sweet-dazzling-heisenberg/mnt/causal_regimes/results'
REGIME_NAMES = ['Normal', 'Elevated', 'Crisis']

# =============================================================================
# DATA LOADING
# =============================================================================

def download_international_data(region_name, url):
    """Download international Fama-French factors from Kenneth French's library."""
    print(f"\n[{region_name}] Downloading from {url}...")
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            # Find CSV file (some regions have different naming conventions)
            csv_files = [f for f in z.namelist() if f.endswith('.csv')]
            if not csv_files:
                raise ValueError(f"No CSV files found in archive")
            csv_name = csv_files[0]
            print(f"  Found: {csv_name}")
            with z.open(csv_name) as f:
                df = pd.read_csv(f, skiprows=3)

        # Standardize column names
        df.columns = df.columns.str.strip()
        df = df.rename(columns={df.columns[0]: 'Date'})

        # Parse dates (try multiple formats)
        df = df[df['Date'].astype(str).str.match(r'^\d{8}$|^\d{4}-\d{2}-\d{2}$')]
        if len(df) == 0:
            raise ValueError(f"No valid dates found after filtering")

        try:
            df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
        except:
            df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

        # Convert numeric columns
        for col in df.columns:
            if col != 'Date':
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.set_index('Date').dropna()

        # Filter to 1990-2024
        df = df.loc['1990-01-01':'2024-12-31']

        print(f"  Loaded {len(df)} trading days ({df.index[0].date()} to {df.index[-1].date()})")
        return df
    except Exception as e:
        print(f"  ERROR: {str(e)}")
        return None


# =============================================================================
# STUDENT-T HMM (from existing pipeline)
# =============================================================================

class StudentTHMM:
    """Student-t HMM with frozen parameter prediction."""

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
        self.xi = None
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
                break
            prev_ll = log_likelihood
        self.log_likelihood_ = log_likelihood
        return self

    def predict_oos(self, X):
        """Predict on new data using frozen parameters (no refit)."""
        X = np.asarray(X)
        log_B = self._compute_emission_probs(X)
        log_alpha = self._forward(log_B)
        log_beta = self._backward(log_B)
        log_gamma = log_alpha + log_beta
        log_gamma = log_gamma - np.logaddexp.reduce(log_gamma, axis=1, keepdims=True)
        gamma = np.exp(log_gamma)
        return np.argmax(gamma, axis=1), gamma


# =============================================================================
# STATISTICAL TESTS
# =============================================================================

def extract_regime_clean_indices(regimes, regime_id, lag=1):
    """Get indices where ALL lags 1..lag fall within the same regime."""
    regime_mask = (regimes == regime_id)
    indices = np.where(regime_mask)[0]
    clean_indices = []
    for idx in indices:
        if idx >= lag:
            if all(regimes[idx - l] == regime_id for l in range(1, lag + 1)):
                clean_indices.append(idx)
    return np.array(clean_indices) if clean_indices else np.array([], dtype=int)


def granger_ftest(y_curr, y_lagged, x_lagged):
    """Standard F-test for Granger causality (x -> y)."""
    n = len(y_curr)
    lag = y_lagged.shape[1]
    X_r = np.column_stack([np.ones(n), y_lagged])
    X_u = np.column_stack([np.ones(n), y_lagged, x_lagged])

    try:
        beta_r = np.linalg.lstsq(X_r, y_curr, rcond=None)[0]
        beta_u = np.linalg.lstsq(X_u, y_curr, rcond=None)[0]
        rss_r = np.sum((y_curr - X_r @ beta_r) ** 2)
        rss_u = np.sum((y_curr - X_u @ beta_u) ** 2)
        df1 = lag
        df2 = n - 2 * lag - 1
        if df2 <= 0 or rss_u <= 0:
            return np.nan, np.nan, np.nan, np.nan
        f_stat = ((rss_r - rss_u) / df1) / (rss_u / df2)
        p_value = 1 - f_dist.cdf(f_stat, df1, df2)
        tss = np.sum((y_curr - y_curr.mean()) ** 2)
        r2_r = 1 - rss_r / tss
        r2_u = 1 - rss_u / tss
        delta_r2 = r2_u - r2_r
        return float(f_stat), float(p_value), float(delta_r2), float(r2_u)
    except Exception:
        return np.nan, np.nan, np.nan, np.nan


def granger_hac_wald(y_curr, y_lagged, x_lagged, lag):
    """HAC (Newey-West) robust Wald test for Granger causality."""
    n = len(y_curr)
    p = y_lagged.shape[1]
    X_u = np.column_stack([np.ones(n), y_lagged, x_lagged])
    try:
        model = sm.OLS(y_curr, X_u)
        result = model.fit(cov_type='HAC', cov_kwds={'maxlags': lag})
        R = np.zeros((p, X_u.shape[1]))
        for i in range(p):
            R[i, 1 + p + i] = 1.0
        beta = result.params
        V = result.cov_params()
        Rb = R @ beta
        RVR = R @ V @ R.T
        wald_stat = float(Rb @ np.linalg.inv(RVR) @ Rb)
        p_value = float(1 - chi2.cdf(wald_stat, p))
        return wald_stat, p_value
    except Exception:
        return np.nan, np.nan


def quandt_andrews_supf(y, x, regime_mask):
    """Quandt-Andrews sup-F test for structural break in strongest regime."""
    indices = np.where(regime_mask)[0]
    if len(indices) < 100:
        return None

    # Focus on regime dates
    regime_x = x[indices]
    regime_y = y[indices]

    # Grid search for break point (exclude first/last 15%)
    n = len(regime_x)
    start_idx = int(0.15 * n)
    end_idx = int(0.85 * n)

    sup_f = 0
    best_break = None

    for break_point in range(start_idx, end_idx):
        # Before break
        y1, x1 = regime_y[:break_point], regime_x[:break_point]
        # After break
        y2, x2 = regime_y[break_point:], regime_x[break_point:]

        if len(y1) < 5 or len(y2) < 5:
            continue

        # Fit both periods
        try:
            X1 = np.column_stack([np.ones(len(y1)), x1])
            X2 = np.column_stack([np.ones(len(y2)), x2])

            beta1 = np.linalg.lstsq(X1, y1, rcond=None)[0]
            beta2 = np.linalg.lstsq(X2, y2, rcond=None)[0]

            rss1 = np.sum((y1 - X1 @ beta1) ** 2)
            rss2 = np.sum((y2 - X2 @ beta2) ** 2)
            rss_total = np.sum((regime_y - np.column_stack([np.ones(n), regime_x]) @
                               np.linalg.lstsq(np.column_stack([np.ones(n), regime_x]), regime_y, rcond=None)[0]) ** 2)

            df1 = 2  # two parameters
            df2 = n - 4  # total - 2*params

            if df2 > 0 and (rss1 + rss2) < rss_total:
                f_stat = ((rss_total - (rss1 + rss2)) / df1) / ((rss1 + rss2) / df2)
                if f_stat > sup_f:
                    sup_f = f_stat
                    best_break = indices[break_point]
        except Exception:
            continue

    if best_break is None:
        return None

    return {
        'sup_f_stat': float(sup_f),
        'break_point_index': int(best_break),
        'break_point_date': str(x.index[best_break]) if hasattr(x, 'index') else 'unknown',
        'critical_value_10pct': 9.01,  # Approximately for 2 breaks, 10% level
    }


def quantile_granger(x, y, tau_values=[0.05, 0.25, 0.50, 0.75, 0.95], lag=1):
    """Quantile Granger causality (x -> y) at multiple quantiles."""
    n = len(x)
    results = {}

    for tau in tau_values:
        try:
            y_curr = y[lag:]
            y_lagged = np.column_stack([y[lag-i-1:-i-1] for i in range(lag)])
            x_lagged = np.column_stack([x[lag-i-1:-i-1] for i in range(lag)])

            # Quantile regression via statsmodels
            X_u = np.column_stack([np.ones(len(y_curr)), y_lagged, x_lagged])
            X_r = np.column_stack([np.ones(len(y_curr)), y_lagged])

            # Simple quantile regression (iterative reweighted least squares)
            from scipy.optimize import minimize

            def quantile_loss(beta, X, y, tau):
                residuals = y - X @ beta
                return np.sum(np.where(residuals > 0, tau * residuals, (tau - 1) * residuals))

            # Fit unrestricted
            beta_init = np.linalg.lstsq(X_u, y[lag:], rcond=None)[0]
            res_u = minimize(lambda b: quantile_loss(b, X_u, y[lag:], tau), beta_init, method='BFGS')
            loss_u = res_u.fun if res_u.success else np.nan

            # Fit restricted
            beta_init_r = np.linalg.lstsq(X_r, y[lag:], rcond=None)[0]
            res_r = minimize(lambda b: quantile_loss(b, X_r, y[lag:], tau), beta_init_r, method='BFGS')
            loss_r = res_r.fun if res_r.success else np.nan

            # Test statistic (simplified)
            test_stat = 2 * (loss_r - loss_u) if not (np.isnan(loss_r) or np.isnan(loss_u)) else np.nan

            results[f'tau_{tau}'] = {
                'tau': tau,
                'test_stat': float(test_stat) if not np.isnan(test_stat) else None,
                'n_obs': len(y_curr),
            }
        except Exception as e:
            results[f'tau_{tau}'] = {
                'tau': tau,
                'test_stat': None,
                'error': str(e),
            }

    return results


# =============================================================================
# MAIN ANALYSIS PIPELINE
# =============================================================================

def run_regional_analysis(region_name, df, output_tag=''):
    """Run complete analysis for a single region."""
    print(f"\n{'='*70}")
    print(f"ANALYZING: {region_name}")
    print(f"{'='*70}")

    # Check data
    if 'HML' not in df.columns or 'SMB' not in df.columns:
        print(f"  WARNING: Missing HML or SMB columns. Available: {df.columns.tolist()}")
        return None

    # Split training/OOS
    train_end = pd.Timestamp('2012-12-31')
    test_end = pd.Timestamp('2024-12-31')

    df_train = df.loc[df.index <= train_end]
    df_oos = df.loc[(df.index > train_end) & (df.index <= test_end)]

    print(f"  Training: {len(df_train)} days ({df_train.index[0].date()} to {df_train.index[-1].date()})")
    print(f"  OOS Test: {len(df_oos)} days ({df_oos.index[0].date()} to {df_oos.index[-1].date()})")

    # Prepare data for HMM (use multiple factors if available)
    factors_for_hmm = ['MKT', 'SMB', 'HML', 'RMW', 'CMA']
    available_factors = [f for f in factors_for_hmm if f in df_train.columns]
    if not available_factors:
        available_factors = [f for f in df_train.columns if f != 'RF'][:3]

    print(f"  Factors for HMM: {available_factors}")

    X_train = df_train[available_factors].values

    # Fit HMM on training data
    print(f"  Fitting Student-t HMM (K=3, seed=28)...")
    hmm = StudentTHMM(n_regimes=3, n_iter=100, random_state=28)
    hmm.fit(X_train)
    print(f"    Log-likelihood: {hmm.log_likelihood_:.2f}")

    # Get regime assignments
    regimes_train = np.argmax(hmm.gamma, axis=1)

    # Predict OOS regimes with frozen parameters
    X_oos = df_oos[available_factors].values
    regimes_oos, gamma_oos = hmm.predict_oos(X_oos)

    # Combine for full period regime path
    full_regimes_train = regimes_train
    full_regimes_oos = regimes_oos
    full_regimes = np.concatenate([full_regimes_train, full_regimes_oos])

    print(f"  Regime distribution (train): {np.bincount(regimes_train, minlength=3)}")
    print(f"  Regime distribution (OOS): {np.bincount(regimes_oos, minlength=3)}")

    # Extract HML and SMB
    hml = df['HML'].values
    smb = df['SMB'].values

    results = {
        'region': region_name,
        'n_train': len(df_train),
        'n_oos': len(df_oos),
        'n_total': len(df),
        'hmm_loglik': float(hmm.log_likelihood_),
        'granger_by_regime': {},
        'quandt_andrews': {},
        'quantile_granger_smb_hml': {},
    }

    # Per-regime Granger test (HML -> SMB)
    print(f"\n  Testing HML → SMB Granger causality by regime:")
    for regime_id in range(3):
        regime_name = REGIME_NAMES[regime_id]

        # In-sample
        clean_idx_train = extract_regime_clean_indices(full_regimes_train, regime_id, lag=1)
        if len(clean_idx_train) > 10:
            y_curr = smb[clean_idx_train]
            y_lagged = np.column_stack([smb[clean_idx_train - i - 1] for i in range(1)])
            x_lagged = np.column_stack([hml[clean_idx_train - i - 1] for i in range(1)])
            f_stat, f_p, delta_r2, r2_u = granger_ftest(y_curr, y_lagged, x_lagged)
            wald_stat, hac_p = granger_hac_wald(y_curr, y_lagged, x_lagged, lag=1)

            results['granger_by_regime'][f'{regime_name}_insample'] = {
                'n_obs': len(clean_idx_train),
                'f_stat': float(f_stat) if not np.isnan(f_stat) else None,
                'f_pvalue': float(f_p) if not np.isnan(f_p) else None,
                'hac_wald_stat': float(wald_stat) if not np.isnan(wald_stat) else None,
                'hac_pvalue': float(hac_p) if not np.isnan(hac_p) else None,
                'delta_r2': float(delta_r2) if not np.isnan(delta_r2) else None,
            }
            print(f"    {regime_name:10s} (in-sample, n={len(clean_idx_train):4d}): F={f_stat:7.2f} p={f_p:.4f} HAC-p={hac_p:.4f}")

        # Out-of-sample
        clean_idx_oos = extract_regime_clean_indices(full_regimes_oos, regime_id, lag=1)
        if len(clean_idx_oos) > 10:
            # Adjust indices to OOS period
            oos_indices = clean_idx_oos + len(df_train)
            y_curr = smb[oos_indices]
            y_lagged = np.column_stack([smb[oos_indices - i - 1] for i in range(1)])
            x_lagged = np.column_stack([hml[oos_indices - i - 1] for i in range(1)])
            f_stat, f_p, delta_r2, r2_u = granger_ftest(y_curr, y_lagged, x_lagged)
            wald_stat, hac_p = granger_hac_wald(y_curr, y_lagged, x_lagged, lag=1)

            results['granger_by_regime'][f'{regime_name}_oos'] = {
                'n_obs': len(oos_indices),
                'f_stat': float(f_stat) if not np.isnan(f_stat) else None,
                'f_pvalue': float(f_p) if not np.isnan(f_p) else None,
                'hac_wald_stat': float(wald_stat) if not np.isnan(wald_stat) else None,
                'hac_pvalue': float(hac_p) if not np.isnan(hac_p) else None,
                'delta_r2': float(delta_r2) if not np.isnan(delta_r2) else None,
            }
            print(f"    {regime_name:10s} (out-of-sample, n={len(oos_indices):4d}): F={f_stat:7.2f} p={f_p:.4f} HAC-p={hac_p:.4f}")

    # Quandt-Andrews sup-F test in strongest regime
    print(f"\n  Quandt-Andrews structural break test (HML → SMB):")
    regime_counts = np.bincount(full_regimes, minlength=3)
    strongest_regime = np.argmax(regime_counts)
    regime_mask = (full_regimes == strongest_regime)

    qa_result = quandt_andrews_supf(df['HML'], df['SMB'], regime_mask)
    if qa_result:
        results['quandt_andrews'] = qa_result
        print(f"    Strongest regime: {REGIME_NAMES[strongest_regime]} (n={regime_counts[strongest_regime]})")
        print(f"    Sup-F statistic: {qa_result['sup_f_stat']:.4f}")
        print(f"    Break point: {qa_result['break_point_date']}")
        print(f"    Critical value (10%): {qa_result['critical_value_10pct']:.2f}")
    else:
        print(f"    Could not compute structural break test")

    # Quantile Granger test (SMB -> HML) for tail dependence
    print(f"\n  Quantile Granger test (SMB → HML, tail dependence):")
    qg_results = quantile_granger(smb, hml, tau_values=[0.05, 0.25, 0.50, 0.75, 0.95], lag=1)
    results['quantile_granger_smb_hml'] = qg_results
    for tau_key, tau_res in qg_results.items():
        if tau_res.get('test_stat') is not None:
            print(f"    {tau_key}: test_stat={tau_res['test_stat']:.4f} (n={tau_res['n_obs']})")

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Regional URLs from Kenneth French's library
    regions = {
        'Developed_ex_US': 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Developed_ex_US_5_Factors_Daily_CSV.zip',
        'Europe': 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Europe_5_Factors_Daily_CSV.zip',
        'Japan': 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Japan_5_Factors_Daily_CSV.zip',
        'Asia_Pacific_ex_Japan': 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Asia_Pacific_ex_Japan_5_Factors_Daily_CSV.zip',
    }

    all_results = {
        'timestamp': str(datetime.now()),
        'protocol': 'International Replication of HML→SMB Granger Predictability',
        'regions': {}
    }

    for region_name, url in regions.items():
        try:
            df = download_international_data(region_name, url)
            if df is None or len(df) < 500:
                print(f"  SKIPPING {region_name}: insufficient data")
                continue

            region_results = run_regional_analysis(region_name, df)
            if region_results:
                all_results['regions'][region_name] = region_results
        except Exception as e:
            print(f"  ERROR processing {region_name}: {str(e)}")
            import traceback
            traceback.print_exc()

    # Save results
    output_path = os.path.join(RESULTS_DIR, 'international_replication.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"RESULTS SAVED: {output_path}")
    print(f"{'='*70}\n")

    # Write summary to text file
    summary_path = os.path.join(RESULTS_DIR, 'international_replication.txt')
    with open(summary_path, 'w') as f:
        f.write("INTERNATIONAL REPLICATION: HML→SMB GRANGER PREDICTABILITY\n")
        f.write("="*70 + "\n\n")

        for region_name, region_res in all_results.get('regions', {}).items():
            f.write(f"\n{region_name.upper()}\n")
            f.write("-"*70 + "\n")
            f.write(f"Sample: {region_res['n_total']} trading days (1990-2024)\n")
            f.write(f"Training: {region_res['n_train']} days | OOS: {region_res['n_oos']} days\n")
            f.write(f"HMM Log-Likelihood: {region_res['hmm_loglik']:.4f}\n\n")

            f.write("GRANGER CAUSALITY (HML → SMB):\n")
            for test_name, test_res in region_res.get('granger_by_regime', {}).items():
                if test_res and test_res.get('f_stat') is not None:
                    f.write(f"  {test_name:30s}: F={test_res['f_stat']:7.2f} (p={test_res['f_pvalue']:.4f}) ")
                    f.write(f"HAC-p={test_res['hac_pvalue']:.4f} (n={test_res['n_obs']})\n")

            if region_res.get('quandt_andrews'):
                f.write(f"\nQUANDT-ANDREWS STRUCTURAL BREAK:\n")
                qa = region_res['quandt_andrews']
                f.write(f"  Sup-F Statistic: {qa['sup_f_stat']:.4f}\n")
                f.write(f"  Break Point: {qa['break_point_date']}\n")
                f.write(f"  Critical Value (10%): {qa['critical_value_10pct']:.2f}\n")

            f.write(f"\nQUANTILE GRANGER (SMB → HML):\n")
            for tau_key, tau_res in region_res.get('quantile_granger_smb_hml', {}).items():
                if tau_res.get('test_stat') is not None:
                    f.write(f"  {tau_key}: {tau_res['test_stat']:.4f}\n")
            f.write("\n")

    print(f"Summary written to: {summary_path}\n")


if __name__ == '__main__':
    main()
