"""
Multi-Start HMM Pipeline for ICAIF 2025
========================================

Fits Student-t HMM with configurable seeds, selects fit via a configurable rule.
From the selected fit, computes ALL fit-dependent tables and analyses:
  - Table 1 (tab:regimes): Regime summary statistics
  - Table 2 (tab:detection): Gaussian vs Student-t crisis detection
  - Table 3 (tab:main): Granger causality by regime (BIC lag + HAC)
  - Table 4 (tab:r2): Incremental R² per regime
  - Table 5 (tab:warning): Early warning lead time
  - Table 6 (tab:events): Event-based validation
  - Table 7 (tab:frozen_events): Frozen OOS validation
  - All-pairs Granger heatmap data
  - Lag sensitivity data
  - Robustness analyses (filtered/smoothed, subsample, transitions, weekly, annual)

Outputs:
  results/multistart_hmm_results.json
  results/selected_fit_regimes.csv
"""

import argparse
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
from scipy.optimize import minimize_scalar
from scipy.cluster.vq import kmeans2
from scipy.stats import f as f_dist, chi2
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

RESULTS_DIR = '/Users/i767700/Github/ai-in-finance/papers/causal_regimes/results'
REGIME_NAMES = ['Normal', 'Elevated', 'Crisis']


def parse_seeds_arg(seeds_arg):
    """Parse a seed list like '0-49,42,77' into sorted unique ints."""
    if not seeds_arg:
        return None
    seeds = set()
    for token in seeds_arg.split(','):
        token = token.strip()
        if not token:
            continue
        if '-' in token:
            left, right = token.split('-', 1)
            start = int(left.strip())
            end = int(right.strip())
            if end < start:
                raise ValueError(f"Invalid seed range: {token}")
            for value in range(start, end + 1):
                seeds.add(value)
        else:
            seeds.add(int(token))
    if not seeds:
        raise ValueError("No valid seeds parsed from --seeds")
    return sorted(seeds)


def tagged_output_path(base_name, output_tag, ext):
    """Build output filename with optional tag before extension."""
    if output_tag:
        return os.path.join(RESULTS_DIR, f"{base_name}_{output_tag}.{ext}")
    return os.path.join(RESULTS_DIR, f"{base_name}.{ext}")

# =============================================================================
# DATA LOADING
# =============================================================================

def download_ff_data():
    """Download Fama-French 5 factors + Momentum daily data."""
    print("Downloading Fama-French 5 factors (daily)...")
    url5 = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip'

    with urllib.request.urlopen(url5, timeout=60) as response:
        data = response.read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        csv_name = z.namelist()[0]
        with z.open(csv_name) as f:
            df5 = pd.read_csv(f, skiprows=3)

    df5.columns = df5.columns.str.strip()
    df5 = df5.rename(columns={df5.columns[0]: 'Date'})
    df5 = df5[df5['Date'].astype(str).str.match(r'^\d{8}$')]
    df5['Date'] = pd.to_datetime(df5['Date'], format='%Y%m%d')
    for col in ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']:
        df5[col] = pd.to_numeric(df5[col], errors='coerce')
    df5 = df5.set_index('Date').dropna()

    print("Downloading Momentum factor (daily)...")
    url_mom = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip'
    with urllib.request.urlopen(url_mom, timeout=60) as response:
        data = response.read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        csv_name = z.namelist()[0]
        with z.open(csv_name) as f:
            mom = pd.read_csv(f, skiprows=13)

    mom.columns = mom.columns.str.strip()
    mom = mom.rename(columns={mom.columns[0]: 'Date'})
    mom = mom[mom['Date'].astype(str).str.match(r'^\d{8}$')]
    mom['Date'] = pd.to_datetime(mom['Date'], format='%Y%m%d')
    mom = mom.rename(columns={mom.columns[1]: 'MOM'})
    mom['MOM'] = pd.to_numeric(mom['MOM'], errors='coerce')
    mom = mom.set_index('Date').dropna()

    df = df5.join(mom[['MOM']], how='inner')
    df = df.rename(columns={'Mkt-RF': 'MKT'})
    df = df.drop('RF', axis=1, errors='ignore')
    df = df.loc['1990-01-01':'2024-12-31']
    print(f"Loaded {len(df)} trading days ({df.index[0].date()} to {df.index[-1].date()})")
    return df


# =============================================================================
# STUDENT-T HMM
# =============================================================================

class StudentTHMM:
    """Student-t HMM with filtered/smoothed probabilities and OOS prediction."""

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

    def predict(self, X, use_filtered=False):
        X = np.asarray(X)
        self._e_step(X)
        if use_filtered:
            return np.argmax(self.alpha, axis=1)
        return np.argmax(self.gamma, axis=1)

    def predict_oos(self, X, use_filtered=False):
        """Predict on new data using frozen parameters (no refit)."""
        X = np.asarray(X)
        log_B = self._compute_emission_probs(X)
        log_alpha = self._forward(log_B)
        if use_filtered:
            log_alpha_norm = log_alpha - np.logaddexp.reduce(log_alpha, axis=1, keepdims=True)
            return np.argmax(np.exp(log_alpha_norm), axis=1), np.exp(log_alpha_norm)
        log_beta = self._backward(log_B)
        log_gamma = log_alpha + log_beta
        log_gamma = log_gamma - np.logaddexp.reduce(log_gamma, axis=1, keepdims=True)
        gamma = np.exp(log_gamma)
        return np.argmax(gamma, axis=1), gamma


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def extract_regime_clean_indices(regimes, regime_id, max_lag):
    """Get indices where ALL lags 1..max_lag fall within the same regime."""
    regime_mask = (regimes == regime_id)
    indices = np.where(regime_mask)[0]
    clean_indices = []
    for idx in indices:
        if idx >= max_lag:
            if all(regimes[idx - l] == regime_id for l in range(1, max_lag + 1)):
                clean_indices.append(idx)
    return np.array(clean_indices) if clean_indices else np.array([], dtype=int)


def select_lag_bic(y_all, x_all, clean_indices, max_lag=15):
    """Select optimal lag using BIC on unrestricted model."""
    best_bic = np.inf
    best_lag = 1
    for lag in range(1, max_lag + 1):
        usable = [idx for idx in clean_indices if idx >= lag]
        if len(usable) < 2 * lag + 10:
            continue
        usable = np.array(usable)
        y_curr = y_all[usable]
        y_lagged = np.column_stack([y_all[usable - i - 1] for i in range(lag)])
        x_lagged = np.column_stack([x_all[usable - i - 1] for i in range(lag)])
        X_u = np.column_stack([np.ones(len(y_curr)), y_lagged, x_lagged])
        n = len(y_curr)
        k = X_u.shape[1]
        try:
            beta = np.linalg.lstsq(X_u, y_curr, rcond=None)[0]
            rss = np.sum((y_curr - X_u @ beta) ** 2)
            bic = n * np.log(rss / n) + k * np.log(n)
            if bic < best_bic:
                best_bic = bic
                best_lag = lag
        except Exception:
            continue
    return best_lag


def granger_ftest(y_curr, y_lagged, x_lagged):
    """Standard F-test for Granger causality (x -> y)."""
    n = len(y_curr)
    lag = y_lagged.shape[1]
    X_r = np.column_stack([np.ones(n), y_lagged])
    X_u = np.column_stack([np.ones(n), y_lagged, x_lagged])
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


def granger_hac_wald(y_curr, y_lagged, x_lagged, lag):
    """HAC (Newey-West) robust Wald test for Granger causality."""
    n = len(y_curr)
    p = y_lagged.shape[1]
    X_u = np.column_stack([np.ones(n), y_lagged, x_lagged])
    model = sm.OLS(y_curr, X_u)
    result = model.fit(cov_type='HAC', cov_kwds={'maxlags': lag})
    n_params = X_u.shape[1]
    R = np.zeros((p, n_params))
    for i in range(p):
        R[i, 1 + p + i] = 1.0
    beta = result.params
    V = result.cov_params()
    Rb = R @ beta
    RVR = R @ V @ R.T
    try:
        wald_stat = float(Rb @ np.linalg.inv(RVR) @ Rb)
        p_value = float(1 - chi2.cdf(wald_stat, p))
    except np.linalg.LinAlgError:
        wald_stat = np.nan
        p_value = np.nan
    return wald_stat, p_value


def granger_test_manual(x, y, max_lag=15):
    """Manual Granger test returning best lag by min-p, with all details."""
    n = len(x)
    best_p = 1.0
    best_lag = 1
    best_f = 0.0
    all_results = {}
    for lag in range(1, max_lag + 1):
        if n - lag < lag * 2 + 10:
            continue
        y_curr = y[lag:]
        y_lagged = np.column_stack([y[lag-i-1:-i-1] for i in range(lag)])
        x_lagged = np.column_stack([x[lag-i-1:-i-1] for i in range(lag)])
        try:
            X_r = np.column_stack([np.ones(len(y_curr)), y_lagged])
            X_u = np.column_stack([np.ones(len(y_curr)), y_lagged, x_lagged])
            beta_r = np.linalg.lstsq(X_r, y_curr, rcond=None)[0]
            beta_u = np.linalg.lstsq(X_u, y_curr, rcond=None)[0]
            rss_r = np.sum((y_curr - X_r @ beta_r) ** 2)
            rss_u = np.sum((y_curr - X_u @ beta_u) ** 2)
            df1 = lag
            df2 = len(y_curr) - 2 * lag - 1
            if df2 > 0 and rss_u > 0:
                f_stat = ((rss_r - rss_u) / df1) / (rss_u / df2)
                p_value = 1 - f_dist.cdf(f_stat, df1, df2)
                tss = np.sum((y_curr - y_curr.mean()) ** 2)
                r2_r = 1 - rss_r / tss
                r2_u = 1 - rss_u / tss
                all_results[lag] = {
                    'p_value': float(p_value),
                    'f_stat': float(f_stat),
                    'r2_restricted': float(r2_r),
                    'r2_unrestricted': float(r2_u),
                    'delta_r2': float(r2_u - r2_r),
                    'n_obs': len(y_curr)
                }
                if p_value < best_p:
                    best_p = p_value
                    best_lag = lag
                    best_f = f_stat
        except Exception:
            continue
    return best_lag, best_p, best_f, all_results


def run_granger_at_lag(y_all, x_all, clean_indices, lag):
    """Run Granger F-test + HAC at a specific lag using clean indices."""
    usable = np.array([idx for idx in clean_indices if idx >= lag])
    if len(usable) < 2 * lag + 10:
        return None
    y_curr = y_all[usable]
    y_lagged = np.column_stack([y_all[usable - i - 1] for i in range(lag)])
    x_lagged = np.column_stack([x_all[usable - i - 1] for i in range(lag)])
    f_stat, f_p, delta_r2, r2_u = granger_ftest(y_curr, y_lagged, x_lagged)
    wald_stat, hac_p = granger_hac_wald(y_curr, y_lagged, x_lagged, lag)
    return {
        'n_obs': len(usable),
        'lag': lag,
        'f_stat': f_stat,
        'f_p_value': f_p,
        'hac_wald_stat': wald_stat,
        'hac_p_value': hac_p,
        'delta_r2': delta_r2,
        'r2_unrestricted': r2_u,
    }


# =============================================================================
# PHASE 1: MULTI-START HMM SELECTION
# =============================================================================

def relabel_regimes_by_data_norm(df, regimes_raw, factor_cols):
    """Relabel regime IDs so that ascending data-based mean norm = Normal/Elevated/Crisis.

    The HMM's _enforce_ordering uses ||mu_k|| (centroid norm), which can differ from
    the mean ||x_t|| of assigned data points. This function relabels by the latter,
    ensuring Normal = lowest-volatility regime, Crisis = highest-volatility regime.
    """
    data_norms = np.linalg.norm(df[factor_cols].values, axis=1)
    mean_norms = []
    for k in range(3):
        mask = regimes_raw == k
        if mask.sum() > 0:
            mean_norms.append(data_norms[mask].mean())
        else:
            mean_norms.append(0.0)

    # order[0] = regime with lowest data norm → becomes 0 (Normal)
    order = np.argsort(mean_norms)
    relabeled = np.zeros_like(regimes_raw)
    for new_k, old_k in enumerate(order):
        relabeled[regimes_raw == old_k] = new_k

    return relabeled, order


def relabel_hmm_params(hmm, order):
    """Reorder HMM parameters to match data-norm-based regime labeling."""
    hmm.mu = hmm.mu[order]
    hmm.Sigma = hmm.Sigma[order]
    hmm.nu = hmm.nu[order]
    hmm.A = hmm.A[order][:, order]
    hmm.pi = hmm.pi[order]
    return hmm


def run_multistart_hmm(df, seeds=None, selection_rule='ll_only'):
    """Fit Student-t HMM with multiple seeds, select best by log-likelihood.

    Uses seeds 0-49 plus seed 42 (paper's original) to ensure wide coverage.
    After selection, relabels regimes by data-based mean norm (volatility).
    """
    if seeds is None:
        seeds = sorted(set(list(range(50)) + [42]))

    print("\n" + "=" * 70)
    print(f"PHASE 1: MULTI-START HMM FITTING ({len(seeds)} seeds)")
    print("=" * 70)

    X = df.values
    factor_cols = ['MKT', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']
    fit_summaries = []

    for seed in seeds:
        print(f"  Seed {seed:>3}...", end=" ", flush=True)
        hmm = StudentTHMM(n_regimes=3, n_iter=100, tol=1e-4, random_state=seed)
        hmm.fit(X)
        regimes_raw = hmm.predict(X, use_filtered=False)

        # Relabel by data-based mean norm for consistent labeling
        regimes, order = relabel_regimes_by_data_norm(df, regimes_raw, factor_cols)
        counts = {REGIME_NAMES[k]: int((regimes == k).sum()) for k in range(3)}

        # Data-based mean norms (what matters for interpretation)
        data_norms = np.linalg.norm(df[factor_cols].values, axis=1)
        data_mean_norms = [float(data_norms[regimes == k].mean()) for k in range(3)]

        summary = {
            'seed': seed,
            'log_likelihood': float(hmm.log_likelihood_),
            'regime_counts': counts,
            'nu': [float(v) for v in hmm.nu[order]],
            'data_mean_norms': data_mean_norms,
            'centroid_norms': [float(np.linalg.norm(hmm.mu[k])) for k in range(3)],
            'transition_diag': [float(hmm.A[order[k], order[k]]) for k in range(3)],
            'relabel_order': [int(o) for o in order],
        }
        fit_summaries.append(summary)
        print(f"LL={hmm.log_likelihood_:.2f}, counts=[N={counts['Normal']},E={counts['Elevated']},C={counts['Crisis']}]")

    # Compute 2008 crisis detection for each fit (quality filter)
    mask_2008 = (df.index >= '2008-07-01') & (df.index <= '2009-06-30')
    idx_2008 = np.where(mask_2008)[0]

    for summary in fit_summaries:
        # Refit to get regime assignments for this check
        hmm_tmp = StudentTHMM(n_regimes=3, n_iter=100, tol=1e-4, random_state=summary['seed'])
        hmm_tmp.fit(X)
        reg_tmp = hmm_tmp.predict(X, use_filtered=False)
        reg_tmp, _ = relabel_regimes_by_data_norm(df, reg_tmp, factor_cols)
        crisis_2008 = float((reg_tmp[idx_2008] == 2).mean() * 100)
        summary['crisis_2008_pct'] = crisis_2008

    # Candidate fits for different selection policies
    valid_fits = [s for s in fit_summaries if s['crisis_2008_pct'] >= 50]
    best_ll_summary = max(fit_summaries, key=lambda s: s['log_likelihood'])
    if valid_fits:
        best_screened_summary = max(valid_fits, key=lambda s: s['log_likelihood'])
    else:
        best_screened_summary = max(fit_summaries, key=lambda s: s['crisis_2008_pct'])

    if selection_rule == 'll_only':
        best_summary = best_ll_summary
        primary_reason = "highest log-likelihood (leakage-safe)"
        sensitivity = {
            'screened_2008_candidate': {
                'seed': best_screened_summary['seed'],
                'log_likelihood': best_screened_summary['log_likelihood'],
                'crisis_2008_pct': best_screened_summary['crisis_2008_pct'],
                'screened_pass_count': len(valid_fits),
                'screened_total': len(fit_summaries),
            }
        }
    elif selection_rule == 'screened_2008':
        best_summary = best_screened_summary
        primary_reason = "highest log-likelihood among fits with >=50% 2008 crisis detection"
        sensitivity = {
            'll_only_candidate': {
                'seed': best_ll_summary['seed'],
                'log_likelihood': best_ll_summary['log_likelihood'],
                'crisis_2008_pct': best_ll_summary['crisis_2008_pct'],
            }
        }
    elif selection_rule == 'dual':
        # Primary remains leakage-safe; screened result is explicitly sensitivity-only.
        best_summary = best_ll_summary
        primary_reason = "dual mode: ll_only primary, screened_2008 sensitivity"
        sensitivity = {
            'screened_2008_candidate': {
                'seed': best_screened_summary['seed'],
                'log_likelihood': best_screened_summary['log_likelihood'],
                'crisis_2008_pct': best_screened_summary['crisis_2008_pct'],
                'screened_pass_count': len(valid_fits),
                'screened_total': len(fit_summaries),
            }
        }
    else:
        raise ValueError(f"Unknown selection_rule: {selection_rule}")

    print(f"\n  Selection rule: {selection_rule}")
    print(f"  Primary selection reason: {primary_reason}")
    print(f"  Screened pass count: {len(valid_fits)}/{len(fit_summaries)}")

    best_seed = best_summary['seed']
    best_ll = best_summary['log_likelihood']
    print(f"  SELECTED: seed={best_seed}, LL={best_ll:.2f}, 2008 detection={best_summary['crisis_2008_pct']:.1f}%")
    print(f"  Counts: {best_summary['regime_counts']}")
    print(f"  Data mean norms: {best_summary['data_mean_norms']}")

    # Refit with best seed and apply data-norm relabeling
    hmm_best = StudentTHMM(n_regimes=3, n_iter=100, tol=1e-4, random_state=best_seed)
    hmm_best.fit(X)
    regimes_raw = hmm_best.predict(X, use_filtered=False)
    regimes, order = relabel_regimes_by_data_norm(df, regimes_raw, factor_cols)
    hmm_best = relabel_hmm_params(hmm_best, order)

    selection_info = {
        'selection_rule': selection_rule,
        'primary_reason': primary_reason,
        'primary_seed': best_seed,
        'primary_log_likelihood': best_ll,
        'primary_crisis_2008_pct': best_summary['crisis_2008_pct'],
        'screened_pass_count': len(valid_fits),
        'screened_total': len(fit_summaries),
        'sensitivity': sensitivity,
    }
    return hmm_best, regimes, fit_summaries, best_seed, selection_info


# =============================================================================
# TABLE 1: REGIME SUMMARY STATISTICS
# =============================================================================

def compute_table1(df, hmm, regimes):
    """Compute Table 1 (tab:regimes) statistics.

    HMM params (mu, Sigma, nu, A) are already relabeled to match regimes.
    """
    print("\n  Computing Table 1 (Regime Summary)...")
    factor_cols = ['MKT', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']
    table1 = {}
    for k in range(3):
        mask = regimes == k
        n = int(mask.sum())
        prop = n / len(regimes) * 100
        # Use boolean indexing on the df
        regime_idx = np.where(mask)[0]
        regime_data = df.iloc[regime_idx]

        mean_returns = {col: float(regime_data[col].mean()) for col in factor_cols}
        # Mean factor norm (for Table 1 "Mean ||x||" column)
        norms = np.linalg.norm(regime_data[factor_cols].values, axis=1)
        mean_norm = float(norms.mean())

        table1[REGIME_NAMES[k]] = {
            'n_days': n,
            'proportion': round(prop, 1),
            'mean_returns': mean_returns,
            'mean_norm': round(mean_norm, 2),
            'nu': round(float(hmm.nu[k]), 1),
            'transition_prob': round(float(hmm.A[k, k]), 3),
        }
        print(f"    {REGIME_NAMES[k]}: {n} days ({prop:.1f}%), "
              f"mean||x||={mean_norm:.2f}, nu={hmm.nu[k]:.1f}, P(stay)={hmm.A[k,k]:.3f}")
    return table1


# =============================================================================
# TABLE 2: GAUSSIAN VS STUDENT-T DETECTION
# =============================================================================

def compute_table2_detection(df, regimes):
    """Compute Table 2 (tab:detection) crisis detection comparison."""
    print("\n  Computing Table 2 (Crisis Detection)...")

    events = [
        ('2008 Financial', '2008-07-01', '2009-06-30'),
        ('2011 EU Debt', '2011-07-01', '2011-10-31'),
        ('2020 COVID-19', '2020-02-01', '2020-06-30'),
    ]

    # Fit Gaussian HMM with multi-start (seeds 0-9, select best LL)
    print("    Fitting Gaussian HMM (multi-start, 10 seeds)...")
    try:
        from hmmlearn.hmm import GaussianHMM
        best_gauss_ll = -np.inf
        best_gauss_model = None
        X = df.values
        for gseed in range(10):
            try:
                gauss = GaussianHMM(n_components=3, covariance_type='full',
                                    n_iter=100, random_state=gseed)
                gauss.fit(X)
                if gauss.score(X) > best_gauss_ll:
                    best_gauss_ll = gauss.score(X)
                    best_gauss_model = gauss
            except Exception:
                continue
        if best_gauss_model is not None:
            gauss_regimes_raw = best_gauss_model.predict(X)
            # Label by ascending mean norm
            gauss_means = best_gauss_model.means_
            gauss_norms = np.linalg.norm(gauss_means, axis=1)
            gauss_order = np.argsort(gauss_norms)
            gauss_regimes = np.zeros_like(gauss_regimes_raw)
            for new_k, old_k in enumerate(gauss_order):
                gauss_regimes[gauss_regimes_raw == old_k] = new_k
            print(f"    Gaussian HMM fitted (best LL={best_gauss_ll:.2f})")
        else:
            gauss_regimes = None
            print("    WARNING: Gaussian HMM fitting failed")
    except ImportError:
        gauss_regimes = None
        print("    WARNING: hmmlearn not available, skipping Gaussian comparison")

    detection = {}
    for event_name, start, end in events:
        mask = (df.index >= start) & (df.index <= end)
        n_days = int(mask.sum())
        # Student-t Crisis detection
        event_regimes = regimes[np.where(mask)[0]]
        student_crisis_pct = float((event_regimes == 2).mean() * 100)

        # Gaussian Crisis detection
        if gauss_regimes is not None:
            gauss_event = gauss_regimes[np.where(mask)[0]]
            gauss_crisis_pct = float((gauss_event == 2).mean() * 100)
        else:
            gauss_crisis_pct = None

        detection[event_name] = {
            'days': n_days,
            'student_t_crisis_pct': round(student_crisis_pct, 1),
            'gaussian_crisis_pct': round(gauss_crisis_pct, 1) if gauss_crisis_pct is not None else None,
        }
        print(f"    {event_name}: Student-t={student_crisis_pct:.1f}%, "
              f"Gaussian={gauss_crisis_pct:.1f}%" if gauss_crisis_pct is not None else
              f"    {event_name}: Student-t={student_crisis_pct:.1f}%")
    return detection


# =============================================================================
# TABLE 3: GRANGER CAUSALITY (BIC + HAC)
# =============================================================================

def compute_table3_granger(df, regimes):
    """Compute Table 3 (tab:main) Granger causality by regime."""
    print("\n  Computing Table 3 (Granger Causality)...")
    hml_all = df['HML'].values
    smb_all = df['SMB'].values
    granger = {}

    for k in range(3):
        regime_name = REGIME_NAMES[k]
        clean_15 = extract_regime_clean_indices(regimes, k, max_lag=15)
        if len(clean_15) < 50:
            granger[regime_name] = {'status': 'insufficient_data'}
            continue

        # BIC lag selection for each direction
        best_lag_h2s = select_lag_bic(smb_all, hml_all, clean_15, max_lag=15)
        best_lag_s2h = select_lag_bic(hml_all, smb_all, clean_15, max_lag=15)

        # Clean indices at BIC-selected lag
        clean_h2s = extract_regime_clean_indices(regimes, k, max_lag=best_lag_h2s)
        clean_s2h = extract_regime_clean_indices(regimes, k, max_lag=best_lag_s2h)

        # Run both tests for HML->SMB
        h2s_result = run_granger_at_lag(smb_all, hml_all, clean_h2s, best_lag_h2s)
        s2h_result = run_granger_at_lag(hml_all, smb_all, clean_s2h, best_lag_s2h)

        granger[regime_name] = {
            'hml_to_smb': h2s_result,
            'smb_to_hml': s2h_result,
            'n_total': int((regimes == k).sum()),
        }

        if h2s_result:
            print(f"    {regime_name} HML->SMB: lag={best_lag_h2s}, "
                  f"F-p={h2s_result['f_p_value']:.2e}, HAC-p={h2s_result['hac_p_value']:.2e}")
        if s2h_result:
            print(f"    {regime_name} SMB->HML: lag={best_lag_s2h}, "
                  f"F-p={s2h_result['f_p_value']:.2e}, HAC-p={s2h_result['hac_p_value']:.2e}")

    return granger


# =============================================================================
# TABLE 4: INCREMENTAL R²
# =============================================================================

def compute_table4_r2(df, regimes):
    """Compute Table 4 (tab:r2) incremental R² per regime."""
    print("\n  Computing Table 4 (Incremental R²)...")
    hml_all = df['HML'].values
    smb_all = df['SMB'].values
    r2_results = {}

    for k in range(3):
        regime_name = REGIME_NAMES[k]
        clean = extract_regime_clean_indices(regimes, k, max_lag=15)
        if len(clean) < 50:
            r2_results[regime_name] = {'status': 'insufficient_data'}
            continue

        usable = np.array([idx for idx in clean if idx >= 15])
        y_curr = smb_all[usable]
        y_lagged = np.column_stack([smb_all[usable - i - 1] for i in range(15)])
        x_lagged = np.column_stack([hml_all[usable - i - 1] for i in range(15)])

        # Restricted model: SMB ~ intercept + SMB_lags
        X_r = np.column_stack([np.ones(len(y_curr)), y_lagged])
        # Unrestricted model: SMB ~ intercept + SMB_lags + HML_lags
        X_u = np.column_stack([np.ones(len(y_curr)), y_lagged, x_lagged])

        beta_r = np.linalg.lstsq(X_r, y_curr, rcond=None)[0]
        beta_u = np.linalg.lstsq(X_u, y_curr, rcond=None)[0]
        rss_r = np.sum((y_curr - X_r @ beta_r) ** 2)
        rss_u = np.sum((y_curr - X_u @ beta_u) ** 2)
        tss = np.sum((y_curr - y_curr.mean()) ** 2)
        r2_ar = 1 - rss_r / tss
        r2_u = 1 - rss_u / tss
        delta_r2 = r2_u - r2_ar

        # F-test
        df1 = 15
        df2 = len(y_curr) - 31
        f_stat = ((rss_r - rss_u) / df1) / (rss_u / df2) if df2 > 0 else np.nan
        p_value = 1 - f_dist.cdf(f_stat, df1, df2) if not np.isnan(f_stat) else np.nan

        r2_results[regime_name] = {
            'n_clean': len(usable),
            'r2_ar': round(float(r2_ar * 100), 2),
            'delta_r2': round(float(delta_r2 * 100), 2),
            'p_value': float(p_value),
            'f_stat': float(f_stat),
        }
        print(f"    {regime_name}: n={len(usable)}, R²_AR={r2_ar*100:.2f}%, "
              f"ΔR²={delta_r2*100:.2f}%, p={p_value:.2e}")

    # Reverse direction: SMB -> HML for Crisis
    clean_crisis = extract_regime_clean_indices(regimes, 2, max_lag=15)
    usable = np.array([idx for idx in clean_crisis if idx >= 15])
    if len(usable) >= 50:
        y_curr = hml_all[usable]
        y_lagged = np.column_stack([hml_all[usable - i - 1] for i in range(15)])
        x_lagged = np.column_stack([smb_all[usable - i - 1] for i in range(15)])
        X_r = np.column_stack([np.ones(len(y_curr)), y_lagged])
        X_u = np.column_stack([np.ones(len(y_curr)), y_lagged, x_lagged])
        beta_r = np.linalg.lstsq(X_r, y_curr, rcond=None)[0]
        beta_u = np.linalg.lstsq(X_u, y_curr, rcond=None)[0]
        rss_r = np.sum((y_curr - X_r @ beta_r) ** 2)
        rss_u = np.sum((y_curr - X_u @ beta_u) ** 2)
        tss = np.sum((y_curr - y_curr.mean()) ** 2)
        r2_ar = 1 - rss_r / tss
        delta_r2_rev = (1 - rss_u / tss) - r2_ar
        df1 = 15
        df2 = len(y_curr) - 31
        f_stat_rev = ((rss_r - rss_u) / df1) / (rss_u / df2)
        p_rev = 1 - f_dist.cdf(f_stat_rev, df1, df2)
        r2_results['Crisis_reverse'] = {
            'delta_r2': round(float(delta_r2_rev * 100), 2),
            'p_value': float(p_rev),
        }
        print(f"    Crisis reverse (SMB->HML): ΔR²={delta_r2_rev*100:.2f}%, p={p_rev:.3f}")

    return r2_results


# =============================================================================
# TABLE 5: EARLY WARNING LEAD TIME
# =============================================================================

def compute_table5_warning(df, regimes):
    """Compute Table 5 (tab:warning) early warning lead time."""
    print("\n  Computing Table 5 (Early Warning)...")
    # Hardcoded volatility peak dates
    vol_peaks = {
        '2008 Financial': pd.Timestamp('2008-09-15'),
        '2011 EU Debt': pd.Timestamp('2011-08-08'),
        '2020 COVID': pd.Timestamp('2020-03-23'),
    }
    search_windows = {
        '2008 Financial': ('2008-01-01', '2008-09-15'),
        '2011 EU Debt': ('2011-06-01', '2011-08-08'),
        '2020 COVID': ('2020-01-01', '2020-03-23'),
    }

    warning = {}
    dates = df.index

    for event_name, (search_start, search_end) in search_windows.items():
        vol_peak = vol_peaks[event_name]
        mask = (dates >= search_start) & (dates <= search_end)
        window_indices = np.where(mask)[0]
        window_regimes = regimes[window_indices]

        # Find first day of 3+ consecutive Crisis assignments
        first_detection = None
        for i in range(len(window_regimes) - 2):
            if (window_regimes[i] == 2 and window_regimes[i+1] == 2 and
                    window_regimes[i+2] == 2):
                first_detection = dates[window_indices[i]]
                break

        if first_detection is not None:
            lead_time = (vol_peak - first_detection).days
            warning[event_name] = {
                'first_detection': str(first_detection.date()),
                'vol_peak': str(vol_peak.date()),
                'lead_time_days': lead_time,
            }
            print(f"    {event_name}: detected {first_detection.date()}, "
                  f"peak {vol_peak.date()}, lead={lead_time} days")
        else:
            warning[event_name] = {
                'first_detection': None,
                'vol_peak': str(vol_peak.date()),
                'lead_time_days': None,
            }
            print(f"    {event_name}: no 3+ consecutive Crisis found before peak")

    return warning


# =============================================================================
# TABLE 6: EVENT-BASED VALIDATION
# =============================================================================

def compute_table6_events(df, regimes):
    """Compute Table 6 (tab:events) event-based validation."""
    print("\n  Computing Table 6 (Event Validation)...")
    events = [
        ('2008 Financial', '2008-07-01', '2009-03-31'),
        ('2011 EU Debt', '2011-07-01', '2011-10-31'),
        ('2015 China', '2015-08-15', '2015-11-15'),
        ('2018 Vol Shock', '2018-12-01', '2019-01-15'),
        ('2020 COVID', '2020-02-15', '2020-06-15'),
        ('2022 Rate Hikes', '2022-01-01', '2022-09-30'),
    ]

    event_results = {}
    for event_name, start, end in events:
        mask = (df.index >= start) & (df.index <= end)
        n_days = int(mask.sum())
        event_regimes = regimes[np.where(mask)[0]]
        crisis_pct = float((event_regimes == 2).mean() * 100)

        hml_ev = df.loc[mask, 'HML'].values
        smb_ev = df.loc[mask, 'SMB'].values
        max_lag_ev = min(10, n_days // 3)

        if n_days > 20 and max_lag_ev >= 1:
            _, p_h2s, _, _ = granger_test_manual(hml_ev, smb_ev, max_lag=max_lag_ev)
            _, p_s2h, _, _ = granger_test_manual(smb_ev, hml_ev, max_lag=max_lag_ev)
        else:
            p_h2s, p_s2h = None, None

        # Result code
        if p_h2s is not None and p_s2h is not None:
            if p_h2s < 0.10 and p_s2h > 0.10:
                result_code = 'checkmark'
            elif p_h2s < 0.10 and p_s2h < 0.10:
                result_code = 'checkmark' if p_h2s < p_s2h else 'x'
            elif p_h2s >= 0.10 and p_s2h < 0.10:
                result_code = 'x'
            elif p_h2s >= 0.10 and p_h2s < p_s2h:
                result_code = 'dir'
            else:
                result_code = 'x'
        else:
            result_code = 'insufficient'

        event_results[event_name] = {
            'days': n_days,
            'crisis_pct': round(crisis_pct, 1),
            'hml_to_smb_p': float(p_h2s) if p_h2s is not None else None,
            'smb_to_hml_p': float(p_s2h) if p_s2h is not None else None,
            'result_code': result_code,
        }
        print(f"    {event_name}: {n_days}d, Crisis={crisis_pct:.0f}%, "
              f"HML->SMB p={p_h2s:.3f}, SMB->HML p={p_s2h:.3f}, {result_code}" if p_h2s else
              f"    {event_name}: insufficient data")

    return event_results


# =============================================================================
# ALL-PAIRS GRANGER HEATMAP DATA
# =============================================================================

def compute_all_pairs_granger(df, regimes, lag=5):
    """Compute all 30 directed-pair Granger p-values for heatmap."""
    print("\n  Computing all-pairs Granger heatmap data (lag=5)...")
    factor_cols = ['MKT', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']
    all_pairs = {}

    for k in range(3):
        regime_name = REGIME_NAMES[k]
        clean = extract_regime_clean_indices(regimes, k, max_lag=lag)
        pairs_data = {}
        n_clean = len(clean)

        for i, src in enumerate(factor_cols):
            for j, tgt in enumerate(factor_cols):
                if i == j:
                    continue
                pair_key = f"{src}->{tgt}"
                usable = np.array([idx for idx in clean if idx >= lag])
                if len(usable) < 2 * lag + 10:
                    pairs_data[pair_key] = {'p_value': 1.0, 'f_stat': 0.0, 'n_obs': 0}
                    continue

                y_all = df[tgt].values
                x_all = df[src].values
                y_curr = y_all[usable]
                y_lagged = np.column_stack([y_all[usable - ii - 1] for ii in range(lag)])
                x_lagged = np.column_stack([x_all[usable - ii - 1] for ii in range(lag)])

                f_stat, p_value, _, _ = granger_ftest(y_curr, y_lagged, x_lagged)
                pairs_data[pair_key] = {
                    'p_value': float(p_value) if not np.isnan(p_value) else 1.0,
                    'f_stat': float(f_stat) if not np.isnan(f_stat) else 0.0,
                    'n_obs': len(usable),
                }

        all_pairs[regime_name] = {
            'n_clean': n_clean,
            'pairs': pairs_data,
        }
        # Count significant at Bonferroni
        bonferroni = 0.01 / 30
        n_sig = sum(1 for v in pairs_data.values() if v['p_value'] < bonferroni)
        print(f"    {regime_name}: {n_clean} clean obs, {n_sig} pairs survive Bonferroni (0.01/30)")

    return all_pairs


# =============================================================================
# LAG SENSITIVITY DATA
# =============================================================================

def compute_lag_sensitivity(df, regimes, max_lag=15):
    """Compute HML->SMB p-values at lags 1-15 per regime with per-lag cleaning."""
    print("\n  Computing lag sensitivity data...")
    hml_all = df['HML'].values
    smb_all = df['SMB'].values
    lag_data = {}

    for k in range(3):
        regime_name = REGIME_NAMES[k]
        lag_results = {}
        for lag in range(1, max_lag + 1):
            clean = extract_regime_clean_indices(regimes, k, max_lag=lag)
            usable = np.array([idx for idx in clean if idx >= lag])
            if len(usable) < 2 * lag + 10:
                lag_results[str(lag)] = {'p_value': 1.0, 'n_obs': 0}
                continue

            y_curr = smb_all[usable]
            y_lagged = np.column_stack([smb_all[usable - i - 1] for i in range(lag)])
            x_lagged = np.column_stack([hml_all[usable - i - 1] for i in range(lag)])
            f_stat, p_value, delta_r2, _ = granger_ftest(y_curr, y_lagged, x_lagged)
            lag_results[str(lag)] = {
                'p_value': float(p_value) if not np.isnan(p_value) else 1.0,
                'f_stat': float(f_stat) if not np.isnan(f_stat) else 0.0,
                'delta_r2': float(delta_r2) if not np.isnan(delta_r2) else 0.0,
                'n_obs': len(usable),
            }

        lag_data[regime_name] = lag_results
        # Report Crisis min p
        if k == 2:
            min_p = min(v['p_value'] for v in lag_results.values())
            max_p = max(v['p_value'] for v in lag_results.values() if v['n_obs'] > 0)
            print(f"    Crisis: min-p={min_p:.2e}, max-p={max_p:.2e}")

    return lag_data


# =============================================================================
# ROBUSTNESS ANALYSES
# =============================================================================

def compute_robustness(df, hmm, regimes):
    """Compute all robustness analyses."""
    print("\n  Computing robustness analyses...")
    hml_all = df['HML'].values
    smb_all = df['SMB'].values
    robustness = {}

    # 1. Filtered vs smoothed
    print("    (a) Filtered vs smoothed...")
    factor_cols = ['MKT', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']
    filtered_regimes_raw = hmm.predict(df.values, use_filtered=True)
    # Relabel filtered regimes by data norm (same logic as smoothed)
    filtered_regimes, _ = relabel_regimes_by_data_norm(df, filtered_regimes_raw, factor_cols)
    agreement = float((regimes == filtered_regimes).mean())
    filtered_crisis_n = int((filtered_regimes == 2).sum())
    smoothed_crisis_n = int((regimes == 2).sum())

    # Granger on filtered Crisis
    clean_filtered = extract_regime_clean_indices(filtered_regimes, 2, max_lag=15)
    if len(clean_filtered) >= 50:
        best_lag_filt = select_lag_bic(smb_all, hml_all, clean_filtered, max_lag=15)
        clean_filt_lag = extract_regime_clean_indices(filtered_regimes, 2, max_lag=best_lag_filt)
        filt_result = run_granger_at_lag(smb_all, hml_all, clean_filt_lag, best_lag_filt)
        filtered_p = filt_result['f_p_value'] if filt_result else None
    else:
        filtered_p = None

    robustness['filtered_vs_smoothed'] = {
        'agreement_pct': round(agreement * 100, 1),
        'filtered_crisis_days': filtered_crisis_n,
        'smoothed_crisis_days': smoothed_crisis_n,
        'filtered_crisis_hml2smb_p': filtered_p,
    }
    print(f"      Agreement: {agreement*100:.1f}%, filtered Crisis: {filtered_crisis_n} vs smoothed: {smoothed_crisis_n}")
    if filtered_p is not None:
        print(f"      Filtered Crisis HML->SMB p={filtered_p:.3f}")

    # 2. Subsample stability
    print("    (b) Subsample stability...")
    crisis_mask = (regimes == 2)
    crisis_dates = df.index[crisis_mask]
    pre_2008 = crisis_dates < '2008-01-01'
    post_2008 = crisis_dates >= '2008-01-01'

    pre_indices = np.where(crisis_mask & (df.index < '2008-01-01'))[0]
    post_indices = np.where(crisis_mask & (df.index >= '2008-01-01'))[0]

    # Pre-2008 Crisis Granger
    pre_clean = []
    for idx in pre_indices:
        if idx >= 15 and all(regimes[idx - l] == 2 for l in range(1, 16)):
            pre_clean.append(idx)
    if len(pre_clean) >= 30:
        pre_clean = np.array(pre_clean)
        hml_pre = hml_all[pre_clean]
        smb_pre = smb_all[pre_clean]
        _, p_pre, _, _ = granger_test_manual(hml_pre, smb_pre, max_lag=15)
        n_pre = len(pre_clean)
    else:
        p_pre, n_pre = None, len(pre_clean)

    post_clean = []
    for idx in post_indices:
        if idx >= 15 and all(regimes[idx - l] == 2 for l in range(1, 16)):
            post_clean.append(idx)
    if len(post_clean) >= 30:
        post_clean = np.array(post_clean)
        hml_post = hml_all[post_clean]
        smb_post = smb_all[post_clean]
        _, p_post, _, _ = granger_test_manual(hml_post, smb_post, max_lag=15)
        n_post = len(post_clean)
    else:
        p_post, n_post = None, len(post_clean)

    robustness['subsample'] = {
        'pre_2008_n': n_pre,
        'pre_2008_p': float(p_pre) if p_pre is not None else None,
        'post_2008_n': n_post,
        'post_2008_p': float(p_post) if p_post is not None else None,
    }
    print(f"      Pre-2008: n={n_pre}, p={p_pre}")
    print(f"      Post-2008: n={n_post}, p={p_post}")

    # 3. Regime transition windows
    print("    (c) Regime transitions...")
    transitions = []
    for i in range(1, len(regimes)):
        if regimes[i-1] == 0 and regimes[i] == 2:
            transitions.append(i)

    pre_trans_ps = []
    post_trans_ps = []
    for t_idx in transitions:
        pre_start = max(0, t_idx - 30)
        post_end = min(len(df), t_idx + 30)
        if t_idx - pre_start < 20 or post_end - t_idx < 20:
            continue
        # Pre-transition 30 days
        hml_pre_t = hml_all[pre_start:t_idx]
        smb_pre_t = smb_all[pre_start:t_idx]
        if len(hml_pre_t) >= 20:
            _, p_pre_t, _, _ = granger_test_manual(hml_pre_t, smb_pre_t, max_lag=min(5, len(hml_pre_t)//4))
            pre_trans_ps.append(p_pre_t)
        # Post-transition 30 days
        hml_post_t = hml_all[t_idx:post_end]
        smb_post_t = smb_all[t_idx:post_end]
        if len(hml_post_t) >= 20:
            _, p_post_t, _, _ = granger_test_manual(hml_post_t, smb_post_t, max_lag=min(5, len(hml_post_t)//4))
            post_trans_ps.append(p_post_t)

    robustness['transitions'] = {
        'n_transitions': len(transitions),
        'pre_transition_median_p': float(np.median(pre_trans_ps)) if pre_trans_ps else None,
        'post_transition_median_p': float(np.median(post_trans_ps)) if post_trans_ps else None,
        'pre_transition_ps': [float(p) for p in pre_trans_ps],
        'post_transition_ps': [float(p) for p in post_trans_ps],
    }
    if pre_trans_ps and post_trans_ps:
        print(f"      {len(transitions)} transitions, "
              f"pre median-p={np.median(pre_trans_ps):.2f}, "
              f"post median-p={np.median(post_trans_ps):.2f}")

    # 4. Weekly aggregation
    print("    (d) Weekly aggregation...")
    crisis_daily = pd.DataFrame({
        'HML': hml_all, 'SMB': smb_all, 'regime': regimes
    }, index=df.index)
    crisis_weekly = crisis_daily[crisis_daily['regime'] == 2].resample('W').agg({
        'HML': 'sum', 'SMB': 'sum', 'regime': 'count'
    }).rename(columns={'regime': 'n_days'})
    crisis_weekly = crisis_weekly[crisis_weekly['n_days'] >= 3]  # at least 3 crisis days in week

    if len(crisis_weekly) >= 30:
        hml_w = crisis_weekly['HML'].values
        smb_w = crisis_weekly['SMB'].values
        _, p_weekly, _, _ = granger_test_manual(hml_w, smb_w, max_lag=min(10, len(hml_w) // 4))
        robustness['weekly'] = {
            'n_weeks': len(crisis_weekly),
            'p_value': float(p_weekly),
        }
        print(f"      {len(crisis_weekly)} weeks, p={p_weekly:.3f}")
    else:
        robustness['weekly'] = {'n_weeks': len(crisis_weekly), 'p_value': None}

    # 5. Elevated annual direction
    print("    (e) Elevated annual direction...")
    elevated_mask = (regimes == 1)
    years = df.index.year
    unique_years = sorted(set(years[elevated_mask]))
    positive_direction_count = 0
    total_years = 0

    for year in unique_years:
        year_mask = elevated_mask & (years == year)
        if year_mask.sum() < 30:
            continue
        indices = np.where(year_mask)[0]
        clean = []
        for idx in indices:
            if idx >= 5 and all(regimes[idx - l] == 1 for l in range(1, 6)):
                clean.append(idx)
        if len(clean) < 20:
            continue
        clean = np.array(clean)
        smb_c = smb_all[clean]
        hml_c = hml_all[clean]
        # Test SMB -> HML (the Elevated direction from paper)
        _, p_s2h, _, details = granger_test_manual(smb_c, hml_c, max_lag=5)
        if p_s2h < 0.5:  # direction check: is SMB->HML positive?
            positive_direction_count += 1
        total_years += 1

    robustness['elevated_annual'] = {
        'total_years': total_years,
        'positive_direction_count': positive_direction_count,
        'fraction': round(positive_direction_count / total_years * 100, 0) if total_years > 0 else None,
    }
    if total_years > 0:
        print(f"      {positive_direction_count}/{total_years} years ({positive_direction_count/total_years*100:.0f}%)")

    return robustness


# =============================================================================
# FROZEN OOS (ACTION 2)
# =============================================================================

def compute_frozen_oos(df, best_seed):
    """Train HMM on 1990-2012, freeze, classify 2013-2024."""
    print("\n  Computing Frozen OOS Validation...")
    train_df = df.loc[:'2012-12-31']
    test_df = df.loc['2013-01-01':]

    print(f"    Train: {len(train_df)} days, Test: {len(test_df)} days")

    hmm_train = StudentTHMM(n_regimes=3, n_iter=100, tol=1e-4, random_state=best_seed)
    hmm_train.fit(train_df.values)

    # Classify test with frozen params
    test_regimes_raw, test_probs = hmm_train.predict_oos(test_df.values, use_filtered=False)
    # Relabel test regimes by data norm for consistent labeling
    factor_cols = ['MKT', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']
    test_regimes, _ = relabel_regimes_by_data_norm(test_df, test_regimes_raw, factor_cols)

    # Raw regime counts (before boundary exclusion)
    raw_counts = {REGIME_NAMES[k]: int((test_regimes == k).sum()) for k in range(3)}
    crisis_fraction = raw_counts['Crisis'] / len(test_df) * 100
    print(f"    Raw test regime counts: {raw_counts}")
    print(f"    Crisis fraction: {crisis_fraction:.1f}%")

    # Aggregate per-regime Granger with fixed lag-15 boundary cleaning
    hml_test = test_df['HML'].values
    smb_test = test_df['SMB'].values

    clean_crisis = extract_regime_clean_indices(test_regimes, 2, max_lag=15)
    if len(clean_crisis) >= 30:
        hml_c = hml_test[clean_crisis]
        smb_c = smb_test[clean_crisis]
        best_lag, best_p, best_f, all_results = granger_test_manual(hml_c, smb_c, max_lag=15)
        aggregate = {
            'n_clean': len(clean_crisis),
            'best_lag': best_lag,
            'p_value': float(best_p),
            'f_stat': float(best_f),
        }
        print(f"    Aggregate Crisis HML->SMB: n={len(clean_crisis)}, lag={best_lag}, p={best_p:.3f}")
    else:
        aggregate = {'n_clean': len(clean_crisis), 'status': 'insufficient'}

    # Event-window tests
    test_events = [
        ('2015 China', '2015-08-10', '2015-09-30'),
        ('2020 COVID', '2020-02-15', '2020-06-30'),
        ('2022 Rate Hikes', '2022-01-01', '2022-06-30'),
    ]

    event_results = {}
    for event_name, start, end in test_events:
        mask = (test_df.index >= start) & (test_df.index <= end)
        n_days = int(mask.sum())
        if n_days < 20:
            continue
        event_regimes = test_regimes[np.where(mask)[0]]
        crisis_pct = float((event_regimes == 2).mean() * 100)
        hml_ev = test_df.loc[mask, 'HML'].values
        smb_ev = test_df.loc[mask, 'SMB'].values
        max_lag_ev = min(10, n_days // 3)
        if max_lag_ev >= 1:
            _, p_h2s, _, _ = granger_test_manual(hml_ev, smb_ev, max_lag=max_lag_ev)
            _, p_s2h, _, _ = granger_test_manual(smb_ev, hml_ev, max_lag=max_lag_ev)
        else:
            p_h2s, p_s2h = None, None

        event_results[event_name] = {
            'days': n_days,
            'crisis_pct': round(crisis_pct, 0),
            'hml_to_smb_p': float(p_h2s) if p_h2s is not None else None,
            'smb_to_hml_p': float(p_s2h) if p_s2h is not None else None,
        }
        print(f"    {event_name}: {n_days}d, Crisis={crisis_pct:.0f}%, "
              f"HML->SMB p={p_h2s:.3f}" if p_h2s else f"    {event_name}: insufficient")

    return {
        'train_period': f"{train_df.index[0].date()} to {train_df.index[-1].date()}",
        'test_period': f"{test_df.index[0].date()} to {test_df.index[-1].date()}",
        'train_n': len(train_df),
        'test_n': len(test_df),
        'raw_counts': raw_counts,
        'crisis_fraction_pct': round(crisis_fraction, 1),
        'aggregate': aggregate,
        'events': event_results,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Multi-start HMM pipeline with configurable selection.")
    parser.add_argument(
        '--selection-rule',
        choices=['ll_only', 'screened_2008', 'dual'],
        default='ll_only',
        help="Primary fit selection rule. 'dual' keeps ll_only primary and records screened sensitivity.",
    )
    parser.add_argument(
        '--seeds',
        default=None,
        help="Seed list/ranges, e.g. '0-49,77'. Default uses 0-49 (+42 deduped).",
    )
    parser.add_argument(
        '--output-tag',
        default='',
        help="Optional suffix for outputs, e.g. 'screened'. Produces *_<tag>.json/csv.",
    )
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    start_time = datetime.now()

    # Download data
    df = download_ff_data()
    assert len(df) == 8817, f"Expected 8,817 trading days, got {len(df)}"
    print(f"Data verification: {len(df)} trading days")

    # Phase 1: Multi-start HMM
    seeds = parse_seeds_arg(args.seeds)
    hmm, regimes, fit_summaries, best_seed, selection_info = run_multistart_hmm(
        df,
        seeds=seeds,
        selection_rule=args.selection_rule,
    )

    # Save regime assignments CSV
    regime_csv = pd.DataFrame({
        'date': df.index.strftime('%Y-%m-%d'),
        'regime_label': [REGIME_NAMES[r] for r in regimes],
    })
    csv_path = tagged_output_path('selected_fit_regimes', args.output_tag, 'csv')
    regime_csv.to_csv(csv_path, index=False)
    print(f"\n  Saved regime assignments to {csv_path}")

    # Compute all tables
    table1 = compute_table1(df, hmm, regimes)
    table2 = compute_table2_detection(df, regimes)
    table3 = compute_table3_granger(df, regimes)
    table4 = compute_table4_r2(df, regimes)
    table5 = compute_table5_warning(df, regimes)
    table6 = compute_table6_events(df, regimes)

    # All-pairs + lag sensitivity
    all_pairs = compute_all_pairs_granger(df, regimes, lag=5)
    lag_sensitivity = compute_lag_sensitivity(df, regimes, max_lag=15)

    # Robustness
    robustness = compute_robustness(df, hmm, regimes)

    # Frozen OOS (Action 2)
    frozen_oos = compute_frozen_oos(df, best_seed)

    # Record versions
    versions = {
        'python': sys.version,
        'numpy': np.__version__,
        'pandas': pd.__version__,
        'scipy': stats.scipy.__version__ if hasattr(stats, 'scipy') else 'unknown',
        'statsmodels': sm.__version__,
    }
    try:
        import scipy
        versions['scipy'] = scipy.__version__
    except Exception:
        pass

    # Build final JSON
    results = {
        'metadata': {
            'timestamp': str(datetime.now()),
            'n_days': len(df),
            'date_range': f"{df.index[0].date()} to {df.index[-1].date()}",
            'n_starts': len(fit_summaries),
            'selection_rule': args.selection_rule,
            'output_tag': args.output_tag,
            'versions': versions,
            'selection_info': selection_info,
        },
        'fit_summaries': fit_summaries,
        'selected_fit': {
            'seed': best_seed,
            'log_likelihood': float(hmm.log_likelihood_),
            'table1': table1,
            'detection': table2,
            'granger': table3,
            'r2': table4,
            'warning': table5,
            'events': table6,
            'all_pairs': all_pairs,
            'lag_sensitivity': lag_sensitivity,
            'robustness': robustness,
        },
        'frozen_oos': frozen_oos,
    }

    json_path = tagged_output_path('multistart_hmm_results', args.output_tag, 'json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n{'='*70}")
    print(f"PIPELINE COMPLETE in {elapsed:.0f}s")
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")
    print(f"  Selected seed: {best_seed}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
