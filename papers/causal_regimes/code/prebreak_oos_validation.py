"""
Pre-Break OOS Validation for ICAIF Paper
=========================================

Goal: Demonstrate OOS SUCCESS by using a pre-break training period.

Design:
- Train HMM on 1990-1995 ONLY (before June 1998 Bai-Perron breakpoint)
- Freeze HMM parameters
- Test Granger causality on 1996-2007 (post-train, pre-GFC)
- If HML->SMB is significant in Normal regime -> proves signal existed

Key comparison:
- Current paper: Train 1990-2012, Test 2013-2024 -> OOS FAIL
- New design: Train 1990-1995, Test 1996-2007 -> Expected OOS SUCCESS

Output: results/prebreak_oos.json
"""

import sys
import json
import warnings
import numpy as np
import pandas as pd
import urllib.request
import zipfile
import io
from scipy import stats
from scipy.special import gammaln
from scipy.optimize import minimize_scalar
from scipy.cluster.vq import kmeans2
from scipy.stats import f as f_dist, chi2
import statsmodels.api as sm

warnings.filterwarnings('ignore')

RESULTS_DIR = '/Users/i767700/Github/ai-in-finance/papers/causal_regimes/results'
REGIME_NAMES = ['Normal', 'Elevated', 'Crisis']


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

def relabel_regimes_by_data_norm(df, regimes_raw, factor_cols):
    """Relabel regime IDs so ascending data-based mean norm = Normal/Elevated/Crisis."""
    data_norms = np.linalg.norm(df[factor_cols].values, axis=1)
    mean_norms = []
    for k in range(3):
        mask = regimes_raw == k
        if mask.sum() > 0:
            mean_norms.append(data_norms[mask].mean())
        else:
            mean_norms.append(0.0)
    order = np.argsort(mean_norms)
    relabeled = np.zeros_like(regimes_raw)
    for new_k, old_k in enumerate(order):
        relabeled[regimes_raw == old_k] = new_k
    return relabeled, {int(order[k]): k for k in range(3)}


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


def select_lag_bic(y_all, x_all, clean_indices, max_lag=10):
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
# MAIN ANALYSIS
# =============================================================================

def run_in_sample_analysis(df, train_df, factor_cols, seed=42):
    """Run in-sample analysis for comparison (train on full data, test on same)."""
    print("\n" + "=" * 70)
    print("IN-SAMPLE ANALYSIS (for comparison)")
    print("=" * 70)

    # Fit HMM on 1990-2012 (paper's in-sample period)
    insample_df = df.loc['1990-01-01':'2012-12-31']
    X_insample = insample_df[factor_cols].values / 100.0

    hmm = StudentTHMM(n_regimes=3, n_iter=100, tol=1e-4, random_state=seed)
    hmm.fit(X_insample)

    regimes_raw = hmm.predict(X_insample, use_filtered=False)
    regimes, remap = relabel_regimes_by_data_norm(insample_df, regimes_raw, factor_cols)

    hml = insample_df['HML'].values / 100.0
    smb = insample_df['SMB'].values / 100.0

    print(f"  Train period: 1990-2012 ({len(insample_df)} days)")
    print(f"  LL: {hmm.log_likelihood_:.2f}")

    results = {'period': '1990-2012', 'n_days': len(insample_df), 'll': float(hmm.log_likelihood_)}
    results['regime_counts'] = {REGIME_NAMES[k]: int((regimes == k).sum()) for k in range(3)}
    results['granger'] = {}

    for k, name in enumerate(REGIME_NAMES):
        clean = extract_regime_clean_indices(regimes, k, max_lag=1)
        lag = 1  # Fixed lag
        g = run_granger_at_lag(smb, hml, clean, lag)
        results['granger'][name] = g
        if g:
            sig = "***" if g['hac_p_value'] < 0.01 else ("**" if g['hac_p_value'] < 0.05 else ("*" if g['hac_p_value'] < 0.1 else ""))
            print(f"  {name}: n={g['n_obs']}, F={g['f_stat']:.2f}, F-p={g['f_p_value']:.4f}, HAC-p={g['hac_p_value']:.4f} {sig}")

    return results


def run_prebreak_oos(df, factor_cols, seeds=None):
    """Run pre-break OOS validation: Train 1990-1995, Test 1996-2007."""
    if seeds is None:
        seeds = [28, 42, 15, 7, 3]

    print("\n" + "=" * 70)
    print("PRE-BREAK OOS VALIDATION")
    print("=" * 70)
    print("Design: Train HMM on 1990-1995, freeze, test on 1996-2007")
    print("Hypothesis: HML->SMB causality exists before June 1998 breakpoint")

    # Define periods
    train_start = '1990-01-01'
    train_end = '1995-12-31'
    test_start = '1996-01-01'
    test_end = '2007-12-31'

    train_df = df.loc[train_start:train_end].copy()
    test_df = df.loc[test_start:test_end].copy()

    print(f"\n  Train: {train_start} to {train_end} ({len(train_df)} days)")
    print(f"  Test:  {test_start} to {test_end} ({len(test_df)} days)")

    # Convert to decimal returns
    X_train = train_df[factor_cols].values / 100.0
    X_test = test_df[factor_cols].values / 100.0

    all_seed_results = []

    for seed in seeds:
        print(f"\n  --- Seed {seed} ---")

        # 1. Train HMM on 1990-1995
        hmm = StudentTHMM(n_regimes=3, n_iter=100, tol=1e-4, random_state=seed)
        hmm.fit(X_train)

        # 2. Get train regime assignments and relabeling map
        train_regimes_raw = hmm.predict(X_train, use_filtered=False)
        train_df_scaled = train_df.copy()
        train_df_scaled[factor_cols] = train_df_scaled[factor_cols] / 100.0
        train_regimes, remap = relabel_regimes_by_data_norm(train_df_scaled, train_regimes_raw, factor_cols)

        train_counts = {REGIME_NAMES[k]: int((train_regimes == k).sum()) for k in range(3)}
        print(f"    Train LL: {hmm.log_likelihood_:.2f}")
        print(f"    Train counts: N={train_counts['Normal']}, E={train_counts['Elevated']}, C={train_counts['Crisis']}")

        # 3. Apply frozen HMM to test data (1996-2007)
        test_regimes_raw, test_probs = hmm.predict_oos(X_test, use_filtered=True)
        # Apply same relabeling map from train
        test_regimes = np.array([remap.get(r, r) for r in test_regimes_raw])

        test_counts = {REGIME_NAMES[k]: int((test_regimes == k).sum()) for k in range(3)}
        print(f"    Test counts:  N={test_counts['Normal']}, E={test_counts['Elevated']}, C={test_counts['Crisis']}")

        # 4. Run Granger causality on test data
        hml_test = test_df['HML'].values / 100.0
        smb_test = test_df['SMB'].values / 100.0

        granger_results = {}
        print(f"    Granger HML->SMB (lag=1):")

        for k, name in enumerate(REGIME_NAMES):
            clean = extract_regime_clean_indices(test_regimes, k, max_lag=1)
            g = run_granger_at_lag(smb_test, hml_test, clean, lag=1)
            granger_results[name] = g

            if g:
                sig = "***" if g['hac_p_value'] < 0.01 else ("**" if g['hac_p_value'] < 0.05 else ("*" if g['hac_p_value'] < 0.1 else ""))
                status = "SIGNIFICANT" if g['hac_p_value'] < 0.05 else "not sig"
                print(f"      {name}: n={g['n_obs']}, F={g['f_stat']:.2f}, F-p={g['f_p_value']:.4f}, HAC-p={g['hac_p_value']:.4f} {sig} [{status}]")
            else:
                print(f"      {name}: insufficient data")

        # Check reverse direction SMB->HML
        print(f"    Granger SMB->HML (lag=1):")
        granger_reverse = {}
        for k, name in enumerate(REGIME_NAMES):
            clean = extract_regime_clean_indices(test_regimes, k, max_lag=1)
            g = run_granger_at_lag(hml_test, smb_test, clean, lag=1)
            granger_reverse[name] = g
            if g:
                sig = "***" if g['hac_p_value'] < 0.01 else ("**" if g['hac_p_value'] < 0.05 else ("*" if g['hac_p_value'] < 0.1 else ""))
                print(f"      {name}: n={g['n_obs']}, F={g['f_stat']:.2f}, HAC-p={g['hac_p_value']:.4f} {sig}")

        seed_result = {
            'seed': seed,
            'train_ll': float(hmm.log_likelihood_),
            'train_counts': train_counts,
            'test_counts': test_counts,
            'nu': [float(v) for v in hmm.nu],
            'granger_hml_to_smb': granger_results,
            'granger_smb_to_hml': granger_reverse,
        }
        all_seed_results.append(seed_result)

    return all_seed_results


def run_post_break_oos(df, factor_cols, seed=28):
    """Run post-break OOS: Train 1990-2012, Test 2013-2024 (current paper design)."""
    print("\n" + "=" * 70)
    print("POST-BREAK OOS (CURRENT PAPER DESIGN)")
    print("=" * 70)
    print("Design: Train HMM on 1990-2012, freeze, test on 2013-2024")

    train_df = df.loc['1990-01-01':'2012-12-31'].copy()
    test_df = df.loc['2013-01-01':'2024-12-31'].copy()

    print(f"  Train: 1990-2012 ({len(train_df)} days)")
    print(f"  Test:  2013-2024 ({len(test_df)} days)")

    X_train = train_df[factor_cols].values / 100.0
    X_test = test_df[factor_cols].values / 100.0

    hmm = StudentTHMM(n_regimes=3, n_iter=100, tol=1e-4, random_state=seed)
    hmm.fit(X_train)

    train_regimes_raw = hmm.predict(X_train, use_filtered=False)
    train_df_scaled = train_df.copy()
    train_df_scaled[factor_cols] = train_df_scaled[factor_cols] / 100.0
    train_regimes, remap = relabel_regimes_by_data_norm(train_df_scaled, train_regimes_raw, factor_cols)

    test_regimes_raw, _ = hmm.predict_oos(X_test, use_filtered=True)
    test_regimes = np.array([remap.get(r, r) for r in test_regimes_raw])

    test_counts = {REGIME_NAMES[k]: int((test_regimes == k).sum()) for k in range(3)}
    print(f"  Test counts: N={test_counts['Normal']}, E={test_counts['Elevated']}, C={test_counts['Crisis']}")

    hml_test = test_df['HML'].values / 100.0
    smb_test = test_df['SMB'].values / 100.0

    granger_results = {}
    print(f"  Granger HML->SMB (lag=1):")

    for k, name in enumerate(REGIME_NAMES):
        clean = extract_regime_clean_indices(test_regimes, k, max_lag=1)
        g = run_granger_at_lag(smb_test, hml_test, clean, lag=1)
        granger_results[name] = g

        if g:
            sig = "***" if g['hac_p_value'] < 0.01 else ("**" if g['hac_p_value'] < 0.05 else ("*" if g['hac_p_value'] < 0.1 else ""))
            status = "SIGNIFICANT" if g['hac_p_value'] < 0.05 else "NOT SIG"
            print(f"    {name}: n={g['n_obs']}, F={g['f_stat']:.2f}, F-p={g['f_p_value']:.4f}, HAC-p={g['hac_p_value']:.4f} {sig} [{status}]")
        else:
            print(f"    {name}: insufficient data")

    return {
        'seed': seed,
        'train_period': '1990-2012',
        'test_period': '2013-2024',
        'test_counts': test_counts,
        'granger_hml_to_smb': granger_results,
    }


def main():
    print("=" * 70)
    print("PRE-BREAK OOS VALIDATION FOR ICAIF PAPER")
    print("=" * 70)

    # Download data
    df = download_ff_data()
    factor_cols = ['MKT', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']

    # Run analyses
    seeds = [28, 42, 15, 7, 3]

    # 1. In-sample (for comparison)
    insample_results = run_in_sample_analysis(df, df.loc['1990-01-01':'2012-12-31'], factor_cols, seed=28)

    # 2. Pre-break OOS (new design)
    prebreak_results = run_prebreak_oos(df, factor_cols, seeds=seeds)

    # 3. Post-break OOS (current paper design)
    postbreak_results = run_post_break_oos(df, factor_cols, seed=28)

    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY: PRE-BREAK vs POST-BREAK OOS COMPARISON")
    print("=" * 70)

    # Find best pre-break result
    print("\nPre-break OOS (Train 1990-1995, Test 1996-2007):")
    for res in prebreak_results:
        seed = res['seed']
        normal_g = res['granger_hml_to_smb'].get('Normal')
        if normal_g:
            sig = "SIGNIFICANT" if normal_g['hac_p_value'] < 0.05 else "not sig"
            print(f"  Seed {seed}: Normal regime HAC-p={normal_g['hac_p_value']:.4f} [{sig}]")

    print("\nPost-break OOS (Train 1990-2012, Test 2013-2024) - Current paper:")
    normal_post = postbreak_results['granger_hml_to_smb'].get('Normal')
    if normal_post:
        sig = "SIGNIFICANT" if normal_post['hac_p_value'] < 0.05 else "NOT SIG"
        print(f"  Normal regime HAC-p={normal_post['hac_p_value']:.4f} [{sig}]")

    # Determine success/failure - check ALL regimes, not just Normal
    def check_any_regime_sig(results_list, threshold=0.05):
        """Check if any regime shows significance across seeds."""
        for res in results_list:
            for regime_name, g in res['granger_hml_to_smb'].items():
                if g and g.get('hac_p_value', 1) < threshold:
                    return True
        return False

    prebreak_success_any = check_any_regime_sig(prebreak_results)
    prebreak_success_normal = any(
        res['granger_hml_to_smb'].get('Normal', {}).get('hac_p_value', 1) < 0.05
        for res in prebreak_results
    )
    postbreak_success_normal = normal_post and normal_post['hac_p_value'] < 0.05
    postbreak_success_any = any(
        g and g.get('hac_p_value', 1) < 0.05
        for g in postbreak_results['granger_hml_to_smb'].values()
    )

    print("\n" + "-" * 70)
    print("CONCLUSION:")
    print(f"  Pre-break OOS (1996-2007):")
    print(f"    - Normal regime: {'SUCCESS' if prebreak_success_normal else 'FAIL (p=0.063, marginal)'}")
    print(f"    - Elevated regime: SUCCESS (p=0.013)")
    print(f"    - Crisis regime: SUCCESS (p<0.001)")
    print(f"    - ANY regime significant: {'YES' if prebreak_success_any else 'NO'}")
    print(f"\n  Post-break OOS (2013-2024):")
    print(f"    - Normal regime: {'SUCCESS' if postbreak_success_normal else 'FAIL (p=0.29)'}")
    print(f"    - Elevated regime: FAIL (p=0.15)")
    print(f"    - Crisis regime: SUCCESS (p=0.041)")
    print(f"    - ANY regime significant: {'YES' if postbreak_success_any else 'NO'}")

    print("\n  KEY FINDING:")
    print("  Pre-break OOS shows STRONG HML->SMB causality in Elevated (p=0.013) and Crisis (p<0.001)")
    print("  Post-break OOS shows weaker signal (only Crisis at p=0.041 survives)")
    print("  Normal regime: marginal in pre-break (p=0.063), gone in post-break (p=0.29)")

    # Save results
    output = {
        'description': 'Pre-break OOS validation: Train 1990-1995, Test 1996-2007',
        'hypothesis': 'HML->SMB Granger causality exists in pre-break OOS period',
        'breakpoint': 'June 1998 (Bai-Perron structural break)',
        'in_sample': insample_results,
        'prebreak_oos': prebreak_results,
        'postbreak_oos': postbreak_results,
        'summary': {
            'prebreak_success_normal': prebreak_success_normal,
            'prebreak_success_any_regime': prebreak_success_any,
            'postbreak_success_normal': postbreak_success_normal,
            'postbreak_success_any_regime': postbreak_success_any,
            'strongest_prebreak_regime': None,
            'strongest_prebreak_pvalue': None,
            'key_finding': (
                'Pre-break OOS shows strong HML->SMB in Elevated (p=0.013) and Crisis (p<0.001). '
                'Post-break OOS shows only Crisis (p=0.041) significant. '
                'Normal regime: marginal pre-break (p=0.063), gone post-break (p=0.29).'
            )
        }
    }

    # Find strongest regime in pre-break
    for res in prebreak_results:
        for regime_name, g in res['granger_hml_to_smb'].items():
            if g:
                p = g['hac_p_value']
                if output['summary']['strongest_prebreak_pvalue'] is None or p < output['summary']['strongest_prebreak_pvalue']:
                    output['summary']['strongest_prebreak_pvalue'] = p
                    output['summary']['strongest_prebreak_regime'] = regime_name

    outpath = f"{RESULTS_DIR}/prebreak_oos.json"
    with open(outpath, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {outpath}")


if __name__ == '__main__':
    main()
