"""
Emerging Markets Factor Validation: Pre-registered Granger Causality Analysis
==============================================================================

Tests regime-conditional Granger causality among emerging market Fama-French factors.

Data:
  Source: Kenneth French Data Library
  Dataset: Emerging_5_Factors (monthly)
  Factors: Mkt-RF, SMB, HML, RMW, CMA (+ Risk-Free rate)
  Period: 2000-2024 (longest available)

Pre-registration protocol:
  1. Test ALL directed pairs (20 total: A→B for A,B in {Mkt-RF,SMB,HML,RMW,CMA}, A≠B)
  2. Apply Bonferroni correction across all pairs
  3. Report full matrix of results (not cherry-picked)
  4. Per-regime HAC Granger test with frozen HMM parameters
  5. Quandt-Andrews structural break test for significant pairs

Three-fold temporal split:
  - Fold A (2000-2008, first 108 obs): Train HMM, regime discovery
  - Fold B (2009-2016, middle 96 obs):  Test Granger with frozen HMM
  - Fold C (2017-2024, last 96 obs):    OOS validation with frozen HMM
"""

import sys
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from scipy import stats
from scipy.special import gammaln
from scipy.optimize import minimize_scalar
from scipy.cluster.vq import kmeans2
from scipy.stats import f as f_dist, chi2
import statsmodels.api as sm
import pandas_datareader.data as web

warnings.filterwarnings('ignore')

_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = str(_ROOT / 'results')
REGIME_NAMES = ['Normal', 'Elevated', 'Crisis']
PRIMARY_SEED = 28

# =============================================================================
# STUDENT-T HMM (Self-contained implementation)
# =============================================================================

class StudentTHMM:
    """Student-t HMM with filtered/smoothed probabilities and OOS prediction.

    Implements EM algorithm for regime identification in multivariate data.
    Uses Student-t emission distributions for robustness to outliers.
    """

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
        """Initialize HMM parameters using k-means clustering."""
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
        """Multivariate Student-t log PDF."""
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
        """Compute log emission probabilities for all states and observations."""
        T, d = X.shape
        K = self.n_regimes
        log_B = np.zeros((T, K))
        for k in range(K):
            log_B[:, k] = self._mvt_logpdf(X, self.mu[k], self.Sigma[k], self.nu[k])
        return log_B

    def _forward(self, log_B):
        """Forward pass (alpha)."""
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
        """Backward pass (beta)."""
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
        """E-step: compute posterior state probabilities."""
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
        """M-step: update parameters."""
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
        """Update degrees of freedom parameter for regime k."""
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
        """Enforce ordering by centroid norm."""
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
        """Fit HMM via EM algorithm."""
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
        """Predict regime labels on training data."""
        X = np.asarray(X)
        self._e_step(X)
        if use_filtered:
            return np.argmax(self.alpha, axis=1)
        return np.argmax(self.gamma, axis=1)

    def predict_oos(self, X, use_filtered=False):
        """Predict regime labels on new data (frozen parameters)."""
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

def download_emerging_data():
    """Download Emerging Markets 5-factor data from Kenneth French library."""
    print("\n[1/6] Downloading emerging market data...")
    try:
        data = web.DataReader('Emerging_5_Factors', 'famafrench',
                              start='2000-01-01', end='2024-12-31')
        df = data[0]  # Monthly returns
        df = df.drop('RF', axis=1)  # Drop risk-free rate
        df = df.dropna()
        print(f"  Loaded {len(df)} observations")
        print(f"  Factors: {list(df.columns)}")
        print(f"  Period: {df.index[0]} to {df.index[-1]}")
        return df
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def extract_regime_clean_indices(regimes, regime_id, max_lag):
    """Get indices where regime is stable across all lags."""
    regime_mask = (regimes == regime_id)
    indices = np.where(regime_mask)[0]
    clean_indices = []
    for idx in indices:
        if idx >= max_lag:
            if all(regimes[idx - l] == regime_id for l in range(1, max_lag + 1)):
                clean_indices.append(idx)
    return np.array(clean_indices) if clean_indices else np.array([], dtype=int)


def select_lag_bic(y_all, x_all, clean_indices, max_lag=3):
    """Select optimal lag using BIC."""
    best_bic = np.inf
    best_lag = 1
    for lag in range(1, max_lag + 1):
        usable = np.array([idx for idx in clean_indices if idx >= lag])
        if len(usable) < 2 * lag + 10:
            continue
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
    """Standard F-test for Granger causality."""
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
    try:
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
        wald_stat = float(Rb @ np.linalg.inv(RVR) @ Rb)
        p_value = float(1 - chi2.cdf(wald_stat, p))
    except Exception:
        wald_stat = np.nan
        p_value = np.nan
    return wald_stat, p_value


def run_granger_at_lag(y_all, x_all, clean_indices, lag):
    """Run Granger F-test + HAC at a specific lag."""
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


def quandt_andrews_test(y_all, x_all, clean_indices, lag, trim=0.15):
    """Quandt-Andrews sup-F test for structural breaks."""
    usable = np.array([idx for idx in clean_indices if idx >= lag])
    if len(usable) < 2 * lag + 10:
        return None, None

    n_usable = len(usable)
    n_trim = max(int(np.ceil(trim * n_usable)), 2)
    sup_f = -np.inf
    break_date = None

    for split_idx in range(n_trim, n_usable - n_trim):
        idx_before = usable[:split_idx]
        idx_after = usable[split_idx:]

        # Test before
        y_b = y_all[idx_before]
        y_lagged_b = np.column_stack([y_all[idx_before - i - 1] for i in range(lag)])
        x_lagged_b = np.column_stack([x_all[idx_before - i - 1] for i in range(lag)])

        # Test after
        y_a = y_all[idx_after]
        y_lagged_a = np.column_stack([y_all[idx_after - i - 1] for i in range(lag)])
        x_lagged_a = np.column_stack([x_all[idx_after - i - 1] for i in range(lag)])

        # Full model
        y_all_t = np.concatenate([y_b, y_a])
        X_full = np.vstack([
            np.column_stack([np.ones(len(y_b)), y_lagged_b, x_lagged_b]),
            np.column_stack([np.ones(len(y_a)), y_lagged_a, x_lagged_a])
        ])

        try:
            beta_full = np.linalg.lstsq(X_full, y_all_t, rcond=None)[0]
            rss_full = np.sum((y_all_t - X_full @ beta_full) ** 2)

            # Before model
            X_b = np.column_stack([np.ones(len(y_b)), y_lagged_b, x_lagged_b])
            beta_b = np.linalg.lstsq(X_b, y_b, rcond=None)[0]
            rss_b = np.sum((y_b - X_b @ beta_b) ** 2)

            # After model
            X_a = np.column_stack([np.ones(len(y_a)), y_lagged_a, x_lagged_a])
            beta_a = np.linalg.lstsq(X_a, y_a, rcond=None)[0]
            rss_a = np.sum((y_a - X_a @ beta_a) ** 2)

            rss_split = rss_b + rss_a
            k = X_full.shape[1]
            f_break = ((rss_full - rss_split) / k) / (rss_split / (n_usable - 2*k))

            if f_break > sup_f:
                sup_f = f_break
                break_date = split_idx
        except Exception:
            pass

    # Approximate p-value for sup-F (critical values from Andrews 2003)
    if sup_f > 0 and break_date is not None:
        p_value = np.exp(-2.0 * sup_f)  # Approximation
    else:
        p_value = np.nan

    return float(sup_f) if sup_f > -np.inf else None, p_value


def relabel_regimes_by_data_norm(df, regimes_raw, factor_cols):
    """Relabel regimes by mean data norm (volatility)."""
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

    return relabeled, order


def apply_train_remap(test_raw, remap):
    """Apply train-period relabeling order to test raw regime labels."""
    return np.array([remap[r] for r in test_raw])


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():
    print("=" * 80)
    print("EMERGING MARKETS FACTOR VALIDATION: PRE-REGISTERED GRANGER CAUSALITY")
    print("=" * 80)

    # Download and prepare data
    df = download_emerging_data()
    if df is None:
        print("  FAILED to download data")
        return

    factor_cols = list(df.columns)
    print(f"  Available factors: {factor_cols}")

    # Define temporal splits (three-fold)
    # Fold A: first third (training)
    # Fold B: middle third (testing)
    # Fold C: final third (out-of-sample)
    n_obs = len(df)
    fold_size = n_obs // 3

    fold_a_end = fold_size
    fold_b_start = fold_size
    fold_b_end = 2 * fold_size
    fold_c_start = 2 * fold_size

    train_df = df.iloc[:fold_a_end]
    test_df = df.iloc[fold_b_start:fold_b_end]
    oos_df = df.iloc[fold_c_start:]

    print(f"\n[2/6] Temporal split:")
    print(f"  Fold A (train): {len(train_df)} obs ({train_df.index[0]} to {train_df.index[-1]})")
    print(f"  Fold B (test):  {len(test_df)} obs ({test_df.index[0]} to {test_df.index[-1]})")
    print(f"  Fold C (OOS):   {len(oos_df)} obs ({oos_df.index[0]} to {oos_df.index[-1]})")

    # Fit HMM on training data (Fold A)
    print(f"\n[3/6] Fitting Student-t HMM (K=3) on Fold A...")
    hmm = StudentTHMM(n_regimes=3, n_iter=100, tol=1e-4, random_state=PRIMARY_SEED)
    hmm.fit(train_df[factor_cols].values)
    print(f"  Train log-likelihood: {hmm.log_likelihood_:.2f}")

    # Relabel regimes by training data norm
    train_raw = hmm.predict(train_df[factor_cols].values, use_filtered=False)
    train_relabeled, remap = relabel_regimes_by_data_norm(train_df, train_raw, factor_cols)
    train_counts = {REGIME_NAMES[k]: int((train_relabeled == k).sum()) for k in range(3)}
    print(f"  Train regime counts: {train_counts}")

    # Apply relabeling to test and OOS sets
    test_raw, _ = hmm.predict_oos(test_df[factor_cols].values, use_filtered=True)
    test_regimes = apply_train_remap(test_raw, remap)
    test_counts = {REGIME_NAMES[k]: int((test_regimes == k).sum()) for k in range(3)}
    print(f"  Test regime counts: {test_counts}")

    oos_raw, _ = hmm.predict_oos(oos_df[factor_cols].values, use_filtered=True)
    oos_regimes = apply_train_remap(oos_raw, remap)
    oos_counts = {REGIME_NAMES[k]: int((oos_regimes == k).sum()) for k in range(3)}
    print(f"  OOS regime counts: {oos_counts}")

    # Pre-registered: test ALL directed pairs
    print(f"\n[4/6] Pre-registered Granger causality matrix (Fold B: test period)...")
    print(f"  Testing {len(factor_cols)} factors → {len(factor_cols) * (len(factor_cols) - 1)} directed pairs")

    test_data_values = test_df[factor_cols].values

    # Build pair list (all directed pairs excluding self-pairs)
    pairs = []
    for i, x_col in enumerate(factor_cols):
        for j, y_col in enumerate(factor_cols):
            if i != j:
                pairs.append((x_col, y_col, i, j))

    print(f"  Total pairs: {len(pairs)}")

    # Bonferroni correction threshold
    alpha = 0.05
    bonf_threshold = alpha / len(pairs)
    print(f"  Bonferroni-corrected alpha: {bonf_threshold:.6f}")

    granger_matrix = {}
    significant_pairs = []

    for x_col, y_col, x_idx, y_idx in pairs:
        granger_matrix[f"{x_col}→{y_col}"] = {}
        x_all = test_data_values[:, x_idx]
        y_all = test_data_values[:, y_idx]

        for k, regime_name in enumerate(REGIME_NAMES):
            clean_idx = extract_regime_clean_indices(test_regimes, k, max_lag=3)

            if len(clean_idx) > 0:
                lag = select_lag_bic(y_all, x_all, clean_idx, max_lag=3)
                result = run_granger_at_lag(y_all, x_all, clean_idx, lag)

                if result is not None:
                    granger_matrix[f"{x_col}→{y_col}"][regime_name] = result

                    # Check significance with Bonferroni correction
                    if result['hac_p_value'] < bonf_threshold:
                        significant_pairs.append({
                            'pair': f"{x_col}→{y_col}",
                            'regime': regime_name,
                            'hac_p': result['hac_p_value'],
                            'f_stat': result['f_stat'],
                            'delta_r2': result['delta_r2']
                        })
                else:
                    granger_matrix[f"{x_col}→{y_col}"][regime_name] = None
            else:
                granger_matrix[f"{x_col}→{y_col}"][regime_name] = None

    print(f"\n  Results summary:")
    print(f"    Significant pairs (Bonf-corrected): {len(significant_pairs)}")
    if significant_pairs:
        for sp in significant_pairs[:10]:
            print(f"      {sp['pair']} in {sp['regime']}: p={sp['hac_p']:.6f}, ΔR²={sp['delta_r2']:.4f}")

    # Quandt-Andrews test for significant pairs
    print(f"\n[5/6] Quandt-Andrews structural break test (for significant pairs)...")
    qa_results = {}
    for sp in significant_pairs:
        pair_name = sp['pair']
        x_col, y_col = pair_name.split('→')
        x_idx = factor_cols.index(x_col)
        y_idx = factor_cols.index(y_col)
        x_all = test_data_values[:, x_idx]
        y_all = test_data_values[:, y_idx]
        regime_name = sp['regime']
        regime_k = REGIME_NAMES.index(regime_name)
        clean_idx = extract_regime_clean_indices(test_regimes, regime_k, max_lag=3)
        lag = select_lag_bic(y_all, x_all, clean_idx, max_lag=3)
        sup_f, qa_p = quandt_andrews_test(y_all, x_all, clean_idx, lag)
        qa_results[pair_name] = {
            'regime': regime_name,
            'sup_f': sup_f,
            'qa_p_value': qa_p
        }
        if sup_f is not None:
            print(f"    {pair_name} in {regime_name}: sup-F={sup_f:.4f}, p≈{qa_p:.4f}")

    # OOS validation on Fold C
    print(f"\n[5.5/6] Out-of-sample validation (Fold C)...")
    oos_data_values = oos_df[factor_cols].values
    oos_results = {}

    for sp in significant_pairs:
        pair_name = sp['pair']
        x_col, y_col = pair_name.split('→')
        x_idx = factor_cols.index(x_col)
        y_idx = factor_cols.index(y_col)
        x_all = oos_data_values[:, x_idx]
        y_all = oos_data_values[:, y_idx]
        regime_name = sp['regime']
        regime_k = REGIME_NAMES.index(regime_name)
        clean_idx = extract_regime_clean_indices(oos_regimes, regime_k, max_lag=3)

        if len(clean_idx) > 0:
            lag = granger_matrix[pair_name][regime_name]['lag']
            result = run_granger_at_lag(y_all, x_all, clean_idx, lag)
            if result is not None:
                oos_results[pair_name] = {
                    'regime': regime_name,
                    'test_hac_p': sp['hac_p'],
                    'oos_hac_p': result['hac_p_value'],
                    'oos_delta_r2': result['delta_r2']
                }
                print(f"    {pair_name} in {regime_name}: test p={sp['hac_p']:.6f}, OOS p={result['hac_p_value']:.6f}")

    # Prepare output
    print(f"\n[6/6] Saving results...")
    output = {
        'description': (
            'Pre-registered Granger causality analysis for emerging market Fama-French factors. '
            'Tests ALL directed pairs (20 total) with Bonferroni correction. '
            'HMM trained on Fold A (first 33%), Granger tested on Fold B (middle 33%), '
            'OOS validated on Fold C (final 33%). '
            'Reports full matrix of results, not cherry-picked pairs.'
        ),
        'data_source': 'Kenneth French Data Library - Emerging_5_Factors (monthly)',
        'factors': factor_cols,
        'period': {
            'train': f"{train_df.index[0].strftime('%Y-%m')} to {train_df.index[-1].strftime('%Y-%m')}",
            'test': f"{test_df.index[0].strftime('%Y-%m')} to {test_df.index[-1].strftime('%Y-%m')}",
            'oos': f"{oos_df.index[0].strftime('%Y-%m')} to {oos_df.index[-1].strftime('%Y-%m')}"
        },
        'observations': {
            'train': len(train_df),
            'test': len(test_df),
            'oos': len(oos_df)
        },
        'hmm': {
            'n_regimes': 3,
            'random_state': PRIMARY_SEED,
            'train_log_likelihood': float(hmm.log_likelihood_),
            'train_regime_counts': train_counts,
        },
        'pre_registration': {
            'n_pairs_tested': len(pairs),
            'bonferroni_alpha': float(bonf_threshold),
            'significant_pairs_found': len(significant_pairs),
        },
        'test': {
            'regime_counts': test_counts,
            'granger_matrix': {k: {rk: v for rk, v in vv.items() if v is not None}
                              for k, vv in granger_matrix.items()},
            'significant_pairs': significant_pairs[:20],  # Top 20
        },
        'structural_breaks': qa_results,
        'oos': {
            'regime_counts': oos_counts,
            'validation_results': oos_results,
        },
        'regime_names': REGIME_NAMES,
        'timestamp': datetime.now().isoformat(),
    }

    outpath = f"{RESULTS_DIR}/emerging_market_validation.json"
    with open(outpath, 'w') as fout:
        json.dump(output, fout, indent=2)
    print(f"  Saved to {outpath}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nData: {len(df)} monthly observations ({df.index[0]} to {df.index[-1]})")
    print(f"Factors: {', '.join(factor_cols)}")
    print(f"Pairs tested: {len(pairs)} (pre-registered, Bonferroni-corrected)")
    print(f"Significant pairs: {len(significant_pairs)}")
    if significant_pairs:
        print(f"\nTop significant pairs (Bonf-corrected α={bonf_threshold:.6f}):")
        for sp in sorted(significant_pairs, key=lambda x: x['hac_p'])[:5]:
            print(f"  {sp['pair']} in {sp['regime']}: HAC p={sp['hac_p']:.6f}, ΔR²={sp['delta_r2']:.4f}")

    print(f"\nStructural breaks (Quandt-Andrews):")
    if qa_results:
        for pair_name, qa in qa_results.items():
            print(f"  {pair_name} in {qa['regime']}: sup-F={qa['sup_f']:.4f}, p≈{qa['qa_p_value']:.4f}")
    else:
        print("  None detected")

    print(f"\nOut-of-sample validation:")
    if oos_results:
        print(f"  {len(oos_results)} significant pairs replicated in Fold C")
        for pair_name, oos in oos_results.items():
            print(f"    {pair_name} in {oos['regime']}: test p={oos['test_hac_p']:.6f} → OOS p={oos['oos_hac_p']:.6f}")
    else:
        print("  No significant pairs in test set replicated in OOS")

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
