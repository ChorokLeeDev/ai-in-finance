"""
Non-Finance Domain Validation for Regime-Conditional Granger Framework
========================================================================

Extends the regime-conditional Granger causality framework to two non-finance domains:

1. CLIMATE TELECONNECTIONS (ENSO → Regional Effects)
   - Uses synthetic ENSO-like data with KNOWN ground truth
   - 3 regimes: El Niño, Neutral, La Niña
   - Causality ONLY in El Niño regime
   - Tests whether framework recovers ground truth

2. ADDITIONAL MACRO PAIRS (FRED data)
   - Oil → CPI inflation
   - Federal funds rate → Unemployment
   - M2 money supply → GDP growth
   - Consumer sentiment → Retail sales

For each pair:
  1. Fit Student-t HMM (K=3)
  2. Run per-regime HAC Granger tests
  3. Compare regime-conditional vs unconditional structure
  4. Identify regimes showing predictability

Key insight: Regime conditioning should reveal structure missed by unconditional Granger tests.
"""

import sys
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats
from scipy.special import gammaln
from scipy.optimize import minimize_scalar
from scipy.cluster.vq import kmeans2
from scipy.stats import f as f_dist, chi2
import statsmodels.api as sm

warnings.filterwarnings('ignore')

RESULTS_DIR = '/sessions/modest-elegant-knuth/mnt/causal_regimes/results'
REGIME_NAMES = ['Low', 'Medium', 'High']
PRIMARY_SEED = 28


# =============================================================================
# STUDENT-T HMM (Reused from macro_regime_granger.py)
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
# SYNTHETIC CLIMATE DATA GENERATION
# =============================================================================

def generate_synthetic_enso_data(T=2000, random_state=42):
    """
    Generate synthetic ENSO-like data with ground truth causality structure.

    Structure:
    - 3 regimes: El Niño (high), Neutral (medium), La Niña (low)
    - Quasi-periodic 2-7 year cycles
    - In El Niño ONLY: X Granger-causes Y with known lag and coefficient
    - In other regimes: no Granger causality (independent)

    Returns:
        X: (T, 2) array [ENSO_proxy, Regional_Effect]
        regimes: (T,) ground truth regime labels (0=La Niña, 1=Neutral, 2=El Niño)
        truth: dict with ground truth parameters
    """
    np.random.seed(random_state)

    # Generate ENSO regime sequence using 3-state Markov chain
    # High persistence within regime, low transition across regimes
    regimes_raw = np.zeros(T, dtype=int)
    P = np.array([
        [0.90, 0.08, 0.02],  # From La Niña
        [0.08, 0.85, 0.07],  # From Neutral
        [0.02, 0.07, 0.91]   # From El Niño
    ])
    regimes_raw[0] = 1  # Start in Neutral
    for t in range(1, T):
        regimes_raw[t] = np.random.choice(3, p=P[regimes_raw[t-1]])

    # Add quasi-periodic modulation (2-7 year cycle ~ 24-84 months)
    time_idx = np.arange(T)
    cycle = 0.3 * np.sin(2 * np.pi * time_idx / 48)  # 48-month (4-year) cycle

    # Generate ENSO proxy (X)
    # Regime-dependent mean + cycle + noise
    regime_means_x = np.array([-1.0, 0.0, 1.0])  # La Niña, Neutral, El Niño
    x_base = regime_means_x[regimes_raw] + cycle
    x_noise = np.random.normal(0, 0.3, T)
    X_enso = x_base + x_noise

    # Generate Regional Effect (Y) with causality ONLY in El Niño regime
    # Y_t = 0.3 * X_{t-1} + noise  (in El Niño)
    # Y_t = noise                  (in Neutral or La Niña)
    lag = 1
    causal_coeff = 0.3
    Y = np.zeros(T)

    for t in range(lag, T):
        if regimes_raw[t] == 2:  # El Niño regime only
            Y[t] = causal_coeff * X_enso[t - lag] + np.random.normal(0, 0.2)
        else:
            Y[t] = np.random.normal(0, 0.2)

    X = np.column_stack([X_enso, Y])

    truth = {
        'causal_lag': lag,
        'causal_coeff': causal_coeff,
        'causal_regime': 2,  # El Niño
        'regime_names': ['La Niña', 'Neutral', 'El Niño'],
        'T': T,
        'noise_level': 0.2
    }

    return X, regimes_raw, truth


# =============================================================================
# GRANGER CAUSALITY FUNCTIONS (Reused from macro_regime_granger.py)
# =============================================================================

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
    r2_r = 1 - rss_r / tss if tss > 0 else 0
    r2_u = 1 - rss_u / tss if tss > 0 else 0
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


def unconditional_granger(y_all, x_all, max_lag=3):
    """Test unconditional (pooled) Granger causality across all data."""
    best_lag = select_lag_bic(y_all, x_all, np.arange(max_lag, len(y_all)), max_lag)
    usable = np.arange(best_lag, len(y_all))
    return run_granger_at_lag(y_all, x_all, usable, best_lag)


# =============================================================================
# SYNTHETIC CLIMATE VALIDATION
# =============================================================================

def validate_synthetic_climate():
    """
    Test regime-conditional Granger on synthetic ENSO data with known ground truth.

    Tests whether the framework:
    1. Identifies regimes similar to ground truth
    2. Detects causality in El Niño regime (where it exists)
    3. Misses causality in other regimes (correct negative results)
    """
    print("\n" + "=" * 70)
    print("DOMAIN 1: SYNTHETIC CLIMATE TELECONNECTIONS (ENSO → REGIONAL EFFECT)")
    print("=" * 70)

    # Generate synthetic data with known ground truth
    print("\n[1/4] Generating synthetic ENSO-like data (T=2000 months)...")
    X, regimes_true, truth = generate_synthetic_enso_data(T=2000, random_state=PRIMARY_SEED)
    print(f"  Ground truth: X causes Y with lag={truth['causal_lag']}, "
          f"coeff={truth['causal_coeff']}, regime={truth['regime_names'][truth['causal_regime']]}")

    # Fit HMM to discover regimes
    print("\n[2/4] Fitting Student-t HMM to discover regimes...")
    hmm = StudentTHMM(n_regimes=3, n_iter=100, tol=1e-4, random_state=PRIMARY_SEED)
    hmm.fit(X)
    print(f"  Train log-likelihood: {hmm.log_likelihood_:.2f}")

    # Predict regime labels
    regimes_pred = hmm.predict(X, use_filtered=False)
    regime_counts = {k: int((regimes_pred == k).sum()) for k in range(3)}
    print(f"  Predicted regime counts: {regime_counts}")

    # Evaluate regime recovery (adjusted rand index)
    from scipy.spatial.distance import pdist, squareform
    def adjusted_rand_index(labels_true, labels_pred):
        """Compute adjusted Rand index."""
        n = len(labels_true)
        a_ij = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                a_ij[i, j] = ((labels_true == i) & (labels_pred == j)).sum()
        # Adjusted Rand Index
        n_choose_2 = n * (n - 1) / 2
        sum_c = np.sum(a_ij ** 2) - n
        sum_r = np.sum(a_ij.sum(axis=1) ** 2) - n
        sum_s = np.sum(a_ij.sum(axis=0) ** 2) - n
        numerator = sum_c - (sum_r * sum_s) / n_choose_2
        denominator = 0.5 * (sum_r + sum_s) - (sum_r * sum_s) / n_choose_2
        ari = numerator / denominator if denominator > 0 else 0
        return float(ari)

    ari = adjusted_rand_index(regimes_true, regimes_pred)
    print(f"  Adjusted Rand Index (regime recovery): {ari:.4f}")

    # Extract data arrays
    X_enso = X[:, 0]
    Y_regional = X[:, 1]

    # Test Granger causality per regime
    print("\n[3/4] Testing regime-conditional Granger causality...")
    granger_results = {}
    regime_labels = ['Low', 'Medium', 'High']

    for k in range(3):
        clean_idx = extract_regime_clean_indices(regimes_pred, k, max_lag=3)
        if len(clean_idx) > 20:
            lag = select_lag_bic(Y_regional, X_enso, clean_idx, max_lag=3)
            result = run_granger_at_lag(Y_regional, X_enso, clean_idx, lag)
            if result is not None:
                granger_results[k] = result
                sig = "**YES**" if result['hac_p_value'] < 0.05 else "NO"
                print(f"  Regime {k} ({regime_labels[k]}): "
                      f"n={result['n_obs']}, lag={lag}, "
                      f"HAC-p={result['hac_p_value']:.4f} [{sig}], "
                      f"ΔR²={result['delta_r2']:.4f}")
            else:
                granger_results[k] = None
                print(f"  Regime {k}: Insufficient observations")
        else:
            granger_results[k] = None
            print(f"  Regime {k}: No clean regime observations")

    # Unconditional test
    print("\n[4/4] Unconditional Granger test (baseline for comparison)...")
    unconditional = unconditional_granger(Y_regional, X_enso)
    if unconditional:
        print(f"  Unconditional: HAC-p={unconditional['hac_p_value']:.4f}, "
              f"ΔR²={unconditional['delta_r2']:.4f}")

    # Build output
    climate_output = {
        'domain': 'Climate Teleconnections',
        'data_type': 'Synthetic ENSO-like',
        'T': 2000,
        'description': 'Synthetic ENSO proxy → regional effect with known causality structure',
        'ground_truth': truth,
        'hmm': {
            'n_regimes': 3,
            'random_state': PRIMARY_SEED,
            'train_log_likelihood': float(hmm.log_likelihood_),
            'regime_counts': regime_counts,
            'adjusted_rand_index': ari
        },
        'granger_results': {
            str(k): granger_results.get(k) for k in range(3)
        },
        'unconditional_granger': unconditional,
        'ground_truth_recovery': {
            'causal_regime_detected': (
                granger_results.get(2) is not None and
                granger_results.get(2)['hac_p_value'] < 0.05
            ),
            'non_causal_regimes_masked': all(
                granger_results.get(k) is None or
                granger_results.get(k)['hac_p_value'] > 0.05
                for k in [0, 1]
            )
        }
    }

    return climate_output


# =============================================================================
# MACRO PAIRS VALIDATION
# =============================================================================

def try_download_fred_series(series_id, start='1990-01-01', end='2024-12-31', max_retries=2):
    """Try to download FRED series via pandas_datareader."""
    try:
        from pandas_datareader.data import DataReader
        data = DataReader(series_id, 'fred', start=start, end=end)
        return data.squeeze()
    except Exception as e:
        print(f"    Warning: Failed to download {series_id}: {e}")
        return None


def prepare_macro_pair(x_series, y_series, x_name, y_name):
    """Align and prepare two macro series for analysis."""
    # Convert to numeric and drop NaN
    x = pd.Series(x_series, copy=True).astype(float).dropna()
    y = pd.Series(y_series, copy=True).astype(float).dropna()

    # Align indices
    common_idx = x.index.intersection(y.index)
    if len(common_idx) < 50:
        print(f"    Insufficient aligned observations: {len(common_idx)}")
        return None

    x = x[common_idx].reset_index(drop=True)
    y = y[common_idx].reset_index(drop=True)

    # Standardize
    x = (x - x.mean()) / (x.std() + 1e-6)
    y = (y - y.mean()) / (y.std() + 1e-6)

    return x, y


def validate_macro_pairs():
    """Test regime-conditional Granger on macro pairs from FRED."""
    print("\n" + "=" * 70)
    print("DOMAIN 2: ADDITIONAL MACRO PAIRS FROM FRED")
    print("=" * 70)

    macro_pairs = [
        ('DCOILWTICO', 'CPIAUCSL', 'Oil Price', 'CPI Inflation'),
        ('FEDFUNDS', 'UNRATE', 'Federal Funds Rate', 'Unemployment'),
        ('M2SL', 'GDPC1', 'M2 Money Supply', 'Real GDP'),
        ('UMCSENT', 'RSAFS', 'Consumer Sentiment', 'Retail Sales'),
    ]

    all_results = []

    for x_id, y_id, x_name, y_name in macro_pairs:
        print(f"\n[Pair] {x_name} → {y_name}")
        print(f"  Downloading {x_id}, {y_id}...")

        x_data = try_download_fred_series(x_id)
        y_data = try_download_fred_series(y_id)

        if x_data is None or y_data is None:
            print(f"  SKIPPED: Could not download data")
            continue

        prep = prepare_macro_pair(x_data, y_data, x_name, y_name)
        if prep is None:
            print(f"  SKIPPED: Could not align data")
            continue

        x_vals, y_vals = prep
        print(f"  Aligned: {len(x_vals)} observations")

        # Fit HMM
        print(f"  Fitting HMM...")
        X = np.column_stack([x_vals, y_vals])
        hmm = StudentTHMM(n_regimes=3, n_iter=100, tol=1e-4, random_state=PRIMARY_SEED)
        try:
            hmm.fit(X)
        except Exception as e:
            print(f"  SKIPPED: HMM fit failed: {e}")
            continue

        regimes = hmm.predict(X, use_filtered=False)
        regime_counts = {k: int((regimes == k).sum()) for k in range(3)}

        # Test Granger per regime
        print(f"  Regime-conditional Granger: ", end='')
        regime_tests = {}
        significant_regimes = []

        for k in range(3):
            clean_idx = extract_regime_clean_indices(regimes, k, max_lag=3)
            if len(clean_idx) > 10:
                lag = select_lag_bic(y_vals, x_vals, clean_idx, max_lag=3)
                result = run_granger_at_lag(y_vals, x_vals, clean_idx, lag)
                regime_tests[k] = result
                if result and result['hac_p_value'] < 0.05:
                    significant_regimes.append(k)
            else:
                regime_tests[k] = None

        # Unconditional test
        print(f"Unconditional: ", end='')
        unconditional = unconditional_granger(np.array(y_vals), np.array(x_vals))

        print(f"\n    Regime tests: ", end='')
        for k, res in regime_tests.items():
            if res:
                sig = "**" if res['hac_p_value'] < 0.05 else "  "
                print(f"R{k}(p={res['hac_p_value']:.3f}){sig} ", end='')
        print()

        if unconditional:
            unc_sig = "**" if unconditional['hac_p_value'] < 0.05 else "  "
            print(f"    Unconditional: p={unconditional['hac_p_value']:.3f}{unc_sig}")

        # Detect regime-conditional structure
        regime_conditional_found = (
            len(significant_regimes) > 0 and
            (unconditional is None or unconditional['hac_p_value'] > 0.05)
        )

        pair_result = {
            'x_series': x_id,
            'y_series': y_id,
            'x_name': x_name,
            'y_name': y_name,
            'n_obs': len(x_vals),
            'hmm': {
                'log_likelihood': float(hmm.log_likelihood_),
                'regime_counts': regime_counts
            },
            'regime_conditional_tests': {
                str(k): regime_tests.get(k) for k in range(3)
            },
            'significant_regimes': significant_regimes,
            'unconditional_granger': unconditional,
            'regime_conditional_structure_found': regime_conditional_found
        }

        all_results.append(pair_result)

    return all_results


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print("=" * 70)
    print("NON-FINANCE DOMAIN VALIDATION")
    print("Regime-Conditional Granger Causality Extension")
    print("=" * 70)

    results = {}

    # Domain 1: Synthetic Climate
    print("\nStarting Domain 1 validation...")
    try:
        results['climate_teleconnections'] = validate_synthetic_climate()
    except Exception as e:
        print(f"ERROR in climate domain: {e}")
        import traceback
        traceback.print_exc()
        results['climate_teleconnections'] = {'error': str(e)}

    # Domain 2: Macro Pairs
    print("\nStarting Domain 2 validation...")
    try:
        results['macro_pairs'] = validate_macro_pairs()
    except Exception as e:
        print(f"ERROR in macro pairs domain: {e}")
        import traceback
        traceback.print_exc()
        results['macro_pairs'] = {'error': str(e)}

    # Save results
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    output = {
        'timestamp': datetime.now().isoformat(),
        'description': 'Non-finance domain validation for regime-conditional Granger framework',
        'domains': results,
        'summary': {
            'climate': 'Synthetic ENSO-like data with known causality structure (ground truth validation)',
            'macro': 'Additional FRED macro pairs: oil→CPI, federal funds→unemployment, M2→GDP, sentiment→retail',
            'key_question': 'Does regime conditioning reveal structure missed by unconditional Granger tests?'
        }
    }

    outpath = f"{RESULTS_DIR}/nonfinance_domain_validation.json"
    with open(outpath, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {outpath}")

    # Summary report
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    if 'climate_teleconnections' in results and isinstance(results['climate_teleconnections'], dict):
        clim = results['climate_teleconnections']
        if 'ground_truth_recovery' in clim:
            print("\nSYNTHETIC CLIMATE:")
            print(f"  Causal regime detected: {clim['ground_truth_recovery']['causal_regime_detected']}")
            print(f"  Non-causal regimes masked: {clim['ground_truth_recovery']['non_causal_regimes_masked']}")
            print(f"  Regime recovery (ARI): {clim['hmm'].get('adjusted_rand_index', 'N/A'):.4f}")

    if 'macro_pairs' in results and isinstance(results['macro_pairs'], list):
        macro = results['macro_pairs']
        if macro:
            regime_cond_found = sum(
                1 for p in macro if p.get('regime_conditional_structure_found')
            )
            print(f"\nMAGRO PAIRS ({len(macro)} pairs):")
            print(f"  Regime-conditional structure found: {regime_cond_found}/{len(macro)}")
            for pair in macro:
                sig_regimes = pair.get('significant_regimes', [])
                print(f"    {pair['x_name']}→{pair['y_name']}: "
                      f"{len(sig_regimes)} significant regime(s): {sig_regimes}")

    print("\n" + "=" * 70)
    print("Validation complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
