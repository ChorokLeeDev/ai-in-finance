"""
Online Regime Detection for ICAIF
=================================

Real-time (streaming) regime detection comparing:
1. Batch HMM (oracle - sees all data)
2. Online HMM (streaming with incremental updates)
3. Rolling Window HMM (re-fit every month)
4. Simple Volatility Threshold (baseline)

Metrics:
- Detection delay: Days after batch HMM detects regime change
- Accuracy: Agreement with batch HMM labels
- F1 per regime
"""

import numpy as np
import pandas as pd
import json
import os
import urllib.request
import zipfile
import io
from datetime import datetime
from scipy import stats
from scipy.special import gammaln
from scipy.optimize import minimize_scalar
from scipy.cluster.vq import kmeans2
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from scipy.optimize import linear_sum_assignment
import warnings
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
    df = df.loc['1990-01-01':'2024-12-31']
    print(f"Loaded {len(df)} trading days ({df.index[0].date()} to {df.index[-1].date()})")
    return df


# =============================================================================
# STUDENT-T HMM (BATCH VERSION)
# =============================================================================

class StudentTHMM:
    """Student-t HMM for batch fitting."""

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
                denom = self.gamma[:-1, j].sum()
                if denom > 0:
                    self.A[j, k] = self.xi[:, j, k].sum() / denom
        self.A = self.A / self.A.sum(axis=1, keepdims=True)
        for k in range(K):
            weights = self.gamma[:, k] * self.u[:, k]
            if weights.sum() > 0:
                self.mu[k] = (weights[:, None] * X).sum(axis=0) / weights.sum()
        for k in range(K):
            diff = X - self.mu[k]
            weights = self.gamma[:, k] * self.u[:, k]
            weighted_outer = np.zeros((d, d))
            for t in range(T):
                weighted_outer += weights[t] * np.outer(diff[t], diff[t])
            denom = self.gamma[:, k].sum()
            if denom > 0:
                self.Sigma[k] = weighted_outer / denom
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
            if self.xi is not None:
                self.xi = self.xi[:, order, :][:, :, order]

    def fit(self, X, verbose=True):
        X = np.asarray(X)
        T, d = X.shape
        self._init_params(X)
        prev_ll = -np.inf
        for iteration in range(self.n_iter):
            log_likelihood = self._e_step(X)
            self._m_step(X)
            if abs(log_likelihood - prev_ll) < self.tol:
                if verbose:
                    print(f"Converged at iteration {iteration + 1}")
                break
            prev_ll = log_likelihood
        self.log_likelihood_ = log_likelihood
        return self

    def predict(self, X):
        X = np.asarray(X)
        self._e_step(X)
        return np.argmax(self.gamma, axis=1)

    def predict_proba(self, X):
        X = np.asarray(X)
        log_B = self._compute_emission_probs(X)
        log_alpha = self._forward(log_B)
        log_beta = self._backward(log_B)
        log_gamma = log_alpha + log_beta
        log_gamma = log_gamma - np.logaddexp.reduce(log_gamma, axis=1, keepdims=True)
        return np.exp(log_gamma)

    def predict_single(self, x, prev_state_prob):
        """Predict regime for a single observation given previous state distribution."""
        x = np.asarray(x).reshape(1, -1)
        K = self.n_regimes
        # Emission probability for single observation
        log_b = np.zeros(K)
        for k in range(K):
            log_b[k] = self._mvt_logpdf(x, self.mu[k], self.Sigma[k], self.nu[k])[0]
        # Filtered probability: P(z_t | x_{1:t})
        # = P(x_t | z_t) * sum_j P(z_t | z_{t-1}=j) * P(z_{t-1}=j | x_{1:t-1})
        log_pred = np.zeros(K)
        log_A = np.log(self.A + 1e-300)
        for k in range(K):
            log_pred[k] = np.logaddexp.reduce(np.log(prev_state_prob + 1e-300) + log_A[:, k])
        log_filtered = log_b + log_pred
        log_filtered = log_filtered - np.logaddexp.reduce(log_filtered)
        return np.exp(log_filtered)


# =============================================================================
# ONLINE HMM WITH INCREMENTAL UPDATES
# =============================================================================

class OnlineStudentTHMM:
    """
    Online Student-t HMM with incremental parameter updates.

    Uses stochastic EM / exponential forgetting for streaming updates.
    """

    def __init__(self, n_regimes=3, learning_rate=0.01, forgetting_factor=0.995):
        self.n_regimes = n_regimes
        self.learning_rate = learning_rate
        self.forgetting_factor = forgetting_factor
        self.mu = None
        self.Sigma = None
        self.nu = None
        self.A = None
        self.pi = None
        self.prev_state_prob = None
        self.n_obs = 0
        # Sufficient statistics for incremental updates
        self.sum_gamma = None
        self.sum_gamma_x = None
        self.sum_gamma_xx = None
        self.sum_xi = None

    def initialize(self, X_init):
        """Initialize parameters using initial batch of data."""
        batch_model = StudentTHMM(n_regimes=self.n_regimes, n_iter=50, random_state=42)
        batch_model.fit(X_init, verbose=False)

        self.mu = batch_model.mu.copy()
        self.Sigma = batch_model.Sigma.copy()
        self.nu = batch_model.nu.copy()
        self.A = batch_model.A.copy()
        self.pi = batch_model.pi.copy()
        self.prev_state_prob = self.pi.copy()
        self.n_obs = len(X_init)

        # Initialize sufficient statistics
        K = self.n_regimes
        d = X_init.shape[1]
        self.sum_gamma = np.ones(K) * self.n_obs / K
        self.sum_gamma_x = np.zeros((K, d))
        self.sum_gamma_xx = np.zeros((K, d, d))
        for k in range(K):
            self.sum_gamma_x[k] = self.mu[k] * self.sum_gamma[k]
            self.sum_gamma_xx[k] = (self.Sigma[k] + np.outer(self.mu[k], self.mu[k])) * self.sum_gamma[k]
        self.sum_xi = self.A * self.sum_gamma[:, None]

    def _mvt_logpdf(self, x, mu, Sigma, nu):
        d = len(mu)
        x = x.reshape(1, -1) if x.ndim == 1 else x
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

    def predict_and_update(self, x):
        """
        Predict regime for new observation and update parameters.

        Returns:
            regime: Predicted regime (argmax of filtered probability)
            proba: Filtered probability distribution over regimes
        """
        x = np.asarray(x).flatten()
        K = self.n_regimes
        d = len(x)

        # === E-step: Compute filtered probability ===
        log_b = np.zeros(K)
        for k in range(K):
            log_b[k] = self._mvt_logpdf(x, self.mu[k], self.Sigma[k], self.nu[k])[0]

        # P(z_t | x_{1:t}) via filtering
        log_A = np.log(self.A + 1e-300)
        log_pred = np.zeros(K)
        for k in range(K):
            log_pred[k] = np.logaddexp.reduce(np.log(self.prev_state_prob + 1e-300) + log_A[:, k])

        log_filtered = log_b + log_pred
        log_filtered = log_filtered - np.logaddexp.reduce(log_filtered)
        gamma_t = np.exp(log_filtered)

        # Compute pairwise posterior for transition update
        xi_t = np.zeros((K, K))
        for j in range(K):
            for k in range(K):
                xi_t[j, k] = self.prev_state_prob[j] * self.A[j, k] * np.exp(log_b[k])
        xi_t = xi_t / (xi_t.sum() + 1e-300)

        # Compute auxiliary variable u for Student-t
        u_t = np.zeros(K)
        for k in range(K):
            diff = x - self.mu[k]
            Sigma_inv = np.linalg.inv(self.Sigma[k])
            mahal = diff @ Sigma_inv @ diff
            u_t[k] = (self.nu[k] + d) / (self.nu[k] + mahal)

        # === M-step: Incremental parameter update ===
        ff = self.forgetting_factor
        lr = self.learning_rate

        # Update sufficient statistics with forgetting
        self.sum_gamma = ff * self.sum_gamma + gamma_t
        self.sum_gamma_x = ff * self.sum_gamma_x + gamma_t[:, None] * u_t[:, None] * x
        for k in range(K):
            self.sum_gamma_xx[k] = ff * self.sum_gamma_xx[k] + gamma_t[k] * u_t[k] * np.outer(x, x)
        self.sum_xi = ff * self.sum_xi + xi_t

        # Update means
        for k in range(K):
            if self.sum_gamma[k] > 1e-6:
                new_mu = self.sum_gamma_x[k] / (self.sum_gamma[k] * u_t[k] + 1e-6)
                self.mu[k] = (1 - lr) * self.mu[k] + lr * new_mu

        # Update covariances
        for k in range(K):
            if self.sum_gamma[k] > 1e-6:
                new_Sigma = self.sum_gamma_xx[k] / self.sum_gamma[k] - np.outer(self.mu[k], self.mu[k])
                new_Sigma = np.maximum(new_Sigma, 1e-6 * np.eye(d))  # Ensure positive definite
                self.Sigma[k] = (1 - lr) * self.Sigma[k] + lr * new_Sigma
                self.Sigma[k] += 1e-6 * np.eye(d)

        # Update transition matrix
        for j in range(K):
            row_sum = self.sum_xi[j].sum()
            if row_sum > 1e-6:
                self.A[j] = self.sum_xi[j] / row_sum

        # Update state for next iteration
        self.prev_state_prob = gamma_t.copy()
        self.n_obs += 1

        return np.argmax(gamma_t), gamma_t


# =============================================================================
# ROLLING WINDOW HMM
# =============================================================================

class RollingWindowHMM:
    """HMM that re-fits periodically on a rolling window."""

    def __init__(self, n_regimes=3, window_size=252, refit_interval=21):
        self.n_regimes = n_regimes
        self.window_size = window_size
        self.refit_interval = refit_interval
        self.model = None
        self.buffer = []
        self.days_since_refit = 0
        self.prev_state_prob = None

    def initialize(self, X_init):
        """Initialize with initial data."""
        self.buffer = list(X_init)
        self.model = StudentTHMM(n_regimes=self.n_regimes, n_iter=50, random_state=42)
        self.model.fit(np.array(self.buffer), verbose=False)
        self.prev_state_prob = self.model.pi.copy()

    def predict_and_update(self, x):
        """Predict regime and potentially refit model."""
        x = np.asarray(x).flatten()

        # Add to buffer
        self.buffer.append(x)
        if len(self.buffer) > self.window_size:
            self.buffer = self.buffer[-self.window_size:]

        self.days_since_refit += 1

        # Refit if needed
        if self.days_since_refit >= self.refit_interval:
            self.model = StudentTHMM(n_regimes=self.n_regimes, n_iter=30, random_state=42)
            self.model.fit(np.array(self.buffer), verbose=False)
            self.days_since_refit = 0

        # Predict using current model
        proba = self.model.predict_single(x, self.prev_state_prob)
        self.prev_state_prob = proba.copy()

        return np.argmax(proba), proba


# =============================================================================
# VOLATILITY THRESHOLD BASELINE
# =============================================================================

class VolatilityThresholdDetector:
    """Simple baseline: detect regimes based on rolling volatility thresholds."""

    def __init__(self, window=60, normal_pct=50, crisis_pct=90):
        self.window = window
        self.normal_pct = normal_pct
        self.crisis_pct = crisis_pct
        self.buffer = []
        self.vol_history = []

    def initialize(self, X_init):
        """Initialize with historical data to compute thresholds."""
        self.buffer = list(X_init)
        # Compute rolling volatility on initialization data
        X = np.array(self.buffer)
        for i in range(self.window, len(X)):
            window_data = X[i-self.window:i]
            vol = np.std(window_data, axis=0).mean()
            self.vol_history.append(vol)

        # Set thresholds based on historical distribution
        self.normal_threshold = np.percentile(self.vol_history, self.normal_pct)
        self.crisis_threshold = np.percentile(self.vol_history, self.crisis_pct)

    def predict_and_update(self, x):
        """Predict regime based on recent volatility."""
        x = np.asarray(x).flatten()
        self.buffer.append(x)

        if len(self.buffer) < self.window:
            return 0, np.array([1.0, 0.0, 0.0])

        # Compute recent volatility
        recent = np.array(self.buffer[-self.window:])
        vol = np.std(recent, axis=0).mean()
        self.vol_history.append(vol)

        # Update thresholds with exponential moving average
        alpha = 0.01
        self.normal_threshold = (1 - alpha) * self.normal_threshold + alpha * np.percentile(self.vol_history[-500:], self.normal_pct)
        self.crisis_threshold = (1 - alpha) * self.crisis_threshold + alpha * np.percentile(self.vol_history[-500:], self.crisis_pct)

        # Classify
        if vol >= self.crisis_threshold:
            regime = 2
            proba = np.array([0.1, 0.2, 0.7])
        elif vol >= self.normal_threshold:
            regime = 1
            proba = np.array([0.2, 0.6, 0.2])
        else:
            regime = 0
            proba = np.array([0.7, 0.2, 0.1])

        return regime, proba


# =============================================================================
# METRICS AND EVALUATION
# =============================================================================

def align_labels(true_labels, pred_labels, n_classes=3):
    """
    Align predicted labels to true labels using Hungarian algorithm.

    This solves the label permutation problem common in unsupervised clustering.
    Returns remapped predicted labels that best match true labels.
    """
    # Build confusion matrix
    cm = confusion_matrix(true_labels, pred_labels, labels=list(range(n_classes)))

    # Use Hungarian algorithm to find optimal assignment
    # We want to maximize agreement, so negate for minimization
    row_ind, col_ind = linear_sum_assignment(-cm)

    # Create mapping from predicted to aligned labels
    mapping = {pred: true for pred, true in zip(col_ind, row_ind)}

    # Apply mapping
    aligned = np.array([mapping.get(p, p) for p in pred_labels])

    return aligned, mapping


def compute_detection_delays(batch_regimes, online_regimes, dates):
    """
    Compute detection delays for regime changes.

    Detection delay = days between batch HMM detecting change and online method detecting it.
    """
    delays = []
    regime_changes = []

    # Find batch regime changes
    for i in range(1, len(batch_regimes)):
        if batch_regimes[i] != batch_regimes[i-1]:
            batch_change_day = i
            from_regime = batch_regimes[i-1]
            to_regime = batch_regimes[i]

            # Find when online detected this change (first day it matched)
            delay = None
            for j in range(batch_change_day, min(batch_change_day + 60, len(online_regimes))):
                if online_regimes[j] == to_regime:
                    delay = j - batch_change_day
                    break

            if delay is not None:
                delays.append(delay)
                regime_changes.append({
                    'date': str(dates[batch_change_day].date()) if hasattr(dates[batch_change_day], 'date') else str(dates[batch_change_day]),
                    'from_regime': int(from_regime),
                    'to_regime': int(to_regime),
                    'delay_days': int(delay)
                })

    return delays, regime_changes


def evaluate_model(batch_regimes, model_regimes, model_name):
    """Compute evaluation metrics for a model."""
    # Overall accuracy
    accuracy = accuracy_score(batch_regimes, model_regimes)

    # Per-regime F1
    f1_per_regime = f1_score(batch_regimes, model_regimes, average=None, labels=[0, 1, 2])
    f1_macro = f1_score(batch_regimes, model_regimes, average='macro')

    # Confusion matrix
    cm = confusion_matrix(batch_regimes, model_regimes, labels=[0, 1, 2])

    return {
        'accuracy': float(accuracy),
        'f1_macro': float(f1_macro),
        'f1_normal': float(f1_per_regime[0]) if len(f1_per_regime) > 0 else 0.0,
        'f1_elevated': float(f1_per_regime[1]) if len(f1_per_regime) > 1 else 0.0,
        'f1_crisis': float(f1_per_regime[2]) if len(f1_per_regime) > 2 else 0.0,
        'confusion_matrix': cm.tolist()
    }


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_online_regime_detection():
    """Run the complete online regime detection experiment."""
    print("=" * 70)
    print("ONLINE REGIME DETECTION EXPERIMENT")
    print("=" * 70)

    # 1. Load data
    print("\n[1] Loading Fama-French data...")
    df = download_ff_data()
    X = df.values
    dates = df.index

    # 2. Fit batch HMM on full data (oracle)
    print("\n[2] Fitting Batch HMM (oracle) on full data...")
    batch_model = StudentTHMM(n_regimes=3, n_iter=100, random_state=42)
    batch_model.fit(X, verbose=True)
    batch_regimes = batch_model.predict(X)
    print(f"Batch HMM regime distribution: Normal={np.mean(batch_regimes==0)*100:.1f}%, "
          f"Elevated={np.mean(batch_regimes==1)*100:.1f}%, Crisis={np.mean(batch_regimes==2)*100:.1f}%")

    # 3. Initialize online models with first 2 years (1990-1991)
    init_end = 504  # ~2 years of trading days
    X_init = X[:init_end]
    X_online = X[init_end:]
    dates_online = dates[init_end:]
    batch_regimes_online = batch_regimes[init_end:]

    print(f"\n[3] Initializing models with {init_end} days (1990-1991)...")
    print(f"Online evaluation period: {dates_online[0].date()} to {dates_online[-1].date()} ({len(X_online)} days)")

    # Initialize models
    online_hmm = OnlineStudentTHMM(n_regimes=3, learning_rate=0.01, forgetting_factor=0.995)
    online_hmm.initialize(X_init)

    rolling_hmm = RollingWindowHMM(n_regimes=3, window_size=252, refit_interval=21)
    rolling_hmm.initialize(X_init)

    vol_detector = VolatilityThresholdDetector(window=60)
    vol_detector.initialize(X_init)

    # 4. Run online prediction
    print("\n[4] Running online prediction...")
    online_preds = []
    rolling_preds = []
    vol_preds = []

    n_total = len(X_online)
    report_interval = n_total // 10

    for i, x in enumerate(X_online):
        # Online HMM
        regime_online, _ = online_hmm.predict_and_update(x)
        online_preds.append(regime_online)

        # Rolling Window HMM
        regime_rolling, _ = rolling_hmm.predict_and_update(x)
        rolling_preds.append(regime_rolling)

        # Volatility Threshold
        regime_vol, _ = vol_detector.predict_and_update(x)
        vol_preds.append(regime_vol)

        if (i + 1) % report_interval == 0:
            print(f"  Processed {i+1}/{n_total} days ({(i+1)/n_total*100:.0f}%)")

    online_preds = np.array(online_preds)
    rolling_preds = np.array(rolling_preds)
    vol_preds = np.array(vol_preds)

    # 5. Align labels using Hungarian algorithm (fix label permutation)
    print("\n[5] Aligning labels (Hungarian algorithm)...")
    online_preds_aligned, online_mapping = align_labels(batch_regimes_online, online_preds)
    rolling_preds_aligned, rolling_mapping = align_labels(batch_regimes_online, rolling_preds)
    vol_preds_aligned, vol_mapping = align_labels(batch_regimes_online, vol_preds)

    print(f"  Online HMM label mapping: {online_mapping}")
    print(f"  Rolling HMM label mapping: {rolling_mapping}")
    print(f"  Volatility label mapping: {vol_mapping}")

    # 6. Evaluate models
    print("\n[6] Evaluating models...")

    results = {
        'experiment': 'online_regime_detection',
        'timestamp': datetime.now().isoformat(),
        'data': {
            'total_days': len(df),
            'init_days': init_end,
            'eval_days': len(X_online),
            'start_date': str(dates[0].date()),
            'end_date': str(dates[-1].date()),
            'eval_start_date': str(dates_online[0].date())
        },
        'batch_hmm': {
            'regime_distribution': {
                'normal': float(np.mean(batch_regimes == 0)),
                'elevated': float(np.mean(batch_regimes == 1)),
                'crisis': float(np.mean(batch_regimes == 2))
            },
            'nu_values': batch_model.nu.tolist(),
            'transition_matrix': batch_model.A.tolist()
        },
        'models': {}
    }

    # Evaluate each model (now using aligned labels)
    models = {
        'online_hmm': (online_preds_aligned, online_mapping),
        'rolling_hmm': (rolling_preds_aligned, rolling_mapping),
        'volatility_threshold': (vol_preds_aligned, vol_mapping)
    }

    for model_name, (preds, mapping) in models.items():
        print(f"\n  {model_name}:")

        # Basic metrics (using aligned labels)
        metrics = evaluate_model(batch_regimes_online, preds, model_name)

        # Detection delays (using aligned labels)
        delays, changes = compute_detection_delays(batch_regimes_online, preds, dates_online)

        avg_delay = np.mean(delays) if delays else float('nan')
        median_delay = np.median(delays) if delays else float('nan')

        print(f"    Accuracy: {metrics['accuracy']*100:.1f}%")
        print(f"    F1 (macro): {metrics['f1_macro']:.3f}")
        print(f"    Avg detection delay: {avg_delay:.1f} days")
        print(f"    Median detection delay: {median_delay:.1f} days")

        results['models'][model_name] = {
            'metrics': metrics,
            'label_mapping': {str(k): int(v) for k, v in mapping.items()},
            'detection_delay': {
                'mean': float(avg_delay) if not np.isnan(avg_delay) else None,
                'median': float(median_delay) if not np.isnan(median_delay) else None,
                'std': float(np.std(delays)) if delays else None,
                'min': int(min(delays)) if delays else None,
                'max': int(max(delays)) if delays else None,
                'n_changes': len(delays)
            },
            'regime_changes': changes[:20]  # Store first 20 for brevity
        }

    # 7. Identify fastest/slowest detected regime changes
    print("\n[7] Analyzing detection speed by regime transition type...")

    # Aggregate delays by transition type
    transition_delays = {}
    for model_name in ['online_hmm', 'rolling_hmm']:
        for change in results['models'][model_name]['regime_changes']:
            key = f"{REGIME_NAMES[change['from_regime']]} -> {REGIME_NAMES[change['to_regime']]}"
            if key not in transition_delays:
                transition_delays[key] = {'online_hmm': [], 'rolling_hmm': []}
            transition_delays[key][model_name].append(change['delay_days'])

    print("\n  Average delay by transition type (Online HMM):")
    transition_summary = {}
    for trans, delays_dict in sorted(transition_delays.items()):
        online_delays = delays_dict['online_hmm']
        if online_delays:
            avg = np.mean(online_delays)
            print(f"    {trans}: {avg:.1f} days (n={len(online_delays)})")
            transition_summary[trans] = {
                'online_hmm_avg': float(avg),
                'online_hmm_n': len(online_delays)
            }

    results['transition_analysis'] = transition_summary

    # 8. Key crisis periods analysis
    print("\n[8] Crisis period detection analysis...")

    crisis_periods = [
        ('2008-09-01', '2008-12-31', 'Global Financial Crisis'),
        ('2011-08-01', '2011-08-31', 'US Debt Downgrade'),
        ('2015-08-15', '2015-09-15', 'China Devaluation'),
        ('2018-12-01', '2018-12-31', 'Dec 2018 Selloff'),
        ('2020-02-20', '2020-04-30', 'COVID-19 Crash'),
        ('2022-01-01', '2022-06-30', '2022 Bear Market')
    ]

    crisis_results = []
    for start, end, name in crisis_periods:
        try:
            mask = (dates_online >= start) & (dates_online <= end)
            if mask.sum() > 0:
                batch_crisis = batch_regimes_online[mask]
                online_crisis = online_preds_aligned[mask]
                rolling_crisis = rolling_preds_aligned[mask]
                vol_crisis = vol_preds_aligned[mask]

                batch_pct = (batch_crisis == 2).mean() * 100
                online_acc = (online_crisis == batch_crisis).mean() * 100
                rolling_acc = (rolling_crisis == batch_crisis).mean() * 100
                vol_acc = (vol_crisis == batch_crisis).mean() * 100

                print(f"  {name}:")
                print(f"    Batch Crisis%: {batch_pct:.0f}%, Online acc: {online_acc:.0f}%, "
                      f"Rolling acc: {rolling_acc:.0f}%, Vol acc: {vol_acc:.0f}%")

                crisis_results.append({
                    'period': name,
                    'start': start,
                    'end': end,
                    'batch_crisis_pct': float(batch_pct),
                    'online_accuracy': float(online_acc),
                    'rolling_accuracy': float(rolling_acc),
                    'volatility_accuracy': float(vol_acc)
                })
        except Exception as e:
            print(f"  Skipped {name}: {e}")

    results['crisis_periods'] = crisis_results

    # 9. Save results
    output_path = os.path.join(RESULTS_DIR, 'online_regime.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[9] Results saved to {output_path}")

    # 10. Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\nModel Comparison:")
    print(f"{'Model':<25} {'Accuracy':<12} {'F1 Macro':<12} {'Avg Delay':<12}")
    print("-" * 61)
    for model_name, data in results['models'].items():
        acc = data['metrics']['accuracy'] * 100
        f1 = data['metrics']['f1_macro']
        delay = data['detection_delay']['mean']
        delay_str = f"{delay:.1f} days" if delay else "N/A"
        print(f"{model_name:<25} {acc:>6.1f}%     {f1:>6.3f}       {delay_str:<12}")

    print("\nKey Findings:")
    online_data = results['models']['online_hmm']
    rolling_data = results['models']['rolling_hmm']
    vol_data = results['models']['volatility_threshold']

    print(f"  - Online HMM achieves {online_data['metrics']['accuracy']*100:.1f}% accuracy vs batch oracle")
    print(f"  - Online HMM average detection delay: {online_data['detection_delay']['mean']:.1f} days")
    print(f"  - Rolling HMM average detection delay: {rolling_data['detection_delay']['mean']:.1f} days")
    print(f"  - Volatility baseline average delay: {vol_data['detection_delay']['mean']:.1f} days")

    # Fastest/slowest transitions for rolling HMM (more regime changes detected)
    print("\nFastest detected transitions (Rolling HMM):")
    fast_transitions = [c for c in results['models']['rolling_hmm']['regime_changes'] if c['delay_days'] == 0][:5]
    for t in fast_transitions:
        print(f"  {t['date']}: {REGIME_NAMES[t['from_regime']]} -> {REGIME_NAMES[t['to_regime']]} (0 days)")

    print("\nSlowest detected transitions (Rolling HMM):")
    slow_transitions = sorted(results['models']['rolling_hmm']['regime_changes'],
                             key=lambda x: x['delay_days'], reverse=True)[:5]
    for t in slow_transitions:
        print(f"  {t['date']}: {REGIME_NAMES[t['from_regime']]} -> {REGIME_NAMES[t['to_regime']]} ({t['delay_days']} days)")

    return results


if __name__ == "__main__":
    results = run_online_regime_detection()
