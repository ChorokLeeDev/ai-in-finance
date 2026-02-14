"""
Hybrid Regime Detector: Frozen HMM + Realized Volatility Fallback
=================================================================

Addresses the critical failure where the frozen HMM classified 0% of COVID-2020
as Crisis. Combines:
  1. Frozen HMM filtered probabilities (trained 1990-2012)
  2. 20-day realized volatility fallback (95th percentile of training period)

When realized vol > threshold AND HMM says non-Crisis => override to "Stress Alert"

All thresholds locked before test evaluation (no tuning on 2013-2024).

Outputs:
  - results/hybrid_detector_results.json
  - figures/hybrid_detector_performance.pdf (3-panel)
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
if BASE_DIR == '':
    BASE_DIR = '/Users/i767700/Github/ai-in-finance/papers/causal_regimes'
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# =============================================================================
# DATA LOADING
# =============================================================================

def download_ff_data():
    """Download Fama-French 5 factors daily data, filter to 1990-2024."""
    print("Downloading Fama-French 5 factors (daily)...")
    url = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip'
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        csv_name = z.namelist()[0]
        with z.open(csv_name) as f:
            df = pd.read_csv(f, skiprows=3)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={df.columns[0]: 'Date'})
    df = df[df['Date'].astype(str).str.match(r'^\d{8}$')]
    df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
    for col in ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['Mkt-RF', 'SMB', 'HML'])
    df = df.set_index('Date').sort_index()
    df = df.loc['1990-01-01':'2024-12-31']
    print(f"  Loaded {len(df)} trading days ({df.index[0].date()} to {df.index[-1].date()})")
    return df


# =============================================================================
# STUDENT-T HMM (from critical_fixes_analysis.py)
# =============================================================================

class StudentTHMM:
    def __init__(self, n_regimes=3, n_iter=100, tol=1e-4, random_state=42):
        self.n_regimes = n_regimes
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state
        self.mu = None; self.Sigma = None; self.nu = None
        self.A = None; self.pi = None
        self.gamma = None; self.alpha = None
        self.log_likelihood_ = None

    def _init_params(self, X):
        np.random.seed(self.random_state)
        T, d = X.shape; K = self.n_regimes
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
        if x.ndim == 1: x = x.reshape(1, -1)
        diff = x - mu
        Sigma_inv = np.linalg.inv(Sigma)
        mahal = np.sum(diff @ Sigma_inv * diff, axis=1)
        sign, logdet = np.linalg.slogdet(Sigma)
        return (gammaln((nu + d) / 2) - gammaln(nu / 2)
                - 0.5 * d * np.log(nu * np.pi) - 0.5 * logdet
                - 0.5 * (nu + d) * np.log(1 + mahal / nu))

    def _compute_emission_probs(self, X):
        T, d = X.shape; K = self.n_regimes
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
                log_alpha[t, k] = np.logaddexp.reduce(log_alpha[t-1] + log_A[:, k]) + log_B[t, k]
        return log_alpha

    def _backward(self, log_B):
        T, K = log_B.shape
        log_beta = np.zeros((T, K))
        log_A = np.log(self.A + 1e-300)
        for t in range(T - 2, -1, -1):
            for k in range(K):
                log_beta[t, k] = np.logaddexp.reduce(log_A[k, :] + log_B[t+1, :] + log_beta[t+1, :])
        return log_beta

    def _e_step(self, X):
        T, d = X.shape; K = self.n_regimes
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
                        log_alpha[t, j] + log_A[j, k] + log_B[t+1, k] + log_beta[t+1, k] - log_likelihood)
        self.u = np.zeros((T, K))
        for k in range(K):
            diff = X - self.mu[k]
            Sigma_inv = np.linalg.inv(self.Sigma[k])
            mahal = np.sum(diff @ Sigma_inv * diff, axis=1)
            self.u[:, k] = (self.nu[k] + d) / (self.nu[k] + mahal)
        return log_likelihood

    def _m_step(self, X):
        T, d = X.shape; K = self.n_regimes
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
            if nu <= 2: return 1e10
            diff = X - self.mu[k]
            Sigma_inv = np.linalg.inv(self.Sigma[k])
            mahal = np.sum(diff @ Sigma_inv * diff, axis=1)
            term1 = gammaln((nu + d) / 2) - gammaln(nu / 2)
            term2 = -0.5 * d * np.log(nu)
            term3 = -0.5 * (nu + d) * np.log(1 + mahal / nu)
            return -(self.gamma[:, k] * (term1 + term2 + term3)).sum()
        result = minimize_scalar(neg_expected_ll, bounds=(2.1, 50), method='bounded')
        self.nu[k] = result.x

    def _enforce_ordering(self):
        norms = np.linalg.norm(self.mu, axis=1)
        order = np.argsort(norms)
        if not np.array_equal(order, np.arange(self.n_regimes)):
            self.mu = self.mu[order]; self.Sigma = self.Sigma[order]
            self.nu = self.nu[order]; self.A = self.A[order][:, order]
            self.pi = self.pi[order]; self.gamma = self.gamma[:, order]
            if self.alpha is not None: self.alpha = self.alpha[:, order]
            if self.xi is not None: self.xi = self.xi[:, order, :][:, :, order]

    def fit(self, X):
        X = np.asarray(X); self._init_params(X)
        prev_ll = -np.inf
        for iteration in range(self.n_iter):
            log_likelihood = self._e_step(X)
            self._m_step(X)
            if abs(log_likelihood - prev_ll) < self.tol:
                print(f"  HMM converged at iteration {iteration + 1}"); break
            prev_ll = log_likelihood
        self.log_likelihood_ = log_likelihood
        return self

    def predict_oos(self, X, use_filtered=False):
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
# CHRISTOFFERSEN TEST
# =============================================================================

def christoffersen_test(violations):
    hits = np.array(violations, dtype=int)
    T = len(hits); n1 = hits.sum(); n0 = T - n1
    pi_hat = n1 / T if T > 0 else 0; alpha = 0.05
    if n1 == 0 or n0 == 0:
        LR_uc = np.nan; p_uc = np.nan
    else:
        LR_uc = -2 * (n1 * np.log(alpha) + n0 * np.log(1 - alpha)
                       - n1 * np.log(pi_hat) - n0 * np.log(1 - pi_hat))
        p_uc = 1 - stats.chi2.cdf(LR_uc, 1)
    n00 = n01 = n10 = n11 = 0
    for t in range(1, T):
        if hits[t-1] == 0 and hits[t] == 0: n00 += 1
        elif hits[t-1] == 0 and hits[t] == 1: n01 += 1
        elif hits[t-1] == 1 and hits[t] == 0: n10 += 1
        elif hits[t-1] == 1 and hits[t] == 1: n11 += 1
    if (n00+n01) == 0 or (n10+n11) == 0 or n01 == 0 or n10 == 0:
        LR_ind = np.nan; p_ind = np.nan
    else:
        pi01 = n01/(n00+n01); pi11 = n11/(n10+n11)
        pi_hat2 = (n01+n11)/(n00+n01+n10+n11)
        if pi01 <= 0 or pi01 >= 1 or pi11 <= 0 or pi11 >= 1 or pi_hat2 <= 0 or pi_hat2 >= 1:
            LR_ind = np.nan; p_ind = np.nan
        else:
            LR_ind = -2 * ((n00+n10)*np.log(1-pi_hat2) + (n01+n11)*np.log(pi_hat2)
                           - n00*np.log(1-pi01) - n01*np.log(pi01)
                           - n10*np.log(1-pi11) - n11*np.log(pi11))
            p_ind = 1 - stats.chi2.cdf(LR_ind, 1)
    LR_cc = (LR_uc if not np.isnan(LR_uc) else 0) + (LR_ind if not np.isnan(LR_ind) else 0)
    p_cc = 1 - stats.chi2.cdf(LR_cc, 2) if not (np.isnan(LR_uc) and np.isnan(LR_ind)) else np.nan
    return {
        'LR_uc': float(LR_uc) if not np.isnan(LR_uc) else None,
        'p_uc': float(p_uc) if not np.isnan(p_uc) else None,
        'LR_ind': float(LR_ind) if not np.isnan(LR_ind) else None,
        'p_ind': float(p_ind) if not np.isnan(p_ind) else None,
        'LR_cc': float(LR_cc),
        'p_cc': float(p_cc) if not np.isnan(p_cc) else None,
    }


# =============================================================================
# VAR MODELS (from risk_monitoring_backtest.py)
# =============================================================================

def rolling_historical_var(returns, window=60, alpha=0.05):
    T = len(returns)
    var_est = np.full(T, np.nan)
    for t in range(window, T):
        var_est[t] = np.percentile(returns[t-window:t], alpha * 100)
    return var_est


def regime_conditional_var(returns, regimes, alpha=0.05,
                           window_calm=75, window_normal=50, window_crisis=30):
    T = len(returns)
    var_est = np.full(T, np.nan)
    windows = {0: window_calm, 1: window_normal, 2: window_crisis}
    max_w = max(window_calm, window_normal, window_crisis)
    for t in range(max_w, T):
        regime = regimes[t-1]
        w = windows.get(regime, window_normal)
        start = max(0, t - w)
        var_est[t] = np.percentile(returns[start:t], alpha * 100)
    return var_est


def hml_informed_var(returns, regimes, hml_cumul, alpha=0.05,
                     window_calm=75, window_normal=50, window_crisis=30,
                     hml_threshold=-0.5, stress_multiplier=2.3):
    T = len(returns)
    var_est = np.full(T, np.nan)
    windows = {0: window_calm, 1: window_normal, 2: window_crisis}
    stress_applied = np.zeros(T, dtype=bool)
    max_w = max(window_calm, window_normal, window_crisis)
    for t in range(max_w, T):
        regime = regimes[t-1]
        w = windows.get(regime, window_normal)
        start = max(0, t - w)
        base_var = np.percentile(returns[start:t], alpha * 100)
        if regime == 2 and hml_cumul[t-1] < hml_threshold:
            var_est[t] = base_var * stress_multiplier
            stress_applied[t] = True
        else:
            var_est[t] = base_var
    return var_est, stress_applied


def hybrid_var(returns, hmm_regimes, realized_vol, hml_cumul, vol_threshold,
               alpha=0.05, window_calm=75, window_normal=50, window_crisis=30,
               hml_threshold=-0.5, stress_multiplier=2.3):
    """
    Hybrid VaR: HML-Informed + Volatility Override.

    When realized vol > vol_threshold AND HMM says non-Crisis:
      override regime to Crisis (Stress Alert), apply stress multiplier.
    """
    T = len(returns)
    var_est = np.full(T, np.nan)
    windows = {0: window_calm, 1: window_normal, 2: window_crisis}
    stress_applied = np.zeros(T, dtype=bool)
    vol_override = np.zeros(T, dtype=bool)
    effective_regimes = hmm_regimes.copy()
    max_w = max(window_calm, window_normal, window_crisis)

    for t in range(max_w, T):
        regime = hmm_regimes[t-1]

        # Volatility override: if vol exceeds threshold and HMM says non-Crisis
        if regime != 2 and not np.isnan(realized_vol[t-1]) and realized_vol[t-1] > vol_threshold:
            regime = 2  # Override to Crisis
            vol_override[t] = True
            effective_regimes[t] = 2

        w = windows.get(regime, window_normal)
        start = max(0, t - w)
        base_var = np.percentile(returns[start:t], alpha * 100)

        # Apply HML stress in Crisis (whether original or overridden)
        if regime == 2 and not np.isnan(hml_cumul[t-1]) and hml_cumul[t-1] < hml_threshold:
            var_est[t] = base_var * stress_multiplier
            stress_applied[t] = True
        elif regime == 2 and vol_override[t]:
            # Vol override but HML not triggered: still use crisis window + modest multiplier
            var_est[t] = base_var * 1.5  # Moderate stress for vol-only override
            stress_applied[t] = True
        else:
            var_est[t] = base_var

    return var_est, stress_applied, vol_override, effective_regimes


def evaluate_var_model(returns, var_estimates, model_name):
    valid = ~np.isnan(var_estimates)
    ret = returns[valid]; var = var_estimates[valid]; T = len(ret)
    violations = ret < var
    n_violations = violations.sum()
    violation_rate = n_violations / T
    avg_violation_mag = np.mean(ret[violations] - var[violations]) if n_violations > 0 else 0.0
    cc_test = christoffersen_test(violations)
    return {
        'model': model_name,
        'n_days': int(T),
        'n_violations': int(n_violations),
        'violation_rate': round(float(violation_rate), 4),
        'violation_rate_pct': round(float(violation_rate * 100), 2),
        'target_rate_pct': 5.0,
        'deviation_from_target_pct': round(float((violation_rate - 0.05) * 100), 2),
        'avg_violation_magnitude': round(float(avg_violation_mag), 4),
        'avg_var_level': round(float(np.mean(var)), 4),
        'christoffersen_LR_cc': cc_test['LR_cc'],
        'christoffersen_p_cc': cc_test['p_cc'],
        'christoffersen_p_uc': cc_test['p_uc'],
        'christoffersen_p_ind': cc_test['p_ind'],
    }, violations


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():
    print("=" * 78)
    print("HYBRID REGIME DETECTOR + VaR BACKTEST")
    print("=" * 78)

    # ---- 1. Load data ----
    df = download_ff_data()
    factor_cols = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
    smb = df['SMB'].values
    hml = df['HML'].values
    dates = df.index
    hml_cumul_9d = pd.Series(hml, index=dates).rolling(9).sum().values

    # ---- 2. Train/Test Split ----
    train_end = pd.Timestamp('2012-12-31')
    test_start = pd.Timestamp('2013-01-01')
    train_mask = dates <= train_end
    test_mask = dates >= test_start

    print(f"\n  Train: {dates[train_mask][0].date()} to {dates[train_mask][-1].date()} ({train_mask.sum()} days)")
    print(f"  Test:  {dates[test_mask][0].date()} to {dates[test_mask][-1].date()} ({test_mask.sum()} days)")

    # ---- 3. Compute 20-day realized volatility ----
    print("\n  Computing 20-day realized volatility...")
    # Use 6-factor norm for realized vol (matches HMM input)
    factor_returns = df[factor_cols].values
    daily_vol = np.sqrt(np.sum(factor_returns**2, axis=1))  # Daily factor norm
    realized_vol_20d = pd.Series(daily_vol, index=dates).rolling(20).std().values

    # ---- 4. Calibrate vol threshold on TRAINING DATA ONLY ----
    train_vol = realized_vol_20d[train_mask]
    train_vol_clean = train_vol[~np.isnan(train_vol)]

    # Fixed design choice: 95th percentile. NO search.
    vol_threshold_95 = np.percentile(train_vol_clean, 95)
    print(f"  Vol threshold (95th pctile of training): {vol_threshold_95:.4f}")

    # Sensitivity table (for robustness reporting only)
    sensitivity = {}
    for pctile in [85, 90, 95, 99]:
        thresh = np.percentile(train_vol_clean, pctile)
        sensitivity[str(pctile)] = round(float(thresh), 4)
        print(f"    {pctile}th percentile: {thresh:.4f}")

    # ---- 5. Fit Student-t HMM on training data ----
    print("\n  Fitting Student-t HMM (K=3) on training data...")
    X_train = df.loc[train_mask, factor_cols].values
    hmm = StudentTHMM(n_regimes=3, n_iter=100, tol=1e-4, random_state=42)
    hmm.fit(X_train)

    regime_names = {0: 'Calm', 1: 'Normal', 2: 'Crisis'}
    for k in range(3):
        print(f"    Regime {k} ({regime_names[k]}): mu_norm={np.linalg.norm(hmm.mu[k]):.3f}, nu={hmm.nu[k]:.1f}")

    # ---- 6. Get regimes for full data (frozen, filtered) ----
    X_all = df[factor_cols].values
    regimes_all, probs_all = hmm.predict_oos(X_all, use_filtered=True)

    # ---- 7. Load calibrated params from existing risk_monitoring_results.json ----
    # These were already calibrated on training data
    cal_params = {
        'hml_threshold': -0.5,
        'stress_multiplier': 2.3,
        'windows': {'calm': 75, 'normal': 50, 'crisis': 30}
    }
    try:
        with open(os.path.join(RESULTS_DIR, 'risk_monitoring_results.json'), 'r') as f:
            existing = json.load(f)
        cal_params['hml_threshold'] = existing['calibrated_params']['hml_threshold']
        cal_params['stress_multiplier'] = existing['calibrated_params']['stress_multiplier']
        cal_params['windows']['calm'] = existing['calibrated_params']['regime_windows']['calm']
        cal_params['windows']['normal'] = existing['calibrated_params']['regime_windows']['normal']
        cal_params['windows']['crisis'] = existing['calibrated_params']['regime_windows']['crisis']
        print(f"\n  Loaded calibrated params from existing results:")
    except Exception:
        print(f"\n  Using default calibrated params:")
    print(f"    HML threshold: {cal_params['hml_threshold']}")
    print(f"    Stress multiplier: {cal_params['stress_multiplier']}")
    print(f"    Windows: {cal_params['windows']}")

    # ---- 8. Compute all VaR models on test data ----
    print("\n  Computing VaR models on test period...")
    test_smb = smb[test_mask]
    test_regimes = regimes_all[test_mask]
    test_hml_cumul = hml_cumul_9d[test_mask]
    test_vol = realized_vol_20d[test_mask]
    test_dates = dates[test_mask]

    w = cal_params['windows']

    # Model 1: Unconditional
    var_uncond = rolling_historical_var(test_smb, window=60, alpha=0.05)

    # Model 2: HML-Informed (existing best model)
    var_hml, stress_hml = hml_informed_var(
        test_smb, test_regimes, test_hml_cumul, alpha=0.05,
        window_calm=w['calm'], window_normal=w['normal'], window_crisis=w['crisis'],
        hml_threshold=cal_params['hml_threshold'],
        stress_multiplier=cal_params['stress_multiplier'])

    # Model 3: HYBRID (new) - uses vol_threshold_95
    var_hybrid, stress_hybrid, vol_override, eff_regimes = hybrid_var(
        test_smb, test_regimes, test_vol, test_hml_cumul, vol_threshold_95,
        alpha=0.05,
        window_calm=w['calm'], window_normal=w['normal'], window_crisis=w['crisis'],
        hml_threshold=cal_params['hml_threshold'],
        stress_multiplier=cal_params['stress_multiplier'])

    # ---- 9. Evaluate ----
    print("\n" + "=" * 78)
    print("VaR BACKTEST RESULTS (2013-2024)")
    print("=" * 78)

    eval_uncond, viol_uncond = evaluate_var_model(test_smb, var_uncond, "Unconditional")
    eval_hml, viol_hml = evaluate_var_model(test_smb, var_hml, "HML-Informed")
    eval_hybrid, viol_hybrid = evaluate_var_model(test_smb, var_hybrid, "Hybrid (HMM+Vol)")

    all_evals = [eval_uncond, eval_hml, eval_hybrid]

    print(f"\n  {'Model':<24} {'Viol%':>7} {'Dev':>8} {'CC p':>8} {'UC p':>8} {'Ind p':>8}")
    print("  " + "-" * 67)
    for ev in all_evals:
        p_cc = ev['christoffersen_p_cc']
        p_uc = ev['christoffersen_p_uc']
        p_ind = ev['christoffersen_p_ind']
        print(f"  {ev['model']:<24} {ev['violation_rate_pct']:>6.2f}% {ev['deviation_from_target_pct']:>+7.2f}% "
              f"{p_cc if p_cc else 'N/A':>8} {p_uc if p_uc else 'N/A':>8} {p_ind if p_ind else 'N/A':>8}")

    # ---- 10. COVID detection analysis ----
    print("\n" + "=" * 78)
    print("COVID DETECTION ANALYSIS")
    print("=" * 78)

    covid_start = pd.Timestamp('2020-02-20')
    covid_end = pd.Timestamp('2020-06-30')
    covid_mask = (test_dates >= covid_start) & (test_dates <= covid_end)
    n_covid = covid_mask.sum()

    covid_hmm_regimes = test_regimes[covid_mask]
    covid_vol_override = vol_override[covid_mask]
    covid_eff_regimes = eff_regimes[covid_mask]
    covid_stress = stress_hybrid[covid_mask]
    covid_vol = test_vol[covid_mask]

    hmm_crisis_days = (covid_hmm_regimes == 2).sum()
    vol_override_days = covid_vol_override.sum()
    hybrid_crisis_days = (covid_eff_regimes == 2).sum()
    covid_detection_rate = hybrid_crisis_days / n_covid * 100

    print(f"\n  COVID window: {covid_start.date()} to {covid_end.date()} ({n_covid} trading days)")
    print(f"  HMM-only Crisis days:    {hmm_crisis_days} ({hmm_crisis_days/n_covid*100:.1f}%)")
    print(f"  Vol override days:       {vol_override_days} ({vol_override_days/n_covid*100:.1f}%)")
    print(f"  Hybrid Crisis/Alert days: {hybrid_crisis_days} ({covid_detection_rate:.1f}%)")
    print(f"  Stress multiplier active: {covid_stress.sum()} days")

    # Sensitivity: different vol thresholds
    print(f"\n  Sensitivity table (COVID detection rate by vol threshold):")
    sensitivity_detection = {}
    for pctile in [85, 90, 95, 99]:
        thresh = sensitivity[str(pctile)]
        override_days = (covid_vol > thresh).sum()
        # Also count HMM crisis days
        detect_days = ((covid_hmm_regimes == 2) | (covid_vol > thresh)).sum()
        detect_rate = detect_days / n_covid * 100
        sensitivity_detection[str(pctile)] = {
            'threshold': thresh,
            'override_days': int(override_days),
            'total_detected': int(detect_days),
            'detection_rate_pct': round(float(detect_rate), 1),
        }
        print(f"    {pctile}th pctile (thresh={thresh:.4f}): {detect_days}/{n_covid} = {detect_rate:.1f}%")

    # ---- 11. False alarm analysis (same definition as var_fixes.py:1112) ----
    print("\n" + "=" * 78)
    print("FALSE ALARM ANALYSIS")
    print("=" * 78)

    valid_hybrid = ~np.isnan(var_hybrid)
    alert_days_mask = stress_hybrid & valid_hybrid
    n_alert = alert_days_mask.sum()

    # False alarm = alert day where unconditional VaR would NOT have been violated
    valid_both = valid_hybrid & ~np.isnan(var_uncond)
    alert_both = stress_hybrid & valid_both

    if alert_both.sum() > 0:
        # Would unconditional VaR have been violated on alert days?
        uncond_would_violate = test_smb[alert_both] < var_uncond[alert_both]
        false_alarm_rate = (1 - uncond_would_violate.mean()) * 100
        print(f"  Alert days: {n_alert}")
        print(f"  Alert days (with valid unconditional): {alert_both.sum()}")
        print(f"  Of those, unconditional would violate: {uncond_would_violate.sum()}")
        print(f"  False alarm rate: {false_alarm_rate:.1f}%")
    else:
        false_alarm_rate = 0.0

    # Compare with old HML-Informed false alarm rate
    alert_hml_mask = stress_hml & ~np.isnan(var_hml) & ~np.isnan(var_uncond)
    if alert_hml_mask.sum() > 0:
        hml_would_violate = test_smb[alert_hml_mask] < var_uncond[alert_hml_mask]
        hml_false_alarm = (1 - hml_would_violate.mean()) * 100
        print(f"\n  For comparison, HML-Informed false alarm rate: {hml_false_alarm:.1f}%")

    # ---- 12. Generate 3-panel figure ----
    print("\n  Generating 3-panel figure...")
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        # Panel 1: Regime classification comparison (COVID zoom)
        ax1 = axes[0]
        zoom_start = pd.Timestamp('2019-07-01')
        zoom_end = pd.Timestamp('2021-06-30')
        zoom_mask = (test_dates >= zoom_start) & (test_dates <= zoom_end)
        zoom_dates = test_dates[zoom_mask]
        zoom_hmm = test_regimes[zoom_mask]
        zoom_eff = eff_regimes[zoom_mask]
        zoom_vol = test_vol[zoom_mask]

        colors_hmm = ['#2ecc71', '#f1c40f', '#e74c3c']
        for i, d in enumerate(zoom_dates):
            ax1.axvspan(d, d + pd.Timedelta(days=1), alpha=0.3,
                       color=colors_hmm[zoom_hmm[i]], linewidth=0)
        # Overlay vol override in blue
        for i, d in enumerate(zoom_dates):
            if vol_override[zoom_mask][i]:
                ax1.axvspan(d, d + pd.Timedelta(days=1), alpha=0.5,
                           color='#3498db', linewidth=0)
        ax1.plot(zoom_dates, zoom_vol, 'k-', linewidth=0.5, alpha=0.7, label='Realized Vol (20d)')
        ax1.axhline(y=vol_threshold_95, color='red', linestyle='--', linewidth=1,
                    label=f'95th pctile threshold ({vol_threshold_95:.3f})')
        ax1.axvspan(covid_start, covid_end, alpha=0.1, color='purple', label='COVID window')
        ax1.set_ylabel('Realized Volatility')
        ax1.set_title('Panel A: Regime Classification (HMM background + Vol Override in blue)')
        ax1.legend(loc='upper left', fontsize=8)

        # Panel 2: VaR comparison
        ax2 = axes[1]
        zoom_smb = test_smb[zoom_mask]
        zoom_uncond = var_uncond[zoom_mask]
        zoom_hml = var_hml[zoom_mask]
        zoom_hybrid = var_hybrid[zoom_mask]

        ax2.plot(zoom_dates, zoom_smb, 'k-', linewidth=0.3, alpha=0.5, label='SMB returns')
        ax2.plot(zoom_dates, zoom_uncond, 'b-', linewidth=1, alpha=0.7, label='Unconditional VaR')
        ax2.plot(zoom_dates, zoom_hml, 'g-', linewidth=1, alpha=0.7, label='HML-Informed VaR')
        ax2.plot(zoom_dates, zoom_hybrid, 'r-', linewidth=1.2, alpha=0.8, label='Hybrid VaR')
        ax2.axvspan(covid_start, covid_end, alpha=0.1, color='purple')
        ax2.set_ylabel('Return / VaR')
        ax2.set_title('Panel B: VaR Models During COVID Period')
        ax2.legend(loc='lower left', fontsize=8)

        # Panel 3: Cumulative violation comparison
        ax3 = axes[2]
        valid_common = ~np.isnan(var_uncond) & ~np.isnan(var_hybrid) & ~np.isnan(var_hml)
        cum_viol_uncond = np.cumsum(test_smb[valid_common] < var_uncond[valid_common])
        cum_viol_hml = np.cumsum(test_smb[valid_common] < var_hml[valid_common])
        cum_viol_hybrid = np.cumsum(test_smb[valid_common] < var_hybrid[valid_common])
        common_dates = test_dates[valid_common]
        n_common = valid_common.sum()
        target_line = np.arange(1, n_common + 1) * 0.05

        ax3.plot(common_dates, cum_viol_uncond, 'b-', linewidth=1, label='Unconditional')
        ax3.plot(common_dates, cum_viol_hml, 'g-', linewidth=1, label='HML-Informed')
        ax3.plot(common_dates, cum_viol_hybrid, 'r-', linewidth=1.2, label='Hybrid')
        ax3.plot(common_dates, target_line, 'k--', linewidth=0.8, alpha=0.5, label='5% target')
        ax3.set_ylabel('Cumulative Violations')
        ax3.set_title('Panel C: Cumulative VaR Violations (2013-2024)')
        ax3.legend(loc='upper left', fontsize=8)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        plt.tight_layout()
        fig_path = os.path.join(FIGURES_DIR, 'hybrid_detector_performance.pdf')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Figure saved to: {fig_path}")
    except ImportError as e:
        print(f"  Could not generate figure: {e}")

    # ---- 13. Save results ----
    results = {
        'description': 'Hybrid Regime Detector: Frozen HMM + Realized Volatility Fallback',
        'train_period': '1990-01-01 to 2012-12-31',
        'test_period': f'{test_dates[0].date()} to {test_dates[-1].date()}',
        'n_test_days': int(test_mask.sum()),
        'vol_threshold': {
            'percentile': 95,
            'value': round(float(vol_threshold_95), 4),
            'calibrated_on': 'training data only (1990-2012)',
            'sensitivity': sensitivity,
        },
        'inherited_params': cal_params,
        'hmm_params': {
            'n_regimes': 3,
            'factors_used': factor_cols,
            'nu': [round(float(v), 2) for v in hmm.nu],
        },
        'covid_detection': {
            'window': f'{covid_start.date()} to {covid_end.date()}',
            'n_days': int(n_covid),
            'hmm_only_crisis_days': int(hmm_crisis_days),
            'hmm_only_crisis_pct': round(float(hmm_crisis_days / n_covid * 100), 1),
            'vol_override_days': int(vol_override_days),
            'hybrid_crisis_alert_days': int(hybrid_crisis_days),
            'hybrid_detection_rate_pct': round(float(covid_detection_rate), 1),
            'sensitivity': sensitivity_detection,
        },
        'var_results': {
            'unconditional': eval_uncond,
            'hml_informed': eval_hml,
            'hybrid': eval_hybrid,
        },
        'false_alarm_analysis': {
            'hybrid_alert_days': int(n_alert),
            'hybrid_false_alarm_rate_pct': round(float(false_alarm_rate), 1),
        },
    }

    output_path = os.path.join(RESULTS_DIR, 'hybrid_detector_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to: {output_path}")

    # ---- Summary ----
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  COVID detection: HMM-only {hmm_crisis_days/n_covid*100:.0f}% -> Hybrid {covid_detection_rate:.0f}%")
    print(f"  VaR violation rate: Unconditional {eval_uncond['violation_rate_pct']:.2f}% | "
          f"HML-Informed {eval_hml['violation_rate_pct']:.2f}% | Hybrid {eval_hybrid['violation_rate_pct']:.2f}%")
    print(f"  Christoffersen CC p-value: Unconditional {eval_uncond['christoffersen_p_cc']} | "
          f"HML-Informed {eval_hml['christoffersen_p_cc']} | Hybrid {eval_hybrid['christoffersen_p_cc']}")
    print(f"  False alarm rate: {false_alarm_rate:.1f}%")
    print("\n  Done.")


if __name__ == '__main__':
    main()
