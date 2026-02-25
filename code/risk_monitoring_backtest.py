"""
Risk Monitoring Backtest: HML-Informed SMB Tail Risk Forecasting
================================================================

Demonstrates practical value of HML->SMB causal regime information
for improving Value-at-Risk forecasts during crises.

Addresses reviewer concern: "no practical application is demonstrated."

Three VaR models compared:
  1. Unconditional: Rolling 60-day historical VaR at 5%
  2. Regime-conditional: Different VaR windows per HMM regime
  3. HML-informed: Regime VaR + HML momentum stress multiplier in Crisis

Evaluation:
  - Violation rate (target: 5%)
  - Christoffersen (1998) conditional coverage test
  - Average violation magnitude (expected shortfall when breached)
  - Unnecessary conservative alerts (false alarms)

Train period: 1990-2012 (calibration)
Test period:  2013-2024 (out-of-sample)
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
from scipy.optimize import minimize_scalar, minimize
from scipy.cluster.vq import kmeans2
import warnings
warnings.filterwarnings('ignore')

RESULTS_DIR = '/Users/i767700/Github/ai-in-finance/papers/causal_regimes/results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# =============================================================================
# DATA LOADING
# =============================================================================

def download_ff_data():
    """Download Fama-French 5 factors daily data."""
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
    df = df[df.index >= '1963-07-01']
    print(f"  Loaded {len(df)} daily observations from {df.index[0].date()} to {df.index[-1].date()}")
    return df


# =============================================================================
# STUDENT-T HMM (from critical_fixes_analysis.py)
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
                print(f"  Converged at iteration {iteration + 1}")
                break
            prev_ll = log_likelihood
        self.log_likelihood_ = log_likelihood
        return self

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
# CHRISTOFFERSEN (1998) CONDITIONAL COVERAGE TEST
# =============================================================================

def christoffersen_test(violations):
    """
    Christoffersen (1998) test for conditional coverage of VaR.
    
    Tests two properties jointly:
      1. Unconditional coverage: violation rate = alpha
      2. Independence: violations are not clustered
    
    Returns dict with LR_uc, LR_ind, LR_cc statistics and p-values.
    """
    hits = np.array(violations, dtype=int)
    T = len(hits)
    n1 = hits.sum()
    n0 = T - n1
    
    # Unconditional coverage test
    pi_hat = n1 / T if T > 0 else 0
    alpha = 0.05  # nominal level
    
    if n1 == 0 or n0 == 0:
        LR_uc = np.nan
        p_uc = np.nan
    else:
        LR_uc = -2 * (n1 * np.log(alpha) + n0 * np.log(1 - alpha)
                       - n1 * np.log(pi_hat) - n0 * np.log(1 - pi_hat))
        p_uc = 1 - stats.chi2.cdf(LR_uc, 1)
    
    # Independence test: count transitions
    n00 = n01 = n10 = n11 = 0
    for t in range(1, T):
        if hits[t-1] == 0 and hits[t] == 0:
            n00 += 1
        elif hits[t-1] == 0 and hits[t] == 1:
            n01 += 1
        elif hits[t-1] == 1 and hits[t] == 0:
            n10 += 1
        elif hits[t-1] == 1 and hits[t] == 1:
            n11 += 1
    
    if (n00 + n01) == 0 or (n10 + n11) == 0 or n01 == 0 or n10 == 0:
        LR_ind = np.nan
        p_ind = np.nan
    else:
        pi01 = n01 / (n00 + n01)
        pi11 = n11 / (n10 + n11)
        pi_hat2 = (n01 + n11) / (n00 + n01 + n10 + n11)
        
        # Handle edge cases
        if pi01 <= 0 or pi01 >= 1 or pi11 <= 0 or pi11 >= 1 or pi_hat2 <= 0 or pi_hat2 >= 1:
            LR_ind = np.nan
            p_ind = np.nan
        else:
            LR_ind = -2 * (
                (n00 + n10) * np.log(1 - pi_hat2) + (n01 + n11) * np.log(pi_hat2)
                - n00 * np.log(1 - pi01) - n01 * np.log(pi01)
                - n10 * np.log(1 - pi11) - n11 * np.log(pi11)
            )
            p_ind = 1 - stats.chi2.cdf(LR_ind, 1)
    
    # Joint conditional coverage
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
# VAR MODELS
# =============================================================================

def rolling_historical_var(returns, window=60, alpha=0.05):
    """
    Rolling historical VaR at alpha level.
    Returns array of VaR estimates (negative numbers = losses).
    VaR[t] is the forecast for day t, using data from [t-window, t-1].
    """
    T = len(returns)
    var_estimates = np.full(T, np.nan)
    for t in range(window, T):
        var_estimates[t] = np.percentile(returns[t-window:t], alpha * 100)
    return var_estimates


def regime_conditional_var(returns, regimes, alpha=0.05, 
                           window_calm=60, window_normal=45, window_crisis=30):
    """
    Regime-conditional VaR: use shorter window in crisis (more reactive).
    """
    T = len(returns)
    var_estimates = np.full(T, np.nan)
    windows = {0: window_calm, 1: window_normal, 2: window_crisis}
    
    for t in range(max(window_calm, window_normal, window_crisis), T):
        regime = regimes[t-1]  # use previous day's regime (real-time)
        w = windows.get(regime, window_normal)
        start = max(0, t - w)
        var_estimates[t] = np.percentile(returns[start:t], alpha * 100)
    return var_estimates


def hml_informed_var(returns, regimes, hml_cumul, alpha=0.05,
                     window_calm=60, window_normal=45, window_crisis=30,
                     hml_threshold=-2.0, stress_multiplier=1.5):
    """
    HML-informed regime VaR:
    - Uses regime-conditional windows
    - In Crisis regime (2): if cumulative HML over past 9 days < threshold,
      applies stress multiplier to make VaR more conservative (more negative)
    """
    T = len(returns)
    var_estimates = np.full(T, np.nan)
    windows = {0: window_calm, 1: window_normal, 2: window_crisis}
    stress_applied = np.zeros(T, dtype=bool)
    
    max_w = max(window_calm, window_normal, window_crisis)
    for t in range(max_w, T):
        regime = regimes[t-1]
        w = windows.get(regime, window_normal)
        start = max(0, t - w)
        base_var = np.percentile(returns[start:t], alpha * 100)
        
        # Apply HML stress in crisis regime
        if regime == 2 and hml_cumul[t-1] < hml_threshold:
            var_estimates[t] = base_var * stress_multiplier  # more negative
            stress_applied[t] = True
        else:
            var_estimates[t] = base_var
    
    return var_estimates, stress_applied


# =============================================================================
# BACKTEST EVALUATION
# =============================================================================

def evaluate_var_model(returns, var_estimates, model_name):
    """Evaluate VaR model performance."""
    valid = ~np.isnan(var_estimates)
    ret = returns[valid]
    var = var_estimates[valid]
    T = len(ret)
    
    # Violations: days where actual return < VaR estimate
    violations = ret < var
    n_violations = violations.sum()
    violation_rate = n_violations / T
    
    # Average violation magnitude (expected shortfall when breached)
    if n_violations > 0:
        avg_violation_mag = np.mean(ret[violations] - var[violations])
    else:
        avg_violation_mag = 0.0
    
    # Christoffersen test
    cc_test = christoffersen_test(violations)
    
    # Average VaR level (how conservative)
    avg_var = np.mean(var)
    
    result = {
        'model': model_name,
        'n_days': int(T),
        'n_violations': int(n_violations),
        'violation_rate': round(float(violation_rate), 4),
        'violation_rate_pct': round(float(violation_rate * 100), 2),
        'target_rate_pct': 5.0,
        'deviation_from_target_pct': round(float((violation_rate - 0.05) * 100), 2),
        'avg_violation_magnitude': round(float(avg_violation_mag), 4),
        'avg_var_level': round(float(avg_var), 4),
        'christoffersen_LR_uc': cc_test['LR_uc'],
        'christoffersen_p_uc': cc_test['p_uc'],
        'christoffersen_LR_ind': cc_test['LR_ind'],
        'christoffersen_p_ind': cc_test['p_ind'],
        'christoffersen_LR_cc': cc_test['LR_cc'],
        'christoffersen_p_cc': cc_test['p_cc'],
    }
    return result, violations


def count_unnecessary_alerts(baseline_var, enhanced_var, returns):
    """
    Count days where enhanced model is MORE conservative than baseline
    but no violation occurred under either model.
    """
    valid = ~np.isnan(baseline_var) & ~np.isnan(enhanced_var)
    b_var = baseline_var[valid]
    e_var = enhanced_var[valid]
    ret = returns[valid]
    
    # Enhanced is more conservative (more negative VaR)
    more_conservative = e_var < b_var
    # No violation under baseline
    no_baseline_violation = ret >= b_var
    # Unnecessary alert: more conservative but didn't need to be
    unnecessary = more_conservative & no_baseline_violation
    
    return int(unnecessary.sum()), int(more_conservative.sum())


# =============================================================================
# CALIBRATION ON TRAINING DATA
# =============================================================================

def calibrate_hml_threshold(train_returns, train_regimes, train_hml_cumul, 
                            alpha=0.05):
    """
    Calibrate the HML cumulative threshold and stress multiplier
    on training data to minimize deviation from target violation rate.
    """
    print("\n  Calibrating HML threshold and stress multiplier on training data...")
    
    best_score = np.inf
    best_params = {'threshold': -2.0, 'multiplier': 1.5}
    
    thresholds = np.arange(-5.0, 0.0, 0.5)
    multipliers = np.arange(1.2, 2.5, 0.1)
    
    for thresh in thresholds:
        for mult in multipliers:
            var_est, _ = hml_informed_var(
                train_returns, train_regimes, train_hml_cumul,
                alpha=alpha, hml_threshold=thresh, stress_multiplier=mult
            )
            valid = ~np.isnan(var_est)
            if valid.sum() < 100:
                continue
            viol_rate = (train_returns[valid] < var_est[valid]).mean()
            # Score: deviation from 5% target + small penalty for being too conservative
            score = abs(viol_rate - alpha) + 0.1 * max(0, alpha - viol_rate)
            if score < best_score:
                best_score = score
                best_params = {'threshold': float(thresh), 'multiplier': float(mult)}
    
    print(f"    Best threshold: {best_params['threshold']:.1f}")
    print(f"    Best multiplier: {best_params['multiplier']:.1f}")
    return best_params


# =============================================================================
# CRISIS PERIOD ANALYSIS
# =============================================================================

def analyze_crisis_periods(returns, var_baseline, var_regime, var_hml, 
                           regimes, dates, stress_applied):
    """Analyze performance during known crisis periods."""
    crisis_periods = {
        'COVID-19 (2020)': ('2020-02-20', '2020-04-30'),
        'China Devaluation (2015)': ('2015-08-10', '2015-09-30'),
        'Volmageddon (2018)': ('2018-01-29', '2018-03-23'),
        'Taper Tantrum (2013)': ('2013-06-01', '2013-08-31'),
        'EU Debt Crisis (2015)': ('2015-06-15', '2015-07-15'),
    }
    
    results = {}
    for name, (start, end) in crisis_periods.items():
        mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
        if mask.sum() == 0:
            continue
        
        ret = returns[mask]
        vb = var_baseline[mask]
        vr = var_regime[mask]
        vh = var_hml[mask]
        sa = stress_applied[mask] if stress_applied is not None else np.zeros_like(mask[mask])
        reg = regimes[mask]
        
        valid_b = ~np.isnan(vb)
        valid_r = ~np.isnan(vr)
        valid_h = ~np.isnan(vh)
        
        crisis_result = {
            'n_days': int(mask.sum()),
            'pct_crisis_regime': round(float((reg == 2).mean() * 100), 1),
            'stress_days': int(sa.sum()),
        }
        
        if valid_b.sum() > 0:
            crisis_result['baseline_violations'] = int((ret[valid_b] < vb[valid_b]).sum())
            crisis_result['baseline_violation_rate'] = round(float((ret[valid_b] < vb[valid_b]).mean() * 100), 1)
        if valid_r.sum() > 0:
            crisis_result['regime_violations'] = int((ret[valid_r] < vr[valid_r]).sum())
            crisis_result['regime_violation_rate'] = round(float((ret[valid_r] < vr[valid_r]).mean() * 100), 1)
        if valid_h.sum() > 0:
            crisis_result['hml_informed_violations'] = int((ret[valid_h] < vh[valid_h]).sum())
            crisis_result['hml_informed_violation_rate'] = round(float((ret[valid_h] < vh[valid_h]).mean() * 100), 1)
        
        results[name] = crisis_result
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 72)
    print("RISK MONITORING BACKTEST: HML-INFORMED SMB TAIL RISK FORECASTING")
    print("=" * 72)
    
    # ---- 1. Load data ----
    df = download_ff_data()
    
    # Use SMB and HML (the causal pair from the paper)
    smb = df['SMB'].values
    hml = df['HML'].values
    dates = df.index
    
    # Compute cumulative HML over past 9 days (HML momentum signal)
    hml_cumul_9d = pd.Series(hml, index=dates).rolling(9).sum().values
    
    # ---- 2. Train/Test Split ----
    train_end = pd.Timestamp('2012-12-31')
    test_start = pd.Timestamp('2013-01-01')
    
    train_mask = dates <= train_end
    test_mask = dates >= test_start
    
    # Require data from 1990+
    full_mask = dates >= pd.Timestamp('1990-01-01')
    train_mask = train_mask & full_mask
    
    print(f"\n  Training: {dates[train_mask][0].date()} to {dates[train_mask][-1].date()} "
          f"({train_mask.sum()} days)")
    print(f"  Testing:  {dates[test_mask][0].date()} to {dates[test_mask][-1].date()} "
          f"({test_mask.sum()} days)")
    
    # ---- 3. Fit Student-t HMM on training data ----
    print("\n  Fitting Student-t HMM (3 regimes) on training data...")
    # Use all 5 factors for regime detection (as in the paper)
    factor_cols = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
    X_train = df.loc[train_mask, factor_cols].values
    
    hmm = StudentTHMM(n_regimes=3, n_iter=100, tol=1e-4, random_state=42)
    hmm.fit(X_train)
    
    regime_names = {0: 'Calm', 1: 'Normal', 2: 'Crisis'}
    for k in range(3):
        print(f"    Regime {k} ({regime_names[k]}): mu_norm={np.linalg.norm(hmm.mu[k]):.3f}, "
              f"nu={hmm.nu[k]:.1f}")
    
    # ---- 4. Get regimes for ALL data (frozen model, forward-filtered) ----
    print("\n  Computing out-of-sample regimes (frozen model, filtered)...")
    X_full = df.loc[full_mask | test_mask, factor_cols].values
    full_dates = dates[full_mask | test_mask]
    
    # We need continuous sequence from 1990 onwards
    combined_mask = dates >= pd.Timestamp('1990-01-01')
    X_all = df.loc[combined_mask, factor_cols].values
    all_dates = dates[combined_mask]
    all_smb = smb[combined_mask]
    all_hml_cumul = hml_cumul_9d[combined_mask]
    
    regimes_all, probs_all = hmm.predict_oos(X_all, use_filtered=True)
    
    # Report regime distribution
    for k in range(3):
        train_pct = (regimes_all[all_dates <= train_end] == k).mean() * 100
        test_pct = (regimes_all[all_dates >= test_start] == k).mean() * 100
        print(f"    Regime {k} ({regime_names[k]}): train={train_pct:.1f}%, test={test_pct:.1f}%")
    
    # ---- 5. Calibrate on training data ----
    train_idx = all_dates <= train_end
    test_idx = all_dates >= test_start
    
    train_smb = all_smb[train_idx]
    train_regimes = regimes_all[train_idx]
    train_hml_cumul = all_hml_cumul[train_idx]
    
    cal_params = calibrate_hml_threshold(train_smb, train_regimes, train_hml_cumul)
    
    # Also calibrate regime-conditional window sizes on training data
    print("\n  Calibrating regime-conditional windows on training data...")
    best_regime_score = np.inf
    best_windows = (60, 45, 30)
    for w_calm in [50, 60, 75, 90]:
        for w_normal in [30, 40, 45, 50]:
            for w_crisis in [15, 20, 25, 30]:
                var_rc = regime_conditional_var(
                    train_smb, train_regimes, alpha=0.05,
                    window_calm=w_calm, window_normal=w_normal, window_crisis=w_crisis
                )
                valid = ~np.isnan(var_rc)
                if valid.sum() < 100:
                    continue
                vr = (train_smb[valid] < var_rc[valid]).mean()
                score = abs(vr - 0.05)
                if score < best_regime_score:
                    best_regime_score = score
                    best_windows = (w_calm, w_normal, w_crisis)
    
    print(f"    Best windows: Calm={best_windows[0]}, Normal={best_windows[1]}, Crisis={best_windows[2]}")
    
    # ---- 6. Compute VaR on TEST data ----
    print("\n  Computing VaR models on test period...")
    test_smb = all_smb[test_idx]
    test_regimes = regimes_all[test_idx]
    test_hml_cumul = all_hml_cumul[test_idx]
    test_dates = all_dates[test_idx]
    
    # Model 1: Unconditional historical VaR
    var_baseline = rolling_historical_var(test_smb, window=60, alpha=0.05)
    
    # Model 2: Regime-conditional VaR
    var_regime = regime_conditional_var(
        test_smb, test_regimes, alpha=0.05,
        window_calm=best_windows[0], window_normal=best_windows[1], 
        window_crisis=best_windows[2]
    )
    
    # Model 3: HML-informed regime VaR
    var_hml, stress_applied = hml_informed_var(
        test_smb, test_regimes, test_hml_cumul, alpha=0.05,
        window_calm=best_windows[0], window_normal=best_windows[1],
        window_crisis=best_windows[2],
        hml_threshold=cal_params['threshold'],
        stress_multiplier=cal_params['multiplier']
    )
    
    # ---- 7. Evaluate ----
    print("\n" + "=" * 72)
    print("OUT-OF-SAMPLE VaR BACKTEST RESULTS (2013-2024)")
    print("=" * 72)
    
    results_all = {}
    
    eval_baseline, viol_baseline = evaluate_var_model(test_smb, var_baseline, "Unconditional Historical VaR")
    eval_regime, viol_regime = evaluate_var_model(test_smb, var_regime, "Regime-Conditional VaR")
    eval_hml, viol_hml = evaluate_var_model(test_smb, var_hml, "HML-Informed Regime VaR")
    
    # Unnecessary alerts
    unnecessary_regime, conservative_regime = count_unnecessary_alerts(
        var_baseline, var_regime, test_smb)
    unnecessary_hml, conservative_hml = count_unnecessary_alerts(
        var_baseline, var_hml, test_smb)
    
    eval_baseline['unnecessary_alerts'] = 0
    eval_baseline['conservative_days'] = 0
    eval_regime['unnecessary_alerts'] = unnecessary_regime
    eval_regime['conservative_days'] = conservative_regime
    eval_hml['unnecessary_alerts'] = unnecessary_hml
    eval_hml['conservative_days'] = conservative_hml
    
    # Print summary table
    print(f"\n{'Model':<32} {'Viol%':>7} {'Target':>7} {'Dev':>7} {'Avg Mag':>9} "
          f"{'CC p-val':>9} {'Ind p-val':>10} {'Alerts':>7}")
    print("-" * 98)
    
    for ev in [eval_baseline, eval_regime, eval_hml]:
        name = ev['model']
        if len(name) > 30:
            name = name[:30]
        p_cc = ev['christoffersen_p_cc']
        p_ind = ev['christoffersen_p_ind']
        p_cc_str = f"{p_cc:.4f}" if p_cc is not None else "N/A"
        p_ind_str = f"{p_ind:.4f}" if p_ind is not None else "N/A"
        print(f"{name:<32} {ev['violation_rate_pct']:>6.2f}% {ev['target_rate_pct']:>6.1f}% "
              f"{ev['deviation_from_target_pct']:>+6.2f}% {ev['avg_violation_magnitude']:>9.4f} "
              f"{p_cc_str:>9} {p_ind_str:>10} {ev['unnecessary_alerts']:>7}")
    
    # Absolute deviation from target
    print(f"\n  Absolute deviation from 5% target:")
    for ev in [eval_baseline, eval_regime, eval_hml]:
        dev = abs(ev['deviation_from_target_pct'])
        print(f"    {ev['model']}: {dev:.2f} pp")
    
    # ---- 8. Crisis period drill-down ----
    print("\n" + "=" * 72)
    print("CRISIS PERIOD ANALYSIS")
    print("=" * 72)
    
    crisis_results = analyze_crisis_periods(
        test_smb, var_baseline, var_regime, var_hml,
        test_regimes, test_dates, stress_applied
    )
    
    for name, cr in crisis_results.items():
        print(f"\n  {name} ({cr['n_days']} days, {cr['pct_crisis_regime']:.0f}% Crisis regime, "
              f"{cr['stress_days']} stress days):")
        if 'baseline_violation_rate' in cr:
            print(f"    Baseline:     {cr['baseline_violations']} violations ({cr['baseline_violation_rate']:.1f}%)")
        if 'regime_violation_rate' in cr:
            print(f"    Regime-cond:  {cr['regime_violations']} violations ({cr['regime_violation_rate']:.1f}%)")
        if 'hml_informed_violation_rate' in cr:
            print(f"    HML-informed: {cr['hml_informed_violations']} violations ({cr['hml_informed_violation_rate']:.1f}%)")
    
    # ---- 9. Regime-stratified analysis ----
    print("\n" + "=" * 72)
    print("REGIME-STRATIFIED VaR PERFORMANCE (Test Period)")
    print("=" * 72)
    
    regime_stratified = {}
    for k in range(3):
        rmask = test_regimes == k
        if rmask.sum() < 60:
            continue
        
        r_smb = test_smb[rmask]
        r_var_b = var_baseline[rmask]
        r_var_h = var_hml[rmask]
        
        valid_b = ~np.isnan(r_var_b)
        valid_h = ~np.isnan(r_var_h)
        
        if valid_b.sum() > 0 and valid_h.sum() > 0:
            vrate_b = (r_smb[valid_b] < r_var_b[valid_b]).mean() * 100
            vrate_h = (r_smb[valid_h] < r_var_h[valid_h]).mean() * 100
            
            print(f"\n  Regime {k} ({regime_names[k]}, {rmask.sum()} days):")
            print(f"    Baseline violation rate:     {vrate_b:.2f}%")
            print(f"    HML-informed violation rate:  {vrate_h:.2f}%")
            print(f"    Improvement: {abs(vrate_b - 5) - abs(vrate_h - 5):+.2f} pp closer to 5%")
            
            regime_stratified[regime_names[k]] = {
                'n_days': int(rmask.sum()),
                'baseline_violation_rate': round(float(vrate_b), 2),
                'hml_informed_violation_rate': round(float(vrate_h), 2),
                'improvement_pp': round(float(abs(vrate_b - 5) - abs(vrate_h - 5)), 2),
            }
    
    # ---- 10. Statistical test: is HML-informed model significantly better? ----
    print("\n" + "=" * 72)
    print("STATISTICAL COMPARISON")
    print("=" * 72)
    
    # Diebold-Mariano-like test on VaR violations
    valid = ~np.isnan(var_baseline) & ~np.isnan(var_hml)
    ret_v = test_smb[valid]
    vb = var_baseline[valid]
    vh = var_hml[valid]
    
    # Quantile loss function (tick loss)
    alpha = 0.05
    loss_baseline = np.where(ret_v < vb, alpha * (vb - ret_v), (1-alpha) * (ret_v - vb))
    loss_hml = np.where(ret_v < vh, alpha * (vh - ret_v), (1-alpha) * (ret_v - vh))
    loss_diff = loss_baseline - loss_hml  # positive = HML model is better
    
    # DM test
    dm_mean = loss_diff.mean()
    dm_se = loss_diff.std() / np.sqrt(len(loss_diff))
    dm_stat = dm_mean / dm_se if dm_se > 0 else 0
    dm_pval = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    
    print(f"\n  Tick loss comparison (Diebold-Mariano style):")
    print(f"    Mean baseline tick loss:     {loss_baseline.mean():.6f}")
    print(f"    Mean HML-informed tick loss: {loss_hml.mean():.6f}")
    print(f"    Mean difference:             {dm_mean:.6f}")
    print(f"    DM statistic:                {dm_stat:.3f}")
    print(f"    DM p-value:                  {dm_pval:.4f}")
    
    if dm_pval < 0.05 and dm_mean > 0:
        print(f"    --> HML-informed model SIGNIFICANTLY better at p<0.05")
    elif dm_pval < 0.10 and dm_mean > 0:
        print(f"    --> HML-informed model better at p<0.10 (marginal)")
    else:
        print(f"    --> No significant difference at p<0.10")
    
    # Also compare regime-conditional vs baseline
    valid_r = ~np.isnan(var_baseline) & ~np.isnan(var_regime)
    ret_vr = test_smb[valid_r]
    vbr = var_baseline[valid_r]
    vrr = var_regime[valid_r]
    
    loss_baseline_r = np.where(ret_vr < vbr, alpha * (vbr - ret_vr), (1-alpha) * (ret_vr - vbr))
    loss_regime_r = np.where(ret_vr < vrr, alpha * (vrr - ret_vr), (1-alpha) * (ret_vr - vrr))
    loss_diff_r = loss_baseline_r - loss_regime_r
    
    dm_mean_r = loss_diff_r.mean()
    dm_se_r = loss_diff_r.std() / np.sqrt(len(loss_diff_r))
    dm_stat_r = dm_mean_r / dm_se_r if dm_se_r > 0 else 0
    dm_pval_r = 2 * (1 - stats.norm.cdf(abs(dm_stat_r)))
    
    print(f"\n  Regime-conditional vs Baseline:")
    print(f"    Mean difference:             {dm_mean_r:.6f}")
    print(f"    DM statistic:                {dm_stat_r:.3f}")
    print(f"    DM p-value:                  {dm_pval_r:.4f}")
    
    # ---- 11. Save results ----
    print("\n" + "=" * 72)
    print("SAVING RESULTS")
    print("=" * 72)
    
    results_json = {
        'description': 'Risk Monitoring Backtest: HML-Informed SMB VaR Forecasting',
        'train_period': '1990-01-01 to 2012-12-31',
        'test_period': f'{test_dates[0].date()} to {test_dates[-1].date()}',
        'n_test_days': int(test_idx.sum()),
        'hmm_params': {
            'n_regimes': 3,
            'factors_used': factor_cols,
            'regime_names': regime_names,
            'nu': [round(float(v), 2) for v in hmm.nu],
        },
        'calibrated_params': {
            'hml_threshold': cal_params['threshold'],
            'stress_multiplier': cal_params['multiplier'],
            'regime_windows': {
                'calm': best_windows[0],
                'normal': best_windows[1],
                'crisis': best_windows[2],
            },
        },
        'overall_results': {
            'unconditional_historical': eval_baseline,
            'regime_conditional': eval_regime,
            'hml_informed': eval_hml,
        },
        'statistical_tests': {
            'dm_hml_vs_baseline': {
                'mean_tick_loss_baseline': round(float(loss_baseline.mean()), 6),
                'mean_tick_loss_hml': round(float(loss_hml.mean()), 6),
                'dm_statistic': round(float(dm_stat), 4),
                'dm_pvalue': round(float(dm_pval), 4),
            },
            'dm_regime_vs_baseline': {
                'mean_tick_loss_baseline': round(float(loss_baseline_r.mean()), 6),
                'mean_tick_loss_regime': round(float(loss_regime_r.mean()), 6),
                'dm_statistic': round(float(dm_stat_r), 4),
                'dm_pvalue': round(float(dm_pval_r), 4),
            },
        },
        'crisis_periods': crisis_results,
        'regime_stratified': regime_stratified,
    }
    
    output_path = os.path.join(RESULTS_DIR, 'risk_monitoring_results.json')
    with open(output_path, 'w') as f:
        json.dump(results_json, f, indent=2, default=str)
    print(f"\n  Results saved to: {output_path}")
    
    # ---- Final summary ----
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    
    best_model = min([eval_baseline, eval_regime, eval_hml], 
                     key=lambda x: abs(x['deviation_from_target_pct']))
    print(f"\n  Best calibrated model: {best_model['model']}")
    print(f"    Violation rate: {best_model['violation_rate_pct']:.2f}% (target: 5%)")
    print(f"    Deviation: {best_model['deviation_from_target_pct']:+.2f} pp")
    
    # Key finding
    print(f"\n  Key finding for paper:")
    hml_dev = abs(eval_hml['deviation_from_target_pct'])
    base_dev = abs(eval_baseline['deviation_from_target_pct'])
    regime_dev = abs(eval_regime['deviation_from_target_pct'])
    print(f"    Deviation from 5% target: Baseline={base_dev:.2f}pp, "
          f"Regime={regime_dev:.2f}pp, HML-informed={hml_dev:.2f}pp")
    
    if hml_dev < base_dev:
        improvement = base_dev - hml_dev
        print(f"    HML-informed model reduces deviation by {improvement:.2f} pp")
    
    if eval_hml['avg_violation_magnitude'] > eval_baseline['avg_violation_magnitude']:
        es_improvement = ((eval_hml['avg_violation_magnitude'] - eval_baseline['avg_violation_magnitude']) 
                          / abs(eval_baseline['avg_violation_magnitude'])) * 100
        print(f"    Average violation magnitude reduced by {es_improvement:.1f}% "
              f"(less severe tail breaches)")
    
    stress_total = int(stress_applied.sum())
    stress_pct = stress_total / test_idx.sum() * 100
    print(f"    Stress multiplier activated on {stress_total} days ({stress_pct:.1f}% of test period)")
    
    print("\n  Done.")


if __name__ == '__main__':
    main()
