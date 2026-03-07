"""
Critical Fixes Analysis for ICAIF 2025 Reviewers
=================================================

Addresses the four most critical reviewer concerns:
1. True out-of-sample validation (train HMM 1990-2012, freeze, test 2013-2024)
2. Markov-Switching VAR comparison (single-stage benchmark)
3. Incremental R² table per regime
4. Filtered vs smoothed regime comparison

Output: JSON file with all results for paper integration.
"""

import numpy as np
import pandas as pd
import json
import urllib.request
import zipfile
import io
import sys
from scipy import stats
from scipy.special import gammaln
from scipy.optimize import minimize_scalar
from scipy.cluster.vq import kmeans2
from scipy.linalg import expm
from scipy.stats import f as f_dist
from statsmodels.tsa.stattools import grangercausalitytests
import warnings
warnings.filterwarnings('ignore')

RESULTS_DIR = '/Users/i767700/Github/ai-in-finance/papers/causal_regimes/results'

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

    # Combine
    df = df5.join(mom[['MOM']], how='inner')
    df = df.rename(columns={'Mkt-RF': 'MKT'})
    df = df.drop('RF', axis=1, errors='ignore')

    # Filter 1990-2024
    df = df.loc['1990-01-01':'2024-12-31']
    print(f"Loaded {len(df)} trading days ({df.index[0].date()} to {df.index[-1].date()})")
    return df


# =============================================================================
# STUDENT-T HMM (from gate2, with filtered probability support)
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
        self.gamma = None       # Smoothed posteriors
        self.alpha = None       # Filtered posteriors (new)
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

        # Smoothed posteriors
        log_gamma = log_alpha + log_beta
        log_gamma = log_gamma - np.logaddexp.reduce(log_gamma, axis=1, keepdims=True)
        self.gamma = np.exp(log_gamma)

        # Filtered posteriors (forward only, normalized)
        log_alpha_norm = log_alpha - np.logaddexp.reduce(log_alpha, axis=1, keepdims=True)
        self.alpha = np.exp(log_alpha_norm)

        # Pairwise posteriors
        log_A = np.log(self.A + 1e-300)
        self.xi = np.zeros((T - 1, K, K))
        for t in range(T - 1):
            for j in range(K):
                for k in range(K):
                    self.xi[t, j, k] = np.exp(
                        log_alpha[t, j] + log_A[j, k] + log_B[t+1, k] + log_beta[t+1, k]
                        - log_likelihood
                    )

        # Expected auxiliary variable for Student-t
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

    def predict(self, X, use_filtered=False):
        """Predict regimes. use_filtered=True for real-time (forward-only)."""
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
# GRANGER CAUSALITY HELPERS
# =============================================================================

def granger_test_manual(x, y, max_lag=15):
    """Manual Granger test returning best lag, p-value, and F-stat."""
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

        X_r = np.column_stack([np.ones(len(y_curr)), y_lagged])
        X_u = np.column_stack([np.ones(len(y_curr)), y_lagged, x_lagged])

        try:
            beta_r = np.linalg.lstsq(X_r, y_curr, rcond=None)[0]
            beta_u = np.linalg.lstsq(X_u, y_curr, rcond=None)[0]

            rss_r = np.sum((y_curr - X_r @ beta_r) ** 2)
            rss_u = np.sum((y_curr - X_u @ beta_u) ** 2)

            df1 = lag
            df2 = len(y_curr) - 2 * lag - 1

            if df2 > 0 and rss_u > 0:
                f_stat = ((rss_r - rss_u) / df1) / (rss_u / df2)
                p_value = 1 - f_dist.cdf(f_stat, df1, df2)

                # Incremental R²
                tss = np.sum((y_curr - y_curr.mean()) ** 2)
                r2_restricted = 1 - rss_r / tss
                r2_unrestricted = 1 - rss_u / tss
                delta_r2 = r2_unrestricted - r2_restricted

                all_results[lag] = {
                    'p_value': float(p_value),
                    'f_stat': float(f_stat),
                    'r2_restricted': float(r2_restricted),
                    'r2_unrestricted': float(r2_unrestricted),
                    'delta_r2': float(delta_r2),
                    'n_obs': len(y_curr)
                }

                if p_value < best_p:
                    best_p = p_value
                    best_lag = lag
                    best_f = f_stat
        except Exception:
            continue

    return best_lag, best_p, best_f, all_results


def extract_regime_data_clean(df, regimes, regime_id, lag=9):
    """Extract regime data ensuring all lags fall within same regime.
    Returns aligned HML, SMB arrays."""
    regime_mask = (regimes == regime_id)
    indices = np.where(regime_mask)[0]

    clean_indices = []
    for idx in indices:
        if idx >= lag:
            # Check all lags are in same regime
            all_in_regime = all(regimes[idx - l] == regime_id for l in range(1, lag + 1))
            if all_in_regime:
                clean_indices.append(idx)

    if len(clean_indices) < 30:
        return None, None, len(clean_indices)

    clean_indices = np.array(clean_indices)
    hml = df['HML'].values[clean_indices]
    smb = df['SMB'].values[clean_indices]
    return hml, smb, len(clean_indices)


# =============================================================================
# FIX 1: TRUE OUT-OF-SAMPLE VALIDATION
# =============================================================================

def fix1_held_out_validation(df):
    """Train HMM on 1990-2012, freeze parameters, test 2013-2024."""
    print("\n" + "=" * 70)
    print("FIX 1: TRUE OUT-OF-SAMPLE VALIDATION")
    print("Train: 1990-2012 | Test: 2013-2024 (frozen HMM parameters)")
    print("=" * 70)

    # Split data
    train_df = df.loc[:'2012-12-31']
    test_df = df.loc['2013-01-01':]

    print(f"  Train: {train_df.index[0].date()} to {train_df.index[-1].date()} ({len(train_df)} days)")
    print(f"  Test:  {test_df.index[0].date()} to {test_df.index[-1].date()} ({len(test_df)} days)")

    # Fit HMM on training data only
    print("\n  Fitting Student-t HMM on training data...")
    hmm = StudentTHMM(n_regimes=3, n_iter=100, tol=1e-4, random_state=28)
    hmm.fit(train_df.values)

    print(f"  Training HMM parameters:")
    print(f"    nu = [{hmm.nu[0]:.1f}, {hmm.nu[1]:.1f}, {hmm.nu[2]:.1f}]")
    print(f"    Diag(A) = [{hmm.A[0,0]:.3f}, {hmm.A[1,1]:.3f}, {hmm.A[2,2]:.3f}]")

    # Classify training data (for reference)
    train_regimes_smoothed = hmm.predict(train_df.values, use_filtered=False)
    train_crisis_days = (train_regimes_smoothed == 2).sum()
    print(f"  Training crisis days: {train_crisis_days} ({train_crisis_days/len(train_df)*100:.1f}%)")

    # Classify test data using FROZEN parameters (no refit)
    print("\n  Classifying test data with frozen HMM parameters...")
    test_regimes_smoothed, test_probs_smoothed = hmm.predict_oos(test_df.values, use_filtered=False)
    test_regimes_filtered, test_probs_filtered = hmm.predict_oos(test_df.values, use_filtered=True)

    # Report regime distribution in test period
    regime_names = ['Normal', 'Elevated', 'Crisis']
    print("\n  Test period regime distribution (smoothed):")
    for k in range(3):
        n = (test_regimes_smoothed == k).sum()
        print(f"    {regime_names[k]}: {n} days ({n/len(test_df)*100:.1f}%)")

    # Run Granger tests on held-out crisis data
    print("\n  Running Granger tests on held-out crisis days...")
    results = {}

    for regime_id, regime_name in enumerate(regime_names):
        hml, smb, n_clean = extract_regime_data_clean(
            test_df, test_regimes_smoothed, regime_id, lag=15
        )

        if hml is None:
            print(f"    {regime_name}: insufficient data ({n_clean} clean obs)")
            results[regime_name] = {'n_obs': n_clean, 'hml_to_smb': None, 'smb_to_hml': None}
            continue

        lag1, p1, f1, details1 = granger_test_manual(hml, smb, max_lag=15)
        lag2, p2, f2, details2 = granger_test_manual(smb, hml, max_lag=15)

        print(f"    {regime_name} ({n_clean} clean obs):")
        print(f"      HML -> SMB: lag={lag1}, p={p1:.2e}, F={f1:.2f}")
        print(f"      SMB -> HML: lag={lag2}, p={p2:.2e}, F={f2:.2f}")

        results[regime_name] = {
            'n_obs': n_clean,
            'hml_to_smb': {'lag': lag1, 'p_value': float(p1), 'f_stat': float(f1)},
            'smb_to_hml': {'lag': lag2, 'p_value': float(p2), 'f_stat': float(f2)},
        }

    # Event-based validation on test period events
    print("\n  Event-based validation (test period only):")
    test_events = [
        ('2015-08-15', '2015-09-15', '2015 China'),
        ('2018-12-01', '2018-12-31', '2018 Vol Shock'),
        ('2020-02-20', '2020-06-30', '2020 COVID'),
        ('2022-01-01', '2022-06-30', '2022 Rate Hikes'),
    ]

    event_results = []
    for start, end, name in test_events:
        mask = (test_df.index >= start) & (test_df.index <= end)
        if mask.sum() < 20:
            continue
        event_data = test_df[mask]
        event_regimes = test_regimes_smoothed[np.where(mask)[0]]
        crisis_pct = (event_regimes == 2).mean() * 100

        # Granger test within event window
        hml_ev = event_data['HML'].values
        smb_ev = event_data['SMB'].values
        if len(hml_ev) > 20:
            lag_ev, p_ev, f_ev, _ = granger_test_manual(hml_ev, smb_ev, max_lag=min(10, len(hml_ev)//3))
            _, p_rev, _, _ = granger_test_manual(smb_ev, hml_ev, max_lag=min(10, len(hml_ev)//3))
        else:
            lag_ev, p_ev, p_rev = None, None, None

        print(f"    {name}: {mask.sum()} days, {crisis_pct:.0f}% crisis, "
              f"HML->SMB p={p_ev:.3f}" if p_ev else f"    {name}: insufficient data")

        event_results.append({
            'event': name,
            'days': int(mask.sum()),
            'crisis_pct': float(crisis_pct),
            'hml_to_smb_p': float(p_ev) if p_ev else None,
            'smb_to_hml_p': float(p_rev) if p_rev else None,
        })

    results['events'] = event_results
    results['train_period'] = f"{train_df.index[0].date()} to {train_df.index[-1].date()}"
    results['test_period'] = f"{test_df.index[0].date()} to {test_df.index[-1].date()}"
    results['train_crisis_days'] = int(train_crisis_days)

    return results, hmm, test_df, test_regimes_smoothed, test_regimes_filtered


# =============================================================================
# FIX 2: MARKOV-SWITCHING VAR COMPARISON
# =============================================================================

def fix2_msvar_comparison(df):
    """Compare with Markov-Switching regression as single-stage benchmark."""
    print("\n" + "=" * 70)
    print("FIX 2: MARKOV-SWITCHING VAR COMPARISON")
    print("=" * 70)

    try:
        from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
        from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    except ImportError:
        print("  statsmodels regime_switching not available")
        return {'available': False}

    # Use HML and SMB
    smb = df['SMB'].values
    hml = df['HML'].values

    # Create lagged HML variables
    max_lag = 9
    data = pd.DataFrame({'SMB': smb, 'HML': hml}, index=df.index)
    for lag in range(1, max_lag + 1):
        data[f'HML_lag{lag}'] = data['HML'].shift(lag)
        data[f'SMB_lag{lag}'] = data['SMB'].shift(lag)
    data = data.dropna()

    print(f"  Data: {len(data)} observations after creating lags")

    # Model 1: Markov-switching AR for SMB (restricted: no HML lags)
    print("\n  Fitting MS-AR(9) for SMB (restricted, no HML lags)...")
    try:
        endog = data['SMB']
        exog_restricted = data[[f'SMB_lag{i}' for i in range(1, max_lag + 1)]]

        ms_restricted = MarkovRegression(
            endog, k_regimes=3, exog=exog_restricted,
            switching_variance=True
        )
        res_restricted = ms_restricted.fit(maxiter=200, search_reps=20)
        print(f"    Log-likelihood (restricted): {res_restricted.llf:.2f}")
    except Exception as e:
        print(f"    Restricted model failed: {e}")
        # Fallback to 2 regimes
        try:
            ms_restricted = MarkovRegression(
                endog, k_regimes=2, exog=exog_restricted,
                switching_variance=True
            )
            res_restricted = ms_restricted.fit(maxiter=200, search_reps=20)
            print(f"    Log-likelihood (restricted, 2 regimes): {res_restricted.llf:.2f}")
        except Exception as e2:
            print(f"    Fallback also failed: {e2}")
            return {'available': True, 'error': str(e2)}

    # Model 2: Markov-switching regression for SMB (unrestricted: with HML lags)
    print("  Fitting MS-VAR for SMB (unrestricted, with HML lags)...")
    try:
        exog_unrestricted = data[[f'SMB_lag{i}' for i in range(1, max_lag + 1)] +
                                  [f'HML_lag{i}' for i in range(1, max_lag + 1)]]

        ms_unrestricted = MarkovRegression(
            endog, k_regimes=res_restricted.k_regimes, exog=exog_unrestricted,
            switching_variance=True
        )
        res_unrestricted = ms_unrestricted.fit(maxiter=200, search_reps=20)
        print(f"    Log-likelihood (unrestricted): {res_unrestricted.llf:.2f}")
    except Exception as e:
        print(f"    Unrestricted model failed: {e}")
        return {'available': True, 'error': str(e)}

    # Likelihood ratio test
    lr_stat = 2 * (res_unrestricted.llf - res_restricted.llf)
    df_diff = max_lag * res_restricted.k_regimes  # additional HML lag params per regime
    lr_pvalue = 1 - stats.chi2.cdf(lr_stat, df_diff)

    print(f"\n  Likelihood Ratio Test:")
    print(f"    LR statistic: {lr_stat:.2f}")
    print(f"    df: {df_diff}")
    print(f"    p-value: {lr_pvalue:.2e}")

    # Extract regime-specific HML coefficients
    print("\n  HML lag coefficients by regime (unrestricted model):")
    k_regimes = res_unrestricted.k_regimes
    hml_coefs_by_regime = {}

    for regime in range(k_regimes):
        coefs = []
        for lag in range(max_lag):
            param_name = f'x{max_lag + lag + 1}'  # HML lags come after SMB lags
            try:
                idx = max_lag + lag + 1  # 0=const, 1-9=SMB lags, 10-18=HML lags
                coef = res_unrestricted.params[regime * (2 * max_lag + 2) + idx + 1]
                coefs.append(float(coef))
            except (IndexError, KeyError):
                coefs.append(0.0)

        hml_coefs_by_regime[f'regime_{regime}'] = coefs
        coef_sum = sum(abs(c) for c in coefs)
        print(f"    Regime {regime}: sum(|coef|) = {coef_sum:.4f}")

    # Get regime probabilities for comparison
    smoothed_probs = res_unrestricted.smoothed_marginal_probabilities

    results = {
        'available': True,
        'k_regimes': int(k_regimes),
        'llf_restricted': float(res_restricted.llf),
        'llf_unrestricted': float(res_unrestricted.llf),
        'lr_statistic': float(lr_stat),
        'lr_df': int(df_diff),
        'lr_pvalue': float(lr_pvalue),
        'hml_significant': lr_pvalue < 0.001,
        'bic_restricted': float(res_restricted.bic),
        'bic_unrestricted': float(res_unrestricted.bic),
    }

    print(f"\n  BIC comparison:")
    print(f"    Restricted (no HML):  {res_restricted.bic:.2f}")
    print(f"    Unrestricted (+ HML): {res_unrestricted.bic:.2f}")
    print(f"    Delta BIC: {res_restricted.bic - res_unrestricted.bic:.2f}")
    print(f"    {'Unrestricted preferred' if res_unrestricted.bic < res_restricted.bic else 'Restricted preferred'}")

    return results


# =============================================================================
# FIX 3: INCREMENTAL R² TABLE
# =============================================================================

def fix3_incremental_r2(df, hmm=None):
    """Compute incremental R² from adding HML lags to SMB prediction, per regime."""
    print("\n" + "=" * 70)
    print("FIX 3: INCREMENTAL R² TABLE")
    print("=" * 70)

    # Fit HMM on full sample for regime assignments
    if hmm is None:
        print("  Fitting HMM on full sample...")
        hmm = StudentTHMM(n_regimes=3, n_iter=100, tol=1e-4, random_state=28)
        hmm.fit(df.values)

    regimes = hmm.predict(df.values)
    regime_names = ['Normal', 'Elevated', 'Crisis']

    results = {}

    for regime_id, regime_name in enumerate(regime_names):
        hml, smb, n_clean = extract_regime_data_clean(df, regimes, regime_id, lag=15)

        if hml is None:
            results[regime_name] = {'n_obs': n_clean, 'delta_r2': None}
            continue

        # Run Granger test with detailed R² output
        lag, p, f, details = granger_test_manual(hml, smb, max_lag=15)

        if lag in details:
            d = details[lag]
            print(f"\n  {regime_name} (n={n_clean}, best lag={lag}):")
            print(f"    R²(SMB ~ SMB lags):         {d['r2_restricted']:.4f}")
            print(f"    R²(SMB ~ SMB + HML lags):   {d['r2_unrestricted']:.4f}")
            print(f"    Delta R²:                    {d['delta_r2']:.4f} ({d['delta_r2']*100:.2f}%)")
            print(f"    F-stat: {f:.2f}, p-value: {p:.2e}")

            results[regime_name] = {
                'n_obs': n_clean,
                'best_lag': lag,
                'r2_restricted': d['r2_restricted'],
                'r2_unrestricted': d['r2_unrestricted'],
                'delta_r2': d['delta_r2'],
                'delta_r2_pct': d['delta_r2'] * 100,
                'f_stat': float(f),
                'p_value': float(p),
            }

            # Also report R² at specific lags (for BIC surface)
            lag_r2 = {}
            for l in sorted(details.keys()):
                lag_r2[str(l)] = {
                    'delta_r2': details[l]['delta_r2'],
                    'p_value': details[l]['p_value']
                }
            results[regime_name]['by_lag'] = lag_r2
        else:
            results[regime_name] = {'n_obs': n_clean, 'delta_r2': None}

    # Also compute for reverse direction (SMB -> HML)
    print("\n  Reverse direction (SMB -> HML):")
    for regime_id, regime_name in enumerate(regime_names):
        smb_r, hml_r, n_clean = extract_regime_data_clean(df, regimes, regime_id, lag=15)
        if smb_r is None:
            continue
        # Note: extract gives (hml, smb), so we swap
        hml_vals = df['HML'].values
        smb_vals = df['SMB'].values
        regime_mask = (regimes == regime_id)
        indices = np.where(regime_mask)[0]
        clean_indices = []
        for idx in indices:
            if idx >= 15:
                all_in = all(regimes[idx - l] == regime_id for l in range(1, 16))
                if all_in:
                    clean_indices.append(idx)
        if len(clean_indices) < 30:
            continue
        clean_indices = np.array(clean_indices)
        smb_c = smb_vals[clean_indices]
        hml_c = hml_vals[clean_indices]
        lag_r, p_r, f_r, details_r = granger_test_manual(smb_c, hml_c, max_lag=15)
        if lag_r in details_r:
            d_r = details_r[lag_r]
            print(f"    {regime_name}: lag={lag_r}, Delta R²={d_r['delta_r2']:.4f}, p={p_r:.2e}")
            results[regime_name + '_reverse'] = {
                'best_lag': lag_r,
                'delta_r2': d_r['delta_r2'],
                'p_value': float(p_r),
            }

    return results


# =============================================================================
# FIX 4: FILTERED VS SMOOTHED REGIME COMPARISON
# =============================================================================

def fix4_filtered_vs_smoothed(df, hmm, test_df, test_regimes_smoothed, test_regimes_filtered):
    """Compare Granger results under filtered (real-time) vs smoothed regimes."""
    print("\n" + "=" * 70)
    print("FIX 4: FILTERED VS SMOOTHED REGIME COMPARISON")
    print("=" * 70)

    regime_names = ['Normal', 'Elevated', 'Crisis']

    # Agreement rate
    agreement = (test_regimes_smoothed == test_regimes_filtered).mean()
    print(f"  Overall agreement (smoothed vs filtered): {agreement*100:.1f}%")

    # Per-regime agreement
    for k in range(3):
        smoothed_k = (test_regimes_smoothed == k).sum()
        filtered_k = (test_regimes_filtered == k).sum()
        both_k = ((test_regimes_smoothed == k) & (test_regimes_filtered == k)).sum()
        print(f"  {regime_names[k]}: smoothed={smoothed_k}, filtered={filtered_k}, overlap={both_k}")

    # Run Granger tests under filtered regimes (test period)
    print("\n  Granger tests under FILTERED regimes (test period):")
    results_filtered = {}
    for regime_id, regime_name in enumerate(regime_names):
        hml, smb, n_clean = extract_regime_data_clean(
            test_df, test_regimes_filtered, regime_id, lag=15
        )
        if hml is None:
            print(f"    {regime_name}: insufficient data ({n_clean})")
            results_filtered[regime_name] = {'n_obs': n_clean}
            continue

        lag1, p1, f1, _ = granger_test_manual(hml, smb, max_lag=15)
        lag2, p2, f2, _ = granger_test_manual(smb, hml, max_lag=15)
        print(f"    {regime_name} ({n_clean} obs): HML->SMB p={p1:.2e} (lag={lag1}), SMB->HML p={p2:.2e} (lag={lag2})")

        results_filtered[regime_name] = {
            'n_obs': n_clean,
            'hml_to_smb': {'lag': lag1, 'p_value': float(p1)},
            'smb_to_hml': {'lag': lag2, 'p_value': float(p2)},
        }

    # Also do full-sample filtered vs smoothed
    print("\n  Full-sample comparison:")
    regimes_smoothed_full = hmm.predict(df.values, use_filtered=False)
    regimes_filtered_full = hmm.predict(df.values, use_filtered=True)
    agreement_full = (regimes_smoothed_full == regimes_filtered_full).mean()
    print(f"    Full-sample agreement: {agreement_full*100:.1f}%")

    results_smoothed_full = {}
    for regime_id, regime_name in enumerate(regime_names):
        hml, smb, n_clean = extract_regime_data_clean(
            df, regimes_filtered_full, regime_id, lag=15
        )
        if hml is None:
            continue
        lag1, p1, _, _ = granger_test_manual(hml, smb, max_lag=15)
        lag2, p2, _, _ = granger_test_manual(smb, hml, max_lag=15)
        print(f"    {regime_name} (filtered, {n_clean} obs): HML->SMB p={p1:.2e}, SMB->HML p={p2:.2e}")
        results_smoothed_full[regime_name] = {
            'n_obs': n_clean,
            'hml_to_smb_p': float(p1),
            'smb_to_hml_p': float(p2),
        }

    return {
        'agreement_test': float(agreement),
        'agreement_full': float(agreement_full),
        'filtered_test_results': results_filtered,
        'filtered_full_results': results_smoothed_full,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    import os
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load data
    df = download_ff_data()

    # Fix 1: Held-out validation
    fix1_results, hmm, test_df, test_smoothed, test_filtered = fix1_held_out_validation(df)

    # Fix 2: MS-VAR comparison
    fix2_results = fix2_msvar_comparison(df)

    # Fix 3: Incremental R²
    fix3_results = fix3_incremental_r2(df)

    # Fix 4: Filtered vs smoothed
    fix4_results = fix4_filtered_vs_smoothed(df, hmm, test_df, test_smoothed, test_filtered)

    # Save all results
    all_results = {
        'fix1_held_out_validation': fix1_results,
        'fix2_msvar_comparison': fix2_results,
        'fix3_incremental_r2': fix3_results,
        'fix4_filtered_vs_smoothed': fix4_results,
    }

    output_path = os.path.join(RESULTS_DIR, 'critical_fixes_results.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n\nAll results saved to: {output_path}")

    # Summary
    print("\n" + "=" * 70)
    print("EXECUTIVE SUMMARY")
    print("=" * 70)

    # Fix 1 summary
    crisis_oos = fix1_results.get('Crisis', {})
    if crisis_oos.get('hml_to_smb'):
        p_oos = crisis_oos['hml_to_smb']['p_value']
        print(f"\n  Fix 1 (OOS Validation): Crisis HML->SMB p = {p_oos:.2e}")
        print(f"    {'PASSES' if p_oos < 0.05 else 'FAILS'} at alpha=0.05")
        print(f"    {'PASSES' if p_oos < 0.00033 else 'FAILS'} Bonferroni (0.00033)")

    # Fix 2 summary
    if fix2_results.get('lr_pvalue') is not None:
        print(f"\n  Fix 2 (MS-VAR): LR test p = {fix2_results['lr_pvalue']:.2e}")
        print(f"    HML lags {'improve' if fix2_results['lr_pvalue'] < 0.01 else 'do not improve'} MS-VAR model")

    # Fix 3 summary
    if fix3_results.get('Crisis', {}).get('delta_r2') is not None:
        dr2 = fix3_results['Crisis']['delta_r2_pct']
        print(f"\n  Fix 3 (Incremental R²): Crisis Delta R² = {dr2:.2f}%")

    # Fix 4 summary
    print(f"\n  Fix 4 (Filtered vs Smoothed): {fix4_results['agreement_full']*100:.1f}% agreement")
    crisis_filt = fix4_results.get('filtered_full_results', {}).get('Crisis', {})
    if crisis_filt.get('hml_to_smb_p') is not None:
        print(f"    Crisis HML->SMB under filtered: p = {crisis_filt['hml_to_smb_p']:.2e}")


if __name__ == '__main__':
    main()
