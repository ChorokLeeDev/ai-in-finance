"""
HAC-Robust Granger Causality with Canonical Regime Assignments
==============================================================

Produces two outputs:
1. canonical_regimes.json — date/regime_id/regime_name for every trading day
2. hac_granger_results.json — standard F-test vs HAC (Newey-West) Wald test

Uses the SAME StudentTHMM implementation as critical_fixes_analysis.py
(K=3, random_state=42, regime ordering by mean factor norm).
"""

import numpy as np
import pandas as pd
import json
import os
import urllib.request
import zipfile
import io
from scipy import stats
from scipy.special import gammaln
from scipy.optimize import minimize_scalar
from scipy.cluster.vq import kmeans2
from scipy.stats import f as f_dist, chi2
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

RESULTS_DIR = '/Users/i767700/Github/ai-in-finance/papers/causal_regimes/results'

# =============================================================================
# DATA LOADING (identical to critical_fixes_analysis.py)
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
# STUDENT-T HMM (identical to critical_fixes_analysis.py)
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

    def predict(self, X, use_filtered=False):
        X = np.asarray(X)
        self._e_step(X)
        if use_filtered:
            return np.argmax(self.alpha, axis=1)
        return np.argmax(self.gamma, axis=1)


# =============================================================================
# REGIME DATA EXTRACTION (boundary-clean, matching the paper)
# =============================================================================

def extract_regime_clean_indices(regimes, regime_id, max_lag):
    """Get indices where ALL lags 1..max_lag fall within the same regime."""
    regime_mask = (regimes == regime_id)
    indices = np.where(regime_mask)[0]

    clean_indices = []
    for idx in indices:
        if idx >= max_lag:
            all_in_regime = all(regimes[idx - l] == regime_id for l in range(1, max_lag + 1))
            if all_in_regime:
                clean_indices.append(idx)

    return np.array(clean_indices) if clean_indices else np.array([], dtype=int)


# =============================================================================
# BIC-OPTIMAL LAG SELECTION
# =============================================================================

def select_lag_bic(y_vals, x_vals, y_all, x_all, clean_indices, max_lag=15):
    """Select optimal lag using BIC on the unrestricted model.
    
    For each candidate lag p, build the unrestricted regression:
        y_t = const + y_{t-1}..y_{t-p} + x_{t-1}..x_{t-p} + e_t
    using only observations where all p lags are within the same regime.
    """
    best_bic = np.inf
    best_lag = 1

    for lag in range(1, max_lag + 1):
        # Build design matrix using clean indices for this lag
        # We need to re-filter for this specific lag
        usable = []
        for idx in clean_indices:
            if idx >= lag:
                usable.append(idx)
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


# =============================================================================
# GRANGER TESTS: STANDARD F-TEST AND HAC WALD TEST
# =============================================================================

def granger_standard_ftest(y_curr, y_lagged, x_lagged):
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

    # Delta R2
    tss = np.sum((y_curr - y_curr.mean()) ** 2)
    r2_r = 1 - rss_r / tss
    r2_u = 1 - rss_u / tss
    delta_r2 = r2_u - r2_r

    return float(f_stat), float(p_value), float(delta_r2), float(r2_u)


def granger_hac_wald(y_curr, y_lagged, x_lagged, lag):
    """HAC (Newey-West) robust Wald test for Granger causality (x -> y).
    
    Fits the unrestricted model via statsmodels OLS with HAC covariance,
    then tests the joint restriction that all x-lag coefficients = 0.
    """
    n = len(y_curr)
    p = y_lagged.shape[1]  # number of lags

    X_u = np.column_stack([np.ones(n), y_lagged, x_lagged])
    
    # Fit with HAC (Newey-West) standard errors
    # maxlags = lag (matching the number of lags in the regression)
    model = sm.OLS(y_curr, X_u)
    result = model.fit(cov_type='HAC', cov_kwds={'maxlags': lag})

    # The x-lag coefficients are at positions (1 + p) through (1 + 2*p - 1)
    # i.e., indices [1+p, 1+p+1, ..., 1+2p-1]
    n_params = X_u.shape[1]
    R = np.zeros((p, n_params))
    for i in range(p):
        R[i, 1 + p + i] = 1.0

    # Wald test: R*beta = 0
    beta = result.params
    V = result.cov_params()
    Rb = R @ beta
    RVR = R @ V @ R.T

    try:
        wald_stat = float(Rb @ np.linalg.inv(RVR) @ Rb)
        # Wald ~ chi2(p) under H0
        p_value = float(1 - chi2.cdf(wald_stat, p))
    except np.linalg.LinAlgError:
        wald_stat = np.nan
        p_value = np.nan

    return wald_stat, p_value


def run_granger_both_methods(y_all, x_all, clean_indices, best_lag, direction_label):
    """Run both standard and HAC Granger tests for a given direction."""
    # Build arrays aligned to clean_indices
    usable = [idx for idx in clean_indices if idx >= best_lag]
    if len(usable) < 2 * best_lag + 10:
        print(f"    {direction_label}: insufficient data ({len(usable)} obs)")
        return None
    usable = np.array(usable)

    y_curr = y_all[usable]
    y_lagged = np.column_stack([y_all[usable - i - 1] for i in range(best_lag)])
    x_lagged = np.column_stack([x_all[usable - i - 1] for i in range(best_lag)])

    # Standard F-test
    f_stat, f_p, delta_r2, r2_u = granger_standard_ftest(y_curr, y_lagged, x_lagged)

    # HAC Wald test
    wald_stat, hac_p = granger_hac_wald(y_curr, y_lagged, x_lagged, best_lag)

    result = {
        'n_obs': len(usable),
        'best_lag': best_lag,
        'standard_f_stat': f_stat,
        'standard_p_value': f_p,
        'hac_wald_stat': wald_stat,
        'hac_p_value': hac_p,
        'delta_r2': delta_r2,
        'r2_unrestricted': r2_u,
    }

    sig_std = "***" if f_p < 0.001 else ("**" if f_p < 0.01 else ("*" if f_p < 0.05 else ""))
    sig_hac = "***" if hac_p < 0.001 else ("**" if hac_p < 0.01 else ("*" if hac_p < 0.05 else ""))

    print(f"    {direction_label} (n={len(usable)}, lag={best_lag}):")
    print(f"      Standard F-test:  F={f_stat:.3f}, p={f_p:.2e} {sig_std}")
    print(f"      HAC Wald test:    W={wald_stat:.3f}, p={hac_p:.2e} {sig_hac}")
    print(f"      Delta R2: {delta_r2:.6f} ({delta_r2*100:.4f}%)")

    return result


# =============================================================================
# MAIN
# =============================================================================

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Step 1: Download data
    df = download_ff_data()
    assert len(df) == 8817, f"Expected 8,817 trading days, got {len(df)}"
    print(f"Data verification: {len(df)} trading days -- matches paper's 8,817")

    # Step 2: Fit Student-t HMM (K=3, random_state=42)
    print("\nFitting Student-t HMM (K=3, random_state=42)...")
    hmm = StudentTHMM(n_regimes=3, n_iter=100, tol=1e-4, random_state=42)
    hmm.fit(df.values)

    regimes = hmm.predict(df.values, use_filtered=False)
    regime_names = ['Normal', 'Elevated', 'Crisis']

    # Step 3: Report regime counts
    print("\nRegime distribution:")
    regime_counts = {}
    for k in range(3):
        count = int((regimes == k).sum())
        pct = count / len(regimes) * 100
        regime_counts[regime_names[k]] = count
        print(f"  {regime_names[k]} (regime {k}): {count} days ({pct:.1f}%)")

    print(f"\nHMM parameters:")
    print(f"  nu = [{hmm.nu[0]:.2f}, {hmm.nu[1]:.2f}, {hmm.nu[2]:.2f}]")
    print(f"  Diag(A) = [{hmm.A[0,0]:.4f}, {hmm.A[1,1]:.4f}, {hmm.A[2,2]:.4f}]")
    for k in range(3):
        norm = np.linalg.norm(hmm.mu[k])
        print(f"  ||mu_{k}|| = {norm:.4f}")

    # Step 4: Save canonical regime assignments
    print("\nSaving canonical regime assignments...")
    canonical = {
        'metadata': {
            'n_days': len(df),
            'date_range': f"{df.index[0].date()} to {df.index[-1].date()}",
            'hmm_params': {
                'n_regimes': 3,
                'random_state': 42,
                'nu': [float(v) for v in hmm.nu],
                'diag_A': [float(hmm.A[k,k]) for k in range(3)],
            },
            'regime_counts': regime_counts,
        },
        'assignments': []
    }
    for i in range(len(df)):
        canonical['assignments'].append({
            'date': str(df.index[i].date()),
            'regime_id': int(regimes[i]),
            'regime_name': regime_names[regimes[i]],
        })

    canonical_path = os.path.join(RESULTS_DIR, 'canonical_regimes.json')
    with open(canonical_path, 'w') as f:
        json.dump(canonical, f, indent=2)
    print(f"  Saved to {canonical_path}")

    # Step 5: Granger causality with HAC
    print("\n" + "=" * 70)
    print("GRANGER CAUSALITY: STANDARD F-TEST vs HAC (NEWEY-WEST) WALD TEST")
    print("=" * 70)

    hml_all = df['HML'].values
    smb_all = df['SMB'].values

    granger_results = {
        'metadata': {
            'n_days': len(df),
            'regime_counts': regime_counts,
            'lag_selection': 'BIC on unrestricted model',
            'hac_method': 'Newey-West (statsmodels cov_type=HAC)',
            'boundary_handling': 'all lags must fall within same regime',
        },
        'regimes': {}
    }

    for regime_id, regime_name in enumerate(regime_names):
        print(f"\n{'='*50}")
        print(f"REGIME {regime_id}: {regime_name} ({regime_counts[regime_name]} total days)")
        print(f"{'='*50}")

        # Get clean indices with max possible lag (15) for initial filter
        # We'll re-filter per lag in BIC selection
        clean_15 = extract_regime_clean_indices(regimes, regime_id, max_lag=15)
        print(f"  Clean obs (max_lag=15): {len(clean_15)}")

        if len(clean_15) < 50:
            print(f"  SKIPPING: insufficient clean observations")
            granger_results['regimes'][regime_name] = {
                'n_days': regime_counts[regime_name],
                'n_clean_max15': len(clean_15),
                'status': 'insufficient_data',
            }
            continue

        # BIC-optimal lag for HML->SMB
        best_lag_hml2smb = select_lag_bic(smb_all, hml_all, smb_all, hml_all, clean_15, max_lag=15)
        # BIC-optimal lag for SMB->HML
        best_lag_smb2hml = select_lag_bic(hml_all, smb_all, hml_all, smb_all, clean_15, max_lag=15)

        print(f"  BIC-optimal lags: HML->SMB={best_lag_hml2smb}, SMB->HML={best_lag_smb2hml}")

        # Get clean indices for the specific optimal lags
        clean_hml2smb = extract_regime_clean_indices(regimes, regime_id, max_lag=best_lag_hml2smb)
        clean_smb2hml = extract_regime_clean_indices(regimes, regime_id, max_lag=best_lag_smb2hml)

        # HML -> SMB
        print(f"\n  Direction: HML -> SMB")
        hml2smb = run_granger_both_methods(
            smb_all, hml_all, clean_hml2smb, best_lag_hml2smb, "HML->SMB"
        )

        # SMB -> HML
        print(f"\n  Direction: SMB -> HML")
        smb2hml = run_granger_both_methods(
            hml_all, smb_all, clean_smb2hml, best_lag_smb2hml, "SMB->HML"
        )

        granger_results['regimes'][regime_name] = {
            'n_days': regime_counts[regime_name],
            'n_clean_max15': len(clean_15),
            'hml_to_smb': hml2smb,
            'smb_to_hml': smb2hml,
        }

    # Step 6: Save results
    granger_path = os.path.join(RESULTS_DIR, 'hac_granger_results.json')
    with open(granger_path, 'w') as f:
        json.dump(granger_results, f, indent=2)
    print(f"\nResults saved to {granger_path}")

    # Step 7: Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY: STANDARD F-TEST vs HAC (NEWEY-WEST) WALD TEST")
    print("=" * 70)
    print(f"\n{'Regime':<12} {'Dir':<12} {'Lag':>4} {'n':>6}  {'F-stat':>8} {'F p-val':>12}  {'W-stat':>8} {'HAC p-val':>12}  {'DeltaR2':>10}")
    print("-" * 100)

    for regime_name in regime_names:
        rd = granger_results['regimes'].get(regime_name, {})
        if rd.get('status') == 'insufficient_data':
            print(f"{regime_name:<12} {'---':<12} {'---':>4} {'---':>6}  {'---':>8} {'---':>12}  {'---':>8} {'---':>12}  {'---':>10}")
            continue

        for direction, key in [('HML->SMB', 'hml_to_smb'), ('SMB->HML', 'smb_to_hml')]:
            d = rd.get(key)
            if d is None:
                continue
            sig_f = "***" if d['standard_p_value'] < 0.001 else ("**" if d['standard_p_value'] < 0.01 else ("*" if d['standard_p_value'] < 0.05 else ""))
            sig_h = "***" if d['hac_p_value'] < 0.001 else ("**" if d['hac_p_value'] < 0.01 else ("*" if d['hac_p_value'] < 0.05 else ""))
            print(f"{regime_name:<12} {direction:<12} {d['best_lag']:>4} {d['n_obs']:>6}  "
                  f"{d['standard_f_stat']:>8.3f} {d['standard_p_value']:>10.2e}{sig_f:>2}  "
                  f"{d['hac_wald_stat']:>8.3f} {d['hac_p_value']:>10.2e}{sig_h:>2}  "
                  f"{d['delta_r2']*100:>9.4f}%")

    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    crisis = granger_results['regimes'].get('Crisis', {})
    if crisis.get('hml_to_smb'):
        d = crisis['hml_to_smb']
        f_sig = d['standard_p_value'] < 0.05
        h_sig = d['hac_p_value'] < 0.05
        if f_sig and h_sig:
            print("  Crisis HML->SMB: Significant under BOTH standard and HAC tests.")
            print("  => Result is robust to heteroskedasticity and autocorrelation.")
        elif f_sig and not h_sig:
            print("  Crisis HML->SMB: Significant under standard F-test but NOT under HAC.")
            print("  => Standard test may be over-rejecting due to serial correlation.")
        elif not f_sig and h_sig:
            print("  Crisis HML->SMB: NOT significant under standard but significant under HAC.")
            print("  => Standard test may be under-powered; HAC captures efficiency gains.")
        else:
            print("  Crisis HML->SMB: NOT significant under either test.")

    print("\nDone.")


if __name__ == '__main__':
    main()
