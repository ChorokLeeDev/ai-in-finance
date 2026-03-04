"""
Three-Fold Temporal Validation for Regime-Conditional Granger Framework
=========================================================================

Implements proper temporal validation without look-ahead bias:
  - Fold A (1990-2000): Fit Student-t HMM, discover regimes ONLY
  - Fold B (2001-2012): Freeze HMM from A, classify B, run Granger tests
  - Fold C (2013-2024): Freeze HMM from A, classify C, run Granger tests

Each fold+regime reports: F-statistic, HAC p-value (Andrews), ΔR², sample size.
Results saved to results/three_fold_validation.json
"""

import sys
import json
import warnings
import urllib.request
import zipfile
import io
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

warnings.filterwarnings('ignore')

_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = str(_ROOT / 'results')
REGIME_NAMES = ['Normal', 'Elevated', 'Crisis']
PRIMARY_SEED = 28

# =============================================================================
# DATA LOADING
# =============================================================================

def download_ff_data():
    """Download Fama-French 5 factors + Momentum daily data (1990-2024)."""
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
# STUDENT-T HMM (Self-contained, no external imports needed)
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
        """Compute emission (observation) log-likelihoods."""
        T, d = X.shape
        K = self.n_regimes
        log_B = np.zeros((T, K))
        for k in range(K):
            log_B[:, k] = self._mvt_logpdf(X, self.mu[k], self.Sigma[k], self.nu[k])
        return log_B

    def _forward(self, log_B):
        """Forward algorithm."""
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
        """Backward algorithm."""
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
        """E-step: compute posteriors and expectations."""
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
        """Fit HMM using EM algorithm."""
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
        """Predict regime labels on new OOS data using frozen parameters."""
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
    """Get indices where regime t and ALL lags 1..max_lag are in same regime."""
    regime_mask = (regimes == regime_id)
    indices = np.where(regime_mask)[0]
    clean_indices = []
    for idx in indices:
        if idx >= max_lag:
            if all(regimes[idx - l] == regime_id for l in range(1, max_lag + 1)):
                clean_indices.append(idx)
    return np.array(clean_indices) if clean_indices else np.array([], dtype=int)


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

    return relabeled, order


def apply_train_remap(test_raw, remap):
    """Apply train-period relabeling order to test raw regime labels."""
    return np.array([remap[r] for r in test_raw])


def run_granger_at_lag(y_all, x_all, clean_idx, lag=1):
    """Run Granger F-test + HAC at specific lag using clean indices."""
    usable = np.array([idx for idx in clean_idx if idx >= lag])
    if len(usable) < 2 * lag + 10:
        return None

    y_curr = y_all[usable]
    y_lagged = np.column_stack([y_all[usable - i - 1] for i in range(lag)])
    x_lagged = np.column_stack([x_all[usable - i - 1] for i in range(lag)])

    # F-test: restricted (y_lag only) vs unrestricted (y_lag + x_lag)
    n = len(usable)
    X_r = np.column_stack([np.ones(n), y_lagged])
    X_u = np.column_stack([np.ones(n), y_lagged, x_lagged])

    beta_r = np.linalg.lstsq(X_r, y_curr, rcond=None)[0]
    beta_u = np.linalg.lstsq(X_u, y_curr, rcond=None)[0]

    rss_r = np.sum((y_curr - X_r @ beta_r) ** 2)
    rss_u = np.sum((y_curr - X_u @ beta_u) ** 2)

    df1 = lag
    df2 = n - 2 * lag - 1

    if df2 <= 0 or rss_u <= 0:
        return None

    f_stat = ((rss_r - rss_u) / df1) / (rss_u / df2)
    f_p = 1.0 - f_dist.cdf(f_stat, df1, df2)

    # HAC (Newey-West) joint Wald test for all x_lagged coefficients
    p = y_lagged.shape[1] if y_lagged.ndim > 1 else 1
    model = sm.OLS(y_curr, X_u)
    result = model.fit(cov_type='HAC', cov_kwds={'maxlags': lag})
    n_params = X_u.shape[1]
    R = np.zeros((lag, n_params))
    for i in range(lag):
        R[i, 1 + p + i] = 1.0
    beta = result.params
    V = result.cov_params()
    Rb = R @ beta
    RVR = R @ V @ R.T
    try:
        wald_stat = float(Rb @ np.linalg.inv(RVR) @ Rb)
        hac_p = float(1 - chi2.cdf(wald_stat, lag))
    except np.linalg.LinAlgError:
        hac_p = float(result.pvalues[1 + p])  # fallback to single-coeff

    # ΔR²
    tss = np.sum((y_curr - y_curr.mean()) ** 2)
    r2_r = 1.0 - rss_r / tss if tss > 0 else 0.0
    r2_u = 1.0 - rss_u / tss if tss > 0 else 0.0
    delta_r2 = r2_u - r2_r

    return {
        'n_obs': len(usable),
        'lag': lag,
        'f_stat': float(f_stat),
        'f_p_value': float(f_p),
        'hac_p_value': float(hac_p),
        'delta_r2': float(delta_r2),
        'r2_unrestricted': float(r2_u),
    }


# =============================================================================
# THREE-FOLD VALIDATION
# =============================================================================

def run_three_fold_validation(df):
    """
    Temporal three-fold validation:
      Fold A (1990-2000): Fit HMM, discover regimes
      Fold B (2001-2012): Freeze HMM, classify, run Granger tests
      Fold C (2013-2024): Freeze HMM, classify, run Granger tests
    """
    factor_cols = ['MKT', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']

    # Split data
    fold_a_df = df.loc[:'2000-12-31']
    fold_b_df = df.loc['2001-01-01':'2012-12-31']
    fold_c_df = df.loc['2013-01-01':]

    print(f"\nFold A (1990-2000): {len(fold_a_df)} obs")
    print(f"Fold B (2001-2012): {len(fold_b_df)} obs")
    print(f"Fold C (2013-2024): {len(fold_c_df)} obs")

    # --- FOLD A: Fit HMM ---
    print("\n" + "=" * 70)
    print("FOLD A: FITTING STUDENT-T HMM (1990-2000)")
    print("=" * 70)

    hmm_a = StudentTHMM(n_regimes=3, n_iter=100, tol=1e-4, random_state=PRIMARY_SEED)
    hmm_a.fit(fold_a_df[factor_cols].values)
    print(f"  LL = {hmm_a.log_likelihood_:.4f}")

    # Get regime assignments and relabel by data norm
    raw_a = hmm_a.predict(fold_a_df[factor_cols].values, use_filtered=False)
    regimes_a, remap = relabel_regimes_by_data_norm(fold_a_df, raw_a, factor_cols)
    print(f"  Regime relabeling order: {remap}")

    # Count regimes in Fold A
    counts_a = {REGIME_NAMES[k]: int((regimes_a == k).sum()) for k in range(3)}
    print(f"  Regime counts: {counts_a}")

    # --- FOLD B: Freeze HMM, classify, Granger test ---
    print("\n" + "=" * 70)
    print("FOLD B: FROZEN CLASSIFICATION & GRANGER TESTS (2001-2012)")
    print("=" * 70)

    raw_b, _ = hmm_a.predict_oos(fold_b_df[factor_cols].values, use_filtered=True)
    regimes_b = apply_train_remap(raw_b, remap)
    counts_b = {REGIME_NAMES[k]: int((regimes_b == k).sum()) for k in range(3)}
    print(f"  Regime counts: {counts_b}")

    hml_b = fold_b_df['HML'].values
    smb_b = fold_b_df['SMB'].values

    granger_b = {}
    for k, name in enumerate(REGIME_NAMES):
        clean = extract_regime_clean_indices(regimes_b, k, max_lag=1)
        result = run_granger_at_lag(smb_b, hml_b, clean, lag=1)
        granger_b[name] = result if result else {}

    print("\n  Fold B Granger Results (HML → SMB, lag=1):")
    for name in REGIME_NAMES:
        g = granger_b[name]
        if g:
            print(f"    {name}: n={g['n_obs']} F={g['f_stat']:.4f} "
                  f"f_p={g['f_p_value']:.4f} hac_p={g['hac_p_value']:.4f} "
                  f"ΔR²={g['delta_r2']:.4f}")
        else:
            print(f"    {name}: insufficient data")

    # --- FOLD C: Freeze HMM, classify, Granger test ---
    print("\n" + "=" * 70)
    print("FOLD C: FROZEN CLASSIFICATION & GRANGER TESTS (2013-2024)")
    print("=" * 70)

    raw_c, _ = hmm_a.predict_oos(fold_c_df[factor_cols].values, use_filtered=True)
    regimes_c = apply_train_remap(raw_c, remap)
    counts_c = {REGIME_NAMES[k]: int((regimes_c == k).sum()) for k in range(3)}
    print(f"  Regime counts: {counts_c}")

    hml_c = fold_c_df['HML'].values
    smb_c = fold_c_df['SMB'].values

    granger_c = {}
    for k, name in enumerate(REGIME_NAMES):
        clean = extract_regime_clean_indices(regimes_c, k, max_lag=1)
        result = run_granger_at_lag(smb_c, hml_c, clean, lag=1)
        granger_c[name] = result if result else {}

    print("\n  Fold C Granger Results (HML → SMB, lag=1):")
    for name in REGIME_NAMES:
        g = granger_c[name]
        if g:
            print(f"    {name}: n={g['n_obs']} F={g['f_stat']:.4f} "
                  f"f_p={g['f_p_value']:.4f} hac_p={g['hac_p_value']:.4f} "
                  f"ΔR²={g['delta_r2']:.4f}")
        else:
            print(f"    {name}: insufficient data")

    # --- Compile results ---
    results = {
        'description': (
            'Three-fold temporal validation: '
            'Fold A (1990-2000) fits HMM, Folds B & C use frozen HMM. '
            'No look-ahead bias. Granger: HML→SMB, lag=1, HAC p-values.'
        ),
        'folds': {
            'fold_a': {
                'period': '1990-01-02 to 2000-12-29',
                'n_obs': len(fold_a_df),
                'hmm_ll': float(hmm_a.log_likelihood_),
                'regime_counts': counts_a,
                'relabel_order': list(map(int, remap)),
            },
            'fold_b': {
                'period': '2001-01-02 to 2012-12-31',
                'n_obs': len(fold_b_df),
                'regime_counts': counts_b,
                'granger_hml_to_smb': granger_b,
            },
            'fold_c': {
                'period': '2013-01-02 to 2024-12-31',
                'n_obs': len(fold_c_df),
                'regime_counts': counts_c,
                'granger_hml_to_smb': granger_c,
            },
        },
    }

    return results


def print_summary_table(results):
    """Print clear summary table of results."""
    print("\n" + "=" * 100)
    print("THREE-FOLD VALIDATION SUMMARY TABLE")
    print("=" * 100)

    # Header
    print(f"\n{'Fold':8} {'Regime':15} {'Period':25} {'N':6} {'F-stat':10} "
          f"{'F p-val':10} {'HAC p-val':10} {'ΔR²':10}")
    print("-" * 100)

    fold_info = [
        ('B', results['folds']['fold_b']['period'], results['folds']['fold_b']['granger_hml_to_smb']),
        ('C', results['folds']['fold_c']['period'], results['folds']['fold_c']['granger_hml_to_smb']),
    ]

    for fold_label, period, granger_dict in fold_info:
        for regime_name in REGIME_NAMES:
            g = granger_dict.get(regime_name, {})
            if g and 'f_stat' in g:
                print(f"{fold_label:8} {regime_name:15} {period:25} "
                      f"{g['n_obs']:6} {g['f_stat']:10.4f} {g['f_p_value']:10.4f} "
                      f"{g['hac_p_value']:10.4f} {g['delta_r2']:10.4f}")
            else:
                print(f"{fold_label:8} {regime_name:15} {period:25} "
                      f"{'—':6} {'—':10} {'—':10} {'—':10} {'—':10}")

    print("=" * 100)
    print("\nNotes:")
    print("  - Fold A (1990-2000): HMM fitting only, regime discovery")
    print("  - Folds B & C: Frozen HMM from Fold A, regime classification, Granger tests")
    print("  - Granger causality: HML → SMB, lag=1")
    print("  - p-values: F-test (standard) and HAC (Andrews, robust to autocorrelation)")
    print("  - ΔR²: Incremental R² from adding HML lags to SMB model")
    print("  - N: Clean observation count (regimes contiguous at lag window)")


def main():
    print("=" * 70)
    print("THREE-FOLD TEMPORAL VALIDATION FOR REGIME-CONDITIONAL GRANGER")
    print("=" * 70)

    # Download data
    print("\nDownloading Fama-French 5-factor + Momentum data...")
    df = download_ff_data()
    df = df / 100.0  # Convert to decimal

    # Run three-fold validation
    results = run_three_fold_validation(df)

    # Print summary
    print_summary_table(results)

    # Save results
    outpath = f"{RESULTS_DIR}/three_fold_validation.json"
    with open(outpath, 'w') as fout:
        json.dump(results, fout, indent=2)
    print(f"\nResults saved → {outpath}")

    return results


if __name__ == '__main__':
    main()
