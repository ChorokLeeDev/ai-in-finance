"""
Regime-Conditional Granger Causality for Macroeconomic Data
============================================================

Tests yield curve slope (T10Y2Y) → industrial production (INDPRO) Granger causality
within regime-conditional framework using HMM trained on macro data.

Data:
  - INDPRO: Industrial Production Index (monthly)
  - T10Y2Y: 10Y-2Y Treasury Spread (daily → resampled monthly)
  - UNRATE: Unemployment Rate (monthly)
  Period: 1990-2024

Three-fold temporal split:
  - Train HMM: 1990-2005 (train regime labels on macro data)
  - Test Granger: 2006-2015 (Fold B: test causality per regime)
  - OOS: 2016-2024 (Fold C: out-of-sample validation)

Motivation: Yield curve slope is known recession predictor.
Expected: Stronger T10Y2Y → INDPRO signal in recession/crisis regime.
"""

import sys
import json
import warnings
import urllib.request
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

def download_macro_data(start_year=1990, end_year=2024):
    """Download macro data from FRED API via pandas_datareader or CSV fallback."""
    try:
        from pandas_datareader.data import DataReader
        print("Downloading FRED data via pandas_datareader...")

        # Try downloading each series
        indpro = DataReader('INDPRO', 'fred', start=f'{start_year}-01-01', end=f'{end_year}-12-31')
        t10y2y = DataReader('T10Y2Y', 'fred', start=f'{start_year}-01-01', end=f'{end_year}-12-31')
        unrate = DataReader('UNRATE', 'fred', start=f'{start_year}-01-01', end=f'{end_year}-12-31')

        return indpro.squeeze(), t10y2y.squeeze(), unrate.squeeze()

    except Exception as e:
        print(f"pandas_datareader failed: {e}")
        print("Falling back to direct CSV download from FRED...")

        return download_macro_data_csv(start_year, end_year)


def download_macro_data_csv(start_year=1990, end_year=2024):
    """Download FRED data via direct CSV download."""
    start_str = f'{start_year}-01-01'
    end_str = f'{end_year}-12-31'

    def fetch_series(series_id):
        url = f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={start_str}&coed={end_str}'
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read().decode('utf-8')
            df = pd.read_csv(io.StringIO(data))
            df['DATE'] = pd.to_datetime(df['DATE'])
            df = df.set_index('DATE')
            return df.iloc[:, 0]
        except Exception as e:
            print(f"Failed to download {series_id}: {e}")
            return None

    print(f"Downloading INDPRO ({start_str} to {end_str})...")
    indpro = fetch_series('INDPRO')

    print(f"Downloading T10Y2Y ({start_str} to {end_str})...")
    t10y2y = fetch_series('T10Y2Y')

    print(f"Downloading UNRATE ({start_str} to {end_str})...")
    unrate = fetch_series('UNRATE')

    return indpro, t10y2y, unrate


def prepare_macro_data(indpro, t10y2y, unrate):
    """Prepare data: convert to growth rates, resample, align frequencies."""
    # INDPRO: monthly log-growth rate
    indpro = pd.Series(indpro, copy=True)
    indpro = indpro.dropna()
    indpro_growth = np.log(indpro / indpro.shift(1)) * 100
    indpro_growth = indpro_growth.dropna()

    # T10Y2Y: daily data, resample to monthly (last business day)
    t10y2y = pd.Series(t10y2y, copy=True)
    t10y2y = t10y2y.dropna()
    # Resample to month-start to match INDPRO and UNRATE
    t10y2y_monthly = t10y2y.resample('MS').last()  # Last business day of month, indexed at month start
    t10y2y_monthly = t10y2y_monthly.dropna()

    # UNRATE: already monthly
    unrate = pd.Series(unrate, copy=True)
    unrate = unrate.dropna()

    # Align all to common index (monthly)
    # All three should have MS frequency now
    idx_list = [indpro_growth.index, t10y2y_monthly.index, unrate.index]
    common_idx = idx_list[0]
    for idx in idx_list[1:]:
        common_idx = common_idx.intersection(idx)

    if len(common_idx) == 0:
        raise ValueError("No common dates found in macro data series")

    df = pd.DataFrame({
        'INDPRO_growth': indpro_growth[common_idx],
        'T10Y2Y': t10y2y_monthly[common_idx],
        'UNRATE': unrate[common_idx]
    })

    df = df.dropna()
    if len(df) == 0:
        raise ValueError("No valid data after alignment and NA removal")

    print(f"Prepared data: {len(df)} monthly observations from {df.index[0].date()} to {df.index[-1].date()}")

    return df


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
    print("=" * 70)
    print("REGIME-CONDITIONAL GRANGER CAUSALITY: MACROECONOMIC DATA")
    print("=" * 70)

    # Download and prepare data
    print("\n[1/5] Downloading macroeconomic data from FRED...")
    indpro, t10y2y, unrate = download_macro_data(1990, 2024)

    print("\n[2/5] Preparing data...")
    df = prepare_macro_data(indpro, t10y2y, unrate)
    print(f"  Data shape: {df.shape}")
    print(f"  Range: {df.index[0].date()} to {df.index[-1].date()}")

    # Define temporal splits
    train_end = '2005-12-31'
    test_start = '2006-01-01'
    test_end = '2015-12-31'
    oos_start = '2016-01-01'
    oos_end = '2024-12-31'

    train_df = df.loc[:train_end]
    test_df = df.loc[test_start:test_end]
    oos_df = df.loc[oos_start:oos_end]

    factor_cols = ['INDPRO_growth', 'T10Y2Y', 'UNRATE']

    print(f"  Train (1990-2005): {len(train_df)} obs")
    print(f"  Test (2006-2015):  {len(test_df)} obs")
    print(f"  OOS (2016-2024):   {len(oos_df)} obs")

    # Fit HMM on training data
    print("\n[3/5] Fitting Student-t HMM on training data...")
    hmm = StudentTHMM(n_regimes=3, n_iter=100, tol=1e-4, random_state=PRIMARY_SEED)
    hmm.fit(train_df[factor_cols].values)
    print(f"  Train log-likelihood: {hmm.log_likelihood_:.2f}")

    # Relabel regimes by training data norm
    print("\n[4/5] Relabeling regimes and running Granger tests...")
    train_raw = hmm.predict(train_df[factor_cols].values, use_filtered=False)
    train_relabeled, remap = relabel_regimes_by_data_norm(train_df, train_raw, factor_cols)

    train_counts = {REGIME_NAMES[k]: int((train_relabeled == k).sum()) for k in range(3)}
    print(f"  Train regime counts: {train_counts}")

    # Apply relabeling to test set
    test_raw, _ = hmm.predict_oos(test_df[factor_cols].values, use_filtered=True)
    test_regimes = apply_train_remap(test_raw, remap)
    test_counts = {REGIME_NAMES[k]: int((test_regimes == k).sum()) for k in range(3)}
    print(f"  Test regime counts: {test_counts}")

    # Apply relabeling to OOS set
    oos_raw, _ = hmm.predict_oos(oos_df[factor_cols].values, use_filtered=True)
    oos_regimes = apply_train_remap(oos_raw, remap)
    oos_counts = {REGIME_NAMES[k]: int((oos_regimes == k).sum()) for k in range(3)}
    print(f"  OOS regime counts: {oos_counts}")

    # Get data arrays
    indpro_test = test_df['INDPRO_growth'].values
    t10y2y_test = test_df['T10Y2Y'].values
    indpro_oos = oos_df['INDPRO_growth'].values
    t10y2y_oos = oos_df['T10Y2Y'].values

    # Test Granger causality per regime on TEST set
    print("\n  Testing Granger causality (T10Y2Y → INDPRO) on Fold B (2006-2015):")
    granger_results = {}
    for k, name in enumerate(REGIME_NAMES):
        clean_idx = extract_regime_clean_indices(test_regimes, k, max_lag=3)
        if len(clean_idx) > 0:
            # Select lag by BIC
            lag = select_lag_bic(indpro_test, t10y2y_test, clean_idx, max_lag=3)
            # Run Granger at selected lag
            result = run_granger_at_lag(indpro_test, t10y2y_test, clean_idx, lag)
            if result is not None:
                granger_results[name] = result
                print(f"    {name} (n={result['n_obs']}, lag={lag}): "
                      f"F-p={result['f_p_value']:.4f}, HAC-p={result['hac_p_value']:.4f}, "
                      f"ΔR²={result['delta_r2']:.4f}")
            else:
                granger_results[name] = None
                print(f"    {name}: Insufficient observations")
        else:
            granger_results[name] = None
            print(f"    {name}: No clean regime observations")

    # OOS validation on TEST set results
    print("\n  OOS validation (Fold C: 2016-2024):")
    oos_granger = {}
    for k, name in enumerate(REGIME_NAMES):
        clean_idx = extract_regime_clean_indices(oos_regimes, k, max_lag=3)
        if len(clean_idx) > 0 and granger_results[name] is not None:
            lag = granger_results[name]['lag']
            result = run_granger_at_lag(indpro_oos, t10y2y_oos, clean_idx, lag)
            if result is not None:
                oos_granger[name] = result
                print(f"    {name} (n={result['n_obs']}, lag={lag}): "
                      f"F-p={result['f_p_value']:.4f}, HAC-p={result['hac_p_value']:.4f}, "
                      f"ΔR²={result['delta_r2']:.4f}")
            else:
                oos_granger[name] = None
                print(f"    {name}: Insufficient observations")
        else:
            oos_granger[name] = None
            print(f"    {name}: Skipped (no test regime or test failed)")

    # Prepare output
    print("\n[5/5] Saving results...")
    output = {
        'description': (
            'Regime-conditional Granger causality for macroeconomic data. '
            'HMM trained 1990-2005 on INDPRO growth, T10Y2Y, UNRATE. '
            'Granger: T10Y2Y → INDPRO tested 2006-2015, validated 2016-2024. '
            'Tests whether yield curve slope predicts industrial production per regime.'
        ),
        'data_source': 'FRED (Federal Reserve Economic Data)',
        'series': {
            'INDPRO': 'Industrial Production Index (log-growth, monthly)',
            'T10Y2Y': '10Y-2Y Treasury Spread (monthly)',
            'UNRATE': 'Unemployment Rate (monthly)'
        },
        'period': {
            'train': '1990-01 to 2005-12',
            'test': '2006-01 to 2015-12',
            'oos': '2016-01 to 2024-12'
        },
        'hmm': {
            'n_regimes': 3,
            'random_state': PRIMARY_SEED,
            'train_log_likelihood': float(hmm.log_likelihood_),
            'train_regime_counts': train_counts,
        },
        'test': {
            'regime_counts': test_counts,
            'granger_t10y2y_to_indpro': granger_results,
        },
        'oos': {
            'regime_counts': oos_counts,
            'granger_t10y2y_to_indpro': oos_granger,
        },
        'regime_names': REGIME_NAMES,
        'timestamp': datetime.now().isoformat(),
    }

    outpath = f"{RESULTS_DIR}/macro_regime_granger.json"
    with open(outpath, 'w') as fout:
        json.dump(output, fout, indent=2)
    print(f"  Saved to {outpath}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nTest Period (2006-2015) Granger Results (T10Y2Y → INDPRO):")
    print("-" * 70)
    for name in REGIME_NAMES:
        res = granger_results[name]
        if res:
            print(f"{name:12s}: F-stat={res['f_stat']:8.3f}  "
                  f"F-p={res['f_p_value']:8.4f}  "
                  f"HAC-p={res['hac_p_value']:8.4f}  "
                  f"ΔR²={res['delta_r2']:8.4f}  "
                  f"n={res['n_obs']}")
        else:
            print(f"{name:12s}: Insufficient data")

    print("\nOOS Period (2016-2024) Granger Results (T10Y2Y → INDPRO):")
    print("-" * 70)
    for name in REGIME_NAMES:
        res = oos_granger[name]
        if res:
            print(f"{name:12s}: F-stat={res['f_stat']:8.3f}  "
                  f"F-p={res['f_p_value']:8.4f}  "
                  f"HAC-p={res['hac_p_value']:8.4f}  "
                  f"ΔR²={res['delta_r2']:8.4f}  "
                  f"n={res['n_obs']}")
        else:
            print(f"{name:12s}: Insufficient data")

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
