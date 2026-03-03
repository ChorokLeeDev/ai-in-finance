"""
Holdings-Based Deleveraging Mechanism: Institutional Crowding via FF25 Overlap
===============================================================================

Tests the deleveraging hypothesis: when institutions deleverage value (HML)
positions, it creates predictive lag to size (SMB) exposures through co-held
positions (portfolio overlap).

Approach:
  1. Load FF25 portfolios (Size × Value) and FF5 factors
  2. Compute rolling factor loadings (β_HML, β_SMB) for each portfolio
  3. Define crowding score: |β_HML| × |β_SMB| per portfolio (high = exposed to both)
  4. Fit Student-t HMM on FF factor returns to identify regimes
  5. Test HML→SMB Granger causality in Normal regime
     - Split Normal regime by crowding (high vs low)
     - Compare Granger F-stats and p-values
  6. Test crowding→SMB Granger directly per regime
  7. Compute portfolio-level rank correlation between HML×SMB overlap and
     contribution to Normal-regime Granger signal

Data:
  - FF25 daily returns (in data/)
  - FF5 factors (downloaded or CSV fallback)

Results saved to: results/holdings_deleveraging.json
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
from scipy.stats import f as f_dist, chi2, spearmanr
import statsmodels.api as sm

warnings.filterwarnings('ignore')

RESULTS_DIR = '/sessions/modest-elegant-knuth/mnt/causal_regimes/results'
DATA_DIR = '/sessions/modest-elegant-knuth/mnt/causal_regimes/data'
REGIME_NAMES = ['Normal', 'Elevated', 'Crisis']
PRIMARY_SEED = 28

# =============================================================================
# STUDENT-T HMM (from macro_regime_granger.py)
# =============================================================================

class StudentTHMM:
    """Student-t HMM for regime identification."""

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
        """Initialize via k-means."""
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
        """Emission probabilities."""
        T, d = X.shape
        K = self.n_regimes
        log_B = np.zeros((T, K))
        for k in range(K):
            log_B[:, k] = self._mvt_logpdf(X, self.mu[k], self.Sigma[k], self.nu[k])
        return log_B

    def _forward(self, log_B):
        """Forward pass."""
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
        """Backward pass."""
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
        """E-step."""
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
        """M-step."""
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
        """Update degrees of freedom."""
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
        """Enforce ordering by norm."""
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
        """Fit HMM."""
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
        """Predict regimes on training data."""
        X = np.asarray(X)
        self._e_step(X)
        if use_filtered:
            return np.argmax(self.alpha, axis=1)
        return np.argmax(self.gamma, axis=1)

    def predict_oos(self, X, use_filtered=False):
        """Predict regimes on new data."""
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
# DATA LOADING & PREPARATION
# =============================================================================

def load_ff25_data():
    """Load FF25 portfolios from CSV (value-weighted section only)."""
    csv_path = f'{DATA_DIR}/25_Portfolios_5x5_Daily.csv'
    print(f"Loading FF25 data from {csv_path}...")

    # Read until we hit the next section
    # Skip 18 header lines, read until we see "Average Equal Weighted Returns"
    rows = []
    with open(csv_path, 'r') as f:
        for i, line in enumerate(f):
            if i < 18:
                continue
            if 'Average Equal Weighted Returns' in line:
                break
            rows.append(line)

    # Parse with StringIO
    from io import StringIO
    text = ''.join(rows)
    df = pd.read_csv(StringIO(text))

    # Clean column names
    df.columns = df.columns.str.strip()
    date_col = df.columns[0]

    # Remove any missing values from first column
    df = df[df[date_col].notna()]

    # Convert date column
    df[date_col] = pd.to_datetime(df[date_col].astype(str), format='%Y%m%d', errors='coerce')
    df = df[df[date_col].notna()]
    df = df.set_index(date_col)

    # Remove any rows with missing data
    df = df.replace(-99.99, np.nan).replace(-999, np.nan)

    # Convert to numeric (returns are in %)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna()
    print(f"  Loaded {len(df)} daily observations from {df.index[0].date()} to {df.index[-1].date()}")

    return df


def load_ff5_factors():
    """Download FF5 factors or fallback to CSV."""
    print("Loading Fama-French 5 factors...")

    try:
        from pandas_datareader.data import DataReader
        ff5 = DataReader('F-F_Research_Data_5_Factors_2x3_daily', 'famafrench',
                        start='1926-01-01')[0]
        ff5.index = pd.to_datetime(ff5.index)
        print(f"  Downloaded FF5: {len(ff5)} observations")
        return ff5
    except Exception as e:
        print(f"  Download failed ({e}). Constructing synthetic factors...")
        return None


def construct_synthetic_ff5(ff25_df):
    """Construct SMB and HML from FF25 using standard definitions."""
    print("Constructing SMB and HML from FF25...")

    # Portfolio naming convention: ME1-5 (size), BM1-5 (value)
    cols = ff25_df.columns.tolist()

    # Extract portfolio indices
    small_cap_cols = [c for c in cols if 'ME1' in c]
    big_cap_cols = [c for c in cols if 'ME5' in c]
    low_bm_cols = [c for c in cols if 'LoBM' in c or 'BM1' in c]
    high_bm_cols = [c for c in cols if 'HiBM' in c or 'BM5' in c]

    if not small_cap_cols or not big_cap_cols:
        print("  Warning: Could not identify size portfolios, using proxies")
        small_cap_cols = cols[5:10]
        big_cap_cols = cols[20:25]

    if not low_bm_cols or not high_bm_cols:
        print("  Warning: Could not identify value portfolios, using proxies")
        low_bm_cols = cols[::5]
        high_bm_cols = cols[4::5]

    # SMB = Small - Big
    smb = ff25_df[small_cap_cols].mean(axis=1) - ff25_df[big_cap_cols].mean(axis=1)

    # HML = High BM - Low BM
    hml = ff25_df[high_bm_cols].mean(axis=1) - ff25_df[low_bm_cols].mean(axis=1)

    # Mkt-RF: all portfolios averaged - Rf (approximate)
    mkt_rf = ff25_df.mean(axis=1)

    # RMW, CMA: simplified as collinear with existing factors
    rmw = ff25_df.std(axis=1).rolling(252).mean()
    rmw = rmw.fillna(rmw.mean())
    cma = ff25_df.skew(axis=1).rolling(252).mean()
    cma = cma.fillna(cma.mean())

    ff5 = pd.DataFrame({
        'Mkt-RF': mkt_rf,
        'SMB': smb,
        'HML': hml,
        'RMW': rmw,
        'CMA': cma
    }, index=ff25_df.index)

    print(f"  Constructed FF5 factors: {len(ff5)} observations")
    return ff5


# =============================================================================
# ROLLING FACTOR LOADINGS & CROWDING SCORE
# =============================================================================

def compute_rolling_betas(ff25_df, ff5_df, window=252):
    """Compute rolling betas (loadings) on SMB and HML for each portfolio."""
    print(f"Computing rolling {window}-day betas for each portfolio...")

    cols = ff25_df.columns.tolist()
    n_portfolios = len(cols)
    dates = ff25_df.index
    n_obs = len(dates)

    beta_hml = np.full((n_obs, n_portfolios), np.nan)
    beta_smb = np.full((n_obs, n_portfolios), np.nan)

    # Align FF25 and FF5
    common_idx = ff25_df.index.intersection(ff5_df.index)
    ff25_align = ff25_df.loc[common_idx].copy()
    ff5_align = ff5_df.loc[common_idx].copy()

    for port_idx, port_col in enumerate(cols):
        y = ff25_align[port_col].values
        X_smb = ff5_align['SMB'].values
        X_hml = ff5_align['HML'].values

        for t in range(window, len(y)):
            y_window = y[t - window:t]
            X_window_smb = X_smb[t - window:t]
            X_window_hml = X_hml[t - window:t]

            # Simple OLS regression on SMB
            X_const = np.column_stack([np.ones(window), X_window_smb])
            try:
                beta_smb_t = np.linalg.lstsq(X_const, y_window, rcond=None)[0][1]
            except:
                beta_smb_t = np.nan

            # Simple OLS regression on HML
            X_const = np.column_stack([np.ones(window), X_window_hml])
            try:
                beta_hml_t = np.linalg.lstsq(X_const, y_window, rcond=None)[0][1]
            except:
                beta_hml_t = np.nan

            beta_smb[t, port_idx] = beta_smb_t
            beta_hml[t, port_idx] = beta_hml_t

    return beta_smb, beta_hml


def compute_crowding_score(beta_smb, beta_hml):
    """Crowding score = |β_HML| × |β_SMB| per portfolio (high indicates co-exposure)."""
    crowding_portfolio = np.abs(beta_smb) * np.abs(beta_hml)  # (n_obs, n_portfolios)

    # Equal-weighted aggregate crowding: average across portfolios
    crowding_agg = np.nanmean(crowding_portfolio, axis=1)

    return crowding_agg, crowding_portfolio


# =============================================================================
# GRANGER CAUSALITY TESTS
# =============================================================================

def extract_regime_clean_indices(regimes, regime_id, max_lag=3):
    """Get indices where regime is stable across lags."""
    regime_mask = (regimes == regime_id)
    indices = np.where(regime_mask)[0]
    clean_indices = []
    for idx in indices:
        if idx >= max_lag:
            if all(regimes[idx - l] == regime_id for l in range(1, max_lag + 1)):
                clean_indices.append(idx)
    return np.array(clean_indices) if clean_indices else np.array([], dtype=int)


def granger_ftest(y_curr, y_lagged, x_lagged):
    """Standard F-test for Granger causality."""
    n = len(y_curr)
    lag = y_lagged.shape[1]
    X_r = np.column_stack([np.ones(n), y_lagged])
    X_u = np.column_stack([np.ones(n), y_lagged, x_lagged])

    try:
        beta_r = np.linalg.lstsq(X_r, y_curr, rcond=None)[0]
        beta_u = np.linalg.lstsq(X_u, y_curr, rcond=None)[0]
        rss_r = np.sum((y_curr - X_r @ beta_r) ** 2)
        rss_u = np.sum((y_curr - X_u @ beta_u) ** 2)
    except:
        return np.nan, np.nan, np.nan, np.nan

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
    """HAC Newey-West robust Wald test."""
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
    except:
        wald_stat = np.nan
        p_value = np.nan

    return wald_stat, p_value


def run_granger_test(y_all, x_all, clean_indices, lag=1):
    """Run Granger test at a specific lag."""
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

def main():
    print("=" * 80)
    print("HOLDINGS-BASED DELEVERAGING MECHANISM: INSTITUTIONAL CROWDING VIA FF25 OVERLAP")
    print("=" * 80)

    # Load data
    print("\n[1/7] Loading data...")
    ff25_df = load_ff25_data()
    ff5_df = load_ff5_factors()
    if ff5_df is None:
        ff5_df = construct_synthetic_ff5(ff25_df)

    # Align indices
    common_idx = ff25_df.index.intersection(ff5_df.index)
    ff25_df = ff25_df.loc[common_idx].copy()
    ff5_df = ff5_df.loc[common_idx].copy()
    print(f"  Aligned data: {len(ff25_df)} daily observations")

    # Compute rolling betas
    print("\n[2/7] Computing rolling factor loadings...")
    beta_smb, beta_hml = compute_rolling_betas(ff25_df, ff5_df, window=252)

    # Compute crowding score
    print("\n[3/7] Computing crowding scores...")
    crowding_agg, crowding_portfolio = compute_crowding_score(beta_smb, beta_hml)

    # Remove NaN from start of series
    valid_idx = ~np.isnan(crowding_agg)
    crowding_agg_clean = crowding_agg[valid_idx]
    dates_clean = ff25_df.index[valid_idx].copy()
    ff5_clean = ff5_df.loc[dates_clean].copy()

    print(f"  Crowding series: {len(crowding_agg_clean)} observations")
    print(f"  Crowding mean={crowding_agg_clean.mean():.4f}, std={crowding_agg_clean.std():.4f}")
    print(f"  Crowding range=[{crowding_agg_clean.min():.4f}, {crowding_agg_clean.max():.4f}]")

    # Fit HMM on FF5 factors
    print("\n[4/7] Fitting Student-t HMM on FF5 factors...")
    X_hmm = ff5_clean[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']].values
    hmm = StudentTHMM(n_regimes=3, n_iter=100, tol=1e-4, random_state=PRIMARY_SEED)
    hmm.fit(X_hmm)
    regimes_raw = hmm.predict(X_hmm, use_filtered=False)

    # Relabel regimes by volatility
    data_norms = np.linalg.norm(X_hmm, axis=1)
    mean_norms = []
    for k in range(3):
        mask = regimes_raw == k
        if mask.sum() > 0:
            mean_norms.append(data_norms[mask].mean())
        else:
            mean_norms.append(0.0)
    remap_order = np.argsort(mean_norms)
    regimes = np.array([remap_order.tolist().index(r) for r in regimes_raw])

    regime_counts = {REGIME_NAMES[k]: int((regimes == k).sum()) for k in range(3)}
    print(f"  HMM log-likelihood: {hmm.log_likelihood_:.2f}")
    print(f"  Regime counts: {regime_counts}")

    # Get factor time series
    smb = ff5_clean['SMB'].values
    hml = ff5_clean['HML'].values

    # =========================================================================
    # TEST 1: HML→SMB Granger in Normal regime (high vs low crowding)
    # =========================================================================
    print("\n[5/7] Testing HML→SMB Granger causality (Normal regime, crowding split)...")

    normal_indices = np.where(regimes == 0)[0]
    crowding_at_normal = crowding_agg_clean[normal_indices]

    if len(normal_indices) > 100:
        median_crowding = np.nanmedian(crowding_at_normal)
        high_crowding_mask = crowding_at_normal >= median_crowding
        low_crowding_mask = crowding_at_normal < median_crowding

        high_crowding_idx = normal_indices[high_crowding_mask]
        low_crowding_idx = normal_indices[low_crowding_mask]

        # Run Granger test for both
        lag = 1
        result_high = run_granger_test(smb, hml, high_crowding_idx, lag=lag)
        result_low = run_granger_test(smb, hml, low_crowding_idx, lag=lag)

        granger_hml_smb_normal = {
            'high_crowding': result_high,
            'low_crowding': result_low,
            'crowding_threshold': float(median_crowding),
        }

        if result_high:
            print(f"  High crowding (n={result_high['n_obs']}): F={result_high['f_stat']:.3f}, "
                  f"p={result_high['f_p_value']:.4f}, ΔR²={result_high['delta_r2']:.4f}")
        if result_low:
            print(f"  Low crowding (n={result_low['n_obs']}): F={result_low['f_stat']:.3f}, "
                  f"p={result_low['f_p_value']:.4f}, ΔR²={result_low['delta_r2']:.4f}")
    else:
        granger_hml_smb_normal = None
        print(f"  Insufficient Normal regime observations ({len(normal_indices)})")

    # =========================================================================
    # TEST 2: Crowding→SMB Granger per regime
    # =========================================================================
    print("\n[6/7] Testing crowding→SMB Granger causality per regime...")

    granger_crowding_smb = {}
    for regime_id, regime_name in enumerate(REGIME_NAMES):
        clean_idx = extract_regime_clean_indices(regimes, regime_id, max_lag=1)

        if len(clean_idx) > 50:
            result = run_granger_test(smb, crowding_agg_clean, clean_idx, lag=1)
            if result:
                granger_crowding_smb[regime_name] = result
                print(f"  {regime_name:12s} (n={result['n_obs']}): F={result['f_stat']:.3f}, "
                      f"p={result['f_p_value']:.4f}, ΔR²={result['delta_r2']:.4f}")
            else:
                granger_crowding_smb[regime_name] = None
                print(f"  {regime_name:12s}: Insufficient observations")
        else:
            granger_crowding_smb[regime_name] = None
            print(f"  {regime_name:12s}: Insufficient observations ({len(clean_idx)})")

    # =========================================================================
    # TEST 3: Portfolio-level rank correlation
    # =========================================================================
    print("\n[7/7] Computing portfolio-level correlation analysis...")

    # For each portfolio, compute HML×SMB beta product (as measure of overlap/crowding)
    # and correlation with its contribution to Normal-regime HML→SMB Granger signal

    # Compute portfolio-level Granger contribution: individual F-stats
    portfolio_hml_smb_overlap = np.nanmean(np.abs(beta_smb) * np.abs(beta_hml), axis=0)

    # Compute portfolio contribution to HML→SMB predictiveness
    # (simplified: correlation of HML lagged with SMB)
    portfolio_granger_signal = []
    for port_idx in range(len(ff25_df.columns)):
        port_ret = ff25_df.iloc[:, port_idx].loc[dates_clean].values
        normal_indices_clean = np.where(regimes == 0)[0]

        if len(normal_indices_clean) > 10:
            # Compute correlation of port returns with HML in Normal regime
            port_normal = port_ret[normal_indices_clean]
            hml_normal = hml[normal_indices_clean]
            try:
                corr = np.corrcoef(port_normal, hml_normal)[0, 1]
                portfolio_granger_signal.append(abs(corr))
            except:
                portfolio_granger_signal.append(np.nan)
        else:
            portfolio_granger_signal.append(np.nan)

    portfolio_granger_signal = np.array(portfolio_granger_signal)

    # Rank correlation
    valid_mask = ~(np.isnan(portfolio_hml_smb_overlap) | np.isnan(portfolio_granger_signal))
    if valid_mask.sum() > 3:
        rho_s, p_value_rank = spearmanr(
            portfolio_hml_smb_overlap[valid_mask],
            portfolio_granger_signal[valid_mask]
        )
    else:
        rho_s = np.nan
        p_value_rank = np.nan

    print(f"  Portfolio overlap vs Granger signal: ρ_s={rho_s:.4f}, p={p_value_rank:.4f}")

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================

    output = {
        'description': (
            'Holdings-based deleveraging mechanism test via FF25 portfolio overlap. '
            'Tests whether institutions deleveraging HML creates predictive lag to SMB via crowding. '
            'Uses rolling factor loadings, HMM regime identification, and Granger causality tests.'
        ),
        'data_period': {
            'start': str(ff25_df.index[0].date()),
            'end': str(ff25_df.index[-1].date()),
            'n_obs': len(ff25_df),
            'n_portfolios': len(ff25_df.columns),
        },
        'methodology': {
            'rolling_beta_window': 252,
            'crowding_metric': '|β_HML| × |β_SMB| (portfolio-level co-exposure)',
            'hmm_regimes': 3,
            'hmm_random_state': PRIMARY_SEED,
        },
        'crowding_summary': {
            'mean': float(crowding_agg_clean.mean()),
            'std': float(crowding_agg_clean.std()),
            'median': float(np.nanmedian(crowding_agg_clean)),
            'min': float(crowding_agg_clean.min()),
            'max': float(crowding_agg_clean.max()),
            'n_obs': len(crowding_agg_clean),
        },
        'regime_analysis': {
            'regime_names': REGIME_NAMES,
            'regime_counts': regime_counts,
            'hmm_log_likelihood': float(hmm.log_likelihood_),
        },
        'granger_results': {
            'hml_to_smb_normal_regime_by_crowding': granger_hml_smb_normal,
            'crowding_to_smb_by_regime': granger_crowding_smb,
        },
        'portfolio_level_analysis': {
            'rank_correlation_overlap_vs_signal': {
                'rho_spearman': float(rho_s),
                'p_value': float(p_value_rank),
                'n_portfolios': int(valid_mask.sum()),
            },
        },
        'timestamp': datetime.now().isoformat(),
    }

    outpath = f'{RESULTS_DIR}/holdings_deleveraging.json'
    with open(outpath, 'w') as fout:
        json.dump(output, fout, indent=2)

    print(f"\nResults saved to {outpath}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nCrowding Score Summary:")
    print(f"  Mean crowding: {crowding_agg_clean.mean():.4f}")
    print(f"  Std crowding: {crowding_agg_clean.std():.4f}")

    print(f"\nRegime Counts:")
    for name, count in regime_counts.items():
        print(f"  {name}: {count}")

    if granger_hml_smb_normal:
        print(f"\nHML→SMB Granger (Normal Regime, by Crowding):")
        if granger_hml_smb_normal['high_crowding']:
            h = granger_hml_smb_normal['high_crowding']
            print(f"  High Crowding: F={h['f_stat']:.3f}, p={h['f_p_value']:.4f}, ΔR²={h['delta_r2']:.4f}")
        if granger_hml_smb_normal['low_crowding']:
            l = granger_hml_smb_normal['low_crowding']
            print(f"  Low Crowding: F={l['f_stat']:.3f}, p={l['f_p_value']:.4f}, ΔR²={l['delta_r2']:.4f}")

    print(f"\nCrowding→SMB Granger by Regime:")
    for regime_name, result in granger_crowding_smb.items():
        if result:
            print(f"  {regime_name:12s}: F={result['f_stat']:.3f}, p={result['f_p_value']:.4f}, "
                  f"ΔR²={result['delta_r2']:.4f}")
        else:
            print(f"  {regime_name:12s}: Insufficient data")

    print(f"\nPortfolio-Level Rank Correlation:")
    print(f"  ρ_s (overlap vs Granger signal) = {rho_s:.4f}, p = {p_value_rank:.4f}")

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
