"""
MOM→SMB Decay Analysis
======================

Compare decay patterns between:
1. HML→SMB (primary finding)
2. MOM→SMB (strongest OOS signal per paper)

Key questions:
- Does MOM→SMB also show decay?
- Is decay a universal phenomenon or HML-specific?
- What's the half-life comparison?
"""

import numpy as np
import pandas as pd
import urllib.request
import zipfile
import io
from scipy.special import gammaln
from scipy.optimize import minimize_scalar, curve_fit
from scipy.cluster.vq import kmeans2
from scipy import stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# DATA
# =============================================================================

def download_ff_factors():
    """Download FF 5 factors + Momentum."""
    print("Downloading Fama-French factors...")

    # 5 factors
    url_5f = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip'
    with urllib.request.urlopen(url_5f, timeout=60) as response:
        data = response.read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        csv_name = z.namelist()[0]
        with z.open(csv_name) as f:
            df_5f = pd.read_csv(f, skiprows=3)

    df_5f.columns = df_5f.columns.str.strip()
    df_5f = df_5f.rename(columns={df_5f.columns[0]: 'Date'})
    df_5f = df_5f[df_5f['Date'].astype(str).str.match(r'^\d{8}$')]
    df_5f['Date'] = pd.to_datetime(df_5f['Date'], format='%Y%m%d')
    for col in ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']:
        df_5f[col] = pd.to_numeric(df_5f[col], errors='coerce')
    df_5f = df_5f.set_index('Date')

    # Momentum
    url_mom = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip'
    with urllib.request.urlopen(url_mom, timeout=60) as response:
        data = response.read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        csv_name = z.namelist()[0]
        with z.open(csv_name) as f:
            df_mom = pd.read_csv(f, skiprows=13)

    df_mom.columns = df_mom.columns.str.strip()
    df_mom = df_mom.rename(columns={df_mom.columns[0]: 'Date', df_mom.columns[1]: 'MOM'})
    df_mom = df_mom[df_mom['Date'].astype(str).str.match(r'^\d{8}$')]
    df_mom['Date'] = pd.to_datetime(df_mom['Date'], format='%Y%m%d')
    df_mom['MOM'] = pd.to_numeric(df_mom['MOM'], errors='coerce')
    df_mom = df_mom.set_index('Date')[['MOM']]

    # Merge
    df = df_5f.join(df_mom, how='inner')
    df = df.dropna(subset=['SMB', 'HML', 'MOM'])

    print(f"  Data: {df.index[0].date()} to {df.index[-1].date()} ({len(df)} days)")
    return df


# =============================================================================
# STUDENT-T HMM
# =============================================================================

class StudentTHMM:
    def __init__(self, n_regimes=3, n_iter=100, tol=1e-4, random_state=28):
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

    def _init_params(self, X):
        np.random.seed(self.random_state)
        T, d = X.shape
        K = self.n_regimes
        centroids, labels = kmeans2(X, K, minit='++')
        norms = np.linalg.norm(centroids, axis=1)
        order = np.argsort(norms)
        centroids = centroids[order]
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
        self.A /= self.A.sum(axis=1, keepdims=True)
        self.pi = np.ones(K) / K

    def _mvt_logpdf(self, x, mu, Sigma, nu):
        d = len(mu)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        diff = x - mu
        Sigma_inv = np.linalg.inv(Sigma + 1e-6 * np.eye(d))
        mahal = np.sum(diff @ Sigma_inv * diff, axis=1)
        sign, logdet = np.linalg.slogdet(Sigma)
        return (gammaln((nu + d) / 2) - gammaln(nu / 2)
                - 0.5 * d * np.log(nu * np.pi) - 0.5 * logdet
                - 0.5 * (nu + d) * np.log(1 + mahal / nu))

    def _forward_backward(self, X):
        T, d = X.shape
        K = self.n_regimes
        log_B = np.zeros((T, K))
        for k in range(K):
            log_B[:, k] = self._mvt_logpdf(X, self.mu[k], self.Sigma[k], self.nu[k])

        log_alpha = np.zeros((T, K))
        log_alpha[0] = np.log(self.pi + 1e-300) + log_B[0]
        log_A = np.log(self.A + 1e-300)
        for t in range(1, T):
            for k in range(K):
                log_alpha[t, k] = np.logaddexp.reduce(log_alpha[t-1] + log_A[:, k]) + log_B[t, k]

        log_beta = np.zeros((T, K))
        for t in range(T - 2, -1, -1):
            for k in range(K):
                log_beta[t, k] = np.logaddexp.reduce(log_A[k, :] + log_B[t+1, :] + log_beta[t+1, :])

        log_gamma = log_alpha + log_beta
        log_gamma -= np.logaddexp.reduce(log_gamma, axis=1, keepdims=True)
        self.gamma = np.exp(log_gamma)

        return np.logaddexp.reduce(log_alpha[-1])

    def fit(self, X):
        X = np.asarray(X)
        self._init_params(X)
        for _ in range(self.n_iter):
            self._forward_backward(X)
            # Simple M-step for means
            for k in range(self.n_regimes):
                weights = self.gamma[:, k]
                self.mu[k] = (weights[:, None] * X).sum(0) / (weights.sum() + 1e-10)
        self._enforce_ordering()
        return self

    def _enforce_ordering(self):
        norms = np.linalg.norm(self.mu, axis=1)
        order = np.argsort(norms)
        self.mu = self.mu[order]
        self.Sigma = self.Sigma[order]
        self.nu = self.nu[order]
        self.A = self.A[order][:, order]
        self.pi = self.pi[order]
        self.gamma = self.gamma[:, order]

    def predict(self, X):
        X = np.asarray(X)
        self._forward_backward(X)
        return np.argmax(self.gamma, axis=1)


# =============================================================================
# GRANGER CAUSALITY WITH HAC
# =============================================================================

def granger_test_hac(Y, X, max_lag=5):
    """
    Granger causality test with HAC standard errors.
    Tests if X Granger-causes Y.
    """
    T = len(Y)

    # Create lagged features
    Y_lags = np.column_stack([Y[max_lag-i-1:T-i-1] for i in range(max_lag)])
    X_lags = np.column_stack([X[max_lag-i-1:T-i-1] for i in range(max_lag)])
    Y_target = Y[max_lag:]

    # Restricted model (only Y lags)
    X_restricted = sm.add_constant(Y_lags)
    model_r = sm.OLS(Y_target, X_restricted).fit(cov_type='HAC', cov_kwds={'maxlags': max_lag})
    rss_r = (model_r.resid ** 2).sum()

    # Unrestricted model (Y lags + X lags)
    X_unrestricted = sm.add_constant(np.column_stack([Y_lags, X_lags]))
    model_u = sm.OLS(Y_target, X_unrestricted).fit(cov_type='HAC', cov_kwds={'maxlags': max_lag})
    rss_u = (model_u.resid ** 2).sum()

    # F-test
    n = len(Y_target)
    k_r = X_restricted.shape[1]
    k_u = X_unrestricted.shape[1]

    F = ((rss_r - rss_u) / (k_u - k_r)) / (rss_u / (n - k_u))
    p_value = 1 - stats.f.cdf(F, k_u - k_r, n - k_u)

    return {'F': F, 'p_value': p_value, 'n': n}


# =============================================================================
# ROLLING DECAY ANALYSIS
# =============================================================================

def rolling_granger_analysis(df, source_col, target_col, regimes, regime_idx=0,
                              window_years=5, step_months=6):
    """
    Rolling window Granger causality analysis within a specific regime.
    """
    results = []

    dates = df.index
    start_year = dates[0].year + window_years
    end_year = dates[-1].year

    for year in range(start_year, end_year + 1):
        for month in [1, 7]:
            window_end = pd.Timestamp(f'{year}-{month:02d}-01')
            window_start = window_end - pd.DateOffset(years=window_years)

            if window_end > dates[-1]:
                continue

            mask = (dates >= window_start) & (dates < window_end) & (regimes == regime_idx)

            if mask.sum() < 100:
                continue

            Y = df.loc[mask, target_col].values
            X = df.loc[mask, source_col].values

            try:
                result = granger_test_hac(Y, X)
                results.append({
                    'window_end': window_end,
                    'year': year + (month - 1) / 12,
                    'F': result['F'],
                    'p_value': result['p_value'],
                    'n': result['n']
                })
            except Exception:
                pass

    return pd.DataFrame(results)


def fit_decay_model(df_rolling):
    """Fit exponential decay to F-statistics."""
    if len(df_rolling) < 5:
        return None

    # Normalize time to start at 0
    t = df_rolling['year'].values - df_rolling['year'].min()
    F = df_rolling['F'].values

    # Exponential decay: F(t) = A * exp(-lambda * t) + C
    def exp_decay(t, A, lam, C):
        return A * np.exp(-lam * t) + C

    try:
        # Initial guess
        p0 = [F.max(), 0.1, F.min()]
        popt, pcov = curve_fit(exp_decay, t, F, p0=p0, maxfev=10000,
                               bounds=([0, 0, 0], [1000, 2, 100]))

        A, lam, C = popt
        half_life = np.log(2) / lam if lam > 0 else np.inf

        # R-squared
        F_pred = exp_decay(t, *popt)
        ss_res = ((F - F_pred) ** 2).sum()
        ss_tot = ((F - F.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot

        return {
            'A': A,
            'lambda': lam,
            'C': C,
            'half_life_years': half_life,
            'r2': r2
        }
    except Exception:
        return None


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("MOM→SMB vs HML→SMB DECAY COMPARISON")
    print("=" * 70)

    # Load data
    df = download_ff_factors()
    df = df[(df.index >= '1990-01-01') & (df.index <= '2024-12-31')]

    # Fit HMM on training period (1990-2012)
    print("\n[1] Fitting HMM on 1990-2012...")
    train_mask = df.index < '2013-01-01'
    X_train = df.loc[train_mask, ['Mkt-RF', 'SMB', 'HML']].values

    hmm = StudentTHMM(n_regimes=3, random_state=28)
    hmm.fit(X_train)

    # Predict regimes for full period
    X_full = df[['Mkt-RF', 'SMB', 'HML']].values
    regimes = hmm.predict(X_full)

    regime_counts = pd.Series(regimes).value_counts().sort_index()
    print(f"  Regime distribution: Normal={regime_counts.get(0, 0)}, "
          f"Elevated={regime_counts.get(1, 0)}, Crisis={regime_counts.get(2, 0)}")

    # Rolling Granger analysis for both pairs
    print("\n[2] Rolling Granger analysis (Normal regime, 5-year windows)...")

    pairs = [
        ('HML', 'SMB', 'HML→SMB'),
        ('MOM', 'SMB', 'MOM→SMB'),
        ('SMB', 'HML', 'SMB→HML'),
        ('SMB', 'MOM', 'SMB→MOM'),
    ]

    all_results = {}
    decay_fits = {}

    for source, target, name in pairs:
        print(f"\n  {name}:")
        rolling_df = rolling_granger_analysis(df, source, target, regimes, regime_idx=0)

        if len(rolling_df) > 0:
            all_results[name] = rolling_df

            # Fit decay model
            decay = fit_decay_model(rolling_df)
            if decay:
                decay_fits[name] = decay
                print(f"    Windows: {len(rolling_df)}")
                print(f"    F range: {rolling_df['F'].min():.2f} - {rolling_df['F'].max():.2f}")
                print(f"    Decay half-life: {decay['half_life_years']:.2f} years (R²={decay['r2']:.3f})")
            else:
                print(f"    Windows: {len(rolling_df)}, decay fit failed")
        else:
            print(f"    Insufficient data")

    # Summary comparison
    print("\n" + "=" * 70)
    print("DECAY COMPARISON SUMMARY")
    print("=" * 70)

    print("\n{:<15} {:>12} {:>12} {:>10} {:>10}".format(
        "Pair", "Half-life", "R²", "Max F", "Min F"))
    print("-" * 60)

    for name in ['HML→SMB', 'MOM→SMB', 'SMB→HML', 'SMB→MOM']:
        if name in decay_fits and name in all_results:
            d = decay_fits[name]
            r = all_results[name]
            print("{:<15} {:>10.2f}y {:>12.3f} {:>10.2f} {:>10.2f}".format(
                name, d['half_life_years'], d['r2'], r['F'].max(), r['F'].min()))

    # Key finding
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    if 'HML→SMB' in decay_fits and 'MOM→SMB' in decay_fits:
        hml_hl = decay_fits['HML→SMB']['half_life_years']
        mom_hl = decay_fits['MOM→SMB']['half_life_years']

        print(f"""
  1. HML→SMB half-life: {hml_hl:.2f} years
  2. MOM→SMB half-life: {mom_hl:.2f} years

  Interpretation:
  - {'Both pairs show similar decay rates' if abs(hml_hl - mom_hl) < 2 else 'Decay rates differ significantly'}
  - {'MOM→SMB decays faster' if mom_hl < hml_hl else 'HML→SMB decays faster'}
  - This {'supports' if abs(hml_hl - mom_hl) < 3 else 'challenges'} the hypothesis that decay is a universal phenomenon
        """)

    # Structural break analysis
    print("\n[3] Structural break analysis (Chow test at 2008)...")

    for name in ['HML→SMB', 'MOM→SMB']:
        if name not in all_results:
            continue

        rolling_df = all_results[name]
        pre_2008 = rolling_df[rolling_df['year'] < 2008]['F']
        post_2008 = rolling_df[rolling_df['year'] >= 2008]['F']

        if len(pre_2008) > 3 and len(post_2008) > 3:
            # Simple t-test for mean difference
            t_stat, p_val = stats.ttest_ind(pre_2008, post_2008)
            print(f"\n  {name}:")
            print(f"    Pre-2008 mean F:  {pre_2008.mean():.2f} (n={len(pre_2008)})")
            print(f"    Post-2008 mean F: {post_2008.mean():.2f} (n={len(post_2008)})")
            print(f"    Difference p-value: {p_val:.2e}")
            print(f"    {'Significant structural break' if p_val < 0.05 else 'No significant break'}")

    return all_results, decay_fits


if __name__ == '__main__':
    results, decay = main()
