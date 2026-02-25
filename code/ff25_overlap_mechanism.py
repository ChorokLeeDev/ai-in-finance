"""
FF 25 Portfolio Overlap Analysis: Mechanism Evidence for HML->SMB Link
======================================================================

Tests the hypothesis that small-value portfolio overlap drives the HML->SMB
Granger causality link. The Small-HighBM portfolio sits in BOTH the SMB long
leg AND the HML long leg, providing a direct mechanism channel.

Analysis:
  1. Download FF 25 Size×BM portfolios (daily, 1990-2024)
  2. Per-portfolio Granger tests (5×5 grid, within Crisis regime)
  3. Spatial gradient test: Spearman correlation between overlap score
     and Granger sensitivity, with permutation p-value
  4. Regime-conditional correlations (overlap vs control portfolios)
  5. Overlap fraction decomposition

Multiple-testing control:
  - 25 Granger tests → Bonferroni correction at α = 0.05/25 = 0.002
  - Pre-specified lag = 1 (BIC-selected in main analysis)
  - Spatial permutation test (10,000 shuffles) as primary inferential test

Outputs:
  - results/ff25_overlap_results.json
  - figures/ff25_overlap_granger_heatmap.pdf
  - data/25_Portfolios_5x5_Daily.csv
"""

import numpy as np
import pandas as pd
import json
import hashlib
import urllib.request
import zipfile
import io
import os
import sys
from datetime import datetime
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
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


# =============================================================================
# DATA LOADING
# =============================================================================

def download_ff5_factors():
    """Download FF 5 factors daily."""
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
    print(f"  Loaded {len(df)} trading days")
    return df


def download_ff25_portfolios():
    """Download FF 25 Size×BM portfolios (daily, value-weighted returns)."""
    print("Downloading FF 25 Size×BM portfolios (daily)...")
    url = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/25_Portfolios_5x5_Daily_CSV.zip'
    retrieval_date = datetime.now().strftime('%Y-%m-%d')

    with urllib.request.urlopen(url, timeout=60) as response:
        raw_data = response.read()

    # Compute SHA256 of raw ZIP
    sha256 = hashlib.sha256(raw_data).hexdigest()

    with zipfile.ZipFile(io.BytesIO(raw_data)) as z:
        csv_name = z.namelist()[0]
        with z.open(csv_name) as f:
            raw_csv = f.read()
            # Save raw CSV
            csv_path = os.path.join(DATA_DIR, '25_Portfolios_5x5_Daily.csv')
            with open(csv_path, 'wb') as out:
                out.write(raw_csv)

    # Parse the CSV: find the header line with "SMALL LoBM" or similar
    # The file has a text header, then "Average Value Weighted Returns -- Daily",
    # then the actual data header + data rows, then possibly equal-weighted section.
    lines = raw_csv.decode('utf-8', errors='replace').split('\n')

    # Find the FIRST header line (contains column names like "SMALL LoBM")
    # There are two sections: Value Weighted and Equal Weighted. Take only the first.
    header_idx = None
    for i, line in enumerate(lines):
        if 'SMALL' in line and 'BM' in line:
            header_idx = i
            break

    if header_idx is None:
        raise ValueError("Could not find FF25 header line")

    # Find the end of the first section (blank line or next section header)
    end_idx = len(lines)
    for i in range(header_idx + 2, len(lines)):
        stripped = lines[i].strip()
        # Stop at blank line, "Equal" header, or non-date line after data has started
        if stripped == '' or 'Equal' in stripped or 'Average' in stripped:
            end_idx = i
            break

    from io import StringIO
    data_text = '\n'.join(lines[header_idx:end_idx])
    df = pd.read_csv(StringIO(data_text))
    df.columns = df.columns.str.strip()
    df = df.rename(columns={df.columns[0]: 'Date'})

    # Filter to valid 8-digit date rows (stops at equal-weighted section or blanks)
    df['Date'] = df['Date'].astype(str).str.strip()
    df = df[df['Date'].str.match(r'^\d{8}$')]
    df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')

    # Convert all portfolio columns to numeric
    for col in df.columns:
        if col != 'Date':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Replace -99.99 and -999 with NaN
    df = df.replace([-99.99, -999], np.nan)
    df = df.set_index('Date').sort_index()
    df = df.loc['1990-01-01':'2024-12-31']
    df = df.dropna(how='all')

    print(f"  FF25 columns: {list(df.columns[:5])} ... (total: {len(df.columns)})")
    print(f"  FF25 shape: {df.shape}")
    print(f"  Date range: {df.index[0].date()} to {df.index[-1].date()}")

    provenance = {
        'url': url,
        'retrieval_date': retrieval_date,
        'sha256': sha256,
        'csv_path': csv_path,
    }

    return df, provenance


# =============================================================================
# STUDENT-T HMM (from critical_fixes_analysis.py)
# =============================================================================

class StudentTHMM:
    def __init__(self, n_regimes=3, n_iter=100, tol=1e-4, random_state=42):
        self.n_regimes = n_regimes; self.n_iter = n_iter
        self.tol = tol; self.random_state = random_state
        self.mu = None; self.Sigma = None; self.nu = None
        self.A = None; self.pi = None
        self.gamma = None; self.alpha = None; self.log_likelihood_ = None

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
            if mask.sum() > d: self.Sigma[k] = np.cov(X[mask].T) + 1e-6 * np.eye(d)
            else: self.Sigma[k] = np.eye(d)
        self.nu = np.array([15.0, 7.0, 4.0])
        self.A = np.eye(K) * 0.95 + np.ones((K, K)) * 0.05 / K
        self.A = self.A / self.A.sum(axis=1, keepdims=True)
        self.pi = np.ones(K) / K

    def _mvt_logpdf(self, x, mu, Sigma, nu):
        d = len(mu)
        if x.ndim == 1: x = x.reshape(1, -1)
        diff = x - mu; Sigma_inv = np.linalg.inv(Sigma)
        mahal = np.sum(diff @ Sigma_inv * diff, axis=1)
        sign, logdet = np.linalg.slogdet(Sigma)
        return (gammaln((nu + d) / 2) - gammaln(nu / 2)
                - 0.5 * d * np.log(nu * np.pi) - 0.5 * logdet
                - 0.5 * (nu + d) * np.log(1 + mahal / nu))

    def _compute_emission_probs(self, X):
        T, d = X.shape; K = self.n_regimes
        log_B = np.zeros((T, K))
        for k in range(K): log_B[:, k] = self._mvt_logpdf(X, self.mu[k], self.Sigma[k], self.nu[k])
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
        log_alpha = self._forward(log_B); log_beta = self._backward(log_B)
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
            diff = X - self.mu[k]; Sigma_inv = np.linalg.inv(self.Sigma[k])
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
            diff = X - self.mu[k]; weights = self.gamma[:, k] * self.u[:, k]
            weighted_outer = np.zeros((d, d))
            for t in range(T): weighted_outer += weights[t] * np.outer(diff[t], diff[t])
            self.Sigma[k] = weighted_outer / self.gamma[:, k].sum()
            self.Sigma[k] += 1e-6 * np.eye(d)
        for k in range(K): self._update_nu(X, k)
        self._enforce_ordering()

    def _update_nu(self, X, k):
        T, d = X.shape
        def neg_expected_ll(nu):
            if nu <= 2: return 1e10
            diff = X - self.mu[k]; Sigma_inv = np.linalg.inv(self.Sigma[k])
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

    def predict(self, X, use_filtered=False):
        X = np.asarray(X); self._e_step(X)
        if use_filtered: return np.argmax(self.alpha, axis=1)
        return np.argmax(self.gamma, axis=1)


# =============================================================================
# GRANGER CAUSALITY
# =============================================================================

def granger_test_manual(x, y, lag=1):
    """Granger test at specific lag. Returns p-value, F-stat, delta_R2."""
    n = len(x)
    if n - lag < lag * 2 + 10:
        return 1.0, 0.0, 0.0

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
            p_value = 1 - stats.f.cdf(f_stat, df1, df2)
            tss = np.sum((y_curr - y_curr.mean()) ** 2)
            delta_r2 = (rss_r - rss_u) / tss if tss > 0 else 0
            return float(p_value), float(f_stat), float(delta_r2)
    except Exception:
        pass
    return 1.0, 0.0, 0.0


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():
    print("=" * 78)
    print("FF 25 PORTFOLIO OVERLAP MECHANISM ANALYSIS")
    print("=" * 78)

    # ---- 1. Load data ----
    df_factors = download_ff5_factors()
    df_25, provenance = download_ff25_portfolios()

    print(f"\n  Data provenance:")
    print(f"    URL: {provenance['url']}")
    print(f"    Retrieved: {provenance['retrieval_date']}")
    print(f"    SHA256: {provenance['sha256'][:16]}...")

    # ---- 2. Align dates ----
    # Remove any duplicate index entries in df_25
    df_25 = df_25[~df_25.index.duplicated(keep='first')]
    common_dates = df_factors.index.intersection(df_25.index)
    df_factors = df_factors.loc[common_dates]
    df_25 = df_25.loc[common_dates]
    print(f"\n  Common dates: {len(common_dates)} ({common_dates[0].date()} to {common_dates[-1].date()})")

    # Name the 25 portfolios systematically
    # FF 25 columns are: SMALL LoBM, ME1 BM2, ME1 BM3, ME1 BM4, SMALL HiBM,
    #                     ME2 BM1, ME2 BM2, ..., BIG LoBM, ME5 BM2, ..., BIG HiBM
    # Order: Size 1 (Small) × BM 1-5, Size 2 × BM 1-5, ..., Size 5 (Big) × BM 1-5
    size_labels = ['S', '2', '3', '4', 'B']
    bm_labels = ['L', '2', '3', '4', 'H']
    portfolio_names = []
    for si in range(5):
        for bi in range(5):
            portfolio_names.append(f"{size_labels[si]}/{bm_labels[bi]}")

    # Map existing columns to our naming
    actual_cols = [c for c in df_25.columns if 'BM' in c or 'SMALL' in c.upper() or 'BIG' in c.upper() or 'ME' in c.upper()]
    if len(actual_cols) < 25:
        # Try using first 25 numeric columns
        actual_cols = list(df_25.columns[:25])

    if len(actual_cols) >= 25:
        cols_25 = actual_cols[:25]
        col_map = dict(zip(cols_25, portfolio_names))
        df_25 = df_25[cols_25].rename(columns=col_map)
        print(f"  Mapped {len(cols_25)} columns to portfolio names")
    else:
        print(f"  WARNING: Expected 25 columns, got {len(actual_cols)}")
        print(f"  Columns: {list(df_25.columns)}")
        return

    print(f"  Portfolio names: {portfolio_names[:5]}...{portfolio_names[-5:]}")

    # ---- 3. Fit HMM on full sample for regime assignments ----
    print("\n  Fitting Student-t HMM (K=3) on full sample...")
    factor_cols = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
    X = df_factors[factor_cols].values
    hmm = StudentTHMM(n_regimes=3, n_iter=100, tol=1e-4, random_state=42)
    hmm.fit(X)
    regimes = hmm.predict(X, use_filtered=False)

    regime_names = {0: 'Normal', 1: 'Elevated', 2: 'Crisis'}
    for k in range(3):
        n = (regimes == k).sum()
        print(f"    Regime {k} ({regime_names[k]}): {n} days ({n/len(regimes)*100:.1f}%)")

    # ---- 4. Define overlap scores ----
    # Overlap score: higher = more overlap with SMB long (small) AND HML long (high BM)
    # overlap_score(i,j) = (5 - size_quintile) + BM_quintile
    # size_quintile: 0=Small to 4=Big; BM_quintile: 0=Low to 4=High
    overlap_scores = np.zeros((5, 5))
    for si in range(5):
        for bi in range(5):
            overlap_scores[si, bi] = (4 - si) + bi  # Small=4, Big=0; LowBM=0, HighBM=4
    # Range: 0 (B/L) to 8 (S/H)

    print(f"\n  Overlap score matrix (higher = more overlap):")
    print(f"  {'':>6}", end='')
    for bi in range(5):
        print(f" {bm_labels[bi]:>5}", end='')
    print()
    for si in range(5):
        print(f"  {size_labels[si]:>6}", end='')
        for bi in range(5):
            print(f" {overlap_scores[si, bi]:>5.0f}", end='')
        print()

    # ---- 5. Per-portfolio Granger tests (HML -> portfolio) ----
    # For each of the 25 portfolios, test if HML Granger-causes that portfolio
    # Use lag=1 (BIC-selected in main analysis) as primary
    print("\n" + "=" * 78)
    print("PER-PORTFOLIO GRANGER TESTS (HML -> Portfolio, lag=1)")
    print("=" * 78)

    hml = df_factors['HML'].values
    granger_results = np.zeros((5, 5))  # p-values
    granger_fstats = np.zeros((5, 5))
    granger_dr2 = np.zeros((5, 5))
    granger_results_detail = {}

    # Extract Crisis regime data
    crisis_mask = regimes == 2
    crisis_indices = np.where(crisis_mask)[0]

    # For regime-conditional Granger: only use crisis days where lag is also crisis
    lag_primary = 1  # BIC-selected

    for si in range(5):
        for bi in range(5):
            pname = f"{size_labels[si]}/{bm_labels[bi]}"
            port_returns = df_25[pname].values

            # Extract clean crisis observations (all lags within crisis)
            clean_idx = []
            for idx in crisis_indices:
                if idx >= lag_primary:
                    all_in = all(regimes[idx - l] == 2 for l in range(1, lag_primary + 1))
                    if all_in:
                        clean_idx.append(idx)

            if len(clean_idx) < 30:
                granger_results[si, bi] = 1.0
                granger_fstats[si, bi] = 0.0
                granger_dr2[si, bi] = 0.0
                granger_results_detail[pname] = {'n_obs': len(clean_idx), 'insufficient': True}
                continue

            clean_idx = np.array(clean_idx)
            hml_clean = hml[clean_idx]
            port_clean = port_returns[clean_idx]

            p_val, f_stat, dr2 = granger_test_manual(hml_clean, port_clean, lag=lag_primary)

            granger_results[si, bi] = p_val
            granger_fstats[si, bi] = f_stat
            granger_dr2[si, bi] = dr2

            granger_results_detail[pname] = {
                'n_obs': len(clean_idx),
                'p_value': round(float(p_val), 6),
                'f_stat': round(float(f_stat), 4),
                'delta_r2': round(float(dr2), 6),
                'bonferroni_significant': p_val < 0.002,  # 0.05/25
            }

    # Print results as grid
    bonferroni_alpha = 0.05 / 25
    print(f"\n  Granger p-values (HML -> Portfolio, Crisis regime, lag={lag_primary}):")
    print(f"  Bonferroni threshold: {bonferroni_alpha:.4f}")
    print(f"  {'':>6}", end='')
    for bi in range(5):
        print(f" {bm_labels[bi]:>8}", end='')
    print()
    for si in range(5):
        print(f"  {size_labels[si]:>6}", end='')
        for bi in range(5):
            p = granger_results[si, bi]
            star = "**" if p < bonferroni_alpha else "*" if p < 0.05 else ""
            print(f" {p:>6.4f}{star:>2}", end='')
        print()

    # ---- 6. Spatial gradient test (primary confirmatory) ----
    print("\n" + "=" * 78)
    print("SPATIAL GRADIENT TEST (PRIMARY CONFIRMATORY)")
    print("=" * 78)

    # Flatten overlap scores and -log10(p-values)
    overlap_flat = overlap_scores.flatten()
    p_clipped = np.maximum(granger_results.flatten(), 1e-300)
    neg_log_p = -np.log10(p_clipped)

    # Observed Spearman correlation
    rho_obs, p_spearman = stats.spearmanr(overlap_flat, neg_log_p)
    print(f"\n  Observed Spearman rho: {rho_obs:.4f}")
    print(f"  Spearman p-value (asymptotic): {p_spearman:.4f}")

    # Spatial permutation test (10,000 shuffles)
    n_perms = 10000
    np.random.seed(42)
    rho_perms = np.zeros(n_perms)
    for perm_i in range(n_perms):
        # Shuffle the assignment of Granger p-values to grid positions
        shuffled_neg_log_p = neg_log_p.copy()
        np.random.shuffle(shuffled_neg_log_p)
        rho_perms[perm_i], _ = stats.spearmanr(overlap_flat, shuffled_neg_log_p)

    perm_p_value = np.mean(rho_perms >= rho_obs)
    print(f"  Spatial permutation p-value (10,000 shuffles): {perm_p_value:.4f}")
    print(f"  Permutation distribution: mean={rho_perms.mean():.4f}, std={rho_perms.std():.4f}")
    print(f"  Observed rho is {(rho_obs - rho_perms.mean()) / rho_perms.std():.1f} SD above permutation mean")

    if perm_p_value < 0.05:
        print(f"  >>> SIGNIFICANT: Spatial gradient confirmed (p < 0.05)")
    else:
        print(f"  >>> NOT SIGNIFICANT at p < 0.05")

    # Also compute Kendall tau for robustness
    tau_obs, p_kendall = stats.kendalltau(overlap_flat, neg_log_p)
    print(f"\n  Kendall tau: {tau_obs:.4f}, p={p_kendall:.4f}")

    # ---- 7. Regime-conditional correlations ----
    print("\n" + "=" * 78)
    print("REGIME-CONDITIONAL CORRELATIONS")
    print("=" * 78)

    # Overlap portfolios: Small/HighBM (S/H), S/4, 2/H
    # Control portfolios: Big/LowBM (B/L), B/2, 4/L
    overlap_ports = ['S/H', 'S/4', '2/H']
    control_ports = ['B/L', 'B/2', '4/L']

    regime_corr_results = {}
    for regime_id in range(3):
        r_mask = regimes == regime_id
        r_dates = common_dates[r_mask]
        r_hml = hml[r_mask]
        r_smb = df_factors['SMB'].values[r_mask]

        regime_corr_results[regime_names[regime_id]] = {
            'n_days': int(r_mask.sum()),
            'overlap': {},
            'control': {},
        }

        print(f"\n  {regime_names[regime_id]} regime ({r_mask.sum()} days):")
        print(f"    Overlap portfolios (S/H, S/4, 2/H) — corr with HML:")
        for pname in overlap_ports:
            port = df_25[pname].values[r_mask]
            corr, p = stats.pearsonr(r_hml, port)
            regime_corr_results[regime_names[regime_id]]['overlap'][pname] = {
                'corr_with_hml': round(float(corr), 4),
                'p_value': round(float(p), 6),
            }
            print(f"      {pname}: r={corr:.4f}, p={p:.2e}")

        print(f"    Control portfolios (B/L, B/2, 4/L) — corr with HML:")
        for pname in control_ports:
            port = df_25[pname].values[r_mask]
            corr, p = stats.pearsonr(r_hml, port)
            regime_corr_results[regime_names[regime_id]]['control'][pname] = {
                'corr_with_hml': round(float(corr), 4),
                'p_value': round(float(p), 6),
            }
            print(f"      {pname}: r={corr:.4f}, p={p:.2e}")

        # Average overlap vs control
        avg_overlap = np.mean([abs(regime_corr_results[regime_names[regime_id]]['overlap'][p]['corr_with_hml'])
                              for p in overlap_ports])
        avg_control = np.mean([abs(regime_corr_results[regime_names[regime_id]]['control'][p]['corr_with_hml'])
                              for p in control_ports])
        regime_corr_results[regime_names[regime_id]]['avg_overlap_corr'] = round(float(avg_overlap), 4)
        regime_corr_results[regime_names[regime_id]]['avg_control_corr'] = round(float(avg_control), 4)
        print(f"    Avg |corr|: overlap={avg_overlap:.4f}, control={avg_control:.4f}")

    # ---- 8. Overlap fraction decomposition ----
    print("\n" + "=" * 78)
    print("OVERLAP FRACTION DECOMPOSITION (lag=15, Crisis)")
    print("=" * 78)

    lag_decomp = 15
    # ΔR²(HML→Small-HighBM, lag 15) / ΔR²(HML→SMB, lag 15)
    smb_full = df_factors['SMB'].values
    sh_port = df_25['S/H'].values

    # Extract clean crisis observations for lag 15
    clean_idx_15 = []
    for idx in crisis_indices:
        if idx >= lag_decomp:
            all_in = all(regimes[idx - l] == 2 for l in range(1, lag_decomp + 1))
            if all_in:
                clean_idx_15.append(idx)
    clean_idx_15 = np.array(clean_idx_15) if len(clean_idx_15) > 0 else np.array([], dtype=int)

    overlap_fraction = None
    if len(clean_idx_15) >= 30:
        hml_c15 = hml[clean_idx_15]
        smb_c15 = smb_full[clean_idx_15]
        sh_c15 = sh_port[clean_idx_15]

        _, _, dr2_hml_smb = granger_test_manual(hml_c15, smb_c15, lag=lag_decomp)
        _, _, dr2_hml_sh = granger_test_manual(hml_c15, sh_c15, lag=lag_decomp)

        overlap_fraction = dr2_hml_sh / dr2_hml_smb if dr2_hml_smb > 0 else None
        print(f"  Clean crisis obs (lag 15): {len(clean_idx_15)}")
        print(f"  ΔR²(HML→SMB, lag 15):    {dr2_hml_smb:.6f}")
        print(f"  ΔR²(HML→S/H, lag 15):    {dr2_hml_sh:.6f}")
        if overlap_fraction is not None:
            print(f"  Overlap fraction:          {overlap_fraction:.2f} ({overlap_fraction*100:.0f}%)")
    else:
        print(f"  Insufficient clean crisis observations for lag 15: {len(clean_idx_15)}")

    # ---- 9. Generate heatmap figure ----
    print("\n  Generating heatmap figure...")
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Panel A: Granger p-value heatmap
        ax1 = axes[0]
        neg_log_p_grid = -np.log10(np.maximum(granger_results, 1e-300))
        im1 = ax1.imshow(neg_log_p_grid, cmap='YlOrRd', aspect='equal',
                         vmin=0, vmax=max(5, neg_log_p_grid.max()))
        ax1.set_xticks(range(5)); ax1.set_xticklabels(bm_labels)
        ax1.set_yticks(range(5)); ax1.set_yticklabels(size_labels)
        ax1.set_xlabel('Book-to-Market Quintile')
        ax1.set_ylabel('Size Quintile')
        ax1.set_title(f'Panel A: $-\\log_{{10}}(p)$ for HML $\\to$ Portfolio\n(Crisis, lag={lag_primary})')

        # Add text annotations
        for si in range(5):
            for bi in range(5):
                p = granger_results[si, bi]
                text = f'{neg_log_p_grid[si, bi]:.1f}'
                if p < bonferroni_alpha:
                    text += '\n**'
                elif p < 0.05:
                    text += '\n*'
                color = 'white' if neg_log_p_grid[si, bi] > 2.5 else 'black'
                ax1.text(bi, si, text, ha='center', va='center', fontsize=8, color=color)

        plt.colorbar(im1, ax=ax1, label='$-\\log_{10}(p)$', shrink=0.8)

        # Panel B: Overlap score vs Granger significance
        ax2 = axes[1]
        scatter_overlap = overlap_scores.flatten()
        scatter_neglogp = neg_log_p_grid.flatten()

        # Color by size quintile
        colors = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#3498db']
        for si in range(5):
            for bi in range(5):
                idx = si * 5 + bi
                ax2.scatter(scatter_overlap[idx], scatter_neglogp[idx],
                           c=colors[si], s=80, edgecolors='black', linewidth=0.5,
                           label=f'{size_labels[si]}' if bi == 0 else '')
                ax2.annotate(f'{size_labels[si]}/{bm_labels[bi]}',
                           (scatter_overlap[idx], scatter_neglogp[idx]),
                           fontsize=6, ha='left', va='bottom',
                           xytext=(3, 3), textcoords='offset points')

        # Add regression line
        z = np.polyfit(scatter_overlap, scatter_neglogp, 1)
        x_line = np.linspace(0, 8, 100)
        ax2.plot(x_line, np.polyval(z, x_line), 'k--', alpha=0.5, linewidth=1)

        ax2.set_xlabel('Overlap Score (higher = more dual-factor exposure)')
        ax2.set_ylabel('$-\\log_{10}(p)$ (Granger significance)')
        ax2.set_title(f'Panel B: Spatial Gradient\n$\\rho_s={rho_obs:.3f}$, perm. $p={perm_p_value:.3f}$')
        ax2.legend(title='Size', fontsize=8, loc='upper left')

        plt.tight_layout()
        fig_path = os.path.join(FIGURES_DIR, 'ff25_overlap_granger_heatmap.pdf')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Figure saved to: {fig_path}")
    except ImportError as e:
        print(f"  Could not generate figure: {e}")

    # ---- 10. Save results ----
    p_grid_raw = granger_results.tolist()
    p_grid_clipped = np.maximum(granger_results, 1e-300).tolist()
    neg_log10_grid = (-np.log10(np.maximum(granger_results, 1e-300))).tolist()

    results = {
        'description': 'FF 25 Portfolio Overlap Mechanism Analysis',
        'data_provenance': provenance,
        'sample': f'{common_dates[0].date()} to {common_dates[-1].date()}',
        'n_days': len(common_dates),
        'overlap_score_definition': '(4 - size_quintile) + BM_quintile; range 0-8; S/H=8, B/L=0',
        'primary_test': {
            'name': 'Spatial gradient test',
            'method': 'Spearman rank correlation between overlap score and -log10(Granger p-value)',
            'spearman_rho': round(float(rho_obs), 4),
            'spearman_p_asymptotic': round(float(p_spearman), 4),
            'permutation_p_value': round(float(perm_p_value), 4),
            'n_permutations': n_perms,
            'kendall_tau': round(float(tau_obs), 4),
            'kendall_p': round(float(p_kendall), 4),
            'significant_at_005': bool(perm_p_value < 0.05),
        },
        'granger_tests': {
            'lag': lag_primary,
            'regime': 'Crisis',
            'bonferroni_threshold': round(float(bonferroni_alpha), 4),
            'per_portfolio': granger_results_detail,
            'p_value_grid_raw': p_grid_raw,
            'p_value_grid_clipped': p_grid_clipped,
            'neg_log10_p_grid': neg_log10_grid,
        },
        'regime_conditional_correlations': regime_corr_results,
        'overlap_decomposition': {
            'lag': lag_decomp,
            'n_clean_crisis_obs': len(clean_idx_15),
            'delta_r2_hml_smb': round(float(dr2_hml_smb), 6) if len(clean_idx_15) >= 30 else None,
            'delta_r2_hml_sh': round(float(dr2_hml_sh), 6) if len(clean_idx_15) >= 30 else None,
            'overlap_fraction': round(float(overlap_fraction), 4) if overlap_fraction is not None else None,
        },
    }

    output_path = os.path.join(RESULTS_DIR, 'ff25_overlap_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {output_path}")

    # ---- Summary ----
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  Primary test: Spearman rho = {rho_obs:.4f}, permutation p = {perm_p_value:.4f}")
    n_bonf_sig = sum(1 for si in range(5) for bi in range(5) if granger_results[si, bi] < bonferroni_alpha)
    n_nom_sig = sum(1 for si in range(5) for bi in range(5) if granger_results[si, bi] < 0.05)
    print(f"  Bonferroni-significant portfolios: {n_bonf_sig}/25")
    print(f"  Nominally significant (p<0.05): {n_nom_sig}/25")
    if overlap_fraction is not None:
        print(f"  Overlap fraction: {overlap_fraction:.2f}")
    print("\n  Done.")


if __name__ == '__main__':
    main()
