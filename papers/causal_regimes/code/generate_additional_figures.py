"""
Generate Additional Figures for Causal Regimes Paper
=====================================================

Addresses reviewer concerns about sparse visualization with 3 publication-quality figures:
  A. All-pairs Granger causality heatmap (3 panels, one per regime)
  B. Lag sensitivity profile for HML->SMB in Crisis regime
  C. Rolling Granger p-value time series with regime-colored background

Usage:
    /usr/bin/python3 -u generate_additional_figures.py
"""

import numpy as np
import pandas as pd
import urllib.request
import zipfile
import io
import warnings
from scipy import stats
from scipy.special import gammaln
from scipy.optimize import minimize_scalar
from scipy.cluster.vq import kmeans2
from statsmodels.tsa.stattools import grangercausalitytests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle

warnings.filterwarnings('ignore')

FIGURES_DIR = '/Users/i767700/Github/ai-in-finance/papers/causal_regimes/figures'

# =============================================================================
# Publication-quality matplotlib settings
# =============================================================================
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'text.usetex': False,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
})

# =============================================================================
# StudentTHMM (copied from critical_fixes_analysis.py)
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
# DATA LOADING
# =============================================================================

def download_ff_data():
    """Download Fama-French 5 factors daily data."""
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
    df5 = df5.dropna()
    df5 = df5.set_index('Date').sort_index()

    # Rename for convenience
    df5 = df5.rename(columns={'Mkt-RF': 'MKT'})
    print(f"  Loaded {len(df5)} daily observations from {df5.index[0].date()} to {df5.index[-1].date()}")
    return df5


def granger_pvalue(data_x, data_y, maxlag):
    """
    Compute Granger causality p-value for x -> y at a specific lag.
    Returns the minimum p-value across F-test variants at that lag.
    """
    try:
        df_test = pd.DataFrame({'y': data_y, 'x': data_x}).dropna()
        if len(df_test) < maxlag + 20:
            return 1.0
        result = grangercausalitytests(df_test[['y', 'x']], maxlag=maxlag, verbose=False)
        # Get p-value at the specified lag
        p_val = result[maxlag][0]['ssr_ftest'][1]
        return p_val
    except Exception:
        return 1.0


def granger_pvalue_at_lag(data_x, data_y, lag):
    """Compute Granger causality p-value for x -> y at exactly the given lag."""
    try:
        df_test = pd.DataFrame({'y': data_y, 'x': data_x}).dropna()
        if len(df_test) < lag + 20:
            return 1.0
        result = grangercausalitytests(df_test[['y', 'x']], maxlag=lag, verbose=False)
        p_val = result[lag][0]['ssr_ftest'][1]
        return p_val
    except Exception:
        return 1.0


# =============================================================================
# FIGURE A: All-Pairs Granger Heatmap
# =============================================================================

def generate_granger_heatmap(df, regimes, regime_labels, factors):
    """Generate 3-panel Granger causality heatmap, one per regime."""
    print("\n=== Figure A: All-Pairs Granger Heatmap ===")
    n_factors = len(factors)
    n_regimes = len(regime_labels)

    # Bonferroni correction: 5 * 4 pairs * 3 regimes = 60 tests (excluding self)
    n_tests = n_factors * (n_factors - 1) * n_regimes
    bonferroni_threshold = 0.05 / n_tests
    print(f"  Bonferroni threshold: p < {bonferroni_threshold:.6f} (n_tests={n_tests})")

    # Compute Granger p-values per regime
    pval_matrices = {}
    for regime_idx, regime_name in enumerate(regime_labels):
        mask = regimes == regime_idx
        regime_data = df.loc[mask, factors]
        n_obs = mask.sum()
        print(f"  Regime {regime_idx} ({regime_name}): {n_obs} observations")

        pval_matrix = np.ones((n_factors, n_factors))
        for i, cause in enumerate(factors):
            for j, effect in enumerate(factors):
                if i == j:
                    continue
                p = granger_pvalue(regime_data[cause].values, regime_data[effect].values, maxlag=5)
                pval_matrix[i, j] = p
        pval_matrices[regime_name] = pval_matrix

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), sharey=True)
    regime_colors_title = ['#2196F3', '#FF9800', '#F44336']

    for idx, (regime_name, pval_matrix) in enumerate(pval_matrices.items()):
        ax = axes[idx]
        # Convert to -log10(p)
        neglog = -np.log10(np.clip(pval_matrix, 1e-20, 1.0))
        # Mask diagonal
        np.fill_diagonal(neglog, np.nan)

        # Plot heatmap
        im = ax.imshow(neglog, cmap='YlOrRd', vmin=0, vmax=8, aspect='equal',
                        interpolation='nearest')

        # Add cell values and significance markers
        for i in range(n_factors):
            for j in range(n_factors):
                if i == j:
                    ax.text(j, i, '--', ha='center', va='center', fontsize=7, color='gray')
                    continue
                val = neglog[i, j]
                p = pval_matrix[i, j]
                color = 'white' if val > 4 else 'black'
                label = f'{val:.1f}'
                if p < bonferroni_threshold:
                    label += '*'
                    # Draw border around significant cells
                    rect = Rectangle((j - 0.5, i - 0.5), 1, 1,
                                     linewidth=1.5, edgecolor='black',
                                     facecolor='none', zorder=3)
                    ax.add_patch(rect)
                ax.text(j, i, label, ha='center', va='center', fontsize=6.5, color=color)

        ax.set_xticks(range(n_factors))
        ax.set_xticklabels(factors, fontsize=8, rotation=45, ha='right')
        if idx == 0:
            ax.set_yticks(range(n_factors))
            ax.set_yticklabels(factors, fontsize=8)
        else:
            ax.set_yticks(range(n_factors))

        ax.set_title(regime_name, fontsize=11, fontweight='bold',
                     color=regime_colors_title[idx])
        ax.tick_params(length=2)

    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(r'$-\log_{10}(p)$', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # Row label
    fig.text(0.02, 0.5, 'Cause', va='center', rotation='vertical', fontsize=10)
    fig.text(0.48, 0.01, 'Effect', ha='center', fontsize=10)

    fig.suptitle('Granger Causality by Regime', fontsize=12, fontweight='bold', y=1.02)

    plt.subplots_adjust(wspace=0.08, left=0.10, right=0.90, bottom=0.18, top=0.88)
    outpath = f'{FIGURES_DIR}/granger_heatmap.pdf'
    fig.savefig(outpath, bbox_inches='tight')
    print(f"  Saved: {outpath}")
    plt.close(fig)


# =============================================================================
# FIGURE B: Lag Sensitivity Profile
# =============================================================================

def generate_lag_sensitivity(df, regimes, regime_labels):
    """Generate lag sensitivity profile for HML->SMB in Crisis regime."""
    print("\n=== Figure B: Lag Sensitivity Profile (HML -> SMB, Crisis) ===")

    crisis_idx = 2  # Crisis is regime 2
    mask = regimes == crisis_idx
    crisis_data = df.loc[mask]
    n_obs = mask.sum()
    print(f"  Crisis regime: {n_obs} observations")

    lags = list(range(1, 16))
    pvalues = []
    for lag in lags:
        p = granger_pvalue_at_lag(crisis_data['HML'].values, crisis_data['SMB'].values, lag)
        pvalues.append(p)
        print(f"    Lag {lag:2d}: p = {p:.6f}")

    neglog_p = [-np.log10(max(p, 1e-20)) for p in pvalues]

    # Bonferroni threshold (same as above: 60 tests)
    bonf_threshold = 0.05 / 60
    neglog_bonf = -np.log10(bonf_threshold)
    neglog_005 = -np.log10(0.05)

    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    # Bar chart with gradient coloring
    colors = []
    for p in pvalues:
        if p < bonf_threshold:
            colors.append('#D32F2F')  # Dark red for Bonferroni-significant
        elif p < 0.05:
            colors.append('#FF9800')  # Orange for nominally significant
        else:
            colors.append('#90CAF9')  # Light blue for non-significant

    bars = ax.bar(lags, neglog_p, color=colors, edgecolor='gray', linewidth=0.5, width=0.7)

    # Threshold lines
    ax.axhline(neglog_bonf, color='#D32F2F', linestyle='--', linewidth=1.0, alpha=0.8,
               label=f'Bonferroni ($p<{bonf_threshold:.4f}$)')
    ax.axhline(neglog_005, color='#FF9800', linestyle=':', linewidth=1.0, alpha=0.8,
               label='$p < 0.05$')

    ax.set_xlabel('Lag (trading days)')
    ax.set_ylabel(r'$-\log_{10}(p)$')
    ax.set_title('Lag Sensitivity: HML $\\rightarrow$ SMB (Crisis)', fontsize=11, fontweight='bold')
    ax.set_xticks(lags)
    ax.legend(fontsize=7.5, loc='upper right', framealpha=0.9)
    ax.set_xlim(0.3, 15.7)
    ax.set_ylim(0, max(neglog_p) * 1.15)

    # Annotate peak
    peak_lag = lags[np.argmax(neglog_p)]
    peak_val = max(neglog_p)
    ax.annotate(f'Peak: lag={peak_lag}', xy=(peak_lag, peak_val),
                xytext=(peak_lag + 2, peak_val * 0.92),
                arrowprops=dict(arrowstyle='->', color='black', lw=0.8),
                fontsize=8, fontweight='bold')

    plt.tight_layout()
    outpath = f'{FIGURES_DIR}/lag_sensitivity.pdf'
    fig.savefig(outpath)
    print(f"  Saved: {outpath}")
    plt.close(fig)


# =============================================================================
# FIGURE C: Rolling Granger P-value Time Series
# =============================================================================

def generate_rolling_granger(df, regimes, regime_labels):
    """Generate rolling Granger p-value time series with regime-colored background."""
    print("\n=== Figure C: Rolling Granger P-value (HML -> SMB, lag=9) ===")

    window_size = 756  # ~3 years of trading days
    lag = 9
    dates = df.index
    n = len(df)

    rolling_pvals = []
    rolling_dates = []

    print(f"  Computing rolling Granger with window={window_size}, lag={lag}...")
    step = 5  # compute every 5 days for speed
    for end in range(window_size, n, step):
        start = end - window_size
        window_data = df.iloc[start:end]
        p = granger_pvalue_at_lag(window_data['HML'].values, window_data['SMB'].values, lag)
        rolling_pvals.append(p)
        rolling_dates.append(dates[end - 1])

    rolling_pvals = np.array(rolling_pvals)
    rolling_dates = pd.DatetimeIndex(rolling_dates)
    neglog_p = -np.log10(np.clip(rolling_pvals, 1e-20, 1.0))

    print(f"  Computed {len(rolling_pvals)} windows")
    print(f"  Range: {rolling_dates[0].date()} to {rolling_dates[-1].date()}")
    print(f"  Max -log10(p): {neglog_p.max():.2f}, Min: {neglog_p.min():.2f}")

    # Regime colors for background
    regime_colors = ['#E3F2FD', '#FFF3E0', '#FFEBEE']  # Light blue, light orange, light red
    regime_edge_colors = ['#2196F3', '#FF9800', '#F44336']

    fig, ax = plt.subplots(figsize=(7.2, 2.8))

    # Draw regime-colored background spans
    # For each contiguous block of the same regime, draw a vertical span
    prev_regime = regimes[0]
    span_start = dates[0]
    for i in range(1, len(dates)):
        if regimes[i] != prev_regime or i == len(dates) - 1:
            span_end = dates[i]
            ax.axvspan(span_start, span_end, alpha=0.35,
                       color=regime_colors[int(prev_regime)], zorder=0)
            span_start = dates[i]
            prev_regime = regimes[i]

    # Plot rolling -log10(p)
    ax.plot(rolling_dates, neglog_p, color='black', linewidth=0.7, alpha=0.9, zorder=2)

    # Threshold lines
    neglog_005 = -np.log10(0.05)
    neglog_001 = -np.log10(0.001)
    ax.axhline(neglog_005, color='#FF9800', linestyle='--', linewidth=0.8,
               label='$p = 0.05$', zorder=1)
    ax.axhline(neglog_001, color='#D32F2F', linestyle='--', linewidth=0.8,
               label='$p = 0.001$', zorder=1)

    # Legend patches for regimes
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=regime_colors[0], edgecolor=regime_edge_colors[0],
              label=regime_labels[0], alpha=0.5),
        Patch(facecolor=regime_colors[1], edgecolor=regime_edge_colors[1],
              label=regime_labels[1], alpha=0.5),
        Patch(facecolor=regime_colors[2], edgecolor=regime_edge_colors[2],
              label=regime_labels[2], alpha=0.5),
        plt.Line2D([0], [0], color='#FF9800', linestyle='--', label='$p=0.05$'),
        plt.Line2D([0], [0], color='#D32F2F', linestyle='--', label='$p=0.001$'),
    ]
    ax.legend(handles=legend_elements, fontsize=7.5, loc='upper left',
              ncol=2, framealpha=0.9)

    ax.set_xlabel('Date')
    ax.set_ylabel(r'$-\log_{10}(p)$')
    ax.set_title('Rolling Granger Causality: HML $\\rightarrow$ SMB (3-year window, lag=9)',
                 fontsize=11, fontweight='bold')
    ax.set_xlim(rolling_dates[0], rolling_dates[-1])
    ax.set_ylim(0, min(neglog_p.max() * 1.1, 20))

    plt.tight_layout()
    outpath = f'{FIGURES_DIR}/rolling_granger.pdf'
    fig.savefig(outpath)
    print(f"  Saved: {outpath}")
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("Generating Additional Figures for Causal Regimes Paper")
    print("=" * 70)

    # 1. Download data
    df = download_ff_data()

    # Filter to 1990-2024 to match paper sample period
    df = df.loc['1990-01-01':'2024-12-31']
    print(f"  Filtered to 1990-2024: {len(df)} observations")

    # Use 5 factors (drop RF)
    factors = ['MKT', 'SMB', 'HML', 'RMW', 'CMA']
    X = df[factors].values

    # 2. Fit Student-t HMM with 3 regimes
    print("\nFitting Student-t HMM (3 regimes)...")
    hmm = StudentTHMM(n_regimes=3, n_iter=100, tol=1e-4, random_state=28)
    hmm.fit(X)

    regimes = np.argmax(hmm.gamma, axis=1)
    regime_labels = ['Normal', 'Elevated', 'Crisis']

    for k in range(3):
        n_k = (regimes == k).sum()
        pct = 100 * n_k / len(regimes)
        print(f"  Regime {k} ({regime_labels[k]}): {n_k} days ({pct:.1f}%), nu={hmm.nu[k]:.1f}")

    # 3. Generate figures
    generate_granger_heatmap(df, regimes, regime_labels, factors)
    generate_lag_sensitivity(df, regimes, regime_labels)
    generate_rolling_granger(df, regimes, regime_labels)

    print("\n" + "=" * 70)
    print("All figures generated successfully.")
    print("=" * 70)


if __name__ == '__main__':
    main()
