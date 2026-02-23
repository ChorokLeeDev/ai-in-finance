"""
BIC Optima Analysis for HMM Local Optima Comparison
=====================================================

This script analyzes the multi-start HMM results to:
1. Cluster seeds by similar log-likelihood (identify distinct local optima)
2. Compute BIC for each optimum
3. Compute Granger causality p-values for each optimum
4. Output a clear comparison table

The goal is to determine if the "null" optimum (seeds like 6, 28, 35, 42 with
poor 2008 crisis detection but best log-likelihood) can be legitimately
excluded based on model fit criteria (BIC).
"""

import numpy as np
import pandas as pd
import json
import os
import sys
import urllib.request
import zipfile
import io
from datetime import datetime
from scipy import stats
from scipy.special import gammaln
from scipy.optimize import minimize_scalar
from scipy.cluster.vq import kmeans2
from scipy.stats import f as f_dist
import warnings
warnings.filterwarnings('ignore')

RESULTS_DIR = '/Users/i767700/Github/ai-in-finance/papers/causal_regimes/results'
REGIME_NAMES = ['Normal', 'Elevated', 'Crisis']


def download_ff_data():
    """Download Fama-French data (same as main pipeline)."""
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


class StudentTHMM:
    """Student-t HMM (same implementation as main pipeline)."""

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

    def count_params(self):
        """Count the number of free parameters in the HMM.

        For a K-state Student-t HMM with d dimensions:
        - Means: K * d parameters
        - Covariances: K * d * (d+1) / 2 (lower triangular)
        - Degrees of freedom: K
        - Transition matrix: K * (K-1) (rows sum to 1)
        - Initial distribution: K-1 (sums to 1)
        """
        K = self.n_regimes
        d = self.mu.shape[1]

        n_means = K * d  # 3 * 6 = 18
        n_covs = K * d * (d + 1) // 2  # 3 * 6 * 7 / 2 = 63
        n_nu = K  # 3
        n_trans = K * (K - 1)  # 3 * 2 = 6
        n_init = K - 1  # 2

        return n_means + n_covs + n_nu + n_trans + n_init


def compute_bic(log_likelihood, n_params, n_obs):
    """Compute Bayesian Information Criterion."""
    return -2 * log_likelihood + n_params * np.log(n_obs)


def relabel_regimes_by_data_norm(df, regimes_raw, factor_cols):
    """Relabel regime IDs so that ascending data-based mean norm = Normal/Elevated/Crisis."""
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


def extract_regime_clean_indices(regimes, regime_id, max_lag):
    """Get indices where ALL lags 1..max_lag fall within the same regime."""
    regime_mask = (regimes == regime_id)
    indices = np.where(regime_mask)[0]
    clean_indices = []
    for idx in indices:
        if idx >= max_lag:
            if all(regimes[idx - l] == regime_id for l in range(1, max_lag + 1)):
                clean_indices.append(idx)
    return np.array(clean_indices) if clean_indices else np.array([], dtype=int)


def granger_ftest(y_curr, y_lagged, x_lagged):
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
        return np.nan, np.nan
    f_stat = ((rss_r - rss_u) / df1) / (rss_u / df2)
    p_value = 1 - f_dist.cdf(f_stat, df1, df2)
    return float(f_stat), float(p_value)


def run_granger_test_for_crisis(df, regimes, lag=5):
    """Run HML->SMB Granger test in Crisis regime at specified lag."""
    hml_all = df['HML'].values
    smb_all = df['SMB'].values

    clean = extract_regime_clean_indices(regimes, 2, max_lag=lag)
    usable = np.array([idx for idx in clean if idx >= lag])

    if len(usable) < 2 * lag + 10:
        return None, None, len(usable)

    y_curr = smb_all[usable]
    y_lagged = np.column_stack([smb_all[usable - i - 1] for i in range(lag)])
    x_lagged = np.column_stack([hml_all[usable - i - 1] for i in range(lag)])

    f_stat, p_value = granger_ftest(y_curr, y_lagged, x_lagged)
    return f_stat, p_value, len(usable)


def cluster_by_log_likelihood(fit_summaries, threshold=10.0):
    """Cluster seeds by similar log-likelihood values.

    Two fits are in the same cluster if their log-likelihoods differ by < threshold.
    Uses a simple greedy clustering approach.
    """
    # Sort by log-likelihood (best first)
    sorted_fits = sorted(fit_summaries, key=lambda x: x['log_likelihood'], reverse=True)

    clusters = []
    for fit in sorted_fits:
        ll = fit['log_likelihood']
        seed = fit['seed']

        # Check if fit belongs to existing cluster
        assigned = False
        for cluster in clusters:
            # Compare to representative (first member)
            rep_ll = cluster['representative_ll']
            if abs(ll - rep_ll) < threshold:
                cluster['seeds'].append(seed)
                cluster['log_likelihoods'].append(ll)
                assigned = True
                break

        if not assigned:
            # Create new cluster
            clusters.append({
                'representative_ll': ll,
                'seeds': [seed],
                'log_likelihoods': [ll],
            })

    return clusters


def main():
    print("=" * 70)
    print("BIC OPTIMA ANALYSIS: Comparing HMM Local Optima")
    print("=" * 70)

    # Load existing results
    results_path = os.path.join(RESULTS_DIR, 'multistart_hmm_results.json')
    print(f"\nLoading results from: {results_path}")

    with open(results_path, 'r') as f:
        existing_results = json.load(f)

    fit_summaries = existing_results['fit_summaries']
    n_seeds = len(fit_summaries)
    print(f"Found {n_seeds} fit summaries")

    # Download data for Granger tests
    df = download_ff_data()
    X = df.values
    n_obs = len(df)
    factor_cols = ['MKT', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']

    # Cluster by log-likelihood
    print("\n" + "=" * 70)
    print("STEP 1: Clustering seeds by log-likelihood")
    print("=" * 70)

    clusters = cluster_by_log_likelihood(fit_summaries, threshold=15.0)

    print(f"\nFound {len(clusters)} distinct local optima clusters:\n")
    for i, cluster in enumerate(clusters):
        mean_ll = np.mean(cluster['log_likelihoods'])
        print(f"  Cluster {i+1}: LL ~ {mean_ll:.2f}, {len(cluster['seeds'])} seeds")
        print(f"    Seeds: {cluster['seeds'][:10]}{'...' if len(cluster['seeds']) > 10 else ''}")

    # For each cluster, compute BIC and Granger p-values using a representative seed
    print("\n" + "=" * 70)
    print("STEP 2: Computing BIC and Granger for each cluster")
    print("=" * 70)

    cluster_results = []

    for i, cluster in enumerate(clusters):
        # Use the first seed as representative (best LL in cluster)
        rep_seed = cluster['seeds'][0]
        rep_ll = cluster['log_likelihoods'][0]

        print(f"\n--- Cluster {i+1} (representative seed={rep_seed}) ---")

        # Find the fit summary for this seed
        fit_info = next(f for f in fit_summaries if f['seed'] == rep_seed)

        # Fit HMM with this seed
        print(f"  Fitting HMM with seed {rep_seed}...")
        hmm = StudentTHMM(n_regimes=3, n_iter=100, tol=1e-4, random_state=rep_seed)
        hmm.fit(X)

        # Get regime assignments
        regimes_raw = hmm.predict(X, use_filtered=False)
        regimes, order = relabel_regimes_by_data_norm(df, regimes_raw, factor_cols)

        # Count parameters and compute BIC
        n_params = hmm.count_params()
        bic = compute_bic(hmm.log_likelihood_, n_params, n_obs)

        print(f"  Log-likelihood: {hmm.log_likelihood_:.2f}")
        print(f"  Parameters: {n_params}")
        print(f"  BIC: {bic:.2f}")

        # Crisis detection for 2008
        mask_2008 = (df.index >= '2008-07-01') & (df.index <= '2009-06-30')
        idx_2008 = np.where(mask_2008)[0]
        crisis_2008 = float((regimes[idx_2008] == 2).mean() * 100)
        print(f"  2008 Crisis detection: {crisis_2008:.1f}%")

        # Regime counts
        counts = {REGIME_NAMES[k]: int((regimes == k).sum()) for k in range(3)}
        print(f"  Regime counts: {counts}")

        # Granger test at multiple lags
        granger_results = {}
        for lag in [1, 5, 10, 15]:
            f_stat, p_value, n_clean = run_granger_test_for_crisis(df, regimes, lag=lag)
            granger_results[f'lag_{lag}'] = {
                'f_stat': f_stat,
                'p_value': p_value,
                'n_clean': n_clean,
            }

        print(f"  Granger HML->SMB in Crisis:")
        for lag_key, result in granger_results.items():
            if result['p_value'] is not None:
                print(f"    {lag_key}: p={result['p_value']:.4f}, n={result['n_clean']}")
            else:
                print(f"    {lag_key}: insufficient data (n={result['n_clean']})")

        cluster_results.append({
            'cluster_id': i + 1,
            'representative_seed': rep_seed,
            'all_seeds': cluster['seeds'],
            'n_seeds': len(cluster['seeds']),
            'log_likelihood': float(hmm.log_likelihood_),
            'mean_log_likelihood': float(np.mean(cluster['log_likelihoods'])),
            'n_params': n_params,
            'n_obs': n_obs,
            'bic': float(bic),
            'crisis_2008_pct': crisis_2008,
            'regime_counts': counts,
            'nu': [float(v) for v in hmm.nu],
            'granger_crisis': granger_results,
        })

    # Sort clusters by BIC (lower is better)
    cluster_results.sort(key=lambda x: x['bic'])

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE: Local Optima Comparison (sorted by BIC)")
    print("=" * 70)

    print("\n{:<10} {:<8} {:<12} {:<10} {:<12} {:<10} {:<15}".format(
        "Cluster", "Seeds", "Log-Lik", "BIC", "2008 Crisis", "Crisis N", "Granger p (L5)"
    ))
    print("-" * 85)

    for result in cluster_results:
        granger_p = result['granger_crisis'].get('lag_5', {}).get('p_value')
        granger_str = f"{granger_p:.4f}" if granger_p is not None else "N/A"

        print("{:<10} {:<8} {:<12.2f} {:<10.2f} {:<12.1f}% {:<10} {:<15}".format(
            f"C{result['cluster_id']}",
            result['n_seeds'],
            result['log_likelihood'],
            result['bic'],
            result['crisis_2008_pct'],
            result['regime_counts']['Crisis'],
            granger_str
        ))

    # Print the key finding
    print("\n" + "=" * 70)
    print("KEY FINDING")
    print("=" * 70)

    best_bic = cluster_results[0]
    print(f"\nBest BIC: Cluster {best_bic['cluster_id']} (seeds: {best_bic['all_seeds'][:5]}...)")
    print(f"  BIC = {best_bic['bic']:.2f}")
    print(f"  Log-likelihood = {best_bic['log_likelihood']:.2f}")
    print(f"  2008 Crisis detection = {best_bic['crisis_2008_pct']:.1f}%")

    # Find the "corroborating" cluster (high 2008 detection)
    corroborating = [c for c in cluster_results if c['crisis_2008_pct'] >= 50]
    if corroborating:
        best_corr = corroborating[0]
        delta_bic = best_corr['bic'] - best_bic['bic']
        print(f"\nBest 'corroborating' cluster: Cluster {best_corr['cluster_id']}")
        print(f"  BIC = {best_corr['bic']:.2f}")
        print(f"  BIC difference from best: {delta_bic:.2f}")
        print(f"  2008 Crisis detection = {best_corr['crisis_2008_pct']:.1f}%")

        if delta_bic > 10:
            print(f"\n*** The null optimum CANNOT be excluded by BIC ***")
            print(f"    (BIC strongly favors null by {delta_bic:.0f} points)")
        elif delta_bic > 2:
            print(f"\n*** BIC weakly favors the null optimum ***")
            print(f"    (BIC difference of {delta_bic:.1f} is moderate)")
        else:
            print(f"\n*** BIC does not distinguish between optima ***")
            print(f"    (BIC difference of {delta_bic:.1f} is negligible)")

    # Save results
    output = {
        'metadata': {
            'timestamp': str(datetime.now()),
            'n_obs': n_obs,
            'n_seeds': n_seeds,
            'n_clusters': len(cluster_results),
            'clustering_threshold': 15.0,
        },
        'clusters': cluster_results,
        'summary': {
            'best_bic_cluster': best_bic['cluster_id'],
            'best_bic_value': best_bic['bic'],
            'best_bic_seeds': best_bic['all_seeds'],
            'best_bic_crisis_2008_pct': best_bic['crisis_2008_pct'],
        }
    }

    if corroborating:
        output['summary']['corroborating_cluster'] = best_corr['cluster_id']
        output['summary']['corroborating_bic'] = best_corr['bic']
        output['summary']['bic_difference'] = delta_bic

    output_path = os.path.join(RESULTS_DIR, 'bic_optima_comparison.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n\nResults saved to: {output_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()
