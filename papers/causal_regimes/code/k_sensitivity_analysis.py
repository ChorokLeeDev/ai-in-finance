"""
K Sensitivity Analysis for HMM Regime Count
============================================

Fits Student-t HMM with K=2 and K=4 regimes on 1990-2012 training data,
then runs frozen OOS Granger causality (lag=1) on 2013-2024 test data.

Goal: Show that HML->SMB Granger causality is not an artifact of choosing K=3.
"""

import sys
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import f as f_dist, chi2
import statsmodels.api as sm

warnings.filterwarnings('ignore')

# Import from existing pipeline
sys.path.insert(0, '/Users/i767700/Github/ai-in-finance/papers/causal_regimes/code')
from multistart_hmm_pipeline import (
    download_ff_data,
    extract_regime_clean_indices,
)

RESULTS_DIR = '/Users/i767700/Github/ai-in-finance/papers/causal_regimes/results'
FIXED_LAG = 1  # in-sample BIC-optimal


class StudentTHMM_K:
    """Student-t HMM with configurable number of regimes."""

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
        from scipy.cluster.vq import kmeans2
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
        # Initialize nu based on K
        if K == 2:
            self.nu = np.array([10.0, 5.0])
        elif K == 4:
            self.nu = np.array([20.0, 10.0, 6.0, 3.5])
        else:
            self.nu = np.linspace(15.0, 4.0, K)
        self.A = np.eye(K) * 0.95 + np.ones((K, K)) * 0.05 / K
        self.A = self.A / self.A.sum(axis=1, keepdims=True)
        self.pi = np.ones(K) / K

    def _mvt_logpdf(self, x, mu, Sigma, nu):
        from scipy.special import gammaln
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
        from scipy.optimize import minimize_scalar
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
                for k_idx in range(K):
                    self.xi[t, j, k_idx] = np.exp(
                        log_alpha[t, j] + log_A[j, k_idx] + log_B[t+1, k_idx] + log_beta[t+1, k_idx]
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
        from scipy.special import gammaln
        from scipy.optimize import minimize_scalar
        T, d = X.shape
        K = self.n_regimes
        self.pi = self.gamma[0] / self.gamma[0].sum()
        for j in range(K):
            for k_idx in range(K):
                self.A[j, k_idx] = self.xi[:, j, k_idx].sum() / self.gamma[:-1, j].sum()
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
        from scipy.special import gammaln
        from scipy.optimize import minimize_scalar
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

    def predict_oos(self, X, use_filtered=True):
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


def relabel_regimes_by_data_norm_k(df, regimes_raw, factor_cols, K):
    """Relabel regime IDs so that ascending data-based mean norm = 0,1,2,...,K-1."""
    data_norms = np.linalg.norm(df[factor_cols].values, axis=1)
    mean_norms = []
    for k in range(K):
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
        return np.nan, np.nan, np.nan
    f_stat = ((rss_r - rss_u) / df1) / (rss_u / df2)
    p_value = 1 - f_dist.cdf(f_stat, df1, df2)
    tss = np.sum((y_curr - y_curr.mean()) ** 2)
    r2_r = 1 - rss_r / tss
    r2_u = 1 - rss_u / tss
    delta_r2 = r2_u - r2_r
    return float(f_stat), float(p_value), float(delta_r2)


def granger_hac_wald(y_curr, y_lagged, x_lagged, lag):
    """HAC (Newey-West) robust Wald test for Granger causality."""
    n = len(y_curr)
    p = y_lagged.shape[1]
    X_u = np.column_stack([np.ones(n), y_lagged, x_lagged])
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
    try:
        wald_stat = float(Rb @ np.linalg.inv(RVR) @ Rb)
        p_value = float(1 - chi2.cdf(wald_stat, p))
    except np.linalg.LinAlgError:
        wald_stat = np.nan
        p_value = np.nan
    return wald_stat, p_value


def run_granger_at_lag_k(y_all, x_all, clean_indices, lag):
    """Run Granger F-test + HAC at a specific lag using clean indices."""
    usable = np.array([idx for idx in clean_indices if idx >= lag])
    if len(usable) < 2 * lag + 10:
        return None
    y_curr = y_all[usable]
    y_lagged = np.column_stack([y_all[usable - i - 1] for i in range(lag)])
    x_lagged = np.column_stack([x_all[usable - i - 1] for i in range(lag)])
    f_stat, f_p, delta_r2 = granger_ftest(y_curr, y_lagged, x_lagged)
    wald_stat, hac_p = granger_hac_wald(y_curr, y_lagged, x_lagged, lag)
    return {
        'n_obs': len(usable),
        'lag': lag,
        'f_stat': f_stat,
        'f_p_value': f_p,
        'hac_wald_stat': wald_stat,
        'hac_p_value': hac_p,
        'delta_r2': delta_r2,
    }


def get_regime_name(k, K):
    """Get regime name based on K."""
    if K == 2:
        names = ['Low-Vol', 'High-Vol']
    elif K == 3:
        names = ['Normal', 'Elevated', 'Crisis']
    elif K == 4:
        names = ['Calm', 'Low-Elevated', 'High-Elevated', 'Crisis']
    else:
        names = [f'Regime_{i}' for i in range(K)]
    return names[k] if k < len(names) else f'Regime_{k}'


def run_k_sensitivity(K, n_seeds=10):
    """Run sensitivity analysis for a specific K value."""
    print(f"\n{'='*70}")
    print(f"Running K={K} Sensitivity Analysis ({n_seeds} seeds)")
    print(f"{'='*70}")

    # Load data
    df = download_ff_data()
    df = df / 100.0  # Convert to decimal
    factor_cols = ['MKT', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']

    train_df = df.loc[:'2012-12-31']
    test_df = df.loc['2013-01-01':]
    print(f"  Train: {len(train_df)} days ({train_df.index[0].date()} to {train_df.index[-1].date()})")
    print(f"  Test: {len(test_df)} days ({test_df.index[0].date()} to {test_df.index[-1].date()})")

    # Fit with multiple seeds, select best LL
    print(f"\n  Fitting K={K} HMM with {n_seeds} seeds...")
    fits = []
    for seed in range(n_seeds):
        print(f"    Seed {seed}...", end=" ", flush=True)
        hmm = StudentTHMM_K(n_regimes=K, n_iter=100, tol=1e-4, random_state=seed)
        hmm.fit(train_df[factor_cols].values)
        print(f"LL={hmm.log_likelihood_:.2f}")
        fits.append({
            'seed': seed,
            'log_likelihood': float(hmm.log_likelihood_),
            'hmm': hmm,
        })

    # Select best by LL
    best_fit = max(fits, key=lambda x: x['log_likelihood'])
    best_seed = best_fit['seed']
    best_hmm = best_fit['hmm']
    print(f"\n  Best seed: {best_seed} (LL={best_fit['log_likelihood']:.2f})")

    # Get train regime assignments for relabeling order
    train_raw = best_hmm.predict(train_df[factor_cols].values, use_filtered=False)
    train_regimes, remap = relabel_regimes_by_data_norm_k(train_df, train_raw, factor_cols, K)
    train_counts = {get_regime_name(k, K): int((train_regimes == k).sum()) for k in range(K)}
    print(f"  Train regime counts: {train_counts}")

    # Apply to test data (frozen OOS)
    test_raw, _ = best_hmm.predict_oos(test_df[factor_cols].values, use_filtered=True)
    test_regimes = np.array([remap[r] for r in test_raw])
    test_counts = {get_regime_name(k, K): int((test_regimes == k).sum()) for k in range(K)}
    print(f"  Test regime counts: {test_counts}")

    # Run Granger causality in each regime
    hml = test_df['HML'].values
    smb = test_df['SMB'].values

    granger_results = {}
    print(f"\n  Granger Causality (HML->SMB, lag={FIXED_LAG}):")
    for k in range(K):
        regime_name = get_regime_name(k, K)
        clean_indices = extract_regime_clean_indices(test_regimes, k, max_lag=FIXED_LAG)
        h2s = run_granger_at_lag_k(smb, hml, clean_indices, FIXED_LAG)
        s2h = run_granger_at_lag_k(hml, smb, clean_indices, FIXED_LAG)

        granger_results[regime_name] = {
            'n_clean': len(clean_indices),
            'hml_to_smb': h2s,
            'smb_to_hml': s2h,
        }

        if h2s:
            sig_mark = '*' if h2s['f_p_value'] < 0.05 else ''
            hac_mark = '*' if h2s['hac_p_value'] < 0.05 else ''
            print(f"    {regime_name:15s}: n={h2s['n_obs']:4d}, "
                  f"F={h2s['f_stat']:6.2f}, F-p={h2s['f_p_value']:.4f}{sig_mark}, "
                  f"HAC-p={h2s['hac_p_value']:.4f}{hac_mark}")
        else:
            print(f"    {regime_name:15s}: insufficient data")

    return {
        'K': K,
        'n_seeds': n_seeds,
        'best_seed': best_seed,
        'best_log_likelihood': best_fit['log_likelihood'],
        'train_counts': train_counts,
        'test_counts': test_counts,
        'granger': granger_results,
        'all_fits': [{k: v for k, v in f.items() if k != 'hmm'} for f in fits],
    }


def main():
    print("K Sensitivity Analysis for HMM Regime Count")
    print("=" * 70)
    print(f"Started at {datetime.now()}")

    results = {
        'metadata': {
            'description': 'K sensitivity analysis: fits K=2 and K=4 HMMs on 1990-2012, '
                          'tests Granger causality on 2013-2024 frozen OOS.',
            'fixed_lag': FIXED_LAG,
            'train_period': '1990-2012',
            'test_period': '2013-2024',
            'timestamp': str(datetime.now()),
        },
        'K2': None,
        'K4': None,
    }

    # Run K=2
    results['K2'] = run_k_sensitivity(K=2, n_seeds=10)

    # Run K=4
    results['K4'] = run_k_sensitivity(K=4, n_seeds=10)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for k_key in ['K2', 'K4']:
        r = results[k_key]
        K = r['K']
        print(f"\nK={K}:")
        sig_regimes = []
        for regime_name, g in r['granger'].items():
            h2s = g.get('hml_to_smb')
            if h2s and h2s['f_p_value'] < 0.05:
                sig_regimes.append(f"{regime_name} (p={h2s['f_p_value']:.4f})")
        if sig_regimes:
            print(f"  Significant HML->SMB: {', '.join(sig_regimes)}")
        else:
            print(f"  No significant HML->SMB at p<0.05")

    # Save results
    outpath = f"{RESULTS_DIR}/k_sensitivity.json"
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {outpath}")
    print(f"Completed at {datetime.now()}")


if __name__ == '__main__':
    main()
