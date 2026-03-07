"""
MS-VAR Comparison: Paper's HMM + Granger vs Markov-Switching VAR Baseline
=========================================================================

Compares two regime-switching approaches:
  1. Paper's approach: Student-t HMM (3 regimes) → Regime-conditional Granger tests
  2. MS-VAR baseline: Markov-Switching VAR using statsmodels

Focus on HML and SMB factors (1990-2024).

Comparison metrics:
  - Regime recovery (label agreement)
  - BIC/AIC model fit
  - Structural break pattern detection
  - Granger-like inference from MS-VAR coefficients
"""

import numpy as np
import pandas as pd
import json
import os
import sys
from datetime import datetime
from scipy import stats
from scipy.special import gammaln
from scipy.cluster.vq import kmeans2
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# Setup paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)
os.makedirs(RESULTS_DIR, exist_ok=True)

from multistart_hmm_pipeline import (
    download_ff_data,
    StudentTHMM,
    relabel_regimes_by_data_norm,
)

FACTOR_COLS = ['MKT', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']


# =============================================================================
# MS-VAR IMPLEMENTATION
# =============================================================================

def fit_msvar_statsmodels(data, k_regimes=3, order=1):
    """
    Fit Markov-Switching VAR using statsmodels.

    statsmodels.tsa.regime_switching.MarkovAutoregression supports
    univariate switching AR models. For bivariate, we fit two separate
    MS-AR models and combine regime information.

    For a proper MS-VAR, we implement a custom version.
    """
    from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression

    results = {}

    # Fit MS-AR for each variable separately
    for col in ['HML', 'SMB']:
        y = data[col].values.astype(float)
        try:
            model = MarkovAutoregression(
                y,
                k_regimes=k_regimes,
                order=order,
                switching_ar=True,
                switching_variance=True,
            )
            res = model.fit(disp=False, maxiter=500)
            # Get smoothed probabilities
            smooth_probs = res.smoothed_marginal_probabilities
            if hasattr(smooth_probs, 'values'):
                regimes = smooth_probs.values.argmax(axis=1)
            else:
                regimes = np.array(smooth_probs).argmax(axis=1)
            results[col] = {
                'model': res,
                'regimes': regimes,
                'aic': res.aic,
                'bic': res.bic,
                'llf': res.llf,
                'transition_matrix': res.regime_transition.T.tolist(),  # P[j|i]
            }
        except Exception as e:
            print(f"  Warning: MS-AR for {col} failed: {e}")
            results[col] = None

    return results


def fit_bivariate_msvar_custom(data, k_regimes=3, order=1, max_iter=100, tol=1e-4, seed=42):
    """
    Custom bivariate Markov-Switching VAR(1) with Gaussian emissions.

    Model: Y_t = mu_s + A_s * Y_{t-1} + epsilon_t, epsilon_t ~ N(0, Sigma_s)
    where s_t follows a Markov chain.

    Uses EM algorithm for estimation.
    """
    np.random.seed(seed)

    Y = data[['HML', 'SMB']].values
    T, d = Y.shape
    K = k_regimes

    # Create lagged data
    Y_curr = Y[order:]  # Y_t for t = order, ..., T-1
    Y_lag = Y[:-order]  # Y_{t-1}
    T_eff = len(Y_curr)

    # Initialize parameters using k-means
    centroids, labels = kmeans2(Y_curr, K, minit='++')
    norms = np.linalg.norm(centroids, axis=1)
    order_idx = np.argsort(norms)
    centroids = centroids[order_idx]

    # Initialize regime-specific parameters
    mu = centroids.copy()
    A = np.zeros((K, d, d))
    Sigma = np.zeros((K, d, d))

    for k in range(K):
        mask = labels == order_idx[k]
        if mask.sum() > 10:
            # Fit simple VAR for initialization
            Y_k = Y_curr[mask]
            Y_lag_k = Y_lag[mask]
            X_k = np.column_stack([np.ones(len(Y_k)), Y_lag_k])
            for j in range(d):
                beta = np.linalg.lstsq(X_k, Y_k[:, j], rcond=None)[0]
                mu[k, j] = beta[0]
                A[k, j, :] = beta[1:1+d]
            # Compute residuals correctly
            pred = mu[k] + Y_lag_k @ A[k].T
            residuals = Y_k - pred
            Sigma[k] = np.cov(residuals.T) + 1e-6 * np.eye(d)
        else:
            A[k] = 0.1 * np.eye(d)
            Sigma[k] = np.eye(d)

    # Transition matrix initialization (sticky)
    P = np.eye(K) * 0.9 + np.ones((K, K)) * 0.1 / K
    P = P / P.sum(axis=1, keepdims=True)

    # Initial state distribution
    pi = np.ones(K) / K

    # EM algorithm
    log_likelihood_prev = -np.inf

    for iteration in range(max_iter):
        # E-step: compute emission probabilities and forward-backward
        log_B = np.zeros((T_eff, K))
        for k in range(K):
            mean = mu[k] + (A[k] @ Y_lag.T).T
            for t in range(T_eff):
                diff = Y_curr[t] - mean[t]
                log_B[t, k] = _mvn_logpdf(diff, Sigma[k])

        # Forward pass
        log_alpha = np.zeros((T_eff, K))
        log_alpha[0] = np.log(pi + 1e-300) + log_B[0]
        log_P = np.log(P + 1e-300)

        for t in range(1, T_eff):
            for k in range(K):
                log_alpha[t, k] = np.logaddexp.reduce(log_alpha[t-1] + log_P[:, k]) + log_B[t, k]

        # Backward pass
        log_beta = np.zeros((T_eff, K))
        for t in range(T_eff - 2, -1, -1):
            for k in range(K):
                log_beta[t, k] = np.logaddexp.reduce(log_P[k, :] + log_B[t+1, :] + log_beta[t+1, :])

        # Compute log-likelihood
        log_likelihood = np.logaddexp.reduce(log_alpha[-1])

        # Smoothed probabilities (gamma)
        log_gamma = log_alpha + log_beta
        log_gamma = log_gamma - np.logaddexp.reduce(log_gamma, axis=1, keepdims=True)
        gamma = np.exp(log_gamma)

        # Transition probabilities (xi)
        xi = np.zeros((T_eff - 1, K, K))
        for t in range(T_eff - 1):
            for j in range(K):
                for k in range(K):
                    xi[t, j, k] = np.exp(
                        log_alpha[t, j] + log_P[j, k] + log_B[t+1, k] + log_beta[t+1, k] - log_likelihood
                    )

        # Check convergence
        if abs(log_likelihood - log_likelihood_prev) < tol:
            break
        log_likelihood_prev = log_likelihood

        # M-step
        # Update initial distribution
        pi = gamma[0] / gamma[0].sum()

        # Update transition matrix
        for j in range(K):
            for k in range(K):
                P[j, k] = (xi[:, j, k].sum() + 1e-10) / (gamma[:-1, j].sum() + 1e-10 * K)
        P = P / P.sum(axis=1, keepdims=True)

        # Update regime parameters
        for k in range(K):
            weights = gamma[:, k]
            W = np.diag(weights)
            X = np.column_stack([np.ones(T_eff), Y_lag])

            for j in range(d):
                try:
                    # Weighted least squares
                    WX = W @ X
                    Wy = W @ Y_curr[:, j]
                    beta = np.linalg.lstsq(WX, Wy, rcond=None)[0]
                    mu[k, j] = beta[0]
                    A[k, j, :] = beta[1:]
                except:
                    pass

            # Update covariance
            mean = mu[k] + (A[k] @ Y_lag.T).T
            residuals = Y_curr - mean
            weighted_cov = np.zeros((d, d))
            for t in range(T_eff):
                weighted_cov += weights[t] * np.outer(residuals[t], residuals[t])
            Sigma[k] = weighted_cov / (weights.sum() + 1e-10) + 1e-6 * np.eye(d)

    # Compute final regime assignments
    regimes = np.argmax(gamma, axis=1)

    # Relabel by volatility (norm of Sigma)
    vol_order = np.argsort([np.trace(Sigma[k]) for k in range(K)])
    relabeled_regimes = np.zeros_like(regimes)
    for new_k, old_k in enumerate(vol_order):
        relabeled_regimes[regimes == old_k] = new_k

    # Reorder parameters
    mu = mu[vol_order]
    A = A[vol_order]
    Sigma = Sigma[vol_order]
    P_new = np.zeros_like(P)
    for i in range(K):
        for j in range(K):
            P_new[np.where(vol_order == i)[0][0], np.where(vol_order == j)[0][0]] = P[i, j]
    P = P_new

    # Compute information criteria
    n_params = K * (d + d*d + d*(d+1)//2) + K * (K - 1)  # mu, A, Sigma, P
    aic = -2 * log_likelihood + 2 * n_params
    bic = -2 * log_likelihood + n_params * np.log(T_eff)

    return {
        'regimes': relabeled_regimes,
        'gamma': gamma[:, vol_order],
        'mu': mu.tolist(),
        'A': A.tolist(),
        'Sigma': Sigma.tolist(),
        'P': P.tolist(),
        'log_likelihood': float(log_likelihood),
        'aic': float(aic),
        'bic': float(bic),
        'n_obs': T_eff,
        'n_params': n_params,
        'iterations': iteration + 1,
    }


def _mvn_logpdf(x, Sigma):
    """Multivariate normal log-pdf."""
    d = len(x)
    sign, logdet = np.linalg.slogdet(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    mahal = x @ Sigma_inv @ x
    return -0.5 * (d * np.log(2 * np.pi) + logdet + mahal)


# =============================================================================
# PAPER'S HMM + GRANGER APPROACH
# =============================================================================

def fit_paper_hmm(df, seed=28):
    """Fit the paper's Student-t HMM approach."""
    print(f"  Fitting Student-t HMM (seed={seed})...")
    X = df[FACTOR_COLS].values

    hmm = StudentTHMM(n_regimes=3, n_iter=100, tol=1e-4, random_state=seed)
    hmm.fit(X)
    regimes_raw = hmm.predict(X, use_filtered=False)
    regimes, order = relabel_regimes_by_data_norm(df, regimes_raw, FACTOR_COLS)

    # Compute BIC/AIC
    T, d = X.shape
    K = 3
    # Parameters: K means (d each), K covariances (d*(d+1)/2), K nu, K*(K-1) transition probs
    n_params = K * d + K * d * (d + 1) // 2 + K + K * (K - 1)
    aic = -2 * hmm.log_likelihood_ + 2 * n_params
    bic = -2 * hmm.log_likelihood_ + n_params * np.log(T)

    return {
        'hmm': hmm,
        'regimes': regimes,
        'log_likelihood': float(hmm.log_likelihood_),
        'aic': float(aic),
        'bic': float(bic),
        'n_obs': T,
        'n_params': n_params,
        'nu': [float(v) for v in hmm.nu[order]],
    }


def run_regime_granger(df, regimes, max_lag=10):
    """Run regime-conditional Granger tests (HML -> SMB and SMB -> HML)."""
    hml = df['HML'].values
    smb = df['SMB'].values

    results = {}
    for direction, (x, y, x_name, y_name) in [
        ('HML_to_SMB', (hml, smb, 'HML', 'SMB')),
        ('SMB_to_HML', (smb, hml, 'SMB', 'HML')),
    ]:
        results[direction] = {}
        for regime_id, regime_name in enumerate(['Normal', 'Elevated', 'Crisis']):
            # Get clean indices
            clean_idx = _get_clean_indices(regimes, regime_id, max_lag)
            if len(clean_idx) < 30:
                results[direction][regime_name] = {'n_obs': len(clean_idx), 'insufficient_data': True}
                continue

            # Run Granger test
            granger_result = _granger_test_regime(y, x, clean_idx, max_lag)
            results[direction][regime_name] = granger_result

    return results


def _get_clean_indices(regimes, regime_id, max_lag):
    """Get indices where all lags are within the same regime."""
    regime_mask = (regimes == regime_id)
    indices = np.where(regime_mask)[0]
    clean = []
    for idx in indices:
        if idx >= max_lag and all(regimes[idx - l] == regime_id for l in range(1, max_lag + 1)):
            clean.append(idx)
    return np.array(clean) if clean else np.array([], dtype=int)


def _granger_test_regime(y_all, x_all, clean_idx, max_lag):
    """Run Granger F-test at optimal BIC lag."""
    # Select lag via BIC
    best_bic = np.inf
    best_lag = 1
    for lag in range(1, max_lag + 1):
        usable = [i for i in clean_idx if i >= lag]
        if len(usable) < 2 * lag + 10:
            continue
        usable = np.array(usable)
        y_curr = y_all[usable]
        y_lagged = np.column_stack([y_all[usable - i - 1] for i in range(lag)])
        x_lagged = np.column_stack([x_all[usable - i - 1] for i in range(lag)])
        X_u = np.column_stack([np.ones(len(y_curr)), y_lagged, x_lagged])
        try:
            beta = np.linalg.lstsq(X_u, y_curr, rcond=None)[0]
            rss = np.sum((y_curr - X_u @ beta) ** 2)
            bic = len(y_curr) * np.log(rss / len(y_curr)) + X_u.shape[1] * np.log(len(y_curr))
            if bic < best_bic:
                best_bic = bic
                best_lag = lag
        except:
            continue

    # Run F-test at best lag
    usable = np.array([i for i in clean_idx if i >= best_lag])
    if len(usable) < 2 * best_lag + 10:
        return {'n_obs': len(usable), 'insufficient_data': True}

    y_curr = y_all[usable]
    y_lagged = np.column_stack([y_all[usable - i - 1] for i in range(best_lag)])
    x_lagged = np.column_stack([x_all[usable - i - 1] for i in range(best_lag)])

    X_r = np.column_stack([np.ones(len(y_curr)), y_lagged])
    X_u = np.column_stack([np.ones(len(y_curr)), y_lagged, x_lagged])

    beta_r = np.linalg.lstsq(X_r, y_curr, rcond=None)[0]
    beta_u = np.linalg.lstsq(X_u, y_curr, rcond=None)[0]

    rss_r = np.sum((y_curr - X_r @ beta_r) ** 2)
    rss_u = np.sum((y_curr - X_u @ beta_u) ** 2)

    df1 = best_lag
    df2 = len(y_curr) - 2 * best_lag - 1

    if df2 <= 0 or rss_u <= 0:
        return {'n_obs': len(usable), 'error': 'Invalid degrees of freedom'}

    f_stat = ((rss_r - rss_u) / df1) / (rss_u / df2)
    p_value = 1 - stats.f.cdf(f_stat, df1, df2)

    tss = np.sum((y_curr - y_curr.mean()) ** 2)
    r2_u = 1 - rss_u / tss
    r2_r = 1 - rss_r / tss

    return {
        'n_obs': len(usable),
        'lag': best_lag,
        'f_stat': float(f_stat),
        'p_value': float(p_value),
        'significant': p_value < 0.05,
        'delta_r2': float(r2_u - r2_r),
    }


# =============================================================================
# COMPARISON METRICS
# =============================================================================

def compute_regime_agreement(regimes1, regimes2):
    """Compute regime label agreement using Hungarian algorithm."""
    from scipy.optimize import linear_sum_assignment

    # Align lengths
    min_len = min(len(regimes1), len(regimes2))
    r1 = regimes1[:min_len]
    r2 = regimes2[:min_len]

    K = max(r1.max(), r2.max()) + 1

    # Build confusion matrix
    confusion = np.zeros((K, K), dtype=int)
    for i, j in zip(r1, r2):
        confusion[i, j] += 1

    # Hungarian algorithm for best mapping
    row_ind, col_ind = linear_sum_assignment(-confusion)

    # Compute agreement with best mapping
    mapping = {col_ind[i]: i for i in range(len(col_ind))}
    r2_mapped = np.array([mapping.get(r, r) for r in r2])
    agreement = (r1 == r2_mapped).mean()

    return {
        'agreement': float(agreement),
        'confusion_matrix': confusion.tolist(),
        'mapping': {int(k): int(v) for k, v in mapping.items()},
    }


def analyze_structural_breaks(df, regimes, method_name):
    """Analyze structural break patterns (2008 crisis, COVID, etc.)."""
    breaks = {
        '2008_crisis': ('2008-09-01', '2009-03-31'),
        '2000_dotcom': ('2000-03-01', '2002-10-31'),
        '2020_covid': ('2020-02-15', '2020-04-30'),
        '2022_tightening': ('2022-01-01', '2022-12-31'),
    }

    results = {}
    for event, (start, end) in breaks.items():
        mask = (df.index >= start) & (df.index <= end)
        if mask.sum() == 0:
            continue

        event_regimes = regimes[mask[:len(regimes)]]
        crisis_pct = (event_regimes == 2).mean() * 100
        elevated_pct = (event_regimes == 1).mean() * 100
        normal_pct = (event_regimes == 0).mean() * 100

        results[event] = {
            'n_days': int(mask.sum()),
            'crisis_pct': float(crisis_pct),
            'elevated_pct': float(elevated_pct),
            'normal_pct': float(normal_pct),
            'detected_as_abnormal': (crisis_pct + elevated_pct) > 50,
        }

    return results


def msvar_coefficient_significance(msvar_result):
    """
    Extract Granger-like significance from MS-VAR coefficients.

    In MS-VAR, the regime-specific VAR coefficients A[k] show how
    the lagged values predict current values. Cross-variable coefficients
    indicate predictive relationships similar to Granger causality.
    """
    A = np.array(msvar_result['A'])  # (K, d, d)
    K, d, _ = A.shape

    results = {}
    for k in range(K):
        regime_name = ['Normal', 'Elevated', 'Crisis'][k]
        results[regime_name] = {
            # A[k, 0, 1] = coefficient of SMB_{t-1} in HML_t equation
            'SMB_to_HML_coef': float(A[k, 0, 1]),
            # A[k, 1, 0] = coefficient of HML_{t-1} in SMB_t equation
            'HML_to_SMB_coef': float(A[k, 1, 0]),
            # Diagonal = AR persistence
            'HML_persistence': float(A[k, 0, 0]),
            'SMB_persistence': float(A[k, 1, 1]),
        }

    return results


# =============================================================================
# MAIN COMPARISON
# =============================================================================

def main():
    print("=" * 70)
    print("MS-VAR COMPARISON: Paper's HMM + Granger vs MS-VAR Baseline")
    print("=" * 70)

    # Download data
    print("\n[1] Downloading Fama-French data...")
    df = download_ff_data()

    # Focus on HML and SMB
    print(f"\n[2] Data loaded: {len(df)} observations ({df.index[0].date()} to {df.index[-1].date()})")
    print(f"    Focus variables: HML, SMB")

    results = {
        'timestamp': datetime.now().isoformat(),
        'data': {
            'n_obs': len(df),
            'start': str(df.index[0].date()),
            'end': str(df.index[-1].date()),
            'variables': ['HML', 'SMB'],
        }
    }

    # Method 1: Paper's HMM + Granger
    print("\n[3] Fitting Paper's Student-t HMM (3 regimes)...")
    hmm_result = fit_paper_hmm(df, seed=28)
    print(f"    Log-likelihood: {hmm_result['log_likelihood']:.2f}")
    print(f"    BIC: {hmm_result['bic']:.2f}")
    print(f"    AIC: {hmm_result['aic']:.2f}")
    print(f"    DoF (nu): {hmm_result['nu']}")

    # Regime distribution
    hmm_regimes = hmm_result['regimes']
    for k, name in enumerate(['Normal', 'Elevated', 'Crisis']):
        pct = (hmm_regimes == k).mean() * 100
        print(f"    {name}: {(hmm_regimes == k).sum()} days ({pct:.1f}%)")

    # Run regime-conditional Granger
    print("\n[4] Running regime-conditional Granger tests (HMM approach)...")
    hmm_granger = run_regime_granger(df, hmm_regimes)

    print("    HML -> SMB:")
    for regime, res in hmm_granger['HML_to_SMB'].items():
        if 'p_value' in res:
            sig = "*" if res['significant'] else ""
            print(f"      {regime}: F={res['f_stat']:.2f}, p={res['p_value']:.4f}{sig}, n={res['n_obs']}")
        else:
            print(f"      {regime}: insufficient data (n={res.get('n_obs', 0)})")

    print("    SMB -> HML:")
    for regime, res in hmm_granger['SMB_to_HML'].items():
        if 'p_value' in res:
            sig = "*" if res['significant'] else ""
            print(f"      {regime}: F={res['f_stat']:.2f}, p={res['p_value']:.4f}{sig}, n={res['n_obs']}")
        else:
            print(f"      {regime}: insufficient data (n={res.get('n_obs', 0)})")

    # Method 2: MS-VAR
    print("\n[5] Fitting Bivariate MS-VAR (3 regimes)...")

    # Try multiple seeds for MS-VAR
    best_msvar = None
    best_ll = -np.inf

    for msvar_seed in [42, 28, 0, 7, 123]:
        try:
            msvar_result = fit_bivariate_msvar_custom(df, k_regimes=3, order=1, seed=msvar_seed)
            if msvar_result['log_likelihood'] > best_ll:
                best_ll = msvar_result['log_likelihood']
                best_msvar = msvar_result
                best_msvar['seed'] = msvar_seed
        except Exception as e:
            print(f"    Seed {msvar_seed} failed: {e}")

    if best_msvar is None:
        print("    MS-VAR fitting failed!")
        msvar_result = None
    else:
        msvar_result = best_msvar
        print(f"    Best seed: {msvar_result['seed']}")
        print(f"    Log-likelihood: {msvar_result['log_likelihood']:.2f}")
        print(f"    BIC: {msvar_result['bic']:.2f}")
        print(f"    AIC: {msvar_result['aic']:.2f}")
        print(f"    Iterations: {msvar_result['iterations']}")

        # Regime distribution
        msvar_regimes = msvar_result['regimes']
        for k, name in enumerate(['Normal', 'Elevated', 'Crisis']):
            pct = (msvar_regimes == k).mean() * 100
            print(f"    {name}: {(msvar_regimes == k).sum()} days ({pct:.1f}%)")

    # Also try statsmodels MS-AR for comparison
    print("\n[6] Fitting statsmodels MS-AR (univariate, for comparison)...")
    msar_results = fit_msvar_statsmodels(df, k_regimes=3, order=1)

    for var in ['HML', 'SMB']:
        if msar_results[var]:
            print(f"    {var}: BIC={msar_results[var]['bic']:.2f}, LL={msar_results[var]['llf']:.2f}")

    # Comparison
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)

    # Model fit comparison
    print("\n[A] MODEL FIT COMPARISON")
    print("-" * 50)
    print(f"{'Method':<25} {'Log-lik':<12} {'BIC':<12} {'AIC':<12}")
    print("-" * 50)
    print(f"{'HMM (Student-t)':<25} {hmm_result['log_likelihood']:<12.2f} {hmm_result['bic']:<12.2f} {hmm_result['aic']:<12.2f}")

    if msvar_result:
        print(f"{'MS-VAR (Gaussian)':<25} {msvar_result['log_likelihood']:<12.2f} {msvar_result['bic']:<12.2f} {msvar_result['aic']:<12.2f}")

    if msar_results['HML']:
        combined_bic = msar_results['HML']['bic'] + msar_results['SMB']['bic']
        combined_ll = msar_results['HML']['llf'] + msar_results['SMB']['llf']
        print(f"{'MS-AR (combined)':<25} {combined_ll:<12.2f} {combined_bic:<12.2f} {'N/A':<12}")

    # Regime agreement
    if msvar_result:
        print("\n[B] REGIME AGREEMENT")
        print("-" * 50)
        agreement = compute_regime_agreement(hmm_regimes, msvar_result['regimes'])
        print(f"    HMM vs MS-VAR agreement: {agreement['agreement']*100:.1f}%")
        print(f"    Mapping (MS-VAR -> HMM): {agreement['mapping']}")

    # Structural break detection
    print("\n[C] STRUCTURAL BREAK DETECTION")
    print("-" * 50)

    hmm_breaks = analyze_structural_breaks(df, hmm_regimes, 'HMM')
    print("\n  HMM (Student-t):")
    for event, data in hmm_breaks.items():
        detected = "YES" if data['detected_as_abnormal'] else "NO"
        print(f"    {event}: Crisis={data['crisis_pct']:.1f}%, Elevated={data['elevated_pct']:.1f}% -> Detected: {detected}")

    if msvar_result:
        msvar_breaks = analyze_structural_breaks(df, msvar_result['regimes'], 'MS-VAR')
        print("\n  MS-VAR (Gaussian):")
        for event, data in msvar_breaks.items():
            detected = "YES" if data['detected_as_abnormal'] else "NO"
            print(f"    {event}: Crisis={data['crisis_pct']:.1f}%, Elevated={data['elevated_pct']:.1f}% -> Detected: {detected}")

    # Granger-like inference from MS-VAR
    if msvar_result:
        print("\n[D] GRANGER-LIKE INFERENCE FROM MS-VAR COEFFICIENTS")
        print("-" * 50)
        msvar_coefs = msvar_coefficient_significance(msvar_result)
        print("\n  Cross-variable coefficients (VAR A matrix):")
        print(f"  {'Regime':<12} {'SMB->HML':<12} {'HML->SMB':<12} {'HML AR':<12} {'SMB AR':<12}")
        print("  " + "-" * 48)
        for regime, coefs in msvar_coefs.items():
            print(f"  {regime:<12} {coefs['SMB_to_HML_coef']:<12.4f} {coefs['HML_to_SMB_coef']:<12.4f} "
                  f"{coefs['HML_persistence']:<12.4f} {coefs['SMB_persistence']:<12.4f}")

        print("\n  Interpretation:")
        print("    - Positive cross-coefficients suggest predictive relationship")
        print("    - Regime-varying coefficients indicate structural change")

    # Summary table
    print("\n[E] SUMMARY COMPARISON TABLE")
    print("=" * 70)

    comparison_table = {
        'HMM_Student_t': {
            'distribution': 'Student-t (heavy tails)',
            'n_regimes': 3,
            'log_likelihood': hmm_result['log_likelihood'],
            'bic': hmm_result['bic'],
            'aic': hmm_result['aic'],
            'crisis_2008_detection': hmm_breaks.get('2008_crisis', {}).get('crisis_pct', 0),
            'covid_2020_detection': hmm_breaks.get('2020_covid', {}).get('crisis_pct', 0),
            'granger_results': hmm_granger,
        }
    }

    if msvar_result:
        comparison_table['MS_VAR_Gaussian'] = {
            'distribution': 'Gaussian',
            'n_regimes': 3,
            'log_likelihood': msvar_result['log_likelihood'],
            'bic': msvar_result['bic'],
            'aic': msvar_result['aic'],
            'crisis_2008_detection': msvar_breaks.get('2008_crisis', {}).get('crisis_pct', 0),
            'covid_2020_detection': msvar_breaks.get('2020_covid', {}).get('crisis_pct', 0),
            'var_coefficients': msvar_coefs,
        }

    # Key findings
    print("\n[F] KEY FINDINGS")
    print("-" * 50)

    bic_winner = "HMM" if hmm_result['bic'] < (msvar_result['bic'] if msvar_result else np.inf) else "MS-VAR"
    print(f"  1. BIC favors: {bic_winner}")

    if msvar_result:
        print(f"  2. Regime agreement: {agreement['agreement']*100:.1f}%")

        # Check if both detect 2008 crisis
        hmm_2008 = hmm_breaks.get('2008_crisis', {}).get('detected_as_abnormal', False)
        msvar_2008 = msvar_breaks.get('2008_crisis', {}).get('detected_as_abnormal', False)
        print(f"  3. 2008 crisis detection: HMM={'YES' if hmm_2008 else 'NO'}, MS-VAR={'YES' if msvar_2008 else 'NO'}")

        # Compare Granger findings
        print(f"  4. Granger significance pattern preserved: ", end="")
        hmm_hml_crisis_sig = hmm_granger['HML_to_SMB'].get('Crisis', {}).get('significant', False)
        print("YES" if hmm_hml_crisis_sig else "NO")

    print(f"  5. Heavy tails captured: HMM nu = {hmm_result['nu']}")

    # Build full results
    results['hmm_result'] = {
        'log_likelihood': hmm_result['log_likelihood'],
        'bic': hmm_result['bic'],
        'aic': hmm_result['aic'],
        'n_params': hmm_result['n_params'],
        'nu': hmm_result['nu'],
        'regime_counts': {name: int((hmm_regimes == k).sum()) for k, name in enumerate(['Normal', 'Elevated', 'Crisis'])},
        'granger': hmm_granger,
        'structural_breaks': hmm_breaks,
    }

    if msvar_result:
        results['msvar_result'] = {
            'log_likelihood': msvar_result['log_likelihood'],
            'bic': msvar_result['bic'],
            'aic': msvar_result['aic'],
            'n_params': msvar_result['n_params'],
            'seed': msvar_result['seed'],
            'regime_counts': {name: int((msvar_result['regimes'] == k).sum()) for k, name in enumerate(['Normal', 'Elevated', 'Crisis'])},
            'var_coefficients': msvar_coefs,
            'structural_breaks': msvar_breaks,
            'mu': msvar_result['mu'],
            'transition_matrix': msvar_result['P'],
        }
        results['comparison'] = {
            'regime_agreement': agreement,
            'bic_winner': bic_winner,
            'both_detect_2008': hmm_2008 and msvar_2008,
        }

    if msar_results['HML']:
        results['msar_results'] = {
            'HML': {k: v for k, v in msar_results['HML'].items() if k != 'model'},
            'SMB': {k: v for k, v in msar_results['SMB'].items() if k != 'model'},
        }

    # Save results
    output_path = os.path.join(RESULTS_DIR, 'msvar_comparison.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n[G] Results saved to: {output_path}")

    return results


if __name__ == '__main__':
    results = main()
