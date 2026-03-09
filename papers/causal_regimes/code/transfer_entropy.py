#!/usr/bin/env python3
"""
Transfer Entropy (Schreiber 2000) between Fama-French factors per HMM regime.

Transfer entropy is a non-linear, information-theoretic generalization of
Granger causality that captures all (linear + non-linear) directed dependencies.

TE(X -> Y) at lag k = CMI(Y_t ; X_{t-k} | Y_{t-k})

Uses Frenzel-Pompe (2007) direct CMI estimator with scipy cKDTree,
vectorized for performance.

Author: Research code for causal_regimes paper
"""

import json
import io
import zipfile
import urllib.request
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.special import digamma, gammaln
from scipy.cluster.vq import kmeans2
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Student-t HMM (from critical_fixes_analysis.py)
# =============================================================================

class StudentTHMM:
    """Student-t HMM with both filtered and smoothed probability outputs."""

    def __init__(self, n_regimes=3, n_iter=100, tol=1e-4, random_state=42):
        self.n_regimes = n_regimes
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state
        self.mu = None; self.Sigma = None; self.nu = None
        self.A = None; self.pi = None; self.gamma = None
        self.alpha = None; self.log_likelihood_ = None

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
        if x.ndim == 1: x = x.reshape(1, -1)
        diff = x - mu
        Sigma_inv = np.linalg.inv(Sigma)
        mahal = np.sum(diff @ Sigma_inv * diff, axis=1)
        _, logdet = np.linalg.slogdet(Sigma)
        return (gammaln((nu + d) / 2) - gammaln(nu / 2)
                - 0.5 * d * np.log(nu * np.pi) - 0.5 * logdet
                - 0.5 * (nu + d) * np.log(1 + mahal / nu))

    def _compute_emission_probs(self, X):
        T, d = X.shape; K = self.n_regimes
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
                log_alpha[t, k] = np.logaddexp.reduce(log_alpha[t-1] + log_A[:, k]) + log_B[t, k]
        return log_alpha

    def _backward(self, log_B):
        T, K = log_B.shape
        log_beta = np.zeros((T, K)); log_beta[-1] = 0
        log_A = np.log(self.A + 1e-300)
        for t in range(T - 2, -1, -1):
            for k in range(K):
                log_beta[t, k] = np.logaddexp.reduce(log_A[k, :] + log_B[t+1, :] + log_beta[t+1, :])
        return log_beta

    def _e_step(self, X):
        T, d = X.shape; K = self.n_regimes
        log_B = self._compute_emission_probs(X)
        log_alpha = self._forward(log_B)
        log_beta = self._backward(log_B)
        log_likelihood = np.logaddexp.reduce(log_alpha[-1])
        log_gamma = log_alpha + log_beta
        log_gamma -= np.logaddexp.reduce(log_gamma, axis=1, keepdims=True)
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
            diff = X - self.mu[k]
            Sigma_inv = np.linalg.inv(self.Sigma[k])
            mahal = np.sum(diff @ Sigma_inv * diff, axis=1)
            self.u[:, k] = (self.nu[k] + d) / (self.nu[k] + mahal)
        return log_likelihood

    def _m_step(self, X):
        T, d = X.shape; K = self.n_regimes
        self.pi = self.gamma[0] / self.gamma[0].sum()
        for j in range(K):
            for k in range(K):
                self.A[j, k] = self.xi[:, j, k].sum() / self.gamma[:-1, j].sum()
        self.A /= self.A.sum(axis=1, keepdims=True)
        for k in range(K):
            weights = self.gamma[:, k] * self.u[:, k]
            self.mu[k] = (weights[:, None] * X).sum(axis=0) / weights.sum()
        for k in range(K):
            diff = X - self.mu[k]
            weights = self.gamma[:, k] * self.u[:, k]
            weighted_outer = np.zeros((d, d))
            for t in range(T):
                weighted_outer += weights[t] * np.outer(diff[t], diff[t])
            self.Sigma[k] = weighted_outer / self.gamma[:, k].sum() + 1e-6 * np.eye(d)
        for k in range(K):
            self._update_nu(X, k)
        self._enforce_ordering()

    def _update_nu(self, X, k):
        T, d = X.shape
        def neg_expected_ll(nu):
            if nu <= 2: return 1e10
            diff = X - self.mu[k]
            Sigma_inv = np.linalg.inv(self.Sigma[k])
            mahal = np.sum(diff @ Sigma_inv * diff, axis=1)
            ll = self.gamma[:, k] * (gammaln((nu + d) / 2) - gammaln(nu / 2)
                                     - 0.5 * d * np.log(nu)
                                     - 0.5 * (nu + d) * np.log(1 + mahal / nu))
            return -ll.sum()
        self.nu[k] = minimize_scalar(neg_expected_ll, bounds=(2.1, 50), method='bounded').x

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
                print(f"  Converged at iteration {iteration + 1}"); break
            prev_ll = log_likelihood
        self.log_likelihood_ = log_likelihood
        return self

    def predict(self, X, use_filtered=False):
        X = np.asarray(X); self._e_step(X)
        if use_filtered: return np.argmax(self.alpha, axis=1)
        return np.argmax(self.gamma, axis=1)


# =============================================================================
# FRENZEL-POMPE (2007) DIRECT CMI ESTIMATOR — VECTORIZED
# =============================================================================

def frenzel_pompe_cmi(x, y, z, k=5):
    """
    Direct conditional mutual information: CMI(X; Y | Z).
    
    Frenzel & Pompe (2007), Phys. Rev. Lett. 99, 204101.
    
    CMI(X;Y|Z) = digamma(k) + <digamma(n_z+1) - digamma(n_xz+1) - digamma(n_yz+1)>
    
    Vectorized: uses cKDTree.query_ball_point with batch mode for speed.
    Adds small jitter to break ties (important for discretized financial data).
    """
    N = len(x)
    if N < k + 5:
        return 0.0
    
    if x.ndim == 1: x = x.reshape(-1, 1)
    if y.ndim == 1: y = y.reshape(-1, 1)
    if z.ndim == 1: z = z.reshape(-1, 1)
    
    # Add tiny jitter to break ties
    rng = np.random.RandomState(0)
    jitter_scale = 1e-10
    x = x + rng.randn(*x.shape) * jitter_scale
    y = y + rng.randn(*y.shape) * jitter_scale
    z = z + rng.randn(*z.shape) * jitter_scale
    
    xyz = np.hstack([x, y, z])
    xz = np.hstack([x, z])
    yz = np.hstack([y, z])
    
    # Build trees with Chebyshev (max) norm
    tree_xyz = cKDTree(xyz)
    tree_xz = cKDTree(xz)
    tree_yz = cKDTree(yz)
    tree_z = cKDTree(z)
    
    # kth neighbor distances in joint space
    dists, _ = tree_xyz.query(xyz, k=k+1, p=np.inf)
    eps = dists[:, -1]
    
    # Batch count neighbors within eps in each subspace
    # query_ball_point with arrays returns list of lists
    eps_arr = eps + 1e-15
    
    # Count in each marginal subspace
    n_xz = np.array([
        tree_xz.query_ball_point(xz[i], eps_arr[i], p=np.inf, return_length=True) - 1
        for i in range(N)
    ], dtype=float)
    n_yz = np.array([
        tree_yz.query_ball_point(yz[i], eps_arr[i], p=np.inf, return_length=True) - 1
        for i in range(N)
    ], dtype=float)
    n_z = np.array([
        tree_z.query_ball_point(z[i], eps_arr[i], p=np.inf, return_length=True) - 1
        for i in range(N)
    ], dtype=float)
    
    # Clamp to >= 1
    n_xz = np.maximum(n_xz, 1)
    n_yz = np.maximum(n_yz, 1)
    n_z = np.maximum(n_z, 1)
    
    cmi = digamma(k) + np.mean(digamma(n_z + 1) - digamma(n_xz + 1) - digamma(n_yz + 1))
    return cmi


def transfer_entropy(source, target, lag, k=5):
    """
    TE(source -> target) at given lag = CMI(Y_t ; X_{t-lag} | Y_{t-lag})
    
    Embedding dimension = 1 (single lag value as conditioning).
    """
    n = len(source)
    if n <= lag + k + 5:
        return 0.0
    
    y_t = target[lag:]                     # Y_t
    x_past = source[lag-1:n-1].copy()      # X_{t-1} (at lag offset)
    y_past = target[lag-1:n-1].copy()      # Y_{t-1}
    
    # For lag > 1, shift further back
    if lag > 1:
        y_t = target[lag:]
        x_past = source[:n-lag].copy()     # X_{t-lag}
        y_past = target[:n-lag].copy()     # Y_{t-lag}
    
    return frenzel_pompe_cmi(y_t, x_past, y_past, k=k)


def transfer_entropy_multilag(source, target, max_lag=9, k=5):
    """
    TE(source -> target) with multi-lag embedding.
    
    Uses embedding: Y_past = [Y_{t-1}], X_past = [X_{t-1}, ..., X_{t-max_lag}]
    This tests whether the full history of X helps predict Y beyond Y's own past.
    """
    n = len(source)
    if n <= max_lag + k + 5:
        return 0.0
    
    y_t = target[max_lag:]
    y_past = target[max_lag-1:n-1].reshape(-1, 1)  # Y_{t-1}
    
    # X_past = [X_{t-1}, X_{t-2}, ..., X_{t-max_lag}]
    x_past_cols = []
    for l in range(1, max_lag + 1):
        x_past_cols.append(source[max_lag - l: n - l])
    x_past = np.column_stack(x_past_cols)
    
    return frenzel_pompe_cmi(y_t, x_past, y_past, k=k)


def permutation_test_te(source, target, lag, k=5, n_perms=200, rng_seed=28):
    """
    Permutation test for TE significance.
    Shuffle source to destroy temporal dependence.
    """
    rng = np.random.RandomState(rng_seed)
    te_observed = transfer_entropy(source, target, lag=lag, k=k)
    
    null_dist = np.zeros(n_perms)
    for i in range(n_perms):
        null_dist[i] = transfer_entropy(rng.permutation(source), target, lag=lag, k=k)
    
    p_value = np.mean(null_dist >= te_observed)
    return te_observed, p_value, null_dist


def permutation_test_te_multilag(source, target, max_lag, k=5, n_perms=200, rng_seed=28):
    """Permutation test for multi-lag TE."""
    rng = np.random.RandomState(rng_seed)
    te_observed = transfer_entropy_multilag(source, target, max_lag=max_lag, k=k)
    
    null_dist = np.zeros(n_perms)
    for i in range(n_perms):
        null_dist[i] = transfer_entropy_multilag(rng.permutation(source), target, max_lag=max_lag, k=k)
    
    p_value = np.mean(null_dist >= te_observed)
    return te_observed, p_value, null_dist


# =============================================================================
# DATA LOADING
# =============================================================================

def load_ff5_daily():
    """Download and parse Fama-French 5 factors daily."""
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
    print("Downloading Fama-French 5 factors daily...")
    response = urllib.request.urlopen(url)
    zipdata = zipfile.ZipFile(io.BytesIO(response.read()))
    csv_name = [f for f in zipdata.namelist() if f.endswith('.CSV') or f.endswith('.csv')][0]
    raw = zipdata.read(csv_name).decode('utf-8')
    lines = raw.split('\n')
    header_idx = None
    for i, line in enumerate(lines):
        if 'Mkt-RF' in line:
            header_idx = i; break
    if header_idx is None:
        raise ValueError("Could not find header row")
    data_lines = []
    for i in range(header_idx, len(lines)):
        line = lines[i].strip()
        if line == '': break
        data_lines.append(line)
    df = pd.read_csv(io.StringIO('\n'.join(data_lines)))
    df.columns = df.columns.str.strip()
    date_col = df.columns[0]
    df = df.rename(columns={date_col: 'date'})
    df['date'] = pd.to_datetime(df['date'].astype(str), format='%Y%m%d')
    for col in ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna().set_index('date').sort_index()
    print(f"  Loaded {len(df)} daily obs from {df.index[0].date()} to {df.index[-1].date()}")
    return df


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("TRANSFER ENTROPY ANALYSIS: Fama-French Factors per HMM Regime")
    print("Schreiber (2000) | Frenzel-Pompe (2007) direct CMI estimator")
    print("=" * 80)
    
    # 1. Load data
    ff = load_ff5_daily()
    factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
    X = ff[factors].values
    
    # 2. Fit Student-t HMM
    print("\nFitting Student-t HMM (3 regimes)...")
    hmm = StudentTHMM(n_regimes=3, n_iter=100, tol=1e-4, random_state=28)
    hmm.fit(X)
    regimes = hmm.predict(X)
    
    regime_names = {0: 'Normal', 1: 'Transition', 2: 'Crisis'}
    for r in range(3):
        n_r = (regimes == r).sum()
        vol = np.std(X[regimes == r, 0])
        print(f"  Regime {r} ({regime_names[r]}): {n_r} days, Mkt vol={vol:.3f}")
    
    # 3. TE: HML <-> SMB per regime, lag 1 through 9
    print("\n" + "=" * 80)
    print("TRANSFER ENTROPY: HML <-> SMB per Regime")
    print("Frenzel-Pompe CMI (k=5), per-lag embedding dim=1, lags 1-9")
    print("=" * 80)
    
    hml = ff['HML'].values
    smb = ff['SMB'].values
    K_NN = 5
    N_PERMS = 200
    LAGS = list(range(1, 10))
    
    results = {
        'method': 'Transfer Entropy (Schreiber 2000)',
        'estimator': 'Frenzel-Pompe (2007) direct CMI, kNN k=5',
        'n_permutations': N_PERMS,
        'lags': LAGS,
        'regimes': {}
    }
    
    for r in range(3):
        mask = regimes == r
        hml_r = hml[mask]
        smb_r = smb[mask]
        n_r = mask.sum()
        
        print(f"\n--- Regime {r} ({regime_names[r]}, n={n_r}) ---")
        
        # Per-lag TE in both directions
        te_fwd = {}  # HML -> SMB
        te_rev = {}  # SMB -> HML
        
        for lag in LAGS:
            te_fwd[lag] = transfer_entropy(hml_r, smb_r, lag=lag, k=K_NN)
            te_rev[lag] = transfer_entropy(smb_r, hml_r, lag=lag, k=K_NN)
        
        te_fwd_total = sum(te_fwd.values())
        te_rev_total = sum(te_rev.values())
        net_te = te_fwd_total - te_rev_total
        
        # Permutation tests at lag=1 (most informative for daily data)
        print(f"  Permutation test (n={N_PERMS}) at lag=1...")
        te_obs_1f, p_1f, null_1f = permutation_test_te(hml_r, smb_r, lag=1, k=K_NN, n_perms=N_PERMS, rng_seed=28+r)
        te_obs_1r, p_1r, null_1r = permutation_test_te(smb_r, hml_r, lag=1, k=K_NN, n_perms=N_PERMS, rng_seed=128+r)
        
        # Multi-lag TE with embedding (lags 1-9 of source, conditioning on lag-1 of target)
        print(f"  Multi-lag TE (max_lag=9)...")
        te_ml_fwd = transfer_entropy_multilag(hml_r, smb_r, max_lag=9, k=K_NN)
        te_ml_rev = transfer_entropy_multilag(smb_r, hml_r, max_lag=9, k=K_NN)
        
        # Permutation test for multi-lag
        print(f"  Permutation test for multi-lag TE...")
        te_ml_fwd_obs, p_ml_fwd, null_ml_fwd = permutation_test_te_multilag(
            hml_r, smb_r, max_lag=9, k=K_NN, n_perms=N_PERMS, rng_seed=228+r)
        te_ml_rev_obs, p_ml_rev, null_ml_rev = permutation_test_te_multilag(
            smb_r, hml_r, max_lag=9, k=K_NN, n_perms=N_PERMS, rng_seed=328+r)
        
        z_1f = (te_obs_1f - np.mean(null_1f)) / np.std(null_1f) if np.std(null_1f) > 0 else 0
        z_1r = (te_obs_1r - np.mean(null_1r)) / np.std(null_1r) if np.std(null_1r) > 0 else 0
        z_ml_f = (te_ml_fwd_obs - np.mean(null_ml_fwd)) / np.std(null_ml_fwd) if np.std(null_ml_fwd) > 0 else 0
        z_ml_r = (te_ml_rev_obs - np.mean(null_ml_rev)) / np.std(null_ml_rev) if np.std(null_ml_rev) > 0 else 0
        
        direction = "HML -> SMB" if net_te > 0 else "SMB -> HML"
        
        print(f"\n  Per-lag TE(HML->SMB): ", end="")
        for lag in LAGS:
            print(f"L{lag}={te_fwd[lag]:.4f}", end="  ")
        print(f"\n  Per-lag TE(SMB->HML): ", end="")
        for lag in LAGS:
            print(f"L{lag}={te_rev[lag]:.4f}", end="  ")
        
        print(f"\n\n  Sum TE(HML->SMB) = {te_fwd_total:.6f}")
        print(f"  Sum TE(SMB->HML) = {te_rev_total:.6f}")
        print(f"  Net TE = {net_te:.6f}  [{direction}]")
        print(f"\n  Lag=1: TE(HML->SMB)={te_obs_1f:.5f} p={p_1f:.4f} z={z_1f:.2f}")
        print(f"         TE(SMB->HML)={te_obs_1r:.5f} p={p_1r:.4f} z={z_1r:.2f}")
        print(f"  Multi-lag(9): TE(HML->SMB)={te_ml_fwd_obs:.5f} p={p_ml_fwd:.4f} z={z_ml_f:.2f}")
        print(f"                TE(SMB->HML)={te_ml_rev_obs:.5f} p={p_ml_rev:.4f} z={z_ml_r:.2f}")
        
        results['regimes'][regime_names[r]] = {
            'n_days': int(n_r),
            'TE_HML_to_SMB': {
                'sum_across_lags': float(te_fwd_total),
                'per_lag': {str(l): float(v) for l, v in te_fwd.items()},
                'lag1_observed': float(te_obs_1f),
                'lag1_pvalue': float(p_1f),
                'lag1_z_score': float(z_1f),
                'lag1_null_mean': float(np.mean(null_1f)),
                'lag1_null_std': float(np.std(null_1f)),
                'multilag9_observed': float(te_ml_fwd_obs),
                'multilag9_pvalue': float(p_ml_fwd),
                'multilag9_z_score': float(z_ml_f),
            },
            'TE_SMB_to_HML': {
                'sum_across_lags': float(te_rev_total),
                'per_lag': {str(l): float(v) for l, v in te_rev.items()},
                'lag1_observed': float(te_obs_1r),
                'lag1_pvalue': float(p_1r),
                'lag1_z_score': float(z_1r),
                'lag1_null_mean': float(np.mean(null_1r)),
                'lag1_null_std': float(np.std(null_1r)),
                'multilag9_observed': float(te_ml_rev_obs),
                'multilag9_pvalue': float(p_ml_rev),
                'multilag9_z_score': float(z_ml_r),
            },
            'net_TE_HML_to_SMB': float(net_te),
            'dominant_direction': direction,
        }
    
    # 4. All 30 directed pairs in Crisis regime (lag=1)
    print("\n" + "=" * 80)
    print("ALL 30 DIRECTED FACTOR PAIRS: Crisis Regime (lag=1)")
    print("=" * 80)
    
    crisis_mask = regimes == 2
    crisis_data = {f: ff[f].values[crisis_mask] for f in factors}
    n_crisis = crisis_mask.sum()
    print(f"Crisis regime: {n_crisis} days\n")
    
    pair_results = []
    for source_name in factors:
        for target_name in factors:
            if source_name == target_name:
                continue
            src = crisis_data[source_name]
            tgt = crisis_data[target_name]
            
            te_val = transfer_entropy(src, tgt, lag=1, k=K_NN)
            
            # Quick permutation (100 perms)
            rng_seed = abs(hash(source_name + target_name)) % (2**31)
            rng = np.random.RandomState(rng_seed)
            null_vals = np.zeros(100)
            for i in range(100):
                null_vals[i] = transfer_entropy(rng.permutation(src), tgt, lag=1, k=K_NN)
            p_val = np.mean(null_vals >= te_val)
            z_sc = (te_val - np.mean(null_vals)) / np.std(null_vals) if np.std(null_vals) > 0 else 0
            
            pair_results.append({
                'source': source_name, 'target': target_name,
                'TE': float(te_val), 'p_value': float(p_val), 'z_score': float(z_sc),
            })
            print(f"  {source_name:>6} -> {target_name:<6}: TE={te_val:+.5f}  p={p_val:.3f}  z={z_sc:+.2f}")
    
    pair_results.sort(key=lambda x: x['TE'], reverse=True)
    
    print(f"\n{'Rank':<5} {'Source':>7} -> {'Target':<7} {'TE':>10} {'p':>8} {'z':>8} {'Sig':>5}")
    print("-" * 55)
    for i, pr in enumerate(pair_results):
        sig = "***" if pr['p_value'] < 0.01 else ("**" if pr['p_value'] < 0.05 else ("*" if pr['p_value'] < 0.10 else ""))
        print(f"{i+1:<5} {pr['source']:>7} -> {pr['target']:<7} {pr['TE']:>+10.5f} {pr['p_value']:>8.3f} {pr['z_score']:>+8.2f} {sig:>5}")
    
    hml_smb_rank = next(i+1 for i, pr in enumerate(pair_results)
                        if pr['source'] == 'HML' and pr['target'] == 'SMB')
    print(f"\nHML -> SMB rank: {hml_smb_rank}/{len(pair_results)}")
    
    results['crisis_all_pairs'] = {
        'n_crisis_days': int(n_crisis),
        'ranked_pairs': pair_results,
        'HML_to_SMB_rank': hml_smb_rank,
    }
    
    # 5. Summary
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"\n{'Regime':<12} {'n':>6} {'TE(H->S)':>10} {'p':>7} {'TE(S->H)':>10} {'p':>7} {'Net TE':>10} {'Dir':>12}")
    print("-" * 80)
    for r_name in ['Normal', 'Transition', 'Crisis']:
        rd = results['regimes'][r_name]
        print(f"{r_name:<12} {rd['n_days']:>6} "
              f"{rd['TE_HML_to_SMB']['lag1_observed']:>+10.5f} {rd['TE_HML_to_SMB']['lag1_pvalue']:>7.4f} "
              f"{rd['TE_SMB_to_HML']['lag1_observed']:>+10.5f} {rd['TE_SMB_to_HML']['lag1_pvalue']:>7.4f} "
              f"{rd['net_TE_HML_to_SMB']:>+10.5f} {rd['dominant_direction']:>12}")
    
    print(f"\nMulti-lag (embedding dim=9):")
    print(f"{'Regime':<12} {'n':>6} {'TE(H->S)':>10} {'p':>7} {'z':>7} {'TE(S->H)':>10} {'p':>7} {'z':>7}")
    print("-" * 75)
    for r_name in ['Normal', 'Transition', 'Crisis']:
        rd = results['regimes'][r_name]
        print(f"{r_name:<12} {rd['n_days']:>6} "
              f"{rd['TE_HML_to_SMB']['multilag9_observed']:>+10.5f} "
              f"{rd['TE_HML_to_SMB']['multilag9_pvalue']:>7.4f} "
              f"{rd['TE_HML_to_SMB']['multilag9_z_score']:>+7.2f} "
              f"{rd['TE_SMB_to_HML']['multilag9_observed']:>+10.5f} "
              f"{rd['TE_SMB_to_HML']['multilag9_pvalue']:>7.4f} "
              f"{rd['TE_SMB_to_HML']['multilag9_z_score']:>+7.2f}")
    
    # 6. Save
    out_path = '/Users/i767700/Github/ai-in-finance/papers/causal_regimes/results/transfer_entropy_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
