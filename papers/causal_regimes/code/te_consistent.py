#!/usr/bin/env python3
"""
Transfer Entropy — Consistent with paper (1990-2024, 8,817 trading days).

Fixes from prior te script:
  1. Filters FF data to 1990-01-02 through 2024-12-31 (NOT full dataset from 1963)
  2. Uses regime labels Normal/Elevated/Crisis (NOT Normal/Transition/Crisis)
  3. Computes 20 directed pairs (5 factors only, no MOM)
  4. Adds z-scores and full per-lag permutation tests

TE(X -> Y) at lag k = CMI(Y_t ; X_{t-k} | Y_{t-k})
Frenzel-Pompe (2007) direct CMI estimator, k=5 neighbors.
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
import time

warnings.filterwarnings('ignore')

RESULTS_DIR = '/Users/i767700/Github/ai-in-finance/papers/causal_regimes/results'


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
# FRENZEL-POMPE (2007) CMI ESTIMATOR
# =============================================================================

def frenzel_pompe_cmi(x, y, z, k=5):
    """
    Direct conditional mutual information: CMI(X; Y | Z).
    Frenzel & Pompe (2007), Phys. Rev. Lett. 99, 204101.
    """
    N = len(x)
    if N < k + 5:
        return 0.0

    if x.ndim == 1: x = x.reshape(-1, 1)
    if y.ndim == 1: y = y.reshape(-1, 1)
    if z.ndim == 1: z = z.reshape(-1, 1)

    rng = np.random.RandomState(0)
    jitter_scale = 1e-10
    x = x + rng.randn(*x.shape) * jitter_scale
    y = y + rng.randn(*y.shape) * jitter_scale
    z = z + rng.randn(*z.shape) * jitter_scale

    xyz = np.hstack([x, y, z])
    xz = np.hstack([x, z])
    yz = np.hstack([y, z])

    tree_xyz = cKDTree(xyz)
    tree_xz = cKDTree(xz)
    tree_yz = cKDTree(yz)
    tree_z = cKDTree(z)

    dists, _ = tree_xyz.query(xyz, k=k+1, p=np.inf)
    eps = dists[:, -1]
    eps_arr = eps + 1e-15

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

    n_xz = np.maximum(n_xz, 1)
    n_yz = np.maximum(n_yz, 1)
    n_z = np.maximum(n_z, 1)

    cmi = digamma(k) + np.mean(digamma(n_z + 1) - digamma(n_xz + 1) - digamma(n_yz + 1))
    return cmi


def transfer_entropy(source, target, lag, k=5):
    """TE(source -> target) at given lag = CMI(Y_t ; X_{t-lag} | Y_{t-lag})."""
    n = len(source)
    if n <= lag + k + 5:
        return 0.0

    y_t = target[lag:]
    x_past = source[:n-lag].copy()    # X_{t-lag}
    y_past = target[:n-lag].copy()    # Y_{t-lag}

    # For lag=1: y_t = target[1:], x_past = source[0:n-1], y_past = target[0:n-1]
    if lag == 1:
        y_t = target[1:]
        x_past = source[:-1].copy()
        y_past = target[:-1].copy()

    return frenzel_pompe_cmi(y_t, x_past, y_past, k=k)


def transfer_entropy_multilag(source, target, max_lag=9, k=5):
    """
    TE with multi-lag embedding.
    Y_past = [Y_{t-1}], X_past = [X_{t-1}, ..., X_{t-max_lag}]
    """
    n = len(source)
    if n <= max_lag + k + 5:
        return 0.0

    y_t = target[max_lag:]
    y_past = target[max_lag-1:n-1].reshape(-1, 1)

    x_past_cols = []
    for l in range(1, max_lag + 1):
        x_past_cols.append(source[max_lag - l: n - l])
    x_past = np.column_stack(x_past_cols)

    return frenzel_pompe_cmi(y_t, x_past, y_past, k=k)


def permutation_test_te(source, target, lag, k=5, n_perms=200, rng_seed=28):
    """Permutation test for TE significance."""
    rng = np.random.RandomState(rng_seed)
    te_observed = transfer_entropy(source, target, lag=lag, k=k)

    null_dist = np.zeros(n_perms)
    for i in range(n_perms):
        null_dist[i] = transfer_entropy(rng.permutation(source), target, lag=lag, k=k)

    p_value = np.mean(null_dist >= te_observed)
    z_score = (te_observed - np.mean(null_dist)) / np.std(null_dist) if np.std(null_dist) > 0 else 0.0
    return te_observed, p_value, z_score, null_dist


def permutation_test_te_multilag(source, target, max_lag, k=5, n_perms=200, rng_seed=28):
    """Permutation test for multi-lag TE."""
    rng = np.random.RandomState(rng_seed)
    te_observed = transfer_entropy_multilag(source, target, max_lag=max_lag, k=k)

    null_dist = np.zeros(n_perms)
    for i in range(n_perms):
        null_dist[i] = transfer_entropy_multilag(rng.permutation(source), target, max_lag=max_lag, k=k)

    p_value = np.mean(null_dist >= te_observed)
    z_score = (te_observed - np.mean(null_dist)) / np.std(null_dist) if np.std(null_dist) > 0 else 0.0
    return te_observed, p_value, z_score, null_dist


# =============================================================================
# DATA LOADING
# =============================================================================

def load_ff5_daily_1990_2024():
    """Download FF5 daily, filter to 1990-01-02 through 2024-12-31."""
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
    print("Downloading Fama-French 5 factors daily...")
    response = urllib.request.urlopen(url, timeout=60)
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

    # CRITICAL: Filter to 1990-2024 only
    df = df.loc['1990-01-02':'2024-12-31']
    print(f"  Loaded {len(df)} trading days ({df.index[0].date()} to {df.index[-1].date()})")
    print(f"  Expected ~8,817 trading days for 1990-2024")
    return df


# =============================================================================
# MAIN
# =============================================================================

def main():
    t0 = time.time()
    print("=" * 80)
    print("TRANSFER ENTROPY — CONSISTENT WITH PAPER (1990-2024)")
    print("Frenzel-Pompe (2007) CMI estimator, kNN k=5")
    print("=" * 80)

    # 1. Load data — 1990-2024 ONLY
    ff = load_ff5_daily_1990_2024()
    factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
    X = ff[factors].values
    print(f"\nTotal trading days: {len(X)}")

    # 2. Fit Student-t HMM (K=3)
    print("\nFitting Student-t HMM (K=3, random_state=28)...")
    hmm = StudentTHMM(n_regimes=3, n_iter=100, tol=1e-4, random_state=28)
    hmm.fit(X)
    regimes = hmm.predict(X)

    # Sort regimes by mean factor norm: Normal (low) / Elevated (mid) / Crisis (high)
    regime_names = {0: 'Normal', 1: 'Elevated', 2: 'Crisis'}
    print("\n--- Regime Verification ---")
    for r in range(3):
        n_r = (regimes == r).sum()
        mean_norm = np.mean(np.linalg.norm(X[regimes == r], axis=1))
        vol = np.std(X[regimes == r, 0])
        print(f"  Regime {r} ({regime_names[r]:>9}): {n_r:>5} days  "
              f"mean_factor_norm={mean_norm:.4f}  Mkt_vol={vol:.4f}")

    print(f"\n  Paper expects: Normal~3721, Elevated~3827, Crisis~1269")
    print(f"  Got:           Normal={int((regimes==0).sum())}, "
          f"Elevated={int((regimes==1).sum())}, "
          f"Crisis={int((regimes==2).sum())}")
    print(f"  Total: {len(regimes)} (paper: 8817)")

    # 3. TE: HML <-> SMB per regime, lags 1-9
    print("\n" + "=" * 80)
    print("TRANSFER ENTROPY: HML <-> SMB per Regime")
    print("Lags 1-9, 200 permutation shuffles each")
    print("=" * 80)

    hml = ff['HML'].values
    smb = ff['SMB'].values
    K_NN = 5
    N_PERMS = 200
    LAGS = list(range(1, 10))

    results = {
        'description': 'Transfer entropy (Schreiber 2000) via Frenzel-Pompe (2007) CMI',
        'data_range': '1990-01-02 to 2024-12-31',
        'n_total_days': int(len(X)),
        'estimator': 'Frenzel-Pompe kNN CMI, k=5',
        'n_permutations': N_PERMS,
        'lags': LAGS,
        'hmm': {
            'type': 'Student-t HMM',
            'K': 3,
            'random_state': 42,
            'log_likelihood': float(hmm.log_likelihood_),
        },
        'regimes': {},
    }

    for r in range(3):
        rname = regime_names[r]
        mask = regimes == r
        hml_r = hml[mask]
        smb_r = smb[mask]
        n_r = int(mask.sum())

        print(f"\n{'='*60}")
        print(f"Regime {r} ({rname}, n={n_r})")
        print(f"{'='*60}")

        te_fwd_lags = {}  # HML -> SMB
        te_rev_lags = {}  # SMB -> HML

        for lag in LAGS:
            t_lag = time.time()
            print(f"  Lag {lag}: ", end="", flush=True)

            # HML -> SMB with permutation test
            te_f, p_f, z_f, null_f = permutation_test_te(
                hml_r, smb_r, lag=lag, k=K_NN, n_perms=N_PERMS, rng_seed=28+r*100+lag)

            # SMB -> HML with permutation test
            te_r, p_r, z_r, null_r = permutation_test_te(
                smb_r, hml_r, lag=lag, k=K_NN, n_perms=N_PERMS, rng_seed=128+r*100+lag)

            sig_f = "***" if p_f < 0.01 else ("**" if p_f < 0.05 else ("*" if p_f < 0.1 else ""))
            sig_r = "***" if p_r < 0.01 else ("**" if p_r < 0.05 else ("*" if p_r < 0.1 else ""))

            print(f"HML->SMB={te_f:+.5f} (p={p_f:.3f}, z={z_f:+.2f}){sig_f}  "
                  f"SMB->HML={te_r:+.5f} (p={p_r:.3f}, z={z_r:+.2f}){sig_r}  "
                  f"[{time.time()-t_lag:.1f}s]")

            te_fwd_lags[lag] = {
                'te': float(te_f), 'p_value': float(p_f), 'z_score': float(z_f),
                'null_mean': float(np.mean(null_f)), 'null_std': float(np.std(null_f)),
            }
            te_rev_lags[lag] = {
                'te': float(te_r), 'p_value': float(p_r), 'z_score': float(z_r),
                'null_mean': float(np.mean(null_r)), 'null_std': float(np.std(null_r)),
            }

        # Multi-lag TE (lags 1-9 jointly)
        print(f"  Multi-lag (1-9): ", end="", flush=True)
        t_ml = time.time()
        te_ml_f, p_ml_f, z_ml_f, null_ml_f = permutation_test_te_multilag(
            hml_r, smb_r, max_lag=9, k=K_NN, n_perms=N_PERMS, rng_seed=228+r)
        te_ml_r, p_ml_r, z_ml_r, null_ml_r = permutation_test_te_multilag(
            smb_r, hml_r, max_lag=9, k=K_NN, n_perms=N_PERMS, rng_seed=328+r)
        print(f"HML->SMB={te_ml_f:+.5f} (p={p_ml_f:.3f}, z={z_ml_f:+.2f})  "
              f"SMB->HML={te_ml_r:+.5f} (p={p_ml_r:.3f}, z={z_ml_r:+.2f})  "
              f"[{time.time()-t_ml:.1f}s]")

        # Sum across lags
        sum_fwd = sum(te_fwd_lags[l]['te'] for l in LAGS)
        sum_rev = sum(te_rev_lags[l]['te'] for l in LAGS)
        net_te = sum_fwd - sum_rev
        direction = "HML -> SMB" if net_te > 0 else "SMB -> HML"

        print(f"\n  Sum TE(HML->SMB) = {sum_fwd:.6f}")
        print(f"  Sum TE(SMB->HML) = {sum_rev:.6f}")
        print(f"  Net TE = {net_te:+.6f}  [{direction}]")

        results['regimes'][rname] = {
            'n_days': n_r,
            'TE_HML_to_SMB': {
                'per_lag': {str(l): te_fwd_lags[l] for l in LAGS},
                'sum_across_lags': float(sum_fwd),
                'multilag_1_9': {
                    'te': float(te_ml_f), 'p_value': float(p_ml_f), 'z_score': float(z_ml_f),
                    'null_mean': float(np.mean(null_ml_f)), 'null_std': float(np.std(null_ml_f)),
                },
            },
            'TE_SMB_to_HML': {
                'per_lag': {str(l): te_rev_lags[l] for l in LAGS},
                'sum_across_lags': float(sum_rev),
                'multilag_1_9': {
                    'te': float(te_ml_r), 'p_value': float(p_ml_r), 'z_score': float(z_ml_r),
                    'null_mean': float(np.mean(null_ml_r)), 'null_std': float(np.std(null_ml_r)),
                },
            },
            'net_TE_HML_to_SMB': float(net_te),
            'dominant_direction': direction,
        }

    # 4. All 20 directed pairs in Crisis regime at lag=1
    print("\n" + "=" * 80)
    print("ALL 20 DIRECTED FACTOR PAIRS: Crisis Regime (lag=1)")
    print("=" * 80)

    crisis_mask = regimes == 2
    crisis_data = {f: ff[f].values[crisis_mask] for f in factors}
    n_crisis = int(crisis_mask.sum())
    print(f"Crisis regime: {n_crisis} days\n")

    pair_results = []
    for source_name in factors:
        for target_name in factors:
            if source_name == target_name:
                continue
            src = crisis_data[source_name]
            tgt = crisis_data[target_name]

            rng_seed = abs(hash(source_name + target_name)) % (2**31)
            te_val, p_val, z_sc, null_vals = permutation_test_te(
                src, tgt, lag=1, k=K_NN, n_perms=N_PERMS, rng_seed=rng_seed)

            pair_results.append({
                'source': source_name, 'target': target_name,
                'TE': float(te_val), 'p_value': float(p_val), 'z_score': float(z_sc),
            })
            sig = "***" if p_val < 0.01 else ("**" if p_val < 0.05 else ("*" if p_val < 0.1 else ""))
            print(f"  {source_name:>6} -> {target_name:<6}: TE={te_val:+.5f}  "
                  f"p={p_val:.3f}  z={z_sc:+.2f} {sig}")

    pair_results.sort(key=lambda x: x['TE'], reverse=True)

    print(f"\n{'Rank':<5} {'Source':>7} -> {'Target':<7} {'TE':>10} {'p':>8} {'z':>8} {'Sig':>5}")
    print("-" * 55)
    for i, pr in enumerate(pair_results):
        sig = "***" if pr['p_value'] < 0.01 else ("**" if pr['p_value'] < 0.05 else ("*" if pr['p_value'] < 0.1 else ""))
        print(f"{i+1:<5} {pr['source']:>7} -> {pr['target']:<7} "
              f"{pr['TE']:>+10.5f} {pr['p_value']:>8.3f} {pr['z_score']:>+8.2f} {sig:>5}")

    hml_smb_rank = next(i+1 for i, pr in enumerate(pair_results)
                        if pr['source'] == 'HML' and pr['target'] == 'SMB')
    print(f"\nHML -> SMB rank: {hml_smb_rank}/{len(pair_results)}")

    results['crisis_all_pairs'] = {
        'n_crisis_days': n_crisis,
        'ranked_pairs': pair_results,
        'HML_to_SMB_rank': hml_smb_rank,
        'n_pairs': len(pair_results),
    }

    # 5. Summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE (lag=1)")
    print("=" * 80)
    print(f"\n{'Regime':<12} {'n':>6} {'TE(H->S)':>10} {'p':>7} {'z':>7} "
          f"{'TE(S->H)':>10} {'p':>7} {'z':>7} {'Net TE':>10} {'Dir':>12}")
    print("-" * 95)
    for rname in ['Normal', 'Elevated', 'Crisis']:
        rd = results['regimes'][rname]
        fwd1 = rd['TE_HML_to_SMB']['per_lag']['1']
        rev1 = rd['TE_SMB_to_HML']['per_lag']['1']
        print(f"{rname:<12} {rd['n_days']:>6} "
              f"{fwd1['te']:>+10.5f} {fwd1['p_value']:>7.3f} {fwd1['z_score']:>+7.2f} "
              f"{rev1['te']:>+10.5f} {rev1['p_value']:>7.3f} {rev1['z_score']:>+7.2f} "
              f"{rd['net_TE_HML_to_SMB']:>+10.5f} {rd['dominant_direction']:>12}")

    print(f"\nMulti-lag (embedding lags 1-9):")
    print(f"{'Regime':<12} {'n':>6} {'TE(H->S)':>10} {'p':>7} {'z':>7} "
          f"{'TE(S->H)':>10} {'p':>7} {'z':>7}")
    print("-" * 75)
    for rname in ['Normal', 'Elevated', 'Crisis']:
        rd = results['regimes'][rname]
        ml_f = rd['TE_HML_to_SMB']['multilag_1_9']
        ml_r = rd['TE_SMB_to_HML']['multilag_1_9']
        print(f"{rname:<12} {rd['n_days']:>6} "
              f"{ml_f['te']:>+10.5f} {ml_f['p_value']:>7.3f} {ml_f['z_score']:>+7.2f} "
              f"{ml_r['te']:>+10.5f} {ml_r['p_value']:>7.3f} {ml_r['z_score']:>+7.2f}")

    # 6. Save
    out_path = f'{RESULTS_DIR}/te_consistent_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print(f"Total runtime: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
