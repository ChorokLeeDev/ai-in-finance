#!/usr/bin/env python3
"""
breakpoint_bootstrap_ci.py
==========================
Bootstrap confidence interval for June 1998 structural breakpoint.

Methodology:
1. Download FF 5 factors + Momentum
2. Fit Student-t HMM (3 regimes) on 1990-2024 data
3. Extract Normal regime (regime 0)
4. Compute rolling Granger F-statistics for HML->SMB
5. Run Bai-Perron structural break test
6. Bootstrap the breakpoint date (1000 samples)
7. Report 90% CI

Output: results/breakpoint_ci.json
"""

import sys
import json
import io
import warnings
import urllib.request
import zipfile
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.optimize import minimize_scalar
from scipy.cluster.vq import kmeans2
from scipy.stats import f as f_dist
import statsmodels.api as sm

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

warnings.filterwarnings('ignore')

BASE = Path('/Users/i767700/Github/ai-in-finance/papers/causal_regimes')
RESULTS_DIR = BASE / 'results'
PRIMARY_SEED = 28
TRIM = 0.15
N_BOOTSTRAP = 1000
CI_LEVEL = 0.90


def download_ff_data():
    """Download FF5 + Momentum data from Ken French's website."""
    print("Downloading FF5 + MOM data...")

    # FF5 factors
    url5 = ('https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/'
            'F-F_Research_Data_5_Factors_2x3_daily_CSV.zip')
    with urllib.request.urlopen(url5, timeout=60) as r:
        data = r.read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        with z.open(z.namelist()[0]) as f:
            df5 = pd.read_csv(f, skiprows=3)
    df5.columns = df5.columns.str.strip()
    df5 = df5.rename(columns={df5.columns[0]: 'Date'})
    df5 = df5[df5['Date'].astype(str).str.match(r'^\d{8}$')]
    df5['Date'] = pd.to_datetime(df5['Date'], format='%Y%m%d')
    for c in ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']:
        df5[c] = pd.to_numeric(df5[c], errors='coerce')
    df5 = df5.set_index('Date').dropna()

    # Momentum
    url_mom = ('https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/'
               'F-F_Momentum_Factor_daily_CSV.zip')
    with urllib.request.urlopen(url_mom, timeout=60) as r:
        data = r.read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        with z.open(z.namelist()[0]) as f:
            mom = pd.read_csv(f, skiprows=13)
    mom.columns = mom.columns.str.strip()
    mom = mom.rename(columns={mom.columns[0]: 'Date'})
    mom = mom[mom['Date'].astype(str).str.match(r'^\d{8}$')]
    mom['Date'] = pd.to_datetime(mom['Date'], format='%Y%m%d')
    mom = mom.rename(columns={mom.columns[1]: 'MOM'})
    mom['MOM'] = pd.to_numeric(mom['MOM'], errors='coerce')
    mom = mom.set_index('Date').dropna()

    df = df5.join(mom[['MOM']], how='inner').rename(columns={'Mkt-RF': 'MKT'})
    df = df.drop('RF', axis=1, errors='ignore')
    df = df.loc['1990-01-01':'2024-12-31']
    print(f"Loaded {len(df)} days from {df.index[0].date()} to {df.index[-1].date()}")
    return df


class StudentTHMM:
    """Student-t HMM with 3 regimes."""

    def __init__(self, n_regimes=3, n_iter=100, tol=1e-4, random_state=42):
        self.n_regimes = n_regimes
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state
        self.mu = self.Sigma = self.nu = self.A = self.pi = None
        self.gamma = self.alpha = self.xi = None
        self.log_likelihood_ = None

    def _init_params(self, X):
        np.random.seed(self.random_state)
        T, d = X.shape
        K = self.n_regimes
        centroids, labels = kmeans2(X, K, minit='++')
        order = np.argsort(np.linalg.norm(centroids, axis=1))
        centroids = centroids[order]
        nl = np.zeros_like(labels)
        for nk, ok in enumerate(order):
            nl[labels == ok] = nk
        labels = nl
        self.mu = centroids
        self.Sigma = np.zeros((K, d, d))
        for k in range(K):
            m = labels == k
            self.Sigma[k] = (np.cov(X[m].T) + 1e-6 * np.eye(d)) if m.sum() > d else np.eye(d)
        self.nu = np.array([15., 7., 4.])
        self.A = np.eye(K) * .95 + np.ones((K, K)) * .05 / K
        self.A /= self.A.sum(axis=1, keepdims=True)
        self.pi = np.ones(K) / K

    def _mvt_logpdf(self, x, mu, Sigma, nu):
        d = len(mu)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        diff = x - mu
        Si = np.linalg.inv(Sigma)
        mahal = np.sum(diff @ Si * diff, axis=1)
        _, ld = np.linalg.slogdet(Sigma)
        return (gammaln((nu + d) / 2) - gammaln(nu / 2) - 0.5 * d * np.log(nu * np.pi)
                - 0.5 * ld - 0.5 * (nu + d) * np.log(1 + mahal / nu))

    def _log_B(self, X):
        T, d = X.shape
        K = self.n_regimes
        lb = np.zeros((T, K))
        for k in range(K):
            lb[:, k] = self._mvt_logpdf(X, self.mu[k], self.Sigma[k], self.nu[k])
        return lb

    def _forward(self, lb):
        T, K = lb.shape
        la = np.zeros((T, K))
        la[0] = np.log(self.pi + 1e-300) + lb[0]
        lA = np.log(self.A + 1e-300)
        for t in range(1, T):
            for k in range(K):
                la[t, k] = np.logaddexp.reduce(la[t - 1] + lA[:, k]) + lb[t, k]
        return la

    def _backward(self, lb):
        T, K = lb.shape
        lb2 = np.zeros((T, K))
        lA = np.log(self.A + 1e-300)
        for t in range(T - 2, -1, -1):
            for k in range(K):
                lb2[t, k] = np.logaddexp.reduce(lA[k, :] + lb[t + 1, :] + lb2[t + 1, :])
        return lb2

    def _e_step(self, X):
        T, d = X.shape
        K = self.n_regimes
        lB = self._log_B(X)
        la = self._forward(lB)
        lb = self._backward(lB)
        ll = np.logaddexp.reduce(la[-1])
        lg = la + lb
        lg -= np.logaddexp.reduce(lg, axis=1, keepdims=True)
        self.gamma = np.exp(lg)
        lan = la - np.logaddexp.reduce(la, axis=1, keepdims=True)
        self.alpha = np.exp(lan)
        lA = np.log(self.A + 1e-300)
        self.xi = np.zeros((T - 1, K, K))
        for t in range(T - 1):
            for j in range(K):
                for k in range(K):
                    self.xi[t, j, k] = np.exp(la[t, j] + lA[j, k] + lB[t + 1, k] + lb[t + 1, k] - ll)
        self.u = np.zeros((T, K))
        for k in range(K):
            diff = X - self.mu[k]
            Si = np.linalg.inv(self.Sigma[k])
            mahal = np.sum(diff @ Si * diff, axis=1)
            self.u[:, k] = (self.nu[k] + d) / (self.nu[k] + mahal)
        return ll

    def _m_step(self, X):
        T, d = X.shape
        K = self.n_regimes
        self.pi = self.gamma[0] / self.gamma[0].sum()
        for j in range(K):
            for k in range(K):
                self.A[j, k] = self.xi[:, j, k].sum() / self.gamma[:-1, j].sum()
        self.A /= self.A.sum(axis=1, keepdims=True)
        for k in range(K):
            w = self.gamma[:, k] * self.u[:, k]
            self.mu[k] = (w[:, None] * X).sum(0) / w.sum()
        for k in range(K):
            diff = X - self.mu[k]
            w = self.gamma[:, k] * self.u[:, k]
            wo = sum(w[t] * np.outer(diff[t], diff[t]) for t in range(T))
            self.Sigma[k] = wo / self.gamma[:, k].sum() + 1e-6 * np.eye(d)
        for k in range(K):
            self._update_nu(X, k)
        self._enforce_ordering()

    def _update_nu(self, X, k):
        T, d = X.shape

        def nl(nu):
            if nu <= 2:
                return 1e10
            diff = X - self.mu[k]
            Si = np.linalg.inv(self.Sigma[k])
            mahal = np.sum(diff @ Si * diff, axis=1)
            return -(self.gamma[:, k] * (gammaln((nu + d) / 2) - gammaln(nu / 2)
                     - 0.5 * d * np.log(nu) - 0.5 * (nu + d) * np.log(1 + mahal / nu))).sum()
        self.nu[k] = minimize_scalar(nl, bounds=(2.1, 50), method='bounded').x

    def _enforce_ordering(self):
        order = np.argsort(np.linalg.norm(self.mu, axis=1))
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
        prev = -np.inf
        for i in range(self.n_iter):
            ll = self._e_step(X)
            self._m_step(X)
            if abs(ll - prev) < self.tol:
                print(f"  HMM converged at iter {i + 1}")
                break
            prev = ll
        self.log_likelihood_ = ll
        return self

    def predict(self, X, use_filtered=False):
        X = np.asarray(X)
        self._e_step(X)
        return np.argmax(self.alpha if use_filtered else self.gamma, axis=1)


def build_granger_data(smb, hml, dates, normal_mask, lag=1):
    """Build boundary-clean Granger regression data for Normal regime."""
    T = len(smb)
    clean = normal_mask.copy()
    for k in range(1, lag + 1):
        shifted = np.zeros(T, dtype=bool)
        shifted[k:] = normal_mask[:T - k]
        clean &= shifted
    clean[:lag] = False
    idx = np.where(clean)[0]

    y = smb[idx]
    smb_lag = smb[idx - 1]
    hml_lag = hml[idx - 1]
    obs_dates = dates[idx]

    # Unrestricted: intercept + SMB lag + HML lag
    Xu = np.column_stack([np.ones(len(y)), smb_lag, hml_lag])

    return y, Xu, obs_dates


def chow_ftest(y, Xu, break_idx):
    """Chow F-test for structural break at break_idx."""
    n, k = Xu.shape
    if break_idx < k + 1 or (n - break_idx) < k + 1:
        return np.nan

    b_r, _, _, _ = np.linalg.lstsq(Xu, y, rcond=None)
    rss_r = float(np.sum((y - Xu @ b_r) ** 2))

    y1, X1 = y[:break_idx], Xu[:break_idx]
    b1, _, _, _ = np.linalg.lstsq(X1, y1, rcond=None)
    rss1 = float(np.sum((y1 - X1 @ b1) ** 2))

    y2, X2 = y[break_idx:], Xu[break_idx:]
    b2, _, _, _ = np.linalg.lstsq(X2, y2, rcond=None)
    rss2 = float(np.sum((y2 - X2 @ b2) ** 2))

    rss_u = rss1 + rss2
    dof_num = k
    dof_den = n - 2 * k

    if dof_den <= 0 or rss_u <= 0:
        return np.nan

    F = ((rss_r - rss_u) / dof_num) / (rss_u / dof_den)
    return float(F)


def find_breakpoint(y, Xu, obs_dates, trim=TRIM):
    """Find the breakpoint that maximizes Chow F-statistic."""
    n = len(y)
    lo = int(np.ceil(trim * n))
    hi = int(np.floor((1 - trim) * n))

    F_stats = np.full(n, np.nan)
    for bi in range(lo, hi + 1):
        F_stats[bi] = chow_ftest(y, Xu, bi)

    valid = ~np.isnan(F_stats)
    if not np.any(valid):
        return None, np.nan, None

    max_bi = int(np.nanargmax(F_stats))
    sup_F = float(F_stats[max_bi])
    break_date = obs_dates[max_bi]

    return max_bi, sup_F, break_date


def bootstrap_breakpoint(y, Xu, obs_dates, n_bootstrap=N_BOOTSTRAP, trim=TRIM, seed=42):
    """
    Bootstrap the breakpoint estimate using residual resampling.

    Method:
    1. Estimate full-sample regression
    2. Resample residuals with replacement
    3. Construct pseudo-sample y* = X*beta + e*
    4. Re-estimate breakpoint on pseudo-sample
    5. Repeat n_bootstrap times
    """
    np.random.seed(seed)
    n = len(y)

    # Full sample OLS
    beta_full, _, _, _ = np.linalg.lstsq(Xu, y, rcond=None)
    residuals = y - Xu @ beta_full

    bootstrap_breaks = []
    bootstrap_dates = []

    print(f"\nRunning {n_bootstrap} bootstrap samples...", flush=True)

    for b in range(n_bootstrap):
        if (b + 1) % 200 == 0:
            print(f"  Bootstrap {b + 1}/{n_bootstrap}", flush=True)

        # Resample residuals
        resamp_idx = np.random.choice(n, size=n, replace=True)
        e_star = residuals[resamp_idx]

        # Construct pseudo-sample
        y_star = Xu @ beta_full + e_star

        # Find breakpoint in pseudo-sample
        bi, sup_f, bdate = find_breakpoint(y_star, Xu, obs_dates, trim=trim)

        if bi is not None and not np.isnan(sup_f):
            bootstrap_breaks.append(bi)
            bootstrap_dates.append(bdate)

    return bootstrap_breaks, bootstrap_dates


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Download data
    df = download_ff_data()

    # 2. Fit Student-t HMM
    print(f"\nFitting Student-t HMM (K=3, seed={PRIMARY_SEED})...")
    hmm = StudentTHMM(n_regimes=3, n_iter=100, tol=1e-4, random_state=PRIMARY_SEED)
    hmm.fit(df.values)
    reg = hmm.predict(df.values, use_filtered=False)

    # Identify Normal regime (largest count)
    counts = [(int((reg == k).sum()), k) for k in range(3)]
    counts.sort(key=lambda x: -x[0])
    normal_idx = counts[0][1]
    print(f"\nRegime counts:")
    for cnt, k in counts:
        lbl = 'Normal' if k == normal_idx else f'regime{k}'
        print(f"  {lbl}: {cnt} days")

    normal_mask = (reg == normal_idx)
    hml = df['HML'].values
    smb = df['SMB'].values
    dates = np.array(df.index, dtype='datetime64[D]')

    # 3. Build boundary-clean Granger data
    y, Xu, obs_dates = build_granger_data(smb, hml, dates, normal_mask, lag=1)
    n = len(y)
    print(f"\nNormal-regime observations: n={n}")
    print(f"  Date range: {obs_dates[0]} to {obs_dates[-1]}")

    # 4. Find point estimate of breakpoint
    print("\n" + "=" * 60)
    print("BAI-PERRON STRUCTURAL BREAK TEST")
    print("=" * 60)

    point_idx, sup_F, point_date = find_breakpoint(y, Xu, obs_dates, trim=TRIM)

    # Andrews (1993) 5% critical value for q=3 regressors
    andrews_cv_5pct = 11.14
    reject = sup_F > andrews_cv_5pct if not np.isnan(sup_F) else False

    point_date_str = str(point_date)[:10] if point_date is not None else None
    print(f"\nPoint estimate:")
    print(f"  Break date: {point_date_str}")
    print(f"  Break index: {point_idx} of {n}")
    print(f"  sup-F: {sup_F:.3f}")
    print(f"  Andrews (1993) 5% CV: {andrews_cv_5pct:.2f}")
    print(f"  Reject H0 (no break): {reject}")

    # 5. Bootstrap CI
    print("\n" + "=" * 60)
    print("BOOTSTRAP CONFIDENCE INTERVAL")
    print("=" * 60)

    bootstrap_breaks, bootstrap_dates = bootstrap_breakpoint(
        y, Xu, obs_dates, n_bootstrap=N_BOOTSTRAP, trim=TRIM, seed=42
    )

    n_detected = len(bootstrap_breaks)
    fraction_detected = n_detected / N_BOOTSTRAP
    print(f"\nBootstrap samples detecting a break: {n_detected}/{N_BOOTSTRAP} ({fraction_detected*100:.1f}%)")

    if n_detected > 0:
        # Convert dates to numeric for quantile computation
        bootstrap_dates_np = np.array(bootstrap_dates, dtype='datetime64[D]')
        bootstrap_dates_sorted = np.sort(bootstrap_dates_np)

        alpha = 1 - CI_LEVEL
        lo_pct = alpha / 2
        hi_pct = 1 - alpha / 2

        lo_idx = int(np.floor(lo_pct * n_detected))
        hi_idx = int(np.ceil(hi_pct * n_detected)) - 1

        ci_lo = bootstrap_dates_sorted[lo_idx]
        ci_hi = bootstrap_dates_sorted[hi_idx]

        ci_lo_str = str(ci_lo)[:10]
        ci_hi_str = str(ci_hi)[:10]

        print(f"\n{int(CI_LEVEL*100)}% Bootstrap CI:")
        print(f"  Lower bound: {ci_lo_str}")
        print(f"  Upper bound: {ci_hi_str}")

        # Median bootstrap estimate
        median_idx = n_detected // 2
        median_date = bootstrap_dates_sorted[median_idx]
        median_date_str = str(median_date)[:10]
        print(f"  Median: {median_date_str}")
    else:
        ci_lo_str = None
        ci_hi_str = None
        median_date_str = None
        print("\nNo breaks detected in bootstrap samples.")

    # 6. Save results
    results = {
        'point_estimate': {
            'break_date': point_date_str,
            'break_index': int(point_idx) if point_idx is not None else None,
            'sup_F': float(sup_F) if not np.isnan(sup_F) else None,
            'andrews_cv_5pct': andrews_cv_5pct,
            'reject_h0': reject,
        },
        'bootstrap': {
            'n_bootstrap': N_BOOTSTRAP,
            'n_detected': n_detected,
            'fraction_detected': fraction_detected,
            'ci_level': CI_LEVEL,
            'ci_lower': ci_lo_str,
            'ci_upper': ci_hi_str,
            'median': median_date_str,
        },
        'data_info': {
            'n_normal_regime': int(n),
            'date_start': str(obs_dates[0])[:10],
            'date_end': str(obs_dates[-1])[:10],
            'hmm_seed': PRIMARY_SEED,
            'trim': TRIM,
        }
    }

    out_path = RESULTS_DIR / 'breakpoint_ci.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Point estimate: {point_date_str}")
    print(f"sup-F = {sup_F:.3f} (CV={andrews_cv_5pct:.2f}, reject H0: {reject})")
    print(f"Bootstrap {int(CI_LEVEL*100)}% CI: [{ci_lo_str}, {ci_hi_str}]")
    print(f"Fraction detecting break: {fraction_detected*100:.1f}%")


if __name__ == '__main__':
    main()
