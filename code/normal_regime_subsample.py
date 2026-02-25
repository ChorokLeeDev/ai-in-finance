#!/usr/bin/env python3
"""
normal_regime_subsample.py
==========================

Two analyses for the "strong accept" revision:

1. NORMAL-REGIME PRE/POST-2008 SPLIT
   Does HML→SMB hold in Normal regime both before and after the GFC?
   Split: Pre-GFC  = 1990-01-02 to 2007-12-31
          Post-GFC = 2008-01-01 to 2024-12-31
   Sensitivity: Pre-2009  = 1990-01-02 to 2008-12-31
                Post-2009 = 2009-01-01 to 2024-12-31

2. ANDREWS HAC P-VALUE (exact) for frozen OOS Elevated result
   Re-runs Table 3 (frozen OOS Elevated, lag=1) with Andrews (1991)
   data-driven bandwidth to confirm "p < 0.05" claim precisely.
   Loads frozen OOS assignments from frozen_oos_primary.json.

Uses PRIMARY fit (random_state=28, highest LL optimum) regime assignments,
matching the paper's main results table (Table 2).
Methodology identical to hac_granger_consistent.py.

Saves results to results/normal_regime_subsample.json
"""

import json, os, io, sys, warnings, urllib.request, zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.optimize import minimize_scalar
from scipy.cluster.vq import kmeans2
from scipy.stats import f as f_dist, chi2
import statsmodels.api as sm

warnings.filterwarnings('ignore')

BASE        = Path('/Users/i767700/Github/ai-in-finance/papers/causal_regimes')
RESULTS_DIR = BASE / 'results'
OUTPUT_PATH = RESULTS_DIR / 'normal_regime_subsample.json'

PRIMARY_SEED = 28   # Paper's primary fit
GFC_SPLIT    = '2008-01-01'   # Pre/post-GFC primary split
ALT_SPLIT    = '2009-01-01'   # Sensitivity: exclude 2008 crisis year
LAG_FIXED    = 1
LAG_MAX      = 15


# =============================================================================
# DATA
# =============================================================================

def download_ff_data():
    print("Downloading Fama-French 5-factor daily data...")
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
    for c in ['Mkt-RF','SMB','HML','RMW','CMA','RF']:
        df5[c] = pd.to_numeric(df5[c], errors='coerce')
    df5 = df5.set_index('Date').dropna()

    print("Downloading Momentum factor daily data...")
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

    df = df5.join(mom[['MOM']], how='inner').rename(columns={'Mkt-RF':'MKT'})
    df = df.drop('RF', axis=1, errors='ignore')
    df = df.loc['1990-01-01':'2024-12-31']
    print(f"Loaded {len(df)} trading days ({df.index[0].date()} to {df.index[-1].date()})")
    return df


# =============================================================================
# STUDENT-T HMM  (identical to hac_granger_consistent.py)
# =============================================================================

class StudentTHMM:
    def __init__(self, n_regimes=3, n_iter=100, tol=1e-4, random_state=42):
        self.n_regimes = n_regimes
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state
        self.mu = self.Sigma = self.nu = self.A = self.pi = None
        self.gamma = self.alpha = None
        self.log_likelihood_ = None

    def _init_params(self, X):
        np.random.seed(self.random_state)
        T, d = X.shape; K = self.n_regimes
        centroids, labels = kmeans2(X, K, minit='++')
        order = np.argsort(np.linalg.norm(centroids, axis=1))
        centroids = centroids[order]
        new_labels = np.zeros_like(labels)
        for nk, ok in enumerate(order):
            new_labels[labels == ok] = nk
        labels = new_labels
        self.mu = centroids
        self.Sigma = np.zeros((K, d, d))
        for k in range(K):
            m = labels == k
            self.Sigma[k] = (np.cov(X[m].T) + 1e-6*np.eye(d)) if m.sum() > d else np.eye(d)
        self.nu = np.array([15.0, 7.0, 4.0])
        self.A  = np.eye(K)*0.95 + np.ones((K,K))*0.05/K
        self.A  = self.A / self.A.sum(axis=1, keepdims=True)
        self.pi = np.ones(K) / K

    def _mvt_logpdf(self, x, mu, Sigma, nu):
        d = len(mu)
        if x.ndim == 1: x = x.reshape(1,-1)
        diff   = x - mu
        Si     = np.linalg.inv(Sigma)
        mahal  = np.sum(diff @ Si * diff, axis=1)
        _, ld  = np.linalg.slogdet(Sigma)
        return (gammaln((nu+d)/2) - gammaln(nu/2)
                - 0.5*d*np.log(nu*np.pi) - 0.5*ld
                - 0.5*(nu+d)*np.log(1 + mahal/nu))

    def _compute_log_B(self, X):
        T, d = X.shape; K = self.n_regimes
        log_B = np.zeros((T, K))
        for k in range(K):
            log_B[:,k] = self._mvt_logpdf(X, self.mu[k], self.Sigma[k], self.nu[k])
        return log_B

    def _forward(self, log_B):
        T, K = log_B.shape
        la = np.zeros((T, K)); la[0] = np.log(self.pi+1e-300) + log_B[0]
        lA = np.log(self.A+1e-300)
        for t in range(1, T):
            for k in range(K):
                la[t,k] = np.logaddexp.reduce(la[t-1] + lA[:,k]) + log_B[t,k]
        return la

    def _backward(self, log_B):
        T, K = log_B.shape
        lb = np.zeros((T, K)); lA = np.log(self.A+1e-300)
        for t in range(T-2, -1, -1):
            for k in range(K):
                lb[t,k] = np.logaddexp.reduce(lA[k,:] + log_B[t+1,:] + lb[t+1,:])
        return lb

    def _e_step(self, X):
        T, d = X.shape; K = self.n_regimes
        log_B = self._compute_log_B(X)
        la = self._forward(log_B); lb = self._backward(log_B)
        ll = np.logaddexp.reduce(la[-1])
        lg = la + lb; lg -= np.logaddexp.reduce(lg, axis=1, keepdims=True)
        self.gamma = np.exp(lg)
        lan = la - np.logaddexp.reduce(la, axis=1, keepdims=True)
        self.alpha = np.exp(lan)
        lA = np.log(self.A+1e-300)
        self.xi = np.zeros((T-1, K, K))
        for t in range(T-1):
            for j in range(K):
                for k in range(K):
                    self.xi[t,j,k] = np.exp(la[t,j]+lA[j,k]+log_B[t+1,k]+lb[t+1,k]-ll)
        self.u = np.zeros((T, K))
        for k in range(K):
            diff  = X - self.mu[k]
            Si    = np.linalg.inv(self.Sigma[k])
            mahal = np.sum(diff @ Si * diff, axis=1)
            self.u[:,k] = (self.nu[k]+d)/(self.nu[k]+mahal)
        return ll

    def _m_step(self, X):
        T, d = X.shape; K = self.n_regimes
        self.pi = self.gamma[0] / self.gamma[0].sum()
        for j in range(K):
            for k in range(K):
                self.A[j,k] = self.xi[:,j,k].sum() / self.gamma[:-1,j].sum()
        self.A /= self.A.sum(axis=1, keepdims=True)
        for k in range(K):
            w = self.gamma[:,k]*self.u[:,k]
            self.mu[k] = (w[:,None]*X).sum(0)/w.sum()
        for k in range(K):
            diff = X - self.mu[k]; w = self.gamma[:,k]*self.u[:,k]
            wo = sum(w[t]*np.outer(diff[t],diff[t]) for t in range(T))
            self.Sigma[k] = wo/self.gamma[:,k].sum() + 1e-6*np.eye(d)
        for k in range(K):
            self._update_nu(X, k)
        self._enforce_ordering()

    def _update_nu(self, X, k):
        T, d = X.shape
        def neg_ll(nu):
            if nu <= 2: return 1e10
            diff = X - self.mu[k]; Si = np.linalg.inv(self.Sigma[k])
            mahal = np.sum(diff @ Si * diff, axis=1)
            t1 = gammaln((nu+d)/2)-gammaln(nu/2)
            t2 = -0.5*d*np.log(nu)
            t3 = -0.5*(nu+d)*np.log(1+mahal/nu)
            return -(self.gamma[:,k]*(t1+t2+t3)).sum()
        res = minimize_scalar(neg_ll, bounds=(2.1,50), method='bounded')
        self.nu[k] = res.x

    def _enforce_ordering(self):
        order = np.argsort(np.linalg.norm(self.mu, axis=1))
        if not np.array_equal(order, np.arange(self.n_regimes)):
            self.mu = self.mu[order]; self.Sigma = self.Sigma[order]
            self.nu = self.nu[order]; self.A = self.A[order][:,order]
            self.pi = self.pi[order]; self.gamma = self.gamma[:,order]
            if self.alpha is not None: self.alpha = self.alpha[:,order]
            if self.xi    is not None: self.xi = self.xi[:,order,:][:,:,order]

    def fit(self, X):
        X = np.asarray(X); self._init_params(X)
        prev_ll = -np.inf
        for i in range(self.n_iter):
            ll = self._e_step(X); self._m_step(X)
            if abs(ll - prev_ll) < self.tol:
                print(f"  Converged at iter {i+1}")
                break
            prev_ll = ll
        self.log_likelihood_ = ll
        return self

    def predict(self, X, use_filtered=False):
        X = np.asarray(X); self._e_step(X)
        return np.argmax(self.alpha if use_filtered else self.gamma, axis=1)


# =============================================================================
# GRANGER TEST FUNCTIONS  (matching hac_granger_consistent.py)
# =============================================================================

def extract_clean_indices(regime_mask, date_mask, lag):
    """
    Returns indices where:
      - current day is in regime AND in period (regime_mask & date_mask)
      - preceding `lag` days are in regime (regime_mask only)
    """
    clean = (regime_mask & date_mask).copy()
    for k in range(1, lag+1):
        shifted = np.zeros(len(regime_mask), dtype=bool)
        shifted[k:] = regime_mask[:len(regime_mask)-k]
        clean &= shifted
    clean[:lag] = False
    return np.where(clean)[0]


def select_lag_bic(y_all, x_all, regime_mask, date_mask, max_lag=15):
    best_bic, best_lag = np.inf, 1
    for lag in range(1, max_lag+1):
        idx = extract_clean_indices(regime_mask, date_mask, lag)
        if len(idx) < 2*lag+10: continue
        y_curr   = y_all[idx]
        y_lagged = np.column_stack([y_all[idx-i-1] for i in range(lag)])
        x_lagged = np.column_stack([x_all[idx-i-1] for i in range(lag)])
        X_u = np.column_stack([np.ones(len(y_curr)), y_lagged, x_lagged])
        n, k = X_u.shape
        try:
            beta = np.linalg.lstsq(X_u, y_curr, rcond=None)[0]
            rss  = np.sum((y_curr - X_u@beta)**2)
            bic  = n*np.log(rss/n) + k*np.log(n)
            if bic < best_bic: best_bic, best_lag = bic, lag
        except Exception: continue
    return best_lag


def standard_ftest(y_curr, y_lagged, x_lagged):
    n, lag = len(y_curr), y_lagged.shape[1]
    Xr = np.column_stack([np.ones(n), y_lagged])
    Xu = np.column_stack([np.ones(n), y_lagged, x_lagged])
    br = np.linalg.lstsq(Xr, y_curr, rcond=None)[0]
    bu = np.linalg.lstsq(Xu, y_curr, rcond=None)[0]
    rss_r = float(np.sum((y_curr - Xr@br)**2))
    rss_u = float(np.sum((y_curr - Xu@bu)**2))
    df1, df2 = lag, n-2*lag-1
    if df2 <= 0 or rss_u <= 0: return np.nan, np.nan, np.nan
    F = ((rss_r-rss_u)/df1)/(rss_u/df2)
    p = float(1 - f_dist.cdf(F, df1, df2))
    tss = float(np.sum((y_curr-y_curr.mean())**2))
    dr2 = (1-rss_u/tss) - (1-rss_r/tss)
    return float(F), p, float(dr2)


def hac_wald(y_curr, y_lagged, x_lagged, bw):
    n, p = len(y_curr), y_lagged.shape[1]
    Xu = np.column_stack([np.ones(n), y_lagged, x_lagged])
    res = sm.OLS(y_curr, Xu).fit(cov_type='HAC', cov_kwds={'maxlags': bw})
    R = np.zeros((p, Xu.shape[1]))
    for i in range(p): R[i, 1+p+i] = 1.0
    Rb = R @ res.params; RVR = R @ res.cov_params() @ R.T
    try:
        W = float(Rb @ np.linalg.inv(RVR) @ Rb)
        return W, float(1 - chi2.cdf(W, p))
    except np.linalg.LinAlgError:
        return np.nan, np.nan


def andrews_bw(residuals):
    """Andrews (1991) AR(1) plug-in bandwidth."""
    n = len(residuals)
    if n < 4: return 1
    rho = float(np.dot(residuals[:-1], residuals[1:]) / np.dot(residuals[:-1], residuals[:-1]))
    rho = np.clip(rho, -0.999, 0.999)
    a   = 4*rho**2 / (1-rho**2)**2
    return max(1, int(np.floor(1.1447*(a*n)**(1/3))))


def run_granger(y_all, x_all, regime_mask, date_mask, label, use_bic=True, fixed_lag=LAG_FIXED):
    opt_lag = select_lag_bic(y_all, x_all, regime_mask, date_mask, LAG_MAX) if use_bic else fixed_lag
    idx = extract_clean_indices(regime_mask, date_mask, opt_lag)
    n   = len(idx)

    if n < 2*opt_lag+10:
        print(f"  {label}: insufficient obs (n={n}, lag={opt_lag})")
        return {'n_obs': n, 'lag': opt_lag, 'status': 'insufficient'}

    y_curr   = y_all[idx]
    y_lagged = np.column_stack([y_all[idx-i-1] for i in range(opt_lag)])
    x_lagged = np.column_stack([x_all[idx-i-1] for i in range(opt_lag)])

    F, f_p, dr2 = standard_ftest(y_curr, y_lagged, x_lagged)

    # HAC bandwidth = lag (paper's primary spec)
    _, hac_p_lag = hac_wald(y_curr, y_lagged, x_lagged, opt_lag)

    # Andrews bandwidth
    Xu  = np.column_stack([np.ones(n), y_lagged, x_lagged])
    bu  = np.linalg.lstsq(Xu, y_curr, rcond=None)[0]
    bw  = andrews_bw(y_curr - Xu@bu)
    _, hac_p_and = hac_wald(y_curr, y_lagged, x_lagged, bw)

    def sig(p):
        return '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else '†' if p<0.10 else ''

    print(f"  {label} (n={n}, lag={opt_lag}, Andrews bw={bw}):")
    print(f"    F-test:        F={F:.3f},  p={f_p:.4e} {sig(f_p)}")
    print(f"    HAC(bw={opt_lag:2d}):    p={hac_p_lag:.4e} {sig(hac_p_lag)}")
    print(f"    HAC(Andrews):  p={hac_p_and:.4e} {sig(hac_p_and)}")
    print(f"    ΔR²:           {dr2*100:.4f}%")

    return {
        'n_obs':         n,
        'lag':           opt_lag,
        'f_stat':        float(F),
        'f_p':           float(f_p),
        'delta_r2_pct':  float(dr2*100),
        'hac_bw_lag':    opt_lag,
        'hac_p_lag':     float(hac_p_lag),
        'andrews_bw':    bw,
        'hac_p_andrews': float(hac_p_and),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── 1. Download factor data ───────────────────────────────────────────
    df = download_ff_data()

    # ── 2. Fit HMM with PRIMARY seed (28) ────────────────────────────────
    print(f"\nFitting Student-t HMM (K=3, seed={PRIMARY_SEED}) — primary fit...")
    hmm = StudentTHMM(n_regimes=3, n_iter=100, tol=1e-4, random_state=PRIMARY_SEED)
    hmm.fit(df.values)

    regimes = hmm.predict(df.values, use_filtered=False)
    regime_names = ['Normal', 'Elevated', 'Crisis']

    print("\nRegime distribution (seed 28, full sample 1990-2024):")
    for k, rn in enumerate(regime_names):
        cnt = int((regimes == k).sum())
        print(f"  {rn}: {cnt} days ({cnt/len(df):.1%})")

    print(f"  Train LL: {hmm.log_likelihood_:.2f}")
    print(f"  nu:  {[round(v,2) for v in hmm.nu]}")
    print(f"  diag(A): {[round(hmm.A[k,k],4) for k in range(3)]}")

    # Verify this matches the paper's primary fit (seed 28)
    # Expected: Normal≈4723, Elevated≈3023, Crisis≈1071
    n_normal = int((regimes == 0).sum())
    if abs(n_normal - 4723) > 100:
        print(f"WARNING: Normal count {n_normal} differs significantly from expected 4723")
        print("  This may indicate a different local optimum. Proceeding anyway.")

    # ── 3. Build arrays and masks ─────────────────────────────────────────
    dates      = df.index
    hml_all    = df['HML'].values
    smb_all    = df['SMB'].values

    normal_mask   = (regimes == 0)
    full_mask     = np.ones(len(dates), dtype=bool)

    # Period masks using numpy datetime64
    dates_np      = np.array(dates, dtype='datetime64[D]')
    pre_gfc_mask  = dates_np < np.datetime64(GFC_SPLIT)
    post_gfc_mask = dates_np >= np.datetime64(GFC_SPLIT)
    alt_pre_mask  = dates_np < np.datetime64(ALT_SPLIT)
    alt_post_mask = dates_np >= np.datetime64(ALT_SPLIT)

    # Summary counts
    def cnt(m1, m2): return int((m1 & m2).sum())
    print(f"\nNormal-regime day counts (seed {PRIMARY_SEED}):")
    print(f"  Total:               {cnt(normal_mask, full_mask)}")
    print(f"  Pre-2008 (1990-2007): {cnt(normal_mask, pre_gfc_mask)}")
    print(f"  Post-2008 (2008-2024):{cnt(normal_mask, post_gfc_mask)}")
    print(f"  Pre-2009 (1990-2008): {cnt(normal_mask, alt_pre_mask)}")
    print(f"  Post-2009 (2009-2024):{cnt(normal_mask, alt_post_mask)}")

    # ── 4. Analysis 1: Pre/post-2008 split — HML→SMB ─────────────────────
    print("\n" + "="*65)
    print("ANALYSIS 1: NORMAL-REGIME HML→SMB PRE/POST-2008 SPLIT")
    print(f"  Split: {GFC_SPLIT}")
    print("="*65)

    print("\n[Full Normal regime, 1990-2024 — replication check]")
    res_full = run_granger(smb_all, hml_all, normal_mask, full_mask, "Normal (full)")

    print("\n[Pre-GFC: Normal regime, 1990-2007]")
    res_pre = run_granger(smb_all, hml_all, normal_mask, pre_gfc_mask, "Normal pre-2008")

    print("\n[Post-GFC: Normal regime, 2008-2024]")
    res_post = run_granger(smb_all, hml_all, normal_mask, post_gfc_mask, "Normal post-2008")

    # ── 5. Analysis 2: Sensitivity pre/post-2009 ─────────────────────────
    print("\n" + "="*65)
    print("ANALYSIS 2: SENSITIVITY — PRE/POST-2009 (excludes 2008)")
    print(f"  Split: {ALT_SPLIT}")
    print("="*65)

    print("\n[Pre-2009: Normal regime, 1990-2008]")
    res_pre2 = run_granger(smb_all, hml_all, normal_mask, alt_pre_mask, "Normal pre-2009")

    print("\n[Post-2009: Normal regime, 2009-2024]")
    res_post2 = run_granger(smb_all, hml_all, normal_mask, alt_post_mask, "Normal post-2009")

    # ── 6. Analysis 3: Reverse direction SMB→HML ─────────────────────────
    print("\n" + "="*65)
    print("ANALYSIS 3: REVERSE DIRECTION — SMB→HML IN NORMAL REGIME")
    print("="*65)

    print("\n[Full Normal regime, SMB→HML]")
    res_rev = run_granger(hml_all, smb_all, normal_mask, full_mask, "Normal SMB→HML (full)")

    print("\n[Pre-2008, SMB→HML]")
    res_rev_pre = run_granger(hml_all, smb_all, normal_mask, pre_gfc_mask, "Normal SMB→HML pre-2008")

    print("\n[Post-2008, SMB→HML]")
    res_rev_post = run_granger(hml_all, smb_all, normal_mask, post_gfc_mask, "Normal SMB→HML post-2008")

    # ── 7. Analysis 4: Andrews HAC for frozen OOS Elevated ────────────────
    print("\n" + "="*65)
    print("ANALYSIS 4: FROZEN OOS ELEVATED — EXACT ANDREWS HAC P-VALUE")
    print("  Re-fitting HMM on 1990-2012 train, classifying 2013-2024 OOS")
    print("="*65)

    # Fit frozen HMM on 1990-2012 training data
    train_mask = dates_np < np.datetime64('2013-01-01')
    oos_mask   = dates_np >= np.datetime64('2013-01-01')
    df_train   = df[train_mask]

    print(f"\nFitting frozen HMM on 1990-2012 ({train_mask.sum()} days), seed={PRIMARY_SEED}...")
    hmm_frozen = StudentTHMM(n_regimes=3, n_iter=100, tol=1e-4, random_state=PRIMARY_SEED)
    hmm_frozen.fit(df_train.values)
    print(f"  Train LL: {hmm_frozen.log_likelihood_:.2f}")

    # Classify OOS using frozen parameters (filtered probabilities only)
    hmm_frozen._e_step(df.values)  # run full sequence using train params
    oos_regimes = np.argmax(hmm_frozen.alpha, axis=1)  # filtered (no lookahead)
    oos_regimes_oos = oos_regimes[oos_mask]

    # Relabel by matching to training regime ordering (by norm)
    print("\nFrozen OOS regime distribution (2013-2024):")
    for k, rn in enumerate(regime_names):
        cnt_k = int((oos_regimes_oos == k).sum())
        print(f"  {rn}: {cnt_k} days ({cnt_k/oos_mask.sum():.1%})")

    elevated_in_oos = np.zeros(len(dates), dtype=bool)
    elevated_in_oos[oos_mask] = (oos_regimes_oos == 1)  # regime 1 = Elevated

    hml_oos_all = df['HML'].values
    smb_oos_all = df['SMB'].values

    print("\n[Frozen OOS Elevated: HML→SMB, fixed lag=1 (paper's Table 3 spec)]")
    res_oos = run_granger(smb_oos_all, hml_oos_all, elevated_in_oos, oos_mask,
                          "OOS Elevated HML→SMB", use_bic=False, fixed_lag=1)

    # ── 8. Summary table ─────────────────────────────────────────────────
    print("\n" + "="*65)
    print("SUMMARY — Normal-regime HML→SMB (seed 28, primary fit)")
    print("="*65)
    hdr = f"{'Period':<24} {'n':>6}  {'Lag':>3}  {'F-p':>12}  {'HAC(bw=lag)':>12}  {'HAC(Andrews)':>13}"
    print(hdr); print("-"*80)

    rows = [
        ("Normal (full, rep check)", res_full),
        ("Normal pre-2008",          res_pre),
        ("Normal post-2008",         res_post),
        ("Normal pre-2009",          res_pre2),
        ("Normal post-2009",         res_post2),
    ]
    for lbl, r in rows:
        if r and r.get('status') != 'insufficient':
            print(f"{lbl:<24} {r['n_obs']:>6}  {r['lag']:>3}  "
                  f"{r['f_p']:>12.4e}  {r['hac_p_lag']:>12.4e}  "
                  f"{r['hac_p_andrews']:>13.4e}  (bw={r['andrews_bw']})")
        else:
            print(f"{lbl:<24} {'—':>6}  {'—':>3}  {'insufficient':>12}")

    print("\nFrozen OOS Elevated (Table 3 replication):")
    if res_oos and res_oos.get('status') != 'insufficient':
        print(f"  n={res_oos['n_obs']}, lag={res_oos['lag']}, "
              f"F-p={res_oos['f_p']:.4e}, HAC(bw=1)={res_oos['hac_p_lag']:.4e}, "
              f"HAC(Andrews bw={res_oos['andrews_bw']})={res_oos['hac_p_andrews']:.4e}")

    # ── 9. Save ───────────────────────────────────────────────────────────
    output = {
        'description': 'Normal-regime HML→SMB pre/post-GFC split + Andrews HAC (seed 28)',
        'primary_seed': PRIMARY_SEED,
        'gfc_split': GFC_SPLIT,
        'alt_split': ALT_SPLIT,
        'regime_counts_seed28': {rn: int((regimes==k).sum()) for k,rn in enumerate(regime_names)},
        'hml_to_smb': {
            'full':      res_full,
            'pre_2008':  res_pre,
            'post_2008': res_post,
            'pre_2009':  res_pre2,
            'post_2009': res_post2,
        },
        'smb_to_hml': {
            'full':     res_rev,
            'pre_2008': res_rev_pre,
            'post_2008':res_rev_post,
        },
        'oos_elevated_andrews': res_oos,
    }

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved → {OUTPUT_PATH}")
    print("Done.")


if __name__ == '__main__':
    main()
