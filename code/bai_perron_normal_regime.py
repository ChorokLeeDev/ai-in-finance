#!/usr/bin/env python3
"""
bai_perron_normal_regime.py
===========================
Bai-Perron / Quandt-Andrews sup-F structural break test on the
Normal-regime HML→SMB Granger coefficient (seed 28, primary fit).

Tests H0: no structural break in the HML lag-1 coefficient over time.
Scans all break dates in [15%, 85%] of Normal-regime observations
(chronologically sorted) and reports:
  - sup-F statistic vs Andrews (1993) 5% critical value
  - Estimated break date (argmax of Chow F)
  - Chow test at pre-specified Jan 2008 (GFC) break
  - CUSUM test via statsmodels
  - Rolling β(HML lag-1) with 200-obs window

Uses identical HMM fit (StudentT, K=3, seed=28) and boundary-clean
Granger design matrix (lag=1) as normal_split_corrected.py.

Output: results/bai_perron_normal.json + .log
"""

import json, io, warnings, urllib.request, zipfile, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.optimize import minimize_scalar
from scipy.cluster.vq import kmeans2
from scipy.stats import f as f_dist, chi2
import statsmodels.api as sm
from statsmodels.stats.diagnostic import breaks_cusumolsresid

warnings.filterwarnings('ignore')

BASE        = Path('/Users/i767700/Github/ai-in-finance/papers/causal_regimes')
RESULTS_DIR = BASE / 'results'
PRIMARY_SEED = 28
GFC_SPLIT    = np.datetime64('2008-01-01')
TRIM         = 0.15      # Bai-Perron trimming: 15%/85%
ROLL_WIN     = 200       # Rolling window for β coefficient


# ── Andrews (1993) asymptotic 5% critical values for sup-F ────────────────
# Table 1, trim=0.15, q = number of restricted coefficients
ANDREWS_CV_5PCT = {1: 8.85, 2: 10.13, 3: 11.14, 4: 11.83, 5: 12.37}


# ── FF data download ───────────────────────────────────────────────────────
def download_ff_data():
    print("Downloading FF5 + MOM data...")
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
    print(f"Loaded {len(df)} days")
    return df


# ── StudentT HMM (identical to normal_split_corrected.py) ─────────────────
class StudentTHMM:
    def __init__(self, n_regimes=3, n_iter=100, tol=1e-4, random_state=42):
        self.n_regimes=n_regimes; self.n_iter=n_iter; self.tol=tol
        self.random_state=random_state
        self.mu=self.Sigma=self.nu=self.A=self.pi=None
        self.gamma=self.alpha=self.xi=None; self.log_likelihood_=None

    def _init_params(self, X):
        np.random.seed(self.random_state); T,d=X.shape; K=self.n_regimes
        centroids,labels=kmeans2(X,K,minit='++')
        order=np.argsort(np.linalg.norm(centroids,axis=1))
        centroids=centroids[order]; nl=np.zeros_like(labels)
        for nk,ok in enumerate(order): nl[labels==ok]=nk
        labels=nl; self.mu=centroids; self.Sigma=np.zeros((K,d,d))
        for k in range(K):
            m=labels==k
            self.Sigma[k]=(np.cov(X[m].T)+1e-6*np.eye(d)) if m.sum()>d else np.eye(d)
        self.nu=np.array([15.,7.,4.]); self.A=np.eye(K)*.95+np.ones((K,K))*.05/K
        self.A/=self.A.sum(axis=1,keepdims=True); self.pi=np.ones(K)/K

    def _mvt_logpdf(self,x,mu,Sigma,nu):
        d=len(mu)
        if x.ndim==1: x=x.reshape(1,-1)
        diff=x-mu; Si=np.linalg.inv(Sigma); mahal=np.sum(diff@Si*diff,axis=1)
        _,ld=np.linalg.slogdet(Sigma)
        return (gammaln((nu+d)/2)-gammaln(nu/2)-0.5*d*np.log(nu*np.pi)
                -0.5*ld-0.5*(nu+d)*np.log(1+mahal/nu))

    def _log_B(self,X):
        T,d=X.shape; K=self.n_regimes; lb=np.zeros((T,K))
        for k in range(K): lb[:,k]=self._mvt_logpdf(X,self.mu[k],self.Sigma[k],self.nu[k])
        return lb

    def _forward(self,lb):
        T,K=lb.shape; la=np.zeros((T,K)); la[0]=np.log(self.pi+1e-300)+lb[0]
        lA=np.log(self.A+1e-300)
        for t in range(1,T):
            for k in range(K): la[t,k]=np.logaddexp.reduce(la[t-1]+lA[:,k])+lb[t,k]
        return la

    def _backward(self,lb):
        T,K=lb.shape; lb2=np.zeros((T,K)); lA=np.log(self.A+1e-300)
        for t in range(T-2,-1,-1):
            for k in range(K): lb2[t,k]=np.logaddexp.reduce(lA[k,:]+lb[t+1,:]+lb2[t+1,:])
        return lb2

    def _e_step(self,X):
        T,d=X.shape; K=self.n_regimes; lB=self._log_B(X)
        la=self._forward(lB); lb=self._backward(lB)
        ll=np.logaddexp.reduce(la[-1])
        lg=la+lb; lg-=np.logaddexp.reduce(lg,axis=1,keepdims=True)
        self.gamma=np.exp(lg)
        lan=la-np.logaddexp.reduce(la,axis=1,keepdims=True); self.alpha=np.exp(lan)
        lA=np.log(self.A+1e-300); self.xi=np.zeros((T-1,K,K))
        for t in range(T-1):
            for j in range(K):
                for k in range(K): self.xi[t,j,k]=np.exp(la[t,j]+lA[j,k]+lB[t+1,k]+lb[t+1,k]-ll)
        self.u=np.zeros((T,K))
        for k in range(K):
            diff=X-self.mu[k]; Si=np.linalg.inv(self.Sigma[k])
            mahal=np.sum(diff@Si*diff,axis=1); self.u[:,k]=(self.nu[k]+d)/(self.nu[k]+mahal)
        return ll

    def _m_step(self,X):
        T,d=X.shape; K=self.n_regimes
        self.pi=self.gamma[0]/self.gamma[0].sum()
        for j in range(K):
            for k in range(K): self.A[j,k]=self.xi[:,j,k].sum()/self.gamma[:-1,j].sum()
        self.A/=self.A.sum(axis=1,keepdims=True)
        for k in range(K):
            w=self.gamma[:,k]*self.u[:,k]; self.mu[k]=(w[:,None]*X).sum(0)/w.sum()
        for k in range(K):
            diff=X-self.mu[k]; w=self.gamma[:,k]*self.u[:,k]
            wo=sum(w[t]*np.outer(diff[t],diff[t]) for t in range(T))
            self.Sigma[k]=wo/self.gamma[:,k].sum()+1e-6*np.eye(d)
        for k in range(K): self._update_nu(X,k)
        self._enforce_ordering()

    def _update_nu(self,X,k):
        T,d=X.shape
        def nl(nu):
            if nu<=2: return 1e10
            diff=X-self.mu[k]; Si=np.linalg.inv(self.Sigma[k])
            mahal=np.sum(diff@Si*diff,axis=1)
            return -(self.gamma[:,k]*(gammaln((nu+d)/2)-gammaln(nu/2)
                    -0.5*d*np.log(nu)-0.5*(nu+d)*np.log(1+mahal/nu))).sum()
        self.nu[k]=minimize_scalar(nl,bounds=(2.1,50),method='bounded').x

    def _enforce_ordering(self):
        order=np.argsort(np.linalg.norm(self.mu,axis=1))
        if not np.array_equal(order,np.arange(self.n_regimes)):
            self.mu=self.mu[order]; self.Sigma=self.Sigma[order]
            self.nu=self.nu[order]; self.A=self.A[order][:,order]
            self.pi=self.pi[order]; self.gamma=self.gamma[:,order]
            if self.alpha is not None: self.alpha=self.alpha[:,order]
            if self.xi    is not None: self.xi=self.xi[:,order,:][:,:,order]

    def fit(self,X):
        X=np.asarray(X); self._init_params(X); prev=-np.inf
        for i in range(self.n_iter):
            ll=self._e_step(X); self._m_step(X)
            if abs(ll-prev)<self.tol:
                print(f"  Converged at iter {i+1}"); break
            prev=ll
        self.log_likelihood_=ll; return self

    def predict(self,X,use_filtered=False):
        X=np.asarray(X); self._e_step(X)
        return np.argmax(self.alpha if use_filtered else self.gamma,axis=1)


# ── Granger design matrix (boundary-clean, lag=1) ─────────────────────────
def build_granger_data(smb, hml, dates, normal_mask, lag=1):
    """
    Returns (y, X_restricted, X_full, obs_dates) for boundary-clean
    Normal-regime observations: both t and t-lag must be in Normal regime.
    """
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

    # Restricted: intercept + SMB lag
    Xr = np.column_stack([np.ones(len(y)), smb_lag])
    # Unrestricted: intercept + SMB lag + HML lag
    Xu = np.column_stack([np.ones(len(y)), smb_lag, hml_lag])

    return y, Xr, Xu, obs_dates, smb_lag, hml_lag


# ── Chow F-test at a given index break ────────────────────────────────────
def chow_ftest(y, Xu, break_idx):
    """
    Chow F-test for structural break in Xu coefficients at break_idx.
    Tests H0: all coefficients identical in [0:break_idx] and [break_idx:].
    """
    n, k = Xu.shape
    if break_idx < k + 1 or (n - break_idx) < k + 1:
        return np.nan, np.nan

    # Full-sample (restricted) OLS
    b_r, _, _, _ = np.linalg.lstsq(Xu, y, rcond=None)
    rss_r = float(np.sum((y - Xu @ b_r) ** 2))

    # Pre-break OLS
    y1, X1 = y[:break_idx], Xu[:break_idx]
    b1, _, _, _ = np.linalg.lstsq(X1, y1, rcond=None)
    rss1 = float(np.sum((y1 - X1 @ b1) ** 2))

    # Post-break OLS
    y2, X2 = y[break_idx:], Xu[break_idx:]
    b2, _, _, _ = np.linalg.lstsq(X2, y2, rcond=None)
    rss2 = float(np.sum((y2 - X2 @ b2) ** 2))

    rss_u = rss1 + rss2
    dof_num = k          # number of restricted params per segment
    dof_den = n - 2 * k

    if dof_den <= 0 or rss_u <= 0:
        return np.nan, np.nan

    F = ((rss_r - rss_u) / dof_num) / (rss_u / dof_den)
    p = float(1 - f_dist.cdf(F, dof_num, dof_den))
    return float(F), p


# ── Quandt-Andrews sup-F test ──────────────────────────────────────────────
def supF_test(y, Xu, obs_dates, trim=TRIM):
    """
    Quandt-Andrews sup-F test: scan break points in [trim*n, (1-trim)*n].
    Returns dict with sup_F, estimated break date, p-value series.
    """
    n = len(y)
    lo = int(np.ceil(trim * n))
    hi = int(np.floor((1 - trim) * n))

    F_stats = np.full(n, np.nan)
    P_stats = np.full(n, np.nan)

    for bi in range(lo, hi + 1):
        F, p = chow_ftest(y, Xu, bi)
        F_stats[bi] = F
        P_stats[bi] = p

    valid = ~np.isnan(F_stats)
    if not np.any(valid):
        return {'sup_F': np.nan, 'break_date': None, 'break_idx': None}

    max_bi = int(np.nanargmax(F_stats))
    sup_F  = float(F_stats[max_bi])
    break_date = str(obs_dates[max_bi])[:10]

    # Approximate p-value for sup-F using Hansen (1997) / Andrews (1993)
    # For q restricted params (= k = columns of Xu), 5% CV:
    q = Xu.shape[1]
    cv_5pct = ANDREWS_CV_5PCT.get(q, ANDREWS_CV_5PCT[5])
    reject_5pct = bool(sup_F > cv_5pct)

    return {
        'sup_F': sup_F,
        'break_date': break_date,
        'break_idx': max_bi,
        'q_restricted': q,
        'andrews_cv_5pct': cv_5pct,
        'reject_5pct': reject_5pct,
        'F_series': F_stats.tolist(),
        'P_series': P_stats.tolist(),
    }


# ── Rolling β(HML lag) ────────────────────────────────────────────────────
def rolling_beta(y, Xu, obs_dates, window=ROLL_WIN):
    """
    Rolling OLS β on HML lag (index 2 in Xu = [1, SMB_lag, HML_lag]).
    Returns (dates, betas, se).
    """
    n = len(y)
    betas, ses, roll_dates = [], [], []
    for i in range(window, n + 1):
        yi = y[i - window:i]
        Xi = Xu[i - window:i]
        try:
            res = sm.OLS(yi, Xi).fit(cov_type='HC3')
            betas.append(float(res.params[2]))
            ses.append(float(res.bse[2]))
            roll_dates.append(str(obs_dates[i - 1])[:10])
        except Exception:
            betas.append(np.nan)
            ses.append(np.nan)
            roll_dates.append(str(obs_dates[i - 1])[:10])
    return roll_dates, betas, ses


# ── CUSUM test ────────────────────────────────────────────────────────────
def cusum_test(y, Xu):
    """
    CUSUM test on OLS residuals (Brown, Durbin, Evans 1975).
    Returns (statistic, p-value) using statsmodels.
    """
    try:
        res = sm.OLS(y, Xu).fit()
        stat, p, _ = breaks_cusumolsresid(res.resid)
        return float(stat), float(p)
    except Exception as e:
        print(f"  CUSUM error: {e}")
        return np.nan, np.nan


# ── main ──────────────────────────────────────────────────────────────────
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = download_ff_data()

    print(f"\nFitting HMM seed={PRIMARY_SEED}...")
    hmm = StudentTHMM(n_regimes=3, n_iter=100, tol=1e-4, random_state=PRIMARY_SEED)
    hmm.fit(df.values)
    reg = hmm.predict(df.values, use_filtered=False)

    # Identify paper's Normal regime = largest count (4723 days)
    counts = [(int((reg == k).sum()), k) for k in range(3)]
    counts.sort(key=lambda x: -x[0])
    paper_normal_idx = counts[0][1]
    print(f"\nRegime counts:")
    for cnt, k in counts:
        lbl = {paper_normal_idx: 'Normal (paper)'}.get(k, f'regime{k}')
        print(f"  {lbl}: {cnt} days  mu_norm={np.linalg.norm(hmm.mu[k]):.3f}")

    normal_mask = (reg == paper_normal_idx)
    hml   = df['HML'].values
    smb   = df['SMB'].values
    dates = np.array(df.index, dtype='datetime64[D]')

    # Build boundary-clean Granger data (lag=1)
    y, Xr, Xu, obs_dates, smb_lag, hml_lag = build_granger_data(
        smb, hml, dates, normal_mask, lag=1
    )
    n = len(y)
    print(f"\nNormal-regime boundary-clean observations (lag=1): n={n}")
    print(f"  Date range: {obs_dates[0]} to {obs_dates[-1]}")

    # Calendar position of GFC split in observation sequence
    gfc_idx = int(np.searchsorted(obs_dates, GFC_SPLIT))
    print(f"  GFC split (Jan 2008) at obs index: {gfc_idx} of {n} "
          f"(pre={gfc_idx}, post={n - gfc_idx})")

    # ── 1. Quandt-Andrews sup-F test ──────────────────────────────────────
    print(f"\n=== QUANDT-ANDREWS sup-F TEST (trim={TRIM}) ===")
    supF = supF_test(y, Xu, obs_dates, trim=TRIM)
    print(f"  sup-F = {supF['sup_F']:.3f}")
    print(f"  Estimated break date: {supF['break_date']}  (obs idx={supF['break_idx']})")
    print(f"  Andrews (1993) 5% CV (q={supF['q_restricted']}): {supF['andrews_cv_5pct']:.2f}")
    print(f"  Reject H0 (no break) at 5%: {supF['reject_5pct']}")

    # ── 2. Chow test at pre-specified GFC date ────────────────────────────
    print(f"\n=== CHOW TEST AT PRE-SPECIFIED GFC BREAK (Jan 2008) ===")
    F_gfc, p_gfc = chow_ftest(y, Xu, gfc_idx)
    print(f"  Break index={gfc_idx}  pre-n={gfc_idx}  post-n={n - gfc_idx}")
    print(f"  Chow F = {F_gfc:.3f}  p = {p_gfc:.4e}")

    # Also test just the HML coefficient (partial Chow on HML lag only)
    # Compare β_HML pre vs post using a simple t-test on subsample estimates
    y_pre,  X_pre  = y[:gfc_idx],  Xu[:gfc_idx]
    y_post, X_post = y[gfc_idx:],  Xu[gfc_idx:]

    res_pre  = sm.OLS(y_pre,  X_pre).fit(cov_type='HC3')
    res_post = sm.OLS(y_post, X_post).fit(cov_type='HC3')

    beta_hml_pre  = float(res_pre.params[2])
    beta_hml_post = float(res_post.params[2])
    se_pre        = float(res_pre.bse[2])
    se_post       = float(res_post.bse[2])
    z_diff = (beta_hml_pre - beta_hml_post) / np.sqrt(se_pre**2 + se_post**2)
    p_diff = float(2 * (1 - f_dist.cdf(abs(z_diff)**2, 1, n - 6)))

    print(f"\n  β(HML lag) pre-2008:  {beta_hml_pre:.6f}  SE={se_pre:.6f}")
    print(f"  β(HML lag) post-2008: {beta_hml_post:.6f}  SE={se_post:.6f}")
    print(f"  Wald z={z_diff:.3f}  p={p_diff:.4e}")

    # ── 3. Grid of Chow tests (candidate break dates) ─────────────────────
    print(f"\n=== CHOW TESTS AT CANDIDATE BREAK DATES ===")
    candidate_dates = [
        ('2007-01-01', 'Jan 2007'),
        ('2008-01-01', 'Jan 2008 (GFC)'),
        ('2009-01-01', 'Jan 2009'),
        ('2010-01-01', 'Jan 2010'),
        ('2011-01-01', 'Jan 2011'),
        ('2012-01-01', 'Jan 2012'),
    ]
    grid_results = {}
    for date_str, label in candidate_dates:
        bi = int(np.searchsorted(obs_dates, np.datetime64(date_str)))
        if bi < 10 or bi > n - 10:
            print(f"  {label}: insufficient data")
            continue
        F_, p_ = chow_ftest(y, Xu, bi)
        sig = '***' if p_ < 0.001 else '**' if p_ < 0.01 else '*' if p_ < 0.05 else '†' if p_ < 0.10 else ''
        print(f"  {label} (obs idx={bi}): F={F_:.3f}  p={p_:.4e} {sig}")
        grid_results[date_str] = {'label': label, 'bi': bi, 'F': F_, 'p': p_}

    # ── 4. CUSUM test ──────────────────────────────────────────────────────
    print(f"\n=== CUSUM TEST (Brown-Durbin-Evans 1975) ===")
    cusum_stat, cusum_p = cusum_test(y, Xu)
    print(f"  CUSUM stat={cusum_stat:.4f}  p={cusum_p:.4e}")

    # ── 5. Rolling β(HML lag) ─────────────────────────────────────────────
    print(f"\n=== ROLLING β(HML lag-1), window={ROLL_WIN} ===")
    roll_dates, roll_betas, roll_ses = rolling_beta(y, Xu, obs_dates, window=ROLL_WIN)

    # Find when β crosses zero (post-GFC attenuation)
    roll_arr = np.array(roll_betas)
    valid_mask = ~np.isnan(roll_arr)
    if valid_mask.any():
        max_beta_idx = int(np.nanargmax(roll_arr))
        min_beta_idx = int(np.nanargmin(roll_arr))
        print(f"  Max β = {roll_arr[max_beta_idx]:.5f} at {roll_dates[max_beta_idx]}")
        print(f"  Min β = {roll_arr[min_beta_idx]:.5f} at {roll_dates[min_beta_idx]}")

        # β at GFC break
        gfc_roll_idx = int(np.searchsorted(
            [d[:10] for d in roll_dates], str(GFC_SPLIT)[:10]
        ))
        if gfc_roll_idx < len(roll_dates):
            print(f"  β at GFC (Jan 2008): {roll_arr[gfc_roll_idx]:.5f} "
                  f"(±{roll_ses[gfc_roll_idx]:.5f})")

    # ── Save results ───────────────────────────────────────────────────────
    out = {
        'n_normal_clean': int(n),
        'gfc_split_idx':  int(gfc_idx),
        'supF': {
            'stat': supF['sup_F'],
            'break_date': supF['break_date'],
            'break_idx': supF['break_idx'],
            'q_restricted': supF['q_restricted'],
            'andrews_cv_5pct': supF['andrews_cv_5pct'],
            'reject_5pct': supF['reject_5pct'],
        },
        'chow_gfc_2008': {
            'F': float(F_gfc),
            'p': float(p_gfc),
            'beta_hml_pre':  beta_hml_pre,
            'beta_hml_post': beta_hml_post,
            'se_pre':        se_pre,
            'se_post':       se_post,
            'z_diff':        float(z_diff),
            'p_diff':        float(p_diff),
        },
        'chow_grid': {k: {'F': v['F'], 'p': v['p']} for k, v in grid_results.items()},
        'cusum': {'stat': cusum_stat, 'p': cusum_p},
        'rolling_beta': {
            'window': ROLL_WIN,
            'dates': roll_dates,
            'betas': [float(b) if not np.isnan(b) else None for b in roll_betas],
            'ses':   [float(s) if not np.isnan(s) else None for s in roll_ses],
        },
    }

    out_path = RESULTS_DIR / 'bai_perron_normal.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY — Bai-Perron / sup-F structural break, Normal regime")
    print("=" * 60)
    print(f"sup-F = {supF['sup_F']:.3f}  (Andrews 5% CV = {supF['andrews_cv_5pct']:.2f})")
    print(f"  → Reject H0 (no break): {supF['reject_5pct']}")
    print(f"  → MLE break date: {supF['break_date']}")
    print(f"Chow at GFC (Jan 2008): F={F_gfc:.3f}  p={p_gfc:.4e}")
    print(f"  β(HML): {beta_hml_pre:.5f} (pre) → {beta_hml_post:.5f} (post)")
    print(f"  Wald for β difference: z={z_diff:.3f}  p={p_diff:.4e}")
    print(f"CUSUM: stat={cusum_stat:.4f}  p={cusum_p:.4e}")
    print("Done.")


if __name__ == '__main__':
    main()
