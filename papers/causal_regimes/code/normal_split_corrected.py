#!/usr/bin/env python3
"""
normal_split_corrected.py
=========================
Same HMM fit as normal_regime_subsample.py (seed=28) but analyses
the CORRECT regime: regime index 1 (the 4,723-day state = paper's "Normal").

The label swap in the subsample script (regime 0 = 3,023 days = paper's Elevated,
regime 1 = 4,723 days = paper's Normal) means we need to target regime 1 here.

Also: for Andrews OOS analysis, uses regime 1 of the frozen fit = OOS Normal.
"""

import json, io, warnings, urllib.request, zipfile, os
from pathlib import Path
import numpy as np, pandas as pd
from scipy.special import gammaln
from scipy.optimize import minimize_scalar
from scipy.cluster.vq import kmeans2
from scipy.stats import f as f_dist, chi2
import statsmodels.api as sm

warnings.filterwarnings('ignore')
BASE        = Path('/Users/i767700/Github/ai-in-finance/papers/causal_regimes')
RESULTS_DIR = BASE / 'results'

PRIMARY_SEED = 28
GFC_SPLIT    = '2008-01-01'
ALT_SPLIT    = '2009-01-01'


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


def extract_clean(regime_mask, date_mask, lag):
    clean=(regime_mask & date_mask).copy()
    for k in range(1,lag+1):
        sh=np.zeros(len(regime_mask),dtype=bool)
        sh[k:]=regime_mask[:len(regime_mask)-k]; clean&=sh
    clean[:lag]=False
    return np.where(clean)[0]

def bic_lag(y,x,rm,dm,mlag=15):
    best_bic,best_lag=np.inf,1
    for lag in range(1,mlag+1):
        idx=extract_clean(rm,dm,lag)
        if len(idx)<2*lag+10: continue
        yc=y[idx]; yl=np.column_stack([y[idx-i-1] for i in range(lag)])
        xl=np.column_stack([x[idx-i-1] for i in range(lag)])
        Xu=np.column_stack([np.ones(len(yc)),yl,xl]); n,k=Xu.shape
        try:
            b=np.linalg.lstsq(Xu,yc,rcond=None)[0]; rss=np.sum((yc-Xu@b)**2)
            bic=n*np.log(rss/n)+k*np.log(n)
            if bic<best_bic: best_bic,best_lag=bic,lag
        except: pass
    return best_lag

def ftest(yc,yl,xl):
    n,lag=len(yc),yl.shape[1]
    Xr=np.column_stack([np.ones(n),yl]); Xu=np.column_stack([np.ones(n),yl,xl])
    br=np.linalg.lstsq(Xr,yc,rcond=None)[0]; bu=np.linalg.lstsq(Xu,yc,rcond=None)[0]
    rr=float(np.sum((yc-Xr@br)**2)); ru=float(np.sum((yc-Xu@bu)**2))
    df1,df2=lag,n-2*lag-1
    if df2<=0 or ru<=0: return np.nan,np.nan,np.nan
    F=((rr-ru)/df1)/(ru/df2); p=float(1-f_dist.cdf(F,df1,df2))
    tss=float(np.sum((yc-yc.mean())**2)); dr2=(1-ru/tss)-(1-rr/tss)
    return float(F),p,float(dr2)

def hac(yc,yl,xl,bw):
    n,p=len(yc),yl.shape[1]
    Xu=np.column_stack([np.ones(n),yl,xl])
    res=sm.OLS(yc,Xu).fit(cov_type='HAC',cov_kwds={'maxlags':bw})
    R=np.zeros((p,Xu.shape[1]))
    for i in range(p): R[i,1+p+i]=1.
    Rb=R@res.params; RVR=R@res.cov_params()@R.T
    try:
        W=float(Rb@np.linalg.inv(RVR)@Rb); return W,float(1-chi2.cdf(W,p))
    except: return np.nan,np.nan

def abw(resid):
    n=len(resid)
    if n<4: return 1
    rho=np.clip(np.dot(resid[:-1],resid[1:])/np.dot(resid[:-1],resid[:-1]),-0.999,0.999)
    a=4*rho**2/(1-rho**2)**2
    return max(1,int(np.floor(1.1447*(a*n)**(1/3))))

def run(ya,xa,rm,dm,lbl,use_bic=True,fl=1):
    ol=bic_lag(ya,xa,rm,dm) if use_bic else fl
    idx=extract_clean(rm,dm,ol); n=len(idx)
    if n<2*ol+10:
        print(f"  {lbl}: insufficient (n={n})"); return {'n_obs':n,'lag':ol,'status':'insufficient'}
    yc=ya[idx]; yl=np.column_stack([ya[idx-i-1] for i in range(ol)])
    xl=np.column_stack([xa[idx-i-1] for i in range(ol)])
    F,fp,dr2=ftest(yc,yl,xl)
    _,hp_lag=hac(yc,yl,xl,ol)
    Xu=np.column_stack([np.ones(n),yl,xl]); bu=np.linalg.lstsq(Xu,yc,rcond=None)[0]
    bw=abw(yc-Xu@bu); _,hp_and=hac(yc,yl,xl,bw)
    def s(p): return '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else '†' if p<0.10 else ''
    print(f"  {lbl} (n={n},lag={ol},Abw={bw}): F-p={fp:.3e}{s(fp)}  HAC(bw={ol})={hp_lag:.3e}{s(hp_lag)}  HAC(And)={hp_and:.3e}{s(hp_and)}  dR2={dr2*100:.3f}%")
    return {'n_obs':n,'lag':ol,'f_p':float(fp),'hac_p_lag':float(hp_lag),'andrews_bw':bw,'hac_p_andrews':float(hp_and),'delta_r2_pct':float(dr2*100)}


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = download_ff_data()

    print(f"\nFitting HMM seed={PRIMARY_SEED}...")
    hmm = StudentTHMM(n_regimes=3, n_iter=100, tol=1e-4, random_state=PRIMARY_SEED)
    hmm.fit(df.values)
    reg = hmm.predict(df.values, use_filtered=False)

    rn = ['Normal','Elevated','Crisis']
    print("Regime counts:")
    for k,n in enumerate(rn): print(f"  {n}: {int((reg==k).sum())} days  mu_norm={np.linalg.norm(hmm.mu[k]):.3f}")

    # Identify which regime index = 4723 days = paper's Normal
    counts = [(int((reg==k).sum()), k) for k in range(3)]
    counts.sort(key=lambda x: -x[0])
    paper_normal_idx = counts[0][1]   # regime with most days = 4723 = paper's Normal
    paper_elev_idx   = counts[1][1]   # regime with 3023 days = paper's Elevated
    paper_crisis_idx = counts[2][1]   # regime with 1071 days

    print(f"\nMapping: paper's Normal=regime{paper_normal_idx} ({counts[0][0]}d), "
          f"Elevated=regime{paper_elev_idx} ({counts[1][0]}d), Crisis=regime{paper_crisis_idx} ({counts[2][0]}d)")

    hml = df['HML'].values; smb = df['SMB'].values
    dates_np = np.array(df.index, dtype='datetime64[D]')

    normal_mask = (reg == paper_normal_idx)
    full_mask   = np.ones(len(dates_np), dtype=bool)
    pre08  = dates_np < np.datetime64(GFC_SPLIT)
    post08 = dates_np >= np.datetime64(GFC_SPLIT)
    pre09  = dates_np < np.datetime64(ALT_SPLIT)
    post09 = dates_np >= np.datetime64(ALT_SPLIT)

    print(f"\nNormal-regime day counts (paper's Normal = regime{paper_normal_idx}):")
    print(f"  Total: {normal_mask.sum()}")
    print(f"  Pre-2008:  {(normal_mask & pre08).sum()}")
    print(f"  Post-2008: {(normal_mask & post08).sum()}")

    print("\n=== NORMAL-REGIME HML→SMB PRE/POST-2008 SPLIT (paper's Normal) ===")
    r_full  = run(smb, hml, normal_mask, full_mask,  "Normal full")
    r_pre   = run(smb, hml, normal_mask, pre08,       "Normal pre-2008")
    r_post  = run(smb, hml, normal_mask, post08,      "Normal post-2008")
    r_pre2  = run(smb, hml, normal_mask, pre09,       "Normal pre-2009")
    r_post2 = run(smb, hml, normal_mask, post09,      "Normal post-2009")

    print("\n=== REVERSE: SMB→HML IN NORMAL REGIME ===")
    r_rev_full = run(hml, smb, normal_mask, full_mask,  "Normal SMB→HML full")
    r_rev_pre  = run(hml, smb, normal_mask, pre08,       "Normal SMB→HML pre-2008")
    r_rev_post = run(hml, smb, normal_mask, post08,      "Normal SMB→HML post-2008")

    print("\n=== FROZEN OOS: ANDREWS HAC FOR ELEVATED REGIME (paper's Elevated) ===")
    # Frozen HMM: train on 1990-2012
    train_mask = dates_np < np.datetime64('2013-01-01')
    oos_mask   = dates_np >= np.datetime64('2013-01-01')
    df_train   = df[train_mask]
    print(f"Fitting frozen HMM on 1990-2012 ({train_mask.sum()} days), seed={PRIMARY_SEED}...")
    hmm_f = StudentTHMM(n_regimes=3, n_iter=100, tol=1e-4, random_state=PRIMARY_SEED)
    hmm_f.fit(df_train.values)
    print(f"  Train LL: {hmm_f.log_likelihood_:.2f}")
    hmm_f._e_step(df.values)
    oos_reg = np.argmax(hmm_f.alpha, axis=1)

    # Identify paper's Elevated in frozen fit (should be 836 days OOS)
    oos_counts = [(int((oos_reg[oos_mask]==k).sum()),k) for k in range(3)]
    oos_counts.sort(key=lambda x: -x[0])
    print("Frozen OOS regime distribution:")
    for cnt,k in oos_counts: print(f"  regime{k}: {cnt} days ({cnt/oos_mask.sum():.1%})")

    # Try each regime as the "Elevated" candidate and find n≈836
    best_k = min(range(3), key=lambda k: abs(int((oos_reg[oos_mask]==k).sum()) - 836))
    elevated_oos = np.zeros(len(dates_np), dtype=bool)
    elevated_oos[oos_mask] = (oos_reg[oos_mask] == best_k)
    print(f"\nUsing regime{best_k} as OOS Elevated (n={elevated_oos.sum()} days, closest to 836)")

    r_oos = run(smb, hml, elevated_oos, oos_mask, "OOS Elevated (Andrews)", use_bic=False, fl=1)

    # Save
    out = {
        'paper_normal_regime_idx': paper_normal_idx,
        'regime_counts': {rn[k]: int((reg==k).sum()) for k in range(3)},
        'hml_to_smb_paper_normal': {
            'full':r_full,'pre_2008':r_pre,'post_2008':r_post,'pre_2009':r_pre2,'post_2009':r_post2
        },
        'smb_to_hml_paper_normal': {'full':r_rev_full,'pre_2008':r_rev_pre,'post_2008':r_rev_post},
        'oos_elevated_andrews': r_oos,
    }
    with open(RESULTS_DIR/'normal_split_corrected.json','w') as f: json.dump(out, f, indent=2)
    print(f"\nSaved → {RESULTS_DIR/'normal_split_corrected.json'}")
    print("Done.")


if __name__ == '__main__':
    main()
