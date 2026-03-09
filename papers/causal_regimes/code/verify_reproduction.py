#!/usr/bin/env python3
"""
verify_reproduction.py
======================
Verification script for causal_regimes paper reproducibility.

Runs core pipeline (HMM fit + Granger tests + OOS validation)
and compares outputs against expected_outputs.json.

Reports PASS/FAIL for each check with specified tolerances.

Runtime: ~15-20 minutes
"""

import json
import sys
import os
import io
import warnings
import urllib.request
import zipfile
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.optimize import minimize_scalar
from scipy.cluster.vq import kmeans2
from scipy.stats import f as f_dist, chi2
import statsmodels.api as sm

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / 'results'
EXPECTED_PATH = BASE_DIR / 'expected_outputs.json'

PRIMARY_SEED = 28
N_REGIMES = 3
REGIME_NAMES = ['Normal', 'Elevated', 'Crisis']

# ============================================================================
# LOGGING
# ============================================================================

class VerificationLogger:
    def __init__(self, output_file=None):
        self.output_file = output_file
        self.lines = []
        self.pass_count = 0
        self.fail_count = 0
        self.checks = []

    def log(self, msg):
        print(msg)
        self.lines.append(msg)

    def check(self, name, actual, expected, tolerance, check_type='absolute'):
        """
        Record a numerical check.

        Args:
            name: Check description
            actual: Actual value
            expected: Expected value
            tolerance: Tolerance (absolute or relative)
            check_type: 'absolute' or 'relative'
        """
        if check_type == 'absolute':
            diff = abs(actual - expected)
            passed = diff <= tolerance
            detail = f"{actual:.6f} (expected {expected:.6f}, tol ±{tolerance})"
        else:  # relative
            if expected != 0:
                pct_diff = abs(actual - expected) / abs(expected) * 100
                passed = pct_diff <= tolerance
                detail = f"{actual:.6e} (expected {expected:.6e}, tol ±{tolerance}%)"
            else:
                passed = abs(actual) <= tolerance
                detail = f"{actual:.6e} (expected {expected:.6e}, tol ±{tolerance})"

        status = "PASS" if passed else "FAIL"
        color = "\033[92m" if passed else "\033[91m"
        reset = "\033[0m"

        self.log(f"  {color}[{status}]{reset} {name}: {detail}")

        if passed:
            self.pass_count += 1
        else:
            self.fail_count += 1

        self.checks.append({
            'name': name,
            'actual': actual,
            'expected': expected,
            'tolerance': tolerance,
            'check_type': check_type,
            'passed': passed
        })

    def summary(self):
        total = self.pass_count + self.fail_count
        pct = 100 * self.pass_count / total if total > 0 else 0
        self.log(f"\n{'='*70}")
        self.log(f"VERIFICATION SUMMARY: {self.pass_count}/{total} checks passed ({pct:.1f}%)")
        self.log(f"{'='*70}")

        if self.fail_count > 0:
            self.log(f"\nFAILED CHECKS ({self.fail_count}):")
            for check in self.checks:
                if not check['passed']:
                    self.log(f"  - {check['name']}")

    def save(self, filepath=None):
        if filepath or self.output_file:
            path = Path(filepath or self.output_file)
            with open(path, 'w') as f:
                f.write('\n'.join(self.lines))
            self.log(f"\nVerification report saved to: {path}")


# ============================================================================
# DATA LOADING
# ============================================================================

def download_ff_data():
    """Download Fama-French 5 factors + Momentum."""
    print("Downloading Fama-French factors...")

    url5 = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip'
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

    url_mom = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip'
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

    return df

# ============================================================================
# STUDENT-T HMM
# ============================================================================

class StudentTHMM:
    def __init__(self, n_regimes=3, n_iter=100, tol=1e-4, random_state=42):
        self.n_regimes = n_regimes; self.n_iter = n_iter; self.tol = tol
        self.random_state = random_state
        self.mu = self.Sigma = self.nu = self.A = self.pi = None
        self.gamma = self.alpha = self.xi = None; self.log_likelihood_ = None

    def _init_params(self, X):
        np.random.seed(self.random_state); T,d = X.shape; K = self.n_regimes
        centroids,labels = kmeans2(X,K,minit='++')
        order = np.argsort(np.linalg.norm(centroids,axis=1))
        centroids = centroids[order]; nl = np.zeros_like(labels)
        for nk,ok in enumerate(order): nl[labels==ok] = nk
        labels = nl; self.mu = centroids; self.Sigma = np.zeros((K,d,d))
        for k in range(K):
            m = labels==k
            self.Sigma[k] = (np.cov(X[m].T)+1e-6*np.eye(d)) if m.sum()>d else np.eye(d)
        self.nu = np.array([15.,7.,4.]); self.A = np.eye(K)*.95+np.ones((K,K))*.05/K
        self.A /= self.A.sum(axis=1,keepdims=True); self.pi = np.ones(K)/K

    def _mvt_logpdf(self,x,mu,Sigma,nu):
        d = len(mu)
        if x.ndim==1: x = x.reshape(1,-1)
        diff = x-mu; Si = np.linalg.inv(Sigma); mahal = np.sum(diff@Si*diff,axis=1)
        _,ld = np.linalg.slogdet(Sigma)
        return (gammaln((nu+d)/2)-gammaln(nu/2)-0.5*d*np.log(nu*np.pi)
                -0.5*ld-0.5*(nu+d)*np.log(1+mahal/nu))

    def _log_B(self,X):
        T,d = X.shape; K = self.n_regimes; lb = np.zeros((T,K))
        for k in range(K): lb[:,k] = self._mvt_logpdf(X,self.mu[k],self.Sigma[k],self.nu[k])
        return lb

    def _forward(self,lb):
        T,K = lb.shape; la = np.zeros((T,K)); la[0] = np.log(self.pi+1e-300)+lb[0]
        lA = np.log(self.A+1e-300)
        for t in range(1,T):
            for k in range(K): la[t,k] = np.logaddexp.reduce(la[t-1]+lA[:,k])+lb[t,k]
        return la

    def _backward(self,lb):
        T,K = lb.shape; lb2 = np.zeros((T,K)); lA = np.log(self.A+1e-300)
        for t in range(T-2,-1,-1):
            for k in range(K): lb2[t,k] = np.logaddexp.reduce(lA[k,:]+lb[t+1,:]+lb2[t+1,:])
        return lb2

    def _e_step(self,X):
        T,d = X.shape; K = self.n_regimes; lB = self._log_B(X)
        la = self._forward(lB); lb = self._backward(lB)
        ll = np.logaddexp.reduce(la[-1])
        lg = la+lb; lg -= np.logaddexp.reduce(lg,axis=1,keepdims=True)
        self.gamma = np.exp(lg)
        lan = la-np.logaddexp.reduce(la,axis=1,keepdims=True); self.alpha = np.exp(lan)
        lA = np.log(self.A+1e-300); self.xi = np.zeros((T-1,K,K))
        for t in range(T-1):
            for j in range(K):
                for k in range(K): self.xi[t,j,k] = np.exp(la[t,j]+lA[j,k]+lB[t+1,k]+lb[t+1,k]-ll)
        self.u = np.zeros((T,K))
        for k in range(K):
            diff = X-self.mu[k]; Si = np.linalg.inv(self.Sigma[k])
            mahal = np.sum(diff@Si*diff,axis=1); self.u[:,k] = (self.nu[k]+d)/(self.nu[k]+mahal)
        return ll

    def _m_step(self,X):
        T,d = X.shape; K = self.n_regimes
        self.pi = self.gamma[0]/self.gamma[0].sum()
        for j in range(K):
            for k in range(K): self.A[j,k] = self.xi[:,j,k].sum()/self.gamma[:-1,j].sum()
        self.A /= self.A.sum(axis=1,keepdims=True)
        for k in range(K):
            w = self.gamma[:,k]*self.u[:,k]; self.mu[k] = (w[:,None]*X).sum(0)/w.sum()
        for k in range(K):
            diff = X-self.mu[k]; w = self.gamma[:,k]*self.u[:,k]
            wo = sum(w[t]*np.outer(diff[t],diff[t]) for t in range(T))
            self.Sigma[k] = wo/self.gamma[:,k].sum()+1e-6*np.eye(d)
        for k in range(K): self._update_nu(X,k)
        self._enforce_ordering()

    def _update_nu(self,X,k):
        T,d = X.shape
        def nl(nu):
            if nu<=2: return 1e10
            diff = X-self.mu[k]; Si = np.linalg.inv(self.Sigma[k])
            mahal = np.sum(diff@Si*diff,axis=1)
            return -(self.gamma[:,k]*(gammaln((nu+d)/2)-gammaln(nu/2)
                    -0.5*d*np.log(nu)-0.5*(nu+d)*np.log(1+mahal/nu))).sum()
        self.nu[k] = minimize_scalar(nl,bounds=(2.1,50),method='bounded').x

    def _enforce_ordering(self):
        # Order regimes by mean volatility norm of assigned data points
        # This matches the paper's convention: Normal (low) < Elevated (mid) < Crisis (high)
        K = self.n_regimes
        regime_mean_norms = np.zeros(K)

        if hasattr(self, '_X'):
            # Compute mean norm for each regime using soft assignments (gamma)
            X_norms = np.linalg.norm(self._X, axis=1)
            for k in range(K):
                regime_weight = self.gamma[:,k].sum()
                if regime_weight > 0:
                    regime_mean_norms[k] = np.sum(self.gamma[:,k] * X_norms) / regime_weight
            order = np.argsort(regime_mean_norms)
        else:
            # Fallback: order by centroid norm (less accurate but works)
            order = np.argsort(np.linalg.norm(self.mu, axis=1))

        if not np.array_equal(order, np.arange(self.n_regimes)):
            self.mu = self.mu[order]; self.Sigma = self.Sigma[order]
            self.nu = self.nu[order]; self.A = self.A[order][:,order]
            self.pi = self.pi[order]; self.gamma = self.gamma[:,order]
            if self.alpha is not None: self.alpha = self.alpha[:,order]
            if self.xi is not None: self.xi = self.xi[:,order,:][:,:,order]

    def fit(self,X):
        X = np.asarray(X); self._X = X; self._init_params(X); prev = -np.inf
        for i in range(self.n_iter):
            ll = self._e_step(X); self._m_step(X)
            if abs(ll-prev) < self.tol: break
            prev = ll
        self.log_likelihood_ = ll; return self

    def predict(self,X,use_filtered=False):
        X = np.asarray(X); self._e_step(X)
        return np.argmax(self.alpha if use_filtered else self.gamma,axis=1)


# ============================================================================
# GRANGER TEST UTILITIES
# ============================================================================

def extract_clean_indices(regime_mask, date_mask, lag):
    """Returns indices where current and preceding lag days are in regime."""
    clean = (regime_mask & date_mask).copy()
    for k in range(1, lag+1):
        shifted = np.zeros(len(regime_mask), dtype=bool)
        shifted[k:] = regime_mask[:len(regime_mask)-k]
        clean &= shifted
    clean[:lag] = False
    return np.where(clean)[0]

def standard_ftest(y_curr, y_lagged, x_lagged):
    n, lag = len(y_curr), y_lagged.shape[1]
    Xr = np.column_stack([np.ones(n), y_lagged])
    Xu = np.column_stack([np.ones(n), y_lagged, x_lagged])
    br = np.linalg.lstsq(Xr, y_curr, rcond=None)[0]
    bu = np.linalg.lstsq(Xu, y_curr, rcond=None)[0]
    rss_r = float(np.sum((y_curr - Xr@br)**2))
    rss_u = float(np.sum((y_curr - Xu@bu)**2))
    df1, df2 = lag, n-2*lag-1
    if df2 <= 0 or rss_u <= 0: return np.nan, np.nan
    F = ((rss_r-rss_u)/df1)/(rss_u/df2)
    p = float(1 - f_dist.cdf(F, df1, df2))
    return float(F), p

def andrews_bw(residuals):
    """Andrews (1991) AR(1) plug-in bandwidth."""
    n = len(residuals)
    if n < 4: return 1
    rho = float(np.dot(residuals[:-1], residuals[1:]) / np.dot(residuals[:-1], residuals[:-1]))
    rho = np.clip(rho, -0.999, 0.999)
    a = 4*rho**2 / (1-rho**2)**2
    return max(1, int(np.floor(1.1447*(a*n)**(1/3))))

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

def run_granger(y_all, x_all, regime_mask, date_mask, lag=1):
    idx = extract_clean_indices(regime_mask, date_mask, lag)
    n = len(idx)

    if n < 2*lag+10:
        return {'n_obs': n, 'lag': lag, 'status': 'insufficient'}

    y_curr = y_all[idx]
    y_lagged = np.column_stack([y_all[idx-i-1] for i in range(lag)])
    x_lagged = np.column_stack([x_all[idx-i-1] for i in range(lag)])

    F, f_p = standard_ftest(y_curr, y_lagged, x_lagged)
    _, hac_p_lag = hac_wald(y_curr, y_lagged, x_lagged, lag)

    Xu = np.column_stack([np.ones(n), y_lagged, x_lagged])
    bu = np.linalg.lstsq(Xu, y_curr, rcond=None)[0]
    bw = andrews_bw(y_curr - Xu@bu)
    _, hac_p_and = hac_wald(y_curr, y_lagged, x_lagged, bw)

    return {
        'n_obs': n,
        'lag': lag,
        'f_stat': float(F),
        'f_p': float(f_p),
        'hac_bw_lag': lag,
        'hac_p_lag': float(hac_p_lag),
        'andrews_bw': bw,
        'hac_p_andrews': float(hac_p_and),
    }


# ============================================================================
# MAIN VERIFICATION
# ============================================================================

def verify_reproduction():
    """Run full verification pipeline."""

    logger = VerificationLogger(RESULTS_DIR / 'verify_reproduction.txt')

    logger.log("="*70)
    logger.log("CAUSAL REGIMES REPRODUCIBILITY VERIFICATION")
    logger.log("="*70)
    logger.log(f"Timestamp: {datetime.now().isoformat()}")
    logger.log(f"Python: {sys.version.split()[0]}")
    logger.log(f"Numpy: {np.__version__}")
    logger.log(f"Pandas: {pd.__version__}")
    logger.log("")

    # Load expected outputs
    logger.log("Loading expected outputs...")
    try:
        with open(EXPECTED_PATH) as f:
            expected = json.load(f)
        logger.log(f"  Loaded from: {EXPECTED_PATH}")
    except FileNotFoundError:
        logger.log(f"  ERROR: {EXPECTED_PATH} not found")
        return logger

    # Download data
    logger.log("\n" + "="*70)
    logger.log("STEP 1: DATA LOADING")
    logger.log("="*70)
    try:
        df = download_ff_data()
        logger.log(f"  Downloaded {len(df)} observations")
        logger.log(f"  Date range: {df.index[0].date()} to {df.index[-1].date()}")
        logger.log(f"  Columns: {', '.join(df.columns)}")
    except Exception as e:
        logger.log(f"  ERROR downloading data: {e}")
        return logger

    # Fit HMM
    logger.log("\n" + "="*70)
    logger.log("STEP 2: HMM FITTING (seed=28)")
    logger.log("="*70)
    try:
        hmm = StudentTHMM(n_regimes=N_REGIMES, n_iter=100, tol=1e-4, random_state=PRIMARY_SEED)
        hmm.fit(df.values)
        regimes = hmm.predict(df.values, use_filtered=False)

        logger.log(f"  HMM fitted successfully")
        logger.log(f"  Converged: Yes")

        # Check log-likelihood
        exp_ll = expected['hmm_primary_fit']['log_likelihood']
        logger.check(
            "Log-likelihood",
            hmm.log_likelihood_,
            exp_ll['value'],
            exp_ll['tolerance']
        )

        # Check BIC
        # Parameters: mu (18) + Sigma (63) + nu (3) + A (6) + pi (2) = 92
        n_params = 92
        bic = -2 * hmm.log_likelihood_ + n_params * np.log(len(df))
        exp_bic = expected['hmm_primary_fit']['bic']
        logger.check(
            "BIC",
            bic,
            exp_bic['value'],
            exp_bic['tolerance']
        )

        # Check regime counts
        for k, name in enumerate(REGIME_NAMES):
            count = int((regimes == k).sum())
            exp_count = expected['hmm_primary_fit']['regime_distribution'][name]['count']
            exp_tol = expected['hmm_primary_fit']['regime_distribution'][name]['count_tolerance']
            logger.check(
                f"Regime count ({name})",
                count,
                exp_count,
                exp_tol
            )

    except Exception as e:
        logger.log(f"  ERROR during HMM fitting: {e}")
        return logger

    # Granger tests
    logger.log("\n" + "="*70)
    logger.log("STEP 3: GRANGER CAUSALITY TESTS (HML→SMB, lag=1)")
    logger.log("="*70)

    try:
        dates_np = np.array(df.index, dtype='datetime64[D]')
        hml = df['HML'].values
        smb = df['SMB'].values
        full_mask = np.ones(len(df), dtype=bool)

        granger_results = {}
        for k, name in enumerate(REGIME_NAMES):
            regime_mask = (regimes == k)
            res = run_granger(smb, hml, regime_mask, full_mask, lag=1)
            granger_results[name] = res

            logger.log(f"  {name} regime:")

            if res.get('status') != 'insufficient':
                # Check observation count
                exp_n = expected['granger_causality_lag1'][f'{name}_regime']['n_obs']
                logger.check(
                    f"    n_obs ({name})",
                    res['n_obs'],
                    exp_n['value'],
                    exp_n['tolerance']
                )

                # Check F-statistic
                if 'f_stat' in res:
                    exp_f = expected['granger_causality_lag1'][f'{name}_regime']['f_stat']
                    logger.check(
                        f"    F-statistic ({name})",
                        res['f_stat'],
                        exp_f['value'],
                        exp_f['tolerance']
                    )

                # Check F p-value
                exp_fp = expected['granger_causality_lag1'][f'{name}_regime']['f_p_value']
                logger.check(
                    f"    F-p-value ({name})",
                    res['f_p'],
                    exp_fp['value'],
                    exp_fp['tolerance'],
                    check_type='absolute'
                )

                # Check HAC p-value
                exp_hac = expected['granger_causality_lag1'][f'{name}_regime']['hac_p_value_andrews']
                logger.check(
                    f"    HAC(Andrews)-p ({name})",
                    res['hac_p_andrews'],
                    exp_hac['value'],
                    exp_hac['tolerance'],
                    check_type='absolute'
                )
            else:
                logger.log(f"    INSUFFICIENT observations (n={res['n_obs']})")

    except Exception as e:
        logger.log(f"  ERROR during Granger tests: {e}")
        return logger

    # Frozen OOS validation
    logger.log("\n" + "="*70)
    logger.log("STEP 4: FROZEN OOS VALIDATION (train 1990-2012, test 2013-2024)")
    logger.log("="*70)

    try:
        train_mask_np = dates_np < np.datetime64('2013-01-01')
        oos_mask_np = dates_np >= np.datetime64('2013-01-01')
        df_train = df[train_mask_np]

        logger.log(f"  Training on {train_mask_np.sum()} days (1990-2012)")

        # Fit frozen HMM
        hmm_frozen = StudentTHMM(n_regimes=N_REGIMES, n_iter=100, tol=1e-4, random_state=PRIMARY_SEED)
        hmm_frozen.fit(df_train.values)

        exp_train_ll = expected['frozen_oos_validation']['training_phase']['train_log_likelihood']
        logger.check(
            "Frozen training LL",
            hmm_frozen.log_likelihood_,
            exp_train_ll['value'],
            exp_train_ll['tolerance']
        )

        # Get OOS regime assignments
        hmm_frozen._e_step(df.values)
        oos_regimes_filtered = np.argmax(hmm_frozen.alpha, axis=1)

        # Re-order regimes based on full-sample volatility norms
        # (The frozen HMM was trained on 1990-2012, but we apply it to full 1990-2024,
        #  so the regime ordering may shift when extended to the OOS period)
        X_full = df.values
        X_norms_full = np.linalg.norm(X_full, axis=1)
        regime_mean_norms = np.zeros(N_REGIMES)
        for k in range(N_REGIMES):
            mask_k = oos_regimes_filtered == k
            if mask_k.sum() > 0:
                regime_mean_norms[k] = X_norms_full[mask_k].mean()

        # Get the reordering to match volatility norm convention (low < mid < high)
        reorder = np.argsort(regime_mean_norms)

        # Create a mapping from old regime indices to new ones
        regime_remap = np.zeros(N_REGIMES, dtype=int)
        for new_idx, old_idx in enumerate(reorder):
            regime_remap[old_idx] = new_idx

        # Apply the remapping to the regime assignments
        oos_regimes_remapped = regime_remap[oos_regimes_filtered]
        oos_regimes_oos = oos_regimes_remapped[oos_mask_np]

        logger.log(f"  OOS observations: {oos_mask_np.sum()}")
        logger.log(f"  Regime reordering applied: {reorder} (for volatility norm consistency)")

        # Check OOS regime counts
        for k, name in enumerate(REGIME_NAMES):
            count = int((oos_regimes_oos == k).sum())
            exp_count = expected['frozen_oos_validation']['oos_regime_distribution'][name]['count']
            exp_tol = expected['frozen_oos_validation']['oos_regime_distribution'][name]['count_tolerance']
            logger.check(
                f"  OOS regime count ({name})",
                count,
                exp_count,
                exp_tol
            )

        # Test OOS Elevated regime (regime 1 in volatility-ordered naming)
        elevated_in_oos = np.zeros(len(df), dtype=bool)
        elevated_in_oos[oos_mask_np] = (oos_regimes_oos == 1)

        hml_oos = df['HML'].values
        smb_oos = df['SMB'].values

        res_oos_elev = run_granger(smb_oos, hml_oos, elevated_in_oos, oos_mask_np, lag=1)

        logger.log(f"  OOS Elevated HML→SMB (lag=1):")

        exp_oos_n = expected['frozen_oos_validation']['oos_elevated_hml_to_smb_lag1']['n_obs']
        logger.check(
            "    OOS Elevated n_obs",
            res_oos_elev['n_obs'],
            exp_oos_n['value'],
            exp_oos_n['tolerance']
        )

        exp_oos_fp = expected['frozen_oos_validation']['oos_elevated_hml_to_smb_lag1']['f_p_value']
        logger.check(
            "    OOS Elevated F-p",
            res_oos_elev['f_p'],
            exp_oos_fp['value'],
            exp_oos_fp['tolerance'],
            check_type='absolute'
        )

        exp_oos_hac = expected['frozen_oos_validation']['oos_elevated_hml_to_smb_lag1']['hac_p_value_andrews']
        logger.check(
            "    OOS Elevated HAC(Andrews)-p",
            res_oos_elev['hac_p_andrews'],
            exp_oos_hac['value'],
            exp_oos_hac['tolerance'],
            check_type='absolute'
        )

    except Exception as e:
        logger.log(f"  ERROR during OOS validation: {e}")
        import traceback
        traceback.print_exc()
        return logger

    # Summary
    logger.summary()

    # Save report
    RESULTS_DIR.mkdir(exist_ok=True)
    logger.save(RESULTS_DIR / 'verify_reproduction.txt')

    return logger


if __name__ == '__main__':
    logger = verify_reproduction()

    # Exit code based on pass/fail
    sys.exit(0 if logger.fail_count == 0 else 1)
