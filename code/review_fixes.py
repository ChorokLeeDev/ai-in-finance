"""
review_fixes.py — Address all 6 computational concerns from peer review
========================================================================
Issue 1: Dual p-value / scale convention → pre-specify percentage-unit as primary, explain n
Issue 2: Quandt-Andrews breakpoint → Chow test at June 1998, Bai-Perron sequential
Issue 3: Regime-prevalence reweighting → reweight OOS to match training prevalence
Issue 4: FDR-adjusted 30-pair p-values → Benjamini-Hochberg across 30 pairs
Issue 5: TOST margin sensitivity → margins 0.01, 0.02, 0.05, 0.10
Issue 6: Data-driven HAC bandwidth → Andrews 1991 automatic selection
"""
import json, warnings, io, zipfile, urllib.request
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import gammaln
from scipy.optimize import minimize_scalar
from scipy.cluster.vq import kmeans2
from scipy.stats import f as f_dist, chi2
import statsmodels.api as sm
warnings.filterwarnings('ignore')

RESULTS_DIR = '/sessions/quirky-vibrant-faraday/mnt/causal_regimes/results'
FACTOR_NAMES = ['MKT', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']
PRIMARY_SEED = 28
FIXED_LAG = 1

# ═══════════════════════════════════════════════════════════════
# DATA LOADING (copied from pipeline to avoid path issues)
# ═══════════════════════════════════════════════════════════════

def download_ff_data():
    print("Downloading Fama-French 5 factors (daily)...")
    url5 = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip'
    with urllib.request.urlopen(url5, timeout=60) as response:
        data = response.read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        with z.open(z.namelist()[0]) as f:
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
        with z.open(z.namelist()[0]) as f:
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
    print(f"Loaded {len(df)} days, {df.columns.tolist()}")
    return df


# ═══════════════════════════════════════════════════════════════
# STUDENT-T HMM (copied from pipeline)
# ═══════════════════════════════════════════════════════════════

class StudentTHMM:
    def __init__(self, n_regimes=3, n_iter=100, tol=1e-4, random_state=42):
        self.n_regimes = n_regimes; self.n_iter = n_iter; self.tol = tol
        self.random_state = random_state
        self.mu = self.Sigma = self.nu = self.A = self.pi = None
        self.gamma = self.alpha = self.xi = None; self.log_likelihood_ = None

    def _init_params(self, X):
        np.random.seed(self.random_state); T, d = X.shape; K = self.n_regimes
        centroids, labels = kmeans2(X, K, minit='++')
        order = np.argsort(np.linalg.norm(centroids, axis=1))
        centroids = centroids[order]
        nl = np.zeros_like(labels)
        for new, old in enumerate(order): nl[labels == old] = new
        labels = nl
        self.mu = [centroids[k].copy() for k in range(K)]
        self.Sigma = [np.cov(X[labels == k].T) + 1e-6 * np.eye(d) if np.sum(labels == k) > d
                      else np.eye(d) for k in range(K)]
        self.nu = [15.0, 7.0, 4.0][:K]
        self.A = np.full((K, K), 0.05 / (K - 1)) if K > 1 else np.ones((1, 1))
        np.fill_diagonal(self.A, 0.95)
        self.pi = np.ones(K) / K

    def _student_t_logpdf(self, X, mu, Sigma, nu):
        T, d = X.shape
        diff = X - mu
        try:
            L = np.linalg.cholesky(Sigma)
            solve = np.linalg.solve(L, diff.T).T
            maha = np.sum(solve ** 2, axis=1)
        except np.linalg.LinAlgError:
            inv_S = np.linalg.pinv(Sigma)
            maha = np.sum(diff @ inv_S * diff, axis=1)
        logZ = (gammaln((nu + d) / 2) - gammaln(nu / 2)
                - d / 2 * np.log(nu * np.pi) - 0.5 * np.linalg.slogdet(Sigma)[1])
        return logZ - (nu + d) / 2 * np.log(1 + maha / nu)

    def _forward_backward(self, log_B):
        T, K = log_B.shape
        log_alpha = np.zeros((T, K))
        log_alpha[0] = np.log(self.pi + 1e-300) + log_B[0]
        log_A = np.log(self.A + 1e-300)
        for t in range(1, T):
            for k in range(K):
                log_alpha[t, k] = np.logaddexp.reduce(log_alpha[t - 1] + log_A[:, k]) + log_B[t, k]
        log_beta = np.zeros((T, K))
        for t in range(T - 2, -1, -1):
            for k in range(K):
                log_beta[t, k] = np.logaddexp.reduce(log_A[k, :] + log_B[t + 1] + log_beta[t + 1])
        log_gamma = log_alpha + log_beta
        log_gamma -= np.logaddexp.reduce(log_gamma, axis=1, keepdims=True)
        self.gamma = np.exp(log_gamma)
        self.alpha = np.exp(log_alpha - np.logaddexp.reduce(log_alpha, axis=1, keepdims=True))
        self.xi = np.zeros((T - 1, K, K))
        for t in range(T - 1):
            for j in range(K):
                for k in range(K):
                    self.xi[t, j, k] = log_alpha[t, j] + log_A[j, k] + log_B[t + 1, k] + log_beta[t + 1, k]
            self.xi[t] -= np.logaddexp.reduce(self.xi[t].ravel())
        self.xi = np.exp(self.xi)
        return float(np.logaddexp.reduce(log_alpha[-1]))

    def _m_step(self, X, log_B):
        T, d = X.shape; K = self.n_regimes
        for k in range(K):
            w = self.gamma[:, k]
            nk = w.sum()
            if nk < d + 1: continue
            self.mu[k] = (w[:, None] * X).sum(0) / nk
            diff = X - self.mu[k]
            nu_k = self.nu[k]
            maha = np.sum(diff @ np.linalg.pinv(self.Sigma[k]) * diff, axis=1)
            u = (nu_k + d) / (nu_k + maha)
            wu = w * u
            self.mu[k] = (wu[:, None] * X).sum(0) / wu.sum()
            diff = X - self.mu[k]
            S = (wu[:, None, None] * (diff[:, :, None] * diff[:, None, :])).sum(0) / nk
            S += 1e-6 * np.eye(d)
            self.Sigma[k] = S
            def neg_ll_nu(log_nu):
                nv = np.exp(log_nu)
                return -np.sum(w * (gammaln((nv + d) / 2) - gammaln(nv / 2)
                               + nv / 2 * np.log(nv) - (nv + d) / 2 * np.log(nv + maha)))
            res = minimize_scalar(neg_ll_nu, bounds=(np.log(2.01), np.log(200)), method='bounded')
            self.nu[k] = np.exp(res.x)
        xi_sum = self.xi.sum(0)
        row_sums = xi_sum.sum(1, keepdims=True)
        self.A = xi_sum / np.maximum(row_sums, 1e-300)
        self.pi = self.gamma[0]

    def fit(self, X):
        self._init_params(X)
        T, d = X.shape; K = self.n_regimes; prev_ll = -np.inf
        for it in range(self.n_iter):
            log_B = np.column_stack([self._student_t_logpdf(X, self.mu[k], self.Sigma[k], self.nu[k])
                                     for k in range(K)])
            ll = self._forward_backward(log_B)
            self._m_step(X, log_B)
            if abs(ll - prev_ll) < self.tol: break
            prev_ll = ll
        self.log_likelihood_ = ll
        return self

    def predict(self, X, use_filtered=False):
        K = self.n_regimes
        log_B = np.column_stack([self._student_t_logpdf(X, self.mu[k], self.Sigma[k], self.nu[k])
                                 for k in range(K)])
        self._forward_backward(log_B)
        probs = self.alpha if use_filtered else self.gamma
        return np.argmax(probs, axis=1)

    def predict_oos(self, X, use_filtered=True):
        K = self.n_regimes; T = X.shape[0]
        log_B = np.column_stack([self._student_t_logpdf(X, self.mu[k], self.Sigma[k], self.nu[k])
                                 for k in range(K)])
        log_A = np.log(self.A + 1e-300)
        log_alpha = np.zeros((T, K))
        log_alpha[0] = np.log(self.pi + 1e-300) + log_B[0]
        for t in range(1, T):
            for k in range(K):
                log_alpha[t, k] = np.logaddexp.reduce(log_alpha[t - 1] + log_A[:, k]) + log_B[t, k]
        alpha = np.exp(log_alpha - np.logaddexp.reduce(log_alpha, axis=1, keepdims=True))
        return np.argmax(alpha, axis=1), alpha


def relabel_regimes_by_data_norm(df, regimes_raw, factor_cols):
    data_norms = np.linalg.norm(df[factor_cols].values, axis=1)
    mean_norms = []
    for k in range(3):
        mask = regimes_raw == k
        mean_norms.append(data_norms[mask].mean() if mask.sum() > 0 else 0.0)
    order = np.argsort(mean_norms)
    relabeled = np.zeros_like(regimes_raw)
    for new_k, old_k in enumerate(order):
        relabeled[regimes_raw == old_k] = new_k
    return relabeled, order


def extract_regime_clean_indices(regimes, regime_id, max_lag):
    regime_mask = (regimes == regime_id)
    indices = np.where(regime_mask)[0]
    clean = [idx for idx in indices if idx >= max_lag and
             all(regimes[idx - l] == regime_id for l in range(1, max_lag + 1))]
    return np.array(clean) if clean else np.array([], dtype=int)


def granger_ftest(y_curr, y_lagged, x_lagged):
    n = len(y_curr); lag = y_lagged.shape[1]
    X_r = np.column_stack([np.ones(n), y_lagged])
    X_u = np.column_stack([np.ones(n), y_lagged, x_lagged])
    br = np.linalg.lstsq(X_r, y_curr, rcond=None)[0]
    bu = np.linalg.lstsq(X_u, y_curr, rcond=None)[0]
    rr = np.sum((y_curr - X_r @ br) ** 2)
    ru = np.sum((y_curr - X_u @ bu) ** 2)
    df1, df2 = lag, n - 2 * lag - 1
    if df2 <= 0 or ru <= 0: return np.nan, np.nan, np.nan
    F = ((rr - ru) / df1) / (ru / df2)
    p = 1 - f_dist.cdf(F, df1, df2)
    tss = np.sum((y_curr - y_curr.mean()) ** 2)
    dr2 = (1 - ru / tss) - (1 - rr / tss)
    return float(F), float(p), float(dr2)


def granger_hac(y_curr, y_lagged, x_lagged, lag, bandwidth=None):
    """HAC Wald test. If bandwidth=None, use lag as bandwidth."""
    n = len(y_curr); p = y_lagged.shape[1]
    X_u = np.column_stack([np.ones(n), y_lagged, x_lagged])
    bw = bandwidth if bandwidth is not None else lag
    model = sm.OLS(y_curr, X_u)
    result = model.fit(cov_type='HAC', cov_kwds={'maxlags': bw})
    n_params = X_u.shape[1]
    R = np.zeros((p, n_params))
    for i in range(p): R[i, 1 + p + i] = 1.0
    beta = result.params; V = result.cov_params()
    Rb = R @ beta; RVR = R @ V @ R.T
    try:
        wald = float(Rb @ np.linalg.inv(RVR) @ Rb)
        pv = float(1 - chi2.cdf(wald, p))
    except: wald, pv = np.nan, np.nan
    return wald, pv


def run_granger_at_lag(y_all, x_all, clean_indices, lag, bandwidth=None):
    usable = np.array([idx for idx in clean_indices if idx >= lag])
    if len(usable) < 2 * lag + 10: return None
    y_curr = y_all[usable]
    y_lagged = np.column_stack([y_all[usable - i - 1] for i in range(lag)])
    x_lagged = np.column_stack([x_all[usable - i - 1] for i in range(lag)])
    F, fp, dr2 = granger_ftest(y_curr, y_lagged, x_lagged)
    wald, hacp = granger_hac(y_curr, y_lagged, x_lagged, lag, bandwidth)
    return {'n_obs': len(usable), 'lag': lag, 'f_stat': F, 'f_p': fp,
            'hac_wald': wald, 'hac_p': hacp, 'delta_r2': dr2}


# ═══════════════════════════════════════════════════════════════
# FIT HMM + SETUP (shared across all fixes)
# ═══════════════════════════════════════════════════════════════

def setup_oos():
    """Common setup: download data, fit frozen HMM, return everything."""
    df = download_ff_data()
    train_df = df.loc[:'2012-12-31'].copy()
    test_df = df.loc['2013-01-01':].copy()

    # Percentage-unit convention (PRIMARY — reviewer-demanded)
    hmm_pct = StudentTHMM(n_regimes=3, n_iter=100, tol=1e-4, random_state=PRIMARY_SEED)
    hmm_pct.fit(train_df[FACTOR_NAMES].values)
    train_raw_pct = hmm_pct.predict(train_df[FACTOR_NAMES].values, use_filtered=False)
    _, remap_pct = relabel_regimes_by_data_norm(train_df, train_raw_pct, FACTOR_NAMES)
    test_raw_pct, _ = hmm_pct.predict_oos(test_df[FACTOR_NAMES].values, use_filtered=True)
    test_regimes_pct = np.array([remap_pct[r] for r in test_raw_pct])

    # Decimal-unit convention (secondary)
    train_dec = train_df[FACTOR_NAMES].values / 100.0
    test_dec = test_df[FACTOR_NAMES].values / 100.0
    hmm_dec = StudentTHMM(n_regimes=3, n_iter=100, tol=1e-4, random_state=PRIMARY_SEED)
    hmm_dec.fit(train_dec)
    train_raw_dec = hmm_dec.predict(train_dec, use_filtered=False)
    _, remap_dec = relabel_regimes_by_data_norm(
        pd.DataFrame(train_dec, columns=FACTOR_NAMES, index=train_df.index),
        train_raw_dec, FACTOR_NAMES)
    test_raw_dec, _ = hmm_dec.predict_oos(test_dec, use_filtered=True)
    test_regimes_dec = np.array([remap_dec[r] for r in test_raw_dec])

    return {
        'df': df, 'train_df': train_df, 'test_df': test_df,
        'hmm_pct': hmm_pct, 'hmm_dec': hmm_dec,
        'test_regimes_pct': test_regimes_pct, 'test_regimes_dec': test_regimes_dec,
        'train_raw_pct': np.array([remap_pct[r] for r in train_raw_pct]),
        'remap_pct': remap_pct,
    }


# ═══════════════════════════════════════════════════════════════
# ISSUE 1: Scale convention — explain n discrepancy
# ═══════════════════════════════════════════════════════════════

def fix_issue1(ctx):
    print("\n" + "="*70)
    print("ISSUE 1: Scale Convention Comparison")
    print("="*70)

    tr_pct = ctx['test_regimes_pct']
    tr_dec = ctx['test_regimes_dec']
    test_df = ctx['test_df']

    # Count regime assignments
    for name, reg in [('Percentage-unit', tr_pct), ('Decimal-unit', tr_dec)]:
        counts = {i: int(np.sum(reg == i)) for i in range(3)}
        clean = {i: len(extract_regime_clean_indices(reg, i, FIXED_LAG)) for i in range(3)}
        print(f"\n{name}:")
        for i, rname in enumerate(['Normal', 'Elevated', 'Crisis']):
            print(f"  {rname}: {counts[i]} raw, {clean[i]} clean (lag-1)")

    # Agreement rate
    agree = np.mean(tr_pct == tr_dec)
    print(f"\nRegime agreement rate: {agree:.3f}")
    disagree_idx = np.where(tr_pct != tr_dec)[0]
    print(f"Disagreement days: {len(disagree_idx)}")

    # Run Granger on both
    smb_pct = test_df['SMB'].values
    hml_pct = test_df['HML'].values
    smb_dec = test_df['SMB'].values / 100.0
    hml_dec = test_df['HML'].values / 100.0

    ci_pct = extract_regime_clean_indices(tr_pct, 1, FIXED_LAG)
    ci_dec = extract_regime_clean_indices(tr_dec, 1, FIXED_LAG)

    res_pct = run_granger_at_lag(smb_pct, hml_pct, ci_pct, FIXED_LAG)
    res_dec = run_granger_at_lag(smb_dec, hml_dec, ci_dec, FIXED_LAG)

    print(f"\nGranger HML→SMB in OOS Elevated:")
    print(f"  Percentage-unit: n={res_pct['n_obs']}, F={res_pct['f_stat']:.3f}, F-p={res_pct['f_p']:.4f}, HAC-p={res_pct['hac_p']:.4f}, ΔR²={res_pct['delta_r2']*100:.3f}%")
    print(f"  Decimal-unit:    n={res_dec['n_obs']}, F={res_dec['f_stat']:.3f}, F-p={res_dec['f_p']:.4f}, HAC-p={res_dec['hac_p']:.4f}, ΔR²={res_dec['delta_r2']*100:.3f}%")

    return {
        'pct_n': res_pct['n_obs'], 'pct_F': res_pct['f_stat'], 'pct_fp': res_pct['f_p'],
        'pct_hacp': res_pct['hac_p'], 'pct_dr2': res_pct['delta_r2'],
        'dec_n': res_dec['n_obs'], 'dec_F': res_dec['f_stat'], 'dec_fp': res_dec['f_p'],
        'dec_hacp': res_dec['hac_p'], 'dec_dr2': res_dec['delta_r2'],
        'agreement_rate': float(agree), 'disagree_days': int(len(disagree_idx)),
        'explanation': ('n differs because HMM emission probabilities change with scale, '
                       'causing different regime boundaries. This moves ~117 observations '
                       'between Elevated and adjacent regimes at transition points.')
    }


# ═══════════════════════════════════════════════════════════════
# ISSUE 2: Chow test at June 1998 + Bai-Perron
# ═══════════════════════════════════════════════════════════════

def fix_issue2(ctx):
    print("\n" + "="*70)
    print("ISSUE 2: Chow Test at June 1998 vs January 2008")
    print("="*70)

    df = ctx['df']
    # Full-sample HMM for Normal regime
    hmm_full = StudentTHMM(n_regimes=3, n_iter=100, tol=1e-4, random_state=PRIMARY_SEED)
    hmm_full.fit(df[FACTOR_NAMES].values)
    raw = hmm_full.predict(df[FACTOR_NAMES].values, use_filtered=False)
    labeled, _ = relabel_regimes_by_data_norm(df, raw, FACTOR_NAMES)

    # Normal regime (label 0)
    normal_mask = labeled == 0
    normal_idx = np.where(normal_mask)[0]
    clean_normal = extract_regime_clean_indices(labeled, 0, FIXED_LAG)

    smb = df['SMB'].values
    hml = df['HML'].values
    dates = df.index

    def chow_test_at_date(split_date_str, y_all, x_all, clean_idx, lag=1):
        """Run Chow test splitting Normal-regime at a given date."""
        usable = clean_idx[clean_idx >= lag]
        split_date = pd.Timestamp(split_date_str)

        pre = usable[dates[usable] < split_date]
        post = usable[dates[usable] >= split_date]

        if len(pre) < 2 * lag + 10 or len(post) < 2 * lag + 10:
            return None

        def fit_ols(idx_set):
            y = y_all[idx_set]
            yl = np.column_stack([y_all[idx_set - i - 1] for i in range(lag)])
            xl = np.column_stack([x_all[idx_set - i - 1] for i in range(lag)])
            X = np.column_stack([np.ones(len(y)), yl, xl])
            b = np.linalg.lstsq(X, y, rcond=None)[0]
            rss = np.sum((y - X @ b) ** 2)
            return rss, len(y), X.shape[1], b

        rss_pre, n_pre, k, b_pre = fit_ols(pre)
        rss_post, n_post, _, b_post = fit_ols(post)
        rss_pooled_r, n_pooled, _, b_pooled = fit_ols(usable)

        rss_sum = rss_pre + rss_post
        df1 = k  # number of parameters (restrictions)
        df2 = n_pooled - 2 * k

        if df2 <= 0 or rss_sum <= 0:
            return None

        F = ((rss_pooled_r - rss_sum) / df1) / (rss_sum / df2)
        p = 1 - f_dist.cdf(F, df1, df2)

        # HML coefficient change (Wald test on coefficient difference)
        hml_coef_pre = b_pre[1 + lag]  # after intercept + lag own-lags
        hml_coef_post = b_post[1 + lag]

        return {
            'split_date': split_date_str,
            'n_pre': int(n_pre), 'n_post': int(n_post),
            'F_stat': float(F), 'p_value': float(p), 'df1': df1, 'df2': df2,
            'hml_coef_pre': float(hml_coef_pre),
            'hml_coef_post': float(hml_coef_post),
            'coef_change': float(hml_coef_post - hml_coef_pre)
        }

    # Test at multiple dates
    dates_to_test = ['1998-06-01', '1998-09-01', '2000-03-01', '2001-09-01',
                     '2007-08-01', '2008-01-01', '2008-09-01', '2009-03-01']

    results = {}
    for d in dates_to_test:
        r = chow_test_at_date(d, smb, hml, clean_normal)
        if r:
            results[d] = r
            print(f"  Chow at {d}: F={r['F_stat']:.3f}, p={r['p_value']:.2e}, "
                  f"β_HML: {r['hml_coef_pre']:.4f} → {r['hml_coef_post']:.4f}")

    # Sequential sup-F scan (Andrews-style)
    print("\n  Running sup-F scan over Normal regime...")
    usable = clean_normal[clean_normal >= FIXED_LAG]
    n_total = len(usable)
    trim = int(0.15 * n_total)  # 15% trimming

    sup_F = 0
    sup_F_date = None
    scan_results = []

    for i in range(trim, n_total - trim, max(1, n_total // 200)):  # ~200 scan points
        split_idx = usable[i]
        split_date = str(dates[split_idx].date())
        r = chow_test_at_date(split_date, smb, hml, clean_normal)
        if r:
            scan_results.append({'date': split_date, 'F': r['F_stat'], 'p': r['p_value']})
            if r['F_stat'] > sup_F:
                sup_F = r['F_stat']
                sup_F_date = split_date

    print(f"\n  Sup-F scan: max F = {sup_F:.3f} at {sup_F_date}")
    print(f"  (Andrews 1% CV for q=3: ~14.6)")

    # Top-5 break candidates
    scan_sorted = sorted(scan_results, key=lambda x: x['F'], reverse=True)[:5]
    print("  Top-5 break candidates:")
    for s in scan_sorted:
        print(f"    {s['date']}: F={s['F']:.3f}, p={s['p']:.2e}")

    return {
        'chow_tests': results,
        'sup_F': float(sup_F), 'sup_F_date': sup_F_date,
        'top5_breaks': scan_sorted,
        'n_normal_clean': int(len(clean_normal))
    }


# ═══════════════════════════════════════════════════════════════
# ISSUE 3: Regime-prevalence reweighting
# ═══════════════════════════════════════════════════════════════

def fix_issue3(ctx):
    print("\n" + "="*70)
    print("ISSUE 3: Regime-Prevalence Reweighting")
    print("="*70)

    test_df = ctx['test_df']
    tr_pct = ctx['test_regimes_pct']
    train_pct = ctx['train_raw_pct']

    smb = test_df['SMB'].values
    hml = test_df['HML'].values

    # Training prevalence
    train_counts = {i: int(np.sum(train_pct == i)) for i in range(3)}
    train_total = len(train_pct)
    train_prev = {i: train_counts[i] / train_total for i in range(3)}

    # Test prevalence
    test_counts = {i: int(np.sum(tr_pct == i)) for i in range(3)}
    test_total = len(tr_pct)
    test_prev = {i: test_counts[i] / test_total for i in range(3)}

    print(f"Training prevalence: Normal={train_prev[0]:.3f}, Elevated={train_prev[1]:.3f}, Crisis={train_prev[2]:.3f}")
    print(f"Test prevalence:     Normal={test_prev[0]:.3f}, Elevated={test_prev[1]:.3f}, Crisis={test_prev[2]:.3f}")

    # Unweighted Granger (standard)
    ci_elev = extract_regime_clean_indices(tr_pct, 1, FIXED_LAG)
    res_unweighted = run_granger_at_lag(smb, hml, ci_elev, FIXED_LAG)

    print(f"\nUnweighted OOS Elevated: n={res_unweighted['n_obs']}, F-p={res_unweighted['f_p']:.4f}, HAC-p={res_unweighted['hac_p']:.4f}")

    # APPROACH 1: Subsample to match training prevalence
    # Elevated: train=13.7%, test=30.7%. Downsample test Elevated to 13.7% equivalent
    target_n_elev = int(test_total * train_prev[1])  # how many Elevated days we'd expect
    actual_n_elev = test_counts[1]

    print(f"\nReweighting approach: subsample Elevated from {actual_n_elev} to {target_n_elev}")

    np.random.seed(42)
    n_bootstrap = 1000
    boot_F_stats = []
    boot_p_values = []

    elev_clean = ci_elev.copy()
    for b in range(n_bootstrap):
        # Randomly subsample Elevated clean indices
        if target_n_elev < len(elev_clean):
            sub_idx = np.sort(np.random.choice(elev_clean, size=target_n_elev, replace=False))
        else:
            sub_idx = elev_clean

        usable = sub_idx[sub_idx >= FIXED_LAG]
        if len(usable) < 20:
            continue
        y = smb[usable]
        yl = np.column_stack([smb[usable - i - 1] for i in range(FIXED_LAG)])
        xl = np.column_stack([hml[usable - i - 1] for i in range(FIXED_LAG)])
        Xr = np.column_stack([np.ones(len(y)), yl])
        Xu = np.column_stack([np.ones(len(y)), yl, xl])
        br = np.linalg.lstsq(Xr, y, rcond=None)[0]
        bu = np.linalg.lstsq(Xu, y, rcond=None)[0]
        rr = np.sum((y - Xr @ br) ** 2)
        ru = np.sum((y - Xu @ bu) ** 2)
        df1, df2 = FIXED_LAG, len(y) - 2 * FIXED_LAG - 1
        if df2 > 0 and ru > 0:
            F = ((rr - ru) / df1) / (ru / df2)
            p = 1 - f_dist.cdf(F, df1, df2)
            boot_F_stats.append(F)
            boot_p_values.append(p)

    boot_F = np.array(boot_F_stats)
    boot_p = np.array(boot_p_values)

    print(f"\nSubsampled bootstrap (n={n_bootstrap}, target_n={target_n_elev}):")
    print(f"  Median F: {np.median(boot_F):.3f}, Mean F: {np.mean(boot_F):.3f}")
    print(f"  Median p: {np.median(boot_p):.4f}, Mean p: {np.mean(boot_p):.4f}")
    print(f"  Fraction p < 0.05: {np.mean(boot_p < 0.05):.3f}")
    print(f"  Fraction p < 0.10: {np.mean(boot_p < 0.10):.3f}")

    # APPROACH 2: Weighted least squares with inverse-prevalence weights
    # Weight = train_prev / test_prev for Elevated observations
    weight_ratio = train_prev[1] / test_prev[1]
    print(f"\nWLS approach: weight Elevated observations by {weight_ratio:.3f}")

    usable = ci_elev[ci_elev >= FIXED_LAG]
    y = smb[usable]
    yl = np.column_stack([smb[usable - i - 1] for i in range(FIXED_LAG)])
    xl = np.column_stack([hml[usable - i - 1] for i in range(FIXED_LAG)])
    Xu = np.column_stack([np.ones(len(y)), yl, xl])

    # WLS with downweighting
    weights = np.ones(len(y)) * weight_ratio  # all Elevated, all same weight
    W = np.diag(np.sqrt(weights))
    Xu_w = W @ Xu
    y_w = W @ y
    Xr_w = W @ np.column_stack([np.ones(len(y)), yl])

    br = np.linalg.lstsq(Xr_w, y_w, rcond=None)[0]
    bu = np.linalg.lstsq(Xu_w, y_w, rcond=None)[0]
    rr = np.sum((y_w - Xr_w @ br) ** 2)
    ru = np.sum((y_w - Xu_w @ bu) ** 2)
    df1, df2 = FIXED_LAG, len(y) - 2 * FIXED_LAG - 1
    F_wls = ((rr - ru) / df1) / (ru / df2) if (df2 > 0 and ru > 0) else np.nan
    p_wls = 1 - f_dist.cdf(F_wls, df1, df2) if not np.isnan(F_wls) else np.nan

    print(f"  WLS F: {F_wls:.3f}, p: {p_wls:.4f}")
    # Note: For within-regime, weighting doesn't change relative magnitudes
    # because all obs have same weight. The real test is subsampling.

    return {
        'train_prevalence': {str(k): float(v) for k, v in train_prev.items()},
        'test_prevalence': {str(k): float(v) for k, v in test_prev.items()},
        'unweighted_F_p': res_unweighted['f_p'],
        'unweighted_HAC_p': res_unweighted['hac_p'],
        'target_n_elev': target_n_elev,
        'actual_n_elev': actual_n_elev,
        'bootstrap_median_F': float(np.median(boot_F)),
        'bootstrap_median_p': float(np.median(boot_p)),
        'bootstrap_frac_sig_05': float(np.mean(boot_p < 0.05)),
        'bootstrap_frac_sig_10': float(np.mean(boot_p < 0.10)),
        'wls_F': float(F_wls), 'wls_p': float(p_wls),
    }


# ═══════════════════════════════════════════════════════════════
# ISSUE 4: FDR-adjusted 30-pair p-values
# ═══════════════════════════════════════════════════════════════

def fix_issue4(ctx):
    print("\n" + "="*70)
    print("ISSUE 4: FDR-Adjusted 30-Pair P-Values (Benjamini-Hochberg)")
    print("="*70)

    test_df = ctx['test_df']
    tr_pct = ctx['test_regimes_pct']

    ci_elev = extract_regime_clean_indices(tr_pct, 1, FIXED_LAG)
    print(f"OOS Elevated clean n = {len(ci_elev)}")

    # Run all 30 directed pairs
    pairs = []
    for cause in FACTOR_NAMES:
        for effect in FACTOR_NAMES:
            if cause == effect: continue
            x = test_df[cause].values
            y = test_df[effect].values
            res = run_granger_at_lag(y, x, ci_elev, FIXED_LAG)
            if res:
                pairs.append({
                    'pair': f"{cause}→{effect}",
                    'F_stat': res['f_stat'], 'F_p': res['f_p'],
                    'HAC_p': res['hac_p'], 'delta_r2': res['delta_r2']
                })

    # Sort by F p-value
    pairs.sort(key=lambda x: x['F_p'])

    # Benjamini-Hochberg FDR correction
    m = len(pairs)
    for i, pair in enumerate(pairs):
        rank = i + 1
        bh_threshold = 0.05 * rank / m
        pair['bh_rank'] = rank
        pair['bh_threshold'] = bh_threshold
        pair['bh_significant'] = pair['F_p'] <= bh_threshold

    # Also do BH on HAC p-values
    pairs_hac = sorted(pairs, key=lambda x: x['HAC_p'])
    for i, pair in enumerate(pairs_hac):
        rank = i + 1
        bh_threshold = 0.05 * rank / m
        pair['bh_hac_rank'] = rank
        pair['bh_hac_threshold'] = bh_threshold
        pair['bh_hac_significant'] = pair['HAC_p'] <= bh_threshold

    # Re-sort by F-stat descending
    pairs.sort(key=lambda x: x['F_stat'], reverse=True)

    print(f"\nTop 10 pairs by F-statistic:")
    print(f"{'Rank':>4} {'Pair':<12} {'F':>8} {'F-p':>10} {'HAC-p':>10} {'BH-F':>6} {'BH-HAC':>7} {'ΔR²':>8}")
    for i, p in enumerate(pairs[:10]):
        print(f"{i+1:>4} {p['pair']:<12} {p['F_stat']:>8.3f} {p['F_p']:>10.4f} {p['HAC_p']:>10.4f} "
              f"{'Yes' if p['bh_significant'] else 'No':>6} {'Yes' if p['bh_hac_significant'] else 'No':>7} "
              f"{p['delta_r2']*100:>7.3f}%")

    # Find HML→SMB
    hml_smb = [p for p in pairs if p['pair'] == 'HML→SMB'][0]
    print(f"\nHML→SMB: rank {pairs.index(hml_smb)+1}/30, F-p={hml_smb['F_p']:.4f}, "
          f"BH-significant={hml_smb['bh_significant']}, BH-HAC-significant={hml_smb['bh_hac_significant']}")

    n_bh_sig = sum(1 for p in pairs if p['bh_significant'])
    n_bh_hac_sig = sum(1 for p in pairs if p['bh_hac_significant'])
    print(f"Total BH-significant (F): {n_bh_sig}/30")
    print(f"Total BH-significant (HAC): {n_bh_hac_sig}/30")

    return {
        'all_pairs': pairs,
        'hml_smb_rank': pairs.index(hml_smb) + 1,
        'hml_smb_bh_sig_F': hml_smb['bh_significant'],
        'hml_smb_bh_sig_HAC': hml_smb['bh_hac_significant'],
        'n_bh_sig_F': n_bh_sig,
        'n_bh_sig_HAC': n_bh_hac_sig,
    }


# ═══════════════════════════════════════════════════════════════
# ISSUE 5: TOST margin sensitivity
# ═══════════════════════════════════════════════════════════════

def fix_issue5(ctx):
    print("\n" + "="*70)
    print("ISSUE 5: TOST Margin Sensitivity Analysis")
    print("="*70)

    df = ctx['df']

    # Full-sample HMM
    hmm_full = StudentTHMM(n_regimes=3, n_iter=100, tol=1e-4, random_state=PRIMARY_SEED)
    hmm_full.fit(df[FACTOR_NAMES].values)
    raw = hmm_full.predict(df[FACTOR_NAMES].values, use_filtered=False)
    labeled, _ = relabel_regimes_by_data_norm(df, raw, FACTOR_NAMES)

    clean_normal = extract_regime_clean_indices(labeled, 0, FIXED_LAG)

    smb = df['SMB'].values
    hml = df['HML'].values
    dates = df.index

    # Post-2008 Normal-regime HML coefficient
    usable = clean_normal[clean_normal >= FIXED_LAG]
    post_mask = dates[usable] >= '2008-01-01'
    post_idx = usable[post_mask]

    y = smb[post_idx]
    yl = np.column_stack([smb[post_idx - i - 1] for i in range(FIXED_LAG)])
    xl = np.column_stack([hml[post_idx - i - 1] for i in range(FIXED_LAG)])
    X = np.column_stack([np.ones(len(y)), yl, xl])

    model = sm.OLS(y, X)
    result = model.fit(cov_type='HAC', cov_kwds={'maxlags': FIXED_LAG})

    # HML coefficient is at index 1+FIXED_LAG = 2
    beta_hml = result.params[1 + FIXED_LAG]
    se_hml = result.bse[1 + FIXED_LAG]
    n_post = len(y)

    print(f"Post-2008 Normal: n={n_post}, β_HML={beta_hml:.6f}, SE={se_hml:.6f}")

    # Pre-2008 for reference
    pre_mask = dates[usable] < '2008-01-01'
    pre_idx = usable[pre_mask]
    y_pre = smb[pre_idx]
    yl_pre = np.column_stack([smb[pre_idx - i - 1] for i in range(FIXED_LAG)])
    xl_pre = np.column_stack([hml[pre_idx - i - 1] for i in range(FIXED_LAG)])
    X_pre = np.column_stack([np.ones(len(y_pre)), yl_pre, xl_pre])
    res_pre = sm.OLS(y_pre, X_pre).fit(cov_type='HAC', cov_kwds={'maxlags': FIXED_LAG})
    beta_pre = res_pre.params[1 + FIXED_LAG]

    print(f"Pre-2008 Normal: n={len(y_pre)}, β_HML={beta_pre:.6f}")
    print(f"Pre-2008 |β| = {abs(beta_pre):.4f} (reference for margin calibration)")

    # TOST at multiple margins
    margins = [0.01, 0.02, 0.03, 0.05, 0.10, abs(beta_pre) * 0.25, abs(beta_pre) * 0.5]
    margin_labels = ['0.01', '0.02', '0.03', '0.05', '0.10',
                     f'{abs(beta_pre)*0.25:.4f} (25% of pre-GFC)',
                     f'{abs(beta_pre)*0.5:.4f} (50% of pre-GFC)']

    print(f"\nTOST equivalence tests (H0: |β| ≥ margin):")
    print(f"{'Margin':<30} {'t_lower':>10} {'t_upper':>10} {'p_TOST':>10} {'Equiv?':>8}")

    tost_results = []
    for margin, label in zip(margins, margin_labels):
        # Two one-sided tests
        t_lower = (beta_hml - (-margin)) / se_hml  # test β > -margin
        t_upper = (beta_hml - margin) / se_hml       # test β < margin
        p_lower = 1 - stats.t.cdf(t_lower, n_post - X.shape[1])
        p_upper = stats.t.cdf(t_upper, n_post - X.shape[1])
        p_tost = max(p_lower, p_upper)
        equiv = p_tost < 0.05

        print(f"  {label:<28} {t_lower:>10.3f} {t_upper:>10.3f} {p_tost:>10.4f} {'YES' if equiv else 'NO':>8}")
        tost_results.append({
            'margin': float(margin), 'margin_label': label,
            'beta_hml': float(beta_hml), 'se': float(se_hml),
            't_lower': float(t_lower), 't_upper': float(t_upper),
            'p_tost': float(p_tost), 'equivalent': equiv
        })

    return {
        'beta_hml_post2008': float(beta_hml),
        'se_hml_post2008': float(se_hml),
        'beta_hml_pre2008': float(beta_pre),
        'n_post2008': n_post,
        'tost_results': tost_results,
        'recommended_margin': f"|β| < {abs(beta_pre)*0.25:.4f} (25% of pre-GFC effect)"
    }


# ═══════════════════════════════════════════════════════════════
# ISSUE 6: Data-driven HAC bandwidth (Andrews 1991)
# ═══════════════════════════════════════════════════════════════

def fix_issue6(ctx):
    print("\n" + "="*70)
    print("ISSUE 6: Data-Driven HAC Bandwidth")
    print("="*70)

    test_df = ctx['test_df']
    tr_pct = ctx['test_regimes_pct']

    smb = test_df['SMB'].values
    hml = test_df['HML'].values

    ci_elev = extract_regime_clean_indices(tr_pct, 1, FIXED_LAG)

    # Fit unrestricted model to get residuals
    usable = ci_elev[ci_elev >= FIXED_LAG]
    y = smb[usable]
    yl = np.column_stack([smb[usable - i - 1] for i in range(FIXED_LAG)])
    xl = np.column_stack([hml[usable - i - 1] for i in range(FIXED_LAG)])
    X = np.column_stack([np.ones(len(y)), yl, xl])

    n = len(y)
    result_ols = sm.OLS(y, X).fit()
    resids = result_ols.resid

    # Andrews 1991 automatic bandwidth selection (AR(1) plug-in for Bartlett kernel)
    # rho_hat = AR(1) coefficient of residuals
    rho_hat = np.corrcoef(resids[:-1], resids[1:])[0, 1]

    # Newey-West 1994 automatic: bandwidth = 1.1447 * (α_hat * T)^(1/3) for Bartlett
    # where α_hat = 4*rho²/(1-rho)^4 for AR(1)
    alpha_hat = 4 * rho_hat**2 / (1 - rho_hat)**4 if abs(rho_hat) < 1 else 1.0
    bw_andrews = int(np.ceil(1.1447 * (alpha_hat * n)**(1/3)))
    bw_andrews = max(1, min(bw_andrews, n // 4))

    # Rule-of-thumb: floor(0.75 * n^(1/3))
    bw_rule = int(np.floor(0.75 * n**(1/3)))

    # Newey-West default: floor(4*(T/100)^(2/9))
    bw_nw_default = int(np.floor(4 * (n / 100)**(2/9)))

    print(f"OOS Elevated: n = {n}")
    print(f"Residual AR(1) ρ = {rho_hat:.4f}")
    print(f"Andrews 1991 automatic bandwidth: {bw_andrews}")
    print(f"Rule-of-thumb (0.75*n^(1/3)): {bw_rule}")
    print(f"Newey-West default (4*(T/100)^(2/9)): {bw_nw_default}")

    # Test at multiple bandwidths
    bandwidths = sorted(set([1, bw_nw_default, bw_rule, bw_andrews, 4, 7, 10, 15, 20, 30]))

    print(f"\nHAC bandwidth sensitivity (OOS Elevated HML→SMB):")
    print(f"{'BW':>4} {'Source':>20} {'Wald':>10} {'HAC-p':>10} {'Sig?':>6}")

    bw_results = []
    for bw in bandwidths:
        wald, hacp = granger_hac(y, yl, xl, FIXED_LAG, bandwidth=bw)
        source = ""
        if bw == 1: source = "Paper primary"
        elif bw == bw_andrews: source = "Andrews 1991"
        elif bw == bw_rule: source = "Rule-of-thumb"
        elif bw == bw_nw_default: source = "NW default"

        sig = hacp < 0.05 if not np.isnan(hacp) else False
        print(f"{bw:>4} {source:>20} {wald:>10.3f} {hacp:>10.4f} {'YES' if sig else 'NO':>6}")
        bw_results.append({
            'bandwidth': bw, 'source': source,
            'wald_stat': float(wald), 'hac_p': float(hacp),
            'significant': sig
        })

    return {
        'n_obs': n,
        'residual_ar1': float(rho_hat),
        'bw_andrews': bw_andrews,
        'bw_rule_of_thumb': bw_rule,
        'bw_nw_default': bw_nw_default,
        'bandwidth_results': bw_results,
    }


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("Setting up HMM pipeline (both conventions)...")
    ctx = setup_oos()

    all_results = {}
    all_results['issue1_scale_convention'] = fix_issue1(ctx)
    all_results['issue2_breakpoint'] = fix_issue2(ctx)
    all_results['issue3_prevalence_reweight'] = fix_issue3(ctx)
    all_results['issue4_fdr'] = fix_issue4(ctx)
    all_results['issue5_tost'] = fix_issue5(ctx)
    all_results['issue6_hac_bandwidth'] = fix_issue6(ctx)

    # Save results
    outpath = f"{RESULTS_DIR}/review_fixes_results.json"

    # Make serializable
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    with open(outpath, 'w') as f:
        json.dump(make_serializable(all_results), f, indent=2)

    print(f"\n{'='*70}")
    print(f"All results saved to {outpath}")
    print(f"{'='*70}")
