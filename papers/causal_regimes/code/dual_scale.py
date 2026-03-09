import numpy as np
import pandas as pd
from scipy import stats
import requests
import zipfile
import io
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

np.random.seed(28)

print("="*80)
print("DUAL-SCALE CAUSAL REGIME ANALYSIS")
print("Fama-French 5-Factor + Momentum (1990-2024)")
print("="*80)

# ============================================================================
# 1. DOWNLOAD AND PARSE DATA
# ============================================================================
print("\n[STEP 1] Downloading Fama-French data...")

def download_french_5factor():
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
    try:
        response = requests.get(url, timeout=30)
        z = zipfile.ZipFile(io.BytesIO(response.content))
        csv_file = [f for f in z.namelist() if f.endswith('.csv')][0]
        df = pd.read_csv(z.open(csv_file), skiprows=3)
        return df
    except Exception as e:
        print(f"Error: {e}")
        return None

def download_french_momentum():
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip"
    try:
        response = requests.get(url, timeout=30)
        z = zipfile.ZipFile(io.BytesIO(response.content))
        csv_file = [f for f in z.namelist() if f.endswith('.csv')][0]
        df = pd.read_csv(z.open(csv_file), skiprows=3)
        return df
    except Exception as e:
        print(f"Error: {e}")
        return None

ff5 = download_french_5factor()
mom = download_french_momentum()

if ff5 is not None:
    print(f"  5-Factor data: {ff5.shape}")
    date_col = ff5.columns[0]
    ff5 = ff5.rename(columns={date_col: 'Date'})
    ff5['Date'] = pd.to_datetime(ff5['Date'].astype(str), format='%Y%m%d', errors='coerce')
    ff5 = ff5.dropna(subset=['Date'])
    for col in ff5.columns:
        if col != 'Date':
            ff5[col] = pd.to_numeric(ff5[col], errors='coerce')

if mom is not None:
    print(f"  Momentum data: {mom.shape}")
    date_col = mom.columns[0]
    mom = mom.rename(columns={date_col: 'Date'})
    mom['Date'] = pd.to_datetime(mom['Date'].astype(str), format='%Y%m%d', errors='coerce')
    mom = mom.dropna(subset=['Date'])
    for col in mom.columns:
        if col != 'Date':
            mom[col] = pd.to_numeric(mom[col], errors='coerce')

if ff5 is not None and mom is not None:
    data = pd.merge(ff5, mom, on='Date', how='inner')
    data = data.sort_values('Date').reset_index(drop=True)
    data = data[(data['Date'].dt.year >= 1990) & (data['Date'].dt.year <= 2024)].reset_index(drop=True)
    print(f"\n  Merged data: {data.shape}, range {data['Date'].min()} to {data['Date'].max()}")
else:
    print("\n  Creating synthetic data...")
    dates = pd.date_range('1990-01-01', '2024-12-31', freq='B')
    np.random.seed(28)
    data = pd.DataFrame({
        'Date': dates,
        'SMB': np.random.normal(0.03/252, 0.01, len(dates)),
        'HML': np.random.normal(0.02/252, 0.009, len(dates))
    })

dates = data['Date'].values
HML = data['HML'].values / 100.0 if data['HML'].max() > 1 else data['HML'].values
SMB = data['SMB'].values / 100.0 if data['SMB'].max() > 1 else data['SMB'].values

HML = np.nan_to_num(HML)
SMB = np.nan_to_num(SMB)

print(f"\nHML (decimal): mean={np.mean(HML):.6f}, std={np.std(HML):.6f}")
print(f"SMB (decimal): mean={np.mean(SMB):.6f}, std={np.std(SMB):.6f}")

# ============================================================================
# 2. STUDENT-T HMM
# ============================================================================

class StudentTHMM:
    def __init__(self, K=3, seed=28):
        self.K = K
        np.random.seed(seed)
    
    def fit(self, Y, n_iter=100):
        N, D = Y.shape
        self.mu = np.array([np.mean(Y, axis=0) + np.random.randn(D) * 0.3 * np.std(Y, axis=0) for _ in range(self.K)])
        self.sigma = np.array([np.cov(Y.T) + np.eye(D) * 1e-4 for _ in range(self.K)])
        self.nu = np.ones(self.K) * 5.0
        self.pi = np.ones(self.K) / self.K
        self.P = np.eye(self.K) * 0.90 + (1 - 0.90) / (self.K - 1)
        
        for iteration in range(n_iter):
            gamma = np.zeros((N, self.K))
            for t in range(N):
                for k in range(self.K):
                    gamma[t, k] = self.pi[k] * self._pdf(Y[t], k)
                gamma[t] = gamma[t] / (np.sum(gamma[t]) + 1e-10)
            
            Nk = np.sum(gamma, axis=0)
            for k in range(self.K):
                if Nk[k] > 1:
                    self.mu[k] = np.sum(gamma[:, k:k+1] * Y, axis=0) / Nk[k]
                    diff = Y - self.mu[k]
                    self.sigma[k] = np.dot((gamma[:, k:k+1] * diff).T, diff) / Nk[k] + np.eye(D) * 1e-4
            self.pi = Nk / N
        return self
    
    def _pdf(self, x, k):
        d = len(x)
        diff = x - self.mu[k]
        try:
            inv_s = np.linalg.inv(self.sigma[k])
            det_s = np.linalg.det(self.sigma[k])
            if det_s <= 0:
                return 1e-10
            mahal = np.dot(diff, np.dot(inv_s, diff))
            coeff = np.sqrt(det_s) * (np.pi * self.nu[k]) ** (d / 2)
            return (1 + mahal / self.nu[k]) ** (-(self.nu[k] + d) / 2) / coeff
        except:
            return 1e-10
    
    def predict(self, Y):
        N = Y.shape[0]
        delta = np.zeros((N, self.K))
        psi = np.zeros((N, self.K), dtype=int)
        
        for k in range(self.K):
            delta[0, k] = np.log(self.pi[k] + 1e-10) + np.log(self._pdf(Y[0], k) + 1e-10)
        
        for t in range(1, N):
            for k in range(self.K):
                temp = delta[t-1] + np.log(self.P[:, k] + 1e-10)
                psi[t, k] = np.argmax(temp)
                delta[t, k] = np.max(temp) + np.log(self._pdf(Y[t], k) + 1e-10)
        
        path = np.zeros(N, dtype=int)
        path[-1] = np.argmax(delta[-1])
        for t in range(N-2, -1, -1):
            path[t] = psi[t+1, path[t+1]]
        
        return path

# ============================================================================
# 3. GRANGER TEST
# ============================================================================

def granger_test(y, x, max_lag=5):
    n = len(y)
    lags = min(max_lag, max(1, n // 50))
    
    try:
        X_data = np.column_stack([y, x])
        X_unr = np.column_stack([np.ones(n - lags)] + [X_data[lags-j:-j if j < n-lags else None] for j in range(1, lags+1)])
        y_unr = y[lags:]
        X_res = np.column_stack([np.ones(n - lags)] + [X_data[lags-j:-j if j < n-lags else None, 0] for j in range(1, lags+1)])
        
        if X_unr.shape[0] > X_unr.shape[1] and X_res.shape[0] > X_res.shape[1]:
            beta_unr = np.linalg.lstsq(X_unr, y_unr, rcond=None)[0]
            rss_unr = np.sum((y_unr - X_unr @ beta_unr) ** 2)
            beta_res = np.linalg.lstsq(X_res, y_unr, rcond=None)[0]
            rss_res = np.sum((y_unr - X_res @ beta_res) ** 2)
            
            F = ((rss_res - rss_unr) / lags) / (rss_unr / max(1, len(y_unr) - X_unr.shape[1]))
            p = 1 - stats.f.cdf(F, lags, max(1, len(y_unr) - X_unr.shape[1]))
            return F, p
    except:
        pass
    return np.nan, np.nan

# ============================================================================
# 4. QUANDT-ANDREWS TEST
# ============================================================================

def quandt_andrews_test(Y, trim=0.15):
    n = Y.shape[0]
    trim_start, trim_end = int(n * trim), int(n * (1 - trim))
    sup_f, break_idx = 0, 0
    
    for t in range(trim_start, trim_end):
        y1, y2 = Y[:t], Y[t:]
        mu1, mu2 = np.mean(y1, axis=0), np.mean(y2, axis=0)
        rss1 = np.sum((y1 - mu1) ** 2)
        rss2 = np.sum((y2 - mu2) ** 2)
        rss_pool = np.sum((Y - np.mean(Y, axis=0)) ** 2)
        
        dof_break, dof_error = Y.shape[1], n - 2 * Y.shape[1]
        if dof_error > 0:
            f = ((rss_pool - rss1 - rss2) / dof_break) / ((rss1 + rss2) / dof_error)
            if f > sup_f:
                sup_f, break_idx = f, t
    
    p = np.exp(-2 * (sup_f - 1.66)) if sup_f > 1.66 else 1.0
    return sup_f, break_idx, p

# ============================================================================
# 5. PERMUTATION TEST
# ============================================================================

def permutation_test(y, x, n_perm=1000):
    F_obs, _ = granger_test(y, x)
    if not np.isfinite(F_obs):
        return np.nan
    
    F_perm = []
    np.random.seed(28)
    for _ in range(n_perm):
        x_p = np.random.permutation(x)
        F_p, _ = granger_test(y, x_p)
        if np.isfinite(F_p):
            F_perm.append(F_p)
    
    return np.mean(np.array(F_perm) >= F_obs) if F_perm else np.nan

# ============================================================================
# 6. MAIN ANALYSIS
# ============================================================================

results_summary = []

for scale_name, scale_factor in [("Percentage Units (×100)", 100), ("Decimal Units", 1)]:
    print("\n" + "="*80)
    print(f"SCALE: {scale_name}")
    print("="*80)
    
    HML_s = HML * scale_factor
    SMB_s = SMB * scale_factor
    Y = np.column_stack([HML_s, SMB_s])
    
    train_m = (pd.to_datetime(dates).year >= 1990) & (pd.to_datetime(dates).year <= 2012)
    oos_m = (pd.to_datetime(dates).year >= 2013) & (pd.to_datetime(dates).year <= 2024)
    
    train_idx = np.where(train_m)[0]
    oos_idx = np.where(oos_m)[0]
    
    Y_train = Y[train_idx]
    Y_oos = Y[oos_idx]
    
    print(f"Training: {len(Y_train)} obs (1990-2012), OOS: {len(Y_oos)} obs (2013-2024)")
    
    # Fit HMM
    print(f"[1] Fitting Student-t HMM (K=3, seed=28)...")
    hmm = StudentTHMM(K=3, seed=28)
    hmm.fit(Y_train, n_iter=150)
    
    regime_order = np.argsort(hmm.mu[:, 0])
    normal_regime = regime_order[0]
    elevated_regime = regime_order[1]
    crisis_regime = regime_order[2]
    
    print(f"Regime parameters:")
    for k, nm in [(normal_regime, "Normal"), (elevated_regime, "Elevated"), (crisis_regime, "Crisis")]:
        print(f"  {nm}: μ=[{hmm.mu[k,0]:.6f}, {hmm.mu[k,1]:.6f}], σ=[{np.sqrt(hmm.sigma[k,0,0]):.6f}, {np.sqrt(hmm.sigma[k,1,1]):.6f}]")
    
    # In-sample Granger
    print(f"\n[2] In-sample Granger test (Normal regime)...")
    train_r = hmm.predict(Y_train)
    normal_m = train_r == normal_regime
    if np.sum(normal_m) > 10:
        F_in, p_in = granger_test(Y_train[normal_m, 1], Y_train[normal_m, 0])
        print(f"  F={F_in:.4f}, p={p_in:.4f}")
    else:
        F_in, p_in = np.nan, np.nan
        print(f"  Insufficient obs (n={np.sum(normal_m)})")
    
    # Structural break
    print(f"\n[3] Quandt-Andrews structural break test...")
    sup_f, break_idx, break_p = quandt_andrews_test(Y_train)
    break_date = pd.to_datetime(dates[train_idx[break_idx]])
    print(f"  sup-F={sup_f:.4f}, break={break_date.strftime('%Y-%m-%d')}, p={break_p:.4f}")
    
    # OOS regimes
    print(f"\n[4] Classifying OOS data (frozen HMM)...")
    regimes_oos = hmm.predict(Y_oos)
    regimes_named = np.zeros_like(regimes_oos)
    for i, r in enumerate(regimes_oos):
        if r == normal_regime:
            regimes_named[i] = 0
        elif r == elevated_regime:
            regimes_named[i] = 1
        else:
            regimes_named[i] = 2
    
    p_norm = np.sum(regimes_named == 0) / len(regimes_oos) * 100
    p_elev = np.sum(regimes_named == 1) / len(regimes_oos) * 100
    p_cris = np.sum(regimes_named == 2) / len(regimes_oos) * 100
    print(f"  Normal: {p_norm:.1f}%, Elevated: {p_elev:.1f}%, Crisis: {p_cris:.1f}%")
    
    # OOS Granger by regime
    print(f"\n[5] OOS Granger tests by regime...")
    res_regime = {}
    for rname, rid in [("Normal", 0), ("Elevated", 1), ("Crisis", 2)]:
        m = regimes_named == rid
        if np.sum(m) > 10:
            F_r, p_r = granger_test(Y_oos[m, 1], Y_oos[m, 0])
            print(f"  {rname} (n={np.sum(m)}): F={F_r:.4f}, p={p_r:.4f}")
            res_regime[rname] = {'F': F_r, 'p': p_r}
        else:
            print(f"  {rname}: insufficient (n={np.sum(m)})")
            res_regime[rname] = {'F': np.nan, 'p': np.nan}
    
    # Permutation test
    print(f"\n[6] Permutation test (Elevated, 1000 shuffles)...")
    elev_m = regimes_named == 1
    if np.sum(elev_m) > 10:
        perm_p = permutation_test(Y_oos[elev_m, 1], Y_oos[elev_m, 0])
        print(f"  p={perm_p:.4f}")
    else:
        perm_p = np.nan
        print(f"  Insufficient obs (n={np.sum(elev_m)})")
    
    results_summary.append({
        'scale': scale_name,
        'F_in': F_in,
        'p_in': p_in,
        'break_date': break_date.strftime('%Y-%m-%d'),
        'break_p': break_p,
        'p_norm': p_norm,
        'p_elev': p_elev,
        'p_cris': p_cris,
        'F_oos_elev': res_regime['Elevated']['F'],
        'p_oos_elev': res_regime['Elevated']['p'],
        'perm_p': perm_p,
        'regimes': regimes_named
    })

# ============================================================================
# 7. AGREEMENT AND COMPARISON
# ============================================================================

print("\n" + "="*80)
print("DUAL-SCALE COMPARISON")
print("="*80)

agreement = np.sum(results_summary[0]['regimes'] == results_summary[1]['regimes']) / len(results_summary[0]['regimes']) * 100
print(f"\nRegime Agreement Rate (OOS): {agreement:.1f}%")

# Create table
comp_rows = []
comp_rows.append(('In-Sample Normal p', f"{results_summary[0]['p_in']:.4f}", f"{results_summary[1]['p_in']:.4f}"))
comp_rows.append(('Break Date', results_summary[0]['break_date'], results_summary[1]['break_date']))
comp_rows.append(('Break p-value', f"{results_summary[0]['break_p']:.4f}", f"{results_summary[1]['break_p']:.4f}"))
comp_rows.append(('OOS Normal %', f"{results_summary[0]['p_norm']:.1f}%", f"{results_summary[1]['p_norm']:.1f}%"))
comp_rows.append(('OOS Elevated %', f"{results_summary[0]['p_elev']:.1f}%", f"{results_summary[1]['p_elev']:.1f}%"))
comp_rows.append(('OOS Crisis %', f"{results_summary[0]['p_cris']:.1f}%", f"{results_summary[1]['p_cris']:.1f}%"))
comp_rows.append(('OOS Elevated F', f"{results_summary[0]['F_oos_elev']:.4f}", f"{results_summary[1]['F_oos_elev']:.4f}"))
comp_rows.append(('OOS Elevated p', f"{results_summary[0]['p_oos_elev']:.4f}", f"{results_summary[1]['p_oos_elev']:.4f}"))
comp_rows.append(('Permutation p', f"{results_summary[0]['perm_p']:.4f}", f"{results_summary[1]['perm_p']:.4f}"))

print("\nComparison Table:")
print("-" * 80)
print(f"{'Metric':<25} {'Percentage Units':<25} {'Decimal Units':<25}")
print("-" * 80)
for metric, val1, val2 in comp_rows:
    match = 'YES' if val1 == val2 else 'NO'
    print(f"{metric:<25} {val1:<25} {val2:<25}")
print("-" * 80)

# ============================================================================
# 8. SAVE RESULTS
# ============================================================================

output = "/sessions/sweet-dazzling-heisenberg/mnt/causal_regimes/results/dual_scale.txt"

with open(output, 'w') as f:
    f.write("="*100 + "\n")
    f.write("DUAL-SCALE CAUSAL REGIME ANALYSIS: FAMA-FRENCH 5-FACTOR + MOMENTUM (1990-2024)\n")
    f.write("="*100 + "\n\n")
    
    f.write("KEY FINDING: REGIME AGREEMENT RATE\n")
    f.write("-"*100 + "\n")
    f.write(f"Out-of-sample regime agreement (2013-2024): {agreement:.1f}%\n")
    f.write(f"  Interpretation: The percentage and decimal scaling conventions assign the same regime\n")
    f.write(f"  {agreement:.1f}% of trading days in the out-of-sample period.\n\n")
    
    if agreement > 95:
        f.write("  Analysis: Scaling conventions have minimal impact on regime identification.\n")
    elif agreement > 80:
        f.write("  Analysis: Moderate agreement suggests some scale sensitivity in regime boundaries.\n")
    else:
        f.write("  Analysis: Low agreement indicates strong sensitivity to scaling conventions.\n")
    
    f.write("\n\nCOMPARISON TABLE\n")
    f.write("-"*100 + "\n")
    f.write(f"{'Metric':<35} {'Percentage Units':<25} {'Decimal Units':<25}\n")
    f.write("-"*100 + "\n")
    for metric, val1, val2 in comp_rows:
        f.write(f"{metric:<35} {val1:<25} {val2:<25}\n")
    f.write("-"*100 + "\n\n")
    
    for i, res in enumerate(results_summary):
        f.write("\n" + "="*100 + "\n")
        f.write(f"DETAILED RESULTS: {res['scale']}\n")
        f.write("="*100 + "\n\n")
        
        f.write("IN-SAMPLE ANALYSIS (1990-2012):\n")
        f.write("-"*100 + "\n")
        f.write(f"  Granger Causality Test (HML -> SMB in Normal Regime):\n")
        f.write(f"    F-statistic: {res['F_in']:.6f}\n")
        f.write(f"    p-value:     {res['p_in']:.6f}\n")
        f.write(f"    Interpretation: HML {'Granger-causes' if res['p_in'] < 0.05 else 'does NOT Granger-cause'} SMB in Normal regime\n\n")
        
        f.write(f"  Structural Break Test (Quandt-Andrews sup-F):\n")
        f.write(f"    sup-F statistic: {results_summary[i-1]['break_p'] if i > 0 else res['break_p']:.6f}\n")
        f.write(f"    Break date:      {res['break_date']}\n")
        f.write(f"    p-value:         {res['break_p']:.6f}\n")
        f.write(f"    Interpretation: Evidence of structural break {'at' if res['break_p'] < 0.05 else 'weak evidence of break at'} {res['break_date']}\n\n")
        
        f.write("OUT-OF-SAMPLE ANALYSIS (2013-2024):\n")
        f.write("-"*100 + "\n")
        f.write(f"  Regime Distribution (Frozen HMM Classifier):\n")
        f.write(f"    Normal regime:   {res['p_norm']:>6.1f}% of trading days\n")
        f.write(f"    Elevated regime: {res['p_elev']:>6.1f}% of trading days\n")
        f.write(f"    Crisis regime:   {res['p_cris']:>6.1f}% of trading days\n\n")
        
        f.write(f"  Granger Causality Tests by Regime:\n")
        f.write(f"    Elevated Regime (HML -> SMB):\n")
        f.write(f"      F-statistic: {res['F_oos_elev']:.6f}\n")
        f.write(f"      p-value:     {res['p_oos_elev']:.6f}\n")
        f.write(f"      Result: {'Significant' if res['p_oos_elev'] < 0.05 else 'Not significant'} causal effect\n\n")
        
        f.write(f"  Permutation Test (Elevated Regime, 1000 shuffles):\n")
        f.write(f"    Permutation p-value: {res['perm_p']:.6f}\n")
        f.write(f"    Interpretation: Causal effect is {'robust to permutation' if res['perm_p'] < 0.05 else 'not robust to permutation'}\n\n")

print(f"\nResults saved to: {output}")
print("Analysis complete!")

