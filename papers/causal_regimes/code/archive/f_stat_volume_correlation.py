"""
F-statistic vs IWM Volume Correlation Analysis
==============================================

Test: Does small cap ETF trading volume correlate with
decay of X→SMB Granger predictability?
"""

import numpy as np
import pandas as pd
import urllib.request
import zipfile
import io
import yfinance as yf
from scipy.special import gammaln
from scipy.cluster.vq import kmeans2
from scipy import stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')


def download_ff_factors():
    """Download FF 5 factors + Momentum."""
    url_5f = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip'
    with urllib.request.urlopen(url_5f, timeout=60) as response:
        data = response.read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        csv_name = z.namelist()[0]
        with z.open(csv_name) as f:
            df_5f = pd.read_csv(f, skiprows=3)

    df_5f.columns = df_5f.columns.str.strip()
    df_5f = df_5f.rename(columns={df_5f.columns[0]: 'Date'})
    df_5f = df_5f[df_5f['Date'].astype(str).str.match(r'^\d{8}$')]
    df_5f['Date'] = pd.to_datetime(df_5f['Date'], format='%Y%m%d')
    for col in ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']:
        df_5f[col] = pd.to_numeric(df_5f[col], errors='coerce')
    df_5f = df_5f.set_index('Date')

    url_mom = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip'
    with urllib.request.urlopen(url_mom, timeout=60) as response:
        data = response.read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        csv_name = z.namelist()[0]
        with z.open(csv_name) as f:
            df_mom = pd.read_csv(f, skiprows=13)

    df_mom.columns = df_mom.columns.str.strip()
    df_mom = df_mom.rename(columns={df_mom.columns[0]: 'Date', df_mom.columns[1]: 'MOM'})
    df_mom = df_mom[df_mom['Date'].astype(str).str.match(r'^\d{8}$')]
    df_mom['Date'] = pd.to_datetime(df_mom['Date'], format='%Y%m%d')
    df_mom['MOM'] = pd.to_numeric(df_mom['MOM'], errors='coerce')
    df_mom = df_mom.set_index('Date')[['MOM']]

    df = df_5f.join(df_mom, how='inner')
    return df.dropna()


class SimpleHMM:
    """Simple regime classifier using k-means."""
    def __init__(self, n_regimes=3, random_state=28):
        self.n_regimes = n_regimes
        self.random_state = random_state

    def fit_predict(self, X):
        np.random.seed(self.random_state)
        centroids, labels = kmeans2(X, self.n_regimes, minit='++')
        norms = np.linalg.norm(centroids, axis=1)
        order = np.argsort(norms)
        new_labels = np.zeros_like(labels)
        for new_k, old_k in enumerate(order):
            new_labels[labels == old_k] = new_k
        return new_labels


def granger_test(Y, X, max_lag=5):
    """Simple Granger causality test."""
    T = len(Y)
    if T < max_lag + 30:
        return None

    Y_lags = np.column_stack([Y[max_lag-i-1:T-i-1] for i in range(max_lag)])
    X_lags = np.column_stack([X[max_lag-i-1:T-i-1] for i in range(max_lag)])
    Y_target = Y[max_lag:]

    X_r = sm.add_constant(Y_lags)
    X_u = sm.add_constant(np.column_stack([Y_lags, X_lags]))

    try:
        model_r = sm.OLS(Y_target, X_r).fit()
        model_u = sm.OLS(Y_target, X_u).fit()

        rss_r = (model_r.resid ** 2).sum()
        rss_u = (model_u.resid ** 2).sum()

        n = len(Y_target)
        k_r, k_u = X_r.shape[1], X_u.shape[1]

        F = ((rss_r - rss_u) / (k_u - k_r)) / (rss_u / (n - k_u))
        return F
    except:
        return None


def get_yearly_f_stats(df, source, target, regimes, regime_idx=0):
    """Get F-statistic for each year."""
    results = []
    dates = df.index

    for year in range(2001, 2024):
        start = pd.Timestamp(f'{year}-01-01')
        end = pd.Timestamp(f'{year}-12-31')

        mask = (dates >= start) & (dates <= end) & (regimes == regime_idx)
        if mask.sum() < 50:
            continue

        Y = df.loc[mask, target].values
        X = df.loc[mask, source].values

        F = granger_test(Y, X)
        if F is not None:
            results.append({'year': year, 'F': F})

    return pd.DataFrame(results)


def main():
    print("=" * 70)
    print("F-STATISTIC vs IWM VOLUME CORRELATION")
    print("=" * 70)

    # 1. Load factor data
    print("\n[1] Loading Fama-French factors...")
    df = download_ff_factors()
    df = df[(df.index >= '1990-01-01') & (df.index <= '2024-12-31')]

    # 2. Load IWM volume
    print("\n[2] Loading IWM volume...")
    iwm = yf.download('IWM', start='2000-01-01', end='2024-12-31', progress=False)
    if isinstance(iwm.columns, pd.MultiIndex):
        iwm.columns = iwm.columns.get_level_values(0)

    # Yearly average volume
    iwm_yearly = iwm.groupby(iwm.index.year)['Volume'].mean()
    iwm_yearly = iwm_yearly.reset_index()
    iwm_yearly.columns = ['year', 'volume']
    print(f"  IWM yearly data: {iwm_yearly['year'].min()} to {iwm_yearly['year'].max()}")

    # 3. Fit HMM
    print("\n[3] Fitting regime model...")
    X = df[['Mkt-RF', 'SMB', 'HML']].values
    hmm = SimpleHMM(n_regimes=3)
    regimes = hmm.fit_predict(X)

    # 4. Get yearly F-stats for X→SMB pairs
    print("\n[4] Computing yearly F-statistics for X→SMB pairs...")

    smb_pairs = [
        ('Mkt-RF', 'SMB'),
        ('HML', 'SMB'),
        ('CMA', 'SMB'),
        ('RMW', 'SMB'),
        ('MOM', 'SMB'),
    ]

    all_f_stats = []

    for source, target in smb_pairs:
        yearly_f = get_yearly_f_stats(df, source, target, regimes, regime_idx=0)
        if len(yearly_f) > 0:
            yearly_f['pair'] = f'{source}→{target}'
            all_f_stats.append(yearly_f)
            print(f"  {source}→{target}: {len(yearly_f)} years")

    f_stats_df = pd.concat(all_f_stats, ignore_index=True)

    # Average F across all X→SMB pairs per year
    avg_f_by_year = f_stats_df.groupby('year')['F'].mean().reset_index()
    avg_f_by_year.columns = ['year', 'avg_F']

    # 5. Merge with IWM volume
    print("\n[5] Merging F-statistics with IWM volume...")
    merged = avg_f_by_year.merge(iwm_yearly, on='year', how='inner')
    merged['log_volume'] = np.log(merged['volume'])

    print(f"  Matched years: {len(merged)}")
    print(f"  Year range: {merged['year'].min()} to {merged['year'].max()}")

    # 6. Correlation analysis
    print("\n" + "=" * 70)
    print("CORRELATION ANALYSIS")
    print("=" * 70)

    # Pearson correlation
    corr_pearson, p_pearson = stats.pearsonr(merged['avg_F'], merged['volume'])
    corr_log, p_log = stats.pearsonr(merged['avg_F'], merged['log_volume'])

    # Spearman (rank) correlation
    corr_spearman, p_spearman = stats.spearmanr(merged['avg_F'], merged['volume'])

    print(f"\nCorrelation: Avg X→SMB F-statistic vs IWM Volume")
    print("-" * 50)
    print(f"  Pearson:  r = {corr_pearson:.3f}, p = {p_pearson:.4f}")
    print(f"  Spearman: ρ = {corr_spearman:.3f}, p = {p_spearman:.4f}")
    print(f"  Pearson (log vol): r = {corr_log:.3f}, p = {p_log:.4f}")

    # 7. Show data
    print("\n" + "=" * 70)
    print("YEARLY DATA")
    print("=" * 70)
    print(f"\n{'Year':>6} {'Avg F':>10} {'IWM Vol (M)':>15}")
    print("-" * 35)
    for _, row in merged.iterrows():
        print(f"{int(row['year']):>6} {row['avg_F']:>10.2f} {row['volume']/1e6:>15.1f}")

    # 8. Regression
    print("\n" + "=" * 70)
    print("REGRESSION: F ~ Volume")
    print("=" * 70)

    X_reg = sm.add_constant(merged['log_volume'])
    model = sm.OLS(merged['avg_F'], X_reg).fit()

    print(f"\n  F = {model.params['const']:.2f} + {model.params['log_volume']:.2f} * log(Volume)")
    print(f"  R² = {model.rsquared:.3f}")
    print(f"  Volume coefficient p-value: {model.pvalues['log_volume']:.4f}")

    # 9. Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    if corr_spearman < -0.5 and p_spearman < 0.05:
        print("""
  FINDING: Significant NEGATIVE correlation between X→SMB
  predictability and IWM trading volume.

  - As small cap ETF volume increased, predictability decreased
  - Spearman ρ = {:.3f} (p = {:.4f})
  - Consistent with market efficiency hypothesis

  CAVEAT: Correlation does not prove causation.
  Alternative explanations:
  - Both may be driven by third factor (e.g., market maturation)
  - Volume increase may be effect, not cause
  - Other factors may dominate
        """.format(corr_spearman, p_spearman))
    elif corr_spearman < 0:
        print(f"""
  FINDING: Negative but {'weak' if abs(corr_spearman) < 0.3 else 'moderate'} correlation.
  Spearman ρ = {corr_spearman:.3f} (p = {p_spearman:.4f})

  Suggestive but not conclusive evidence for liquidity-efficiency link.
        """)
    else:
        print(f"""
  FINDING: No negative correlation found.
  Spearman ρ = {corr_spearman:.3f} (p = {p_spearman:.4f})

  Liquidity hypothesis not supported by this analysis.
        """)

    return merged, model


if __name__ == '__main__':
    merged, model = main()

    # Save results
    merged.to_csv('results/f_stat_vs_volume.csv', index=False)
    print("\nResults saved to results/f_stat_vs_volume.csv")
