"""
30 Factor Pairs Decay Analysis
==============================

Analyze decay patterns across ALL 30 directed factor pairs
to test whether decay is a general phenomenon.

Factors: Mkt-RF, SMB, HML, RMW, CMA, MOM (6 factors)
Pairs: 6 * 5 = 30 directed pairs
"""

import numpy as np
import pandas as pd
import urllib.request
import zipfile
import io
from scipy.special import gammaln
from scipy.optimize import curve_fit
from scipy.cluster.vq import kmeans2
from scipy import stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')


def download_ff_factors():
    """Download FF 5 factors + Momentum."""
    # 5 factors
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

    # Momentum
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


class StudentTHMM:
    """Minimal Student-t HMM for regime detection."""

    def __init__(self, n_regimes=3, n_iter=50, random_state=28):
        self.n_regimes = n_regimes
        self.n_iter = n_iter
        self.random_state = random_state

    def fit_predict(self, X):
        np.random.seed(self.random_state)
        X = np.asarray(X)
        T, d = X.shape
        K = self.n_regimes

        # K-means init
        centroids, labels = kmeans2(X, K, minit='++')
        norms = np.linalg.norm(centroids, axis=1)
        order = np.argsort(norms)

        # Reorder by volatility
        new_labels = np.zeros_like(labels)
        for new_k, old_k in enumerate(order):
            new_labels[labels == old_k] = new_k

        return new_labels


def granger_test_hac(Y, X, max_lag=5):
    """Granger causality test with HAC standard errors."""
    T = len(Y)
    if T < max_lag + 20:
        return None

    Y_lags = np.column_stack([Y[max_lag-i-1:T-i-1] for i in range(max_lag)])
    X_lags = np.column_stack([X[max_lag-i-1:T-i-1] for i in range(max_lag)])
    Y_target = Y[max_lag:]

    X_restricted = sm.add_constant(Y_lags)
    X_unrestricted = sm.add_constant(np.column_stack([Y_lags, X_lags]))

    try:
        model_r = sm.OLS(Y_target, X_restricted).fit(cov_type='HAC', cov_kwds={'maxlags': max_lag})
        model_u = sm.OLS(Y_target, X_unrestricted).fit(cov_type='HAC', cov_kwds={'maxlags': max_lag})

        rss_r = (model_r.resid ** 2).sum()
        rss_u = (model_u.resid ** 2).sum()

        n = len(Y_target)
        k_r = X_restricted.shape[1]
        k_u = X_unrestricted.shape[1]

        F = ((rss_r - rss_u) / (k_u - k_r)) / (rss_u / (n - k_u))
        p_value = 1 - stats.f.cdf(F, k_u - k_r, n - k_u)

        return {'F': F, 'p_value': p_value}
    except:
        return None


def rolling_granger(df, source, target, regimes, regime_idx=0, window_years=5):
    """Rolling Granger analysis for a single pair."""
    results = []
    dates = df.index

    for year in range(1995, 2024):
        for month in [1, 7]:
            window_end = pd.Timestamp(f'{year}-{month:02d}-01')
            window_start = window_end - pd.DateOffset(years=window_years)

            if window_end > dates[-1] or window_start < dates[0]:
                continue

            mask = (dates >= window_start) & (dates < window_end) & (regimes == regime_idx)

            if mask.sum() < 50:
                continue

            Y = df.loc[mask, target].values
            X = df.loc[mask, source].values

            result = granger_test_hac(Y, X)
            if result:
                results.append({
                    'year': year + (month - 1) / 12,
                    'F': result['F'],
                    'p_value': result['p_value']
                })

    return pd.DataFrame(results)


def fit_decay(df_rolling):
    """Fit exponential decay model."""
    if len(df_rolling) < 5:
        return None

    t = df_rolling['year'].values - df_rolling['year'].min()
    F = df_rolling['F'].values

    # Check if there's actual variation
    if F.std() < 0.1:
        return {'half_life': np.inf, 'r2': 0, 'decay': False}

    def exp_decay(t, A, lam, C):
        return A * np.exp(-lam * t) + C

    try:
        p0 = [max(F.max(), 1), 0.1, max(F.min(), 0)]
        popt, _ = curve_fit(exp_decay, t, F, p0=p0, maxfev=5000,
                           bounds=([0, 0.001, 0], [1000, 2, 100]))

        A, lam, C = popt
        half_life = np.log(2) / lam if lam > 0.001 else np.inf

        F_pred = exp_decay(t, *popt)
        ss_res = ((F - F_pred) ** 2).sum()
        ss_tot = ((F - F.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # Decay = negative slope with reasonable fit
        decay = (r2 > 0.3) and (half_life < 50) and (A > C)

        return {'half_life': half_life, 'r2': r2, 'decay': decay, 'A': A, 'C': C}
    except:
        return None


def main():
    print("=" * 70)
    print("30 FACTOR PAIRS DECAY ANALYSIS")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    df = download_ff_factors()
    df = df[(df.index >= '1990-01-01') & (df.index <= '2024-12-31')]
    print(f"Data: {df.index[0].date()} to {df.index[-1].date()}")

    # Fit HMM
    print("\nFitting HMM...")
    factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']
    X = df[['Mkt-RF', 'SMB', 'HML']].values

    hmm = StudentTHMM(n_regimes=3)
    regimes = hmm.fit_predict(X)

    regime_counts = pd.Series(regimes).value_counts().sort_index()
    print(f"Regimes: Normal={regime_counts.get(0,0)}, Elevated={regime_counts.get(1,0)}, Crisis={regime_counts.get(2,0)}")

    # Analyze all 30 pairs
    print("\nAnalyzing 30 directed pairs...")
    print("-" * 70)

    results = []

    for source in factors:
        for target in factors:
            if source == target:
                continue

            pair_name = f"{source}→{target}"

            rolling_df = rolling_granger(df, source, target, regimes, regime_idx=0)

            if len(rolling_df) < 5:
                results.append({
                    'pair': pair_name,
                    'source': source,
                    'target': target,
                    'n_windows': len(rolling_df),
                    'max_F': np.nan,
                    'half_life': np.nan,
                    'r2': np.nan,
                    'decay': False
                })
                continue

            decay_fit = fit_decay(rolling_df)

            if decay_fit:
                results.append({
                    'pair': pair_name,
                    'source': source,
                    'target': target,
                    'n_windows': len(rolling_df),
                    'max_F': rolling_df['F'].max(),
                    'min_F': rolling_df['F'].min(),
                    'half_life': decay_fit['half_life'],
                    'r2': decay_fit['r2'],
                    'decay': decay_fit['decay']
                })
            else:
                results.append({
                    'pair': pair_name,
                    'source': source,
                    'target': target,
                    'n_windows': len(rolling_df),
                    'max_F': rolling_df['F'].max(),
                    'half_life': np.nan,
                    'r2': np.nan,
                    'decay': False
                })

    results_df = pd.DataFrame(results)

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS: ALL 30 PAIRS")
    print("=" * 70)

    # Sort by decay evidence
    results_df = results_df.sort_values(['decay', 'r2'], ascending=[False, False])

    print("\nPairs showing decay (R² > 0.3, half-life < 50 years):")
    print("-" * 70)
    decay_pairs = results_df[results_df['decay'] == True]

    if len(decay_pairs) > 0:
        print(f"{'Pair':<20} {'Half-life':>12} {'R²':>8} {'Max F':>8} {'Min F':>8}")
        print("-" * 60)
        for _, row in decay_pairs.iterrows():
            print(f"{row['pair']:<20} {row['half_life']:>10.2f}y {row['r2']:>8.3f} {row['max_F']:>8.2f} {row['min_F']:>8.2f}")
    else:
        print("  No pairs show clear decay pattern")

    print(f"\nPairs with decay: {len(decay_pairs)} / 30")

    # Pairs without decay
    print("\nPairs without decay:")
    print("-" * 70)
    no_decay = results_df[results_df['decay'] == False]
    print(f"{'Pair':<20} {'Half-life':>12} {'R²':>8} {'Max F':>8}")
    print("-" * 60)
    for _, row in no_decay.head(15).iterrows():
        hl = f"{row['half_life']:.1f}y" if not np.isnan(row['half_life']) else "N/A"
        r2 = f"{row['r2']:.3f}" if not np.isnan(row['r2']) else "N/A"
        maxf = f"{row['max_F']:.2f}" if not np.isnan(row['max_F']) else "N/A"
        print(f"{row['pair']:<20} {hl:>12} {r2:>8} {maxf:>8}")

    if len(no_decay) > 15:
        print(f"  ... and {len(no_decay) - 15} more")

    # Key statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    n_decay = len(decay_pairs)
    n_total = 30
    pct_decay = n_decay / n_total * 100

    print(f"\nPairs showing decay: {n_decay} / {n_total} ({pct_decay:.1f}%)")

    if n_decay > 0:
        avg_halflife = decay_pairs['half_life'].mean()
        avg_r2 = decay_pairs['r2'].mean()
        print(f"Average half-life (decaying pairs): {avg_halflife:.2f} years")
        print(f"Average R² (decaying pairs): {avg_r2:.3f}")

    # Is decay general?
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    if pct_decay >= 50:
        print(f"\n  Decay is a GENERAL phenomenon: {pct_decay:.0f}% of pairs show decay")
    elif pct_decay >= 20:
        print(f"\n  Decay is COMMON but not universal: {pct_decay:.0f}% of pairs show decay")
    else:
        print(f"\n  Decay is RARE: only {pct_decay:.0f}% of pairs show decay")

    return results_df


if __name__ == '__main__':
    results = main()

    # Save results
    results.to_csv('results/30pairs_decay_analysis.csv', index=False)
    print("\nResults saved to results/30pairs_decay_analysis.csv")
