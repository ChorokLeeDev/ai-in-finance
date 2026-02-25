"""
Generate all fit-dependent figures from selected HMM fit.
Reads regime assignments from selected_fit_regimes.csv and precomputed data from JSON.

Outputs:
  figures/regime_timeline.pdf
  figures/granger_heatmap.pdf
  figures/lag_sensitivity.pdf
  figures/rolling_granger.pdf
"""

import numpy as np
import pandas as pd
import json
import urllib.request
import zipfile
import io
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import f as f_dist

RESULTS_DIR = '/Users/i767700/Github/ai-in-finance/papers/causal_regimes/results'
FIGURES_DIR = '/Users/i767700/Github/ai-in-finance/papers/causal_regimes/figures'
REGIME_COLORS = {'Normal': '#2ecc71', 'Elevated': '#f1c40f', 'Crisis': '#e74c3c'}
FACTOR_COLS = ['MKT', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']


def load_ff_data():
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
    url_mom = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip'
    with urllib.request.urlopen(url_mom, timeout=60) as response:
        data = response.read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        with z.open(z.namelist()[0]) as f:
            mom = pd.read_csv(f, skiprows=13)
    mom.columns = mom.columns.str.strip()
    mom = mom.rename(columns={mom.columns[0]: 'Date', mom.columns[1]: 'MOM'})
    mom = mom[mom['Date'].astype(str).str.match(r'^\d{8}$')]
    mom['Date'] = pd.to_datetime(mom['Date'], format='%Y%m%d')
    mom['MOM'] = pd.to_numeric(mom['MOM'], errors='coerce')
    mom = mom.set_index('Date').dropna()
    df = df5.join(mom[['MOM']], how='inner')
    df = df.rename(columns={'Mkt-RF': 'MKT'}).drop('RF', axis=1, errors='ignore')
    return df.loc['1990-01-01':'2024-12-31']


def load_selected_regimes():
    df = pd.read_csv(f"{RESULTS_DIR}/selected_fit_regimes.csv")
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    return df


# =============================================================================
# FIGURE 1: REGIME TIMELINE
# =============================================================================
def generate_regime_timeline(ff_df, regime_df):
    print("  Generating regime_timeline.pdf...", flush=True)
    common = ff_df.index.intersection(regime_df.index).sort_values()
    df = ff_df.loc[common].copy()
    df['regime'] = regime_df.loc[common, 'regime_label']
    df['factor_norm'] = np.linalg.norm(df[FACTOR_COLS].values, axis=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5), height_ratios=[3, 1],
                                     sharex=True, gridspec_kw={'hspace': 0.05})

    # Top: factor norm with regime background
    dates = df.index
    regime_vals = df['regime'].values
    for i in range(len(dates) - 1):
        ax1.axvspan(dates[i], dates[i+1], alpha=0.3,
                    color=REGIME_COLORS.get(regime_vals[i], 'gray'), linewidth=0)

    ax1.plot(dates, df['factor_norm'].rolling(5).mean(), color='black', linewidth=0.3, alpha=0.7)
    ax1.set_ylabel('Factor Norm (5-day MA)')
    ax1.set_ylim(0, df['factor_norm'].quantile(0.999) * 1.1)

    # Crisis events
    events = [('2001-03-01', 'Dot-com'), ('2008-09-15', 'Lehman'),
              ('2011-08-01', 'EU Debt'), ('2015-08-24', 'China'),
              ('2018-02-05', 'Vol'), ('2020-03-16', 'COVID'),
              ('2022-06-15', 'Rates')]
    for date, label in events:
        try:
            ax1.axvline(pd.Timestamp(date), color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
            ax1.text(pd.Timestamp(date), ax1.get_ylim()[1] * 0.95, label,
                     fontsize=6, rotation=90, va='top', ha='right')
        except Exception:
            pass

    # Bottom: regime assignments
    regime_map = {'Normal': 0, 'Elevated': 1, 'Crisis': 2}
    regime_numeric = np.array([regime_map.get(r, 0) for r in regime_vals])
    for i in range(len(dates) - 1):
        ax2.axvspan(dates[i], dates[i+1],
                    color=REGIME_COLORS.get(regime_vals[i], 'gray'), alpha=0.8, linewidth=0)
    ax2.set_ylabel('Regime')
    ax2.set_yticks([])
    ax2.xaxis.set_major_locator(mdates.YearLocator(5))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=REGIME_COLORS[r], alpha=0.6, label=r) for r in ['Normal', 'Elevated', 'Crisis']]
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=8, framealpha=0.8)

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/regime_timeline.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    print("    Done.", flush=True)


# =============================================================================
# FIGURE 2: GRANGER HEATMAP
# =============================================================================
def generate_granger_heatmap(hmm_json):
    print("  Generating granger_heatmap.pdf...", flush=True)
    all_pairs = hmm_json['selected_fit']['all_pairs']
    bonferroni = 0.01 / 30  # 0.000333

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    regime_names = ['Normal', 'Elevated', 'Crisis']

    for ax_idx, regime_name in enumerate(regime_names):
        ax = axes[ax_idx]
        pairs_data = all_pairs[regime_name]['pairs']

        matrix = np.zeros((6, 6))
        for i, src in enumerate(FACTOR_COLS):
            for j, tgt in enumerate(FACTOR_COLS):
                if i == j:
                    matrix[i, j] = 0
                else:
                    key = f"{src}->{tgt}"
                    p = pairs_data.get(key, {}).get('p_value', 1.0)
                    matrix[i, j] = -np.log10(max(p, 1e-15))

        im = ax.imshow(matrix, cmap='YlOrRd', vmin=0, vmax=8, aspect='auto')

        # Mark significant cells
        for i in range(6):
            for j in range(6):
                if i == j:
                    continue
                key = f"{FACTOR_COLS[i]}->{FACTOR_COLS[j]}"
                p = pairs_data.get(key, {}).get('p_value', 1.0)
                if p < bonferroni:
                    ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                                                fill=False, edgecolor='black', linewidth=2))
                    ax.text(j, i, '*', ha='center', va='center', fontsize=10, fontweight='bold')

        ax.set_xticks(range(6))
        ax.set_xticklabels(FACTOR_COLS, fontsize=8, rotation=45)
        ax.set_yticks(range(6))
        ax.set_yticklabels(FACTOR_COLS, fontsize=8)
        ax.set_title(f'{regime_name}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Target')
        if ax_idx == 0:
            ax.set_ylabel('Source')

    fig.colorbar(im, ax=axes, label='$-\\log_{10}(p)$', shrink=0.8)
    fig.suptitle('All-Pairs Granger Causality by Regime (lag=5)', fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/granger_heatmap.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    print("    Done.", flush=True)


# =============================================================================
# FIGURE 3: LAG SENSITIVITY
# =============================================================================
def generate_lag_sensitivity(hmm_json):
    print("  Generating lag_sensitivity.pdf...", flush=True)
    lag_data = hmm_json['selected_fit']['lag_sensitivity']
    bonferroni = 0.01 / 30

    fig, ax = plt.subplots(figsize=(8, 5))
    lags = range(1, 16)
    regime_colors_line = {'Normal': '#2ecc71', 'Elevated': '#f1c40f', 'Crisis': '#e74c3c'}

    for regime_name in ['Normal', 'Elevated', 'Crisis']:
        rdata = lag_data[regime_name]
        neg_log_ps = []
        for lag in lags:
            p = rdata.get(str(lag), {}).get('p_value', 1.0)
            neg_log_ps.append(-np.log10(max(p, 1e-15)))
        ax.plot(lags, neg_log_ps, 'o-', label=regime_name,
                color=regime_colors_line[regime_name], linewidth=2, markersize=5)

    ax.axhline(-np.log10(0.05), color='gray', linestyle='--', alpha=0.5, label='p=0.05')
    ax.axhline(-np.log10(bonferroni), color='black', linestyle=':', alpha=0.5, label=f'Bonferroni ({bonferroni:.1e})')
    ax.set_xlabel('Lag')
    ax.set_ylabel('$-\\log_{10}(p)$')
    ax.set_title('HML→SMB Granger Causality: Lag Sensitivity by Regime')
    ax.legend(fontsize=9)
    ax.set_xticks(list(lags))
    ax.set_xlim(0.5, 15.5)

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/lag_sensitivity.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    print("    Done.", flush=True)


# =============================================================================
# FIGURE 4: ROLLING GRANGER
# =============================================================================
def generate_rolling_granger(ff_df, regime_df):
    print("  Generating rolling_granger.pdf...", flush=True)
    common = ff_df.index.intersection(regime_df.index).sort_values()
    df = ff_df.loc[common].copy()
    df['regime'] = regime_df.loc[common, 'regime_label']

    hml = df['HML'].values
    smb = df['SMB'].values
    dates = df.index
    window = 756  # ~3 years
    step = 5
    lag = 9

    roll_dates = []
    roll_neg_logp = []

    for start in range(0, len(df) - window, step):
        end = start + window
        hml_w = hml[start:end]
        smb_w = smb[start:end]
        n = len(hml_w)
        if n < lag * 4:
            continue
        y = smb_w[lag:]
        y_lagged = np.column_stack([smb_w[lag-i-1:-i-1] for i in range(lag)])
        x_lagged = np.column_stack([hml_w[lag-i-1:-i-1] for i in range(lag)])
        X_r = np.column_stack([np.ones(len(y)), y_lagged])
        X_u = np.column_stack([np.ones(len(y)), y_lagged, x_lagged])
        try:
            beta_r = np.linalg.lstsq(X_r, y, rcond=None)[0]
            beta_u = np.linalg.lstsq(X_u, y, rcond=None)[0]
            rss_r = np.sum((y - X_r @ beta_r) ** 2)
            rss_u = np.sum((y - X_u @ beta_u) ** 2)
            df1, df2 = lag, len(y) - 2 * lag - 1
            if df2 > 0 and rss_u > 0:
                f_stat = ((rss_r - rss_u) / df1) / (rss_u / df2)
                p_value = 1 - f_dist.cdf(f_stat, df1, df2)
                roll_dates.append(dates[end - 1])
                roll_neg_logp.append(-np.log10(max(p_value, 1e-15)))
        except Exception:
            continue

    fig, ax = plt.subplots(figsize=(12, 5))

    # Regime background
    regime_vals = df['regime'].values
    for i in range(len(dates) - 1):
        ax.axvspan(dates[i], dates[i+1], alpha=0.2,
                   color=REGIME_COLORS.get(regime_vals[i], 'gray'), linewidth=0)

    ax.plot(roll_dates, roll_neg_logp, color='darkblue', linewidth=1)
    ax.axhline(-np.log10(0.05), color='gray', linestyle='--', alpha=0.7, label='p=0.05')
    ax.axhline(-np.log10(0.001), color='black', linestyle=':', alpha=0.5, label='p=0.001')
    ax.set_ylabel('$-\\log_{10}(p)$')
    ax.set_title('Rolling 3-Year Granger Causality: HML→SMB (lag=9)')
    ax.legend(fontsize=9)
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/rolling_granger.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    print("    Done.", flush=True)


# =============================================================================
# MAIN
# =============================================================================
def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print("=" * 70)
    print("GENERATING ALL FIT-DEPENDENT FIGURES")
    print("=" * 70, flush=True)

    ff_df = load_ff_data()
    regime_df = load_selected_regimes()

    with open(f"{RESULTS_DIR}/multistart_hmm_results.json") as f:
        hmm_json = json.load(f)

    generate_regime_timeline(ff_df, regime_df)
    generate_granger_heatmap(hmm_json)
    generate_lag_sensitivity(hmm_json)
    generate_rolling_granger(ff_df, regime_df)

    print("\nAll figures generated.", flush=True)


if __name__ == '__main__':
    main()
