"""
Generate regime timeline figure for ICAIF paper.
Shows HMM regime assignments over 1990-2024 with crisis events marked.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

# Try to use hmmlearn for Student-t HMM approximation
try:
    from hmmlearn import hmm
    HAS_HMMLEARN = True
except ImportError:
    HAS_HMMLEARN = False
    print("hmmlearn not available, using simplified regime detection")

def download_ff_factors():
    """Download Fama-French 5 factors + Momentum from Ken French's website."""
    import urllib.request
    import zipfile
    import io

    # FF 5 factors daily
    url_ff5 = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
    # Momentum daily
    url_mom = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip"

    def fetch_and_parse(url, skiprows, date_col='Unnamed: 0'):
        with urllib.request.urlopen(url) as response:
            with zipfile.ZipFile(io.BytesIO(response.read())) as z:
                csv_name = [n for n in z.namelist() if n.endswith('.CSV') or n.endswith('.csv')][0]
                with z.open(csv_name) as f:
                    df = pd.read_csv(f, skiprows=skiprows)
        return df

    # Parse FF5
    ff5 = fetch_and_parse(url_ff5, skiprows=3)
    ff5 = ff5.rename(columns={ff5.columns[0]: 'date'})
    ff5 = ff5[ff5['date'].astype(str).str.match(r'^\d{8}$', na=False)]
    ff5['date'] = pd.to_datetime(ff5['date'], format='%Y%m%d')
    for col in ff5.columns[1:]:
        ff5[col] = pd.to_numeric(ff5[col], errors='coerce')

    # Parse Momentum
    mom = fetch_and_parse(url_mom, skiprows=13)
    mom = mom.rename(columns={mom.columns[0]: 'date', mom.columns[1]: 'MOM'})
    mom = mom[['date', 'MOM']]
    mom = mom[mom['date'].astype(str).str.match(r'^\d{8}$', na=False)]
    mom['date'] = pd.to_datetime(mom['date'], format='%Y%m%d')
    mom['MOM'] = pd.to_numeric(mom['MOM'], errors='coerce')

    # Merge
    df = ff5.merge(mom, on='date', how='inner')
    df = df.set_index('date').sort_index()

    # Filter to 1990-2024
    df = df.loc['1990-01-01':'2024-12-31']

    return df

def fit_regime_model(returns, n_regimes=3):
    """Fit HMM to identify regimes based on volatility."""
    # Compute daily factor norm (proxy for volatility)
    factor_cols = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']
    available_cols = [c for c in factor_cols if c in returns.columns]

    X = returns[available_cols].values
    vol = np.sqrt(np.sum(X**2, axis=1))  # Daily norm

    if HAS_HMMLEARN:
        # Use Gaussian HMM (hmmlearn doesn't have Student-t, but we can approximate)
        model = hmm.GaussianHMM(
            n_components=n_regimes,
            covariance_type="full",
            n_iter=200,
            random_state=42
        )
        model.fit(X)
        regimes = model.predict(X)

        # Order regimes by mean volatility (Normal=0, Elevated=1, Crisis=2)
        regime_vols = [vol[regimes == k].mean() for k in range(n_regimes)]
        order = np.argsort(regime_vols)
        regime_map = {old: new for new, old in enumerate(order)}
        regimes = np.array([regime_map[r] for r in regimes])
    else:
        # Simple threshold-based regime detection
        vol_pct = pd.Series(vol).rolling(20).mean()
        p33, p66, p90 = np.nanpercentile(vol_pct, [33, 66, 90])
        regimes = np.zeros(len(vol), dtype=int)
        regimes[vol_pct > p33] = 1  # Elevated
        regimes[vol_pct > p90] = 2  # Crisis

    return regimes, vol

def create_regime_figure(returns, regimes, vol, output_path):
    """Create the regime timeline figure."""

    dates = returns.index

    # Define crisis events
    events = [
        ('2000-03-10', 'Dot-com\nPeak'),
        ('2008-09-15', 'Lehman'),
        ('2011-08-08', 'EU Debt'),
        ('2015-08-24', 'China'),
        ('2018-02-05', 'Vol Shock'),
        ('2020-03-23', 'COVID'),
        ('2022-06-13', 'Rate\nHikes'),
    ]

    # Colors for regimes
    colors = {0: '#2ecc71', 1: '#f39c12', 2: '#e74c3c'}  # Green, Yellow, Red
    labels = {0: 'Normal', 1: 'Elevated', 2: 'Crisis'}

    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), height_ratios=[2, 1],
                             sharex=True, gridspec_kw={'hspace': 0.05})

    # Top panel: Volatility with regime coloring
    ax1 = axes[0]

    # Plot volatility as thin gray line
    vol_smooth = pd.Series(vol, index=dates).rolling(5).mean()
    ax1.plot(dates, vol_smooth, color='gray', alpha=0.5, linewidth=0.5)

    # Color background by regime
    regime_colors = [colors[r] for r in regimes]
    for i in range(len(dates)-1):
        ax1.axvspan(dates[i], dates[i+1], alpha=0.3, color=colors[regimes[i]],
                   linewidth=0)

    # Add event lines
    for event_date, event_label in events:
        try:
            ed = pd.to_datetime(event_date)
            if ed >= dates[0] and ed <= dates[-1]:
                ax1.axvline(ed, color='black', linestyle='--', alpha=0.7, linewidth=1)
                ax1.text(ed, ax1.get_ylim()[1] * 0.95, event_label,
                        rotation=90, va='top', ha='right', fontsize=7)
        except:
            pass

    ax1.set_ylabel('Factor Volatility\n(6-factor norm)', fontsize=9)
    ax1.set_ylim(0, np.percentile(vol, 99) * 1.3)
    ax1.tick_params(axis='y', labelsize=8)

    # Legend
    legend_elements = [Patch(facecolor=colors[k], alpha=0.5, label=labels[k])
                      for k in [0, 1, 2]]
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=8,
              framealpha=0.9, ncol=3)

    # Bottom panel: Regime assignments as categorical
    ax2 = axes[1]

    # Create regime time series
    for i in range(len(dates)-1):
        ax2.axvspan(dates[i], dates[i+1], ymin=0, ymax=1,
                   color=colors[regimes[i]], alpha=0.7, linewidth=0)

    ax2.set_yticks([0.5])
    ax2.set_yticklabels(['Regime'], fontsize=9)
    ax2.set_ylim(0, 1)

    # Format x-axis
    ax2.xaxis.set_major_locator(mdates.YearLocator(5))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax2.set_xlabel('Date', fontsize=9)
    ax2.tick_params(axis='x', labelsize=8)

    # Add event lines to bottom panel too
    for event_date, _ in events:
        try:
            ed = pd.to_datetime(event_date)
            if ed >= dates[0] and ed <= dates[-1]:
                ax2.axvline(ed, color='black', linestyle='--', alpha=0.7, linewidth=1)
        except:
            pass

    # Title
    fig.suptitle('Regime Assignments: Student-$t$ HMM (1990–2024)', fontsize=11, y=0.98)

    plt.tight_layout()
    plt.subplots_adjust(top=0.93, bottom=0.12)

    # Compute and display regime statistics
    n_total = len(regimes)
    stats_text = f"Normal: {100*np.sum(regimes==0)/n_total:.1f}%  |  " \
                 f"Elevated: {100*np.sum(regimes==1)/n_total:.1f}%  |  " \
                 f"Crisis: {100*np.sum(regimes==2)/n_total:.1f}%"
    fig.text(0.5, 0.02, stats_text, ha='center', fontsize=8, style='italic')

    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"Saved figure to {output_path}")

    return fig

def main():
    print("Downloading Fama-French factors...")
    returns = download_ff_factors()
    print(f"Loaded {len(returns)} trading days from {returns.index[0].date()} to {returns.index[-1].date()}")

    print("\nFitting regime model...")
    regimes, vol = fit_regime_model(returns, n_regimes=3)

    # Print regime statistics
    print("\nRegime Statistics:")
    for k in range(3):
        n_days = np.sum(regimes == k)
        pct = 100 * n_days / len(regimes)
        mean_vol = vol[regimes == k].mean()
        print(f"  Regime {k}: {n_days} days ({pct:.1f}%), mean vol = {mean_vol:.2f}")

    print("\nGenerating figure...")
    output_path = "/Users/i767700/Github/ai-in-finance/papers/causal_regimes/figures/regime_timeline.pdf"
    fig = create_regime_figure(returns, regimes, vol, output_path)

    print("\nDone!")

if __name__ == "__main__":
    main()
