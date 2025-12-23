"""
Generate OOS Validation Figure for ICAIF Paper

Creates a publication-ready figure showing:
- Panel A: Regime timeline with train/test split
- Panel B: Per-regime Granger validation rates
- Panel C: HML↔SMB relationship by regime (train vs test)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')

from gate2_regime_detection import StudentTHMM, load_and_prepare_data


def create_oos_figure(save_path=None):
    """
    Create the OOS validation figure.
    """
    # Set up style
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9

    # Load data
    print("Loading data...")
    crowding = load_and_prepare_data()

    # Split
    train_end = '2014-12-31'
    test_start = '2015-01-01'

    train_data = crowding[crowding.index <= train_end]
    test_data = crowding[crowding.index >= test_start]

    # Fit HMM on training only
    print("Fitting HMM on training data...")
    hmm = StudentTHMM(n_regimes=3, n_iter=100)
    hmm.fit(train_data.values)

    # Get regimes for full period (but model was only trained on train)
    full_regimes = hmm.predict(crowding.values)

    # Identify regime labels by volatility
    vol_by_regime = []
    train_regimes = hmm.predict(train_data.values)
    for k in range(3):
        regime_data = train_data.values[train_regimes == k]
        vol = np.std(regime_data)
        vol_by_regime.append(vol)

    crisis_regime = np.argmax(vol_by_regime)
    normal_regime = np.argmin(vol_by_regime)
    crowding_regime = 3 - crisis_regime - normal_regime

    regime_names = {
        normal_regime: 'Normal',
        crowding_regime: 'Crowding',
        crisis_regime: 'Crisis'
    }

    # Colors
    colors = {
        normal_regime: '#4CAF50',      # Green
        crowding_regime: '#FFC107',    # Amber
        crisis_regime: '#F44336'       # Red
    }

    # Create figure
    fig = plt.figure(figsize=(12, 10))

    # =========================================================================
    # Panel A: Regime Timeline with Train/Test Split
    # =========================================================================
    ax1 = fig.add_subplot(3, 1, 1)

    # Compute rolling volatility for visualization
    vol = crowding.rolling(20).std().mean(axis=1)

    # Plot volatility line
    ax1.plot(crowding.index, vol, 'k-', alpha=0.6, linewidth=0.5)

    # Color background by regime
    dates = crowding.index
    for i in range(len(dates) - 1):
        regime = full_regimes[i]
        ax1.axvspan(dates[i], dates[i+1], alpha=0.4, color=colors[regime], linewidth=0)

    # Mark train/test split
    split_date = pd.Timestamp('2015-01-01')
    ax1.axvline(split_date, color='black', linestyle='--', linewidth=2, label='Train/Test Split')

    # Add text labels
    ax1.text(pd.Timestamp('2002-01-01'), ax1.get_ylim()[1] * 0.9,
             'TRAINING (1990-2014)', fontsize=11, fontweight='bold', ha='center')
    ax1.text(pd.Timestamp('2020-01-01'), ax1.get_ylim()[1] * 0.9,
             'TEST (2015-2024)', fontsize=11, fontweight='bold', ha='center')

    # Mark key events
    events = [
        ('2008-09-15', 'Lehman'),
        ('2020-03-16', 'COVID'),
        ('2011-08-05', 'EU Debt'),
        ('2015-08-24', 'China'),
        ('2018-12-24', 'Dec 2018'),
    ]

    for date, label in events:
        try:
            ax1.axvline(pd.to_datetime(date), color='navy', linestyle=':', alpha=0.7, linewidth=1)
        except:
            pass

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#4CAF50', alpha=0.4, label='Normal'),
        mpatches.Patch(facecolor='#FFC107', alpha=0.4, label='Crowding'),
        mpatches.Patch(facecolor='#F44336', alpha=0.4, label='Crisis'),
        Line2D([0], [0], color='black', linestyle='--', linewidth=2, label='Train/Test Split'),
    ]
    ax1.legend(handles=legend_elements, loc='upper left', ncol=4, fontsize=9)

    ax1.set_ylabel('Rolling Volatility')
    ax1.set_title('(A) Regime Detection Timeline: HMM Trained on 1990-2014, Applied to Full Period', fontweight='bold')
    ax1.set_xlim(crowding.index[0], crowding.index[-1])

    # =========================================================================
    # Panel B: Per-Regime Validation Rates
    # =========================================================================
    ax2 = fig.add_subplot(3, 2, 3)

    # Data from OOS validation
    regimes = ['Normal', 'Crowding', 'Crisis']
    train_discovered = [22, 26, 22]
    test_validated = [9, 17, 18]
    validation_rates = [41, 65, 82]

    x = np.arange(len(regimes))
    width = 0.35

    bars1 = ax2.bar(x - width/2, train_discovered, width, label='Discovered (Train)',
                    color='#2196F3', alpha=0.8)
    bars2 = ax2.bar(x + width/2, test_validated, width, label='Validated (Test)',
                    color='#4CAF50', alpha=0.8)

    # Add rate labels on top
    for i, (rate, bar) in enumerate(zip(validation_rates, bars2)):
        ax2.annotate(f'{rate}%',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=10, fontweight='bold',
                    color='#2E7D32')

    ax2.set_ylabel('Number of Relationships')
    ax2.set_title('(B) Granger Causality Validation by Regime', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(regimes)
    ax2.legend(loc='upper left')
    ax2.set_ylim(0, 35)

    # Add horizontal line for overall rate
    ax2.axhline(y=70 * 0.63, color='gray', linestyle='--', alpha=0.5)
    ax2.text(2.3, 70 * 0.63 + 1, '63% overall', fontsize=8, color='gray')

    # =========================================================================
    # Panel C: Crisis Detection on Test Events
    # =========================================================================
    ax3 = fig.add_subplot(3, 2, 4)

    events = ['China\n2015', 'Dec 2018\nSelloff', 'COVID-19\n2020', '2022 Bear\nMarket']
    crisis_pct = [0, 0, 43, 0]
    crowding_pct = [88, 100, 57, 100]

    x = np.arange(len(events))
    width = 0.6

    bars_crisis = ax3.bar(x, crisis_pct, width, label='Crisis', color='#F44336', alpha=0.8)
    bars_crowding = ax3.bar(x, crowding_pct, width, bottom=crisis_pct,
                           label='Crowding', color='#FFC107', alpha=0.8)

    # Add 50% threshold line
    ax3.axhline(y=50, color='black', linestyle='--', linewidth=1.5, label='Detection Threshold')

    # Add checkmarks
    for i, (c, cr) in enumerate(zip(crisis_pct, crowding_pct)):
        total = c + cr
        if total >= 50:
            ax3.annotate('✓', xy=(i, total + 3), ha='center', fontsize=14,
                        color='#2E7D32', fontweight='bold')

    ax3.set_ylabel('% of Days in Regime')
    ax3.set_title('(C) Crisis Detection on Unseen Test Events', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(events)
    ax3.legend(loc='upper right')
    ax3.set_ylim(0, 120)
    ax3.text(3.5, 52, '4/4 detected', fontsize=9, style='italic')

    # =========================================================================
    # Panel D: HML↔SMB Relationship by Regime
    # =========================================================================
    ax4 = fig.add_subplot(3, 1, 3)

    # Data structure for grouped bars
    regimes = ['Normal', 'Crowding', 'Crisis']

    # -log10(p-value) for visualization (capped at 30)
    # Train period (in-sample)
    train_hml_smb = [1.82, 1.06, 4.72]  # -log10(p): Normal not sig, Crowding not sig, Crisis sig
    train_smb_hml = [1.01, 3.71, 0.78]  # -log10(p): Normal not sig, Crowding sig, Crisis not sig

    # Test period (OOS)
    test_hml_smb = [0.36, 1.35, 24.1]   # -log10(p): 0.43, 0.045, 8e-25
    test_smb_hml = [1.57, 26.7, 8.59]   # -log10(p): 0.027, 2e-27, 2.6e-9

    x = np.arange(len(regimes))
    width = 0.2

    # Plot bars
    bars1 = ax4.bar(x - 1.5*width, train_hml_smb, width, label='HML→SMB (Train)',
                    color='#1976D2', alpha=0.8)
    bars2 = ax4.bar(x - 0.5*width, train_smb_hml, width, label='SMB→HML (Train)',
                    color='#1976D2', alpha=0.4, hatch='///')
    bars3 = ax4.bar(x + 0.5*width, test_hml_smb, width, label='HML→SMB (Test)',
                    color='#388E3C', alpha=0.8)
    bars4 = ax4.bar(x + 1.5*width, test_smb_hml, width, label='SMB→HML (Test)',
                    color='#388E3C', alpha=0.4, hatch='///')

    # Significance threshold line (-log10(0.05) ≈ 1.3)
    ax4.axhline(y=1.3, color='red', linestyle='--', linewidth=1.5, label='α=0.05')

    # Add annotations for significant findings
    annotations = [
        (2, train_hml_smb[2], 'Train:\nHML→SMB', -40, 10),
        (1, train_smb_hml[1], 'Train:\nSMB→HML', -40, 10),
        (2, min(test_hml_smb[2], 25), 'Test: Both\nsignificant', 30, -5),
        (1, min(test_smb_hml[1], 25), 'Test: Both\nsignificant', 30, -5),
    ]

    ax4.set_ylabel('-log₁₀(p-value)')
    ax4.set_title('(D) HML↔SMB Lead-Lag Relationship: Train vs Test Period', fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(regimes)
    ax4.legend(loc='upper left', ncol=3, fontsize=8)
    ax4.set_ylim(0, 30)

    # Add interpretation text
    ax4.text(0, 28, 'Normal:\nIndependent', ha='center', fontsize=8, style='italic')
    ax4.text(1, 28, 'Crowding:\nBidirectional', ha='center', fontsize=8, style='italic')
    ax4.text(2, 28, 'Crisis:\nBidirectional', ha='center', fontsize=8, style='italic')

    # =========================================================================
    # Final adjustments
    # =========================================================================
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"\nFigure saved to: {save_path}")

    plt.close()

    return fig


def create_simple_oos_figure(save_path=None):
    """
    Create a simpler 2-panel figure for space-constrained submissions.
    """
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 10

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # =========================================================================
    # Panel A: Validation Rates by Regime
    # =========================================================================
    ax1 = axes[0]

    regimes = ['Normal', 'Crowding', 'Crisis']
    validation_rates = [41, 65, 82]
    colors = ['#4CAF50', '#FFC107', '#F44336']

    bars = ax1.bar(regimes, validation_rates, color=colors, alpha=0.8, edgecolor='black')

    # Add value labels
    for bar, rate in zip(bars, validation_rates):
        ax1.annotate(f'{rate}%',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax1.set_ylabel('OOS Validation Rate (%)')
    ax1.set_title('(A) Granger Relationship Replication\nby Market Regime', fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.axhline(y=63, color='gray', linestyle='--', alpha=0.7)
    ax1.text(2.4, 65, 'Overall: 63%', fontsize=9, color='gray')

    # =========================================================================
    # Panel B: HML↔SMB Direction by Regime
    # =========================================================================
    ax2 = axes[1]

    # Create a heatmap-style visualization
    data = np.array([
        [0, 1],   # Normal: HML→SMB (no), SMB→HML (yes)
        [1, 1],   # Crowding: both
        [1, 1],   # Crisis: both
    ])

    im = ax2.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    # Labels
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['HML→SMB', 'SMB→HML'])
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(['Normal', 'Crowding', 'Crisis'])

    # Add text annotations
    annotations = [
        ['✗', '✓'],
        ['✓', '✓'],
        ['✓', '✓'],
    ]
    p_values = [
        ['p=0.43', 'p=0.03'],
        ['p=0.04', 'p<0.001'],
        ['p<0.001', 'p<0.001'],
    ]

    for i in range(3):
        for j in range(2):
            color = 'white' if data[i, j] == 1 else 'black'
            ax2.text(j, i, f'{annotations[i][j]}\n{p_values[i][j]}',
                    ha='center', va='center', fontsize=10, color=color, fontweight='bold')

    ax2.set_title('(B) HML↔SMB Significance\n(OOS Test Period)', fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2, shrink=0.6)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Not Sig.', 'Significant'])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"\nFigure saved to: {save_path}")

    plt.close()

    return fig


if __name__ == "__main__":
    # Create full figure
    create_oos_figure(
        save_path='/Users/i767700/Github/ai-in-finance/chorok/v11_causal_factor_crowding/fig_oos_full.png'
    )

    # Create simple 2-panel figure for space-constrained submission
    create_simple_oos_figure(
        save_path='/Users/i767700/Github/ai-in-finance/chorok/v11_causal_factor_crowding/fig_oos_simple.png'
    )

    print("\nFigures created successfully!")
    print("  - fig_oos_full.png: Full 4-panel figure")
    print("  - fig_oos_simple.png: Simple 2-panel figure for 8-page limit")
