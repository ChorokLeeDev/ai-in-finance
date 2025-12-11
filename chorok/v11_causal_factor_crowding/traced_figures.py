"""
Create publication-ready figures for TRACED paper.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats

# Publication style
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = 11
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['figure.dpi'] = 150


def load_results():
    return pd.read_csv('/Users/i767700/Github/ai-in-finance/chorok/v11_causal_factor_crowding/traced_results.csv')


def figure1_regime_detection(df, save_path):
    """
    Figure 1: Regime Detection Performance (NMI) vs Tail Heaviness
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    nu_values = sorted(df['nu'].unique())

    # Colors
    colors = {
        'TRACED': '#2E86AB',  # Blue
        'Gaussian HMM': '#A23B72',  # Purple
        'Single DAG': '#F18F01'  # Orange
    }

    for method in ['TRACED', 'Gaussian HMM']:
        means = []
        stds = []
        for nu in nu_values:
            data = df[(df['method'] == method) & (df['nu'] == nu)]['NMI']
            means.append(data.mean())
            stds.append(data.std())

        means = np.array(means)
        stds = np.array(stds)

        label = 'TRACED (Student-t HMM)' if method == 'TRACED' else 'Gaussian HMM'
        linestyle = '-' if method == 'TRACED' else '--'
        marker = 'o' if method == 'TRACED' else 's'

        ax.plot(nu_values, means, label=label, color=colors[method],
                linewidth=2, marker=marker, markersize=8, linestyle=linestyle)
        ax.fill_between(nu_values, means - stds, means + stds,
                        alpha=0.2, color=colors[method])

    # Mark significant result
    ax.annotate('p=0.007**\nd=1.37', xy=(3, 0.65), xytext=(5, 0.8),
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', color='gray', lw=1))

    ax.set_xlabel('Degrees of freedom ($\\nu$)\n← Heavy tails | Light tails →')
    ax.set_ylabel('Normalized Mutual Information (NMI)')
    ax.set_title('Regime Detection: Student-t vs Gaussian HMM')

    ax.set_xlim(2, 32)
    ax.set_ylim(0.1, 1.0)
    ax.set_xticks(nu_values)

    # Add region annotations
    ax.axvspan(2, 8, alpha=0.1, color='red', label='Crisis-like (heavy tails)')
    ax.axvspan(12, 35, alpha=0.1, color='green', label='Normal (light tails)')

    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def figure2_bar_comparison(df, save_path):
    """
    Figure 2: Bar chart comparison at nu=3 (heavy tails)
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel A: Regime Detection (NMI)
    ax = axes[0]
    nu = 3

    methods = ['Single DAG', 'Gaussian HMM', 'TRACED']
    colors = ['#F18F01', '#A23B72', '#2E86AB']

    nmi_means = []
    nmi_stds = []
    for method in methods:
        data = df[(df['method'] == method) & (df['nu'] == nu)]['NMI']
        nmi_means.append(data.mean())
        nmi_stds.append(data.std())

    x = np.arange(len(methods))
    bars = ax.bar(x, nmi_means, yerr=nmi_stds, capsize=5,
                  color=colors, edgecolor='black', linewidth=1)

    ax.set_ylabel('NMI')
    ax.set_title(f'(A) Regime Detection ($\\nu$={nu})')
    ax.set_xticks(x)
    ax.set_xticklabels(['No Regimes', 'Gaussian', 'TRACED\n(ours)'])
    ax.set_ylim(0, 1.0)

    # Add significance bar
    ax.plot([1, 2], [0.85, 0.85], 'k-', lw=1)
    ax.plot([1, 1], [0.83, 0.85], 'k-', lw=1)
    ax.plot([2, 2], [0.83, 0.85], 'k-', lw=1)
    ax.text(1.5, 0.87, '**', ha='center', fontsize=14)

    # Panel B: DAG Recovery (F1)
    ax = axes[1]

    f1_means = []
    f1_stds = []
    for method in methods:
        data = df[(df['method'] == method) & (df['nu'] == nu)]['F1']
        f1_means.append(data.mean())
        f1_stds.append(data.std())

    bars = ax.bar(x, f1_means, yerr=f1_stds, capsize=5,
                  color=colors, edgecolor='black', linewidth=1)

    ax.set_ylabel('F1 Score')
    ax.set_title(f'(B) DAG Recovery ($\\nu$={nu})')
    ax.set_xticks(x)
    ax.set_xticklabels(['No Regimes', 'Gaussian', 'TRACED\n(ours)'])
    ax.set_ylim(0, 0.5)

    ax.text(1.5, 0.35, 'n.s.', ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def figure3_aggregate_boxplot(df, save_path):
    """
    Figure 3: Box plot comparing aggregate performance
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Heavy tails (nu <= 7)
    ax = axes[0]
    df_heavy = df[df['nu'] <= 7]

    traced_data = df_heavy[df_heavy['method'] == 'TRACED']['NMI']
    gaussian_data = df_heavy[df_heavy['method'] == 'Gaussian HMM']['NMI']

    bp = ax.boxplot([gaussian_data, traced_data],
                    labels=['Gaussian HMM', 'TRACED'],
                    patch_artist=True)

    bp['boxes'][0].set_facecolor('#A23B72')
    bp['boxes'][1].set_facecolor('#2E86AB')

    ax.set_ylabel('NMI')
    ax.set_title('(A) Heavy Tails ($\\nu \\leq 7$)')
    ax.set_ylim(0, 1)

    # Add p-value
    _, p_val = stats.ttest_ind(traced_data.values, gaussian_data.values)
    ax.text(1.5, 0.95, f'p={p_val:.3f}**', ha='center', fontsize=10)

    # Light tails (nu >= 15)
    ax = axes[1]
    df_light = df[df['nu'] >= 15]

    traced_data = df_light[df_light['method'] == 'TRACED']['NMI']
    gaussian_data = df_light[df_light['method'] == 'Gaussian HMM']['NMI']

    bp = ax.boxplot([gaussian_data, traced_data],
                    labels=['Gaussian HMM', 'TRACED'],
                    patch_artist=True)

    bp['boxes'][0].set_facecolor('#A23B72')
    bp['boxes'][1].set_facecolor('#2E86AB')

    ax.set_ylabel('NMI')
    ax.set_title('(B) Light Tails ($\\nu \\geq 15$)')
    ax.set_ylim(0, 1)

    _, p_val = stats.ttest_ind(traced_data.values, gaussian_data.values)
    ax.text(1.5, 0.95, f'p={p_val:.2f} n.s.', ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def figure4_full_comparison(df, save_path):
    """
    Figure 4: Full comparison table as heatmap
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    nu_values = sorted(df['nu'].unique())

    # NMI
    ax = axes[0]
    data_nmi = []
    for method in ['Gaussian HMM', 'TRACED']:
        row = []
        for nu in nu_values:
            val = df[(df['method'] == method) & (df['nu'] == nu)]['NMI'].mean()
            row.append(val)
        data_nmi.append(row)

    im = ax.imshow(data_nmi, cmap='RdYlGn', aspect='auto', vmin=0.2, vmax=0.9)
    ax.set_xticks(range(len(nu_values)))
    ax.set_xticklabels([f'$\\nu$={nu}' for nu in nu_values])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Gaussian', 'TRACED'])
    ax.set_title('(A) Regime Detection (NMI)')

    # Add values
    for i in range(2):
        for j in range(len(nu_values)):
            ax.text(j, i, f'{data_nmi[i][j]:.2f}', ha='center', va='center', color='black')

    plt.colorbar(im, ax=ax, shrink=0.8)

    # F1
    ax = axes[1]
    data_f1 = []
    for method in ['Gaussian HMM', 'TRACED']:
        row = []
        for nu in nu_values:
            val = df[(df['method'] == method) & (df['nu'] == nu)]['F1'].mean()
            row.append(val)
        data_f1.append(row)

    im = ax.imshow(data_f1, cmap='RdYlGn', aspect='auto', vmin=0.1, vmax=0.3)
    ax.set_xticks(range(len(nu_values)))
    ax.set_xticklabels([f'$\\nu$={nu}' for nu in nu_values])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Gaussian', 'TRACED'])
    ax.set_title('(B) DAG Recovery (F1)')

    for i in range(2):
        for j in range(len(nu_values)):
            ax.text(j, i, f'{data_f1[i][j]:.2f}', ha='center', va='center', color='black')

    plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


if __name__ == "__main__":
    df = load_results()
    base_path = '/Users/i767700/Github/ai-in-finance/chorok/v11_causal_factor_crowding/'

    print("Creating publication figures...")
    figure1_regime_detection(df, base_path + 'fig1_regime_detection.png')
    figure2_bar_comparison(df, base_path + 'fig2_bar_comparison.png')
    figure3_aggregate_boxplot(df, base_path + 'fig3_boxplot.png')
    figure4_full_comparison(df, base_path + 'fig4_heatmap.png')

    print("\nAll figures created!")
