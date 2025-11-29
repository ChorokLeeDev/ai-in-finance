"""
Generate Figures for NeurIPS Paper
==================================

Creates all figures needed for the RelUQ paper.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import json
import os

# Set style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.dpi'] = 150

FIGURES_DIR = '/Users/i767700/Github/ai-in-finance/paper/figures'
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_ablation_results():
    """Load ablation study results."""
    with open('/Users/i767700/Github/ai-in-finance/experiments/ablation/results.json', 'r') as f:
        return json.load(f)


def fig1_overview():
    """Create overview diagram of RelUQ pipeline."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis('off')

    # Boxes
    boxes = [
        (0.5, 1.5, 1.8, 1, 'Relational\nDatabase', '#E8F4FD'),
        (3, 1.5, 1.8, 1, 'FK-aware\nFeatures', '#FDF4E8'),
        (5.5, 1.5, 1.8, 1, 'Ensemble\nModels', '#E8FDE8'),
        (8, 1.5, 1.8, 1, 'FK-level\nAttribution', '#FDE8E8'),
    ]

    for x, y, w, h, text, color in boxes:
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.05,rounding_size=0.1",
            facecolor=color, edgecolor='black', linewidth=1.5
        )
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=10, fontweight='bold')

    # Arrows
    arrow_style = dict(arrowstyle='->', color='black', lw=1.5)
    for i in range(3):
        x_start = boxes[i][0] + boxes[i][2]
        x_end = boxes[i+1][0]
        y = boxes[i][1] + boxes[i][3]/2
        ax.annotate('', xy=(x_end, y), xytext=(x_start, y),
                    arrowprops=arrow_style)

    # Labels below
    labels = ['Schema', 'col_to_fk mapping', 'Variance estimation', 'Permutation']
    positions = [1.4, 3.9, 6.4, 8.9]
    for pos, label in zip(positions, labels):
        ax.text(pos, 1.2, label, ha='center', va='top', fontsize=8, style='italic', color='gray')

    # Title
    ax.set_title('RelUQ: Relational Uncertainty Quantification Pipeline', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/fig1_overview.pdf', bbox_inches='tight')
    plt.savefig(f'{FIGURES_DIR}/fig1_overview.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Created: fig1_overview.pdf")


def fig2_baseline_comparison():
    """Create baseline comparison bar chart."""
    methods = ['Feature\n(24 groups)', 'Correlation\n(5 groups)', 'Random\n(5 groups)', 'RelUQ (FK)\n(5 groups)']
    stability = [0.956, 0.933, -0.400, 0.933]
    actionable = [False, False, False, True]

    colors = ['#FFB3B3' if not a else '#B3FFB3' for a in actionable]

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(methods))
    bars = ax.bar(x, stability, color=colors, edgecolor='black', linewidth=1.5)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, stability)):
        height = bar.get_height()
        va = 'bottom' if height >= 0 else 'top'
        offset = 0.02 if height >= 0 else -0.02
        ax.text(bar.get_x() + bar.get_width()/2, height + offset, f'{val:.3f}',
                ha='center', va=va, fontweight='bold', fontsize=11)

    ax.set_ylabel('Attribution Stability (Spearman ρ)', fontsize=12)
    ax.set_xlabel('Method', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylim(-0.6, 1.2)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax.axhline(y=0.8, color='green', linestyle=':', linewidth=1, alpha=0.5)
    ax.text(3.5, 0.82, 'Target: ρ ≥ 0.8', fontsize=8, color='green')

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#B3FFB3', edgecolor='black', label='Actionable'),
        mpatches.Patch(facecolor='#FFB3B3', edgecolor='black', label='Not Actionable')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    ax.set_title('Baseline Comparison: Stability and Actionability', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/fig2_baseline_comparison.pdf', bbox_inches='tight')
    plt.savefig(f'{FIGURES_DIR}/fig2_baseline_comparison.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Created: fig2_baseline_comparison.pdf")


def fig3_ablation_ensemble_size():
    """Ablation: Ensemble size K."""
    results = load_ablation_results()

    K_values = [int(k) for k in results['K'].keys()]
    stability = [results['K'][str(k)]['stability'] for k in K_values]

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(K_values, stability, 'o-', color='#2E86AB', linewidth=2, markersize=8)
    ax.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='Threshold (0.9)')

    ax.set_xlabel('Ensemble Size (K)', fontsize=12)
    ax.set_ylabel('Attribution Stability (ρ)', fontsize=12)
    ax.set_title('Sensitivity to Ensemble Size', fontsize=14, fontweight='bold')
    ax.set_ylim(0.7, 1.05)
    ax.set_xticks(K_values)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    # Annotation
    ax.annotate('K=3: Unstable\n(ρ=0.83)', xy=(3, 0.833), xytext=(4.5, 0.78),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=9, color='red')
    ax.annotate('K≥5: Stable\n(ρ≥0.93)', xy=(5, 0.933), xytext=(6.5, 0.87),
                arrowprops=dict(arrowstyle='->', color='green'),
                fontsize=9, color='green')

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/fig3_ablation_K.pdf', bbox_inches='tight')
    plt.savefig(f'{FIGURES_DIR}/fig3_ablation_K.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Created: fig3_ablation_K.pdf")


def fig4_ablation_sample_size():
    """Ablation: Sample size n."""
    results = load_ablation_results()

    n_values = [int(k) for k in results['n'].keys()]
    stability = [results['n'][str(n)]['stability'] for n in n_values]

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(n_values, stability, 's-', color='#E94F37', linewidth=2, markersize=8)
    ax.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='Threshold (0.9)')

    ax.set_xlabel('Sample Size (n)', fontsize=12)
    ax.set_ylabel('Attribution Stability (ρ)', fontsize=12)
    ax.set_title('Sensitivity to Sample Size', fontsize=14, fontweight='bold')
    ax.set_ylim(0.7, 1.05)
    ax.set_xscale('log')
    ax.set_xticks(n_values)
    ax.set_xticklabels([str(n) for n in n_values])
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    # Annotation
    ax.annotate('n=500: Unstable\n(ρ=0.80)', xy=(500, 0.80), xytext=(800, 0.75),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=9, color='red')

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/fig4_ablation_n.pdf', bbox_inches='tight')
    plt.savefig(f'{FIGURES_DIR}/fig4_ablation_n.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Created: fig4_ablation_n.pdf")


def fig5_ablation_subsampling():
    """Ablation: Subsampling rate and its effect on base uncertainty."""
    results = load_ablation_results()

    rates = [float(k) for k in results['subsample'].keys()]
    base_unc = [results['subsample'][str(r)]['base_uncertainty'] for r in rates]
    stability = [results['subsample'][str(r)]['stability'] for r in rates]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Left: Base uncertainty
    ax1.plot(rates, base_unc, 'o-', color='#9B5DE5', linewidth=2, markersize=8)
    ax1.set_xlabel('Subsampling Rate', fontsize=12)
    ax1.set_ylabel('Base Uncertainty (Ensemble Variance)', fontsize=12)
    ax1.set_title('Effect on Ensemble Diversity', fontsize=12, fontweight='bold')
    ax1.set_xticks(rates)
    ax1.grid(True, alpha=0.3)

    # Critical annotation
    ax1.annotate('rate=1.0:\nUQ ≈ 0\n(No diversity!)',
                 xy=(1.0, base_unc[-1]), xytext=(0.85, 0.03),
                 arrowprops=dict(arrowstyle='->', color='red'),
                 fontsize=9, color='red', fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    # Right: Stability
    ax2.plot(rates, stability, 's-', color='#00F5D4', linewidth=2, markersize=8)
    ax2.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='Threshold (0.9)')
    ax2.set_xlabel('Subsampling Rate', fontsize=12)
    ax2.set_ylabel('Attribution Stability (ρ)', fontsize=12)
    ax2.set_title('Effect on Attribution Stability', fontsize=12, fontweight='bold')
    ax2.set_ylim(0.7, 1.05)
    ax2.set_xticks(rates)
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/fig5_ablation_subsample.pdf', bbox_inches='tight')
    plt.savefig(f'{FIGURES_DIR}/fig5_ablation_subsample.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Created: fig5_ablation_subsample.pdf")


def fig6_hierarchy_comparison():
    """Visual comparison of hierarchy structures."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Feature-level: flat
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis('off')
    ax.set_title('Feature-level\n(Flat)', fontsize=11, fontweight='bold')

    for i, feat in enumerate(['f1', 'f2', 'f3', '...', 'f24']):
        x = 1 + i * 1.8
        rect = mpatches.FancyBboxPatch((x, 1.5), 1.2, 0.8, boxstyle="round,pad=0.02",
                                        facecolor='#FFB3B3', edgecolor='black')
        ax.add_patch(rect)
        ax.text(x + 0.6, 1.9, feat, ha='center', va='center', fontsize=9)
    ax.text(5, 0.8, 'No drill-up possible', ha='center', fontsize=10, style='italic', color='red')

    # Correlation: unstable groups
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis('off')
    ax.set_title('Correlation Clustering\n(Data-driven)', fontsize=11, fontweight='bold')

    # Groups at top
    for i, grp in enumerate(['G1', 'G2', 'G3']):
        x = 1.5 + i * 2.5
        rect = mpatches.FancyBboxPatch((x, 2.8), 1.5, 0.6, boxstyle="round,pad=0.02",
                                        facecolor='#FFFFB3', edgecolor='black')
        ax.add_patch(rect)
        ax.text(x + 0.75, 3.1, grp, ha='center', va='center', fontsize=9)
        # Features below
        for j in range(2):
            fx = x + j * 0.8
            frect = mpatches.FancyBboxPatch((fx, 1.8), 0.6, 0.5, boxstyle="round,pad=0.02",
                                            facecolor='#FFB3B3', edgecolor='black')
            ax.add_patch(frect)
            ax.text(fx + 0.3, 2.05, f'f{i*2+j+1}', ha='center', va='center', fontsize=7)
            ax.plot([fx + 0.3, x + 0.75], [2.3, 2.8], 'k-', lw=0.5)

    ax.text(5, 0.8, 'Groups change with data!', ha='center', fontsize=10, style='italic', color='orange')
    ax.annotate('', xy=(4, 1.3), xytext=(6, 1.3),
                arrowprops=dict(arrowstyle='<->', color='orange', lw=1.5))

    # FK: stable hierarchy
    ax = axes[2]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis('off')
    ax.set_title('RelUQ (FK-based)\n(Schema-defined)', fontsize=11, fontweight='bold')

    # FK groups at top
    fk_groups = ['DRIVER', 'CIRCUIT', 'RACE']
    colors = ['#B3FFB3', '#B3E0FF', '#FFE0B3']
    for i, (grp, col) in enumerate(zip(fk_groups, colors)):
        x = 1.5 + i * 2.5
        rect = mpatches.FancyBboxPatch((x, 2.8), 1.5, 0.6, boxstyle="round,pad=0.02",
                                        facecolor=col, edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + 0.75, 3.1, grp, ha='center', va='center', fontsize=9, fontweight='bold')
        # Features below
        for j in range(2):
            fx = x + j * 0.8
            frect = mpatches.FancyBboxPatch((fx, 1.8), 0.6, 0.5, boxstyle="round,pad=0.02",
                                            facecolor=col, edgecolor='black', alpha=0.6)
            ax.add_patch(frect)
            ax.plot([fx + 0.3, x + 0.75], [2.3, 2.8], 'k-', lw=0.5)

    ax.text(5, 0.8, 'Stable, actionable grouping', ha='center', fontsize=10, style='italic', color='green')

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/fig6_hierarchy.pdf', bbox_inches='tight')
    plt.savefig(f'{FIGURES_DIR}/fig6_hierarchy.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Created: fig6_hierarchy.pdf")


def fig7_multi_domain():
    """Multi-domain validation results."""
    domains = ['rel-f1\n(Sports)', 'rel-stack\n(Q&A)', 'rel-amazon\n(E-commerce)']
    stability = [0.933, 0.867, 1.000]  # From our experiments
    top_fk = ['DRIVER\n(29%)', 'ANSWER\n(41%)', 'REVIEW\n(100%)']

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(domains))
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    bars = ax.bar(x, stability, color=colors, edgecolor='black', linewidth=1.5)

    # Add value labels
    for bar, val, fk in zip(bars, stability, top_fk):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.02, f'ρ={val:.3f}',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
        ax.text(bar.get_x() + bar.get_width()/2, height/2, fk,
                ha='center', va='center', fontsize=9, color='white', fontweight='bold')

    ax.axhline(y=0.85, color='green', linestyle='--', linewidth=1.5, label='Threshold (ρ ≥ 0.85)')

    ax.set_ylabel('Attribution Stability (Spearman ρ)', fontsize=12)
    ax.set_xlabel('Dataset (Domain)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(domains, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.legend(loc='lower right')

    ax.set_title('Multi-Domain Validation: Consistent Stability Across Domains', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/fig7_multi_domain.pdf', bbox_inches='tight')
    plt.savefig(f'{FIGURES_DIR}/fig7_multi_domain.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Created: fig7_multi_domain.pdf")


def main():
    print("=" * 60)
    print("Generating Figures for NeurIPS Paper")
    print("=" * 60)

    fig1_overview()
    fig2_baseline_comparison()
    fig3_ablation_ensemble_size()
    fig4_ablation_sample_size()
    fig5_ablation_subsampling()
    fig6_hierarchy_comparison()
    fig7_multi_domain()

    print("=" * 60)
    print(f"All figures saved to: {FIGURES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
