"""
Create Figure 3: Feature Importance Analysis (4-panel layout)

Panel A: Catastrophic task - Feature importance comparison (val vs test)
Panel B: Robust task - Feature importance comparison (val vs test)
Panel C: Importance increase ratio - Shows concentration vs redistribution
Panel D: Feature ranking stability - Alluvial-style flow diagram

Author: UAI 2026 Conformal COVID Paper
Date: 2025-12-26
"""

import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path


def load_results(results_dir: Path):
    """Load SHAP results for both tasks."""
    with open(results_dir / 'shap_rel-salt_sales-shipcond.pkl', 'rb') as f:
        cat_results = pickle.load(f)

    with open(results_dir / 'shap_rel-salt_sales-office.pkl', 'rb') as f:
        rob_results = pickle.load(f)

    return cat_results, rob_results


def panel_a_catastrophic(ax, results):
    """Panel A: Catastrophic task feature importance."""
    features_val = [f for f, _ in results['top_features_val'][:5]]
    scores_val = [s for _, s in results['top_features_val'][:5]]

    # Get test scores for same features
    test_dict = {f: s for f, s in results['top_features_test']}
    scores_test = [test_dict.get(f, 0) for f in features_val]

    x = np.arange(len(features_val))
    width = 0.35

    ax.bar(x - width/2, scores_val, width, label='Pre-COVID (val)',
           color='#2E86AB', alpha=0.8)
    ax.bar(x + width/2, scores_test, width, label='Post-COVID (test)',
           color='#A23B72', alpha=0.8)

    # Abbreviate feature names
    short_names = []
    for f in features_val:
        if f == 'SALESDOCUMENT':
            short_names.append('SALESDOC')
        elif f == 'SALESORGANIZATION':
            short_names.append('SALESORG')
        elif f == 'BILLINGCOMPANYCODE':
            short_names.append('BILLCODE')
        elif f == 'TRANSACTIONCURRENCY':
            short_names.append('CURRENCY')
        elif f == 'SALESDOCUMENTTYPE':
            short_names.append('DOCTYPE')
        else:
            short_names.append(f[:8])

    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Mean |SHAP| Value', fontsize=10)
    ax.set_title('(A) Catastrophic Task: sales-shipcond\n(Coverage drop: 71.6%)',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    # Highlight dominant feature
    ax.axvspan(-0.5, 0.5, alpha=0.1, color='red')
    ax.text(0, scores_test[0] + 0.5, '4.5×', ha='center', fontsize=9,
            fontweight='bold', color='red')


def panel_b_robust(ax, results):
    """Panel B: Robust task feature importance."""
    features_val = [f for f, _ in results['top_features_val'][:5]]
    scores_val = [s for _, s in results['top_features_val'][:5]]

    # Get test scores for same features
    test_dict = {f: s for f, s in results['top_features_test']}
    scores_test = [test_dict.get(f, 0) for f in features_val]

    x = np.arange(len(features_val))
    width = 0.35

    ax.bar(x - width/2, scores_val, width, label='Pre-COVID (val)',
           color='#2E86AB', alpha=0.8)
    ax.bar(x + width/2, scores_test, width, label='Post-COVID (test)',
           color='#06A77D', alpha=0.8)

    # Abbreviate feature names
    short_names = []
    for f in features_val:
        if f == 'SALESDOCUMENT':
            short_names.append('SALESDOC')
        elif f == 'SALESORGANIZATION':
            short_names.append('SALESORG')
        elif f == 'BILLINGCOMPANYCODE':
            short_names.append('BILLCODE')
        elif f == 'TRANSACTIONCURRENCY':
            short_names.append('CURRENCY')
        elif f == 'DISTRIBUTIONCHANNEL':
            short_names.append('DISTCHAN')
        else:
            short_names.append(f[:8])

    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Mean |SHAP| Value', fontsize=10)
    ax.set_title('(B) Robust Task: sales-office\n(Coverage drop: 0.1%)',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(axis='y', alpha=0.3)


def panel_c_increase_ratio(ax, cat_results, rob_results):
    """Panel C: Importance increase ratios (test/val)."""
    # Get top 5 features for each task
    cat_features = [f for f, _ in cat_results['top_features_val'][:5]]
    rob_features = [f for f, _ in rob_results['top_features_val'][:5]]

    # Compute ratios
    cat_val_dict = {f: s for f, s in cat_results['top_features_val']}
    cat_test_dict = {f: s for f, s in cat_results['top_features_test']}
    cat_ratios = [cat_test_dict.get(f, 0) / cat_val_dict[f] if cat_val_dict[f] > 0 else 0
                  for f in cat_features]

    rob_val_dict = {f: s for f, s in rob_results['top_features_val']}
    rob_test_dict = {f: s for f, s in rob_results['top_features_test']}
    rob_ratios = [rob_test_dict.get(f, 0) / rob_val_dict[f] if rob_val_dict[f] > 0 else 0
                  for f in rob_features]

    x = np.arange(5)
    width = 0.35

    ax.bar(x - width/2, cat_ratios, width, label='Catastrophic (71.6% drop)',
           color='#A23B72', alpha=0.8)
    ax.bar(x + width/2, rob_ratios, width, label='Robust (0.1% drop)',
           color='#06A77D', alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([f'Top-{i+1}' for i in range(5)], fontsize=9)
    ax.set_ylabel('Importance Increase Ratio\n(test / validation)', fontsize=10)
    ax.set_title('(C) Feature Importance Dynamics', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    # Add interpretation text
    ax.text(0.5, 0.95, 'Catastrophic: Single dominant feature explodes',
            transform=ax.transAxes, fontsize=8, va='top', ha='center',
            bbox=dict(boxstyle='round', facecolor='#A23B72', alpha=0.2))
    ax.text(0.5, 0.85, 'Robust: Importance redistributes across features',
            transform=ax.transAxes, fontsize=8, va='top', ha='center',
            bbox=dict(boxstyle='round', facecolor='#06A77D', alpha=0.2))


def panel_d_ranking_shift(ax, cat_results, rob_results):
    """Panel D: Feature ranking stability visualization."""
    # Get top 5 features and their ranks
    cat_val_features = [f for f, _ in cat_results['top_features_val'][:5]]
    cat_test_ranks = {}
    for i, (f, _) in enumerate(cat_results['top_features_test']):
        cat_test_ranks[f] = i

    rob_val_features = [f for f, _ in rob_results['top_features_val'][:5]]
    rob_test_ranks = {}
    for i, (f, _) in enumerate(rob_results['top_features_test']):
        rob_test_ranks[f] = i

    # Compute rank changes
    cat_rank_changes = [abs(cat_test_ranks.get(f, 10) - i) for i, f in enumerate(cat_val_features)]
    rob_rank_changes = [abs(rob_test_ranks.get(f, 10) - i) for i, f in enumerate(rob_val_features)]

    cat_mean_shift = np.mean(cat_rank_changes)
    rob_mean_shift = np.mean(rob_rank_changes)

    # Bar chart of mean rank shift
    tasks = ['Catastrophic\n(71.6% drop)', 'Robust\n(0.1% drop)']
    shifts = [cat_mean_shift, rob_mean_shift]
    colors = ['#A23B72', '#06A77D']

    bars = ax.bar(tasks, shifts, color=colors, alpha=0.8, width=0.6)

    ax.set_ylabel('Mean Rank Change\n(top-5 features)', fontsize=10)
    ax.set_title('(D) Feature Ranking Stability', fontsize=10, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(shifts) * 1.3)

    # Add values on bars
    for bar, shift in zip(bars, shifts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{shift:.1f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add interpretation
    ax.text(0.5, 0.95, 'Lower = More stable feature importance',
            transform=ax.transAxes, fontsize=8, va='top', ha='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


def create_figure3(output_path: Path):
    """Create 4-panel Figure 3."""
    results_dir = Path('papers/conformal_covid/results/shap')
    cat_results, rob_results = load_results(results_dir)

    # Create figure with 2x2 layout
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    # Removed title to avoid cutoff - rely on LaTeX caption for full context
    # fig.suptitle('Feature Importance Analysis: Mechanism of Catastrophic Failure',
    #              fontsize=13, fontweight='bold', y=0.995)

    # Create panels
    panel_a_catastrophic(axes[0, 0], cat_results)
    panel_b_robust(axes[0, 1], rob_results)
    panel_c_increase_ratio(axes[1, 0], cat_results, rob_results)
    panel_d_ranking_shift(axes[1, 1], cat_results, rob_results)

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Figure 3 saved to: {output_path}")

    # Also save as PNG for quick viewing
    png_path = output_path.with_suffix('.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"✓ PNG preview saved to: {png_path}")

    plt.close()


if __name__ == '__main__':
    output_path = Path('papers/conformal_covid/figures/figure3_feature_importance.pdf')
    create_figure3(output_path)
    print("\n" + "="*80)
    print("Figure 3 creation complete!")
    print("="*80)
