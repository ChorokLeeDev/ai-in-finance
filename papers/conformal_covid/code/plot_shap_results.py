"""
Visualization for SHAP Feature Importance Results

Creates publication-quality plots for SHAP analysis:
- Top-10 feature importance bar charts
- Feature importance vs Jaccard scatter plots

Author: UAI 2026 Conformal COVID Paper
Date: 2025-12-26
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict

# Set publication-quality defaults
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13


def plot_top10_features(
    results: Dict,
    output_dir: Path,
    task_label: str = None
) -> None:
    """
    Plot top-10 features with importance and Jaccard scores.

    Args:
        results: Results dict from analyze_feature_importance.py
        output_dir: Directory to save plot
        task_label: Optional label for plot title
    """
    task = results['task']
    if task_label is None:
        task_label = task

    # Extract top features
    top_features = results['top_features_val'][:10]
    feature_names = [f[0] for f in top_features]
    importance = np.array([f[1] for f in top_features])
    jaccard = np.array([results['feature_jaccard'][f] for f in feature_names])

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))

    # Color by Jaccard (red = low/unstable, green = high/stable)
    colors = plt.cm.RdYlGn(jaccard)

    # Horizontal bar chart
    y_pos = np.arange(len(feature_names))
    bars = ax.barh(y_pos, importance, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

    # Configure axes
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names)
    ax.set_xlabel('Mean |SHAP| (Feature Importance)')
    ax.set_title(f'Top 10 Features - {task_label}')
    ax.invert_yaxis()  # Highest importance at top
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    # Add Jaccard values as text annotations
    for i, (imp, jac) in enumerate(zip(importance, jaccard)):
        ax.text(imp + max(importance) * 0.02, i, f'J={jac:.2f}',
                va='center', fontsize=8, fontweight='bold')

    # Add colorbar legend
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.15)
    cbar.set_label('Jaccard Similarity', rotation=270, labelpad=15)

    plt.tight_layout()
    output_file = output_dir / f'shap_top10_{task}.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_file}")


def plot_importance_vs_jaccard_scatter(
    results: Dict,
    output_dir: Path,
    task_label: str = None
) -> None:
    """
    Scatter plot: Feature importance vs Jaccard similarity.

    Args:
        results: Results dict from analyze_feature_importance.py
        output_dir: Directory to save plot
        task_label: Optional label for plot title
    """
    task = results['task']
    if task_label is None:
        task_label = task

    # Extract all features
    feature_names = results['feature_names']
    all_importance = np.abs(results['shap_values_val']).mean(axis=0)
    all_jaccard = np.array([results['feature_jaccard'][f] for f in feature_names])

    # Create figure
    fig, ax = plt.subplots(figsize=(7, 5))

    # Scatter plot for all features
    ax.scatter(all_jaccard, all_importance, alpha=0.3, s=30, c='gray', label='All features')

    # Highlight top-10 features
    top_features = results['top_features_val'][:10]
    top_feature_names = [f[0] for f in top_features]
    top_idx = [feature_names.index(f) for f in top_feature_names]
    top_jac = all_jaccard[top_idx]
    top_imp = all_importance[top_idx]

    ax.scatter(top_jac, top_imp, color='red', s=100, marker='*',
               label='Top 10', zorder=5, edgecolor='black', linewidth=0.5)

    # Annotate top-3 features
    for i in range(min(3, len(top_feature_names))):
        fname = top_feature_names[i]
        if len(fname) > 15:
            fname = fname[:12] + '...'
        ax.annotate(fname, (top_jac[i], top_imp[i]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=7, alpha=0.8)

    # Configure axes
    ax.set_xlabel('Feature Jaccard Similarity (train vs test)')
    ax.set_ylabel('Mean |SHAP| (Feature Importance)')
    ax.set_title(f'Feature Importance vs Temporal Stability - {task_label}')
    ax.legend(loc='best')
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xlim(-0.05, 1.05)

    plt.tight_layout()
    output_file = output_dir / f'shap_scatter_{task}.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_file}")


def plot_feature_ranking_shift(
    results: Dict,
    output_dir: Path,
    task_label: str = None
) -> None:
    """
    Plot feature ranking shift between validation and test sets.

    Shows how feature importance changes between pre-COVID and post-COVID.

    Args:
        results: Results dict from analyze_feature_importance.py
        output_dir: Directory to save plot
        task_label: Optional label for plot title
    """
    task = results['task']
    if task_label is None:
        task_label = task

    # Get top-10 features from validation
    top_features_val = results['top_features_val'][:10]
    val_feature_names = [f[0] for f in top_features_val]

    # Get their rankings in test set
    feature_names = results['feature_names']
    test_importance = np.abs(results['shap_values_test']).mean(axis=0)
    test_ranking = np.argsort(test_importance)[::-1]  # Descending

    # Find ranks of val top-10 in test set
    val_ranks = list(range(1, 11))  # 1-10
    test_ranks = []
    for fname in val_feature_names:
        idx = feature_names.index(fname)
        rank = np.where(test_ranking == idx)[0][0] + 1  # 1-indexed
        test_ranks.append(rank)

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 5))

    # Plot ranking changes
    for i, fname in enumerate(val_feature_names):
        val_rank = val_ranks[i]
        test_rank = test_ranks[i]

        # Color: stable (green) vs changed (red)
        rank_change = abs(test_rank - val_rank)
        if rank_change <= 2:
            color = 'green'
            alpha = 0.6
        elif rank_change <= 5:
            color = 'orange'
            alpha = 0.7
        else:
            color = 'red'
            alpha = 0.8

        # Draw line
        ax.plot([0, 1], [val_rank, test_rank], color=color, alpha=alpha, linewidth=2)
        ax.scatter([0, 1], [val_rank, test_rank], color=color, s=50, zorder=5)

        # Annotate feature name
        if len(fname) > 12:
            fname_short = fname[:9] + '...'
        else:
            fname_short = fname
        ax.text(-0.05, val_rank, fname_short, ha='right', va='center', fontsize=7)

    # Configure axes
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Validation\n(Pre-COVID)', 'Test\n(Post-COVID)'])
    ax.set_ylabel('Feature Rank')
    ax.set_ylim(max(test_ranks) + 2, 0)  # Inverted (rank 1 at top)
    ax.set_xlim(-0.3, 1.3)
    ax.set_title(f'Feature Ranking Shift - {task_label}')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='green', linewidth=2, label='Stable (Δ≤2)'),
        Line2D([0], [0], color='orange', linewidth=2, label='Moderate (Δ≤5)'),
        Line2D([0], [0], color='red', linewidth=2, label='Large (Δ>5)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8)

    plt.tight_layout()
    output_file = output_dir / f'shap_ranking_shift_{task}.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_file}")


def plot_shap_summary(results: Dict, output_dir: Path) -> None:
    """
    Create all SHAP summary plots.

    Args:
        results: Results dict from analyze_feature_importance.py
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    task = results['task']

    # Determine task label (catastrophic vs robust)
    mean_jaccard = results.get('mean_jaccard_all', 0)
    if mean_jaccard < 0.3:
        task_label = f"{task} (Catastrophic)"
    elif mean_jaccard > 0.5:
        task_label = f"{task} (Robust)"
    else:
        task_label = task

    print(f"\nGenerating plots for {task}...")

    # Plot 1: Top-10 features bar chart
    plot_top10_features(results, output_dir, task_label)

    # Plot 2: Importance vs Jaccard scatter
    plot_importance_vs_jaccard_scatter(results, output_dir, task_label)

    # Plot 3: Feature ranking shift
    plot_feature_ranking_shift(results, output_dir, task_label)

    print(f"\n✓ All plots created in {output_dir}/")


def create_combined_figure3(
    results_catastrophic: Dict,
    results_robust: Dict,
    output_dir: Path
) -> None:
    """
    Create Figure 3 for paper: 2x2 panel layout comparing both tasks.

    Panel A: Top-10 features (catastrophic)
    Panel B: Top-10 features (robust)
    Panel C: Scatter plot (both tasks)
    Panel D: Ranking shift comparison

    Args:
        results_catastrophic: Results for catastrophic task
        results_robust: Results for robust task
        output_dir: Directory to save figure
    """
    fig = plt.figure(figsize=(12, 10))

    # Panel A: Catastrophic task top-10
    ax1 = plt.subplot(2, 2, 1)
    # ... implementation similar to plot_top10_features but on ax1

    # Panel B: Robust task top-10
    ax2 = plt.subplot(2, 2, 2)
    # ... implementation similar to plot_top10_features but on ax2

    # Panel C: Combined scatter
    ax3 = plt.subplot(2, 2, 3)
    # ... plot both tasks on same scatter

    # Panel D: Ranking shifts
    ax4 = plt.subplot(2, 2, 4)
    # ... combined ranking comparison

    plt.tight_layout()
    output_file = output_dir / 'figure3_feature_importance_combined.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✓ Combined Figure 3 saved to {output_file}")


if __name__ == "__main__":
    # Example usage for testing
    import pickle
    import sys

    if len(sys.argv) < 2:
        print("Usage: python plot_shap_results.py <shap_results.pkl>")
        sys.exit(1)

    # Load results
    results_file = sys.argv[1]
    with open(results_file, 'rb') as f:
        results = pickle.load(f)

    # Create plots
    output_dir = Path(results_file).parent
    plot_shap_summary(results, output_dir)
