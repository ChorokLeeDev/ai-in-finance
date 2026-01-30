"""
Visualization for Retraining Experiment Results

Creates Figure 4 for paper showing how retraining restores coverage:
- Coverage over time for different retraining frequencies
- Coverage vs retraining cost (Pareto curve)
- Jaccard decay over time
- Decision framework

Author: UAI 2026 Conformal COVID Paper
Date: 2025-12-26
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

# Publication-quality settings
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13


def load_results(results_dir: Path, task: str = 'sales-shipcond') -> Dict[str, List[Dict]]:
    """
    Load retraining results for all frequencies.

    Args:
        results_dir: Directory containing result files
        task: Task name

    Returns:
        Dict mapping frequency ('none', '1M', '3M', '6M') to results list
    """
    results = {}
    for freq in ['none', '1M', '3M', '6M']:
        pkl_file = results_dir / f'retrain_{freq}_{task}.pkl'
        if pkl_file.exists():
            with open(pkl_file, 'rb') as f:
                results[freq] = pickle.load(f)
        else:
            print(f"Warning: {pkl_file} not found")

    return results


def plot_coverage_over_time(
    all_results: Dict[str, List[Dict]],
    output_dir: Path,
    task: str = 'sales-shipcond'
) -> None:
    """
    Panel A: Coverage over time for all retraining frequencies.

    Shows how coverage degrades without retraining and is maintained with retraining.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Color scheme
    colors = {
        'none': '#d62728',      # Red (failure)
        '6M': '#ff7f0e',        # Orange (borderline)
        '3M': '#2ca02c',        # Green (good)
        '1M': '#1f77b4',        # Blue (best)
    }

    labels = {
        'none': 'No retrain',
        '6M': 'Semi-annual (2/year)',
        '3M': 'Quarterly (4/year)',
        '1M': 'Monthly (12/year)',
    }

    # Plot each frequency
    for freq in ['none', '6M', '3M', '1M']:
        if freq not in all_results:
            continue

        results = all_results[freq]
        months = [r['month'] for r in results]
        coverages = [r['coverage'] * 100 for r in results]  # Convert to percentage
        retrains = [r['retrained'] for r in results]

        # Plot coverage line
        ax.plot(months, coverages, color=colors[freq], linewidth=2.5,
                label=labels[freq], marker='o', markersize=4, alpha=0.9)

        # Add vertical lines for retrain points (except 'none' and skip first point)
        if freq != 'none':
            for i, retrained in enumerate(retrains):
                if retrained and i > 0:  # Skip initial training
                    ax.axvline(x=months[i], color=colors[freq], alpha=0.2,
                              linestyle='--', linewidth=1)

    # Target coverage line
    ax.axhline(y=90, color='black', linestyle=':', linewidth=1.5,
               alpha=0.5, label='Target (90%)')

    # Configure axes
    ax.set_xlabel('Month Index (0=Feb 2020, 10=Dec 2020)')
    ax.set_ylabel('Coverage (%)')
    # Removed title to avoid cutoff - rely on LaTeX caption
    # ax.set_title(f'Coverage Degradation and Recovery - {task}')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_ylim([0, 100])

    plt.tight_layout()
    output_file = output_dir / 'retrain_coverage_over_time.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {output_file}")


def plot_coverage_vs_cost(
    all_results: Dict[str, List[Dict]],
    output_dir: Path
) -> None:
    """
    Panel B: Coverage vs retraining cost (Pareto curve).

    Shows trade-off between coverage quality and retraining frequency.
    """
    fig, ax = plt.subplots(figsize=(9, 8))

    # Extract summary statistics
    freq_order = ['none', '6M', '3M', '1M']
    mean_coverages = []
    n_retrains = []
    labels_short = {'none': 'None', '6M': '6M', '3M': '3M', '1M': '1M'}

    for freq in freq_order:
        if freq not in all_results:
            continue
        results = all_results[freq]
        coverages = [r['coverage'] * 100 for r in results]
        retrains = sum(r['retrained'] for r in results) - 1  # Subtract initial training

        mean_coverages.append(np.mean(coverages))
        n_retrains.append(retrains)

    # Plot Pareto curve
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']
    for i, (freq, mean_cov, n_ret) in enumerate(zip(freq_order, mean_coverages, n_retrains)):
        ax.scatter(n_ret, mean_cov, s=200, color=colors[i],
                  marker='o', edgecolor='black', linewidth=1.5,
                  label=labels_short[freq], zorder=5)

        # Annotate
        ax.annotate(f'{mean_cov:.1f}%', (n_ret, mean_cov),
                   xytext=(8, 0), textcoords='offset points',
                   fontsize=9, va='center')

    # Connect with line
    ax.plot(n_retrains, mean_coverages, color='gray', linestyle='--',
            linewidth=1.5, alpha=0.5, zorder=1)

    # Highlight recommended point (3M = quarterly)
    rec_idx = freq_order.index('3M')
    ax.scatter(n_retrains[rec_idx], mean_coverages[rec_idx],
              s=400, facecolors='none', edgecolors='green',
              linewidth=3, zorder=6)
    ax.annotate('Recommended', (n_retrains[rec_idx], mean_coverages[rec_idx]),
               xytext=(15, 15), textcoords='offset points',
               fontsize=10, fontweight='bold', color='green',
               arrowprops=dict(arrowstyle='->', color='green', lw=2))

    # Configure axes
    ax.set_xlabel('Retrains per Year')
    ax.set_ylabel('Mean Coverage (%)')
    ax.set_title('Coverage vs Retraining Cost')
    ax.legend(loc='lower right', title='Frequency')
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_ylim([0, 100])
    ax.set_xlim([-0.5, max(n_retrains) + 0.5])

    plt.tight_layout()
    output_file = output_dir / 'retrain_coverage_vs_cost.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {output_file}")


def plot_jaccard_decay(
    all_results: Dict[str, List[Dict]],
    output_dir: Path
) -> None:
    """
    Panel C: Jaccard decay over time.

    Shows why coverage degrades (feature distributions drift).
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot Jaccard for baseline (no retrain)
    if 'none' in all_results:
        results = all_results['none']
        months = [r['month'] for r in results]
        jaccards = [r['jaccard'] for r in results]

        ax.plot(months, jaccards, color='#d62728', linewidth=2.5,
                marker='o', markersize=5, label='Feature Jaccard (baseline)')

    # Overlay coverage for comparison
    if 'none' in all_results:
        ax2 = ax.twinx()
        results = all_results['none']
        coverages = [r['coverage'] * 100 for r in results]
        ax2.plot(months, coverages, color='#1f77b4', linewidth=2.5,
                marker='s', markersize=5, linestyle='--',
                label='Coverage (baseline)', alpha=0.7)
        ax2.set_ylabel('Coverage (%)', color='#1f77b4')
        ax2.tick_params(axis='y', labelcolor='#1f77b4')
        ax2.set_ylim([0, 100])

    # Configure axes
    ax.set_xlabel('Month Index (Feb 2020 - Dec 2020)')
    ax.set_ylabel('Mean Jaccard Similarity', color='#d62728')
    ax.tick_params(axis='y', labelcolor='#d62728')
    ax.set_title('Feature Drift Explains Coverage Degradation')
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_ylim([0, 1])

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    if 'none' in all_results:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='best')

    plt.tight_layout()
    output_file = output_dir / 'retrain_jaccard_decay.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {output_file}")


def plot_decision_framework(output_dir: Path) -> None:
    """
    Panel D: Decision framework flowchart.

    Provides practitioners with actionable guidance.
    """
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.axis('off')

    # Create flowchart using text boxes
    # Title
    ax.text(0.5, 0.95, 'Retraining Decision Framework',
           ha='center', va='top', fontsize=14, fontweight='bold')

    # Step 1: Measure Jaccard
    ax.add_patch(plt.Rectangle((0.2, 0.80), 0.6, 0.08, facecolor='lightblue',
                               edgecolor='black', linewidth=2))
    ax.text(0.5, 0.84, 'Step 1: Measure Mean Jaccard',
           ha='center', va='center', fontsize=11, fontweight='bold')

    # Arrow down
    ax.annotate('', xy=(0.5, 0.80), xytext=(0.5, 0.75),
               arrowprops=dict(arrowstyle='->', lw=2))

    # Decision: High Jaccard (> 0.4)
    ax.add_patch(plt.Rectangle((0.05, 0.62), 0.35, 0.10, facecolor='#90ee90',
                               edgecolor='black', linewidth=2))
    ax.text(0.225, 0.69, 'High Jaccard (> 0.4)',
           ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(0.225, 0.65, 'Stable features',
           ha='center', va='center', fontsize=9)

    # Decision: Medium Jaccard (0.1 - 0.4)
    ax.add_patch(plt.Rectangle((0.45, 0.62), 0.35, 0.10, facecolor='#ffe4b5',
                               edgecolor='black', linewidth=2))
    ax.text(0.625, 0.69, 'Medium Jaccard (0.1-0.4)',
           ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(0.625, 0.65, 'Moderate drift',
           ha='center', va='center', fontsize=9)

    # Decision: Low Jaccard (< 0.1)
    ax.add_patch(plt.Rectangle((0.85, 0.62), 0.12, 0.10, facecolor='#ffcccb',
                               edgecolor='black', linewidth=2))
    ax.text(0.91, 0.69, 'Low (<0.1)',
           ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(0.91, 0.65, 'High drift',
           ha='center', va='center', fontsize=8)

    # Arrows down
    for x in [0.225, 0.625, 0.91]:
        ax.annotate('', xy=(x, 0.62), xytext=(x, 0.57),
                   arrowprops=dict(arrowstyle='->', lw=1.5))

    # Recommendations
    # Yearly
    ax.add_patch(plt.Rectangle((0.05, 0.45), 0.35, 0.10, facecolor='white',
                               edgecolor='green', linewidth=2))
    ax.text(0.225, 0.52, 'Yearly Retrain',
           ha='center', va='center', fontsize=10, fontweight='bold', color='green')
    ax.text(0.225, 0.48, 'Cost: Low (1-2/year)',
           ha='center', va='center', fontsize=8)

    # Quarterly
    ax.add_patch(plt.Rectangle((0.45, 0.45), 0.35, 0.10, facecolor='white',
                               edgecolor='orange', linewidth=2))
    ax.text(0.625, 0.52, 'Quarterly Retrain ⭐',
           ha='center', va='center', fontsize=10, fontweight='bold', color='orange')
    ax.text(0.625, 0.48, 'Cost: Medium (4/year)',
           ha='center', va='center', fontsize=8)

    # Monthly
    ax.add_patch(plt.Rectangle((0.85, 0.45), 0.12, 0.10, facecolor='white',
                               edgecolor='red', linewidth=2))
    ax.text(0.91, 0.52, 'Monthly',
           ha='center', va='center', fontsize=9, fontweight='bold', color='red')
    ax.text(0.91, 0.48, '12/year',
           ha='center', va='center', fontsize=7)

    # Bottom note
    ax.text(0.5, 0.35, 'Note: Adjust based on application requirements and computational budget',
           ha='center', va='center', fontsize=9, style='italic')

    # Example box
    ax.add_patch(plt.Rectangle((0.15, 0.10), 0.7, 0.20, facecolor='#f0f0f0',
                               edgecolor='black', linewidth=1))
    ax.text(0.5, 0.26, 'Example: sales-shipcond (Jaccard=0.02)',
           ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(0.5, 0.22, '• Without retrain: Coverage drops to 0.1%',
           ha='center', va='center', fontsize=9)
    ax.text(0.5, 0.18, '• With quarterly retrain: Coverage maintained at 80%+',
           ha='center', va='center', fontsize=9)
    ax.text(0.5, 0.14, '• Recommended: Quarterly (good coverage/cost balance)',
           ha='center', va='center', fontsize=9)

    plt.tight_layout()
    output_file = output_dir / 'retrain_decision_framework.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {output_file}")


def create_figure4_combined(
    all_results: Dict[str, List[Dict]],
    output_dir: Path,
    task: str = 'sales-shipcond'
) -> None:
    """
    Create Figure 4 for paper: 2x2 panel layout.

    Panel A: Coverage over time
    Panel B: Coverage vs cost
    Panel C: Jaccard decay
    Panel D: Decision framework
    """
    fig = plt.figure(figsize=(14, 10))

    # This is a placeholder - would need full implementation
    # For now, individual plots are created separately

    print("Note: Use individual plot files for Figure 4 panels")
    print("  Panel A: retrain_coverage_over_time.pdf")
    print("  Panel B: retrain_coverage_vs_cost.pdf")
    print("  Panel C: retrain_jaccard_decay.pdf")
    print("  Panel D: retrain_decision_framework.pdf")


def generate_summary_table(
    all_results: Dict[str, List[Dict]],
    output_dir: Path
) -> None:
    """Generate LaTeX table for paper."""

    table_lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Retraining Frequency Impact on Coverage}",
        "\\label{tab:retrain}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Frequency & Retrains/Year & Mean Cov. & Min Cov. & Std Cov. \\\\",
        "\\midrule",
    ]

    freq_labels = {
        'none': 'No retrain',
        '6M': 'Semi-annual',
        '3M': 'Quarterly',
        '1M': 'Monthly',
    }

    for freq in ['none', '6M', '3M', '1M']:
        if freq not in all_results:
            continue

        results = all_results[freq]
        coverages = [r['coverage'] * 100 for r in results]
        n_retrains = sum(r['retrained'] for r in results) - 1

        mean_cov = np.mean(coverages)
        min_cov = np.min(coverages)
        std_cov = np.std(coverages)

        line = f"{freq_labels[freq]} & {n_retrains} & {mean_cov:.1f}\\% & {min_cov:.1f}\\% & {std_cov:.1f}\\% \\\\"
        table_lines.append(line)

    table_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    output_file = output_dir / 'retraining_table.tex'
    with open(output_file, 'w') as f:
        f.write('\n'.join(table_lines))

    print(f"\n✓ LaTeX table saved: {output_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Plot retraining experiment results")
    parser.add_argument('--results_dir', type=str,
                       default='papers/conformal_covid/results/retraining',
                       help='Directory with result files')
    parser.add_argument('--task', type=str, default='sales-shipcond',
                       help='Task name')
    parser.add_argument('--output_dir', type=str,
                       default='papers/conformal_covid/results/retraining',
                       help='Output directory for plots')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print("GENERATING RETRAINING PLOTS")
    print(f"{'='*80}\n")

    # Load results
    print("Loading results...")
    all_results = load_results(results_dir, args.task)
    print(f"  Loaded {len(all_results)} frequency scenarios\n")

    # Generate plots
    print("Creating plots...\n")

    plot_coverage_over_time(all_results, output_dir, args.task)
    plot_coverage_vs_cost(all_results, output_dir)
    plot_jaccard_decay(all_results, output_dir)
    plot_decision_framework(output_dir)

    # Generate summary table
    generate_summary_table(all_results, output_dir)

    print(f"\n{'='*80}")
    print("All plots generated successfully!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
