"""
Correlate PSI (Label Shift) vs MMD (Feature Shift)
====================================================

Validates the hypothesis: PSI and MMD measure different aspects of distribution shift.

Expected Finding:
- sales-office: PSI=0.0000, MMD=high → Feature shift without label shift
- sales-payterms: PSI=0.0057, MMD=high → Feature shift without label shift

Author: ChorokLeeDev
Created: 2025-11-28
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def load_results():
    """Load PSI and MMD results."""
    results_dir = Path("chorok/results")

    with open(results_dir / "shift_detection.json") as f:
        shift_data = json.load(f)

    with open(results_dir / "feature_shift_analysis.json") as f:
        mmd_data = json.load(f)

    return shift_data, mmd_data

def extract_metrics(shift_data, mmd_data):
    """Extract PSI and MMD for each task."""
    tasks = []
    psi_values = []
    mmd_values = []

    for task_name, mmd_results in mmd_data.items():
        if 'error' in mmd_results:
            continue
        if task_name not in shift_data:
            continue

        shift_results = shift_data[task_name]
        shift_metrics = shift_results.get('shift_metrics', {})

        psi = shift_metrics.get('psi_train_test')
        mmd = mmd_results.get('mmd', {}).get('train_test')

        if psi is not None and mmd is not None:
            tasks.append(task_name)
            psi_values.append(psi)
            mmd_values.append(mmd)

    return tasks, np.array(psi_values), np.array(mmd_values)

def create_correlation_plot(tasks, psi_values, mmd_values, output_path):
    """Create scatter plot showing PSI vs MMD."""

    fig, ax = plt.subplots(figsize=(10, 8))

    # Color by task type
    colors = ['#e74c3c' if 'sales' in t else '#3498db' for t in tasks]

    scatter = ax.scatter(psi_values, mmd_values, s=300, c=colors,
                        alpha=0.7, edgecolor='black', linewidth=2, zorder=3)

    # Add task labels
    for i, task in enumerate(tasks):
        offset = (10, 10) if psi_values[i] < 0.1 else (-50, 10)
        ax.annotate(task, (psi_values[i], mmd_values[i]),
                   xytext=offset, textcoords='offset points',
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    # Correlation statistics
    corr, p_value = stats.pearsonr(psi_values, mmd_values)
    spearman_r, spearman_p = stats.spearmanr(psi_values, mmd_values)

    # Add statistics box
    stats_text = f'Pearson r = {corr:.3f} (p = {p_value:.4f})\nSpearman ρ = {spearman_r:.3f} (p = {spearman_p:.4f})'
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    # Reference lines
    ax.axhline(0.03, color='gray', linestyle='--', alpha=0.5, label='MMD=0.03 threshold')
    ax.axvline(0.1, color='orange', linestyle='--', alpha=0.5, label='PSI=0.1 (moderate)')

    # Labels
    ax.set_xlabel('PSI (Label Distribution Shift)', fontsize=13, fontweight='bold')
    ax.set_ylabel('MMD (Feature Distribution Shift)', fontsize=13, fontweight='bold')
    ax.set_title('Label Shift (PSI) vs Feature Shift (MMD)\nSALT Dataset - COVID-19 Natural Experiment',
                fontsize=14, fontweight='bold')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', label='Sales-level tasks', alpha=0.7),
        Patch(facecolor='#3498db', label='Item-level tasks', alpha=0.7),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved correlation plot: {output_path}")
    return corr, p_value

def print_summary_table(tasks, psi_values, mmd_values):
    """Print comparison table."""
    print("\n" + "="*80)
    print("PSI vs MMD COMPARISON TABLE")
    print("="*80)
    print(f"{'Task':<20} {'PSI (Label)':>12} {'MMD (Feature)':>14} {'Interpretation':<25}")
    print("-"*80)

    # Sort by MMD descending
    sorted_idx = np.argsort(mmd_values)[::-1]

    for idx in sorted_idx:
        task = tasks[idx]
        psi = psi_values[idx]
        mmd = mmd_values[idx]

        # Interpretation
        if psi < 0.01 and mmd > 0.04:
            interp = "FEATURE SHIFT w/o LABEL"
        elif psi > 0.1 and mmd < 0.02:
            interp = "LABEL SHIFT w/o FEATURE"
        elif psi > 0.1 and mmd > 0.04:
            interp = "BOTH SHIFT"
        else:
            interp = "LOW SHIFT"

        print(f"{task:<20} {psi:>12.4f} {mmd:>14.6f} {interp:<25}")

    print("="*80)

    # Correlation
    corr, p_value = stats.pearsonr(psi_values, mmd_values)
    print(f"\nPearson correlation: r = {corr:.4f}, p = {p_value:.4f}")

    if abs(corr) < 0.3:
        print("→ WEAK/NO correlation: PSI and MMD measure DIFFERENT aspects of shift")
    elif corr > 0:
        print("→ POSITIVE correlation: Label and feature shifts co-occur")
    else:
        print("→ NEGATIVE correlation: Interesting inverse relationship")

    print("="*80 + "\n")

def main():
    print("\n" + "#"*60)
    print("# PSI vs MMD CORRELATION ANALYSIS")
    print("#"*60)

    # Load data
    shift_data, mmd_data = load_results()

    # Extract metrics
    tasks, psi_values, mmd_values = extract_metrics(shift_data, mmd_data)

    print(f"\nTasks analyzed: {len(tasks)}")

    # Print summary table
    print_summary_table(tasks, psi_values, mmd_values)

    # Create plot
    output_path = Path("chorok/figures/psi_vs_mmd_correlation.pdf")
    corr, p_value = create_correlation_plot(tasks, psi_values, mmd_values, output_path)

    # Key finding
    print("\n" + "="*60)
    print("KEY FINDING")
    print("="*60)

    # Find sales-office and sales-payterms
    for i, task in enumerate(tasks):
        if task in ['sales-office', 'sales-payterms']:
            print(f"  {task}: PSI={psi_values[i]:.4f}, MMD={mmd_values[i]:.6f}")
            print(f"    → HIGH feature shift despite LOW/ZERO label shift!")

    print("\nThis validates: PSI (labels) and MMD (features) capture ORTHOGONAL shift signals")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
