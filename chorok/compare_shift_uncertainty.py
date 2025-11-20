"""
Multi-Task Comparison: Distribution Shift vs Uncertainty Increase
==================================================================

**Research Goal**: Validate the core hypothesis across multiple tasks (Phase 4).
This is the final validation step of the UQ-based shift detection research.

**Hypothesis**: Epistemic uncertainty increase correlates with distribution shift magnitude
- Null (H₀): No correlation between PSI and uncertainty increase
- Alternative (H₁): Positive correlation exists (r > 0.7, p < 0.05)

**Why This Matters**:
- Single-task result (item-plant) could be coincidence
- Multi-task validation (8 tasks) provides statistical evidence
- Strong correlation proves UQ can reliably detect distribution shift

**Workflow**:
1. Load shift detection results (Phase 1: temporal_shift_detection.py)
   - PSI scores for each task
   - Ground truth shift magnitude

2. Load uncertainty analysis results (Phase 3: temporal_uncertainty_analysis.py)
   - Epistemic uncertainty increase % for each task
   - Derived from ensemble disagreement (predictions as tool)

3. Correlation analysis
   - Scatter plot: PSI (x-axis) vs Uncertainty Increase (y-axis)
   - Pearson correlation coefficient and p-value
   - Linear regression fit

4. Interpretation
   - If r > 0.7 and p < 0.05: HYPOTHESIS VALIDATED ✅
   - UQ can detect shift without ground truth labels
   - Practical application: Early warning system for model monitoring

**Expected Result**: Strong positive correlation
- Tasks with high PSI (sales-group: 0.200) → High uncertainty increase
- Tasks with low PSI (item-plant: 0.092) → Moderate uncertainty increase
- Consistent relationship across all 8 tasks

**Output**:
- Correlation plot (chorok/figures/shift_vs_uncertainty_correlation.png)
- Statistical validation results (Pearson r, p-value)
- Scientific evidence for publication

**Key Reminder**: Predictions are NOT the goal. They are merely the mechanism
to measure epistemic uncertainty, which is our actual proxy for shift detection.

Author: ChorokLeeDev
Created: 2025-01-19
Last Updated: 2025-01-20
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def load_results(results_dir: Path = Path("chorok/results")):
    """Load all analysis results."""
    # Load shift detection results
    with open(results_dir / "shift_detection.json", 'r') as f:
        shift_data = json.load(f)

    # Load uncertainty analysis results
    with open(results_dir / "temporal_uncertainty.json", 'r') as f:
        uncertainty_data = json.load(f)

    return shift_data, uncertainty_data


def extract_metrics(shift_data, uncertainty_data):
    """Extract PSI and uncertainty increase for available tasks."""
    tasks = []
    psi_values = []
    uncertainty_increases = []
    js_divs = []

    for task_name in uncertainty_data.keys():
        if task_name in shift_data:
            # Get shift metrics
            shift_metrics = shift_data[task_name].get('shift_metrics', {})
            psi = shift_metrics.get('psi_train_test')
            js = shift_metrics.get('js_train_test')

            # Get uncertainty metrics
            unc_results = uncertainty_data[task_name].get('uncertainty_results', {})
            unc_shift = unc_results.get('uncertainty_shift', {})
            unc_increase_pct = unc_shift.get('train_to_test_pct')

            if psi is not None and unc_increase_pct is not None:
                tasks.append(task_name)
                psi_values.append(psi)
                uncertainty_increases.append(unc_increase_pct)
                js_divs.append(js if js is not None else 0)

    return tasks, np.array(psi_values), np.array(uncertainty_increases), np.array(js_divs)


def plot_correlation(tasks, psi_values, uncertainty_increases, js_divs, output_path):
    """Create scatter plot showing PSI vs Uncertainty correlation."""

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: PSI vs Uncertainty Increase
    colors = plt.cm.RdYlGn_r(psi_values / psi_values.max() if len(psi_values) > 0 else [0])

    axes[0].scatter(psi_values, uncertainty_increases, s=300, c=colors,
                   alpha=0.7, edgecolor='black', linewidth=2, zorder=3)

    # Add task labels
    for i, task in enumerate(tasks):
        axes[0].annotate(task, (psi_values[i], uncertainty_increases[i]),
                        xytext=(10, 5), textcoords='offset points',
                        fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.6))

    # Fit regression line
    if len(psi_values) > 1:
        z = np.polyfit(psi_values, uncertainty_increases, 1)
        p = np.poly1d(z)
        x_line = np.linspace(psi_values.min(), psi_values.max(), 100)
        axes[0].plot(x_line, p(x_line), 'r--', linewidth=2, alpha=0.7, label='Linear fit')

        # Calculate correlation
        corr, p_value = stats.pearsonr(psi_values, uncertainty_increases)
        axes[0].text(0.05, 0.95, f'Pearson r = {corr:.3f}\np = {p_value:.4f}',
                    transform=axes[0].transAxes, fontsize=11,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # PSI threshold lines
    axes[0].axvline(0.1, color='orange', linestyle='--', linewidth=1.5, alpha=0.5, label='PSI=0.1 (Moderate)')
    axes[0].axvline(0.2, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='PSI=0.2 (Significant)')

    axes[0].set_xlabel('Distribution Shift (PSI)', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Uncertainty Increase (%)', fontsize=13, fontweight='bold')
    axes[0].set_title('Distribution Shift vs Epistemic Uncertainty Increase', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10, loc='lower right')
    axes[0].grid(True, alpha=0.3)

    # Plot 2: JS Divergence vs Uncertainty Increase
    colors_js = plt.cm.plasma(js_divs / js_divs.max() if len(js_divs) > 0 and js_divs.max() > 0 else [0])

    axes[1].scatter(js_divs, uncertainty_increases, s=300, c=colors_js,
                   alpha=0.7, edgecolor='black', linewidth=2, zorder=3)

    # Add task labels
    for i, task in enumerate(tasks):
        axes[1].annotate(task, (js_divs[i], uncertainty_increases[i]),
                        xytext=(10, 5), textcoords='offset points',
                        fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='cyan', alpha=0.6))

    # Fit regression line
    if len(js_divs) > 1:
        z_js = np.polyfit(js_divs, uncertainty_increases, 1)
        p_js = np.poly1d(z_js)
        x_line_js = np.linspace(js_divs.min(), js_divs.max(), 100)
        axes[1].plot(x_line_js, p_js(x_line_js), 'b--', linewidth=2, alpha=0.7, label='Linear fit')

        # Calculate correlation
        corr_js, p_value_js = stats.pearsonr(js_divs, uncertainty_increases)
        axes[1].text(0.05, 0.95, f'Pearson r = {corr_js:.3f}\np = {p_value_js:.4f}',
                    transform=axes[1].transAxes, fontsize=11,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    axes[1].set_xlabel('Jensen-Shannon Divergence', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Uncertainty Increase (%)', fontsize=13, fontweight='bold')
    axes[1].set_title('JS Divergence vs Epistemic Uncertainty Increase', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10, loc='lower right')
    axes[1].grid(True, alpha=0.3)

    fig.suptitle('Validation: Uncertainty Quantification Tracks Distribution Shift',
                fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved correlation plot: {output_path}")


def print_summary_table(tasks, psi_values, uncertainty_increases, js_divs):
    """Print summary table of all tasks."""
    print("\n" + "="*80)
    print("DISTRIBUTION SHIFT vs UNCERTAINTY INCREASE")
    print("="*80)
    print(f"{'Task':<20} {'PSI':>8} {'JS Div':>8} {'Unc. Increase':>14} {'Category':<15}")
    print("-"*80)

    # Sort by PSI
    sorted_idx = np.argsort(psi_values)[::-1]

    for idx in sorted_idx:
        task = tasks[idx]
        psi = psi_values[idx]
        js = js_divs[idx]
        unc_inc = uncertainty_increases[idx]

        if psi >= 0.2:
            category = "High Shift"
        elif psi >= 0.1:
            category = "Moderate Shift"
        else:
            category = "Low Shift"

        print(f"{task:<20} {psi:>8.4f} {js:>8.4f} {unc_inc:>13.2f}% {category:<15}")

    print("="*80)

    # Correlation statistics
    if len(psi_values) > 1:
        corr_psi, p_psi = stats.pearsonr(psi_values, uncertainty_increases)
        corr_js, p_js = stats.pearsonr(js_divs, uncertainty_increases)

        print("\nCORRELATION ANALYSIS:")
        print(f"  PSI vs Uncertainty Increase:")
        print(f"    Pearson r = {corr_psi:.4f}, p-value = {p_psi:.4e}")
        print(f"    {'SIGNIFICANT' if p_psi < 0.05 else 'Not significant'} at α=0.05")
        print(f"\n  JS Divergence vs Uncertainty Increase:")
        print(f"    Pearson r = {corr_js:.4f}, p-value = {p_js:.4e}")
        print(f"    {'SIGNIFICANT' if p_js < 0.05 else 'Not significant'} at α=0.05")

        # Interpretation
        print("\nINTERPRETATION:")
        if corr_psi > 0.7 and p_psi < 0.05:
            print("  ✅ STRONG POSITIVE CORRELATION - Hypothesis VALIDATED")
            print("     Epistemic uncertainty reliably tracks distribution shift magnitude")
        elif corr_psi > 0.5 and p_psi < 0.05:
            print("  ✓ MODERATE POSITIVE CORRELATION - Hypothesis SUPPORTED")
            print("     Epistemic uncertainty generally increases with distribution shift")
        elif corr_psi > 0 and p_psi < 0.05:
            print("  ~ WEAK POSITIVE CORRELATION - Hypothesis PARTIALLY SUPPORTED")
        else:
            print("  ✗ NO SIGNIFICANT CORRELATION - Hypothesis NOT VALIDATED")

    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Compare distribution shift vs uncertainty across tasks")
    parser.add_argument("--results_dir", type=str, default="chorok/results",
                       help="Directory with analysis results")
    parser.add_argument("--output_dir", type=str, default="chorok/figures",
                       help="Output directory for figures")

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "#"*80)
    print("# MULTI-TASK COMPARISON: DISTRIBUTION SHIFT vs UNCERTAINTY")
    print("#"*80 + "\n")

    # Load results
    print("Loading results...")
    shift_data, uncertainty_data = load_results(results_dir)

    print(f"  Shift data: {len(shift_data)} tasks")
    print(f"  Uncertainty data: {len(uncertainty_data)} tasks")

    # Extract metrics
    tasks, psi_values, uncertainty_increases, js_divs = extract_metrics(shift_data, uncertainty_data)

    print(f"\n  Tasks with both metrics: {len(tasks)}")
    if len(tasks) == 0:
        print("\nERROR: No tasks with both shift and uncertainty metrics found.")
        print("Please run temporal_uncertainty_analysis.py on tasks first.")
        return

    print(f"  Tasks: {', '.join(tasks)}")

    # Print summary table
    print_summary_table(tasks, psi_values, uncertainty_increases, js_divs)

    # Create correlation plot
    output_path = output_dir / "shift_uncertainty_correlation.pdf"
    plot_correlation(tasks, psi_values, uncertainty_increases, js_divs, output_path)

    # Save numerical results
    results = {
        'tasks': tasks,
        'psi_values': psi_values.tolist(),
        'uncertainty_increases': uncertainty_increases.tolist(),
        'js_divs': js_divs.tolist(),
    }

    if len(psi_values) > 1:
        corr_psi, p_psi = stats.pearsonr(psi_values, uncertainty_increases)
        corr_js, p_js = stats.pearsonr(js_divs, uncertainty_increases)
        results['correlation'] = {
            'psi_pearson_r': float(corr_psi),
            'psi_p_value': float(p_psi),
            'js_pearson_r': float(corr_js),
            'js_p_value': float(p_js),
        }

    output_json = results_dir / "shift_uncertainty_correlation.json"
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Correlation results saved: {output_json}")
    print("\n" + "#"*80)
    print("# ANALYSIS COMPLETE")
    print("#"*80 + "\n")


if __name__ == "__main__":
    main()
