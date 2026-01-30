"""
Generate Revised Figures for Reviewer Response

Changes from original:
1. Figure 1 Panel A: Use median + IQR error bars for high-variance tasks
2. Add annotations for bimodal distributions
3. Generate retraining figure excluding month 9-10 anomaly
4. Enhanced Table 1 with both mean±std and median(IQR)

Usage:
    python code/generate_figures_revised.py

Output:
    figure1_main_results_REVISED.pdf/png
    figure_retraining_CLEANED.pdf/png
    table1_enhanced.tex

Author: UAI 2026 Reviewer Response
Date: 2025-12-27
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pickle
from pathlib import Path

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['axes.labelsize'] = 10

# Load actual 50-seed ensemble results
def load_ensemble_results():
    """Load 50-seed results with median/IQR."""
    results_file = Path(__file__).parent.parent / 'results' / 'ensemble_50seeds.pkl'

    if not results_file.exists():
        print(f"Warning: {results_file} not found, using placeholder data")
        return None

    with open(results_file, 'rb') as f:
        data = pickle.load(f)

    return data


def generate_figure1_revised():
    """Generate Figure 1 Panel A with MEDIAN + IQR error bars."""

    # Load real data
    ensemble_data = load_ensemble_results()

    if ensemble_data is None:
        print("Cannot generate revised figure without ensemble results")
        return

    # Prepare data
    task_results = []
    for r in ensemble_data:
        task_short = r['task'].replace('sales-', 's-').replace('item-', 'i-')

        # Calculate drop from median values (more robust)
        drop_median = (r['val_coverage_median'] - r['test_coverage_median']) * 100

        task_results.append({
            'task': task_short,
            'full_name': r['task'],
            'val_mean': r['val_coverage_mean'] * 100,
            'val_std': r['val_coverage_std'] * 100,
            'val_median': r['val_coverage_median'] * 100,
            'val_iqr': r['val_coverage_iqr'] * 100,
            'test_mean': r['test_coverage_mean'] * 100,
            'test_std': r['test_coverage_std'] * 100,
            'test_median': r['test_coverage_median'] * 100,
            'test_iqr': r['test_coverage_iqr'] * 100,
            'drop': drop_median,
            'classes': r['num_classes'],
        })

    # Sort by drop
    task_results = sorted(task_results, key=lambda x: -x['drop'])

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    tasks = [r['task'] for r in task_results]
    val_medians = np.array([r['val_median'] for r in task_results])
    test_medians = np.array([r['test_median'] for r in task_results])
    val_iqrs = np.array([r['val_iqr'] for r in task_results])
    test_iqrs = np.array([r['test_iqr'] for r in task_results])
    test_stds = np.array([r['test_std'] for r in task_results])

    x = np.arange(len(tasks))
    width = 0.35

    # Colors
    val_color = '#2ecc71'
    test_color = '#e74c3c'

    # Bar plots with median values
    bars1 = ax.bar(x - width/2, val_medians, width,
                   label='Val (median)', color=val_color,
                   alpha=0.85, edgecolor='white', linewidth=0.5)

    bars2 = ax.bar(x + width/2, test_medians, width,
                   label='Test (median)', color=test_color,
                   alpha=0.85, edgecolor='white', linewidth=0.5)

    # Error bars using IQR (more robust than std for skewed distributions)
    ax.errorbar(x - width/2, val_medians, yerr=val_iqrs/2,
                fmt='none', ecolor='darkgreen', elinewidth=2, capsize=4,
                capthick=2, alpha=0.7)

    ax.errorbar(x + width/2, test_medians, yerr=test_iqrs/2,
                fmt='none', ecolor='darkred', elinewidth=2, capsize=4,
                capthick=2, alpha=0.7)

    # Add markers for high-variance tasks (std > 30%)
    for i, (task, std) in enumerate(zip(tasks, test_stds)):
        if std > 30:
            # Add asterisk above bar
            y_pos = max(test_medians[i] + test_iqrs[i]/2 + 5, 10)
            ax.text(x[i] + width/2, y_pos, '*',
                   fontsize=16, fontweight='bold', ha='center',
                   color='red')

    # Target line
    ax.axhline(90, color='black', linestyle='--', lw=2,
              label='Target (90%)')

    # Configure axes
    ax.set_ylabel('Coverage (%)')
    ax.set_title('Panel A: Coverage Degradation (Median + IQR across 50 seeds)',
                fontweight='bold', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=9)
    ax.set_ylim([0, 105])
    ax.legend(loc='lower left', fontsize=9)
    ax.set_xlabel('Task')

    # Add footnote about high variance
    ax.text(0.98, 0.02,
           '*High variance (std>30%) indicates bimodal distribution',
           transform=ax.transAxes, fontsize=8, ha='right', va='bottom',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save
    output_path = Path(__file__).parent.parent / 'figure1_panel_A_REVISED.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_path}")
    plt.close()


def generate_retraining_figure_cleaned():
    """Generate retraining figure EXCLUDING months 9-10 (data anomaly)."""

    # Load retraining results
    results_dir = Path(__file__).parent.parent / 'results' / 'retraining'

    results = {}
    for freq in ['none', '1M', '3M', '6M']:
        pkl_file = results_dir / f'retrain_{freq}_sales-shipcond.pkl'
        if pkl_file.exists():
            with open(pkl_file, 'rb') as f:
                results[freq] = pickle.load(f)

    if not results:
        print("Warning: No retraining results found")
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    colors = {
        'none': '#d62728',
        '6M': '#ff7f0e',
        '3M': '#2ca02c',
        '1M': '#1f77b4',
    }

    labels = {
        'none': 'No retrain',
        '6M': 'Semi-annual (2/year)',
        '3M': 'Quarterly (4/year)',
        '1M': 'Monthly (12/year)',
    }

    # Plot each frequency, EXCLUDING months 9-10
    for freq in ['none', '6M', '3M', '1M']:
        if freq not in results:
            continue

        data = results[freq]

        # Filter out months 9-10
        filtered = [r for r in data if r['month'] < 9]

        months = [r['month'] for r in filtered]
        coverages = [r['coverage'] * 100 for r in filtered]
        retrains = [r['retrained'] for r in filtered]

        # Plot
        ax.plot(months, coverages, color=colors[freq], linewidth=2.5,
               label=labels[freq], marker='o', markersize=5, alpha=0.9)

        # Mark retrain points
        if freq != 'none':
            for i, retrained in enumerate(retrains):
                if retrained and i > 0:
                    ax.axvline(x=months[i], color=colors[freq], alpha=0.2,
                             linestyle='--', linewidth=1)

    # Target line
    ax.axhline(y=90, color='black', linestyle=':', linewidth=1.5,
              alpha=0.5, label='Target (90%)')

    # Add shaded region for excluded months
    ax.axvspan(8.5, 11, alpha=0.1, color='gray', label='Months 9-10 (excluded)')

    # Annotation about exclusion
    ax.text(9.5, 50, 'Data anomaly\n(excluded)',
           ha='center', va='center', fontsize=10, style='italic',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Configure axes
    ax.set_xlabel('Month Index (0=Feb 2020, 8=Oct 2020)')
    ax.set_ylabel('Coverage (%)')
    ax.set_title('Retraining Restores Coverage (Months 9-10 excluded due to data anomaly)',
                fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_ylim([0, 100])
    ax.set_xlim([-0.5, 11])

    plt.tight_layout()

    # Save
    output_path = Path(__file__).parent.parent / 'figure_retraining_CLEANED.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_path}")
    plt.close()


def generate_enhanced_table1():
    """Generate enhanced Table 1 with both mean±std and median(IQR)."""

    ensemble_data = load_ensemble_results()

    if ensemble_data is None:
        print("Cannot generate table without ensemble results")
        return

    # LaTeX table
    table_lines = [
        "% Enhanced Table 1 with Median + IQR",
        "% Generated: 2025-12-27",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Coverage Degradation Under COVID-19 Distribution Shift (50 model seeds). ",
        "High-variance tasks ($^*$) show severely skewed distributions where mean $\\pm$ std ",
        "is misleading; median and IQR provide robust statistics.}",
        "\\label{tab:main_results}",
        "\\small",
        "\\begin{tabular}{@{}lcccccc@{}}",
        "\\toprule",
        "\\multirow{2}{*}{Task} & \\multirow{2}{*}{Classes} &",
        "\\multicolumn{2}{c}{Val Coverage (\\%)} &",
        "\\multicolumn{2}{c}{Test Coverage (\\%)} &",
        "\\multirow{2}{*}{Cat} \\\\",
        "\\cmidrule(lr){3-4} \\cmidrule(lr){5-6}",
        "& & Mean$\\pm$SD & Med(IQR) & Mean$\\pm$SD & Med(IQR) & \\\\",
        "\\midrule",
    ]

    # Sort by drop (descending)
    sorted_data = sorted(ensemble_data,
                        key=lambda x: (x['val_coverage_median'] - x['test_coverage_median']),
                        reverse=True)

    for r in sorted_data:
        task = r['task'].replace('sales-', 's-').replace('item-', 'i-')
        classes = r['num_classes']

        # Val statistics
        val_mean = r['val_coverage_mean'] * 100
        val_std = r['val_coverage_std'] * 100
        val_med = r['val_coverage_median'] * 100
        val_iqr = r['val_coverage_iqr'] * 100

        # Test statistics
        test_mean = r['test_coverage_mean'] * 100
        test_std = r['test_coverage_std'] * 100
        test_med = r['test_coverage_median'] * 100
        test_iqr = r['test_coverage_iqr'] * 100

        # Determine category
        drop = (r['val_coverage_median'] - r['test_coverage_median']) * 100
        if drop > 70:
            cat = "SEV"
        elif drop > 15:
            cat = "MOD"
        else:
            cat = "ROB"

        # Mark high variance
        if test_std > 30:
            cat += "$^*$"

        # Highlight extreme medians
        test_med_str = f"{test_med:.1f}" if test_med > 5 else f"\\textbf{{{test_med:.1f}}}"

        # Build row
        row = (f"{task} & {classes} & "
               f"{val_mean:.1f}$\\pm${val_std:.1f} & {val_med:.1f}({val_iqr:.1f}) & "
               f"{test_mean:.1f}$\\pm${test_std:.1f} & {test_med_str}({test_iqr:.1f}) & "
               f"{cat} \\\\")

        table_lines.append(row)

    table_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\vspace{1mm}",
        "",
        "\\raggedright",
        "\\footnotesize",
        "SEV = Severe ($>$70\\% drop), MOD = Moderate (15-70\\%), ROB = Robust ($<$15\\%).",
        "$^*$High model variance (std $>$ 30\\%) indicates knife-edge regime where small ",
        "initialization changes lead to qualitatively different learned representations.",
        "\\end{table*}",
    ])

    # Save
    output_file = Path(__file__).parent.parent / 'table1_enhanced.tex'
    with open(output_file, 'w') as f:
        f.write('\n'.join(table_lines))

    print(f"✓ Saved: {output_file}")


def main():
    print("=" * 80)
    print("GENERATING REVISED FIGURES FOR REVIEWER RESPONSE")
    print("=" * 80)
    print()

    print("1. Figure 1 Panel A with median + IQR...")
    generate_figure1_revised()
    print()

    print("2. Cleaned retraining figure (excluding month 9-10 anomaly)...")
    generate_retraining_figure_cleaned()
    print()

    print("3. Enhanced Table 1 with median + IQR...")
    generate_enhanced_table1()
    print()

    print("=" * 80)
    print("DONE! Generated files:")
    print("  - figure1_panel_A_REVISED.pdf/png")
    print("  - figure_retraining_CLEANED.pdf/png")
    print("  - table1_enhanced.tex")
    print("=" * 80)


if __name__ == "__main__":
    main()
