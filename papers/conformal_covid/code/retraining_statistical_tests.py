"""
Statistical Significance Tests for Retraining Experiments

Tests whether differences in retraining frequencies are statistically significant.
Critical for UAI 2026 paper revision - addresses reviewer concern about
overclaiming without statistical tests.

Usage:
    python retraining_statistical_tests.py

Author: UAI 2026 Conformal COVID Paper
Date: 2025-12-27
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


def load_retraining_results(task: str, frequency: str) -> List[float]:
    """
    Load coverage values for a retraining frequency.

    Args:
        task: Task name (e.g., 'sales-shipcond')
        frequency: Retraining frequency ('none', '1M', '3M', '6M')

    Returns:
        coverages: List of coverage values over time
    """
    results_dir = Path('papers/conformal_covid/results/retraining')
    filename = f'retrain_{frequency}_{task}.json'
    filepath = results_dir / filename

    if not filepath.exists():
        print(f"Warning: {filepath} not found")
        return []

    with open(filepath, 'r') as f:
        data = json.load(f)

    # Extract coverage values
    coverages = [entry['coverage'] * 100 for entry in data]  # Convert to percentage
    return coverages


def compute_summary_statistics(coverages: List[float]) -> Dict:
    """
    Compute summary statistics for coverage time series.

    Returns:
        stats: Dict with mean, std, min, max, median, IQR
    """
    arr = np.array(coverages)

    return {
        'mean': np.mean(arr),
        'std': np.std(arr, ddof=1),  # Sample std
        'min': np.min(arr),
        'max': np.max(arr),
        'median': np.median(arr),
        'q25': np.percentile(arr, 25),
        'q75': np.percentile(arr, 75),
        'iqr': np.percentile(arr, 75) - np.percentile(arr, 25),
        'n': len(arr),
    }


def paired_wilcoxon_test(coverage1: List[float], coverage2: List[float]) -> Tuple[float, float]:
    """
    Run Wilcoxon signed-rank test for paired samples.

    This is appropriate because we're comparing same time points across
    different retraining frequencies.

    Returns:
        statistic: Test statistic
        p_value: Two-tailed p-value
    """
    # Ensure same length
    min_len = min(len(coverage1), len(coverage2))
    coverage1 = np.array(coverage1[:min_len])
    coverage2 = np.array(coverage2[:min_len])

    # Check if all differences are zero (happens when coverage is identical)
    diff = coverage1 - coverage2
    if np.all(np.abs(diff) < 1e-10):
        # No difference, p=1.0 (perfectly not significant)
        return 0.0, 1.0

    # Wilcoxon signed-rank test (non-parametric paired test)
    # Good for non-normal distributions
    try:
        statistic, p_value = stats.wilcoxon(coverage1, coverage2, alternative='two-sided')
        return statistic, p_value
    except ValueError:
        # Fallback if test fails (e.g., all zeros)
        return 0.0, 1.0


def mann_whitney_u_test(coverage1: List[float], coverage2: List[float]) -> Tuple[float, float]:
    """
    Run Mann-Whitney U test for independent samples.

    Alternative to paired test if time points don't align perfectly.

    Returns:
        statistic: Test statistic
        p_value: Two-tailed p-value
    """
    statistic, p_value = stats.mannwhitneyu(
        coverage1, coverage2, alternative='two-sided'
    )

    return statistic, p_value


def analyze_catastrophic_task():
    """
    Analyze statistical significance for sales-shipcond (catastrophic task).
    """
    task = 'sales-shipcond'
    frequencies = ['none', '6M', '3M', '1M']

    print(f"\n{'='*80}")
    print(f"Statistical Analysis: {task} (Catastrophic Task)")
    print(f"{'='*80}\n")

    # Load data
    data = {}
    for freq in frequencies:
        coverages = load_retraining_results(task, freq)
        data[freq] = coverages
        stats_dict = compute_summary_statistics(coverages)

        print(f"{freq:8s}: Mean={stats_dict['mean']:5.1f}%, Std={stats_dict['std']:5.1f}%, "
              f"Min={stats_dict['min']:5.1f}%, Max={stats_dict['max']:5.1f}%, "
              f"Median={stats_dict['median']:5.1f}%, N={stats_dict['n']}")

    # Pairwise comparisons
    print(f"\n{'Pairwise Comparisons (Wilcoxon Signed-Rank Test)':-^80}")
    print(f"{'Comparison':<30s} {'Mean Diff':>10s} {'Statistic':>12s} {'p-value':>10s} {'Significant?':>15s}")
    print(f"{'-'*80}")

    comparisons = [
        ('3M', 'none'),    # Quarterly vs no retrain
        ('3M', '1M'),      # Quarterly vs monthly
        ('3M', '6M'),      # Quarterly vs bi-annual
        ('1M', 'none'),    # Monthly vs no retrain
        ('6M', 'none'),    # Bi-annual vs no retrain
    ]

    results = []

    for freq1, freq2 in comparisons:
        cov1 = data[freq1]
        cov2 = data[freq2]

        mean1 = np.mean(cov1)
        mean2 = np.mean(cov2)
        mean_diff = mean1 - mean2

        stat, p = paired_wilcoxon_test(cov1, cov2)

        # Determine significance
        if p < 0.001:
            sig = '***'
        elif p < 0.01:
            sig = '**'
        elif p < 0.05:
            sig = '*'
        else:
            sig = 'n.s.'

        print(f"{freq1:>8s} vs {freq2:<8s}     {mean_diff:>+7.1f}%  {stat:>12.1f}  {p:>10.4f}  {sig:>15s}")

        results.append({
            'freq1': freq1,
            'freq2': freq2,
            'mean_diff': mean_diff,
            'statistic': stat,
            'p_value': p,
            'significant': sig,
        })

    print(f"\n{'Note: * p<0.05, ** p<0.01, *** p<0.001, n.s. = not significant'}")

    return data, results


def analyze_robust_task():
    """
    Analyze statistical significance for sales-office (robust task).
    """
    task = 'sales-office'
    frequencies = ['none', '6M', '3M', '1M']

    print(f"\n{'='*80}")
    print(f"Statistical Analysis: {task} (Robust Task)")
    print(f"{'='*80}\n")

    # Load data
    data = {}
    for freq in frequencies:
        coverages = load_retraining_results(task, freq)
        data[freq] = coverages
        stats_dict = compute_summary_statistics(coverages)

        print(f"{freq:8s}: Mean={stats_dict['mean']:5.1f}%, Std={stats_dict['std']:5.1f}%, "
              f"Min={stats_dict['min']:5.1f}%, Max={stats_dict['max']:5.1f}%, "
              f"Median={stats_dict['median']:5.1f}%, N={stats_dict['n']}")

    # Test for differences (expect no significant differences)
    print(f"\n{'Pairwise Comparisons (Wilcoxon Signed-Rank Test)':-^80}")
    print(f"{'Comparison':<30s} {'Mean Diff':>10s} {'Statistic':>12s} {'p-value':>10s} {'Significant?':>15s}")
    print(f"{'-'*80}")

    comparisons = [
        ('3M', 'none'),
        ('1M', 'none'),
        ('6M', 'none'),
    ]

    for freq1, freq2 in comparisons:
        cov1 = data[freq1]
        cov2 = data[freq2]

        mean1 = np.mean(cov1)
        mean2 = np.mean(cov2)
        mean_diff = mean1 - mean2

        stat, p = paired_wilcoxon_test(cov1, cov2)

        if p < 0.05:
            sig = '*'
        else:
            sig = 'n.s.'

        print(f"{freq1:>8s} vs {freq2:<8s}     {mean_diff:>+7.1f}%  {stat:>12.1f}  {p:>10.4f}  {sig:>15s}")

    return data


def generate_updated_latex_table(catastrophic_results: List[Dict]):
    """
    Generate updated LaTeX table with significance indicators.
    """
    print(f"\n{'='*80}")
    print("Updated LaTeX Table for Paper (Table 4)")
    print(f"{'='*80}\n")

    # Find p-values for comparisons
    p_3M_vs_1M = None
    p_3M_vs_none = None
    p_3M_vs_6M = None

    for r in catastrophic_results:
        if r['freq1'] == '3M' and r['freq2'] == '1M':
            p_3M_vs_1M = r['p_value']
        elif r['freq1'] == '3M' and r['freq2'] == 'none':
            p_3M_vs_none = r['p_value']
        elif r['freq1'] == '3M' and r['freq2'] == '6M':
            p_3M_vs_6M = r['p_value']

    # Load summary stats
    task = 'sales-shipcond'
    frequencies = ['none', '6M', '3M', '1M']
    stats_summary = {}
    for freq in frequencies:
        coverages = load_retraining_results(task, freq)
        stats_summary[freq] = compute_summary_statistics(coverages)

    latex = r"""\begin{table}[h]
\centering
\caption{Retraining Frequency Impact on Catastrophic Task (sales-shipcond).
Statistical significance tested using Wilcoxon signed-rank test (paired samples
across 11 time points). * $p<0.05$, ** $p<0.01$, n.s. = not significant.}
\label{tab:retrain}
\begin{tabular}{lcccc}
\toprule
Frequency & Retrains/Year & Mean Cov. & Min Cov. & Std Cov. \\
\midrule
"""

    # Add rows
    retrains_per_year = {'none': 0, '6M': 1, '3M': 3, '1M': 10}

    for freq in ['none', '6M', '3M', '1M']:
        s = stats_summary[freq]
        retrains = retrains_per_year[freq]

        # Add significance marker for 3M row
        if freq == '3M':
            mean_str = r"\textbf{" + f"{s['mean']:.1f}\\%" + "}"
            min_str = r"\textbf{" + f"{s['min']:.1f}\\%" + "}"

            # Add footnote markers based on p-values
            markers = []
            if p_3M_vs_none and p_3M_vs_none < 0.05:
                markers.append(r"$^{\dagger}$")
            if p_3M_vs_1M and p_3M_vs_1M < 0.05:
                markers.append(r"$^{\ddagger}$")

            freq_display = r"\textbf{Quarterly (3M)}" + ''.join(markers)
        else:
            mean_str = f"{s['mean']:.1f}\\%"
            min_str = f"{s['min']:.1f}\\%"
            freq_display = freq.replace('none', 'No retrain').replace('6M', 'Bi-annual (6M)').replace('1M', 'Monthly (1M)')

        latex += f"{freq_display:25s} & {retrains:>2d} & {mean_str:>8s} & {min_str:>8s} & {s['std']:.1f}\\% \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\vspace{2mm}

\raggedright
\footnotesize
"""

    # Add significance notes
    if p_3M_vs_none:
        if p_3M_vs_none < 0.001:
            sig_vs_none = "p<0.001"
        elif p_3M_vs_none < 0.01:
            sig_vs_none = f"p={p_3M_vs_none:.3f}"
        else:
            sig_vs_none = f"p={p_3M_vs_none:.2f}"
        latex += r"$^{\dagger}$ Quarterly vs no retrain: " + sig_vs_none + r". "

    if p_3M_vs_1M:
        if p_3M_vs_1M < 0.05:
            sig_vs_monthly = f"p={p_3M_vs_1M:.2f}"
            latex += r"$^{\ddagger}$ Quarterly vs monthly: " + sig_vs_monthly + r". "
        else:
            latex += r"$^{\ddagger}$ Quarterly vs monthly: not significant (p=" + f"{p_3M_vs_1M:.2f}" + r"). "

    latex += r"""
\end{table}
"""

    print(latex)

    # Save to file
    output_dir = Path('papers/conformal_covid/results/retraining')
    output_file = output_dir / 'table4_updated_with_stats.tex'
    with open(output_file, 'w') as f:
        f.write(latex)

    print(f"\n✓ Saved updated table to {output_file}")

    return latex


def create_visualization(catastrophic_data: Dict, robust_data: Dict):
    """
    Create visualization showing coverage distributions by frequency.
    """
    print(f"\nCreating visualization...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Catastrophic task
    ax = axes[0]
    frequencies = ['none', '6M', '3M', '1M']
    labels = ['No Retrain', 'Bi-annual\n(6M)', 'Quarterly\n(3M)', 'Monthly\n(1M)']
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']

    positions = []
    data_list = []
    for i, freq in enumerate(frequencies):
        positions.append(i)
        data_list.append(catastrophic_data[freq])

    bp = ax.boxplot(data_list, positions=positions, labels=labels,
                     patch_artist=True, widths=0.6)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel('Coverage (%)', fontsize=12, fontweight='bold')
    ax.set_title('(A) Catastrophic Task (sales-shipcond)', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 100)

    # Add mean markers
    for i, freq in enumerate(frequencies):
        mean_val = np.mean(catastrophic_data[freq])
        ax.plot(i, mean_val, 'D', color='black', markersize=8, zorder=10)

    # Panel B: Robust task
    ax = axes[1]

    positions = []
    data_list = []
    for i, freq in enumerate(frequencies):
        positions.append(i)
        data_list.append(robust_data[freq])

    bp = ax.boxplot(data_list, positions=positions, labels=labels,
                     patch_artist=True, widths=0.6)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel('Coverage (%)', fontsize=12, fontweight='bold')
    ax.set_title('(B) Robust Task (sales-office)', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(95, 100.5)

    # Add mean markers
    for i, freq in enumerate(frequencies):
        mean_val = np.mean(robust_data[freq])
        ax.plot(i, mean_val, 'D', color='black', markersize=8, zorder=10)

    plt.tight_layout()

    # Save
    output_dir = Path('papers/conformal_covid/results/retraining')
    output_file = output_dir / 'retraining_statistical_comparison.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved figure to {output_file}")

    output_file_png = output_dir / 'retraining_statistical_comparison.png'
    plt.savefig(output_file_png, dpi=300, bbox_inches='tight')
    print(f"✓ Saved figure to {output_file_png}")

    plt.close()


def main():
    print(f"\n{'='*80}")
    print("Statistical Significance Tests for Retraining Experiments")
    print("UAI 2026 Paper Revision - Priority 1 Critical Fix")
    print(f"{'='*80}\n")

    # 1. Analyze catastrophic task
    catastrophic_data, catastrophic_results = analyze_catastrophic_task()

    # 2. Analyze robust task
    robust_data = analyze_robust_task()

    # 3. Generate updated LaTeX table
    latex_table = generate_updated_latex_table(catastrophic_results)

    # 4. Create visualization
    create_visualization(catastrophic_data, robust_data)

    # 5. Summary
    print(f"\n{'='*80}")
    print("Key Findings")
    print(f"{'='*80}\n")

    # Find key comparisons
    p_3M_vs_1M = None
    p_3M_vs_none = None

    for r in catastrophic_results:
        if r['freq1'] == '3M' and r['freq2'] == '1M':
            p_3M_vs_1M = r['p_value']
            mean_diff_3M_1M = r['mean_diff']
        elif r['freq1'] == '3M' and r['freq2'] == 'none':
            p_3M_vs_none = r['p_value']
            mean_diff_3M_none = r['mean_diff']

    print(f"Catastrophic Task (sales-shipcond):")
    print(f"  • Quarterly vs No Retrain: +{mean_diff_3M_none:.1f}% (p={p_3M_vs_none:.4f})")
    if p_3M_vs_none < 0.05:
        print(f"    → STATISTICALLY SIGNIFICANT improvement")
    else:
        print(f"    → Not statistically significant")

    print(f"\n  • Quarterly vs Monthly: +{mean_diff_3M_1M:.1f}% (p={p_3M_vs_1M:.4f})")
    if p_3M_vs_1M < 0.05:
        print(f"    → STATISTICALLY SIGNIFICANT difference")
    else:
        print(f"    → Not statistically significant")
        print(f"    → MUST SOFTEN CLAIM in paper ('numerically higher' not 'outperforms')")

    print(f"\n{'='*80}")
    print("Recommendations for Paper Text")
    print(f"{'='*80}\n")

    if p_3M_vs_1M < 0.05:
        print("✓ Can keep 'quarterly outperforms monthly' language")
    else:
        print("⚠️  MUST CHANGE 'outperforms' to 'shows numerically higher mean coverage'")
        print("   Current text (line 311): 'quarterly retraining outperforms monthly'")
        print("   Revised text: 'quarterly retraining shows highest mean coverage (41.1%), though the difference vs monthly (32.0%) is not statistically significant (p=XX)'")

    print(f"\n{'='*80}")
    print("Analysis Complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
