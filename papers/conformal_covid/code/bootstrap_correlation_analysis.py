"""
Bootstrap Confidence Intervals for Correlation Analysis

Computes bootstrap CIs and significance tests for:
1. Feature Jaccard ↔ Coverage Drop correlation
2. Entropy ↔ Coverage Drop correlation (for 0% Jaccard tasks)

This addresses UAI reviewer requirements for statistical rigor.

Usage:
    python bootstrap_correlation_analysis.py

Output:
    - Bootstrap CIs for correlation coefficients
    - Permutation test p-values
    - Publication-ready LaTeX table
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Number of bootstrap samples
N_BOOTSTRAP = 10000
N_PERMUTATION = 10000

# =============================================================================
# Data from experiments
# =============================================================================

# Main results with Jaccard similarity and entropy
DATA = [
    {'task': 'sales-shipcond', 'jaccard': 0.02, 'drop': 93.1, 'entropy': 3.16, 'classes': 45},
    {'task': 'sales-group', 'jaccard': 0.02, 'drop': 86.7, 'entropy': 7.61, 'classes': 459},
    {'task': 'sales-payterms', 'jaccard': 0.05, 'drop': 33.8, 'entropy': 4.21, 'classes': 137},
    {'task': 'item-plant', 'jaccard': 0.08, 'drop': 29.1, 'entropy': 2.94, 'classes': 35},
    {'task': 'item-shippoint', 'jaccard': 0.06, 'drop': 18.9, 'entropy': 3.42, 'classes': 69},
    {'task': 'sales-incoterms', 'jaccard': 0.50, 'drop': 3.6, 'entropy': 2.08, 'classes': 13},
    {'task': 'item-incoterms', 'jaccard': 0.58, 'drop': 0.5, 'entropy': 1.83, 'classes': 13},
    {'task': 'sales-office', 'jaccard': 0.61, 'drop': 0.1, 'entropy': 0.05, 'classes': 25},
]

df = pd.DataFrame(DATA)


# =============================================================================
# Bootstrap Functions
# =============================================================================

def bootstrap_correlation(x, y, n_bootstrap=N_BOOTSTRAP, method='pearson'):
    """
    Compute bootstrap confidence interval for correlation coefficient.

    Args:
        x: First variable
        y: Second variable
        n_bootstrap: Number of bootstrap samples
        method: 'pearson' or 'spearman'

    Returns:
        dict with point estimate, CI, and bootstrap distribution
    """
    n = len(x)

    # Compute observed correlation
    if method == 'pearson':
        r_obs, _ = stats.pearsonr(x, y)
    elif method == 'spearman':
        r_obs, _ = stats.spearmanr(x, y)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Bootstrap resampling
    correlations = []
    for _ in range(n_bootstrap):
        # Resample with replacement
        idx = np.random.choice(n, size=n, replace=True)
        x_boot = x[idx]
        y_boot = y[idx]

        # Compute correlation
        if method == 'pearson':
            r, _ = stats.pearsonr(x_boot, y_boot)
        else:
            r, _ = stats.spearmanr(x_boot, y_boot)
        correlations.append(r)

    correlations = np.array(correlations)

    # Compute 95% CI using percentile method
    ci_lower = np.percentile(correlations, 2.5)
    ci_upper = np.percentile(correlations, 97.5)

    # Compute standard error
    se = np.std(correlations)

    return {
        'r': r_obs,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'se': se,
        'bootstrap_dist': correlations,
        'method': method,
    }


def permutation_test(x, y, n_permutations=N_PERMUTATION, method='pearson'):
    """
    Permutation test for correlation coefficient.

    Tests H0: No correlation between x and y

    Args:
        x: First variable
        y: Second variable
        n_permutations: Number of permutations
        method: 'pearson' or 'spearman'

    Returns:
        dict with p-value and permutation distribution
    """
    # Compute observed correlation
    if method == 'pearson':
        r_obs, _ = stats.pearsonr(x, y)
    else:
        r_obs, _ = stats.spearmanr(x, y)

    # Permutation test
    perm_correlations = []
    for _ in range(n_permutations):
        # Permute y (break association with x)
        y_perm = np.random.permutation(y)

        # Compute correlation
        if method == 'pearson':
            r, _ = stats.pearsonr(x, y_perm)
        else:
            r, _ = stats.spearmanr(x, y_perm)
        perm_correlations.append(r)

    perm_correlations = np.array(perm_correlations)

    # Two-tailed p-value
    p_value = np.mean(np.abs(perm_correlations) >= np.abs(r_obs))

    return {
        'r': r_obs,
        'p_value': p_value,
        'perm_dist': perm_correlations,
        'method': method,
    }


# =============================================================================
# Analysis 1: Jaccard ↔ Coverage Drop (All Tasks)
# =============================================================================

def analyze_jaccard_correlation():
    """Analyze correlation between Jaccard similarity and coverage drop."""
    print("=" * 80)
    print("ANALYSIS 1: Feature Jaccard ↔ Coverage Drop")
    print("=" * 80)
    print(f"Sample size: n = {len(df)}")
    print()

    x = df['jaccard'].values
    y = df['drop'].values

    # Pearson correlation
    print("Pearson Correlation:")
    print("-" * 80)

    boot_pearson = bootstrap_correlation(x, y, method='pearson')
    perm_pearson = permutation_test(x, y, method='pearson')

    print(f"  r = {boot_pearson['r']:.3f}")
    print(f"  95% Bootstrap CI: [{boot_pearson['ci_lower']:.3f}, {boot_pearson['ci_upper']:.3f}]")
    print(f"  SE = {boot_pearson['se']:.3f}")
    print(f"  p-value (permutation): {perm_pearson['p_value']:.4f}")

    # Also compute standard p-value for comparison
    r_standard, p_standard = stats.pearsonr(x, y)
    print(f"  p-value (parametric): {p_standard:.4f}")
    print()

    # Spearman correlation (more robust)
    print("Spearman Correlation (robust to outliers):")
    print("-" * 80)

    boot_spearman = bootstrap_correlation(x, y, method='spearman')
    perm_spearman = permutation_test(x, y, method='spearman')

    print(f"  ρ = {boot_spearman['r']:.3f}")
    print(f"  95% Bootstrap CI: [{boot_spearman['ci_lower']:.3f}, {boot_spearman['ci_upper']:.3f}]")
    print(f"  SE = {boot_spearman['se']:.3f}")
    print(f"  p-value (permutation): {perm_spearman['p_value']:.4f}")
    print()

    # Interpretation
    if perm_pearson['p_value'] < 0.05:
        print("✓ RESULT: Statistically significant negative correlation (p < 0.05)")
        print("  → Tasks with low feature overlap experience larger coverage drops")
    else:
        print("✗ WARNING: Correlation not statistically significant (p ≥ 0.05)")
        print("  → Cannot reject null hypothesis of no correlation")
    print()

    return {
        'pearson': boot_pearson,
        'pearson_pvalue': perm_pearson['p_value'],
        'spearman': boot_spearman,
        'spearman_pvalue': perm_spearman['p_value'],
    }


# =============================================================================
# Analysis 2: Entropy ↔ Coverage Drop (Low Jaccard Tasks Only)
# =============================================================================

def analyze_entropy_correlation():
    """Analyze correlation between entropy and coverage drop for low-Jaccard tasks."""
    print("=" * 80)
    print("ANALYSIS 2: Entropy ↔ Coverage Drop (Low Jaccard Tasks, J < 0.1)")
    print("=" * 80)

    # Filter to low-Jaccard tasks
    df_low_jaccard = df[df['jaccard'] < 0.1].copy()
    print(f"Sample size: n = {len(df_low_jaccard)}")
    print(f"Tasks included: {', '.join(df_low_jaccard['task'].values)}")
    print()

    if len(df_low_jaccard) < 3:
        print("⚠ WARNING: Too few tasks (n < 3) for meaningful correlation")
        return None

    x = df_low_jaccard['entropy'].values
    y = df_low_jaccard['drop'].values

    # Pearson correlation
    print("Pearson Correlation:")
    print("-" * 80)

    boot_pearson = bootstrap_correlation(x, y, method='pearson')
    perm_pearson = permutation_test(x, y, method='pearson')

    print(f"  r = {boot_pearson['r']:.3f}")
    print(f"  95% Bootstrap CI: [{boot_pearson['ci_lower']:.3f}, {boot_pearson['ci_upper']:.3f}]")
    print(f"  SE = {boot_pearson['se']:.3f}")
    print(f"  p-value (permutation): {perm_pearson['p_value']:.4f}")
    print()

    # Spearman correlation
    print("Spearman Correlation:")
    print("-" * 80)

    boot_spearman = bootstrap_correlation(x, y, method='spearman')
    perm_spearman = permutation_test(x, y, method='spearman')

    print(f"  ρ = {boot_spearman['r']:.3f}")
    print(f"  95% Bootstrap CI: [{boot_spearman['ci_lower']:.3f}, {boot_spearman['ci_upper']:.3f}]")
    print(f"  SE = {boot_spearman['se']:.3f}")
    print(f"  p-value (permutation): {perm_spearman['p_value']:.4f}")
    print()

    # Interpretation
    if perm_pearson['p_value'] < 0.05:
        print("✓ RESULT: Statistically significant correlation (p < 0.05)")
        print("  → Among low-overlap tasks, entropy predicts vulnerability")
    else:
        print("✗ WARNING: Correlation not statistically significant (p ≥ 0.05)")
        print("  → Small sample size (n = {}) limits statistical power".format(len(df_low_jaccard)))
    print()

    return {
        'pearson': boot_pearson,
        'pearson_pvalue': perm_pearson['p_value'],
        'spearman': boot_spearman,
        'spearman_pvalue': perm_spearman['p_value'],
        'n': len(df_low_jaccard),
    }


# =============================================================================
# Visualization
# =============================================================================

def plot_bootstrap_distributions(results_jaccard, results_entropy):
    """Plot bootstrap distributions with CIs."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Jaccard correlation
    ax = axes[0]
    boot_dist = results_jaccard['pearson']['bootstrap_dist']
    ax.hist(boot_dist, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(results_jaccard['pearson']['r'], color='red', linestyle='--',
               linewidth=2, label=f"Observed r = {results_jaccard['pearson']['r']:.3f}")
    ax.axvline(results_jaccard['pearson']['ci_lower'], color='orange', linestyle=':',
               linewidth=1.5, label='95% CI')
    ax.axvline(results_jaccard['pearson']['ci_upper'], color='orange', linestyle=':', linewidth=1.5)
    ax.set_xlabel('Correlation Coefficient')
    ax.set_ylabel('Frequency')
    ax.set_title('Bootstrap Distribution: Jaccard ↔ Coverage Drop')
    ax.legend()
    ax.grid(alpha=0.3)

    # Entropy correlation
    if results_entropy is not None:
        ax = axes[1]
        boot_dist = results_entropy['pearson']['bootstrap_dist']
        ax.hist(boot_dist, bins=50, alpha=0.7, color='coral', edgecolor='black')
        ax.axvline(results_entropy['pearson']['r'], color='red', linestyle='--',
                   linewidth=2, label=f"Observed r = {results_entropy['pearson']['r']:.3f}")
        ax.axvline(results_entropy['pearson']['ci_lower'], color='orange', linestyle=':',
                   linewidth=1.5, label='95% CI')
        ax.axvline(results_entropy['pearson']['ci_upper'], color='orange', linestyle=':', linewidth=1.5)
        ax.set_xlabel('Correlation Coefficient')
        ax.set_ylabel('Frequency')
        ax.set_title('Bootstrap Distribution: Entropy ↔ Coverage Drop\n(Low Jaccard Tasks Only)')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()

    # Save
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "bootstrap_distributions.pdf", bbox_inches='tight', dpi=300)
    plt.savefig(output_dir / "bootstrap_distributions.png", bbox_inches='tight', dpi=300)
    print(f"✓ Saved bootstrap distribution plots to {output_dir}")


# =============================================================================
# LaTeX Table Generator
# =============================================================================

def generate_latex_table(results_jaccard, results_entropy):
    """Generate publication-ready LaTeX table."""

    latex = r"""
\begin{table}[t]
\centering
\caption{Statistical Analysis of Correlation Coefficients}
\label{tab:correlations}
\begin{tabular}{lcccc}
\toprule
Analysis & $r$ & 95\% CI & $p$-value & Interpretation \\
\midrule
"""

    # Jaccard correlation
    r = results_jaccard['pearson']['r']
    ci_low = results_jaccard['pearson']['ci_lower']
    ci_high = results_jaccard['pearson']['ci_upper']
    p = results_jaccard['pearson_pvalue']

    sig_marker = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

    latex += f"Jaccard $\\leftrightarrow$ Drop & {r:.2f} & [{ci_low:.2f}, {ci_high:.2f}] & "
    latex += f"{p:.3f}{sig_marker} & Strong negative \\\\\n"

    # Entropy correlation
    if results_entropy is not None:
        r = results_entropy['pearson']['r']
        ci_low = results_entropy['pearson']['ci_lower']
        ci_high = results_entropy['pearson']['ci_upper']
        p = results_entropy['pearson_pvalue']
        n = results_entropy['n']

        sig_marker = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

        latex += f"Entropy $\\leftrightarrow$ Drop & {r:.2f} & [{ci_low:.2f}, {ci_high:.2f}] & "
        latex += f"{p:.3f}{sig_marker} & Moderate positive \\\\\n"
        latex += f"\\quad (low Jaccard, $n={n}$) & & & & \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\vspace{2mm}
\small{$p$-values from permutation test (10,000 permutations).
*$p<0.05$, **$p<0.01$, ***$p<0.001$}
\end{table}
"""

    print()
    print("=" * 80)
    print("LATEX TABLE (copy to paper)")
    print("=" * 80)
    print(latex)

    # Save to file
    output_dir = Path(__file__).parent.parent / "results"
    with open(output_dir / "correlation_table.tex", 'w') as f:
        f.write(latex)
    print(f"✓ Saved LaTeX table to {output_dir}/correlation_table.tex")


# =============================================================================
# Summary for Paper
# =============================================================================

def print_paper_text(results_jaccard, results_entropy):
    """Generate text for the paper."""

    print()
    print("=" * 80)
    print("TEXT FOR PAPER (Section 5.3 - Correlation Analysis)")
    print("=" * 80)
    print()

    r_j = results_jaccard['pearson']['r']
    ci_j_low = results_jaccard['pearson']['ci_lower']
    ci_j_high = results_jaccard['pearson']['ci_upper']
    p_j = results_jaccard['pearson_pvalue']

    print("Feature temporal stability is the primary predictor of coverage failure.")
    print(f"We find a strong negative correlation between Jaccard similarity and ")
    print(f"coverage drop (r = {r_j:.2f}, 95% CI [{ci_j_low:.2f}, {ci_j_high:.2f}], ")
    print(f"p = {p_j:.3f}, permutation test). This indicates that tasks with low ")
    print("feature overlap experience significantly larger coverage degradation.")
    print()

    if results_entropy is not None:
        r_e = results_entropy['pearson']['r']
        ci_e_low = results_entropy['pearson']['ci_lower']
        ci_e_high = results_entropy['pearson']['ci_upper']
        p_e = results_entropy['pearson_pvalue']
        n_e = results_entropy['n']

        print(f"Among tasks with low feature overlap (Jaccard < 0.1, n = {n_e}), ")
        print(f"we observe a moderate positive correlation between entropy and ")
        print(f"coverage drop (r = {r_e:.2f}, 95% CI [{ci_e_low:.2f}, {ci_e_high:.2f}], ")
        print(f"p = {p_e:.3f}). However, the small sample size limits statistical power.")

    print()


# =============================================================================
# Main
# =============================================================================

def main():
    """Run all analyses."""

    print()
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 15 + "BOOTSTRAP CORRELATION ANALYSIS" + " " * 33 + "║")
    print("║" + " " * 15 + "Conformal Prediction Under COVID" + " " * 31 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    print(f"Bootstrap samples: {N_BOOTSTRAP:,}")
    print(f"Permutation samples: {N_PERMUTATION:,}")
    print()

    # Analysis 1: Jaccard correlation
    results_jaccard = analyze_jaccard_correlation()

    # Analysis 2: Entropy correlation
    results_entropy = analyze_entropy_correlation()

    # Generate visualizations
    plot_bootstrap_distributions(results_jaccard, results_entropy)

    # Generate LaTeX table
    generate_latex_table(results_jaccard, results_entropy)

    # Generate paper text
    print_paper_text(results_jaccard, results_entropy)

    # Save results
    output_dir = Path(__file__).parent.parent / "results"
    results = {
        'jaccard_analysis': {
            'r': results_jaccard['pearson']['r'],
            'ci': [results_jaccard['pearson']['ci_lower'], results_jaccard['pearson']['ci_upper']],
            'p_value': results_jaccard['pearson_pvalue'],
            'se': results_jaccard['pearson']['se'],
        },
        'entropy_analysis': {
            'r': results_entropy['pearson']['r'] if results_entropy else None,
            'ci': [results_entropy['pearson']['ci_lower'], results_entropy['pearson']['ci_upper']] if results_entropy else None,
            'p_value': results_entropy['pearson_pvalue'] if results_entropy else None,
            'n': results_entropy['n'] if results_entropy else None,
        } if results_entropy else None,
    }

    with open(output_dir / "bootstrap_correlation_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print()
    print("=" * 80)
    print(f"✓ ALL RESULTS SAVED TO {output_dir}")
    print("=" * 80)
    print()
    print("RECOMMENDATION FOR UAI SUBMISSION:")
    print("-" * 80)

    if results_jaccard['pearson_pvalue'] < 0.05:
        print("✓ Jaccard correlation is statistically significant")
        print("  → This is your strongest finding, emphasize it in the paper")
    else:
        print("✗ WARNING: Jaccard correlation not significant - need more data or better metric")

    if results_entropy and results_entropy['pearson_pvalue'] < 0.05:
        print("✓ Entropy correlation is statistically significant")
    else:
        print("⚠ Entropy correlation weak or not significant (small sample size)")
        print("  → Acknowledge this limitation in the paper")

    print()
    print("NEXT STEPS:")
    print("1. Copy the LaTeX table to your paper (Section 5.3)")
    print("2. Update paper text with CI and p-values")
    print("3. Include bootstrap_distributions.pdf in supplementary materials")
    print()

    return results


if __name__ == "__main__":
    results = main()
