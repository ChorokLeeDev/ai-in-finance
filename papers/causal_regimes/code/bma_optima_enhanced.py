"""
Enhanced BMA Visualization with Log-Scale Weights and Sensitivity Analysis
"""

import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import LogFormatterSciNotation
import os

RESULTS_DIR = '/sessions/festive-youthful-mccarthy/mnt/causal_regimes/results'
FIGURES_DIR = '/sessions/festive-youthful-mccarthy/mnt/causal_regimes/figures'


def create_enhanced_figure():
    """Create enhanced publication figure with better weight visualization."""

    # Load results
    with open(os.path.join(RESULTS_DIR, 'bma_optima_results.json')) as f:
        results = json.load(f)

    with open(os.path.join(RESULTS_DIR, 'bic_optima_comparison.json')) as f:
        bic_data = json.load(f)

    clusters = results['bma_weights']['clusters']
    granger_results = results['granger_elevated_hml_smb']['cluster_results']
    bma_est = results['granger_elevated_hml_smb']['bma_estimate']

    # Create figure with 3 subpanels
    fig = plt.figure(figsize=(17, 5.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1.3, 1.4], hspace=0.3, wspace=0.35)

    # ===== Panel A: BIC and Log-Weights =====
    ax = fig.add_subplot(gs[0])

    cluster_ids = np.array([c['cluster_id'] for c in clusters])
    bic_vals = np.array([c['bic'] for c in clusters])
    weights = np.array([c['posterior_weight'] for c in clusters])

    # Normalize BIC relative to minimum
    bic_relative = bic_vals - np.min(bic_vals)

    # Sort by BIC for display
    sort_idx = np.argsort(bic_relative)
    cluster_ids_sorted = cluster_ids[sort_idx]
    bic_relative_sorted = bic_relative[sort_idx]
    weights_sorted = weights[sort_idx]

    # Color scale based on weights
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(clusters)))
    colors_sorted = colors[sort_idx]

    x_pos = np.arange(len(clusters))

    # Primary axis: BIC relative to minimum
    bars1 = ax.bar(x_pos - 0.2, bic_relative_sorted, width=0.4, label='BIC - BIC min',
                    color=colors_sorted, alpha=0.7, edgecolor='black', linewidth=1.5)

    # Secondary axis: Log-scale weights
    ax2 = ax.twinx()
    # Add small constant to avoid log(0)
    weights_plot = np.maximum(weights_sorted, 1e-120)
    ax2.semilogy(x_pos, weights_sorted, 'o-', color='steelblue', linewidth=2.5, markersize=8,
                  label='Posterior Weight', zorder=3)

    ax.set_xlabel('Cluster (sorted by BIC)', fontsize=12, fontweight='bold')
    ax.set_ylabel('BIC - BIC(min)', fontsize=12, fontweight='bold', color='black')
    ax2.set_ylabel('Posterior Model Weight (log scale)', fontsize=12, fontweight='bold',
                    color='steelblue')
    ax.set_title('(A) Cluster Bayesian Model Weights', fontsize=13, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'C{int(cid)}' for cid in cluster_ids_sorted], fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars1, bic_relative_sorted)):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, val + 5, f'{val:.0f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Add value labels on weights
    for i, (x, w) in enumerate(zip(x_pos, weights_sorted)):
        if w > 1e-10:
            label = f'{w:.2e}' if w < 0.01 else f'{w:.4f}'
            ax2.text(x, w * 5, label, ha='center', va='bottom', fontsize=8, color='darkblue',
                    fontweight='bold')

    ax2.set_ylim([1e-125, 10])
    ax.set_ylim([0, max(bic_relative_sorted) * 1.15])

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)

    # ===== Panel B: Per-Cluster Granger p-values =====
    ax = fig.add_subplot(gs[1])

    p_values = np.array([c['cluster_p_value'] for c in granger_results])
    cluster_ids_granger = np.array([c['cluster_id'] for c in granger_results])

    # Find sort order from panel A
    sort_order_granger = [np.where(cluster_ids_sorted == cid)[0][0] for cid in cluster_ids_granger]
    colors_p = colors_sorted[sort_order_granger]

    ax.scatter(cluster_ids_granger, p_values, s=250, c=colors_p, alpha=0.6,
               edgecolor='black', linewidth=2, zorder=3, label='Cluster p-value')

    # Significance threshold
    ax.axhline(0.05, color='red', linestyle='--', linewidth=2.5, label='α=0.05', zorder=2)

    ax.set_xlabel('Cluster', fontsize=12, fontweight='bold')
    ax.set_ylabel('HAC p-value (HML→SMB)', fontsize=12, fontweight='bold')
    ax.set_title('(B) Per-Cluster Granger Causality', fontsize=13, fontweight='bold')
    ax.set_yscale('log')
    ax.set_xticks(cluster_ids_granger)
    ax.set_xticklabels([f'C{int(cid)}' for cid in cluster_ids_granger], fontsize=11)
    ax.grid(True, alpha=0.3, which='both', linestyle='--')
    ax.set_ylim([1e-8, 1.5])
    ax.legend(loc='lower left', fontsize=10)

    # Add significance indicators
    for cid, p in zip(cluster_ids_granger, p_values):
        sig = '***' if p < 0.05 else 'ns'
        ax.text(cid, p * 0.7, sig, ha='center', va='center', fontsize=10,
               fontweight='bold', color='white',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    # ===== Panel C: BMA Result =====
    ax = fig.add_subplot(gs[2])

    # Show individual cluster results
    x_offset = np.linspace(0.7, 1.3, len(granger_results))
    for x, result in zip(x_offset, granger_results):
        p = result['cluster_p_value']
        color = 'green' if p < 0.05 else 'red'
        weight = result['weight']
        size = 100 + 150 * (weight / np.max([r['weight'] for r in granger_results]))
        ax.scatter(x, p, s=max(50, size), c=color, alpha=0.4, edgecolor='black', linewidth=1, zorder=2)

    # BMA estimate with confidence interval
    point = bma_est['point']
    ci_lower = bma_est['ci_lower']
    ci_upper = bma_est['ci_upper']

    ax.errorbar(2, point, yerr=[[point - ci_lower], [ci_upper - point]],
                fmt='D', markersize=14, color='darkblue', ecolor='darkblue', elinewidth=3,
                capsize=12, capthick=2.5, label='BMA ± 95% CI', zorder=4, alpha=0.9)

    # Significance threshold
    ax.axhline(0.05, color='red', linestyle='--', linewidth=2.5, label='α=0.05', zorder=2)

    ax.set_xlim([0.4, 2.6])
    ax.set_yscale('log')
    ax.set_ylabel('HAC p-value', fontsize=12, fontweight='bold')
    ax.set_title('(C) Bayesian Model Average', fontsize=13, fontweight='bold')
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Cluster\nResults\n(variability)', 'BMA\nEstimate\n(posterior avg)'],
                       fontsize=11)
    ax.set_ylim([1e-8, 1.5])
    ax.grid(True, alpha=0.3, which='both', linestyle='--')
    ax.legend(loc='lower left', fontsize=11)

    # Annotation for BMA result
    sig_text = 'Significant\nat 5% level' if point < 0.05 else 'Not significant\nat 5% level'
    ax.text(2, point * 0.45, sig_text, fontsize=11, fontweight='bold', ha='center',
           bbox=dict(boxstyle='round,pad=0.7', facecolor='yellow', alpha=0.8, edgecolor='black', linewidth=2))

    # Add p-value annotation
    ax.text(2, point * 0.15, f'p = {point:.4f}\n95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]',
           fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    # Overall title
    fig.suptitle('Bayesian Model Averaging over HMM Local Optima Clusters\n' +
                 'Frozen Out-of-Sample Validation: Granger Causality HML→SMB in Elevated Regime (Lag=1)',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def create_sensitivity_table():
    """Create a detailed sensitivity analysis table."""

    with open(os.path.join(RESULTS_DIR, 'bma_optima_results.json')) as f:
        results = json.load(f)

    with open(os.path.join(RESULTS_DIR, 'bic_optima_comparison.json')) as f:
        bic_data = json.load(f)

    # Create detailed table
    print("\n" + "="*120)
    print("DETAILED SENSITIVITY ANALYSIS: GRANGER CAUSALITY ACROSS CLUSTERS")
    print("="*120)
    print(f"\n{'Cluster':<10} {'Seeds':<15} {'BIC':<15} {'BIC Δ':<12} {'Weight':<15} {'p-value':<12} {'Sig?':<8} {'Min p':<12} {'Max p':<12}")
    print("-"*120)

    clusters = bic_data['clusters']
    granger_results = results['granger_elevated_hml_smb']['cluster_results']

    min_bic = min([c['bic'] for c in clusters])

    for cluster, granger in zip(clusters, granger_results):
        cid = cluster['cluster_id']
        n_seeds = cluster['n_seeds']
        bic = cluster['bic']
        bic_delta = bic - min_bic
        weight = cluster['bic'] if 'bic' in locals() else 0  # Will be computed
        p_val = granger['cluster_p_value']

        seed_ps = granger['seed_p_values']
        min_p = min(seed_ps) if seed_ps else np.nan
        max_p = max(seed_ps) if seed_ps else np.nan

        # Compute weight
        bic_vals = [c['bic'] for c in clusters]
        min_bic_val = min(bic_vals)
        bic_centered = np.array(bic_vals) - min_bic_val
        weights_raw = np.exp(-0.5 * bic_centered)
        weights_norm = weights_raw / np.sum(weights_raw)
        weight = weights_norm[cid - 1]

        sig = '***' if p_val < 0.01 else ('**' if p_val < 0.05 else 'ns')

        print(f"{cid:<10} {n_seeds:<15} {bic:<15.2f} {bic_delta:<12.2f} {weight:<15.2e} "
              f"{p_val:<12.6f} {sig:<8} {min_p:<12.6f} {max_p:<12.6f}")

    # Summary statistics
    bma_point = results['granger_elevated_hml_smb']['bma_estimate']['point']
    bma_ci_lower = results['granger_elevated_hml_smb']['bma_estimate']['ci_lower']
    bma_ci_upper = results['granger_elevated_hml_smb']['bma_estimate']['ci_upper']

    print("-"*120)
    print(f"\nBMA SUMMARY:")
    print(f"  Point estimate: {bma_point:.6f}")
    print(f"  95% Credible Interval: [{bma_ci_lower:.6f}, {bma_ci_upper:.6f}]")
    print(f"  Significant at 5% level: {'YES' if bma_point < 0.05 else 'NO'}")

    # Weight concentration
    weight_vals = weights_norm
    entropy = -np.sum([w * np.log(w + 1e-300) for w in weight_vals])
    concentration = weight_vals[0] * 100  # Top cluster weight

    print(f"\n  Weight Concentration Analysis:")
    print(f"    Top cluster weight: {concentration:.2f}%")
    print(f"    Shannon entropy: {entropy:.4f}")
    print(f"    Effective # clusters: {np.exp(entropy):.2f}")

    print("\n" + "="*120 + "\n")


def main():
    """Generate enhanced figure and sensitivity analysis."""

    print("\nGenerating enhanced BMA visualization...")
    fig = create_enhanced_figure()

    fig_path = os.path.join(FIGURES_DIR, 'bma_optima_weights_enhanced.pdf')
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Enhanced figure saved: {fig_path}")

    fig_png = fig_path.replace('.pdf', '.png')
    fig.savefig(fig_png, dpi=150, bbox_inches='tight')
    print(f"PNG version saved: {fig_png}")

    plt.close(fig)

    # Sensitivity analysis
    create_sensitivity_table()


if __name__ == '__main__':
    main()
