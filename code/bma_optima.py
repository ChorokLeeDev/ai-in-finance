"""
Bayesian Model Averaging (BMA) over HMM Local Optima Clusters
==============================================================

Implements BMA to combine inference across the 7 local optima clusters found
in the 50-seed multistart analysis. Uses BIC-based posterior model weights
to produce cluster-weighted Granger causality p-values.

Key insight: The best-BIC cluster (seeds 28,20,6; BIC=75,587) assigns 0% of
2008 GFC days to Crisis, while economically interpretable clusters (BIC=75,805+)
detect 90%+ of 2008. BMA averages across all clusters weighted by their
posterior probabilities.

Data sources:
  - bic_optima_comparison.json: BIC values and cluster memberships
  - frozen_oos_50seeds.json: Pre-computed Granger causality p-values for each seed

Output:
  - Cluster-weighted Granger p-values for Elevated regime (HML→SMB)
  - BMA posterior weights and confidence intervals
  - Publication-quality figure
"""

import numpy as np
import pandas as pd
import json
import os
import sys
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Path configuration
RESULTS_DIR = '/sessions/festive-youthful-mccarthy/mnt/causal_regimes/results'
FIGURES_DIR = '/sessions/festive-youthful-mccarthy/mnt/causal_regimes/figures'

REGIME_NAMES = ['Normal', 'Elevated', 'Crisis']


def compute_bma_weights(bic_values):
    """
    Compute BMA posterior model weights using BIC.

    w_k = exp(-0.5 * BIC_k) / sum(exp(-0.5 * BIC_j))

    This uses the standard Bayesian Model Averaging formula based on BIC.
    """
    bic_values = np.array(bic_values)
    # For numerical stability, subtract minimum BIC
    min_bic = np.min(bic_values)
    bic_centered = bic_values - min_bic

    # Compute weights
    weights_raw = np.exp(-0.5 * bic_centered)
    weights = weights_raw / np.sum(weights_raw)

    return weights


def load_bic_analysis():
    """Load pre-computed BIC analysis results."""
    path = os.path.join(RESULTS_DIR, 'bic_optima_comparison.json')
    with open(path, 'r') as f:
        return json.load(f)


def load_frozen_oos_results():
    """Load frozen OOS 50-seed Granger results."""
    path = os.path.join(RESULTS_DIR, 'frozen_oos_50seeds.json')
    with open(path, 'r') as f:
        return json.load(f)


def extract_granger_for_seed(oos_results, seed, regime='Elevated'):
    """Extract Granger HAC p-value for a specific seed and regime."""
    for seed_result in oos_results['all_seeds']:
        if seed_result['seed'] == seed:
            granger_regime = seed_result['granger'].get(regime, {})
            hml_to_smb = granger_regime.get('hml_to_smb', {})
            return hml_to_smb.get('hac_p_value')
    return None


def run_bma_analysis():
    """Main BMA analysis."""
    print("=" * 80)
    print("BAYESIAN MODEL AVERAGING OVER HMM LOCAL OPTIMA")
    print("=" * 80)

    # Load BIC analysis
    print("\n[Step 1] Loading BIC optima analysis...")
    bic_analysis = load_bic_analysis()
    clusters = bic_analysis['clusters']
    n_clusters = len(clusters)
    print(f"  Found {n_clusters} clusters from multistart analysis")

    # Load frozen OOS results
    print("\n[Step 2] Loading frozen OOS Granger results...")
    oos_results = load_frozen_oos_results()
    print(f"  Loaded results for {oos_results['n_seeds']} seeds")

    # Extract BIC values and compute weights
    print("\n[Step 3] Computing BMA posterior model weights...")
    bic_values = np.array([c['bic'] for c in clusters])
    bma_weights = compute_bma_weights(bic_values)

    print(f"\n  Cluster-level posterior model weights (w_k):")
    print(f"  {'Cluster':<10} {'Seeds':<15} {'BIC':<15} {'Weight':<12} {'Cumul %':<10}")
    print(f"  {'-'*70}")
    cumul = 0
    for i, (cluster, weight) in enumerate(zip(clusters, bma_weights)):
        cumul += weight * 100
        seed_str = f"{cluster['n_seeds']} seeds"
        print(f"  {cluster['cluster_id']:<10} {seed_str:<15} {cluster['bic']:<15.2f} "
              f"{weight:<12.4f} {cumul:<10.1f}%")

    # Compute Granger p-values for each cluster (average across seeds in cluster)
    print("\n[Step 4] Extracting Granger p-values for each cluster...")
    cluster_granger_results = []

    for cluster in clusters:
        print(f"\n  Cluster {cluster['cluster_id']} ({cluster['n_seeds']} seeds):")

        # Extract Granger HAC p-values for all seeds in this cluster
        p_values_in_cluster = []
        for seed in cluster['all_seeds']:
            p_val = extract_granger_for_seed(oos_results, seed, regime='Elevated')
            if p_val is not None:
                p_values_in_cluster.append(p_val)
                print(f"    Seed {seed}: p={p_val:.6f}")

        if not p_values_in_cluster:
            print(f"    WARNING: No valid p-values for this cluster")
            cluster_p = None
        else:
            # Cluster p-value as median of its seeds
            cluster_p = float(np.median(p_values_in_cluster))
            print(f"    Cluster median p-value: {cluster_p:.6f}")

        cluster_granger_results.append({
            'cluster_id': cluster['cluster_id'],
            'representative_seed': cluster['representative_seed'],
            'n_seeds': cluster['n_seeds'],
            'all_seeds': cluster['all_seeds'],
            'bic': cluster['bic'],
            'weight': float(bma_weights[len(cluster_granger_results)]),
            'seed_p_values': p_values_in_cluster,
            'cluster_p_value': cluster_p,
            'granger_elevated': {
                'p_value': cluster_p,
                'n_seeds': len(p_values_in_cluster),
            }
        })

    # Compute BMA-weighted p-values
    print("\n[Step 5] Computing BMA-weighted Granger p-values...")

    valid_clusters = [r for r in cluster_granger_results
                      if r['granger_elevated']['p_value'] is not None]

    if not valid_clusters:
        print("  ERROR: No valid cluster results")
        return None

    # Normalize weights for valid clusters only
    valid_weights = np.array([r['weight'] for r in valid_clusters])
    valid_weights = valid_weights / np.sum(valid_weights)

    # Compute BMA p-value (weighted average)
    p_values = np.array([r['granger_elevated']['p_value'] for r in valid_clusters])
    bma_p_value = np.average(p_values, weights=valid_weights)

    # Compute confidence interval for BMA estimate
    # Using bootstrap-like approach with posterior sample
    np.random.seed(42)
    n_bootstrap = 10000
    p_bootstrap = []
    for _ in range(n_bootstrap):
        # Sample cluster according to posterior weights
        sampled_idx = np.random.choice(len(valid_clusters), p=valid_weights)
        # Sample seed within cluster according to its p-value distribution
        cluster = valid_clusters[sampled_idx]
        if cluster['seed_p_values']:
            sampled_p = np.random.choice(cluster['seed_p_values'])
            p_bootstrap.append(sampled_p)

    if p_bootstrap:
        p_ci_lower = np.percentile(p_bootstrap, 2.5)
        p_ci_upper = np.percentile(p_bootstrap, 97.5)
    else:
        p_ci_lower = bma_p_value
        p_ci_upper = bma_p_value

    print(f"\n  BMA-weighted p-value (Elevated, HML→SMB):")
    print(f"    Point estimate: {bma_p_value:.6f}")
    print(f"    95% CI:         [{p_ci_lower:.6f}, {p_ci_upper:.6f}]")
    print(f"    Interpretation: {('Significant (p<0.05)' if bma_p_value < 0.05 else 'Not significant (p≥0.05)')}")

    # Prepare summary results
    results_summary = {
        'metadata': {
            'timestamp': str(datetime.now()),
            'description': 'BMA over 7 local optima clusters using frozen OOS results',
            'n_clusters': n_clusters,
            'data_source_bic': 'bic_optima_comparison.json',
            'data_source_granger': 'frozen_oos_50seeds.json',
            'lag': 1,
            'regime': 'Elevated',
            'test': 'HML→SMB',
        },
        'bma_weights': {
            'clusters': [
                {
                    'cluster_id': r['cluster_id'],
                    'n_seeds': r['n_seeds'],
                    'all_seeds': r['all_seeds'],
                    'bic': r['bic'],
                    'posterior_weight': r['weight'],
                }
                for r in cluster_granger_results
            ]
        },
        'granger_elevated_hml_smb': {
            'cluster_results': [
                {
                    'cluster_id': r['cluster_id'],
                    'representative_seed': r['representative_seed'],
                    'n_seeds': r['n_seeds'],
                    'cluster_p_value': r['granger_elevated']['p_value'],
                    'seed_p_values': r['seed_p_values'],
                    'weight': r['weight'],
                }
                for r in cluster_granger_results
                if r['granger_elevated']['p_value'] is not None
            ],
            'bma_estimate': {
                'point': float(bma_p_value),
                'ci_lower': float(p_ci_lower),
                'ci_upper': float(p_ci_upper),
                'significant_at_5pct': bool(bma_p_value < 0.05),
            }
        }
    }

    return {
        'cluster_granger_results': cluster_granger_results,
        'bma_weights': bma_weights,
        'valid_clusters': valid_clusters,
        'valid_weights': valid_weights,
        'bma_p_value': bma_p_value,
        'p_ci_lower': p_ci_lower,
        'p_ci_upper': p_ci_upper,
        'results_summary': results_summary,
    }


def create_publication_figure(analysis_results):
    """Create publication-quality figure showing BMA results."""

    cluster_granger_results = analysis_results['cluster_granger_results']
    bma_weights = analysis_results['bma_weights']
    valid_clusters = analysis_results['valid_clusters']
    bma_p_value = analysis_results['bma_p_value']
    p_ci_lower = analysis_results['p_ci_lower']
    p_ci_upper = analysis_results['p_ci_upper']

    # Create figure with 3 subpanels
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), gridspec_kw={'width_ratios': [1.2, 1.2, 1.3]})

    # Panel A: Cluster BIC and Weights
    ax = axes[0]
    cluster_ids = np.arange(1, len(cluster_granger_results) + 1)
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(cluster_granger_results)))

    # Sort by BIC for visualization
    sorted_idx = np.argsort([r['bic'] for r in cluster_granger_results])
    cluster_ids_sorted = cluster_ids[sorted_idx]
    bic_sorted = np.array([r['bic'] for r in cluster_granger_results])[sorted_idx]
    weights_sorted = bma_weights[sorted_idx]
    colors_sorted = colors[sorted_idx]

    bars1 = ax.bar(cluster_ids_sorted - 0.2, bic_sorted / 1000, width=0.4,
                    label='BIC (÷1000)', color=colors_sorted, alpha=0.7, edgecolor='black', linewidth=1.5)

    ax2 = ax.twinx()
    bars2 = ax2.bar(cluster_ids_sorted + 0.2, weights_sorted * 100, width=0.4,
                     label='Posterior Weight (%)', color='steelblue', alpha=0.5, edgecolor='black', linewidth=1.5)

    ax.set_xlabel('Cluster (sorted by BIC)', fontsize=11, fontweight='bold')
    ax.set_ylabel('BIC (÷1000)', fontsize=11, fontweight='bold', color='black')
    ax2.set_ylabel('Posterior Weight (%)', fontsize=11, fontweight='bold', color='steelblue')
    ax.set_title('(A) Cluster Weights', fontsize=12, fontweight='bold')
    ax.set_xticks(cluster_ids_sorted)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([75.5, 76.2])
    ax2.set_ylim([0, 50])

    # Legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)

    # Panel B: Per-Cluster Granger p-values
    ax = axes[1]

    p_values = np.array([r['granger_elevated']['p_value'] for r in cluster_granger_results])
    valid_mask = ~np.isnan(p_values) & (p_values > 0)

    colors_p = ['green' if p < 0.05 else 'red' for p in p_values[valid_mask]]

    ax.scatter(cluster_ids_sorted, p_values[sorted_idx], s=200, c=colors_sorted,
               alpha=0.6, edgecolor='black', linewidth=2, zorder=3, label='Cluster p-value')

    # Add significance threshold line
    ax.axhline(0.05, color='red', linestyle='--', linewidth=2, label='α=0.05', zorder=2)

    ax.set_xlabel('Cluster', fontsize=11, fontweight='bold')
    ax.set_ylabel('HAC-adjusted p-value (HML→SMB)', fontsize=11, fontweight='bold')
    ax.set_title('(B) Per-Cluster Granger p-values', fontsize=12, fontweight='bold')
    ax.set_xticks(cluster_ids_sorted)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, which='both', linestyle='--')
    ax.set_ylim([1e-8, 1.0])
    ax.legend(loc='lower left', fontsize=10)

    # Panel C: BMA Result with CI
    ax = axes[2]

    # Prepare data for visualization
    bma_methods = ['Per-cluster\np-values', 'BMA\nEstimate']

    # Show individual cluster p-values as scatter, then BMA result
    cluster_labels = [f"C{int(r['cluster_id'])}" for r in cluster_granger_results]
    x_positions = np.linspace(0.8, 1.2, len(valid_clusters))

    for x_pos, cluster in zip(x_positions, valid_clusters):
        p_val = cluster['granger_elevated']['p_value']
        color_val = 'green' if p_val < 0.05 else 'red'
        ax.scatter(x_pos, p_val, s=150, c=color_val, alpha=0.5, edgecolor='black', linewidth=1.5, zorder=2)

    # BMA estimate with CI
    ax.errorbar(2, bma_p_value, yerr=[[bma_p_value - p_ci_lower], [p_ci_upper - bma_p_value]],
                fmt='D', markersize=12, color='darkblue', ecolor='darkblue', elinewidth=2.5,
                capsize=10, capthick=2, label='BMA ± 95% CI', zorder=3)

    # Significance threshold
    ax.axhline(0.05, color='red', linestyle='--', linewidth=2.5, label='α=0.05', zorder=2)

    ax.set_xlim([0.5, 2.5])
    ax.set_yscale('log')
    ax.set_ylabel('HAC-adjusted p-value', fontsize=11, fontweight='bold')
    ax.set_title('(C) BMA Result', fontsize=12, fontweight='bold')
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Cluster\nResults', 'BMA\nEstimate'], fontsize=10)
    ax.set_ylim([1e-8, 1.0])
    ax.grid(True, alpha=0.3, which='both', linestyle='--')
    ax.legend(loc='lower left', fontsize=10)

    # Add text annotation for BMA result
    sig_text = "Significant (p<0.05)" if bma_p_value < 0.05 else "Not significant (p≥0.05)"
    ax.text(2, bma_p_value * 2, f'{sig_text}\np={bma_p_value:.4f}',
            fontsize=10, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.suptitle('Bayesian Model Averaging over HMM Local Optima Clusters\n' +
                 'Granger Causality: HML→SMB in Elevated Regime (Frozen OOS, Lag=1)',
                 fontsize=13, fontweight='bold', y=1.00)

    plt.tight_layout()

    return fig


def save_results(analysis_results):
    """Save analysis results to JSON."""
    output_path = os.path.join(RESULTS_DIR, 'bma_optima_results.json')
    with open(output_path, 'w') as f:
        json.dump(analysis_results['results_summary'], f, indent=2)
    print(f"\nResults saved to: {output_path}")
    return output_path


def main():
    """Run full BMA analysis."""

    # Run BMA analysis
    analysis_results = run_bma_analysis()

    if analysis_results is None:
        print("\nERROR: BMA analysis failed")
        return

    # Create and save figure
    print("\n[Step 6] Creating publication-quality figure...")
    fig = create_publication_figure(analysis_results)

    fig_path = os.path.join(FIGURES_DIR, 'bma_optima_weights.pdf')
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"  Figure saved to: {fig_path}")

    fig_png = fig_path.replace('.pdf', '.png')
    fig.savefig(fig_png, dpi=150, bbox_inches='tight')
    print(f"  PNG version saved to: {fig_png}")

    plt.close(fig)

    # Save results
    results_path = save_results(analysis_results)

    # Print final summary
    print("\n" + "=" * 80)
    print("FINAL BMA SUMMARY")
    print("=" * 80)
    print(f"\nGranger Causality: HML→SMB in Elevated Regime (lag=1, Frozen OOS)")
    print(f"\n  BMA-weighted p-value: {analysis_results['bma_p_value']:.6f}")
    print(f"  95% Credible Interval: [{analysis_results['p_ci_lower']:.6f}, {analysis_results['p_ci_upper']:.6f}]")

    if analysis_results['bma_p_value'] < 0.05:
        print(f"\n  RESULT: SIGNIFICANT at 5% level")
    else:
        print(f"\n  RESULT: NOT SIGNIFICANT at 5% level")

    print(f"\nCluster Posterior Weights:")
    for r in analysis_results['cluster_granger_results']:
        p_str = f"{r['granger_elevated']['p_value']:.6f}" if r['granger_elevated']['p_value'] else "N/A"
        print(f"  Cluster {int(r['cluster_id'])}: w={r['weight']:.4f} ({r['weight']*100:.2f}%), p={p_str}")

    print(f"\nOutput files:")
    print(f"  - Figure: {fig_path}")
    print(f"  - Results: {results_path}")
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
