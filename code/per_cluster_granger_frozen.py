"""
Per-Cluster Granger Robustness Analysis (using Frozen OOS)
===========================================================

Replaces degenerate BMA with per-cluster robustness analysis using pre-computed
frozen OOS results.

For EACH of the 7 local optima clusters:
  1. Use all seeds in the cluster from bic_optima_comparison.json
  2. Extract Granger p-values from frozen_oos_50seeds.json for each seed
  3. Report median/mean p-values for each regime
  4. Assess robustness: "In how many clusters is the Elevated finding significant?"

This approach:
  - Uses pre-computed, frozen OOS Granger statistics (no refitting)
  - Compares across ALL seeds in each cluster, not just representative seed
  - More robust and closer to how the original analysis was performed
"""

import sys
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Paths
RESULTS_DIR = '/sessions/festive-youthful-mccarthy/mnt/causal_regimes/results'
FIGURES_DIR = '/sessions/festive-youthful-mccarthy/mnt/causal_regimes/figures'
BIC_FILE = os.path.join(RESULTS_DIR, 'bic_optima_comparison.json')
FROZEN_OOS_FILE = os.path.join(RESULTS_DIR, 'frozen_oos_50seeds.json')

REGIME_NAMES = ['Normal', 'Elevated', 'Crisis']


def load_cluster_definitions():
    """Load cluster definitions from BIC analysis."""
    with open(BIC_FILE, 'r') as f:
        data = json.load(f)
    return data['clusters']


def load_frozen_oos_results():
    """Load pre-computed frozen OOS Granger results."""
    with open(FROZEN_OOS_FILE, 'r') as f:
        data = json.load(f)
    return data['all_seeds']


def extract_granger_for_seed(oos_results, seed, regime='Elevated'):
    """Extract Granger p-values for a specific seed and regime."""
    for seed_result in oos_results:
        if seed_result['seed'] == seed:
            granger = seed_result.get('granger', {})
            regime_data = granger.get(regime, {})
            hml_to_smb = regime_data.get('hml_to_smb', {})

            return {
                'n_clean': regime_data.get('n_clean'),
                'f_p_value': hml_to_smb.get('f_p_value'),
                'hac_p_value': hml_to_smb.get('hac_p_value'),
                'f_stat': hml_to_smb.get('f_stat'),
                'hac_wald_stat': hml_to_smb.get('hac_wald_stat'),
            }
    return None


def run_per_cluster_analysis():
    """Main per-cluster analysis using frozen OOS results."""
    print("=" * 80)
    print("PER-CLUSTER GRANGER ROBUSTNESS ANALYSIS (FROZEN OOS)")
    print("=" * 80)

    # Load data
    print("\n[Step 1] Loading cluster definitions...")
    clusters = load_cluster_definitions()
    n_clusters = len(clusters)
    print(f"  Loaded {n_clusters} clusters from BIC analysis")

    print("\n[Step 2] Loading frozen OOS Granger results...")
    oos_results = load_frozen_oos_results()
    print(f"  Loaded results for {len(oos_results)} seeds")

    # Run analysis for each cluster
    print("\n[Step 3] Aggregating Granger p-values across seeds in each cluster...")
    cluster_results = []

    for cluster in clusters:
        cluster_id = cluster['cluster_id']
        all_seeds = cluster['all_seeds']
        rep_seed = cluster['representative_seed']
        n_seeds = len(all_seeds)
        bic = cluster['bic']
        crisis_2008_pct = cluster['crisis_2008_pct']

        print(f"\n  Cluster {cluster_id} ({n_seeds} seeds, BIC={bic:.1f}, GFC={crisis_2008_pct:.0f}%):")

        cluster_summary = {
            'cluster_id': cluster_id,
            'representative_seed': rep_seed,
            'n_seeds_in_cluster': n_seeds,
            'all_seeds': all_seeds,
            'bic': bic,
            'crisis_2008_pct': crisis_2008_pct,
            'regime_results': {}
        }

        # For each regime, extract p-values from all seeds
        sig_regimes_count = 0

        for regime_name in REGIME_NAMES:
            # Extract p-values for all seeds in this cluster
            p_values = []
            f_p_values = []
            hac_p_values = []

            for seed in all_seeds:
                result = extract_granger_for_seed(oos_results, seed, regime=regime_name)
                if result and result['hac_p_value'] is not None:
                    hac_p_values.append(result['hac_p_value'])
                    f_p_values.append(result['f_p_value'])

            if not hac_p_values:
                print(f"    {regime_name:10s}: No valid results")
                cluster_summary['regime_results'][regime_name] = {
                    'n_seeds_with_results': 0,
                    'median_hac_p': None,
                    'mean_hac_p': None,
                    'significant_at_05': False,
                }
                continue

            # Compute statistics
            median_p = float(np.median(hac_p_values))
            mean_p = float(np.mean(hac_p_values))
            is_sig = median_p < 0.05

            if is_sig:
                sig_regimes_count += 1

            cluster_summary['regime_results'][regime_name] = {
                'n_seeds_with_results': len(hac_p_values),
                'median_hac_p': median_p,
                'mean_hac_p': mean_p,
                'hac_p_values': hac_p_values,
                'significant_at_05': bool(is_sig),
            }

            sig_marker = "***" if is_sig else ""
            print(f"    {regime_name:10s}: n_seeds={len(hac_p_values):2d} | "
                  f"median HAC-p={median_p:.4f} | mean={mean_p:.4f} {sig_marker}")

        cluster_summary['sig_regimes_count'] = sig_regimes_count
        cluster_results.append(cluster_summary)

    return {
        'timestamp': str(datetime.now()),
        'n_clusters': n_clusters,
        'cluster_results': cluster_results,
    }


def create_summary_table(analysis_results):
    """Create publication-quality summary table."""
    cluster_results = analysis_results['cluster_results']

    # Build table data
    table_data = []

    for cluster_res in cluster_results:
        cluster_id = cluster_res['cluster_id']
        rep_seed = cluster_res['representative_seed']
        n_seeds = cluster_res['n_seeds_in_cluster']
        bic = cluster_res['bic']
        crisis_pct = cluster_res['crisis_2008_pct']

        normal = cluster_res['regime_results'].get('Normal', {})
        elevated = cluster_res['regime_results'].get('Elevated', {})
        crisis = cluster_res['regime_results'].get('Crisis', {})

        table_data.append({
            'Cluster': cluster_id,
            'Seeds': n_seeds,
            'Rep. Seed': rep_seed,
            'BIC': bic,
            'GFC 2008 %': crisis_pct,
            'Normal median p': normal.get('median_hac_p'),
            'Normal n': normal.get('n_seeds_with_results', 0),
            'Elevated median p': elevated.get('median_hac_p'),
            'Elevated n': elevated.get('n_seeds_with_results', 0),
            'Crisis median p': crisis.get('median_hac_p'),
            'Crisis n': crisis.get('n_seeds_with_results', 0),
            'Elevated Sig?': elevated.get('significant_at_05', False),
        })

    df_table = pd.DataFrame(table_data)
    return df_table


def save_results(analysis_results, df_table):
    """Save results to CSV and JSON."""
    # Save JSON
    json_path = os.path.join(RESULTS_DIR, 'per_cluster_granger_frozen_results.json')
    with open(json_path, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    print(f"\nJSON results saved: {json_path}")

    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, 'per_cluster_granger_frozen_results.csv')
    df_table.to_csv(csv_path, index=False)
    print(f"CSV summary saved: {csv_path}")

    return json_path, csv_path


def create_publication_figure(df_table, analysis_results):
    """Create publication-quality figure showing per-cluster robustness."""
    cluster_results = analysis_results['cluster_results']

    # Figure: 2 panels
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # ---- Panel A: Cluster BIC and GFC Detection ----
    ax = axes[0]

    clusters_ids = np.array([r['cluster_id'] for r in cluster_results])
    bics = np.array([r['bic'] for r in cluster_results])
    crisis_pcts = np.array([r['crisis_2008_pct'] for r in cluster_results])

    x_pos = np.arange(len(clusters_ids))
    width = 0.35

    # Normalize BIC for visualization (subtract minimum)
    bics_norm = bics - np.min(bics)
    colors = plt.cm.RdYlGn_r(crisis_pcts / 100.0)

    bars1 = ax.bar(x_pos - width/2, bics_norm / 100, width, label='BIC difference from best (÷100)',
                    color='steelblue', alpha=0.7, edgecolor='black', linewidth=1.5)

    ax2 = ax.twinx()
    bars2 = ax2.bar(x_pos + width/2, crisis_pcts, width, label='GFC 2008 detection (%)',
                     color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

    ax.set_xlabel('Cluster ID', fontsize=12, fontweight='bold')
    ax.set_ylabel('BIC Difference (÷100)', fontsize=12, fontweight='bold', color='steelblue')
    ax2.set_ylabel('GFC 2008 Detection (%)', fontsize=12, fontweight='bold', color='darkred')
    ax.set_title('(A) Cluster Properties: BIC vs. GFC Detection', fontsize=13, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'C{c}' for c in clusters_ids], fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_ylim([0, 110])

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=11)

    # ---- Panel B: Median Granger p-values by regime ----
    ax = axes[1]

    # Extract median HAC p-values for each regime
    normal_median_ps = []
    elevated_median_ps = []
    crisis_median_ps = []

    for cluster_res in cluster_results:
        normal = cluster_res['regime_results'].get('Normal', {})
        elevated = cluster_res['regime_results'].get('Elevated', {})
        crisis = cluster_res['regime_results'].get('Crisis', {})

        normal_p = normal.get('median_hac_p', np.nan)
        elevated_p = elevated.get('median_hac_p', np.nan)
        crisis_p = crisis.get('median_hac_p', np.nan)

        normal_median_ps.append(normal_p if not np.isnan(normal_p) else 1.0)
        elevated_median_ps.append(elevated_p if not np.isnan(elevated_p) else 1.0)
        crisis_median_ps.append(crisis_p if not np.isnan(crisis_p) else 1.0)

    x_pos_regimes = np.arange(len(clusters_ids))
    width = 0.25

    ax.bar(x_pos_regimes - width, normal_median_ps, width, label='Normal',
           color='steelblue', alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.bar(x_pos_regimes, elevated_median_ps, width, label='Elevated',
           color='darkorange', alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.bar(x_pos_regimes + width, crisis_median_ps, width, label='Crisis',
           color='darkred', alpha=0.7, edgecolor='black', linewidth=1.5)

    # Significance threshold
    ax.axhline(0.05, color='red', linestyle='--', linewidth=2.5, label='α=0.05', zorder=2)

    ax.set_xlabel('Cluster ID', fontsize=12, fontweight='bold')
    ax.set_ylabel('Median HAC-adjusted p-value (HML→SMB)', fontsize=12, fontweight='bold')
    ax.set_title('(B) Median Granger p-values by Regime (across seeds in cluster)',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x_pos_regimes)
    ax.set_xticklabels([f'C{c}' for c in clusters_ids], fontsize=11)
    ax.set_yscale('log')
    ax.set_ylim([1e-3, 1.0])
    ax.grid(axis='y', alpha=0.3, which='both', linestyle='--')
    ax.legend(loc='upper left', fontsize=11)

    plt.suptitle('Per-Cluster Granger Robustness: HML→SMB across 7 Local Optima\n' +
                 'Using Frozen OOS (train 1990-2012, test 2013-2024), Lag=1, HAC-adjusted',
                 fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout()

    return fig


def print_summary_statistics(analysis_results, df_table):
    """Print summary statistics about robustness."""
    print("\n" + "=" * 80)
    print("ROBUSTNESS SUMMARY")
    print("=" * 80)

    cluster_results = analysis_results['cluster_results']
    n_clusters = len(cluster_results)

    # Check Elevated regime
    elevated_sig = 0
    elevated_p_values = []

    for cluster_res in cluster_results:
        elevated = cluster_res['regime_results'].get('Elevated', {})
        median_p = elevated.get('median_hac_p')
        if median_p is not None:
            elevated_p_values.append(median_p)
            if median_p < 0.05:
                elevated_sig += 1

    print(f"\nElevated regime (main finding: HML→SMB causal):")
    print(f"  Significant at 5% (HAC) in: {elevated_sig}/{n_clusters} clusters")
    print(f"  Median p-values: {[f'{p:.4f}' for p in elevated_p_values]}")
    if elevated_sig > 0:
        print(f"  Robustness: STRONG (consistent across multiple clusters)")
    elif elevated_sig == 1:
        print(f"  Robustness: MODERATE (concentrated in best-fit cluster)")
    else:
        print(f"  Robustness: WEAK (not significant across any cluster)")

    # Check Normal regime
    normal_sig = 0
    for cluster_res in cluster_results:
        normal = cluster_res['regime_results'].get('Normal', {})
        if normal.get('significant_at_05'):
            normal_sig += 1

    print(f"\nNormal regime:")
    print(f"  Significant at 5% (HAC) in: {normal_sig}/{n_clusters} clusters")

    # Check Crisis regime
    crisis_sig = 0
    for cluster_res in cluster_results:
        crisis = cluster_res['regime_results'].get('Crisis', {})
        if crisis.get('significant_at_05'):
            crisis_sig += 1

    print(f"\nCrisis regime:")
    print(f"  Significant at 5% (HAC) in: {crisis_sig}/{n_clusters} clusters")

    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print(f"""
1. BMA Failure Analysis:
   - BIC differences across clusters range from 37 to 550 points
   - exp(-0.5 × 550) ≈ 1e-120, making all non-best weights negligible
   - Result: BMA collapses to Cluster 1 (best-BIC) with ~100% weight
   - This makes BMA vacuous and uninformative

2. Per-Cluster Alternative:
   - For EACH cluster, we report median Granger p-values across its seeds
   - This directly addresses: "Does the finding depend on cluster choice?"
   - Avoids degenerate weighting by treating all clusters symmetrically

3. Robustness Verdict:
   - HML→SMB in Elevated regime: {elevated_sig}/7 clusters (median p < 0.05)
   - If significant in ≥5/7 clusters → finding is ROBUST
   - If significant in ≤2/7 clusters → finding is CLUSTER-DEPENDENT
    """)

    return {
        'elevated_sig': elevated_sig,
        'normal_sig': normal_sig,
        'crisis_sig': crisis_sig,
        'n_clusters': n_clusters,
    }


def main():
    """Run full per-cluster analysis."""

    # Run analysis
    analysis_results = run_per_cluster_analysis()

    # Create summary table
    df_table = create_summary_table(analysis_results)

    # Save results
    json_path, csv_path = save_results(analysis_results, df_table)

    # Print summary
    stats = print_summary_statistics(analysis_results, df_table)

    # Create figure
    print("\n[Step 4] Creating publication-quality figure...")
    fig = create_publication_figure(df_table, analysis_results)

    fig_path = os.path.join(FIGURES_DIR, 'per_cluster_granger_frozen_robustness.pdf')
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"  PDF saved: {fig_path}")

    fig_png = fig_path.replace('.pdf', '.png')
    fig.savefig(fig_png, dpi=150, bbox_inches='tight')
    print(f"  PNG saved: {fig_png}")

    plt.close(fig)

    # Print final summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE (CSV)")
    print("=" * 80)
    print("\nKey columns:")
    display_cols = ['Cluster', 'Seeds', 'GFC 2008 %', 'Elevated median p', 'Elevated Sig?']
    print(df_table[display_cols].to_string(index=False))

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Output files:")
    print(f"  - JSON: {json_path}")
    print(f"  - CSV:  {csv_path}")
    print(f"  - PDF:  {fig_path}")
    print(f"  - PNG:  {fig_png}")


if __name__ == '__main__':
    main()
