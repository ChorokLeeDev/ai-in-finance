"""
Aggregate Validation Analysis
=============================

This script analyzes validation results across ALL domains/tasks and computes
aggregate statistics. Key insight: with 55 FK data points across 13 tasks,
we get much more robust correlations than per-domain analysis.

Outputs:
1. Aggregate correlation between uncertainty and corruption sensitivity
2. Publication-ready figures
3. Statistical tests with proper sample sizes
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def load_all_results(results_dir):
    """Load all validation results from JSON files."""
    results_dir = Path(results_dir)
    all_results = []

    for json_file in results_dir.glob('*_validation.json'):
        if json_file.name == 'all_validation_results.json':
            continue
        with open(json_file) as f:
            result = json.load(f)
            all_results.append(result)

    return all_results


def extract_fk_datapoints(all_results):
    """Extract all FK-level data points across all domains."""
    fk_datapoints = []

    for result in all_results:
        dataset = result['dataset']
        task = result['task']

        # Get corruption results
        corruption = result['experiments'].get('corruption', {})
        if 'fk_results' not in corruption:
            continue

        # Get EP detection results
        ep = result['experiments'].get('ep_detection', {})
        if 'fk_uncertainty_attribution' not in ep:
            continue

        unc_attr = ep['fk_uncertainty_attribution']
        err_attr = ep['fk_error_attribution']

        for fk_name, fk_data in corruption['fk_results'].items():
            if fk_name not in unc_attr or fk_name not in err_attr:
                continue

            fk_datapoints.append({
                'dataset': dataset,
                'task': task,
                'fk': fk_name,
                'uncertainty_contribution': unc_attr[fk_name],
                'error_attribution': err_attr[fk_name],
                'corruption_sensitivity': fk_data['corruption_sensitivity'],
            })

    return fk_datapoints


def compute_aggregate_correlations(fk_datapoints):
    """Compute correlations across ALL FK data points."""
    print("\n" + "="*70)
    print("AGGREGATE CORRELATION ANALYSIS")
    print("="*70)

    n = len(fk_datapoints)
    print(f"\nTotal FK data points: {n}")

    if n < 5:
        print("ERROR: Not enough data points for robust analysis")
        return {}

    uncertainties = [dp['uncertainty_contribution'] for dp in fk_datapoints]
    errors = [dp['error_attribution'] for dp in fk_datapoints]
    sensitivities = [dp['corruption_sensitivity'] for dp in fk_datapoints]

    # 1. Uncertainty vs Error (EP validation)
    rho_ue, p_ue = stats.spearmanr(uncertainties, errors)
    print(f"\n1. Uncertainty vs Error Attribution:")
    print(f"   Spearman ρ = {rho_ue:.3f} (p = {p_ue:.4f}, n = {n})")

    if rho_ue > 0.5 and p_ue < 0.05:
        print("   ✅ Significant positive correlation - EP property holds")
    elif rho_ue > 0.3:
        print("   ⚠️  Moderate correlation - partial EP property")
    else:
        print("   ❌ Weak/negative correlation - EP property fails")

    # 2. Uncertainty vs Corruption Sensitivity (causal validation)
    rho_us, p_us = stats.spearmanr(uncertainties, sensitivities)
    print(f"\n2. Uncertainty vs Corruption Sensitivity:")
    print(f"   Spearman ρ = {rho_us:.3f} (p = {p_us:.4f}, n = {n})")

    if rho_us > 0.5 and p_us < 0.05:
        print("   ✅ Significant correlation - uncertainty predicts corruption impact")
    elif rho_us > 0.3:
        print("   ⚠️  Moderate correlation - some predictive power")
    else:
        print("   ❌ Weak correlation - uncertainty doesn't predict corruption well")

    # 3. Error vs Corruption Sensitivity (sanity check)
    rho_es, p_es = stats.spearmanr(errors, sensitivities)
    print(f"\n3. Error Attribution vs Corruption Sensitivity:")
    print(f"   Spearman ρ = {rho_es:.3f} (p = {p_es:.4f}, n = {n})")

    if rho_es > 0.5 and p_es < 0.05:
        print("   ✅ Significant correlation - error attribution predicts corruption")
    else:
        print("   ⚠️  Weaker than expected")

    return {
        'n_datapoints': n,
        'uncertainty_vs_error': {'rho': float(rho_ue), 'p': float(p_ue)},
        'uncertainty_vs_corruption': {'rho': float(rho_us), 'p': float(p_us)},
        'error_vs_corruption': {'rho': float(rho_es), 'p': float(p_es)},
    }


def analyze_by_domain(fk_datapoints):
    """Analyze patterns by domain."""
    print("\n" + "="*70)
    print("ANALYSIS BY DOMAIN")
    print("="*70)

    # Group by dataset
    by_dataset = {}
    for dp in fk_datapoints:
        dataset = dp['dataset']
        if dataset not in by_dataset:
            by_dataset[dataset] = []
        by_dataset[dataset].append(dp)

    domain_stats = {}

    for dataset, datapoints in sorted(by_dataset.items()):
        print(f"\n{dataset}:")

        uncertainties = [dp['uncertainty_contribution'] for dp in datapoints]
        sensitivities = [dp['corruption_sensitivity'] for dp in datapoints]

        n = len(datapoints)
        if n >= 3:
            rho, p = stats.spearmanr(uncertainties, sensitivities)
            print(f"  FK groups: {n}, ρ(unc, corruption) = {rho:.3f}")
        else:
            rho = np.nan
            print(f"  FK groups: {n} (too few for correlation)")

        # Show top FKs
        sorted_by_unc = sorted(datapoints, key=lambda x: -x['uncertainty_contribution'])
        print(f"  Top uncertainty: {sorted_by_unc[0]['fk']} ({sorted_by_unc[0]['uncertainty_contribution']:+.1f}%)")

        sorted_by_sens = sorted(datapoints, key=lambda x: -x['corruption_sensitivity'])
        print(f"  Top corruption: {sorted_by_sens[0]['fk']} ({sorted_by_sens[0]['corruption_sensitivity']:+.1f}%)")

        domain_stats[dataset] = {
            'n_fks': n,
            'correlation': float(rho) if not np.isnan(rho) else None,
            'top_uncertainty_fk': sorted_by_unc[0]['fk'],
            'top_corruption_fk': sorted_by_sens[0]['fk'],
            'match': sorted_by_unc[0]['fk'] == sorted_by_sens[0]['fk']
        }

    return domain_stats


def create_aggregate_figures(fk_datapoints, output_dir):
    """Create publication-ready figures."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Figure 1: Scatter plot - Uncertainty vs Corruption Sensitivity
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Colors by domain
    domain_colors = {
        'rel-f1': '#1f77b4',
        'rel-trial': '#2ca02c',
        'rel-salt': '#ff7f0e',
        'rel-avito': '#9467bd',
        'rel-event': '#d62728',
    }

    uncertainties = np.array([dp['uncertainty_contribution'] for dp in fk_datapoints])
    errors = np.array([dp['error_attribution'] for dp in fk_datapoints])
    sensitivities = np.array([dp['corruption_sensitivity'] for dp in fk_datapoints])
    colors = [domain_colors.get(dp['dataset'], 'gray') for dp in fk_datapoints]

    # Panel A: Uncertainty vs Corruption
    ax = axes[0]
    ax.scatter(uncertainties, sensitivities, c=colors, s=80, alpha=0.7, edgecolors='white')

    # Add regression line
    z = np.polyfit(uncertainties, sensitivities, 1)
    p = np.poly1d(z)
    x_line = np.linspace(uncertainties.min(), uncertainties.max(), 100)
    ax.plot(x_line, p(x_line), 'r--', alpha=0.5, label='Trend')

    rho, pval = stats.spearmanr(uncertainties, sensitivities)
    ax.set_xlabel('FK Uncertainty Contribution (%)', fontsize=11)
    ax.set_ylabel('Corruption Sensitivity (%)', fontsize=11)
    ax.set_title(f'(A) Causal Validation\nρ = {rho:.2f}, p = {pval:.3f}', fontsize=12)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)

    # Panel B: Uncertainty vs Error
    ax = axes[1]
    ax.scatter(uncertainties, errors, c=colors, s=80, alpha=0.7, edgecolors='white')

    z = np.polyfit(uncertainties, errors, 1)
    p = np.poly1d(z)
    ax.plot(x_line, p(x_line), 'r--', alpha=0.5)

    rho, pval = stats.spearmanr(uncertainties, errors)
    ax.set_xlabel('FK Uncertainty Contribution (%)', fontsize=11)
    ax.set_ylabel('FK Error Attribution (%)', fontsize=11)
    ax.set_title(f'(B) EP Validation\nρ = {rho:.2f}, p = {pval:.3f}', fontsize=12)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)

    # Panel C: Error vs Corruption
    ax = axes[2]
    ax.scatter(errors, sensitivities, c=colors, s=80, alpha=0.7, edgecolors='white')

    z = np.polyfit(errors, sensitivities, 1)
    p = np.poly1d(z)
    x_line = np.linspace(errors.min(), errors.max(), 100)
    ax.plot(x_line, p(x_line), 'r--', alpha=0.5)

    rho, pval = stats.spearmanr(errors, sensitivities)
    ax.set_xlabel('FK Error Attribution (%)', fontsize=11)
    ax.set_ylabel('Corruption Sensitivity (%)', fontsize=11)
    ax.set_title(f'(C) Importance Validation\nρ = {rho:.2f}, p = {pval:.3f}', fontsize=12)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)

    # Legend
    handles = [mpatches.Patch(color=c, label=d.replace('rel-', '').upper())
              for d, c in domain_colors.items()]
    axes[2].legend(handles=handles, loc='lower right', fontsize=9)

    plt.tight_layout()
    fig_path = output_dir / 'aggregate_validation.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.savefig(str(fig_path).replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to: {fig_path}")
    plt.close()

    # Figure 2: Domain-level summary
    fig, ax = plt.subplots(figsize=(10, 6))

    # Group by domain and compute stats
    by_domain = {}
    for dp in fk_datapoints:
        d = dp['dataset']
        if d not in by_domain:
            by_domain[d] = {'unc': [], 'sens': []}
        by_domain[d]['unc'].append(dp['uncertainty_contribution'])
        by_domain[d]['sens'].append(dp['corruption_sensitivity'])

    domains = list(by_domain.keys())
    x_pos = np.arange(len(domains))

    correlations = []
    for d in domains:
        if len(by_domain[d]['unc']) >= 3:
            rho, _ = stats.spearmanr(by_domain[d]['unc'], by_domain[d]['sens'])
            correlations.append(rho)
        else:
            correlations.append(0)

    colors = ['green' if r > 0.5 else 'orange' if r > 0 else 'red' for r in correlations]
    bars = ax.bar(x_pos, correlations, color=colors, edgecolor='black', alpha=0.7)

    ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Strong (0.7)')
    ax.axhline(y=0.3, color='orange', linestyle='--', alpha=0.5, label='Moderate (0.3)')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([d.replace('rel-', '').upper() for d in domains], fontsize=11)
    ax.set_ylabel('Spearman ρ (Uncertainty vs Corruption)', fontsize=11)
    ax.set_title('Causal Validation by Domain', fontsize=14)
    ax.set_ylim(-1, 1)
    ax.legend(loc='upper right')

    fig_path = output_dir / 'domain_correlations.png'
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.savefig(str(fig_path).replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {fig_path}")
    plt.close()


def generate_paper_statistics(fk_datapoints, correlations, domain_stats):
    """Generate statistics for paper."""
    print("\n" + "="*70)
    print("STATISTICS FOR PAPER")
    print("="*70)

    n = len(fk_datapoints)

    print(f"""
Key Statistics:
- Total FK data points: {n}
- Domains validated: {len(domain_stats)}
- Aggregate ρ (uncertainty vs corruption): {correlations['uncertainty_vs_corruption']['rho']:.2f}
  (p = {correlations['uncertainty_vs_corruption']['p']:.4f})
- Aggregate ρ (uncertainty vs error): {correlations['uncertainty_vs_error']['rho']:.2f}
  (p = {correlations['uncertainty_vs_error']['p']:.4f})

Domain-level matching (top uncertainty FK = top corruption FK):
""")

    matches = sum(1 for d, s in domain_stats.items() if s['match'])
    total = len(domain_stats)
    print(f"- Matches: {matches}/{total} ({matches/total*100:.0f}%)")

    # Per-domain breakdown
    for domain, stats in domain_stats.items():
        match_symbol = "✓" if stats['match'] else "✗"
        print(f"  {domain}: {match_symbol} (unc={stats['top_uncertainty_fk']}, corr={stats['top_corruption_fk']})")


def main():
    results_dir = Path(__file__).parent / 'validation_results'

    if not results_dir.exists():
        print(f"ERROR: Results directory not found: {results_dir}")
        print("Run causal_validation_suite.py --all first")
        return

    # Load results
    print("Loading validation results...")
    all_results = load_all_results(results_dir)
    print(f"Loaded {len(all_results)} domain/task results")

    # Extract FK datapoints
    fk_datapoints = extract_fk_datapoints(all_results)
    print(f"Extracted {len(fk_datapoints)} FK data points")

    if len(fk_datapoints) < 5:
        print("ERROR: Not enough FK data points. Run more experiments.")
        return

    # Compute aggregate correlations
    correlations = compute_aggregate_correlations(fk_datapoints)

    # Analyze by domain
    domain_stats = analyze_by_domain(fk_datapoints)

    # Create figures
    create_aggregate_figures(fk_datapoints, results_dir)

    # Generate paper statistics
    generate_paper_statistics(fk_datapoints, correlations, domain_stats)

    # Save analysis results
    analysis = {
        'n_datapoints': len(fk_datapoints),
        'aggregate_correlations': correlations,
        'domain_stats': domain_stats,
        'fk_datapoints': fk_datapoints
    }

    output_file = results_dir / 'aggregate_analysis.json'
    with open(output_file, 'w') as f:
        json.dump(analysis, f, indent=2)

    print(f"\nAnalysis saved to: {output_file}")

    # Final verdict
    print("\n" + "="*70)
    print("FINAL VERDICT FOR PAPER")
    print("="*70)

    rho_uc = correlations['uncertainty_vs_corruption']['rho']
    p_uc = correlations['uncertainty_vs_corruption']['p']

    if rho_uc > 0.5 and p_uc < 0.05:
        print(f"""
✅ CAUSAL VALIDATION PASSED (ρ = {rho_uc:.2f}, p = {p_uc:.4f})

You can claim:
"FK-level uncertainty contribution significantly predicts corruption
sensitivity across {len(fk_datapoints)} FK-task combinations (Spearman ρ = {rho_uc:.2f},
p < {p_uc:.3f}), validating that high-uncertainty FKs are indeed more
sensitive to data quality issues."
""")
    elif rho_uc > 0.3:
        print(f"""
⚠️  CAUSAL VALIDATION MARGINAL (ρ = {rho_uc:.2f}, p = {p_uc:.4f})

You can claim (with caveats):
"FK-level uncertainty shows moderate correlation with corruption
sensitivity (ρ = {rho_uc:.2f}), suggesting some predictive power for
identifying data quality bottlenecks."
""")
    else:
        print(f"""
❌ CAUSAL VALIDATION FAILED (ρ = {rho_uc:.2f}, p = {p_uc:.4f})

This is a significant problem for the paper. Consider:
1. Reframing the contribution away from causal claims
2. Focusing on other validated aspects (learning curves, etc.)
3. Investigating why uncertainty doesn't predict corruption
""")


if __name__ == '__main__':
    main()
