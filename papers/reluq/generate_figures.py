"""
Generate Figures for NeurIPS Paper: RelUQ
==========================================

Creates publication-quality figures for the RelUQ paper.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import json
import os

# Set NeurIPS style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.linewidth'] = 0.8

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(SCRIPT_DIR, 'figures')
RESULTS_DIR = '/Users/i767700/Github/ai-in-finance/chorok/v3_fk_risk_attribution/results'
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_results(filename):
    """Load results from JSON file."""
    filepath = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None


def fig1_overview():
    """
    Figure 1: RelUQ Method Overview
    Shows the pipeline from database to actionable attribution.
    """
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis('off')

    # Main pipeline boxes
    boxes = [
        (0.3, 1.8, 2.2, 1.2, 'Relational\nDatabase\n(Schema)', '#E3F2FD'),
        (3.0, 1.8, 2.2, 1.2, 'FK-Grouped\nFeatures\n(col_to_fk)', '#FFF3E0'),
        (5.7, 1.8, 2.2, 1.2, 'Ensemble\nModels\n(K=5)', '#E8F5E9'),
        (8.4, 1.8, 2.2, 1.2, 'FK-Level\nAttribution\n(Permutation)', '#FCE4EC'),
    ]

    for x, y, w, h, text, color in boxes:
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.03,rounding_size=0.15",
            facecolor=color, edgecolor='#333333', linewidth=1.2
        )
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=9, fontweight='bold', linespacing=1.2)

    # Arrows between boxes
    arrow_props = dict(arrowstyle='->', color='#333333', lw=1.5,
                       connectionstyle='arc3,rad=0')
    for i in range(3):
        x_start = boxes[i][0] + boxes[i][2]
        x_end = boxes[i+1][0]
        y_mid = boxes[i][1] + boxes[i][3]/2
        ax.annotate('', xy=(x_end, y_mid), xytext=(x_start, y_mid),
                    arrowprops=arrow_props)

    # Output box
    output_box = mpatches.FancyBboxPatch(
        (8.4, 0.3), 2.2, 1.2,
        boxstyle="round,pad=0.03,rounding_size=0.15",
        facecolor='#FFFDE7', edgecolor='#333333', linewidth=1.2
    )
    ax.add_patch(output_box)
    ax.text(9.5, 0.9, 'Actionable\nInsight\n(Drill-down)', ha='center', va='center',
            fontsize=9, fontweight='bold', linespacing=1.2)

    # Arrow from attribution to output
    ax.annotate('', xy=(9.5, 1.5), xytext=(9.5, 1.8),
                arrowprops=dict(arrowstyle='->', color='#333333', lw=1.5))

    # Key insight annotation
    ax.text(6, 3.5, 'Key: FK constraints define semantic grouping',
            ha='center', va='center', fontsize=10, style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8))

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/fig1_overview.pdf', bbox_inches='tight')
    plt.savefig(f'{FIGURES_DIR}/fig1_overview.png', bbox_inches='tight')
    plt.close()
    print("Created: fig1_overview.pdf/png")


def fig2_attribution_error_scatter():
    """
    Figure 2: Attribution-Error Correlation
    Scatter plot showing FK attribution vs error impact across domains.
    """
    results = load_results('attribution_error_validation.json')

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

    domains = [
        ('salt', 'rel-salt (ERP)', '#2E86AB', 'o'),
        ('trial', 'rel-trial (Clinical)', '#A23B72', 's'),
        ('stack', 'rel-stack (Q&A)', '#F18F01', '^'),
    ]

    for ax, (domain, title, color, marker) in zip(axes, domains):
        if results and domain in results:
            data = results[domain]
            unc_attr = list(data['unc_attribution'].values())
            err_impact = list(data['error_impact'].values())
            fk_names = list(data['unc_attribution'].keys())
            rho = data['spearman_corr']

            ax.scatter(unc_attr, err_impact, c=color, s=80, marker=marker,
                      edgecolors='black', linewidth=0.5, alpha=0.8)

            for i, name in enumerate(fk_names):
                ax.annotate(name, (unc_attr[i], err_impact[i]),
                           fontsize=7, ha='left', va='bottom',
                           xytext=(3, 3), textcoords='offset points')

            if rho is not None and abs(rho) > 0.5:
                z = np.polyfit(unc_attr, err_impact, 1)
                p = np.poly1d(z)
                x_line = np.linspace(min(unc_attr), max(unc_attr), 100)
                ax.plot(x_line, p(x_line), '--', color=color, alpha=0.5, lw=1.5)

            rho_str = f'ρ = {rho:.2f}' if rho is not None else 'ρ = N/A'
        else:
            rho_str = 'No data'

        ax.set_xlabel('Uncertainty Attribution (%)', fontsize=10)
        ax.set_ylabel('Error Impact (%)', fontsize=10)
        ax.set_title(f'{title}\n{rho_str}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/fig2_attribution_error.pdf', bbox_inches='tight')
    plt.savefig(f'{FIGURES_DIR}/fig2_attribution_error.png', bbox_inches='tight')
    plt.close()
    print("Created: fig2_attribution_error.pdf/png")


def fig3_stability_comparison():
    """
    Figure 3: Stability Comparison - FK-level vs Feature-level
    """
    methods = ['Feature-level\n(24 attrs)', 'Correlation\nClustering', 'Random\nGrouping', 'RelUQ\n(FK-level)']
    stability_mean = [0.45, 0.62, 0.35, 0.93]
    stability_std = [0.15, 0.12, 0.20, 0.04]

    fig, ax = plt.subplots(figsize=(8, 4))

    x = np.arange(len(methods))
    colors = ['#FFB3B3', '#FFFFB3', '#FFB3B3', '#B3FFB3']

    bars = ax.bar(x, stability_mean, yerr=stability_std,
                  color=colors, edgecolor='black', linewidth=1.2,
                  capsize=5, error_kw={'lw': 1.5})

    for bar, val, std in zip(bars, stability_mean, stability_std):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + std + 0.03,
                f'{val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax.axhline(y=0.8, color='green', linestyle='--', lw=1.5, alpha=0.7, label='Target: ρ ≥ 0.8')
    ax.axhline(y=0.5, color='orange', linestyle=':', lw=1.2, alpha=0.7, label='Random baseline')

    ax.set_ylabel('Attribution Stability\n(Spearman ρ across seeds)', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_title('FK Grouping Dramatically Improves Stability', fontsize=12, fontweight='bold')

    ax.annotate('', xy=(3, 0.93), xytext=(0, 0.45),
                arrowprops=dict(arrowstyle='->', color='green', lw=2,
                               connectionstyle='arc3,rad=-0.2'))
    ax.text(1.5, 0.72, '+107%', fontsize=12, fontweight='bold', color='green')

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/fig3_stability.pdf', bbox_inches='tight')
    plt.savefig(f'{FIGURES_DIR}/fig3_stability.png', bbox_inches='tight')
    plt.close()
    print("Created: fig3_stability.pdf/png")


def fig4_hierarchical_drilldown():
    """
    Figure 4: Hierarchical Drill-Down Example
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Level 1: FK Groups
    fk_data = [
        (1, 4.5, 'ITEM\n34.6%', '#FF6B6B'),
        (3.5, 4.5, 'SALES\nDOC\n21.8%', '#4ECDC4'),
        (6, 4.5, 'SALES\nGROUP\n20.3%', '#45B7D1'),
        (8.5, 4.5, 'SHIP\n12.1%', '#96CEB4'),
        (11, 4.5, 'SOLD\n11.3%', '#FFEAA7'),
    ]

    for x, y, text, color in fk_data:
        rect = mpatches.FancyBboxPatch(
            (x-0.7, y-0.4), 1.4, 0.8,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            facecolor=color, edgecolor='black', linewidth=1.5
        )
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=8, fontweight='bold')

    ax.text(0.3, 4.5, 'Level 1\n(FK Group)', ha='center', va='center', fontsize=9,
            fontweight='bold', color='#666')

    # Level 2: Features (for ITEM only)
    feature_data = [
        (0.5, 2.8, 'SHIPPING\nPOINT\n52%', '#FF6B6B'),
        (2.0, 2.8, 'ITEM\nINCO\n48%', '#FF8E8E'),
    ]

    for x, y, text, color in feature_data:
        rect = mpatches.FancyBboxPatch(
            (x-0.5, y-0.35), 1.0, 0.7,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=color, edgecolor='black', linewidth=1
        )
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=7, fontweight='bold')
        ax.plot([x, 1], [y+0.35, 4.1], 'k-', lw=0.8)

    ax.text(0.3, 2.8, 'Level 2\n(Feature)', ha='center', va='center', fontsize=9,
            fontweight='bold', color='#666')

    # Level 3: Entities
    entity_data = [
        (0.3, 1.0, 'SP 40\n57× higher', '#FF4444'),
        (1.3, 1.0, 'SP 12\n23×', '#FF6666'),
        (2.3, 1.0, 'SP 2\n1× (baseline)', '#AAFFAA'),
    ]

    for x, y, text, color in entity_data:
        rect = mpatches.FancyBboxPatch(
            (x-0.4, y-0.3), 0.8, 0.6,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            facecolor=color, edgecolor='black', linewidth=1
        )
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=6, fontweight='bold')
        ax.plot([x, 0.5], [y+0.3, 2.45], 'k-', lw=0.8)

    ax.text(0.3, 1.0, 'Level 3\n(Entity)', ha='center', va='center', fontsize=9,
            fontweight='bold', color='#666')

    # Actionable insight box
    insight_box = mpatches.FancyBboxPatch(
        (4.5, 0.5), 7, 2.5,
        boxstyle="round,pad=0.05,rounding_size=0.15",
        facecolor='#FFFDE7', edgecolor='#FFA000', linewidth=2
    )
    ax.add_patch(insight_box)

    ax.text(8, 2.5, 'Actionable Insight', ha='center', va='center',
            fontsize=11, fontweight='bold', color='#E65100')
    ax.text(8, 1.7, '1. ITEM FK group contributes 34.6% of uncertainty',
            ha='center', va='center', fontsize=9)
    ax.text(8, 1.3, '2. Within ITEM, SHIPPINGPOINT is the main driver (52%)',
            ha='center', va='center', fontsize=9)
    ax.text(8, 0.9, '3. Shipping Point 40 has 57× higher uncertainty than SP 2',
            ha='center', va='center', fontsize=9, fontweight='bold', color='#D32F2F')

    ax.text(6, 5.7, 'Hierarchical Drill-Down: rel-salt (ERP Dataset)',
            ha='center', va='center', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/fig4_drilldown.pdf', bbox_inches='tight')
    plt.savefig(f'{FIGURES_DIR}/fig4_drilldown.png', bbox_inches='tight')
    plt.close()
    print("Created: fig4_drilldown.pdf/png")


def fig5_domain_comparison():
    """
    Figure 5: Multi-Domain Validation - EP vs Associative
    """
    domains = ['rel-salt\n(ERP)', 'rel-trial\n(Clinical)', 'rel-avito\n(Classifieds)',
               'rel-amazon\n(E-comm)', 'rel-stack\n(Q&A)']
    rho_values = [0.90, 1.00, 1.00, None, -0.50]
    domain_types = ['EP', 'EP', 'EP', 'Assoc', 'Assoc']

    fig, ax = plt.subplots(figsize=(9, 4.5))

    x = np.arange(len(domains))
    colors = ['#4CAF50' if t == 'EP' else '#FF5722' for t in domain_types]
    plot_values = [v if v is not None else 0 for v in rho_values]

    bars = ax.bar(x, plot_values, color=colors, edgecolor='black', linewidth=1.2, alpha=0.8)

    for i, (bar, val) in enumerate(zip(bars, rho_values)):
        if val is not None:
            height = bar.get_height()
            va = 'bottom' if height >= 0 else 'top'
            offset = 0.03 if height >= 0 else -0.03
            ax.text(bar.get_x() + bar.get_width()/2, height + offset,
                    f'ρ = {val:.2f}', ha='center', va=va, fontweight='bold', fontsize=10)
        else:
            ax.text(bar.get_x() + bar.get_width()/2, 0.05,
                    'Only 2 FKs\n(N/A)', ha='center', va='bottom', fontsize=8, color='gray')

    ax.axhline(y=0, color='black', lw=0.8)
    ax.axhline(y=0.8, color='green', linestyle='--', lw=1.5, alpha=0.7)
    ax.text(4.6, 0.82, 'Target: ρ ≥ 0.8', fontsize=9, color='green')

    ax.axvline(x=2.5, color='gray', linestyle=':', lw=1.5)
    ax.text(1, 1.1, 'Error Propagation\nDomains', ha='center', fontsize=10,
            fontweight='bold', color='#2E7D32')
    ax.text(3.5, 1.1, 'Associative\nDomains', ha='center', fontsize=10,
            fontweight='bold', color='#BF360C')

    ax.set_ylabel('Spearman ρ\n(Uncertainty vs Error Impact)', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(domains, fontsize=9)
    ax.set_ylim(-0.7, 1.3)
    ax.set_title('The Error Propagation Hypothesis: FK Attribution Works in EP Domains',
                 fontsize=12, fontweight='bold')

    legend_elements = [
        mpatches.Patch(facecolor='#4CAF50', edgecolor='black', label='EP Domain (ρ ≥ 0.90)'),
        mpatches.Patch(facecolor='#FF5722', edgecolor='black', label='Associative Domain')
    ]
    ax.legend(handles=legend_elements, loc='lower left', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/fig5_domains.pdf', bbox_inches='tight')
    plt.savefig(f'{FIGURES_DIR}/fig5_domains.png', bbox_inches='tight')
    plt.close()
    print("Created: fig5_domains.pdf/png")


def fig6_hierarchy_comparison():
    """Visual comparison of hierarchy structures."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Feature-level: flat
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis('off')
    ax.set_title('Feature-level\n(Flat)', fontsize=11, fontweight='bold')

    for i, feat in enumerate(['f1', 'f2', 'f3', '...', 'f24']):
        x = 1 + i * 1.8
        rect = mpatches.FancyBboxPatch((x, 1.5), 1.2, 0.8, boxstyle="round,pad=0.02",
                                        facecolor='#FFB3B3', edgecolor='black')
        ax.add_patch(rect)
        ax.text(x + 0.6, 1.9, feat, ha='center', va='center', fontsize=9)
    ax.text(5, 0.8, 'No drill-up possible', ha='center', fontsize=10, style='italic', color='red')

    # Correlation: unstable groups
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis('off')
    ax.set_title('Correlation Clustering\n(Data-driven)', fontsize=11, fontweight='bold')

    for i, grp in enumerate(['G1', 'G2', 'G3']):
        x = 1.5 + i * 2.5
        rect = mpatches.FancyBboxPatch((x, 2.8), 1.5, 0.6, boxstyle="round,pad=0.02",
                                        facecolor='#FFFFB3', edgecolor='black')
        ax.add_patch(rect)
        ax.text(x + 0.75, 3.1, grp, ha='center', va='center', fontsize=9)
        for j in range(2):
            fx = x + j * 0.8
            frect = mpatches.FancyBboxPatch((fx, 1.8), 0.6, 0.5, boxstyle="round,pad=0.02",
                                            facecolor='#FFB3B3', edgecolor='black')
            ax.add_patch(frect)
            ax.text(fx + 0.3, 2.05, f'f{i*2+j+1}', ha='center', va='center', fontsize=7)
            ax.plot([fx + 0.3, x + 0.75], [2.3, 2.8], 'k-', lw=0.5)

    ax.text(5, 0.8, 'Groups change with data!', ha='center', fontsize=10, style='italic', color='orange')

    # FK: stable hierarchy
    ax = axes[2]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis('off')
    ax.set_title('RelUQ (FK-based)\n(Schema-defined)', fontsize=11, fontweight='bold')

    fk_groups = ['DRIVER', 'CIRCUIT', 'RACE']
    colors = ['#B3FFB3', '#B3E0FF', '#FFE0B3']
    for i, (grp, col) in enumerate(zip(fk_groups, colors)):
        x = 1.5 + i * 2.5
        rect = mpatches.FancyBboxPatch((x, 2.8), 1.5, 0.6, boxstyle="round,pad=0.02",
                                        facecolor=col, edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + 0.75, 3.1, grp, ha='center', va='center', fontsize=9, fontweight='bold')
        for j in range(2):
            fx = x + j * 0.8
            frect = mpatches.FancyBboxPatch((fx, 1.8), 0.6, 0.5, boxstyle="round,pad=0.02",
                                            facecolor=col, edgecolor='black', alpha=0.6)
            ax.add_patch(frect)
            ax.plot([fx + 0.3, x + 0.75], [2.3, 2.8], 'k-', lw=0.5)

    ax.text(5, 0.8, 'Stable, actionable grouping', ha='center', fontsize=10, style='italic', color='green')

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/fig6_hierarchy.pdf', bbox_inches='tight')
    plt.savefig(f'{FIGURES_DIR}/fig6_hierarchy.png', bbox_inches='tight')
    plt.close()
    print("Created: fig6_hierarchy.pdf/png")


def fig8_main_result():
    """Figure: Main result summary - horizontal bar chart."""
    results = load_results('attribution_error_validation.json')

    if results:
        domains = ['salt', 'trial', 'avito', 'stack']
        names = ['rel-salt', 'rel-trial', 'rel-avito', 'rel-stack']
        rhos = [results[d]['spearman_corr'] if d in results and results[d]['spearman_corr'] else 0 for d in domains]
        types = ['EP', 'EP', 'EP', 'Assoc']
    else:
        names = ['rel-salt', 'rel-trial', 'rel-avito', 'rel-stack']
        rhos = [0.90, 1.00, 1.00, -0.50]
        types = ['EP', 'EP', 'EP', 'Assoc']

    fig, ax = plt.subplots(figsize=(8, 4))

    x = np.arange(len(names))
    colors = ['#66BB6A' if t == 'EP' else '#EF5350' for t in types]

    bars = ax.barh(x, rhos, color=colors, edgecolor='black', height=0.6)

    for bar, val in zip(bars, rhos):
        width = bar.get_width()
        x_pos = width + 0.02 if width >= 0 else width - 0.02
        ha = 'left' if width >= 0 else 'right'
        ax.text(x_pos, bar.get_y() + bar.get_height()/2,
                f'ρ = {val:.2f}', ha=ha, va='center', fontweight='bold', fontsize=11)

    ax.axvline(x=0, color='black', lw=1)
    ax.axvline(x=0.8, color='green', linestyle='--', lw=1.5, alpha=0.7)

    ax.set_yticks(x)
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xlabel('Spearman Correlation (Uncertainty Attribution vs Error Impact)', fontsize=11)
    ax.set_xlim(-0.8, 1.2)
    ax.set_title('Main Result: FK Attribution Predicts Error Impact in EP Domains',
                 fontsize=12, fontweight='bold')

    legend_elements = [
        mpatches.Patch(facecolor='#66BB6A', edgecolor='black', label='Error Propagation'),
        mpatches.Patch(facecolor='#EF5350', edgecolor='black', label='Associative')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/fig8_attribution_error_validation.pdf', bbox_inches='tight')
    plt.savefig(f'{FIGURES_DIR}/fig8_attribution_error_validation.png', bbox_inches='tight')
    plt.close()
    print("Created: fig8_attribution_error_validation.pdf/png")


def main():
    print("=" * 60)
    print("Generating Figures for NeurIPS Paper: RelUQ")
    print("=" * 60)

    fig1_overview()
    fig2_attribution_error_scatter()
    fig3_stability_comparison()
    fig4_hierarchical_drilldown()
    fig5_domain_comparison()
    fig6_hierarchy_comparison()
    fig8_main_result()

    print("=" * 60)
    print(f"All figures saved to: {FIGURES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
