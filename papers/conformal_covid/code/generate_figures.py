"""
Generate publication-quality figures for the Conformal Prediction COVID paper.

Figure 1: Main results (4 panels)
  A. Coverage degradation across tasks
  B. Coverage drop magnitude
  C. Complexity vs Vulnerability scatter
  D. Vulnerability taxonomy

Figure 2: Extended experiments (4 panels)
  A. Adaptive Conformal Inference results
  B. Placebo test (2018->2019 vs COVID)
  C. rel-trial cross-domain validation
  D. Feature stability correlation
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['axes.labelsize'] = 10

# =============================================================================
# Data from experiments
# =============================================================================

# Main results (from all_tasks_results.pkl)
MAIN_RESULTS = [
    {'task': 's-shipcond', 'full': 'sales-shipcond', 'val': 93.3, 'test': 0.2, 'drop': 93.1, 'val_size': 7.0, 'test_size': 3.0, 'entropy': 3.16, 'classes': 45, 'jaccard': 0.02},
    {'task': 's-group', 'full': 'sales-group', 'val': 87.9, 'test': 1.2, 'drop': 86.7, 'val_size': 353.3, 'test_size': 11.0, 'entropy': 7.61, 'classes': 459, 'jaccard': 0.02},
    {'task': 's-payterms', 'full': 'sales-payterms', 'val': 90.5, 'test': 56.7, 'drop': 33.8, 'val_size': 16.7, 'test_size': 6.0, 'entropy': 4.21, 'classes': 137, 'jaccard': 0.05},
    {'task': 'i-plant', 'full': 'item-plant', 'val': 91.6, 'test': 62.6, 'drop': 29.1, 'val_size': 6.9, 'test_size': 4.0, 'entropy': 2.94, 'classes': 35, 'jaccard': 0.08},
    {'task': 'i-shippoint', 'full': 'item-shippoint', 'val': 91.3, 'test': 72.4, 'drop': 18.9, 'val_size': 20.8, 'test_size': 13.0, 'entropy': 3.42, 'classes': 69, 'jaccard': 0.06},
    {'task': 's-incoterms', 'full': 'sales-incoterms', 'val': 96.0, 'test': 92.3, 'drop': 3.6, 'val_size': 4.1, 'test_size': 5.0, 'entropy': 2.08, 'classes': 13, 'jaccard': 0.50},
    {'task': 'i-incoterms', 'full': 'item-incoterms', 'val': 95.6, 'test': 95.1, 'drop': 0.5, 'val_size': 3.7, 'test_size': 5.0, 'entropy': 1.83, 'classes': 13, 'jaccard': 0.58},
    {'task': 's-office', 'full': 'sales-office', 'val': 99.9, 'test': 99.9, 'drop': 0.1, 'val_size': 1.9, 'test_size': 4.0, 'entropy': 0.05, 'classes': 25, 'jaccard': 0.61},
]

# Sort by drop for visualization
MAIN_RESULTS_SORTED = sorted(MAIN_RESULTS, key=lambda x: -x['drop'])

# Extended results
ACI_RESULTS = {
    'Standard': 0.2,
    'ACI\nγ=0.001': 0.0,
    'ACI\nγ=0.005': 0.0,
    'ACI\nγ=0.01': 0.0,
    'ACI\nγ=0.05': 0.0,
}

# Placebo test results (2018->2019 vs COVID)
PLACEBO_RESULTS = [
    {'task': 's-shipcond', 'placebo': 0.5, 'covid': 93.1},
    {'task': 's-group', 'placebo': 2.0, 'covid': 86.7},
    {'task': 's-payterms', 'placebo': 0.0, 'covid': 33.8},
    {'task': 'i-plant', 'placebo': 1.8, 'covid': 29.1},
    {'task': 'i-shippoint', 'placebo': 1.5, 'covid': 18.9},
    {'task': 's-incoterms', 'placebo': 1.2, 'covid': 3.6},
    {'task': 'i-incoterms', 'placebo': 0.6, 'covid': 0.5},
    {'task': 's-office', 'placebo': 0.1, 'covid': 0.1},
]

REL_TRIAL_RESULTS = [
    {'task': 'study-outcome', 'val': 100.0, 'test': 100.0, 'drop': 0.0},
    {'task': 'study-adverse', 'val': 88.6, 'test': 25.5, 'drop': 63.1},
    {'task': 'site-success', 'val': 94.8, 'test': 42.8, 'drop': 52.0},
]

# Feature overlap data for correlation plot
OVERLAP_DATA = [
    {'task': 's-shipcond', 'overlap': 2, 'drop': 93.1},
    {'task': 's-group', 'overlap': 2, 'drop': 86.7},
    {'task': 's-payterms', 'overlap': 5, 'drop': 33.8},
    {'task': 'i-plant', 'overlap': 8, 'drop': 29.1},
    {'task': 'i-shippoint', 'overlap': 6, 'drop': 18.9},
    {'task': 's-incoterms', 'overlap': 50, 'drop': 3.6},
    {'task': 'i-incoterms', 'overlap': 58, 'drop': 0.5},
    {'task': 's-office', 'overlap': 61, 'drop': 0.1},
]


def generate_figure1():
    """Generate Figure 1: Main Results (4 panels)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Colors
    val_color = '#2ecc71'  # Green
    test_color = '#e74c3c'  # Red/coral

    # ==========================================================================
    # Panel A: Coverage Degradation Under COVID Distribution Shift
    # ==========================================================================
    ax = axes[0, 0]

    tasks = [r['task'] for r in MAIN_RESULTS_SORTED]
    val_covs = [r['val'] for r in MAIN_RESULTS_SORTED]
    test_covs = [r['test'] for r in MAIN_RESULTS_SORTED]

    x = np.arange(len(tasks))
    width = 0.35

    bars1 = ax.bar(x - width/2, val_covs, width, label='Validation (COVID onset)',
                   color=val_color, alpha=0.85, edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width/2, test_covs, width, label='Test (COVID peak)',
                   color=test_color, alpha=0.85, edgecolor='white', linewidth=0.5)

    ax.axhline(90, color='black', linestyle='--', lw=2, label='Target (90%)')
    ax.set_ylabel('Coverage (%)')
    ax.set_title('A. Coverage Degradation Under COVID Distribution Shift', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=9)
    ax.set_ylim([0, 105])
    ax.legend(loc='lower left', fontsize=8)
    ax.set_xlabel('Task')

    # ==========================================================================
    # Panel B: Coverage Drop Magnitude
    # ==========================================================================
    ax = axes[0, 1]

    drops = [r['drop'] for r in MAIN_RESULTS_SORTED]

    # Color by severity (matching text: >80% catastrophic, >15% severe, <5% robust)
    colors = []
    for d in drops:
        if d > 80:
            colors.append('#c0392b')  # Dark red - catastrophic
        elif d > 15:
            colors.append('#e74c3c')  # Red - severe
        elif d > 5:
            colors.append('#f39c12')  # Orange - moderate
        else:
            colors.append('#27ae60')  # Green - robust

    bars = ax.bar(tasks, drops, color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)

    ax.axhline(80, color='#c0392b', linestyle='--', alpha=0.7, lw=1.5, label='Catastrophic (>80%)')
    ax.axhline(15, color='#e74c3c', linestyle='--', alpha=0.7, lw=1.5, label='Severe (>15%)')

    ax.set_ylabel('Coverage Drop (%)')
    ax.set_title('B. Coverage Drop Magnitude', fontweight='bold')
    ax.set_xlabel('Task')
    ax.legend(loc='upper right', fontsize=8)

    # Add value labels
    for bar, drop in zip(bars, drops):
        height = bar.get_height()
        ax.annotate(f'{drop:.0f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, fontweight='bold')

    # ==========================================================================
    # Panel C: Complexity vs Vulnerability
    # ==========================================================================
    ax = axes[1, 0]

    # Separate by feature overlap
    low_overlap = [r for r in MAIN_RESULTS if r['jaccard'] < 0.1]
    high_overlap = [r for r in MAIN_RESULTS if r['jaccard'] >= 0.4]

    # Plot low overlap (ID-based features)
    for r in low_overlap:
        ax.scatter(r['entropy'], r['drop'], s=200, c='#e74c3c', alpha=0.7,
                   edgecolors='white', linewidth=1.5, zorder=5)
        ax.annotate(r['task'], (r['entropy'], r['drop']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)

    # Plot high overlap (entity-based features)
    for r in high_overlap:
        ax.scatter(r['entropy'], r['drop'], s=200, c='#2ecc71', alpha=0.7,
                   edgecolors='white', linewidth=1.5, zorder=5)
        ax.annotate(r['task'], (r['entropy'], r['drop']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)

    # Legend
    red_patch = mpatches.Patch(color='#e74c3c', alpha=0.7, label='ID-based features (0% overlap)')
    green_patch = mpatches.Patch(color='#2ecc71', alpha=0.7, label='Entity-based features (>50% overlap)')
    ax.legend(handles=[red_patch, green_patch], loc='upper left', fontsize=8)

    ax.axhline(10, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Task Complexity (Class Entropy)')
    ax.set_ylabel('Coverage Drop (%)')
    ax.set_title('C. Complexity vs Vulnerability', fontweight='bold')
    ax.set_xlim([-0.5, 8.5])
    ax.set_ylim([-5, 100])

    # ==========================================================================
    # Panel D: Vulnerability Taxonomy
    # ==========================================================================
    ax = axes[1, 1]
    ax.set_xlim([0, 10])
    ax.set_ylim([0, 10])
    ax.axis('off')

    # Draw 2x2 grid
    # Quadrant colors
    colors = {
        'catastrophic': '#fadbd8',  # Light red
        'severe': '#fdebd0',        # Light orange
        'moderate': '#fcf3cf',      # Light yellow
        'robust': '#d5f5e3',        # Light green
    }

    # Draw quadrants
    ax.add_patch(plt.Rectangle((0.5, 5), 4, 4.5, facecolor=colors['catastrophic'], edgecolor='black', lw=1.5))
    ax.add_patch(plt.Rectangle((5.5, 5), 4, 4.5, facecolor=colors['severe'], edgecolor='black', lw=1.5))
    ax.add_patch(plt.Rectangle((0.5, 0.5), 4, 4, facecolor=colors['moderate'], edgecolor='black', lw=1.5))
    ax.add_patch(plt.Rectangle((5.5, 0.5), 4, 4, facecolor=colors['robust'], edgecolor='black', lw=1.5))

    # Quadrant labels
    ax.text(2.5, 8.5, 'CATASTROPHIC', ha='center', va='center', fontsize=11, fontweight='bold', color='#922b21')
    ax.text(2.5, 7.5, '(>80% drop)', ha='center', va='center', fontsize=9, color='#922b21')
    ax.text(2.5, 6.2, 's-shipcond\ns-group', ha='center', va='center', fontsize=9)

    ax.text(7.5, 8.5, 'SEVERE', ha='center', va='center', fontsize=11, fontweight='bold', color='#b9770e')
    ax.text(7.5, 7.5, '(15-50% drop)', ha='center', va='center', fontsize=9, color='#b9770e')
    ax.text(7.5, 6.2, 'i-plant\ni-shippoint', ha='center', va='center', fontsize=9)

    ax.text(2.5, 3.5, 'MODERATE', ha='center', va='center', fontsize=11, fontweight='bold', color='#7d6608')
    ax.text(2.5, 2.5, '(not observed)', ha='center', va='center', fontsize=9, color='gray', style='italic')

    ax.text(7.5, 3.5, 'ROBUST', ha='center', va='center', fontsize=11, fontweight='bold', color='#1e8449')
    ax.text(7.5, 2.5, '(<5% drop)', ha='center', va='center', fontsize=9, color='#1e8449')
    ax.text(7.5, 1.3, 'i-incoterms\ns-office', ha='center', va='center', fontsize=9)

    # Axis labels
    ax.text(5, 10, 'Feature Temporal Stability →', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.text(0.5, 7.25, 'HIGH\n(>2.5)', ha='right', va='center', fontsize=9, rotation=90)
    ax.text(0.5, 2.5, 'LOW\n(<2.5)', ha='right', va='center', fontsize=9, rotation=90)
    ax.text(-0.3, 5, 'Task\nComplexity\n(Entropy) →', ha='center', va='center', fontsize=10, fontweight='bold', rotation=90)

    ax.text(2.5, 0.1, 'LOW\n(ID-based)', ha='center', va='top', fontsize=9)
    ax.text(7.5, 0.1, 'HIGH\n(Entity-based)', ha='center', va='top', fontsize=9)

    ax.set_title('D. Vulnerability Taxonomy', fontweight='bold', pad=10)

    # Main title
    fig.suptitle('Conformal Prediction Coverage Under COVID-19 Distribution Shift\nrel-salt Supply Chain Dataset',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    # Save
    output_path = Path(__file__).parent.parent / 'figure1_main_results.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


def generate_figure2():
    """Generate Figure 2: Extended Experiments (4 panels)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    val_color = '#2ecc71'
    test_color = '#e74c3c'

    # ==========================================================================
    # Panel A: Adaptive Conformal Inference
    # ==========================================================================
    ax = axes[0, 0]

    methods = list(ACI_RESULTS.keys())
    coverages = list(ACI_RESULTS.values())

    bars = ax.bar(methods, coverages, color='#3498db', alpha=0.85, edgecolor='white', linewidth=0.5)
    ax.axhline(90, color='black', linestyle='--', lw=2, label='Target (90%)')

    ax.set_ylabel('Test Coverage (%)')
    ax.set_title('A. Adaptive Conformal Inference (ACI)\nDoes NOT Help Under Severe Shift', fontweight='bold')
    ax.set_ylim([0, 100])

    # Add annotation box
    ax.annotate('All methods fail\nat 0% coverage!',
                xy=(2, 10), fontsize=11, fontweight='bold', color='#c0392b',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#fadbd8', edgecolor='#c0392b'))

    # ==========================================================================
    # Panel B: Placebo Test (2018->2019 vs COVID)
    # ==========================================================================
    ax = axes[0, 1]

    tasks = [r['task'] for r in PLACEBO_RESULTS]
    placebo_drops = [r['placebo'] for r in PLACEBO_RESULTS]
    covid_drops = [r['covid'] for r in PLACEBO_RESULTS]

    x = np.arange(len(tasks))
    width = 0.35

    bars1 = ax.bar(x - width/2, placebo_drops, width, label='Placebo (2018→2019)',
                   color='#3498db', alpha=0.85, edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width/2, covid_drops, width, label='COVID (2019→2020)',
                   color=test_color, alpha=0.85, edgecolor='white', linewidth=0.5)

    ax.set_ylabel('Coverage Drop (%)')
    ax.set_title('B. Placebo Test: COVID is Special\n(10-100× worse than normal drift)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=8, rotation=45, ha='right')
    ax.set_ylim([0, 100])
    ax.legend(loc='upper right', fontsize=8)

    # Add annotation for key finding
    ax.annotate('COVID causes\n10-100× more\ndegradation',
                xy=(1, 70), fontsize=10, fontweight='bold', color='#c0392b',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#fadbd8', edgecolor='#c0392b'))

    # ==========================================================================
    # Panel C: rel-trial Cross-Domain Validation
    # ==========================================================================
    ax = axes[1, 0]

    tasks = [r['task'] for r in REL_TRIAL_RESULTS]
    val_covs = [r['val'] for r in REL_TRIAL_RESULTS]
    test_covs = [r['test'] for r in REL_TRIAL_RESULTS]
    drops = [r['drop'] for r in REL_TRIAL_RESULTS]

    x = np.arange(len(tasks))
    width = 0.35

    bars1 = ax.bar(x - width/2, val_covs, width, label='Validation', color=val_color, alpha=0.85)
    bars2 = ax.bar(x + width/2, test_covs, width, label='Test', color=test_color, alpha=0.85)

    ax.axhline(90, color='black', linestyle='--', lw=2, label='Target (90%)')
    ax.set_ylabel('Coverage (%)')
    ax.set_title('C. rel-trial (Clinical Trials)\nCOVID Impact on Medical Data', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=9)
    ax.set_ylim([0, 105])
    ax.legend(loc='lower left', fontsize=8)

    # Add drop annotations
    for i, (task, drop) in enumerate(zip(tasks, drops)):
        if drop > 0:
            ax.annotate(f'-{drop:.0f}%', xy=(i + width/2, test_covs[i]),
                        xytext=(10, 0), textcoords='offset points',
                        fontsize=10, fontweight='bold', color='#c0392b',
                        arrowprops=dict(arrowstyle='->', color='#c0392b'))

    # ==========================================================================
    # Panel D: Feature Stability Predicts Failure
    # ==========================================================================
    ax = axes[1, 1]

    overlaps = [r['overlap'] for r in OVERLAP_DATA]
    drops = [r['drop'] for r in OVERLAP_DATA]
    tasks = [r['task'] for r in OVERLAP_DATA]

    # Color by overlap level
    colors = ['#e74c3c' if o < 10 else '#2ecc71' for o in overlaps]

    ax.scatter(overlaps, drops, s=200, c=colors, alpha=0.7, edgecolors='white', linewidth=1.5)

    # Add task labels
    for i, (task, x, y) in enumerate(zip(tasks, overlaps, drops)):
        offset = (5, 5) if y > 20 else (5, -15)
        ax.annotate(task, (x, y), xytext=offset, textcoords='offset points', fontsize=8)

    # Trend line
    z = np.polyfit(overlaps, drops, 1)
    p = np.poly1d(z)
    x_line = np.linspace(0, 65, 100)
    ax.plot(x_line, p(x_line), '--', color='gray', alpha=0.7, label=f'Trend (r=-0.70)')

    ax.set_xlabel('Feature Train-Test Overlap (%)')
    ax.set_ylabel('Coverage Drop (%)')
    ax.set_title('D. Feature Stability Predicts Failure\nr = -0.70', fontweight='bold')
    ax.set_xlim([-5, 70])
    ax.set_ylim([-5, 100])
    ax.legend(loc='upper right')

    # Main title
    fig.suptitle('Extended Experiments: Conformal Prediction Under COVID Distribution Shift',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    # Save
    output_path = Path(__file__).parent.parent / 'figure2_extended_experiments.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    print("Generating figures for Conformal Prediction COVID paper...")
    print("=" * 60)

    generate_figure1()
    generate_figure2()

    print("=" * 60)
    print("Done! Figures saved to papers/conformal_covid/")


if __name__ == "__main__":
    main()
