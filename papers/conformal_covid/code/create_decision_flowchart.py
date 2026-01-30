#!/usr/bin/env python3
"""
Create Decision Flowchart for 2D Framework

Generates a visual flowchart showing the 2D decision process:
1. Check concentration threshold (40%)
2. If high, check for protective factors
3. Classify as ROBUST or VULNERABLE

Output: PDF figure for paper (Figure 4 or similar)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.patches import Rectangle
import numpy as np

def create_flowchart():
    """Create 2D framework decision flowchart."""

    # Create figure with good size for paper
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Colors
    color_input = '#E8F4F8'      # Light blue
    color_decision = '#FFF4E6'   # Light orange
    color_robust = '#E8F5E9'     # Light green
    color_vulnerable = '#FFEBEE' # Light red
    color_text = '#333333'

    # Font sizes
    fs_title = 11
    fs_box = 10
    fs_arrow = 9

    # === START: Input box ===
    start_x, start_y = 5, 9
    input_box = FancyBboxPatch(
        (start_x - 1.5, start_y - 0.3), 3, 0.6,
        boxstyle="round,pad=0.1",
        edgecolor='black', facecolor=color_input, linewidth=2
    )
    ax.add_patch(input_box)
    ax.text(start_x, start_y, 'Compute SHAP\nConcentration',
            ha='center', va='center', fontsize=fs_box, weight='bold',
            color=color_text)

    # Arrow down
    ax.annotate('', xy=(start_x, start_y - 0.3), xytext=(start_x, start_y - 0.8),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # === DECISION 1: Concentration threshold ===
    dec1_x, dec1_y = 5, 7.8
    decision1_box = FancyBboxPatch(
        (dec1_x - 1.8, dec1_y - 0.4), 3.6, 0.8,
        boxstyle="round,pad=0.1",
        edgecolor='black', facecolor=color_decision, linewidth=2
    )
    ax.add_patch(decision1_box)
    ax.text(dec1_x, dec1_y, 'Concentration\n> 40%?',
            ha='center', va='center', fontsize=fs_box, weight='bold',
            color=color_text)

    # === LEFT BRANCH: No (concentration ≤ 40%) → ROBUST ===
    # Arrow left and down
    robust1_x, robust1_y = 2, 6.5
    ax.annotate('', xy=(robust1_x + 1.2, robust1_y),
                xytext=(dec1_x - 1.8, dec1_y),
                arrowprops=dict(arrowstyle='->', lw=2, color='green',
                               connectionstyle="arc3,rad=0.3"))
    ax.text(2.8, dec1_y - 0.5, 'NO', ha='center', va='center',
            fontsize=fs_arrow, color='green', weight='bold')

    # ROBUST outcome (left)
    robust1_box = FancyBboxPatch(
        (robust1_x - 1.2, robust1_y - 0.35), 2.4, 0.7,
        boxstyle="round,pad=0.1",
        edgecolor='green', facecolor=color_robust, linewidth=2.5
    )
    ax.add_patch(robust1_box)
    ax.text(robust1_x, robust1_y, '✓ ROBUST\n(Low concentration)',
            ha='center', va='center', fontsize=fs_box, weight='bold',
            color='green')

    # === RIGHT BRANCH: Yes (concentration > 40%) ===
    # Arrow down
    ax.annotate('', xy=(dec1_x, dec1_y - 0.4), xytext=(dec1_x, dec1_y - 1.2),
                arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    ax.text(dec1_x + 0.5, dec1_y - 0.8, 'YES', ha='left', va='center',
            fontsize=fs_arrow, color='red', weight='bold')

    # === DECISION 2: Protective factors ===
    dec2_x, dec2_y = 5, 5.2
    decision2_box = FancyBboxPatch(
        (dec2_x - 2, dec2_y - 0.5), 4, 1,
        boxstyle="round,pad=0.1",
        edgecolor='black', facecolor=color_decision, linewidth=2
    )
    ax.add_patch(decision2_box)
    ax.text(dec2_x, dec2_y + 0.15, 'Has protective factor?',
            ha='center', va='center', fontsize=fs_box, weight='bold',
            color=color_text)
    ax.text(dec2_x, dec2_y - 0.25, '(Jaccard > 0.5 AND\nImportance > 15%)',
            ha='center', va='center', fontsize=8, style='italic',
            color=color_text)

    # === LEFT BRANCH from DECISION 2: Yes → ROBUST ===
    # Arrow left
    robust2_x, robust2_y = 2, 4
    ax.annotate('', xy=(robust2_x, robust2_y + 0.35), xytext=(dec2_x - 2, dec2_y - 0.2),
                arrowprops=dict(arrowstyle='->', lw=2, color='green',
                               connectionstyle="arc3,rad=0.3"))
    ax.text(2.8, dec2_y - 0.6, 'YES', ha='center', va='top',
            fontsize=fs_arrow, color='green', weight='bold')

    # ROBUST outcome (bottom left)
    robust2_box = FancyBboxPatch(
        (robust2_x - 1.2, robust2_y - 0.35), 2.4, 0.7,
        boxstyle="round,pad=0.1",
        edgecolor='green', facecolor=color_robust, linewidth=2.5
    )
    ax.add_patch(robust2_box)
    ax.text(robust2_x, robust2_y, '✓ ROBUST\n(Protected)',
            ha='center', va='center', fontsize=fs_box, weight='bold',
            color='green')

    # Example annotation for sales-office
    ax.text(robust2_x, robust2_y - 0.7, 'e.g., sales-office\n(SALESORGANIZATION)',
            ha='center', va='top', fontsize=7.5, style='italic',
            color='green', bbox=dict(boxstyle='round,pad=0.3',
                                    facecolor='white', edgecolor='green',
                                    linewidth=1, alpha=0.8))

    # === RIGHT BRANCH from DECISION 2: No → VULNERABLE ===
    # Arrow right
    vuln_x, vuln_y = 8, 4
    ax.annotate('', xy=(vuln_x, vuln_y + 0.35), xytext=(dec2_x + 2, dec2_y - 0.2),
                arrowprops=dict(arrowstyle='->', lw=2, color='red',
                               connectionstyle="arc3,rad=-0.3"))
    ax.text(7.2, dec2_y - 0.6, 'NO', ha='center', va='top',
            fontsize=fs_arrow, color='red', weight='bold')

    # VULNERABLE outcome (bottom right)
    vuln_box = FancyBboxPatch(
        (vuln_x - 1.2, vuln_y - 0.35), 2.4, 0.7,
        boxstyle="round,pad=0.1",
        edgecolor='red', facecolor=color_vulnerable, linewidth=2.5
    )
    ax.add_patch(vuln_box)
    ax.text(vuln_x, vuln_y, '✗ VULNERABLE\n(Retraining needed)',
            ha='center', va='center', fontsize=fs_box, weight='bold',
            color='red')

    # Example annotation
    ax.text(vuln_x, vuln_y - 0.7, 'e.g., sales-group,\nsales-payterms',
            ha='center', va='top', fontsize=7.5, style='italic',
            color='red', bbox=dict(boxstyle='round,pad=0.3',
                                  facecolor='white', edgecolor='red',
                                  linewidth=1, alpha=0.8))

    # === ACTIONS box at bottom ===
    action_y = 1.8

    # Robust action
    ax.text(2, action_y, '→ Skip retraining\n   (save cost)',
            ha='center', va='center', fontsize=8.5,
            color='green', weight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                     edgecolor='green', linewidth=1.5))

    # Vulnerable action
    ax.text(8, action_y, '→ Quarterly retraining\n   (maintain coverage)',
            ha='center', va='center', fontsize=8.5,
            color='red', weight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                     edgecolor='red', linewidth=1.5))

    # Arrows to actions
    ax.annotate('', xy=(2, action_y + 0.4), xytext=(2, robust2_y - 0.35),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='green',
                               linestyle='--', alpha=0.7))
    ax.annotate('', xy=(8, action_y + 0.4), xytext=(8, vuln_y - 0.35),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='red',
                               linestyle='--', alpha=0.7))

    # === Title ===
    ax.text(5, 9.7, '2D Decision Framework for Task Vulnerability',
            ha='center', va='bottom', fontsize=12, weight='bold',
            color=color_text)

    # === Legend/Note ===
    note_text = (
        'Framework improves accuracy from 75% (1D) to 87.5% (2D) by checking\n'
        'both concentration AND secondary feature stability.'
    )
    ax.text(5, 0.5, note_text,
            ha='center', va='center', fontsize=8, style='italic',
            color=color_text,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#F5F5F5',
                     edgecolor='#CCCCCC', linewidth=1))

    plt.tight_layout()
    return fig


def main():
    """Generate flowchart and save to file."""
    import os
    os.chdir('/Users/i767700/Github/ai-in-finance/papers/conformal_covid')

    print("Creating 2D decision framework flowchart...")
    fig = create_flowchart()

    # Save as PDF (high quality for paper)
    output_pdf = 'figure_decision_framework_2d.pdf'
    fig.savefig(output_pdf, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"✓ PDF saved: {output_pdf}")

    # Save as PNG (for preview)
    output_png = 'figure_decision_framework_2d.png'
    fig.savefig(output_png, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"✓ PNG saved: {output_png}")

    print("\nFlowchart complete! Ready to include in paper.")
    print("Suggested LaTeX caption:")
    print("=" * 70)
    print(r"\caption{2D Decision Framework for Task Vulnerability Assessment.")
    print(r"The framework first checks SHAP concentration (threshold: 40\%).")
    print(r"Tasks with high concentration are further evaluated for protective")
    print(r"factors (stable secondary features with Jaccard $> 0.5$ and")
    print(r"importance $> 15\%$). This 2D approach improves classification")
    print(r"accuracy from 75\% to 87.5\%, correctly handling outliers like")
    print(r"\texttt{sales-office} that have high concentration but remain")
    print(r"robust due to protective factors.}")
    print("=" * 70)


if __name__ == "__main__":
    main()
