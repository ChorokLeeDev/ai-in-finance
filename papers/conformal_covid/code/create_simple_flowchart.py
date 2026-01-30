#!/usr/bin/env python3
"""
Simple, clean flowchart with straight arrows only.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def create_simple_flowchart():
    """Create clean flowchart with straight lines only."""

    fig, ax = plt.subplots(1, 1, figsize=(10, 9))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Colors
    color_input = '#E8F4F8'
    color_decision = '#FFF4E6'
    color_robust = '#E8F5E9'
    color_vulnerable = '#FFEBEE'

    # Box width
    w_input = 3.0
    w_decision = 3.6
    w_outcome = 2.8

    # Y positions
    y_start = 9.0
    y_dec1 = 7.5
    y_robust1 = 6.0
    y_dec2 = 4.5
    y_outcomes = 3.0
    y_actions = 1.2

    # X positions (3 columns)
    x_left = 2.0
    x_center = 5.0
    x_right = 8.0

    # === Title ===
    ax.text(5, 9.7, '2D Decision Framework for Task Vulnerability',
            ha='center', va='bottom', fontsize=12, weight='bold')

    # === 1. START ===
    box = FancyBboxPatch((x_center - w_input/2, y_start - 0.3), w_input, 0.6,
                         boxstyle="round,pad=0.1", edgecolor='black',
                         facecolor=color_input, linewidth=2)
    ax.add_patch(box)
    ax.text(x_center, y_start, 'Compute SHAP\nConcentration',
            ha='center', va='center', fontsize=10, weight='bold')

    # Arrow down
    ax.arrow(x_center, y_start - 0.35, 0, -0.6, head_width=0.15,
             head_length=0.1, fc='black', ec='black', lw=2)

    # === 2. DECISION 1: Concentration threshold ===
    box = FancyBboxPatch((x_center - w_decision/2, y_dec1 - 0.4), w_decision, 0.8,
                         boxstyle="round,pad=0.1", edgecolor='black',
                         facecolor=color_decision, linewidth=2)
    ax.add_patch(box)
    ax.text(x_center, y_dec1, 'Concentration\n> 40%?',
            ha='center', va='center', fontsize=10, weight='bold')

    # === 3. LEFT: NO branch (concentration ≤ 40%) ===
    # Horizontal arrow left
    ax.arrow(x_center - w_decision/2 - 0.05, y_dec1, -(x_center - x_left - w_outcome/2 - w_decision/2 - 0.1), 0,
             head_width=0.15, head_length=0.15, fc='green', ec='green', lw=2)
    ax.text(3.5, y_dec1 + 0.25, 'NO', ha='center', va='bottom',
            fontsize=9, color='green', weight='bold')

    # Then down
    ax.arrow(x_left, y_dec1 - 0.05, 0, -(y_dec1 - y_robust1 - 0.3),
             head_width=0.15, head_length=0.1, fc='green', ec='green', lw=2)

    # ROBUST outcome (left)
    box = FancyBboxPatch((x_left - w_outcome/2, y_robust1 - 0.35), w_outcome, 0.7,
                         boxstyle="round,pad=0.1", edgecolor='green',
                         facecolor=color_robust, linewidth=2.5)
    ax.add_patch(box)
    ax.text(x_left, y_robust1, '✓ ROBUST\n(Low concentration)',
            ha='center', va='center', fontsize=10, weight='bold', color='green')

    # === 4. DOWN: YES branch (concentration > 40%) ===
    ax.arrow(x_center, y_dec1 - 0.45, 0, -(y_dec1 - y_dec2 - 0.45),
             head_width=0.15, head_length=0.1, fc='red', ec='red', lw=2)
    ax.text(x_center + 0.4, y_dec1 - 1.0, 'YES', ha='left', va='center',
            fontsize=9, color='red', weight='bold')

    # === 5. DECISION 2: Protective factors ===
    box = FancyBboxPatch((x_center - w_decision/2, y_dec2 - 0.5), w_decision, 1.0,
                         boxstyle="round,pad=0.1", edgecolor='black',
                         facecolor=color_decision, linewidth=2)
    ax.add_patch(box)
    ax.text(x_center, y_dec2 + 0.15, 'Has protective factor?',
            ha='center', va='center', fontsize=10, weight='bold')
    ax.text(x_center, y_dec2 - 0.25, '(Jaccard > 0.5 AND\nImportance > 15%)',
            ha='center', va='center', fontsize=8, style='italic')

    # === 6. LEFT: YES (has protective) → ROBUST ===
    # Horizontal arrow left
    ax.arrow(x_center - w_decision/2 - 0.05, y_dec2, -(x_center - x_left - w_outcome/2 - w_decision/2 - 0.1), 0,
             head_width=0.15, head_length=0.15, fc='green', ec='green', lw=2)
    ax.text(3.5, y_dec2 + 0.25, 'YES', ha='center', va='bottom',
            fontsize=9, color='green', weight='bold')

    # Then down
    ax.arrow(x_left, y_dec2 - 0.05, 0, -(y_dec2 - y_outcomes - 0.3),
             head_width=0.15, head_length=0.1, fc='green', ec='green', lw=2)

    # ROBUST outcome (bottom left)
    box = FancyBboxPatch((x_left - w_outcome/2, y_outcomes - 0.35), w_outcome, 0.7,
                         boxstyle="round,pad=0.1", edgecolor='green',
                         facecolor=color_robust, linewidth=2.5)
    ax.add_patch(box)
    ax.text(x_left, y_outcomes, '✓ ROBUST\n(Protected)',
            ha='center', va='center', fontsize=10, weight='bold', color='green')

    # Example
    ax.text(x_left, y_outcomes - 0.65, 'e.g., sales-office\n(SALESORGANIZATION)',
            ha='center', va='top', fontsize=7.5, style='italic', color='green',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                     edgecolor='green', linewidth=1, alpha=0.8))

    # === 7. RIGHT: NO (no protective) → VULNERABLE ===
    # Horizontal arrow right
    ax.arrow(x_center + w_decision/2 + 0.05, y_dec2, (x_right - x_center - w_outcome/2 - w_decision/2 - 0.1), 0,
             head_width=0.15, head_length=0.15, fc='red', ec='red', lw=2)
    ax.text(6.5, y_dec2 + 0.25, 'NO', ha='center', va='bottom',
            fontsize=9, color='red', weight='bold')

    # Then down
    ax.arrow(x_right, y_dec2 - 0.05, 0, -(y_dec2 - y_outcomes - 0.3),
             head_width=0.15, head_length=0.1, fc='red', ec='red', lw=2)

    # VULNERABLE outcome (bottom right)
    box = FancyBboxPatch((x_right - w_outcome/2, y_outcomes - 0.35), w_outcome, 0.7,
                         boxstyle="round,pad=0.1", edgecolor='red',
                         facecolor=color_vulnerable, linewidth=2.5)
    ax.add_patch(box)
    ax.text(x_right, y_outcomes, '✗ VULNERABLE\n(Retraining needed)',
            ha='center', va='center', fontsize=10, weight='bold', color='red')

    # Example
    ax.text(x_right, y_outcomes - 0.65, 'e.g., sales-group,\nsales-payterms',
            ha='center', va='top', fontsize=7.5, style='italic', color='red',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                     edgecolor='red', linewidth=1, alpha=0.8))

    # === 8. ACTIONS ===
    # Left action
    ax.text(x_left, y_actions, '→ Skip retraining\n   (save cost)',
            ha='center', va='center', fontsize=8.5, color='green', weight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                     edgecolor='green', linewidth=1.5))

    # Right action
    ax.text(x_right, y_actions, '→ Quarterly retraining\n   (maintain coverage)',
            ha='center', va='center', fontsize=8.5, color='red', weight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                     edgecolor='red', linewidth=1.5))

    # Arrows to actions (dashed)
    ax.plot([x_left, x_left], [y_outcomes - 0.35, y_actions + 0.4],
            'g--', lw=1.5, alpha=0.7)
    ax.plot([x_right, x_right], [y_outcomes - 0.35, y_actions + 0.4],
            'r--', lw=1.5, alpha=0.7)

    # Also from top left robust
    ax.plot([x_left, x_left], [y_robust1 - 0.35, y_actions + 0.4],
            'g--', lw=1.5, alpha=0.7)

    # === Note ===
    note_text = (
        'Framework improves accuracy from 75% (1D) to 87.5% (2D) by checking\n'
        'both concentration AND secondary feature stability.'
    )
    ax.text(5, 0.3, note_text, ha='center', va='center', fontsize=8,
            style='italic', bbox=dict(boxstyle='round,pad=0.5',
            facecolor='#F5F5F5', edgecolor='#CCCCCC', linewidth=1))

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    import os
    os.chdir('/Users/i767700/Github/ai-in-finance/papers/conformal_covid')

    print("Creating simple, clean flowchart...")
    fig = create_simple_flowchart()

    # Save
    fig.savefig('figure_decision_framework_2d.pdf', dpi=300,
                bbox_inches='tight', facecolor='white')
    print("✓ PDF saved")

    fig.savefig('figure_decision_framework_2d.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    print("✓ PNG saved")
    print("\n✓ Simple flowchart with straight arrows only - no weird curves!")
