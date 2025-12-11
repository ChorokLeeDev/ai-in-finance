"""
Figure 2: Regime-Dependent Causal Structure

This is the KEY figure showing that causal links emerge only in specific regimes:
- HML → SMB: Only in Crisis regime (p = 1.89e-05, lag = 9 days)
- SMB → HML: Only in Crowding regime (p = 1.94e-04, lag = 3 days)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import warnings
warnings.filterwarnings('ignore')


def create_regime_dag_figure():
    """Create the key figure showing regime-dependent causal structure."""

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Factor positions (circular layout for 6 factors)
    factor_names = ['MKT', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']
    n_factors = len(factor_names)

    # Positions in a circle
    angles = np.linspace(0, 2*np.pi, n_factors, endpoint=False) - np.pi/2
    radius = 0.35
    positions = {name: (0.5 + radius * np.cos(angle), 0.5 + radius * np.sin(angle))
                 for name, angle in zip(factor_names, angles)}

    # Regime-specific edges (significant at p < 0.01)
    # From Gate 3 Granger results
    regime_edges = {
        'Normal': [
            ('MKT', 'SMB', 5, 2.91e-16),
            ('MKT', 'HML', 5, 2.88e-12),
            ('MKT', 'RMW', 5, None),  # significant
            ('MKT', 'CMA', 5, 4.34e-09),
            ('MKT', 'MOM', 5, 1.51e-18),
            ('MOM', 'CMA', 7, 1.36e-13),
            ('HML', 'CMA', 7, None),
            ('HML', 'MOM', 7, None),
        ],
        'Crowding': [
            ('MKT', 'SMB', 7, None),
            ('MKT', 'HML', 7, 4.97e-09),
            ('MKT', 'RMW', 7, 7.65e-19),
            ('MKT', 'CMA', 13, 9.45e-07),
            ('MKT', 'MOM', 5, 1.01e-12),
            ('SMB', 'HML', 3, 1.94e-04),  # KEY: SMB → HML only here
            ('SMB', 'RMW', 5, None),
            ('SMB', 'CMA', 2, None),
            ('SMB', 'MOM', 6, None),
            ('MOM', 'RMW', 2, 3.23e-05),
        ],
        'Crisis': [
            ('MKT', 'SMB', 4, 6.49e-21),
            ('MKT', 'HML', 2, None),
            ('MKT', 'RMW', 3, None),
            ('MKT', 'CMA', 6, None),
            ('MKT', 'MOM', 1, 1.79e-05),
            ('HML', 'SMB', 9, 1.89e-05),  # KEY: HML → SMB only here
            ('HML', 'CMA', 2, None),
            ('HML', 'MOM', 4, None),
            ('SMB', 'MKT', 4, 3.12e-06),
            ('RMW', 'CMA', 13, 4.08e-08),
        ]
    }

    # Highlight edges (the key finding)
    highlight_edges = {
        'Normal': [],
        'Crowding': [('SMB', 'HML')],
        'Crisis': [('HML', 'SMB')]
    }

    regime_names = ['Normal', 'Crowding', 'Crisis']
    regime_colors = ['#2ecc71', '#f39c12', '#e74c3c']  # Green, Orange, Red

    for idx, (ax, regime) in enumerate(zip(axes, regime_names)):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')

        # Background color
        ax.set_facecolor('#f8f9fa')

        # Draw nodes
        for name, (x, y) in positions.items():
            # Highlight HML and SMB
            if name in ['HML', 'SMB']:
                circle = plt.Circle((x, y), 0.08, color=regime_colors[idx],
                                   alpha=0.3, zorder=1)
                ax.add_patch(circle)
                circle2 = plt.Circle((x, y), 0.06, color='white', zorder=2)
                ax.add_patch(circle2)
            else:
                circle = plt.Circle((x, y), 0.06, color='lightgray',
                                   alpha=0.5, zorder=1)
                ax.add_patch(circle)

            ax.text(x, y, name, ha='center', va='center', fontsize=10,
                   fontweight='bold', zorder=3)

        # Draw edges
        edges = regime_edges[regime]
        highlights = highlight_edges[regime]

        for edge_info in edges:
            src, dst = edge_info[0], edge_info[1]
            lag = edge_info[2]

            x1, y1 = positions[src]
            x2, y2 = positions[dst]

            # Adjust for node radius
            dx, dy = x2 - x1, y2 - y1
            dist = np.sqrt(dx**2 + dy**2)

            # Start and end points adjusted for node size
            node_radius = 0.07
            x1_adj = x1 + node_radius * dx / dist
            y1_adj = y1 + node_radius * dy / dist
            x2_adj = x2 - node_radius * dx / dist
            y2_adj = y2 - node_radius * dy / dist

            # Check if this is a highlight edge
            is_highlight = (src, dst) in highlights

            if is_highlight:
                # Prominent arrow for key edges
                arrow = FancyArrowPatch(
                    (x1_adj, y1_adj), (x2_adj, y2_adj),
                    arrowstyle='-|>', mutation_scale=20,
                    color=regime_colors[idx], linewidth=3,
                    zorder=5
                )
                ax.add_patch(arrow)

                # Add lag label
                mid_x = (x1_adj + x2_adj) / 2
                mid_y = (y1_adj + y2_adj) / 2
                ax.text(mid_x + 0.05, mid_y + 0.05, f'{lag}d',
                       fontsize=9, color=regime_colors[idx], fontweight='bold')
            else:
                # Subtle arrow for other edges
                arrow = FancyArrowPatch(
                    (x1_adj, y1_adj), (x2_adj, y2_adj),
                    arrowstyle='-|>', mutation_scale=10,
                    color='gray', linewidth=0.8, alpha=0.4,
                    zorder=4
                )
                ax.add_patch(arrow)

        # Title
        ax.set_title(f'{regime} Regime', fontsize=14, fontweight='bold',
                    color=regime_colors[idx], pad=10)

        # Subtitle with key info
        if regime == 'Normal':
            subtitle = 'No HML↔SMB link'
        elif regime == 'Crowding':
            subtitle = 'SMB → HML emerges\n(p = 1.94e-04, lag = 3d)'
        else:  # Crisis
            subtitle = 'HML → SMB emerges\n(p = 1.89e-05, lag = 9d)'

        ax.text(0.5, 0.02, subtitle, ha='center', va='bottom',
               fontsize=10, style='italic', transform=ax.transAxes)

    # Main title
    fig.suptitle('Regime-Dependent Causal Structure Between Factors',
                fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig('/Users/i767700/Github/ai-in-finance/chorok/v11_causal_factor_crowding/fig2_regime_causal.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    print("Saved: fig2_regime_causal.png")
    plt.close()


def create_simplified_dag_figure():
    """Create a simplified version focusing on HML-SMB relationship."""

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    regime_names = ['Normal', 'Crowding', 'Crisis']
    regime_colors = ['#27ae60', '#f39c12', '#c0392b']

    for idx, (ax, regime) in enumerate(zip(axes, regime_names)):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')

        # Draw HML and SMB nodes
        hml_pos = (0.3, 0.5)
        smb_pos = (0.7, 0.5)

        # Nodes
        for name, pos in [('HML', hml_pos), ('SMB', smb_pos)]:
            circle = plt.Circle(pos, 0.12, color=regime_colors[idx],
                               alpha=0.2, zorder=1)
            ax.add_patch(circle)
            circle2 = plt.Circle(pos, 0.10, color='white', zorder=2)
            ax.add_patch(circle2)
            ax.text(pos[0], pos[1], name, ha='center', va='center',
                   fontsize=14, fontweight='bold', zorder=3)

        # Draw arrow based on regime
        if regime == 'Normal':
            # No arrow - just dashed line
            ax.plot([0.42, 0.58], [0.5, 0.5], 'k--', alpha=0.3, linewidth=2)
            ax.text(0.5, 0.35, 'No significant\ncausal link', ha='center',
                   fontsize=10, style='italic', color='gray')

        elif regime == 'Crowding':
            # SMB → HML (right to left)
            arrow = FancyArrowPatch(
                (0.58, 0.5), (0.42, 0.5),
                arrowstyle='-|>', mutation_scale=25,
                color=regime_colors[idx], linewidth=4,
                zorder=5
            )
            ax.add_patch(arrow)
            ax.text(0.5, 0.65, 'p = 1.94e-04', ha='center',
                   fontsize=10, fontweight='bold', color=regime_colors[idx])
            ax.text(0.5, 0.35, '3-day lag', ha='center',
                   fontsize=10, color=regime_colors[idx])

        else:  # Crisis
            # HML → SMB (left to right)
            arrow = FancyArrowPatch(
                (0.42, 0.5), (0.58, 0.5),
                arrowstyle='-|>', mutation_scale=25,
                color=regime_colors[idx], linewidth=4,
                zorder=5
            )
            ax.add_patch(arrow)
            ax.text(0.5, 0.65, 'p = 1.89e-05', ha='center',
                   fontsize=10, fontweight='bold', color=regime_colors[idx])
            ax.text(0.5, 0.35, '9-day lag', ha='center',
                   fontsize=10, color=regime_colors[idx])

        # Title
        ax.set_title(f'{regime}', fontsize=14, fontweight='bold',
                    color=regime_colors[idx], pad=15)

    # Main title
    fig.suptitle('Key Finding: Causal Direction Between HML and SMB is Regime-Dependent',
                fontsize=14, fontweight='bold', y=1.05)

    # Annotation
    fig.text(0.5, -0.05,
            'Value (HML) → Size (SMB) causality emerges only during Crisis.\n'
            'Size (SMB) → Value (HML) causality emerges only during Crowding.',
            ha='center', fontsize=11, style='italic')

    plt.tight_layout()
    plt.savefig('/Users/i767700/Github/ai-in-finance/chorok/v11_causal_factor_crowding/fig2_hml_smb_simple.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    print("Saved: fig2_hml_smb_simple.png")
    plt.close()


def create_timeline_with_causality():
    """Create timeline showing when causal links appear."""

    # Load data and run regime detection
    from gate2_regime_detection import StudentTHMM, load_and_prepare_data

    crowding = load_and_prepare_data()
    X = crowding.values

    # Fit HMM
    hmm = StudentTHMM(n_regimes=3, n_iter=100)
    hmm.fit(X)
    regimes = hmm.predict(X)

    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    dates = crowding.index

    # Panel 1: HML volatility
    ax = axes[0]
    hml_idx = list(crowding.columns).index('HML')
    ax.plot(dates, X[:, hml_idx], 'k-', alpha=0.7, linewidth=0.5)
    ax.set_ylabel('HML Crowding\n(Rolling Vol)', fontsize=10)
    ax.set_title('Factor Crowding and Regime-Dependent Causality', fontsize=14, fontweight='bold')

    # Panel 2: SMB volatility
    ax = axes[1]
    smb_idx = list(crowding.columns).index('SMB')
    ax.plot(dates, X[:, smb_idx], 'k-', alpha=0.7, linewidth=0.5)
    ax.set_ylabel('SMB Crowding\n(Rolling Vol)', fontsize=10)

    # Panel 3: Regimes with causality annotation
    ax = axes[2]

    regime_colors = ['#27ae60', '#f39c12', '#c0392b']
    regime_names = ['Normal', 'Crowding', 'Crisis']

    for k in range(3):
        mask = regimes == k
        ax.fill_between(dates, 0, 1, where=mask,
                       color=regime_colors[k], alpha=0.5,
                       label=regime_names[k])

    ax.set_ylabel('Regime', fontsize=10)
    ax.set_yticks([])
    ax.legend(loc='upper right', ncol=3)

    # Add causality annotations
    # Find crisis periods
    crisis_mask = regimes == 2
    crisis_starts = np.where(np.diff(crisis_mask.astype(int)) == 1)[0]

    for start in crisis_starts[:3]:  # Annotate first 3 crisis periods
        if start < len(dates):
            ax.annotate('HML→SMB\nactive',
                       xy=(dates[start], 0.5),
                       xytext=(dates[min(start+100, len(dates)-1)], 0.8),
                       fontsize=8, color='#c0392b',
                       arrowprops=dict(arrowstyle='->', color='#c0392b', alpha=0.7))

    # X-axis formatting
    ax.set_xlabel('Date', fontsize=10)

    plt.tight_layout()
    plt.savefig('/Users/i767700/Github/ai-in-finance/chorok/v11_causal_factor_crowding/fig3_timeline_causality.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    print("Saved: fig3_timeline_causality.png")
    plt.close()


if __name__ == "__main__":
    print("Creating Figure 2: Regime-Dependent Causal Structure")
    print("=" * 60)

    # Create main figure (full DAG)
    create_regime_dag_figure()

    # Create simplified figure (HML-SMB only)
    create_simplified_dag_figure()

    # Create timeline with causality
    print("\nCreating timeline figure...")
    create_timeline_with_causality()

    print("\n" + "=" * 60)
    print("KEY MESSAGE FOR PAPER:")
    print("=" * 60)
    print("""
The causal direction between Value (HML) and Size (SMB) factors
is REGIME-DEPENDENT:

  Normal regime:   No significant causal link
  Crowding regime: SMB → HML (Size drives Value, 3-day lag)
  Crisis regime:   HML → SMB (Value drives Size, 9-day lag)

This is a NOVEL finding - existing literature studies correlation,
not regime-dependent directed causality.
""")
