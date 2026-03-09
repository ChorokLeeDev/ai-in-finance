#!/usr/bin/env python3
"""
Generate Figure: SHAP Concentration vs Coverage Drop, n=16 multiclass tasks (9 domains).
Primary result: rho=0.853, p<0.001.

Output: results/figure_n16_correlation.pdf
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import spearmanr

# ── Data ──────────────────────────────────────────────────────────────────────
# 8 SALT tasks (supply-chain domain, COVID temporal shift)
salt_tasks = [
    ("s-shipcond",   50.7,  71.6),
    ("s-payterms",   54.2,  77.1),
    ("s-group",      47.3,  71.2),
    ("i-shippoint",  48.8,  18.5),
    ("s-office",     42.6,   0.1),
    ("i-incoterms",  28.9,  11.3),
    ("i-plant",      23.9,  10.6),
    ("s-incoterms",  23.7,   8.5),
]

# 8 external multiclass tasks (7 non-supply-chain domains, 10-seed means)
ext_tasks = [
    ("Covertype",   49.78,  81.8),
    ("KDDCup99",    21.13,  15.9),
    ("Gas Sensor",   7.27,  -3.8),
    ("Avila",       20.49,   1.0),
    ("Shuttle",     30.66,   0.3),
    ("PAMAP2",      19.24,  -2.1),
    ("Pendigits",   14.45,  -1.7),
    ("Satimage",     9.04,  -0.3),
]

# Verify correlation
all_c   = [t[1] for t in salt_tasks + ext_tasks]
all_d   = [t[2] for t in salt_tasks + ext_tasks]
rho, p  = spearmanr(all_c, all_d)
print(f"n=16 Spearman rho={rho:.3f}, p={p:.4f}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.5, 4.2))

# SALT points (dark filled circles)
salt_c = [t[1] for t in salt_tasks]
salt_d = [t[2] for t in salt_tasks]
ax.scatter(salt_c, salt_d, color='#1f4e79', s=60, zorder=5,
           marker='o', label='SALT (supply-chain, COVID)')

# External points (orange triangles)
ext_c = [t[1] for t in ext_tasks]
ext_d = [t[2] for t in ext_tasks]
ax.scatter(ext_c, ext_d, color='#c55a11', s=60, zorder=5,
           marker='^', label='External (8 domains, 10 seeds)')

# 40% threshold
ax.axvline(40, color='gray', linestyle='--', linewidth=1.2, alpha=0.8, label='40\\% threshold')

# Labels for key points
labels = {
    "Covertype":   ( 1.5, -3),
    "s-payterms":  (-12,   2),
    "s-shipcond":  (-13,   2),
    "s-group":     ( 1.5,  2),
    "s-office":    ( 1.5,  2),
    "KDDCup99":    ( 1.5,  2),
    "Gas Sensor":  ( 1.5, -5),
}

# Add task labels for SALT
for name, c, d in salt_tasks:
    short = name.replace("s-", "s-").replace("i-", "i-")
    ax.annotate(short, (c, d), fontsize=6.5, color='#1f4e79',
                xytext=(2, 2), textcoords='offset points')

for name, c, d in ext_tasks:
    ax.annotate(name, (c, d), fontsize=6.5, color='#c55a11',
                xytext=(2, 2), textcoords='offset points')

# Annotation box
ax.text(0.04, 0.97,
        f'$\\rho={rho:.3f}$, $p<0.001$\n$n=16$, 9 domains',
        transform=ax.transAxes, fontsize=9, va='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9))

ax.set_xlabel('SHAP Concentration $C$ (\\%)', fontsize=10)
ax.set_ylabel('Coverage Drop (pp)', fontsize=10)
ax.set_title('SHAP Concentration vs.~Coverage Drop\n(16 multiclass tasks, 9 domains)',
             fontsize=10)
ax.legend(fontsize=8, loc='lower right')
ax.set_xlim(-2, 65)
ax.set_ylim(-15, 90)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = '/Users/i767700/Github/ai-in-finance/papers/conformal_covid/results/figure_n16_correlation.pdf'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved: {out}")
