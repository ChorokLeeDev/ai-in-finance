"""
Generate complexity spectrum figure: grouped bar chart of MSE improvement %
across 3 regimes for 4 model types (Linear, RF, MLP, LSTM).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- rcParams for academic style ----------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
})

# ---------- data ----------
regimes = ["Normal", "Elevated", "Crisis"]
models = ["Linear", "RF", "MLP", "LSTM"]
colors = ["#4878CF", "#6ACC65", "#EE854A", "#D65F5F"]  # blue, green, orange, red

# MSE improvement % (positive = HML helps)
# Linear/RF/MLP from neural_granger_selected.json (selected HMM fit)
# LSTM from lstm_granger_results.json (separate initialization, see footnote 1)
data = {
    "Normal":   [0.73, -1.42, -5.85, -8.80],
    "Elevated": [0.30,  0.13, -4.98, -15.44],
    "Crisis":   [1.25,  0.49, -6.51, -13.56],
}

# Significance (p-values)
p_values = {
    "Normal":   [0.0005, 1.00, 0.64, 0.63],
    "Elevated": [0.595,  0.96, 0.91, 0.93],
    "Crisis":   [0.013,  0.96, 0.93, 0.83],
}

# ---------- plot ----------
n_regimes = len(regimes)
n_models = len(models)
bar_width = 0.18
x = np.arange(n_regimes)

fig, ax = plt.subplots(figsize=(6, 4))

# Draw zero line
ax.axhline(0, color="black", linewidth=0.6, linestyle="--", zorder=1)

# Bars
for i, model in enumerate(models):
    offsets = x + (i - (n_models - 1) / 2) * bar_width
    vals = [data[r][i] for r in regimes]
    ax.bar(
        offsets, vals, bar_width,
        label=model, color=colors[i], edgecolor="white", linewidth=0.5, zorder=2,
    )

# Significance asterisks + collect positions for bracket
sig_xs = []
sig_ys = []
for ri, regime in enumerate(regimes):
    for mi in range(n_models):
        p = p_values[regime][mi]
        if p < 0.05:
            bar_x = x[ri] + (mi - (n_models - 1) / 2) * bar_width
            bar_y = data[regime][mi]
            offset_y = 0.6
            star = "**" if p < 0.001 else "*"
            ax.text(
                bar_x, bar_y + offset_y, star,
                ha="center", va="bottom", fontsize=10, fontweight="bold",
            )
            sig_xs.append(bar_x)
            sig_ys.append(bar_y + offset_y)

# Bracket annotation: "Linear only" spanning the three significant bars
bracket_y = max(sig_ys) + 2.0
ax.annotate(
    "", xy=(sig_xs[0], bracket_y), xytext=(sig_xs[-1], bracket_y),
    arrowprops=dict(arrowstyle="-", lw=0.8, color="0.35"),
)
# Small ticks at bracket ends
tick_h = 0.5
for bx in [sig_xs[0], sig_xs[-1]]:
    ax.plot([bx, bx], [bracket_y - tick_h, bracket_y], color="0.35", lw=0.8)
ax.text(
    np.mean(sig_xs), bracket_y + 0.3, "Linear only",
    ha="center", va="bottom", fontsize=8, fontstyle="italic", color="0.25",
)

# ---------- formatting ----------
ax.set_xticks(x)
ax.set_xticklabels(regimes)
ax.set_ylabel("MSE Improvement (%)")
ax.set_xlabel("HMM Regime")

# Remove top/right spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Legend
ax.legend(loc="lower left", frameon=True, framealpha=0.9, edgecolor="0.8", fontsize=8)

# Y-axis limits with some padding
all_vals = [v for regime_vals in data.values() for v in regime_vals]
y_min = min(all_vals) - 3
y_max = bracket_y + 2.5
ax.set_ylim(y_min, y_max)

fig.tight_layout()

# ---------- save ----------
out_dir = "/Users/i767700/Github/ai-in-finance/papers/causal_regimes/figures"
fig.savefig(f"{out_dir}/complexity_spectrum.pdf", dpi=300)
fig.savefig(f"{out_dir}/complexity_spectrum.png", dpi=300)
print(f"Saved: {out_dir}/complexity_spectrum.pdf")
print(f"Saved: {out_dir}/complexity_spectrum.png")
