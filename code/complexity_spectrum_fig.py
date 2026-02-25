"""
Generate complexity spectrum figure from recomputed JSON artifacts.

Reads:
  - results/neural_granger_selected.json
  - results/lstm_granger_results.json

Writes:
  - figures/complexity_spectrum.pdf
  - figures/complexity_spectrum.png
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REGIMES = ["Normal", "Elevated", "Crisis"]
MODELS = ["Linear", "RF", "MLP", "LSTM"]
COLORS = ["#4878CF", "#6ACC65", "#EE854A", "#D65F5F"]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_metrics(results_dir):
    neural_path = Path(results_dir) / "neural_granger_selected.json"
    lstm_path = Path(results_dir) / "lstm_granger_results.json"

    if not neural_path.exists():
        raise FileNotFoundError(f"Missing required file: {neural_path}")
    if not lstm_path.exists():
        raise FileNotFoundError(f"Missing required file: {lstm_path}")

    neural = load_json(neural_path)
    lstm = load_json(lstm_path)

    data = {}
    p_values = {}
    for regime in REGIMES:
        n_row = neural["results"].get(regime, {})
        l_row = lstm["per_regime_granger"].get(regime, {})
        if not n_row or not l_row:
            raise KeyError(f"Missing regime '{regime}' in neural or LSTM results.")

        data[regime] = [
            float(n_row["linear_mse_improvement_pct"]),
            float(n_row["rf_mse_improvement_pct"]),
            float(n_row["mlp_mse_improvement_pct"]),
            float(l_row["mse_improvement_pct"]),
        ]
        p_values[regime] = [
            float(n_row["linear_p_value"]),
            float(n_row["rf_p_value"]),
            float(n_row["mlp_p_value"]),
            float(l_row["permutation_p_value"]),
        ]
    return data, p_values


def plot_complexity_spectrum(data, p_values, output_dir):
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
        }
    )

    n_regimes = len(REGIMES)
    n_models = len(MODELS)
    bar_width = 0.18
    x = np.arange(n_regimes)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--", zorder=1)

    sig_xs = []
    sig_ys = []
    for model_idx, model in enumerate(MODELS):
        offsets = x + (model_idx - (n_models - 1) / 2) * bar_width
        vals = [data[r][model_idx] for r in REGIMES]
        ax.bar(
            offsets,
            vals,
            bar_width,
            label=model,
            color=COLORS[model_idx],
            edgecolor="white",
            linewidth=0.5,
            zorder=2,
        )
        for regime_idx, regime in enumerate(REGIMES):
            p = p_values[regime][model_idx]
            if p < 0.05:
                bar_x = offsets[regime_idx]
                bar_y = vals[regime_idx]
                star = "**" if p < 0.001 else "*"
                y_offset = 0.6 if bar_y >= 0 else -0.8
                va = "bottom" if bar_y >= 0 else "top"
                ax.text(
                    bar_x,
                    bar_y + y_offset,
                    star,
                    ha="center",
                    va=va,
                    fontsize=10,
                    fontweight="bold",
                )
                sig_xs.append(bar_x)
                sig_ys.append(bar_y + y_offset)

    if sig_xs:
        bracket_y = max(sig_ys) + 2.0
        left, right = min(sig_xs), max(sig_xs)
        ax.annotate(
            "",
            xy=(left, bracket_y),
            xytext=(right, bracket_y),
            arrowprops=dict(arrowstyle="-", lw=0.8, color="0.35"),
        )
        tick_h = 0.5
        for bx in [left, right]:
            ax.plot([bx, bx], [bracket_y - tick_h, bracket_y], color="0.35", lw=0.8)
        ax.text(
            (left + right) / 2.0,
            bracket_y + 0.3,
            "Significant bars",
            ha="center",
            va="bottom",
            fontsize=8,
            fontstyle="italic",
            color="0.25",
        )
    else:
        bracket_y = max(v for row in data.values() for v in row) + 2.0

    ax.set_xticks(x)
    ax.set_xticklabels(REGIMES)
    ax.set_ylabel("MSE Improvement (%)")
    ax.set_xlabel("HMM Regime")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower left", frameon=True, framealpha=0.9, edgecolor="0.8", fontsize=8)

    all_vals = [v for row in data.values() for v in row]
    y_min = min(all_vals) - 3
    y_max = bracket_y + 2.5
    ax.set_ylim(y_min, y_max)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "complexity_spectrum.pdf"
    png_path = output_dir / "complexity_spectrum.png"
    fig.savefig(pdf_path, dpi=300)
    fig.savefig(png_path, dpi=300)
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate complexity spectrum figure from result JSONs.")
    parser.add_argument(
        "--results-dir",
        default="/Users/i767700/Github/ai-in-finance/papers/causal_regimes/results",
        help="Directory containing recomputed JSON artifacts.",
    )
    parser.add_argument(
        "--figures-dir",
        default="/Users/i767700/Github/ai-in-finance/papers/causal_regimes/figures",
        help="Output directory for figures.",
    )
    args = parser.parse_args()

    data, p_values = load_metrics(args.results_dir)
    plot_complexity_spectrum(data, p_values, Path(args.figures_dir))


if __name__ == "__main__":
    main()
