"""
Extended Scale-Up Validation for NeurIPS Paper
===============================================

Validates FK Attribution at large scales (10K, 50K, 100K samples).
Generates publication-ready figure and results table.

Author: ChorokLeeDev
Created: 2025-12-23
"""

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from collections import defaultdict
from scipy.stats import spearmanr
import lightgbm as lgb
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Paths
CACHE_DIR = Path('/Users/i767700/Github/ai-in-finance/chorok/v3_fk_risk_attribution/cache')
RESULTS_DIR = Path('/Users/i767700/Github/ai-in-finance/papers/reluq/results')
FIGURES_DIR = Path('/Users/i767700/Github/ai-in-finance/papers/reluq/figures')

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def train_ensemble(X, y, n_models=5, base_seed=42):
    """Train ensemble for uncertainty estimation."""
    models = []
    for i in range(n_models):
        model = lgb.LGBMRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            random_state=base_seed + i, verbose=-1
        )
        model.fit(X, y)
        models.append(model)
    return models


def get_uncertainty(models, X):
    """Compute ensemble uncertainty (variance)."""
    preds = np.array([m.predict(X) for m in models])
    return preds.var(axis=0)


def get_prediction(models, X):
    """Get ensemble mean prediction."""
    preds = np.array([m.predict(X) for m in models])
    return preds.mean(axis=0)


def get_fk_grouping(col_to_fk):
    """Convert column->FK mapping to FK->columns mapping."""
    fk_to_cols = defaultdict(list)
    for col, fk in col_to_fk.items():
        fk_to_cols[fk].append(col)
    return dict(fk_to_cols)


def compute_fk_attribution(models, X, fk_grouping, n_permute=10):
    """Compute FK attribution via permutation."""
    base_unc = get_uncertainty(models, X).mean()
    results = {}

    for fk_group, cols in fk_grouping.items():
        valid_cols = [c for c in cols if c in X.columns]
        if not valid_cols:
            continue

        deltas = []
        for _ in range(n_permute):
            X_perm = X.copy()
            for col in valid_cols:
                X_perm[col] = np.random.permutation(X_perm[col].values)
            perm_unc = get_uncertainty(models, X_perm).mean()
            deltas.append(perm_unc - base_unc)

        results[fk_group] = np.mean(deltas)

    total = sum(max(0, v) for v in results.values())
    if total > 0:
        return {g: max(0, v) / total * 100 for g, v in results.items()}
    return {g: 0 for g in results}


def compute_error_impact(models, X, y, fk_grouping, n_permute=10):
    """Compute error impact (ground truth)."""
    from sklearn.metrics import mean_absolute_error

    base_pred = get_prediction(models, X)
    base_mae = mean_absolute_error(y, base_pred)
    results = {}

    for fk_group, cols in fk_grouping.items():
        valid_cols = [c for c in cols if c in X.columns]
        if not valid_cols:
            continue

        deltas = []
        for _ in range(n_permute):
            X_perm = X.copy()
            for col in valid_cols:
                X_perm[col] = np.random.permutation(X_perm[col].values)
            perm_pred = get_prediction(models, X_perm)
            perm_mae = mean_absolute_error(y, perm_pred)
            deltas.append(perm_mae - base_mae)

        results[fk_group] = np.mean(deltas)

    total = sum(max(0, v) for v in results.values())
    if total > 0:
        return {g: max(0, v) / total * 100 for g, v in results.items()}
    return {g: 0 for g in results}


def run_scale_experiment(X_full, y_full, col_to_fk, domain_name,
                         sample_sizes=[1000, 5000, 10000, 20000, 50000],
                         n_runs=5):
    """Run scale-up experiment with statistical rigor."""
    print(f"\n{'='*70}")
    print(f"SCALE-UP: {domain_name}")
    print(f"{'='*70}")

    fk_grouping = get_fk_grouping(col_to_fk)
    fk_list = list(fk_grouping.keys())

    print(f"Total samples available: {len(X_full)}")
    print(f"FK groups ({len(fk_list)}): {fk_list}")

    all_results = []

    for sample_size in sample_sizes:
        if sample_size > len(X_full):
            print(f"\n  Skipping n={sample_size} (need {sample_size}, have {len(X_full)})")
            continue

        print(f"\n  --- n = {sample_size:,} ---")

        correlations = []
        times = []

        for run in range(n_runs):
            seed = 42 + run * 10
            np.random.seed(seed)

            # Sample data
            idx = np.random.choice(len(X_full), size=sample_size, replace=False)
            X = X_full.iloc[idx].copy()
            y = y_full.iloc[idx].copy()

            # Train and evaluate
            import time
            start = time.time()
            models = train_ensemble(X, y, n_models=5, base_seed=seed)
            attr = compute_fk_attribution(models, X, fk_grouping, n_permute=10)
            error = compute_error_impact(models, X, y, fk_grouping, n_permute=10)
            elapsed = time.time() - start
            times.append(elapsed)

            # Compute correlation
            attr_vals = [attr.get(fk, 0) for fk in fk_list]
            err_vals = [error.get(fk, 0) for fk in fk_list]

            if len(fk_list) >= 3:
                corr, _ = spearmanr(attr_vals, err_vals)
                correlations.append(corr)

        # Statistics
        if correlations:
            mean_corr = np.mean(correlations)
            std_corr = np.std(correlations)
            ci_low = np.percentile(correlations, 2.5)
            ci_high = np.percentile(correlations, 97.5)
            mean_time = np.mean(times)

            print(f"    ρ = {mean_corr:.3f} ± {std_corr:.3f} [95% CI: {ci_low:.3f}, {ci_high:.3f}]")
            print(f"    Runtime: {mean_time:.1f}s per run")

            all_results.append({
                'sample_size': sample_size,
                'mean_correlation': mean_corr,
                'std_correlation': std_corr,
                'ci_low': ci_low,
                'ci_high': ci_high,
                'mean_runtime': mean_time,
                'n_runs': n_runs
            })

    return all_results


def load_salt_data():
    """Load SALT data from largest available cache."""
    cache_files = [
        ('data_salt_temporal_50000.pkl', 50000),
        ('data_salt_PLANT_20000.pkl', 20000),
        ('data_salt_PLANT_10000.pkl', 10000),
    ]

    for cache_name, expected in cache_files:
        cache_path = CACHE_DIR / cache_name
        if cache_path.exists():
            print(f"Loading {cache_name}...")
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)

            # Handle different cache formats
            if len(data) == 4:
                X, y, feature_cols, col_to_fk = data
            elif len(data) == 6:
                X, y, feature_cols, col_to_fk, _, _ = data
            else:
                continue

            if isinstance(col_to_fk, dict) and len(col_to_fk) > 0:
                print(f"  Loaded {len(X)} samples")
                return X, y, col_to_fk

    return None, None, None


def generate_scale_figure(results, output_path):
    """Generate publication-ready scale-up figure."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Colors
    color_main = '#2E86AB'
    color_fill = '#2E86AB'

    # Extract data
    sizes = [r['sample_size'] for r in results]
    corrs = [r['mean_correlation'] for r in results]
    ci_lows = [r['ci_low'] for r in results]
    ci_highs = [r['ci_high'] for r in results]
    times = [r['mean_runtime'] for r in results]

    # Left: Correlation vs Sample Size
    ax1 = axes[0]
    ax1.plot(sizes, corrs, 'o-', color=color_main, linewidth=2, markersize=8, label='Mean ρ')
    ax1.fill_between(sizes, ci_lows, ci_highs, alpha=0.2, color=color_fill, label='95% CI')
    ax1.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='ρ = 0.9 threshold')
    ax1.set_xlabel('Sample Size', fontsize=11)
    ax1.set_ylabel('Spearman Correlation (ρ)', fontsize=11)
    ax1.set_title('(a) Attribution-Error Correlation', fontsize=12, fontweight='bold')
    ax1.set_xscale('log')
    ax1.set_ylim(0.5, 1.05)
    ax1.legend(loc='lower right', fontsize=9)
    ax1.set_xticks(sizes)
    ax1.set_xticklabels([f'{s//1000}K' for s in sizes])

    # Right: Runtime vs Sample Size
    ax2 = axes[1]
    ax2.plot(sizes, times, 's-', color='#E94E77', linewidth=2, markersize=8)
    ax2.set_xlabel('Sample Size', fontsize=11)
    ax2.set_ylabel('Runtime (seconds)', fontsize=11)
    ax2.set_title('(b) Computational Cost', fontsize=12, fontweight='bold')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xticks(sizes)
    ax2.set_xticklabels([f'{s//1000}K' for s in sizes])

    plt.tight_layout()

    # Save
    plt.savefig(output_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved figure to {output_path}")


def main():
    print("="*70)
    print("EXTENDED SCALE-UP VALIDATION FOR NEURIPS")
    print("Testing FK Attribution at 1K - 50K samples")
    print("="*70)

    # Load data
    X, y, col_to_fk = load_salt_data()

    if X is None:
        print("ERROR: Could not load SALT data")
        return

    # Run experiment
    sample_sizes = [1000, 5000, 10000, 20000, 50000]
    sample_sizes = [s for s in sample_sizes if s <= len(X)]

    results = run_scale_experiment(
        X, y, col_to_fk, "SALT (ERP)",
        sample_sizes=sample_sizes,
        n_runs=5
    )

    # Print summary table
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"{'Sample Size':<15} {'ρ Mean':<10} {'95% CI':<20} {'Runtime':<10}")
    print("-"*55)
    for r in results:
        ci_str = f"[{r['ci_low']:.2f}, {r['ci_high']:.2f}]"
        print(f"{r['sample_size']:<15,} {r['mean_correlation']:<10.3f} {ci_str:<20} {r['mean_runtime']:<10.1f}s")

    # Save results
    output_json = RESULTS_DIR / 'scale_up_extended.json'

    def convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        return obj

    with open(output_json, 'w') as f:
        json.dump([{k: convert(v) for k, v in r.items()} for r in results], f, indent=2)
    print(f"\n[SAVED] Results: {output_json}")

    # Generate figure
    if results:
        fig_path = FIGURES_DIR / 'fig_scale_up'
        generate_scale_figure(results, fig_path)
        print(f"[SAVED] Figure: {fig_path}.pdf/png")

    # Print LaTeX table
    print("\n" + "="*70)
    print("LATEX TABLE")
    print("="*70)
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{Scale-up validation results on SALT dataset.}")
    print(r"\label{tab:scale_up}")
    print(r"\begin{tabular}{rcccc}")
    print(r"\toprule")
    print(r"Sample Size & $\rho$ & 95\% CI & Runtime (s) \\")
    print(r"\midrule")
    for r in results:
        print(f"{r['sample_size']:,} & {r['mean_correlation']:.2f} & [{r['ci_low']:.2f}, {r['ci_high']:.2f}] & {r['mean_runtime']:.1f} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()
