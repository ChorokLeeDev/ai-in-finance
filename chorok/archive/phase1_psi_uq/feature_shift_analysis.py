"""
Feature-Level Distribution Shift Analysis for SALT Dataset
============================================================

**Research Goal**: Measure feature-level distribution shift using MMD and KS-test,
and correlate with epistemic uncertainty to validate the "complementary signals" hypothesis.

**Hypothesis**:
- PSI measures label shift P(y)
- MMD measures feature shift P(X)
- Epistemic uncertainty correlates more with feature shift (MMD) than label shift (PSI)

**Key Finding to Validate**:
- sales-office: PSI=0.0000 but high uncertainty → expect HIGH MMD
- sales-payterms: PSI=0.0057 but +293% uncertainty → expect HIGH MMD

**Metrics**:
1. Maximum Mean Discrepancy (MMD) - Kernel-based distribution distance
2. Kolmogorov-Smirnov test - Per-feature distribution difference
3. Feature importance shift - Changes in which features matter most

Author: ChorokLeeDev
Created: 2025-11-28
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# SALT task names
SALT_TASKS = [
    "item-plant",
    "item-shippoint",
    "item-incoterms",
    "sales-office",
    "sales-group",
    "sales-payterms",
    "sales-shipcond",
    "sales-incoterms",
]

# COVID-19 temporal markers
VAL_TIMESTAMP = pd.Timestamp("2020-02-01")
TEST_TIMESTAMP = pd.Timestamp("2020-07-01")


def compute_mmd_rbf(X: np.ndarray, Y: np.ndarray, gamma: float = None) -> float:
    """
    Compute Maximum Mean Discrepancy with RBF kernel.

    MMD measures the distance between two distributions in a reproducing kernel Hilbert space.
    Higher values indicate greater distribution shift.

    Args:
        X: Samples from first distribution (n1, d)
        Y: Samples from second distribution (n2, d)
        gamma: RBF kernel bandwidth (if None, uses median heuristic)

    Returns:
        MMD^2 value (non-negative, 0 = identical distributions)
    """
    if len(X) == 0 or len(Y) == 0:
        return np.nan

    # Subsample if too large (for computational efficiency)
    max_samples = 2000
    if len(X) > max_samples:
        idx = np.random.choice(len(X), max_samples, replace=False)
        X = X[idx]
    if len(Y) > max_samples:
        idx = np.random.choice(len(Y), max_samples, replace=False)
        Y = Y[idx]

    # Compute pairwise distances
    XX = cdist(X, X, 'sqeuclidean')
    YY = cdist(Y, Y, 'sqeuclidean')
    XY = cdist(X, Y, 'sqeuclidean')

    # Median heuristic for bandwidth
    if gamma is None:
        all_dists = np.concatenate([XX.flatten(), YY.flatten(), XY.flatten()])
        median_dist = np.median(all_dists[all_dists > 0])
        gamma = 1.0 / (2 * median_dist) if median_dist > 0 else 1.0

    # RBF kernel
    K_XX = np.exp(-gamma * XX)
    K_YY = np.exp(-gamma * YY)
    K_XY = np.exp(-gamma * XY)

    # MMD^2 = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]
    n = len(X)
    m = len(Y)

    # Unbiased estimator (exclude diagonal for XX and YY)
    mmd2 = (np.sum(K_XX) - n) / (n * (n - 1)) if n > 1 else 0
    mmd2 += (np.sum(K_YY) - m) / (m * (m - 1)) if m > 1 else 0
    mmd2 -= 2 * np.mean(K_XY)

    return max(0, mmd2)  # Ensure non-negative


def compute_ks_statistics(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Dict:
    """
    Compute KS test statistics for each feature.

    Returns dict with per-feature KS statistics and p-values.
    """
    results = {}

    for col in X_train.columns:
        if X_train[col].dtype in ['int64', 'float64', 'int32', 'float32']:
            try:
                ks_stat, p_value = stats.ks_2samp(
                    X_train[col].dropna(),
                    X_test[col].dropna()
                )
                results[col] = {
                    'ks_statistic': float(ks_stat),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05
                }
            except Exception as e:
                results[col] = {'error': str(e)}

    return results


def load_task_features(task_name: str, sample_size: int = 10000) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """
    Load features for a SALT task, split by temporal periods.

    Returns:
        train_features, val_features, test_features, target_col
    """
    from relbench.datasets import get_dataset
    from relbench.tasks import get_task

    # Load task
    task = get_task("rel-salt", task_name, download=False)
    target_col = task.target_col
    entity_table_name = task.entity_table

    # Load dataset
    dataset = get_dataset("rel-salt", download=False)
    db = dataset.get_db(upto_test_timestamp=False)

    # Get entity table
    entity_table = db.table_dict[entity_table_name]
    df_full = entity_table.df.copy()

    # Split by temporal periods
    train_df = df_full[df_full['CREATIONTIMESTAMP'] < VAL_TIMESTAMP].copy()
    val_df = df_full[(df_full['CREATIONTIMESTAMP'] >= VAL_TIMESTAMP) &
                     (df_full['CREATIONTIMESTAMP'] < TEST_TIMESTAMP)].copy()
    test_df = df_full[df_full['CREATIONTIMESTAMP'] >= TEST_TIMESTAMP].copy()

    # Sample if too large
    if len(train_df) > sample_size:
        train_df = train_df.sample(sample_size, random_state=42)
    if len(val_df) > sample_size:
        val_df = val_df.sample(sample_size, random_state=42)
    if len(test_df) > sample_size:
        test_df = test_df.sample(sample_size, random_state=42)

    # Get numeric features only (exclude keys, timestamps, target)
    exclude_cols = {
        entity_table.pkey_col,
        'CREATIONTIMESTAMP',
        target_col
    }

    # Add FK columns to exclude
    if hasattr(entity_table, 'fkey_col_to_pkey_table'):
        exclude_cols.update(entity_table.fkey_col_to_pkey_table.keys())

    # Get numeric columns
    feature_cols = []
    for col in df_full.columns:
        if col in exclude_cols:
            continue
        if df_full[col].dtype in ['int64', 'float64', 'int32', 'float32']:
            feature_cols.append(col)
        elif df_full[col].dtype == 'object':
            # Try to encode categorical as numeric
            try:
                train_df[col] = train_df[col].astype('category').cat.codes
                val_df[col] = val_df[col].astype('category').cat.codes
                test_df[col] = test_df[col].astype('category').cat.codes
                feature_cols.append(col)
            except:
                pass

    return train_df[feature_cols], val_df[feature_cols], test_df[feature_cols], target_col


def analyze_task_feature_shift(task_name: str, sample_size: int = 10000) -> Dict:
    """
    Analyze feature-level distribution shift for a single task.
    """
    print(f"\n{'='*60}")
    print(f"Feature Shift Analysis: {task_name}")
    print(f"{'='*60}")

    # Load features
    train_feats, val_feats, test_feats, target_col = load_task_features(task_name, sample_size)

    print(f"Train samples: {len(train_feats)}, features: {len(train_feats.columns)}")
    print(f"Val samples: {len(val_feats)}, features: {len(val_feats.columns)}")
    print(f"Test samples: {len(test_feats)}, features: {len(test_feats.columns)}")

    # Prepare numeric arrays (handle NaN)
    scaler = StandardScaler()

    common_cols = list(set(train_feats.columns) & set(val_feats.columns) & set(test_feats.columns))

    if len(common_cols) == 0:
        return {"error": "No common features found", "task": task_name}

    X_train = train_feats[common_cols].fillna(0).values
    X_val = val_feats[common_cols].fillna(0).values
    X_test = test_feats[common_cols].fillna(0).values

    # Standardize
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    results = {
        "task": task_name,
        "n_features": len(common_cols),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
    }

    # Compute MMD
    print("\nComputing MMD...")
    mmd_train_val = compute_mmd_rbf(X_train_scaled, X_val_scaled)
    mmd_train_test = compute_mmd_rbf(X_train_scaled, X_test_scaled)
    mmd_val_test = compute_mmd_rbf(X_val_scaled, X_test_scaled)

    results["mmd"] = {
        "train_val": float(mmd_train_val),
        "train_test": float(mmd_train_test),
        "val_test": float(mmd_val_test),
    }

    print(f"  MMD (Train→Val): {mmd_train_val:.6f}")
    print(f"  MMD (Train→Test): {mmd_train_test:.6f}")
    print(f"  MMD (Val→Test): {mmd_val_test:.6f}")

    # Compute per-feature KS statistics
    print("\nComputing KS statistics...")
    ks_train_test = compute_ks_statistics(
        train_feats[common_cols],
        test_feats[common_cols]
    )

    # Summary statistics
    ks_stats = [v['ks_statistic'] for v in ks_train_test.values() if 'ks_statistic' in v]
    ks_pvals = [v['p_value'] for v in ks_train_test.values() if 'p_value' in v]
    n_significant = sum(1 for v in ks_train_test.values() if v.get('significant', False))

    results["ks_summary"] = {
        "mean_ks_statistic": float(np.mean(ks_stats)) if ks_stats else None,
        "max_ks_statistic": float(np.max(ks_stats)) if ks_stats else None,
        "n_significant_features": n_significant,
        "pct_significant": float(n_significant / len(ks_train_test)) if ks_train_test else 0,
    }

    # Top shifted features
    sorted_features = sorted(
        [(k, v['ks_statistic']) for k, v in ks_train_test.items() if 'ks_statistic' in v],
        key=lambda x: -x[1]
    )[:5]

    results["top_shifted_features"] = [
        {"feature": f, "ks_statistic": float(ks)} for f, ks in sorted_features
    ]

    print(f"  Mean KS statistic: {results['ks_summary']['mean_ks_statistic']:.4f}")
    print(f"  Significant features: {n_significant}/{len(ks_train_test)} ({results['ks_summary']['pct_significant']*100:.1f}%)")
    print(f"  Top shifted features: {[f[0] for f in sorted_features[:3]]}")

    return results


def run_all_tasks(sample_size: int = 10000) -> Dict:
    """Run feature shift analysis for all SALT tasks."""
    all_results = {}

    for task_name in SALT_TASKS:
        try:
            results = analyze_task_feature_shift(task_name, sample_size)
            all_results[task_name] = results
        except Exception as e:
            print(f"Error analyzing {task_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results[task_name] = {"error": str(e)}

    return all_results


def plot_mmd_comparison(all_results: Dict, output_path: Path):
    """Create bar plot comparing MMD across tasks."""
    tasks = []
    mmd_values = []

    for task_name, results in all_results.items():
        if 'mmd' in results:
            tasks.append(task_name)
            mmd_values.append(results['mmd']['train_test'])

    if not tasks:
        print("No MMD data to plot")
        return

    # Sort by MMD value
    sorted_idx = np.argsort(mmd_values)[::-1]
    tasks = [tasks[i] for i in sorted_idx]
    mmd_values = [mmd_values[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(tasks)))
    bars = ax.bar(tasks, mmd_values, color=colors, edgecolor='black', linewidth=1.5)

    ax.set_xlabel('Task', fontsize=12, fontweight='bold')
    ax.set_ylabel('MMD (Train → Test)', fontsize=12, fontweight='bold')
    ax.set_title('Feature-Level Distribution Shift (MMD) Across SALT Tasks', fontsize=14, fontweight='bold')

    plt.xticks(rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, val in zip(bars, mmd_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved MMD comparison plot: {output_path}")


def print_summary_table(all_results: Dict):
    """Print summary table of feature shift metrics."""
    print("\n" + "="*80)
    print("FEATURE-LEVEL SHIFT SUMMARY (MMD and KS)")
    print("="*80)
    print(f"{'Task':<20} {'MMD':>10} {'Mean KS':>10} {'% Sig':>10} {'Top Feature':<20}")
    print("-"*80)

    for task_name in SALT_TASKS:
        results = all_results.get(task_name, {})
        if 'error' in results:
            print(f"{task_name:<20} ERROR: {results['error']}")
            continue

        mmd = results.get('mmd', {}).get('train_test', 0)
        ks_summary = results.get('ks_summary', {})
        mean_ks = ks_summary.get('mean_ks_statistic', 0)
        pct_sig = ks_summary.get('pct_significant', 0) * 100

        top_feats = results.get('top_shifted_features', [])
        top_feat = top_feats[0]['feature'] if top_feats else 'N/A'

        print(f"{task_name:<20} {mmd:>10.6f} {mean_ks:>10.4f} {pct_sig:>9.1f}% {top_feat:<20}")

    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Feature-level shift analysis for SALT dataset")
    parser.add_argument("--task", type=str, default=None,
                       help="Single task to analyze (default: all tasks)")
    parser.add_argument("--sample_size", type=int, default=10000,
                       help="Max samples per split")
    parser.add_argument("--output_dir", type=str, default="chorok/results",
                       help="Output directory for results")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "#"*60)
    print("# FEATURE-LEVEL DISTRIBUTION SHIFT ANALYSIS")
    print("# Using MMD and KS-test")
    print("#"*60)

    if args.task:
        # Single task
        results = {args.task: analyze_task_feature_shift(args.task, args.sample_size)}
    else:
        # All tasks
        results = run_all_tasks(args.sample_size)

    # Print summary
    print_summary_table(results)

    # Save results
    output_file = output_dir / "feature_shift_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    # Create plot
    figures_dir = Path("chorok/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_mmd_comparison(results, figures_dir / "feature_shift_mmd.pdf")

    print("\n" + "#"*60)
    print("# ANALYSIS COMPLETE")
    print("#"*60)


if __name__ == "__main__":
    main()
