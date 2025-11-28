"""
COVID Timeline Analysis: Monthly FK Uncertainty Attribution
============================================================

Track how FK uncertainty contribution changes month-by-month
to show alignment with COVID onset (Feb 2020).

Research Question: Do FK attribution patterns change around COVID?
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import entropy
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from relbench.datasets import get_dataset
from relbench.tasks import get_task

warnings.filterwarnings('ignore')

# Cache directory
CACHE_DIR = Path("chorok/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Key dates
COVID_ONSET = pd.Timestamp("2020-02-01")
DATA_START = pd.Timestamp("2019-07-01")
DATA_END = pd.Timestamp("2020-07-01")


def get_prediction_entropy(proba):
    """Calculate entropy of prediction probabilities."""
    proba = np.clip(proba, 1e-10, 1 - 1e-10)
    return entropy(proba, axis=1)


def load_full_data(task_name: str, use_cache: bool = True):
    """Load task data without temporal splits. Uses pickle cache for speed."""
    cache_file = CACHE_DIR / f"salt_{task_name}_full.pkl"

    if use_cache and cache_file.exists():
        print(f"  Loading from cache: {cache_file}")
        with open(cache_file, 'rb') as f:
            cached = pickle.load(f)
        return cached['df'], cached['target_col'], cached['entity_table']

    print(f"  Loading from relbench (first time, will cache)...")
    task = get_task("rel-salt", task_name, download=False)
    dataset = get_dataset("rel-salt", download=False)
    db = dataset.get_db(upto_test_timestamp=False)

    entity_table = db.table_dict[task.entity_table]
    df = entity_table.df.copy()

    # Cache for next time
    if use_cache:
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'df': df,
                'target_col': task.target_col,
                'entity_table': entity_table
            }, f)
        print(f"  Cached to: {cache_file}")

    return df, task.target_col, entity_table


def prepare_features_for_month(df, entity_table, target_col, sample_size=3000):
    """Prepare features for a single month's data."""
    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)
    elif len(df) < 100:
        return None, None, None, None  # Not enough data

    # Exclude non-feature columns
    exclude = {'CREATIONTIMESTAMP', target_col, entity_table.pkey_col}
    feature_cols = [c for c in df.columns if c not in exclude]

    # Feature groups (each column is its own FK group)
    feature_groups = {col: [col] for col in feature_cols}

    # Prepare numeric features
    X_df = df[feature_cols].copy()
    for col in X_df.columns:
        if X_df[col].dtype == 'object':
            X_df[col] = X_df[col].astype('category').cat.codes
        X_df[col] = X_df[col].fillna(-999)

    y = df[target_col].copy()
    if y.dtype == 'object':
        y = y.astype('category').cat.codes

    return X_df, y, feature_groups, feature_cols


def measure_monthly_attribution(X, y, feature_groups, feature_cols):
    """Measure FK uncertainty contribution for a single month."""
    if X is None or len(X) < 100:
        return None, None

    # Train full model
    model_full = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    model_full.fit(X, y)
    proba_full = model_full.predict_proba(X)
    entropy_full = get_prediction_entropy(proba_full).mean()

    # Leave-one-out for each FK
    contributions = {}
    for group_name, group_features in feature_groups.items():
        keep_features = [f for f in feature_cols if f not in group_features]
        if len(keep_features) == 0:
            continue

        X_reduced = X[keep_features]
        model_reduced = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
        model_reduced.fit(X_reduced, y)
        proba_reduced = model_reduced.predict_proba(X_reduced)
        entropy_reduced = get_prediction_entropy(proba_reduced).mean()

        contributions[group_name] = entropy_reduced - entropy_full

    return contributions, entropy_full


def run_monthly_analysis(task_name: str, sample_size: int = 3000):
    """Run FK attribution for each month."""
    print(f"\n{'='*60}")
    print(f"COVID Timeline Analysis: {task_name}")
    print(f"{'='*60}")

    # Load data
    df, target_col, entity_table = load_full_data(task_name)
    print(f"Total samples: {len(df):,}")
    print(f"Date range: {df['CREATIONTIMESTAMP'].min()} to {df['CREATIONTIMESTAMP'].max()}")

    # Generate monthly periods
    months = pd.date_range(start=DATA_START, end=DATA_END, freq='MS')

    results = {
        'task': task_name,
        'months': [],
        'monthly_entropy': [],
        'monthly_contributions': [],
        'covid_onset': str(COVID_ONSET)
    }

    print(f"\nAnalyzing {len(months)} months...")

    for month_start in months:
        month_end = month_start + pd.DateOffset(months=1)
        month_str = month_start.strftime('%Y-%m')

        # Filter data for this month
        month_df = df[(df['CREATIONTIMESTAMP'] >= month_start) &
                      (df['CREATIONTIMESTAMP'] < month_end)]

        if len(month_df) < 100:
            print(f"  {month_str}: Skipping (only {len(month_df)} samples)")
            continue

        print(f"  {month_str}: {len(month_df):,} samples", end="")

        # Prepare features
        X, y, feature_groups, feature_cols = prepare_features_for_month(
            month_df, entity_table, target_col, sample_size
        )

        if X is None:
            print(" - skipped (insufficient data)")
            continue

        # Measure attribution
        contributions, entropy_val = measure_monthly_attribution(
            X, y, feature_groups, feature_cols
        )

        if contributions is None:
            print(" - skipped (model error)")
            continue

        # Find top contributor
        top_fk = max(contributions.keys(), key=lambda k: abs(contributions[k]))
        print(f" | entropy={entropy_val:.4f} | top={top_fk} ({contributions[top_fk]:+.4f})")

        results['months'].append(month_str)
        results['monthly_entropy'].append(entropy_val)
        results['monthly_contributions'].append(contributions)

    return results


def plot_timeline(results: Dict, output_dir: Path):
    """Create timeline visualization."""
    months = [pd.Timestamp(m) for m in results['months']]
    entropy_vals = results['monthly_entropy']
    contributions = results['monthly_contributions']

    if len(months) < 2:
        print("Not enough data for visualization")
        return

    # Get all FK names
    all_fks = set()
    for contrib in contributions:
        all_fks.update(contrib.keys())

    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot 1: Overall entropy over time
    ax1 = axes[0]
    ax1.plot(months, entropy_vals, 'b-o', linewidth=2, markersize=6)
    ax1.axvline(x=COVID_ONSET, color='red', linestyle='--', linewidth=2, label='COVID Onset')
    ax1.set_ylabel('Mean Prediction Entropy', fontsize=12)
    ax1.set_title(f"FK Uncertainty Timeline: {results['task']}", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Shade pre/post COVID
    ax1.axvspan(months[0], COVID_ONSET, alpha=0.1, color='green', label='Pre-COVID')
    ax1.axvspan(COVID_ONSET, months[-1], alpha=0.1, color='red', label='COVID Period')

    # Plot 2: Top FK contributions over time
    ax2 = axes[1]

    # Track top 3 FKs by average absolute contribution
    fk_avg_contrib = {}
    for fk in all_fks:
        vals = [abs(c.get(fk, 0)) for c in contributions]
        fk_avg_contrib[fk] = np.mean(vals)

    top_fks = sorted(fk_avg_contrib.keys(), key=lambda k: fk_avg_contrib[k], reverse=True)[:5]

    for fk in top_fks:
        fk_vals = [c.get(fk, 0) for c in contributions]
        ax2.plot(months, fk_vals, '-o', linewidth=1.5, markersize=4, label=fk)

    ax2.axvline(x=COVID_ONSET, color='red', linestyle='--', linewidth=2)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Month', fontsize=12)
    ax2.set_ylabel('FK Contribution (Δ Entropy)', fontsize=12)
    ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Format x-axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)

    plt.tight_layout()

    # Save
    output_file = output_dir / f"covid_timeline_{results['task']}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")
    plt.close()


def analyze_pre_post_covid(results: Dict):
    """Compare pre-COVID vs post-COVID attribution patterns."""
    months = results['months']
    contributions = results['monthly_contributions']

    pre_covid_contribs = []
    post_covid_contribs = []

    for month_str, contrib in zip(months, contributions):
        month = pd.Timestamp(month_str)
        if month < COVID_ONSET:
            pre_covid_contribs.append(contrib)
        else:
            post_covid_contribs.append(contrib)

    if not pre_covid_contribs or not post_covid_contribs:
        print("Not enough data for pre/post COVID comparison")
        return None

    # Average contributions
    all_fks = set()
    for c in contributions:
        all_fks.update(c.keys())

    print("\n" + "="*60)
    print("Pre-COVID vs Post-COVID FK Attribution")
    print("="*60)

    comparison = {}
    print(f"\n{'FK':<25} {'Pre-COVID':>12} {'Post-COVID':>12} {'Change':>12}")
    print("-" * 63)

    for fk in sorted(all_fks):
        pre_vals = [c.get(fk, 0) for c in pre_covid_contribs]
        post_vals = [c.get(fk, 0) for c in post_covid_contribs]

        pre_mean = np.mean(pre_vals)
        post_mean = np.mean(post_vals)
        change = post_mean - pre_mean

        comparison[fk] = {
            'pre_covid': pre_mean,
            'post_covid': post_mean,
            'change': change
        }

        print(f"{fk:<25} {pre_mean:>+12.4f} {post_mean:>+12.4f} {change:>+12.4f}")

    # Find biggest change
    biggest_change_fk = max(comparison.keys(), key=lambda k: abs(comparison[k]['change']))
    print(f"\n>>> Biggest change: {biggest_change_fk} ({comparison[biggest_change_fk]['change']:+.4f})")

    return comparison


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="sales-group")
    parser.add_argument("--sample_size", type=int, default=3000)
    parser.add_argument("--all_tasks", action="store_true")
    args = parser.parse_args()

    output_dir = Path("chorok/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    figures_dir = Path("chorok/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    if args.all_tasks:
        tasks = [
            "sales-group", "sales-office", "sales-payterms", "sales-shipcond",
            "sales-incoterms", "item-plant", "item-shippoint", "item-incoterms"
        ]
    else:
        tasks = [args.task]

    all_results = {}
    all_comparisons = {}

    for task in tasks:
        try:
            results = run_monthly_analysis(task, args.sample_size)
            all_results[task] = results

            # Plot timeline
            plot_timeline(results, figures_dir)

            # Pre/post COVID comparison
            comparison = analyze_pre_post_covid(results)
            if comparison:
                all_comparisons[task] = comparison

        except Exception as e:
            print(f"Error on {task}: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    output_file = output_dir / "covid_timeline_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")

    # Summary across tasks
    if all_comparisons:
        print("\n" + "="*60)
        print("SUMMARY: Biggest FK Changes Across Tasks")
        print("="*60)

        for task, comparison in all_comparisons.items():
            biggest = max(comparison.keys(), key=lambda k: abs(comparison[k]['change']))
            print(f"{task}: {biggest} ({comparison[biggest]['change']:+.4f})")


if __name__ == "__main__":
    main()
