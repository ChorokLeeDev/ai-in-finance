"""
Temporal Exploration of SALT Dataset
=====================================

This script explores temporal characteristics of the SALT dataset to understand
distribution changes before, during, and after COVID-19 pandemic.

Key analyses:
1. Data volume over time (monthly/quarterly)
2. Class distribution evolution
3. Pre-COVID vs COVID comparison
4. Visualizations for all 8 SALT tasks

Author: ChorokLeeDev
Created: 2025-01-18
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from relbench.datasets import get_dataset
from relbench.tasks import get_task

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
VAL_TIMESTAMP = pd.Timestamp("2020-02-01")  # COVID onset
TEST_TIMESTAMP = pd.Timestamp("2020-07-01")  # COVID impact


def load_salt_data(task_name: str, download: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train/val/test splits for a SALT task."""
    print(f"Loading task: {task_name}")
    task = get_task("rel-salt", task_name, download=download)

    train_table = task.get_table("train")
    val_table = task.get_table("val")
    test_table = task.get_table("test")

    return train_table.df, val_table.df, test_table.df


def add_temporal_features(df: pd.DataFrame, time_col: str = "CREATIONTIMESTAMP") -> pd.DataFrame:
    """Add temporal features for analysis."""
    df = df.copy()

    if time_col not in df.columns:
        print(f"Warning: {time_col} not found in columns: {df.columns.tolist()}")
        return df

    df[time_col] = pd.to_datetime(df[time_col])
    df['year'] = df[time_col].dt.year
    df['month'] = df[time_col].dt.month
    df['quarter'] = df[time_col].dt.quarter
    df['year_month'] = df[time_col].dt.to_period('M')
    df['year_quarter'] = df[time_col].dt.to_period('Q')

    # COVID period markers
    df['covid_period'] = 'pre_covid'
    df.loc[df[time_col] >= VAL_TIMESTAMP, 'covid_period'] = 'covid_onset'
    df.loc[df[time_col] >= TEST_TIMESTAMP, 'covid_period'] = 'covid_impact'

    return df


def analyze_data_volume(df: pd.DataFrame, time_col: str = "CREATIONTIMESTAMP") -> Dict:
    """Analyze data volume over time."""
    if time_col not in df.columns:
        return {"error": f"{time_col} not found"}

    stats = {
        "total_samples": len(df),
        "date_range": {
            "start": str(df[time_col].min()),
            "end": str(df[time_col].max()),
        },
        "monthly_avg": df.groupby('year_month').size().mean(),
        "monthly_std": df.groupby('year_month').size().std(),
        "covid_periods": {
            "pre_covid": len(df[df['covid_period'] == 'pre_covid']),
            "covid_onset": len(df[df['covid_period'] == 'covid_onset']),
            "covid_impact": len(df[df['covid_period'] == 'covid_impact']),
        }
    }

    return stats


def analyze_class_distribution(df: pd.DataFrame, target_col: str) -> Dict:
    """Analyze class distribution and changes over time."""
    if target_col not in df.columns:
        return {"error": f"{target_col} not found"}

    # Overall distribution
    overall_dist = df[target_col].value_counts(normalize=True).to_dict()

    # Distribution by COVID period
    period_dist = {}
    for period in ['pre_covid', 'covid_onset', 'covid_impact']:
        period_df = df[df['covid_period'] == period]
        if len(period_df) > 0:
            period_dist[period] = period_df[target_col].value_counts(normalize=True).to_dict()

    stats = {
        "num_classes": df[target_col].nunique(),
        "overall_distribution": {k: float(v) for k, v in overall_dist.items()},
        "period_distributions": {
            k: {cls: float(v) for cls, v in dist.items()}
            for k, dist in period_dist.items()
        },
        "class_imbalance": float(df[target_col].value_counts(normalize=True).max()),
    }

    return stats


def plot_temporal_volume(df: pd.DataFrame, task_name: str, output_dir: Path, time_col: str = "CREATIONTIMESTAMP"):
    """Plot data volume over time."""
    if time_col not in df.columns:
        print(f"Skipping volume plot for {task_name}: {time_col} not found")
        return

    fig, ax = plt.subplots(figsize=(14, 6))

    # Monthly volume
    monthly_counts = df.groupby('year_month').size()
    monthly_counts.index = monthly_counts.index.to_timestamp()

    ax.plot(monthly_counts.index, monthly_counts.values, marker='o', linewidth=2, markersize=4)

    # Add COVID markers
    ax.axvline(VAL_TIMESTAMP, color='orange', linestyle='--', linewidth=2, label='COVID Onset (Val Split)')
    ax.axvline(TEST_TIMESTAMP, color='red', linestyle='--', linewidth=2, label='COVID Impact (Test Split)')

    # Shade COVID periods
    ax.axvspan(VAL_TIMESTAMP, TEST_TIMESTAMP, alpha=0.2, color='orange', label='COVID Onset Period')
    ax.axvspan(TEST_TIMESTAMP, df[time_col].max(), alpha=0.2, color='red', label='COVID Impact Period')

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title(f'Temporal Data Volume: {task_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / f"{task_name}_temporal_volume.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")


def plot_class_distribution_over_time(df: pd.DataFrame, task_name: str, target_col: str, output_dir: Path):
    """Plot class distribution evolution over time."""
    if target_col not in df.columns:
        print(f"Skipping class distribution plot for {task_name}: {target_col} not found")
        return

    # Get top 10 classes by frequency
    top_classes = df[target_col].value_counts().head(10).index.tolist()

    # Monthly class distribution
    monthly_dist = df.groupby(['year_month', target_col]).size().unstack(fill_value=0)
    monthly_dist = monthly_dist[top_classes]
    monthly_dist_pct = monthly_dist.div(monthly_dist.sum(axis=1), axis=0) * 100
    monthly_dist_pct.index = monthly_dist_pct.index.to_timestamp()

    fig, ax = plt.subplots(figsize=(14, 8))

    # Stacked area plot
    ax.stackplot(monthly_dist_pct.index,
                 [monthly_dist_pct[col].values for col in monthly_dist_pct.columns],
                 labels=monthly_dist_pct.columns,
                 alpha=0.8)

    # Add COVID markers
    ax.axvline(VAL_TIMESTAMP, color='black', linestyle='--', linewidth=2, label='COVID Onset')
    ax.axvline(TEST_TIMESTAMP, color='black', linestyle=':', linewidth=2, label='COVID Impact')

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Class Distribution (%)', fontsize=12)
    ax.set_title(f'Class Distribution Evolution: {task_name} (Top 10 Classes)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = output_dir / f"{task_name}_class_evolution.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")


def plot_covid_period_comparison(df: pd.DataFrame, task_name: str, target_col: str, output_dir: Path):
    """Compare class distributions across COVID periods."""
    if target_col not in df.columns:
        print(f"Skipping COVID comparison for {task_name}: {target_col} not found")
        return

    # Get top 15 classes
    top_classes = df[target_col].value_counts().head(15).index.tolist()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    periods = ['pre_covid', 'covid_onset', 'covid_impact']
    period_names = ['Pre-COVID\n(2018-2020.02)', 'COVID Onset\n(2020.02-07)', 'COVID Impact\n(2020.07+)']
    colors = ['#2ecc71', '#f39c12', '#e74c3c']

    for ax, period, period_name, color in zip(axes, periods, period_names, colors):
        period_df = df[df['covid_period'] == period]
        if len(period_df) == 0:
            continue

        class_counts = period_df[target_col].value_counts()
        class_counts = class_counts[class_counts.index.isin(top_classes)]
        class_pct = (class_counts / len(period_df) * 100).sort_values(ascending=True)

        ax.barh(range(len(class_pct)), class_pct.values, color=color, alpha=0.7)
        ax.set_yticks(range(len(class_pct)))
        ax.set_yticklabels(class_pct.index, fontsize=9)
        ax.set_xlabel('Percentage (%)', fontsize=11)
        ax.set_title(f'{period_name}\n(n={len(period_df):,})', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

    fig.suptitle(f'Class Distribution by COVID Period: {task_name}', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_path = output_dir / f"{task_name}_covid_comparison.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")


def explore_task(task_name: str, output_dir: Path, download: bool = False) -> Dict:
    """Complete temporal exploration for a single task."""
    print(f"\n{'='*60}")
    print(f"Exploring: {task_name}")
    print(f"{'='*60}")

    # Load data
    train_df, val_df, test_df = load_salt_data(task_name, download=download)

    # Combine for temporal analysis
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'
    combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # Determine target column
    task_config = {
        "item-plant": "PLANT",
        "item-shippoint": "SHIPPINGPOINT",
        "item-incoterms": "ITEMINCOTERMSCLASSIFICATION",
        "sales-office": "SALESOFFICE",
        "sales-group": "SALESGROUP",
        "sales-payterms": "CUSTOMERPAYMENTTERMS",
        "sales-shipcond": "SHIPPINGCONDITION",
        "sales-incoterms": "HEADERINCOTERMSCLASSIFICATION",
    }
    target_col = task_config.get(task_name)

    if target_col not in combined_df.columns:
        print(f"Warning: Target column {target_col} not found. Available: {combined_df.columns.tolist()}")
        return {"error": f"Target column {target_col} not found"}

    # Add temporal features
    combined_df = add_temporal_features(combined_df)

    # Analyze
    volume_stats = analyze_data_volume(combined_df)
    class_stats = analyze_class_distribution(combined_df, target_col)

    # Visualize
    plot_temporal_volume(combined_df, task_name, output_dir)
    plot_class_distribution_over_time(combined_df, task_name, target_col, output_dir)
    plot_covid_period_comparison(combined_df, task_name, target_col, output_dir)

    results = {
        "task_name": task_name,
        "target_column": target_col,
        "volume_stats": volume_stats,
        "class_stats": class_stats,
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Temporal exploration of SALT dataset")
    parser.add_argument("--tasks", nargs="+", default=SALT_TASKS,
                       help="List of tasks to explore (default: all 8 tasks)")
    parser.add_argument("--output_dir", type=str, default="chorok/figures/temporal_stats",
                       help="Output directory for figures")
    parser.add_argument("--download", action="store_true",
                       help="Download dataset if not cached")
    parser.add_argument("--save_json", type=str, default="chorok/results/temporal_exploration.json",
                       help="Path to save JSON results")

    args = parser.parse_args()

    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_dir = Path(args.save_json).parent
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"# SALT Dataset Temporal Exploration")
    print(f"{'#'*60}")
    print(f"Tasks to explore: {args.tasks}")
    print(f"Output directory: {output_dir}")
    print(f"Results JSON: {args.save_json}")

    # Explore each task
    all_results = {}
    for task_name in tqdm(args.tasks, desc="Exploring tasks"):
        try:
            results = explore_task(task_name, output_dir, download=args.download)
            all_results[task_name] = results
        except Exception as e:
            print(f"Error exploring {task_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results[task_name] = {"error": str(e)}

    # Save results
    with open(args.save_json, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"Exploration complete! Results saved to: {args.save_json}")
    print(f"Figures saved to: {output_dir}")
    print(f"{'='*60}")

    # Summary statistics
    print("\n## Summary Statistics ##")
    for task_name, results in all_results.items():
        if "error" in results:
            print(f"{task_name}: ERROR - {results['error']}")
            continue

        volume = results['volume_stats']
        classes = results['class_stats']

        print(f"\n{task_name}:")
        print(f"  Total samples: {volume['total_samples']:,}")
        print(f"  Num classes: {classes['num_classes']}")
        print(f"  COVID period split:")
        print(f"    Pre-COVID: {volume['covid_periods']['pre_covid']:,} ({volume['covid_periods']['pre_covid']/volume['total_samples']*100:.1f}%)")
        print(f"    COVID Onset: {volume['covid_periods']['covid_onset']:,} ({volume['covid_periods']['covid_onset']/volume['total_samples']*100:.1f}%)")
        print(f"    COVID Impact: {volume['covid_periods']['covid_impact']:,} ({volume['covid_periods']['covid_impact']/volume['total_samples']*100:.1f}%)")


if __name__ == "__main__":
    main()
