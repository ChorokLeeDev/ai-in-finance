"""
Investigation: item-shippoint Uncertainty Anomaly
==================================================

The item-shippoint task reportedly showed -82% uncertainty change (DECREASE)
despite distribution shift. This is anomalous and needs investigation.

Possible explanations:
1. Autocomplete artifact: Zero-dominated predictions
2. Adaptive learning: Models successfully adapted
3. Data quality: Post-COVID data more structured
4. Task characteristic: Simpler prediction post-COVID

Author: ChorokLeeDev
Created: 2025-11-28
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from relbench.datasets import get_dataset
from relbench.tasks import get_task

# COVID-19 temporal markers
VAL_TIMESTAMP = pd.Timestamp("2020-02-01")
TEST_TIMESTAMP = pd.Timestamp("2020-07-01")


def load_task_data(task_name: str = "item-shippoint"):
    """Load task data for analysis."""
    task = get_task("rel-salt", task_name, download=False)
    dataset = get_dataset("rel-salt", download=False)
    db = dataset.get_db(upto_test_timestamp=False)

    entity_table = db.table_dict[task.entity_table]
    df = entity_table.df.copy()

    target_col = task.target_col

    # Split by temporal periods
    train_df = df[df['CREATIONTIMESTAMP'] < VAL_TIMESTAMP]
    val_df = df[(df['CREATIONTIMESTAMP'] >= VAL_TIMESTAMP) &
                (df['CREATIONTIMESTAMP'] < TEST_TIMESTAMP)]
    test_df = df[df['CREATIONTIMESTAMP'] >= TEST_TIMESTAMP]

    return train_df, val_df, test_df, target_col, df


def analyze_target_distribution(train_df, val_df, test_df, target_col):
    """Analyze target distribution across splits."""
    print("\n" + "="*60)
    print("TARGET DISTRIBUTION ANALYSIS")
    print("="*60)

    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        print(f"\n{name} ({len(df):,} samples):")
        value_counts = df[target_col].value_counts()
        top_5 = value_counts.head(5)
        total = len(df)

        print(f"  Unique values: {df[target_col].nunique()}")
        print(f"  Top 5 classes:")
        for val, count in top_5.items():
            pct = count / total * 100
            print(f"    {val}: {count:,} ({pct:.1f}%)")

        # Check concentration
        top1_pct = value_counts.iloc[0] / total * 100
        top3_pct = value_counts.head(3).sum() / total * 100
        print(f"  Top-1 concentration: {top1_pct:.1f}%")
        print(f"  Top-3 concentration: {top3_pct:.1f}%")


def analyze_temporal_patterns(df, target_col):
    """Analyze how target distribution changes over time."""
    print("\n" + "="*60)
    print("TEMPORAL PATTERN ANALYSIS")
    print("="*60)

    # Add month column
    df = df.copy()
    df['month'] = df['CREATIONTIMESTAMP'].dt.to_period('M')

    # Calculate monthly class concentration
    monthly_stats = []
    for month, group in df.groupby('month'):
        value_counts = group[target_col].value_counts()
        total = len(group)
        top1_pct = value_counts.iloc[0] / total * 100 if len(value_counts) > 0 else 0
        n_classes = group[target_col].nunique()
        monthly_stats.append({
            'month': str(month),
            'samples': total,
            'n_classes': n_classes,
            'top1_pct': top1_pct
        })

    monthly_df = pd.DataFrame(monthly_stats)

    # Print key periods
    print("\nClass concentration over time:")
    print(f"{'Month':<10} {'Samples':>10} {'Classes':>10} {'Top-1 %':>10}")
    print("-"*45)

    for _, row in monthly_df.iterrows():
        month = row['month']
        # Highlight COVID periods
        marker = ""
        if "2020-02" <= month <= "2020-06":
            marker = " ← COVID onset"
        elif month >= "2020-07":
            marker = " ← Post-lockdown"

        print(f"{month:<10} {row['samples']:>10,} {row['n_classes']:>10} {row['top1_pct']:>9.1f}%{marker}")

    return monthly_df


def analyze_feature_quality(train_df, test_df, target_col):
    """Check if feature quality improved post-COVID."""
    print("\n" + "="*60)
    print("FEATURE QUALITY ANALYSIS")
    print("="*60)

    # Exclude target and timestamp
    feature_cols = [c for c in train_df.columns
                   if c not in [target_col, 'CREATIONTIMESTAMP', 'SALESDOCUMENTITEM']]

    print(f"\nFeatures analyzed: {feature_cols}")

    for col in feature_cols:
        train_null = train_df[col].isna().mean() * 100
        test_null = test_df[col].isna().mean() * 100

        train_unique = train_df[col].nunique()
        test_unique = test_df[col].nunique()

        print(f"\n{col}:")
        print(f"  Missing: Train={train_null:.1f}%, Test={test_null:.1f}%")
        print(f"  Unique: Train={train_unique}, Test={test_unique}")


def plot_temporal_analysis(monthly_df, output_path):
    """Create temporal analysis plot."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    months = monthly_df['month'].values
    x = range(len(months))

    # Plot 1: Sample count
    axes[0].bar(x, monthly_df['samples'].values, color='steelblue', alpha=0.7)
    axes[0].axvline(months.tolist().index('2020-02'), color='red', linestyle='--',
                   label='COVID onset (Feb 2020)')
    axes[0].axvline(months.tolist().index('2020-07'), color='orange', linestyle='--',
                   label='Post-lockdown (Jul 2020)')
    axes[0].set_ylabel('Sample Count', fontsize=12, fontweight='bold')
    axes[0].set_title('item-shippoint: Monthly Volume', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].set_xticks(x[::3])
    axes[0].set_xticklabels([months[i] for i in x[::3]], rotation=45)

    # Plot 2: Class concentration
    axes[1].plot(x, monthly_df['top1_pct'].values, 'o-', color='darkgreen',
                linewidth=2, markersize=6)
    axes[1].axvline(months.tolist().index('2020-02'), color='red', linestyle='--')
    axes[1].axvline(months.tolist().index('2020-07'), color='orange', linestyle='--')
    axes[1].axhline(monthly_df['top1_pct'].mean(), color='gray', linestyle=':',
                   label=f'Mean: {monthly_df["top1_pct"].mean():.1f}%')
    axes[1].set_ylabel('Top-1 Class %', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Month', fontsize=12)
    axes[1].set_title('Class Concentration Over Time', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].set_xticks(x[::3])
    axes[1].set_xticklabels([months[i] for i in x[::3]], rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved temporal analysis plot: {output_path}")


def main():
    print("\n" + "#"*60)
    print("# INVESTIGATION: item-shippoint Anomaly")
    print("# Why did uncertainty DECREASE (-82%)?")
    print("#"*60)

    # Load data
    print("\nLoading data...")
    train_df, val_df, test_df, target_col, full_df = load_task_data()

    print(f"Target column: {target_col}")
    print(f"Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")

    # Analyze target distribution
    analyze_target_distribution(train_df, val_df, test_df, target_col)

    # Analyze temporal patterns
    monthly_df = analyze_temporal_patterns(full_df, target_col)

    # Analyze feature quality
    analyze_feature_quality(train_df, test_df, target_col)

    # Create plot
    output_dir = Path("chorok/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_temporal_analysis(monthly_df, output_dir / "item_shippoint_temporal.pdf")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: Possible Explanations for Uncertainty Decrease")
    print("="*60)

    # Calculate key metrics
    train_top1 = train_df[target_col].value_counts().iloc[0] / len(train_df) * 100
    test_top1 = test_df[target_col].value_counts().iloc[0] / len(test_df) * 100

    print(f"\n1. CLASS CONCENTRATION:")
    print(f"   Train Top-1: {train_top1:.1f}%")
    print(f"   Test Top-1: {test_top1:.1f}%")
    if test_top1 > train_top1:
        print(f"   → Test is MORE concentrated → Models become MORE confident")
    else:
        print(f"   → Test is LESS concentrated → Not the explanation")

    print(f"\n2. CLASS COUNT:")
    print(f"   Train unique: {train_df[target_col].nunique()}")
    print(f"   Test unique: {test_df[target_col].nunique()}")

    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    print("""
If test data is MORE concentrated (fewer classes dominate),
models naturally become more confident (lower entropy/uncertainty).

This is NOT necessarily "adaptive learning" but rather a shift to
an easier prediction task (less diverse target distribution).

For publication: Frame as "Distribution shift that simplifies the task"
rather than "Models successfully adapted."
    """)


if __name__ == "__main__":
    main()
