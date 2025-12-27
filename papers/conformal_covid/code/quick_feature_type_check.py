#!/usr/bin/env python3
"""
Quick Feature Type Check - Incremental Phase 1

Goal: Just check if existing 12 tasks have ANY continuous features.
- If all categorical → No hope, stop here
- If some continuous → Proceed to Phase 2

NO SHAP recomputation, NO new experiments.
Just load existing data and check feature types.

Time: ~10 minutes
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add relbench to path
sys.path.insert(0, '/Users/i767700/Github/ai-in-finance')

from relbench.datasets import get_dataset
from relbench.tasks import get_task

# All 12 tasks from our analysis
TASKS = [
    # Severe shift (n=8, rel-salt)
    ('rel-salt', 'sales-group'),
    ('rel-salt', 'sales-payterms'),
    ('rel-salt', 'sales-shipcond'),
    ('rel-salt', 'item-shippoint'),
    ('rel-salt', 'item-incoterms'),
    ('rel-salt', 'item-plant'),
    ('rel-salt', 'sales-incoterms'),
    ('rel-salt', 'sales-office'),

    # Moderate shift (n=4, exploratory)
    ('rel-trial', 'study-outcome'),
    ('rel-trial', 'study-adverse'),
    ('rel-trial', 'site-success'),
    ('rel-f1', 'driver-dnf'),
]


def quick_classify_feature(series: pd.Series) -> str:
    """Quick classification: continuous vs categorical."""
    dtype = series.dtype

    # Object/category → categorical
    if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_categorical_dtype(dtype):
        return 'categorical'

    # Boolean → categorical
    if pd.api.types.is_bool_dtype(dtype):
        return 'categorical'

    # Numeric: check unique count
    if pd.api.types.is_numeric_dtype(dtype):
        n_unique = series.nunique()
        # Many unique → continuous
        return 'continuous' if n_unique > 20 else 'categorical'

    return 'categorical'


def analyze_task(dataset_name: str, task_name: str) -> dict:
    """Analyze one task's feature types."""
    try:
        # Load task
        dataset = get_dataset(dataset_name, download=False)  # Use cached
        task = get_task(dataset_name, task_name, download=False)

        # Get training data
        train_table = task.get_table('train')

        # Identify feature columns (exclude target, time, entity)
        exclude_cols = {task.target_col}

        # Some tasks have time_col, some don't
        if hasattr(task, 'time_col') and task.time_col:
            exclude_cols.add(task.time_col)
        if hasattr(task, 'entity_col') and task.entity_col:
            exclude_cols.add(task.entity_col)
        if hasattr(task, 'entity_table') and task.entity_table:
            exclude_cols.add(task.entity_table)

        # Also check for common column names
        for col in ['timestamp', 'time', 'entity_id', 'id']:
            if col in train_table.df.columns:
                exclude_cols.add(col)

        feature_cols = [col for col in train_table.df.columns if col not in exclude_cols]

        # Classify features
        categorical = []
        continuous = []

        for col in feature_cols:
            feat_type = quick_classify_feature(train_table.df[col])
            if feat_type == 'continuous':
                continuous.append(col)
            else:
                categorical.append(col)

        n_total = len(feature_cols)
        n_cat = len(categorical)
        n_cont = len(continuous)
        pct_cont = (n_cont / n_total * 100) if n_total > 0 else 0

        return {
            'dataset': dataset_name,
            'task': task_name,
            'n_total': n_total,
            'n_categorical': n_cat,
            'n_continuous': n_cont,
            'pct_continuous': pct_cont,
            'continuous_features': continuous[:5],  # First 5 for display
            'dominant': 'continuous' if n_cont > n_cat else 'categorical'
        }

    except Exception as e:
        return {
            'dataset': dataset_name,
            'task': task_name,
            'error': str(e)
        }


def main():
    print("="*80)
    print("QUICK FEATURE TYPE CHECK - Phase 1")
    print("="*80)
    print("\nGoal: Check if existing 12 tasks have ANY continuous features")
    print("Time: ~10 minutes (using cached data)")
    print()

    results = []

    for dataset_name, task_name in TASKS:
        print(f"\n{dataset_name:15} / {task_name:20}", end=" ... ")
        result = analyze_task(dataset_name, task_name)
        results.append(result)

        if 'error' in result:
            print(f"ERROR: {result['error']}")
        else:
            print(f"Features: {result['n_total']:3} = {result['n_categorical']:3} cat + {result['n_continuous']:3} cont ({result['pct_continuous']:5.1f}% cont)")
            if result['n_continuous'] > 0:
                print(f"    Continuous: {result['continuous_features']}")

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    # Summary statistics
    df = pd.DataFrame([r for r in results if 'error' not in r])

    total_tasks = len(df)
    tasks_with_continuous = len(df[df['n_continuous'] > 0])
    continuous_dominated = len(df[df['dominant'] == 'continuous'])

    print(f"\nTotal tasks analyzed: {total_tasks}")
    print(f"Tasks with ANY continuous features: {tasks_with_continuous} ({tasks_with_continuous/total_tasks*100:.1f}%)")
    print(f"Tasks DOMINATED by continuous: {continuous_dominated} ({continuous_dominated/total_tasks*100:.1f}%)")
    print()

    # Show distribution
    print("Distribution by % continuous:")
    for pct_range in [(0, 10), (10, 30), (30, 50), (50, 70), (70, 100)]:
        count = len(df[(df['pct_continuous'] >= pct_range[0]) & (df['pct_continuous'] < pct_range[1])])
        print(f"  {pct_range[0]:3}%-{pct_range[1]:3}%: {count:2} tasks")

    print("\n" + "="*80)
    print("DECISION")
    print("="*80)

    if tasks_with_continuous == 0:
        print("\n❌ NO HOPE - All tasks are 100% categorical")
        print("   Continuous features validation is IMPOSSIBLE with current data")
        print("   Recommendation: STOP HERE, submit paper as-is")

    elif continuous_dominated == 0:
        print("\n⚠️  LIMITED HOPE - Some continuous features but all tasks categorical-dominated")
        print(f"   {tasks_with_continuous} tasks have continuous features but they're minority")
        print("   Recommendation: Can try but unlikely to validate threshold")

    else:
        print(f"\n✓ HOPEFUL - {continuous_dominated} tasks are continuous-dominated")
        print("   Can proceed to Phase 2: Compute SHAP for continuous features")
        print("   Estimated time: 2-3 days")

    # Save results
    output_dir = Path(__file__).parent.parent / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_dir / 'feature_type_analysis.csv', index=False)
    print(f"\n✓ Results saved to {output_dir}/feature_type_analysis.csv")

    return df


if __name__ == '__main__':
    df = main()
