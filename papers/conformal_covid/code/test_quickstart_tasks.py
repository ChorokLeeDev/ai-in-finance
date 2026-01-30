#!/usr/bin/env python3
"""
Test Quick Start Tasks - Day 1 Verification

Verifies that all 4 target tasks can be loaded and examines their characteristics.
"""

import sys
sys.path.insert(0, '/Users/i767700/Github/ai-in-finance')

from relbench.datasets import get_dataset
from relbench.tasks import get_task
import pandas as pd

# Quick Start target tasks
TASKS = [
    ('rel-trial', 'study-outcome'),
    ('rel-trial', 'condition-sponsor-run'),
    ('rel-trial', 'site-sponsor-run'),
    ('rel-f1', 'driver-dnf'),
]


def test_task(dataset_name: str, task_name: str):
    """Test loading and examine characteristics of one task."""

    print(f"\n{'='*80}")
    print(f"TESTING: {dataset_name} / {task_name}")
    print(f"{'='*80}")

    try:
        # Load dataset and task
        print(f"Loading dataset: {dataset_name}...")
        dataset = get_dataset(dataset_name, download=True)

        print(f"Loading task: {task_name}...")
        task = get_task(dataset_name, task_name, download=True)

        # Get splits
        train_table = task.get_table('train')
        val_table = task.get_table('val')
        test_table = task.get_table('test')

        # Basic info
        print(f"\n✓ TASK LOADED SUCCESSFULLY")
        print(f"\nBasic Information:")
        print(f"  Task type: {task.task_type}")

        # Handle different task types (entity vs link prediction)
        entity_col = getattr(task, 'entity_col', None)
        if entity_col:
            print(f"  Entity column: {entity_col}")
        else:
            # Link prediction tasks don't have entity_col
            src_entity = getattr(task, 'src_entity_table', None)
            dst_entity = getattr(task, 'dst_entity_table', None)
            if src_entity and dst_entity:
                print(f"  Link prediction: {src_entity} -> {dst_entity}")

        target_col = getattr(task, 'target_col', None)
        if target_col:
            print(f"  Target column: {target_col}")

        time_col = getattr(task, 'time_col', None)
        if time_col:
            print(f"  Time column: {time_col}")

        # Split info
        print(f"\nData Splits:")
        print(f"  Train: {len(train_table.df):,} rows")
        print(f"  Val:   {len(val_table.df):,} rows")
        print(f"  Test:  {len(test_table.df):,} rows")
        print(f"  Total: {len(train_table.df) + len(val_table.df) + len(test_table.df):,} rows")

        # Temporal info
        if hasattr(dataset, 'val_timestamp') and hasattr(dataset, 'test_timestamp'):
            print(f"\nTemporal Splits:")
            print(f"  Val timestamp:  {dataset.val_timestamp}")
            print(f"  Test timestamp: {dataset.test_timestamp}")

        # Target info (if available)
        target_col = getattr(task, 'target_col', None)
        if target_col and target_col in train_table.df.columns:
            print(f"\nTarget Variable ({target_col}):")
            target_train = train_table.df[target_col]

            # Check if target is in val/test (may not be for prediction tasks)
            has_val_target = target_col in val_table.df.columns
            has_test_target = target_col in test_table.df.columns

            if 'regression' in str(task.task_type).lower():
                print(f"  Train - Mean: {target_train.mean():.2f}, Std: {target_train.std():.2f}, Range: [{target_train.min():.2f}, {target_train.max():.2f}]")
                if has_val_target:
                    target_val = val_table.df[target_col]
                    print(f"  Val   - Mean: {target_val.mean():.2f}, Std: {target_val.std():.2f}")
                if has_test_target:
                    target_test = test_table.df[target_col]
                    print(f"  Test  - Mean: {target_test.mean():.2f}, Std: {target_test.std():.2f}")
            else:  # classification
                print(f"  Train - Classes: {target_train.nunique()}, Distribution:")
                print(f"    {target_train.value_counts().to_dict()}")
                if has_val_target:
                    target_val = val_table.df[target_col]
                    print(f"  Val   - Classes: {target_val.nunique()}")
                if has_test_target:
                    target_test = test_table.df[target_col]
                    print(f"  Test  - Classes: {target_test.nunique()}")

            if not (has_val_target and has_test_target):
                print(f"  ⚠️ Target not in val/test tables (prediction task)")
        else:
            print(f"\n⚠️ Target variable structure different (link prediction task)")

        # Feature info
        exclude_cols = []
        if hasattr(task, 'entity_col') and task.entity_col:
            exclude_cols.append(task.entity_col)
        if hasattr(task, 'target_col') and task.target_col:
            exclude_cols.append(task.target_col)
        if hasattr(task, 'time_col') and task.time_col:
            exclude_cols.append(task.time_col)

        feature_cols = [col for col in train_table.df.columns if col not in exclude_cols]

        print(f"\nFeatures:")
        print(f"  Total columns: {len(train_table.df.columns)}")
        print(f"  Feature columns: {len(feature_cols)}")

        if len(feature_cols) == 0:
            print(f"  ⚠️ NO FEATURES IN TABLE - Will need feature engineering from DB")
        else:
            print(f"  ✓ Features available: {feature_cols[:5]}{'...' if len(feature_cols) > 5 else ''}")

            # Sample feature types
            print(f"\nSample Feature Types:")
            for col in feature_cols[:5]:
                dtype = train_table.df[col].dtype
                nunique = train_table.df[col].nunique()
                print(f"    {col:30} {str(dtype):15} unique={nunique:5}")

        # Database info
        print(f"\nDatabase:")
        db = dataset.get_db()
        print(f"  Tables: {list(db.table_dict.keys())}")

        # Entity table
        entity_table = db.table_dict.get(task.entity_table)
        if entity_table is not None:
            print(f"\nEntity Table ({task.entity_table}):")
            print(f"  Rows: {len(entity_table.df):,}")
            print(f"  Columns: {list(entity_table.df.columns)[:10]}")

        return {
            'dataset': dataset_name,
            'task': task_name,
            'task_type': task.task_type,
            'train_size': len(train_table.df),
            'val_size': len(val_table.df),
            'test_size': len(test_table.df),
            'n_features': len(feature_cols),
            'needs_feature_engineering': len(feature_cols) == 0,
            'status': 'SUCCESS'
        }

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {
            'dataset': dataset_name,
            'task': task_name,
            'status': 'FAILED',
            'error': str(e)
        }


def main():
    """Test all Quick Start tasks."""

    print("="*80)
    print("QUICK START TASKS - DAY 1 VERIFICATION")
    print("="*80)
    print(f"\nTesting {len(TASKS)} tasks...")

    results = []
    for dataset_name, task_name in TASKS:
        result = test_task(dataset_name, task_name)
        results.append(result)

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    success_count = sum(1 for r in results if r['status'] == 'SUCCESS')
    print(f"\nSuccessfully loaded: {success_count}/{len(TASKS)}")

    if success_count == len(TASKS):
        print("✓ ALL TASKS VERIFIED - Ready for feature engineering")
    else:
        print("✗ Some tasks failed - Need to investigate")

    # Feature engineering assessment
    needs_eng = sum(1 for r in results if r.get('needs_feature_engineering', False))
    print(f"\nTasks needing feature engineering: {needs_eng}/{success_count}")

    # Create summary table
    print(f"\n{'Dataset':<15} {'Task':<30} {'Type':<15} {'Train':<10} {'Features':<10} {'Status'}")
    print("-"*100)
    for r in results:
        if r['status'] == 'SUCCESS':
            print(f"{r['dataset']:<15} {r['task']:<30} {r['task_type']:<15} "
                  f"{r['train_size']:<10} {r['n_features']:<10} {r['status']}")
        else:
            print(f"{r['dataset']:<15} {r['task']:<30} {'FAILED':<15} {'-':<10} {'-':<10} {r['status']}")

    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)

    if needs_eng > 0:
        print(f"\nDAY 2-4: Feature Engineering Required")
        print(f"  - {needs_eng} tasks need feature engineering from database joins")
        print(f"  - Examine entity tables and relationships")
        print(f"  - Create feature engineering pipelines")
    else:
        print(f"\nDAY 2-4: Features already available!")
        print(f"  - All tasks have features ready")
        print(f"  - Can proceed directly to experiments")

    print(f"\nDAY 5-6: Run Conformal Experiments + SHAP")
    print(f"  - Train LightGBM (50 seeds × {success_count} tasks)")
    print(f"  - Compute conformal prediction coverage")
    print(f"  - Compute SHAP concentration")

    print(f"\nDAY 7: Analyze & Update Paper")
    print(f"  - Combine n=8 + n={success_count} = n={8+success_count}")
    print(f"  - Target: p < 0.02 (from p=0.047)")

    return results


if __name__ == "__main__":
    results = main()
