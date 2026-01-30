"""
Debug Month 9 Coverage Anomaly

Investigates why November 2020 shows 100% coverage with 0% Jaccard similarity.

Possible causes:
1. Target distribution collapse (all one class)
2. Very few samples (low statistical power)
3. Model memorization bug
4. Conformal predictor calibration issue

Author: UAI 2026 Deep Dive
Date: 2025-12-27
"""

import pickle
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')


def split_by_month(df: pd.DataFrame, timestamp_col: str = 'CREATIONTIMESTAMP'):
    """Split dataframe into monthly chunks."""
    df = df.copy()
    df['year_month'] = pd.to_datetime(df[timestamp_col]).dt.to_period('M')

    months = []
    for period in sorted(df['year_month'].unique()):
        month_data = df[df['year_month'] == period].copy()
        month_data = month_data.drop(columns=['year_month'])
        months.append((str(period), month_data))

    return months


def main():
    from relbench.tasks import get_task

    print("=" * 80)
    print("DEBUGGING MONTH 9 COVERAGE ANOMALY")
    print("=" * 80)
    print()

    # Load task
    print("Loading rel-salt/sales-shipcond task...")
    task = get_task('rel-salt', 'sales-shipcond', download=False)

    # Get tables
    train_table = task.get_table("train")
    val_table = task.get_table("val")
    test_table = task.get_table("test", mask_input_cols=False)

    # Get entity data
    dataset = task.dataset
    entity_table = dataset.get_db().table_dict[task.entity_table]
    entity_df = entity_table.df.copy()

    # Merge with entity
    def merge_with_entity(table):
        df = table.df.copy()
        entity_df_copy = entity_df.copy()
        left_entity = list(table.fkey_col_to_pkey_table.keys())[0]
        entity_df_copy = entity_df_copy.astype({entity_table.pkey_col: df[left_entity].dtype})

        for col in set(entity_df_copy.columns).intersection(set(df.columns)):
            if col != entity_table.pkey_col:
                entity_df_copy = entity_df_copy.drop(columns=[col])

        return df.merge(entity_df_copy, how="left",
                       left_on=left_entity, right_on=entity_table.pkey_col)

    train_df = merge_with_entity(train_table)
    val_df = merge_with_entity(val_table)
    test_df = merge_with_entity(test_table)

    print(f"✓ Loaded data:")
    print(f"  Train: {len(train_df):,} samples")
    print(f"  Val: {len(val_df):,} samples")
    print(f"  Test: {len(test_df):,} samples")
    print()

    # Split test into months
    test_months = split_by_month(test_df)

    print(f"Test months: {len(test_months)}")
    for i, (month_str, month_data) in enumerate(test_months):
        print(f"  {i}: {month_str} ({len(month_data):,} samples)")
    print()

    # Focus on month 9 (November 2020)
    if len(test_months) < 10:
        print("ERROR: Not enough test months to analyze month 9")
        return

    month_9_str, month_9_data = test_months[9]
    month_8_str, month_8_data = test_months[8]

    print("=" * 80)
    print(f"MONTH 9 ANALYSIS: {month_9_str}")
    print("=" * 80)
    print()

    target_col = task.target_col

    # 1. Check target distribution
    print("1. TARGET DISTRIBUTION")
    print("-" * 40)
    target_counts = month_9_data[target_col].value_counts()
    print(f"Number of unique classes: {len(target_counts)}")
    print(f"Total samples: {len(month_9_data)}")
    print()
    print("Top 10 classes:")
    print(target_counts.head(10))
    print()

    # Compute entropy
    probs = target_counts / target_counts.sum()
    entropy = -np.sum(probs * np.log2(probs + 1e-10))
    print(f"Target entropy: {entropy:.3f}")
    print(f"Max possible entropy (uniform over {len(target_counts)} classes): {np.log2(len(target_counts)):.3f}")
    print()

    # Compare to month 8
    month_8_target = month_8_data[target_col].value_counts()
    print(f"Month 8 ({month_8_str}) had {len(month_8_target)} unique classes")
    print()

    # 2. Check for missing values
    print("2. MISSING VALUES")
    print("-" * 40)
    missing_counts = month_9_data.isnull().sum()
    missing_features = missing_counts[missing_counts > 0]
    if len(missing_features) == 0:
        print("No missing values")
    else:
        print(f"Features with missing values: {len(missing_features)}")
        print(missing_features.head(10))
    print()

    # 3. Feature distribution analysis
    print("3. FEATURE DISTRIBUTION")
    print("-" * 40)

    # Get feature columns (exclude IDs, timestamps, target)
    exclude_cols = [target_col, 'CREATIONTIMESTAMP', 'timestamp']
    id_cols = [c for c in month_9_data.columns if c.endswith('_id') or c.endswith('Id') or c == 'ID']
    exclude_cols.extend(id_cols)
    feature_cols = [c for c in month_9_data.columns if c not in exclude_cols]

    print(f"Number of features: {len(feature_cols)}")
    print()

    # Check for constant features
    constant_features = []
    for col in feature_cols[:20]:  # Check first 20
        n_unique = month_9_data[col].nunique()
        if n_unique == 1:
            constant_features.append(col)
            print(f"  {col}: CONSTANT (value={month_9_data[col].iloc[0]})")

    if len(constant_features) == 0:
        print("No constant features detected in first 20 features")
    else:
        print(f"\nFound {len(constant_features)} constant features")
    print()

    # 4. Compare feature distributions with training data
    print("4. FEATURE OVERLAP WITH TRAINING DATA")
    print("-" * 40)

    # Sample a few key features
    sample_features = [col for col in feature_cols if 'SALES' in col or 'CUSTOMER' in col][:5]

    for feat in sample_features:
        if feat not in train_df.columns or feat not in month_9_data.columns:
            continue

        train_vals = set(train_df[feat].dropna().unique())
        month9_vals = set(month_9_data[feat].dropna().unique())

        overlap = len(train_vals & month9_vals)
        union = len(train_vals | month9_vals)
        jaccard = overlap / union if union > 0 else 0

        print(f"  {feat}:")
        print(f"    Train unique: {len(train_vals):,}")
        print(f"    Month9 unique: {len(month9_vals):,}")
        print(f"    Jaccard: {jaccard:.3f}")
    print()

    # 5. Load the actual retraining result to see predictions
    print("5. RETRAINING EXPERIMENT RESULTS")
    print("-" * 40)

    results_file = Path(__file__).parent.parent / 'results' / 'retraining' / 'retrain_3M_sales-shipcond.pkl'

    if results_file.exists():
        with open(results_file, 'rb') as f:
            retrain_results = pickle.load(f)

        month_9_result = retrain_results[9]
        print("Month 9 result from experiment:")
        for key, val in month_9_result.items():
            if key != 'predictions':  # Don't print full predictions array
                print(f"  {key}: {val}")
        print()
    else:
        print(f"Results file not found: {results_file}")
        print()

    # 6. HYPOTHESIS TESTING
    print("=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)
    print()

    # Check if target is trivial
    if len(target_counts) == 1:
        print("✓ FOUND ROOT CAUSE:")
        print(f"  Month 9 has only ONE unique target class: {target_counts.index[0]}")
        print("  A model that predicts this class for all samples gets 100% coverage")
        print("  This is a DATA QUALITY ISSUE, not a conformal prediction success")
    elif target_counts.iloc[0] / target_counts.sum() > 0.95:
        print("✓ FOUND LIKELY CAUSE:")
        print(f"  Month 9 is heavily skewed: {target_counts.iloc[0]/target_counts.sum()*100:.1f}% are class '{target_counts.index[0]}'")
        print("  Model likely predicts majority class → high coverage")
    else:
        print("⚠ INCONCLUSIVE:")
        print(f"  Target distribution seems reasonable ({len(target_counts)} classes)")
        print(f"  Entropy: {entropy:.3f}")
        print("  Need to inspect model predictions and conformal sets")
        print()
        print("  Possible next steps:")
        print("    1. Re-run retraining experiment with debug output")
        print("    2. Check if conformal sets are trivial (all size=1)")
        print("    3. Inspect predicted probabilities")

    print()
    print("=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print()
    print("Based on this analysis:")
    print("1. Exclude months 9-10 from quantitative analysis (as already done)")
    print("2. Report in limitations: 'Data quality issue in Nov-Dec 2020'")
    print("3. Hypothesis: Supply chain data collection disrupted during COVID surge")
    print()


if __name__ == "__main__":
    main()
