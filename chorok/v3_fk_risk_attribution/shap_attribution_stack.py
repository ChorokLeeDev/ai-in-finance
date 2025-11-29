"""
SHAP-based FK Attribution for Stack Overflow Dataset
=====================================================

Baseline comparison: Use SHAP to attribute model predictions to FK groups.

Stack has posts table with 3 FK relationships:
- OwnerUserId → users (who posted)
- ParentId → posts (parent question for answers)
- AcceptedAnswerId → posts (accepted answer)

Task: post-votes (regression) - predict post popularity
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb

try:
    import shap
except ImportError:
    print("Please install shap: pip install shap")
    raise

from relbench.datasets import get_dataset
from relbench.tasks import get_task

warnings.filterwarnings('ignore')


def load_task_data(task_name: str = "post-votes"):
    """Load Stack task and get train/val split."""
    task = get_task("rel-stack", task_name, download=True)
    dataset = get_dataset("rel-stack", download=True)
    db = dataset.get_db()

    # Get entity table (posts)
    entity_table = db.table_dict[task.entity_table]
    df = entity_table.df.copy()

    # Get task tables for temporal splits
    train_table = task.get_table("train")
    val_table = task.get_table("val")

    # Join with entity table to get features
    entity_col = 'PostId' if 'PostId' in train_table.df.columns else entity_table.pkey_col
    train_df = df.merge(train_table.df, left_on=entity_table.pkey_col, right_on=entity_col)
    val_df = df.merge(val_table.df, left_on=entity_table.pkey_col, right_on=entity_col)

    return train_df, val_df, task.target_col, entity_table, db


def join_fk_features(df, entity_table, db):
    """Join features from FK-related tables to entity table."""
    result_df = df.copy()
    fk_feature_map = {}  # maps fk_name -> list of feature columns from that FK

    for fk_col, target_table_name in entity_table.fkey_col_to_pkey_table.items():
        target_table = db.table_dict[target_table_name]
        target_df = target_table.df.copy()

        # Get numeric/categorical columns from target table (exclude text/blob)
        exclude_cols = {'Body', 'Title', 'Tags', 'OwnerDisplayName', 'ContentLicense',
                        target_table.pkey_col}
        target_cols = [c for c in target_df.columns if c not in exclude_cols]

        if not target_cols:
            continue

        # Prefix columns with FK name to avoid collisions
        rename_map = {c: f"{fk_col}__{c}" for c in target_cols}
        target_df = target_df[[target_table.pkey_col] + target_cols].copy()
        target_df = target_df.rename(columns=rename_map)

        # Join
        result_df = result_df.merge(
            target_df,
            left_on=fk_col,
            right_on=target_table.pkey_col,
            how='left',
            suffixes=('', '_fk')
        )

        # Track which columns came from this FK
        fk_feature_map[fk_col] = list(rename_map.values())

    return result_df, fk_feature_map


def prepare_features(df, entity_table, target_col, db, sample_size=5000):
    """Prepare features and identify FK groups by joining related tables."""
    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)

    # Join features from FK-related tables
    df_enriched, fk_feature_map = join_fk_features(df, entity_table, db)

    # Exclude non-feature columns
    exclude = {'timestamp', target_col, entity_table.pkey_col, 'Body', 'Title', 'Tags',
               'OwnerDisplayName', 'ContentLicense', 'PostId', 'Id'}
    # Also exclude the raw FK ID columns and any pkey columns from joined tables
    for fk_col in entity_table.fkey_col_to_pkey_table.keys():
        exclude.add(fk_col)
        target_table = db.table_dict[entity_table.fkey_col_to_pkey_table[fk_col]]
        exclude.add(target_table.pkey_col)

    # Get all feature columns
    feature_cols = [c for c in df_enriched.columns
                    if c not in exclude and not c.startswith('__')]

    # Remove duplicate columns
    feature_cols = list(dict.fromkeys(feature_cols))

    # Build feature groups from FK feature map
    feature_groups = {}
    assigned_cols = set()

    for fk_name, fk_cols in fk_feature_map.items():
        available_cols = [c for c in fk_cols if c in feature_cols]
        if available_cols:
            feature_groups[fk_name] = available_cols
            assigned_cols.update(available_cols)

    # Non-FK columns (direct attributes of entity table)
    direct_cols = [c for c in feature_cols if c not in assigned_cols]
    if direct_cols:
        feature_groups['_direct_attributes'] = direct_cols

    # Prepare numeric features
    X_df = df_enriched[feature_cols].copy()
    X_df = X_df.loc[:, ~X_df.columns.duplicated()]

    for col in X_df.columns:
        col_data = X_df[col]
        if hasattr(col_data, 'iloc'):
            if col_data.dtype == 'object':
                X_df[col] = col_data.astype('category').cat.codes
            elif str(col_data.dtype).startswith('datetime'):
                X_df[col] = col_data.astype(np.int64) // 10**9
            X_df[col] = pd.to_numeric(X_df[col], errors='coerce').fillna(-999)

    feature_cols = list(X_df.columns)

    y = df_enriched[target_col].copy()
    y = y.fillna(y.median())

    return X_df, y, feature_groups, feature_cols


def compute_shap_attribution(X, y, feature_cols, feature_groups):
    """
    Compute SHAP-based feature attribution for regression.
    """
    # Train model
    model = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
    model.fit(X, y)

    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Mean absolute SHAP per feature
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

    # Create feature-level attribution
    feature_attribution = {}
    for i, col in enumerate(feature_cols):
        feature_attribution[col] = float(mean_abs_shap[i]) if i < len(mean_abs_shap) else 0.0

    # Aggregate by FK group
    group_attribution = {}
    for group_name, group_features in feature_groups.items():
        group_shap = sum(feature_attribution.get(f, 0) for f in group_features)
        group_attribution[group_name] = group_shap

    # Normalize
    total = sum(group_attribution.values())
    group_attribution_norm = {k: v / total if total > 0 else 0
                              for k, v in group_attribution.items()}

    return group_attribution, group_attribution_norm, feature_attribution


def run_shap_comparison(task_name: str = "post-votes", sample_size: int = 5000):
    """Run SHAP attribution comparison for train vs val."""

    print(f"\n{'='*60}")
    print(f"SHAP Attribution (Stack): {task_name}")
    print(f"{'='*60}")

    # Load data
    train_df, val_df, target_col, entity_table, db = load_task_data(task_name)

    print(f"Train samples: {len(train_df):,}")
    print(f"Val samples: {len(val_df):,}")
    print(f"Target: {target_col}")
    print(f"FK relationships: {entity_table.fkey_col_to_pkey_table}")

    # Prepare features
    X_train, y_train, feature_groups, feature_cols = prepare_features(
        train_df, entity_table, target_col, db, sample_size
    )
    X_val, y_val, _, _ = prepare_features(
        val_df, entity_table, target_col, db, sample_size
    )

    print(f"\nFeature groups: {list(feature_groups.keys())}")
    print(f"Total features: {len(feature_cols)}")

    # SHAP attribution on TRAIN
    print(f"\n--- TRAIN ---")
    train_attr, train_attr_norm, train_feat = compute_shap_attribution(
        X_train, y_train, feature_cols, feature_groups
    )

    print("SHAP importance by FK group (normalized):")
    for k, v in sorted(train_attr_norm.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v:.4f}")

    # SHAP attribution on VAL
    print(f"\n--- VAL ---")
    val_attr, val_attr_norm, val_feat = compute_shap_attribution(
        X_val, y_val, feature_cols, feature_groups
    )

    print("SHAP importance by FK group (normalized):")
    for k, v in sorted(val_attr_norm.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v:.4f}")

    # Compare: delta
    print(f"\n--- COMPARISON: Delta (Val - Train) ---")
    deltas = {}
    for group in feature_groups.keys():
        delta = val_attr_norm.get(group, 0) - train_attr_norm.get(group, 0)
        deltas[group] = {
            'train_shap': train_attr_norm.get(group, 0),
            'val_shap': val_attr_norm.get(group, 0),
            'delta': delta
        }
        print(f"  {group}: {train_attr_norm.get(group, 0):.4f} → {val_attr_norm.get(group, 0):.4f} (delta={delta:+.4f})")

    # Find biggest change
    if deltas:
        max_delta_group = max(deltas.keys(), key=lambda k: abs(deltas[k]['delta']))
        print(f"\n>>> Biggest SHAP change: {max_delta_group} (delta={deltas[max_delta_group]['delta']:+.4f})")

    results = {
        'task': task_name,
        'dataset': 'rel-stack',
        'method': 'SHAP',
        'train_attribution': train_attr_norm,
        'val_attribution': val_attr_norm,
        'deltas': deltas,
        'fk_relationships': dict(entity_table.fkey_col_to_pkey_table)
    }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="post-votes")
    parser.add_argument("--sample_size", type=int, default=5000)
    args = parser.parse_args()

    output_dir = Path("chorok/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = run_shap_comparison(args.task, args.sample_size)

    # Save results
    output_file = output_dir / "shap_attribution_stack.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
