"""
[ARCHIVED - 2025-11-29]
========================
원래 목적: Stack Overflow 데이터셋에서 LOO FK attribution 구현
폐기 이유: Stack 데이터셋 실험 중단 - rel-salt에 집중하기로 결정
           Stack의 FK 구조 (OwnerUserId, ParentId, AcceptedAnswerId)가 단순함
           rel-salt의 SAP 도메인 (7개 FK 그룹)이 actionability 검증에 더 적합
========================

FK Uncertainty Attribution for Stack Overflow Dataset
======================================================

Adapts the Leave-One-Out FK attribution method for rel-stack dataset.

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
from scipy.stats import entropy, spearmanr

from relbench.datasets import get_dataset
from relbench.tasks import get_task

warnings.filterwarnings('ignore')


def get_prediction_variance(predictions_list):
    """Calculate variance across ensemble predictions (epistemic uncertainty for regression)."""
    stacked = np.stack(predictions_list, axis=0)
    return np.var(stacked, axis=0)


def get_prediction_entropy(proba):
    """Calculate entropy of prediction probabilities (for classification)."""
    proba = np.clip(proba, 1e-10, 1 - 1e-10)
    return entropy(proba, axis=1)


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
    # Note: train_table uses PostId, entity table uses Id
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

    # Remove duplicate columns (can happen from multiple joins)
    feature_cols = [c for c in feature_cols if c in df_enriched.columns]
    feature_cols = list(dict.fromkeys(feature_cols))  # Remove duplicates preserving order

    # Prepare numeric features
    X_df = df_enriched[feature_cols].copy()
    # Handle duplicate columns in dataframe
    X_df = X_df.loc[:, ~X_df.columns.duplicated()]

    for col in X_df.columns:
        col_data = X_df[col]
        if hasattr(col_data, 'iloc'):  # It's a Series
            if col_data.dtype == 'object':
                X_df[col] = col_data.astype('category').cat.codes
            elif str(col_data.dtype).startswith('datetime'):
                X_df[col] = col_data.astype(np.int64) // 10**9
            X_df[col] = pd.to_numeric(X_df[col], errors='coerce').fillna(-999)

    feature_cols = list(X_df.columns)

    y = df_enriched[target_col].copy()
    y = y.fillna(y.median())

    return X_df, y, feature_groups, feature_cols


def measure_fk_uncertainty_contribution_regression(X, y, feature_groups, feature_cols, n_ensemble=5):
    """
    Measure each FK group's contribution to uncertainty via leave-one-out.
    For regression: uses ensemble variance as uncertainty measure.

    Returns dict: {fk_name: uncertainty_contribution}
    """
    # Train ensemble of models for full feature set
    # Use bagging and feature sampling to induce variance
    print(f"  Training {n_ensemble} ensemble models (full features)...")
    predictions_full = []
    for seed in range(n_ensemble):
        model = lgb.LGBMRegressor(
            n_estimators=100,
            random_state=seed,
            verbose=-1,
            bagging_fraction=0.8,
            bagging_freq=1,
            feature_fraction=0.8,
        )
        model.fit(X, y)
        predictions_full.append(model.predict(X))

    variance_full = get_prediction_variance(predictions_full).mean()
    print(f"  Full model variance (uncertainty): {variance_full:.4f}")

    # Leave-one-out for each FK group
    contributions = {}

    for group_name, group_features in feature_groups.items():
        # Features to keep (remove this group)
        keep_features = [f for f in feature_cols if f not in group_features]

        if len(keep_features) == 0:
            continue

        X_reduced = X[keep_features]

        # Train ensemble without this group
        predictions_reduced = []
        for seed in range(n_ensemble):
            model = lgb.LGBMRegressor(
                n_estimators=100,
                random_state=seed,
                verbose=-1,
                bagging_fraction=0.8,
                bagging_freq=1,
                feature_fraction=0.8,
            )
            model.fit(X_reduced, y)
            predictions_reduced.append(model.predict(X_reduced))

        variance_reduced = get_prediction_variance(predictions_reduced).mean()

        # Contribution = how much variance increases when this group is removed
        contribution = variance_reduced - variance_full
        contributions[group_name] = contribution

        print(f"  Without {group_name}: variance={variance_reduced:.4f}, delta={contribution:+.4f}")

    # Normalize contributions
    total = sum(abs(v) for v in contributions.values())
    if total > 0:
        contributions_norm = {k: abs(v) / total for k, v in contributions.items()}
    else:
        contributions_norm = contributions

    return contributions, contributions_norm, variance_full


def run_attribution_comparison(task_name: str = "post-votes", sample_size: int = 5000):
    """Run FK uncertainty attribution comparison for train vs val."""

    print(f"\n{'='*60}")
    print(f"FK Uncertainty Attribution (Stack): {task_name}")
    print(f"{'='*60}")

    # Load data
    train_df, val_df, target_col, entity_table, db = load_task_data(task_name)

    print(f"Train samples: {len(train_df):,}")
    print(f"Val samples: {len(val_df):,}")
    print(f"Target: {target_col}")
    print(f"Entity table: {entity_table.pkey_col}")
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

    # Measure uncertainty contribution on TRAIN
    print(f"\n--- TRAIN ---")
    train_contrib, train_contrib_norm, train_variance = measure_fk_uncertainty_contribution_regression(
        X_train, y_train, feature_groups, feature_cols
    )

    # Measure uncertainty contribution on VAL
    print(f"\n--- VAL ---")
    val_contrib, val_contrib_norm, val_variance = measure_fk_uncertainty_contribution_regression(
        X_val, y_val, feature_groups, feature_cols
    )

    # Compare: delta
    print(f"\n--- COMPARISON: Delta (Val - Train) ---")
    deltas = {}
    for group in train_contrib.keys():
        if group in val_contrib:
            delta = val_contrib[group] - train_contrib[group]
            delta_pct = (val_contrib_norm.get(group, 0) - train_contrib_norm.get(group, 0)) * 100
            deltas[group] = {
                'train': train_contrib[group],
                'val': val_contrib[group],
                'delta': delta,
                'train_norm': train_contrib_norm.get(group, 0),
                'val_norm': val_contrib_norm.get(group, 0),
                'delta_pct': delta_pct
            }
            print(f"  {group}: Train={train_contrib[group]:+.4f} → Val={val_contrib[group]:+.4f} (delta={delta:+.4f})")

    # Find biggest change
    if deltas:
        max_delta_group = max(deltas.keys(), key=lambda k: abs(deltas[k]['delta']))
        print(f"\n>>> Biggest change: {max_delta_group} (delta={deltas[max_delta_group]['delta']:+.4f})")

    results = {
        'task': task_name,
        'dataset': 'rel-stack',
        'train_variance': train_variance,
        'val_variance': val_variance,
        'variance_change_pct': (val_variance - train_variance) / train_variance * 100 if train_variance > 0 else 0,
        'train_contributions': train_contrib,
        'val_contributions': val_contrib,
        'train_contributions_norm': train_contrib_norm,
        'val_contributions_norm': val_contrib_norm,
        'deltas': deltas,
        'fk_relationships': dict(entity_table.fkey_col_to_pkey_table)
    }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="post-votes")
    parser.add_argument("--sample_size", type=int, default=5000)
    parser.add_argument("--all_tasks", action="store_true", help="Run all Stack entity tasks")
    args = parser.parse_args()

    output_dir = Path("chorok/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.all_tasks:
        # Stack entity tasks with proper FK structure
        tasks = ["post-votes"]  # user-engagement and user-badge use users table (no FKs)
    else:
        tasks = [args.task]

    all_results = {}
    for task in tasks:
        try:
            results = run_attribution_comparison(task, args.sample_size)
            all_results[task] = results
        except Exception as e:
            print(f"Error on {task}: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    output_file = output_dir / "fk_attribution_stack.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: FK Uncertainty Attribution (Stack)")
    print("="*60)

    for task, results in all_results.items():
        print(f"\n{task}:")
        print(f"  Variance: {results['train_variance']:.4f} → {results['val_variance']:.4f} ({results['variance_change_pct']:+.1f}%)")
        print(f"  FK relationships: {results['fk_relationships']}")

        if results['deltas']:
            max_group = max(results['deltas'].keys(), key=lambda k: abs(results['deltas'][k]['delta']))
            print(f"  Biggest FK delta: {max_group} ({results['deltas'][max_group]['delta']:+.4f})")


if __name__ == "__main__":
    main()
