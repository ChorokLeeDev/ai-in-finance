"""
Quick Validation Test: FK-level uncertainty vs feature importance

Purpose: Check if FK-grouped uncertainty contribution differs from
         FK-grouped feature importance (to validate research direction)

Test:
1. Train LightGBM with all features → get feature importance & uncertainty
2. Group by FK → compare rankings
3. If rankings differ significantly → proceed with research
4. If rankings similar → reconsider direction
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import lightgbm as lgb
from scipy.stats import entropy, spearmanr
from collections import defaultdict

from relbench.datasets import get_dataset
from relbench.tasks import get_task


def get_prediction_entropy(proba):
    """Calculate entropy of prediction probabilities"""
    # Add small epsilon to avoid log(0)
    proba = np.clip(proba, 1e-10, 1 - 1e-10)
    return entropy(proba, axis=1)


def run_test(dataset_name, task_name, sample_size=10000):
    """Run quick validation test"""

    print(f"\n{'='*60}")
    print(f"Quick Validation Test: {dataset_name}/{task_name}")
    print(f"{'='*60}\n")

    # Load data
    print("Loading dataset...")
    task = get_task(dataset_name, task_name, download=True)
    dataset = task.dataset
    db = dataset.get_db()

    # Get entity table info
    entity_table = db.table_dict[task.entity_table]
    print(f"Entity table: {task.entity_table}")
    print(f"FKs: {entity_table.fkey_col_to_pkey_table}")

    # Get train/val tables
    train_table = task.get_table("train")
    val_table = task.get_table("val")

    # Prepare features by joining with FK tables
    print("\nPreparing features...")

    # Get entity df
    entity_df = entity_table.df.copy()

    # Merge train table with entity
    train_df = train_table.df.copy()

    # Find the FK column that links to entity table
    fkey_cols = list(train_table.fkey_col_to_pkey_table.keys())
    entity_fkey = None
    for fk in fkey_cols:
        if train_table.fkey_col_to_pkey_table[fk] == task.entity_table:
            entity_fkey = fk
            break

    if entity_fkey is None:
        print(f"Warning: No FK to entity table found in train table")
        print(f"Train FKs: {train_table.fkey_col_to_pkey_table}")
        # Try direct merge
        entity_fkey = entity_table.pkey_col

    print(f"Merging on: {entity_fkey} -> {entity_table.pkey_col}")

    # Merge
    merged_train = train_df.merge(
        entity_df,
        left_on=entity_fkey if entity_fkey != entity_table.pkey_col else entity_table.pkey_col,
        right_on=entity_table.pkey_col,
        how='left',
        suffixes=('', '_entity')
    )

    # Sample if too large
    if len(merged_train) > sample_size:
        merged_train = merged_train.sample(sample_size, random_state=42)

    print(f"Training samples: {len(merged_train)}")

    # Identify feature columns by FK source
    print("\nIdentifying feature groups by FK...")

    # Features from entity table (excluding pkey, time, target)
    exclude_cols = {entity_table.pkey_col, entity_table.time_col, task.target_col}
    exclude_cols.update(entity_table.fkey_col_to_pkey_table.keys())
    exclude_cols.update(train_table.fkey_col_to_pkey_table.keys())
    exclude_cols.discard(None)

    entity_features = [c for c in entity_df.columns
                       if c not in exclude_cols and c in merged_train.columns]

    # Features from each FK's target table
    fk_features = {}
    for fk_col, target_table_name in entity_table.fkey_col_to_pkey_table.items():
        target_table = db.table_dict[target_table_name]
        target_cols = [c for c in target_table.df.columns
                      if c != target_table.pkey_col and c != target_table.time_col]
        # Find which of these are in merged_train
        available = [c for c in target_cols if c in merged_train.columns]
        if available:
            fk_features[fk_col] = available

    print(f"\nFeature groups:")
    print(f"  Entity own features ({len(entity_features)}): {entity_features[:5]}...")
    for fk, feats in fk_features.items():
        print(f"  FK '{fk}' features ({len(feats)}): {feats[:5]}...")

    # All features
    all_features = entity_features.copy()
    for feats in fk_features.values():
        all_features.extend(feats)
    all_features = list(set(all_features))

    # Remove non-numeric columns
    numeric_features = []
    for f in all_features:
        if merged_train[f].dtype in ['int64', 'float64', 'int32', 'float32']:
            numeric_features.append(f)
        elif merged_train[f].dtype == 'object':
            # Try to convert categorical
            try:
                merged_train[f] = merged_train[f].astype('category').cat.codes
                numeric_features.append(f)
            except:
                pass

    all_features = numeric_features
    print(f"\nTotal numeric features: {len(all_features)}")

    if len(all_features) < 5:
        print("ERROR: Not enough features for meaningful test")
        return None

    # Prepare X, y
    X = merged_train[all_features].fillna(-999)
    y = merged_train[task.target_col]

    # Encode target if categorical
    if y.dtype == 'object':
        y = y.astype('category').cat.codes

    # Check if classification or regression
    n_unique = y.nunique()
    is_classification = n_unique < 50
    print(f"Task type: {'classification' if is_classification else 'regression'} ({n_unique} unique values)")

    # Train LightGBM
    print("\nTraining LightGBM...")

    if is_classification:
        model = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    else:
        model = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)

    model.fit(X, y)

    # Get feature importance
    importance = dict(zip(all_features, model.feature_importances_))

    # Group importance by FK
    print("\n" + "="*60)
    print("RESULT A: Feature Importance grouped by FK")
    print("="*60)

    fk_importance = {}

    # Entity own features
    entity_imp = sum(importance.get(f, 0) for f in entity_features)
    fk_importance['entity_own'] = entity_imp
    print(f"  entity_own: {entity_imp:.2f}")

    # FK features
    for fk, feats in fk_features.items():
        imp = sum(importance.get(f, 0) for f in feats)
        fk_importance[fk] = imp
        print(f"  {fk}: {imp:.2f}")

    # Normalize
    total_imp = sum(fk_importance.values())
    if total_imp > 0:
        fk_importance = {k: v/total_imp for k, v in fk_importance.items()}

    print("\nNormalized:")
    for k, v in sorted(fk_importance.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v:.3f}")

    # Now calculate uncertainty contribution
    print("\n" + "="*60)
    print("RESULT B: Uncertainty contribution by FK")
    print("="*60)

    if is_classification:
        # Get baseline entropy
        proba_full = model.predict_proba(X)
        entropy_full = get_prediction_entropy(proba_full).mean()
        print(f"Baseline entropy (all features): {entropy_full:.4f}")

        fk_uncertainty_contribution = {}

        # Remove entity own features
        if entity_features:
            X_no_entity = X.drop(columns=[f for f in entity_features if f in X.columns], errors='ignore')
            if len(X_no_entity.columns) > 0:
                model_no_entity = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
                model_no_entity.fit(X_no_entity, y)
                proba_no_entity = model_no_entity.predict_proba(X_no_entity)
                entropy_no_entity = get_prediction_entropy(proba_no_entity).mean()
                # Contribution = how much entropy increases when removed
                fk_uncertainty_contribution['entity_own'] = entropy_no_entity - entropy_full
                print(f"  Without entity_own: entropy={entropy_no_entity:.4f}, delta={entropy_no_entity - entropy_full:.4f}")

        # Remove each FK's features
        for fk, feats in fk_features.items():
            X_no_fk = X.drop(columns=[f for f in feats if f in X.columns], errors='ignore')
            if len(X_no_fk.columns) > 0:
                model_no_fk = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
                model_no_fk.fit(X_no_fk, y)
                proba_no_fk = model_no_fk.predict_proba(X_no_fk)
                entropy_no_fk = get_prediction_entropy(proba_no_fk).mean()
                fk_uncertainty_contribution[fk] = entropy_no_fk - entropy_full
                print(f"  Without {fk}: entropy={entropy_no_fk:.4f}, delta={entropy_no_fk - entropy_full:.4f}")

        # Normalize
        total_contrib = sum(abs(v) for v in fk_uncertainty_contribution.values())
        if total_contrib > 0:
            fk_uncertainty_contribution = {k: abs(v)/total_contrib for k, v in fk_uncertainty_contribution.items()}

        print("\nNormalized uncertainty contribution:")
        for k, v in sorted(fk_uncertainty_contribution.items(), key=lambda x: -x[1]):
            print(f"  {k}: {v:.3f}")

    else:
        print("Skipping uncertainty analysis for regression task")
        fk_uncertainty_contribution = fk_importance.copy()

    # Compare rankings
    print("\n" + "="*60)
    print("COMPARISON: Feature Importance vs Uncertainty Contribution")
    print("="*60)

    common_keys = set(fk_importance.keys()) & set(fk_uncertainty_contribution.keys())
    if len(common_keys) >= 2:
        imp_values = [fk_importance[k] for k in common_keys]
        unc_values = [fk_uncertainty_contribution[k] for k in common_keys]

        correlation, pvalue = spearmanr(imp_values, unc_values)
        print(f"\nSpearman correlation: {correlation:.3f} (p={pvalue:.3f})")

        print("\nRanking comparison:")
        imp_rank = sorted(common_keys, key=lambda k: -fk_importance[k])
        unc_rank = sorted(common_keys, key=lambda k: -fk_uncertainty_contribution[k])

        print(f"  Feature Importance ranking: {imp_rank}")
        print(f"  Uncertainty Contribution ranking: {unc_rank}")

        if imp_rank == unc_rank:
            print("\n>>> SAME RANKING - Possibly trivial (just aggregated SHAP)")
        else:
            print("\n>>> DIFFERENT RANKING - Potential for novel contribution!")

        return {
            'correlation': correlation,
            'pvalue': pvalue,
            'imp_rank': imp_rank,
            'unc_rank': unc_rank,
            'same_ranking': imp_rank == unc_rank
        }
    else:
        print("Not enough FK groups for comparison")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="rel-stack")
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--sample_size", type=int, default=10000)
    args = parser.parse_args()

    if args.task is None:
        # Try to find a suitable task
        from relbench.tasks import get_task_names
        tasks = get_task_names(args.dataset)
        print(f"Available tasks: {tasks}")
        if tasks:
            args.task = tasks[0]
        else:
            print("No tasks found")
            exit(1)

    result = run_test(args.dataset, args.task, args.sample_size)

    if result:
        print("\n" + "="*60)
        print("CONCLUSION")
        print("="*60)
        if result['same_ranking']:
            print("Rankings are SAME - uncertainty contribution ≈ aggregated feature importance")
            print("Research direction may be TRIVIAL")
        else:
            print("Rankings are DIFFERENT - uncertainty behaves differently from prediction")
            print("Research direction has POTENTIAL")
