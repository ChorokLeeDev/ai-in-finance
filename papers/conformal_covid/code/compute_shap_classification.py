#!/usr/bin/env python3
"""
Compute SHAP Concentration for Classification Tasks

Extends SHAP analysis to binary classification tasks from rel-trial and rel-f1.
Tests if concentration-degradation correlation holds for classification.

Tasks:
- rel-trial/study-outcome (binary: study outcome positive/negative)
- rel-trial/study-adverse (binary: adverse events occurred)
- rel-trial/site-success (binary: site recruitment success)
- rel-f1/driver-dnf (binary: driver did not finish)

Usage:
    python compute_shap_classification.py --dataset rel-trial --task study-outcome
    python compute_shap_classification.py --dataset rel-f1 --task driver-dnf

Author: UAI 2026 Conformal COVID Paper
Date: 2025-12-27
"""

import argparse
import pickle
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

# Add relbench to path
sys.path.insert(0, '/Users/i767700/Github/ai-in-finance')

from relbench.datasets import get_dataset
from relbench.tasks import get_task


def load_and_preprocess_data(
    task,
    sample_size: int = 30000,
    seed: int = 42
) -> Tuple[Dict, Dict, List[str], Dict]:
    """
    Load task data and preprocess features.

    Returns:
        X_data: Dict of {'train', 'val', 'test'} -> feature matrices
        y_data: Dict of {'train', 'val', 'test'} -> labels
        feature_names: List of feature column names
        raw_data: Raw dataframes for Jaccard computation
    """
    print("Loading data...")
    train_table = task.get_table("train")
    val_table = task.get_table("val")
    test_table = task.get_table("test", mask_input_cols=False)

    dataset = task.dataset
    entity_table = dataset.get_db().table_dict[task.entity_table]
    entity_df = entity_table.df.copy()

    # Merge with entity table
    dfs = {}
    for split, table in [("train", train_table), ("val", val_table), ("test", test_table)]:
        entity_df_copy = entity_df.copy()
        left_entity = list(table.fkey_col_to_pkey_table.keys())[0]
        entity_df_copy = entity_df_copy.astype({entity_table.pkey_col: table.df[left_entity].dtype})

        # Drop duplicate columns
        for col in set(entity_df_copy.columns).intersection(set(table.df.columns)):
            if col != entity_table.pkey_col:
                entity_df_copy = entity_df_copy.drop(columns=[col])

        dfs[split] = table.df.merge(
            entity_df_copy, how="left",
            left_on=left_entity, right_on=entity_table.pkey_col,
        )

    # Subsample training data
    if sample_size and sample_size < len(dfs["train"]):
        np.random.seed(seed)
        idx = np.random.permutation(len(dfs["train"]))[:sample_size]
        dfs["train"] = dfs["train"].iloc[idx].copy()

    target_col = task.target_col

    # Define feature columns (exclude target, timestamps, IDs)
    all_data = pd.concat([dfs["train"], dfs["val"], dfs["test"]], ignore_index=True)
    exclude_cols = [target_col, 'CREATIONTIMESTAMP', 'timestamp']
    id_cols = [c for c in all_data.columns if c.endswith('_id') or c.endswith('Id') or c == 'ID']
    exclude_cols.extend(id_cols)
    feature_cols = [c for c in all_data.columns if c not in exclude_cols]

    print(f"  Features: {len(feature_cols)}")
    print(f"  Training samples: {len(dfs['train'])}")
    print(f"  Validation samples: {len(dfs['val'])}")
    print(f"  Test samples: {len(dfs['test'])}")

    # Encode categorical features
    label_encoders = {}
    for col in feature_cols:
        if all_data[col].dtype == 'object' or all_data[col].dtype.name == 'category':
            le = LabelEncoder()
            all_data[col] = all_data[col].astype(str).fillna('__MISSING__')
            le.fit(all_data[col])
            label_encoders[col] = le

    # Create feature matrices
    X_data, y_data = {}, {}
    for split, df in dfs.items():
        X = df[feature_cols].copy()
        for col, le in label_encoders.items():
            X[col] = X[col].astype(str).fillna('__MISSING__')
            X[col] = X[col].apply(lambda x: x if x in le.classes_ else '__MISSING__')
            if '__MISSING__' not in le.classes_:
                le.classes_ = np.append(le.classes_, '__MISSING__')
            X[col] = le.transform(X[col])
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(-999)
        X_data[split] = X.values.astype(np.float32)
        y_data[split] = df[target_col].values.astype(int)

    # Store raw data for Jaccard computation
    raw_data = {split: dfs[split][feature_cols].copy() for split in dfs}

    return X_data, y_data, feature_cols, raw_data


def train_binary_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int = 42
) -> lgb.Booster:
    """Train LightGBM binary classifier."""
    print("Training LightGBM binary classifier...")

    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': seed,
        'n_jobs': -1,
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )

    print(f"  Model trained with {model.num_trees()} trees")
    return model


def compute_shap_values(
    model: lgb.Booster,
    X: np.ndarray,
    subsample_size: int = 10000,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute SHAP values using TreeExplainer.

    Returns:
        shap_values: SHAP values (n_samples, n_features)
        X_sample: Subsampled features used
    """
    if len(X) > subsample_size:
        np.random.seed(seed)
        idx = np.random.choice(len(X), subsample_size, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X

    print(f"  Computing SHAP values for {len(X_sample)} samples...")

    explainer = shap.TreeExplainer(model)
    shap_values_raw = explainer.shap_values(X_sample)

    # For binary classification, shap_values might be:
    # - Single array (n_samples, n_features) for class 1
    # - List of 2 arrays for [class 0, class 1]
    if isinstance(shap_values_raw, list):
        # Use class 1 SHAP values
        shap_values = shap_values_raw[1]
    else:
        shap_values = shap_values_raw

    print(f"  SHAP values shape: {shap_values.shape}")
    return shap_values, X_sample


def compute_feature_importance(
    shap_values: np.ndarray,
    feature_names: List[str]
) -> List[Tuple[str, float]]:
    """
    Compute feature importance from SHAP values.

    Returns:
        List of (feature_name, importance) sorted by importance (descending)
    """
    # Mean absolute SHAP value per feature
    importance = np.mean(np.abs(shap_values), axis=0)

    # Create list and sort
    feature_importance = list(zip(feature_names, importance))
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    return feature_importance


def compute_jaccard_similarity(
    raw_data: Dict[str, pd.DataFrame],
    feature_names: List[str]
) -> Dict[str, float]:
    """
    Compute Jaccard similarity for each feature between val and test sets.

    Returns:
        Dict mapping feature_name -> jaccard_similarity
    """
    print("Computing Jaccard similarity for features...")

    val_df = raw_data['val']
    test_df = raw_data['test']

    jaccard_scores = {}

    for col in feature_names:
        # Get unique values
        val_set = set(val_df[col].dropna().unique())
        test_set = set(test_df[col].dropna().unique())

        # Compute Jaccard
        if len(val_set) == 0 and len(test_set) == 0:
            jaccard = 1.0
        else:
            intersection = len(val_set.intersection(test_set))
            union = len(val_set.union(test_set))
            jaccard = intersection / union if union > 0 else 0.0

        jaccard_scores[col] = jaccard

    return jaccard_scores


def compute_concentration_metric(
    feature_importance: List[Tuple[str, float]]
) -> Tuple[float, float, float]:
    """
    Compute SHAP concentration.

    Returns:
        top_feature_importance: Importance of top feature
        total_importance: Sum of all importances
        concentration_pct: (top / total) * 100
    """
    top_feature_importance = feature_importance[0][1]
    total_importance = sum(imp for _, imp in feature_importance)
    concentration_pct = (top_feature_importance / total_importance * 100) if total_importance > 0 else 0

    return top_feature_importance, total_importance, concentration_pct


def main():
    parser = argparse.ArgumentParser(
        description="Compute SHAP concentration for classification tasks"
    )
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (e.g., rel-trial, rel-f1)')
    parser.add_argument('--task', type=str, required=True,
                       help='Task name (e.g., study-outcome, driver-dnf)')
    parser.add_argument('--subsample', type=int, default=10000,
                       help='SHAP subsample size')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"SHAP Analysis: {args.dataset}/{args.task}")
    print(f"{'='*80}\n")

    # Load task
    print("Loading task...")
    dataset = get_dataset(args.dataset, download=True)
    task = get_task(args.dataset, args.task, download=True)
    task.dataset = dataset

    # Load and preprocess data
    X_data, y_data, feature_names, raw_data = load_and_preprocess_data(
        task, sample_size=30000, seed=args.seed
    )

    # Train model
    model = train_binary_classifier(
        X_data['train'], y_data['train'],
        X_data['val'], y_data['val'],
        seed=args.seed
    )

    # Compute SHAP values on validation set
    print("\nComputing SHAP values on validation set...")
    shap_values_val, X_val_sample = compute_shap_values(
        model, X_data['val'], subsample_size=args.subsample, seed=args.seed
    )

    # Compute SHAP values on test set
    print("\nComputing SHAP values on test set...")
    shap_values_test, X_test_sample = compute_shap_values(
        model, X_data['test'], subsample_size=args.subsample, seed=args.seed
    )

    # Compute feature importance
    print("\nComputing feature importance...")
    feature_importance_val = compute_feature_importance(shap_values_val, feature_names)
    feature_importance_test = compute_feature_importance(shap_values_test, feature_names)

    # Compute Jaccard similarity
    jaccard_scores = compute_jaccard_similarity(raw_data, feature_names)

    # Compute concentration
    top_imp_val, total_imp_val, concentration_val = compute_concentration_metric(feature_importance_val)
    top_imp_test, total_imp_test, concentration_test = compute_concentration_metric(feature_importance_test)

    # Top feature analysis
    top_feature_val = feature_importance_val[0][0]
    top_feature_test = feature_importance_test[0][0]
    top_feature_jaccard = jaccard_scores.get(top_feature_val, 0.0)

    # Mean Jaccard for top 10 features
    top10_features = [feat for feat, _ in feature_importance_val[:10]]
    top10_jaccard = [jaccard_scores.get(feat, 0.0) for feat in top10_features]
    mean_jaccard_top10 = np.mean(top10_jaccard)

    # Print results
    print(f"\n{'='*80}")
    print("SHAP Concentration Analysis Results")
    print(f"{'='*80}\n")

    print(f"Validation Set:")
    print(f"  Top feature: {top_feature_val}")
    print(f"  Top feature importance: {top_imp_val:.4f}")
    print(f"  Total importance: {total_imp_val:.4f}")
    print(f"  Concentration: {concentration_val:.2f}%")
    print(f"  Top feature Jaccard: {top_feature_jaccard:.3f}")
    print(f"  Mean Jaccard (top 10): {mean_jaccard_top10:.3f}")

    print(f"\nTest Set:")
    print(f"  Top feature: {top_feature_test}")
    print(f"  Top feature importance: {top_imp_test:.4f}")
    print(f"  Total importance: {total_imp_test:.4f}")
    print(f"  Concentration: {concentration_test:.2f}%")

    print(f"\nTop 10 Features (Validation):")
    for i, (feat, imp) in enumerate(feature_importance_val[:10], 1):
        jac = jaccard_scores.get(feat, 0.0)
        print(f"  {i:2d}. {feat:30s} | Importance: {imp:.4f} | Jaccard: {jac:.3f}")

    # Save results
    output_dir = Path('papers/conformal_covid/results/shap')
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'dataset': args.dataset,
        'task': args.task,
        'concentration_val': concentration_val,
        'concentration_test': concentration_test,
        'top_feature_val': top_feature_val,
        'top_feature_test': top_feature_test,
        'top_feature_jaccard': top_feature_jaccard,
        'mean_jaccard_top10': mean_jaccard_top10,
        'feature_importance_val': feature_importance_val,
        'feature_importance_test': feature_importance_test,
        'jaccard_scores': jaccard_scores,
        'shap_values_val': shap_values_val,
        'shap_values_test': shap_values_test,
    }

    output_file = output_dir / f'shap_{args.dataset}_{args.task}.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)

    print(f"\n✓ Results saved to {output_file}")

    print(f"\n{'='*80}")
    print("Analysis Complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
