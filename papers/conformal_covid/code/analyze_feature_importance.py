"""
Feature Importance Analysis with SHAP

Analyzes which features models rely on and their temporal stability.
Tests hypothesis: catastrophic tasks rely on unstable (low-Jaccard) features.

Usage:
    python analyze_feature_importance.py --dataset rel-salt --task sales-shipcond
    python analyze_feature_importance.py --dataset rel-salt --task sales-office

Author: UAI 2026 Conformal COVID Paper
Date: 2025-12-26
"""

import argparse
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')


def load_and_preprocess_data(
    task,
    sample_size: int = 30000,
    seed: int = 42
) -> Tuple[Dict, Dict, List[str], LabelEncoder, Dict]:
    """
    Load task data and preprocess features.
    Reuses pipeline from compute_confidence_intervals.py.

    Returns:
        X_data: Dict of {'train', 'val', 'test'} -> feature matrices
        y_data: Dict of {'train', 'val', 'test'} -> labels
        feature_names: List of feature column names
        target_le: Label encoder for target
        label_encoders: Dict of label encoders for categorical features
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
        y_data[split] = df[target_col].values

    # Encode target
    target_le = LabelEncoder()
    all_y = np.concatenate([y_data['train'], y_data['val'], y_data['test']])
    target_le.fit(all_y)
    for split in y_data:
        y_data[split] = target_le.transform(y_data[split])

    # Store raw data for Jaccard computation
    raw_data = {split: dfs[split][feature_cols].copy() for split in dfs}

    return X_data, y_data, feature_cols, target_le, label_encoders, raw_data


def train_lightgbm_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    seed: int = 42
) -> lgb.Booster:
    """Train LightGBM classifier."""
    print("Training LightGBM model...")

    params = {
        'objective': 'multiclass',
        'num_class': num_classes,
        'metric': 'multi_logloss',
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

    Args:
        model: Trained LightGBM model
        X: Feature matrix (n_samples, n_features)
        subsample_size: Max samples for SHAP computation
        seed: Random seed

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

    # TreeExplainer is fast for LightGBM
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # For multiclass, take mean absolute across classes
    if isinstance(shap_values, list):
        # Older SHAP versions: list of arrays, one per class
        shap_values = np.abs(shap_values).mean(axis=0)
    else:
        shap_values = np.abs(shap_values)
        # Newer SHAP versions: 3D array (n_samples, n_features, n_classes)
        if shap_values.ndim == 3:
            shap_values = shap_values.mean(axis=2)  # Average across classes

    return shap_values, X_sample


def analyze_top_features(
    shap_values: np.ndarray,
    feature_names: List[str],
    top_k: int = 10
) -> Tuple[List[Tuple[str, float]], np.ndarray]:
    """
    Identify top-k most important features.

    Args:
        shap_values: SHAP values (n_samples, n_features)
        feature_names: List of feature names
        top_k: Number of top features to return

    Returns:
        top_features: List of (feature_name, importance_score) tuples
        top_indices: Indices of top features in original array
    """
    # Mean absolute SHAP value per feature
    feature_importance = np.abs(shap_values).mean(axis=0).ravel()

    # Cap top_k to number of available features
    n_features = min(len(feature_names), len(feature_importance))
    top_k = min(top_k, n_features)

    # Sort descending and get top-k
    top_indices = np.argsort(feature_importance)[-top_k:][::-1]

    top_features = [
        (feature_names[int(i)], float(feature_importance[int(i)]))
        for i in top_indices
    ]

    return top_features, top_indices


def compute_feature_jaccard(
    train_values: pd.Series,
    test_values: pd.Series
) -> float:
    """
    Compute Jaccard similarity for a single feature.

    Args:
        train_values: Feature values from training set
        test_values: Feature values from test set

    Returns:
        jaccard: Jaccard similarity (0 to 1)
    """
    # Remove NaN values
    train_clean = train_values.dropna().unique()
    test_clean = test_values.dropna().unique()

    if len(train_clean) == 0 or len(test_clean) == 0:
        return 0.0

    # Convert to sets
    train_set = set(train_clean)
    test_set = set(test_clean)

    # Compute Jaccard
    intersection = len(train_set & test_set)
    union = len(train_set | test_set)

    return intersection / union if union > 0 else 0.0


def compute_all_feature_jaccards(
    raw_train: pd.DataFrame,
    raw_test: pd.DataFrame,
    feature_names: List[str]
) -> Dict[str, float]:
    """Compute Jaccard similarity for all features."""
    print("Computing Jaccard similarity for all features...")

    feature_jaccard = {}
    for fname in feature_names:
        jaccard = compute_feature_jaccard(raw_train[fname], raw_test[fname])
        feature_jaccard[fname] = jaccard

    return feature_jaccard


def main():
    parser = argparse.ArgumentParser(
        description="Feature importance analysis with SHAP"
    )
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (e.g., rel-salt)')
    parser.add_argument('--task', type=str, required=True,
                       help='Task name (e.g., sales-shipcond)')
    parser.add_argument('--subsample', type=int, default=10000,
                       help='Max samples for SHAP computation')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output_dir', type=str,
                       default='papers/conformal_covid/results/shap',
                       help='Output directory for results')
    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"Feature Importance Analysis: {args.dataset}/{args.task}")
    print(f"{'='*80}\n")

    # Import here to avoid startup delay
    from relbench.tasks import get_task

    # 1. Load data and preprocess
    # Data already cached locally, no need to download
    task = get_task(args.dataset, args.task, download=False)
    X_data, y_data, feature_names, target_le, label_encoders, raw_data = \
        load_and_preprocess_data(task, seed=args.seed)

    num_classes = len(target_le.classes_)

    # 2. Train model
    model = train_lightgbm_model(
        X_data['train'], y_data['train'],
        X_data['val'], y_data['val'],
        num_classes, args.seed
    )

    # 3. Compute SHAP on validation (pre-COVID)
    print("\nComputing SHAP values on validation set (pre-COVID)...")
    shap_val, X_val_sample = compute_shap_values(
        model, X_data['val'], args.subsample, args.seed
    )
    top_features_val, top_idx_val = analyze_top_features(
        shap_val, feature_names, top_k=10
    )

    print("\nTop 10 features (validation):")
    for i, (fname, score) in enumerate(top_features_val):
        print(f"  {i+1:2d}. {fname:30s}: {score:.4f}")

    # 4. Compute SHAP on test (post-COVID)
    print("\nComputing SHAP values on test set (post-COVID)...")
    shap_test, X_test_sample = compute_shap_values(
        model, X_data['test'], args.subsample, args.seed
    )
    top_features_test, top_idx_test = analyze_top_features(
        shap_test, feature_names, top_k=10
    )

    print("\nTop 10 features (test):")
    for i, (fname, score) in enumerate(top_features_test):
        print(f"  {i+1:2d}. {fname:30s}: {score:.4f}")

    # 5. Compute Jaccard for all features
    feature_jaccard = compute_all_feature_jaccards(
        raw_data['train'], raw_data['test'], feature_names
    )

    # 6. Show Jaccard for top features
    print("\nJaccard similarity for top 10 features:")
    for fname, score in top_features_val:
        jaccard = feature_jaccard[fname]
        print(f"  {fname:30s}: SHAP={score:.4f}, Jaccard={jaccard:.4f}")

    # Compute mean Jaccard for top-10 vs all features
    top_feature_names = [f[0] for f in top_features_val]
    mean_jaccard_top10 = np.mean([feature_jaccard[f] for f in top_feature_names])
    mean_jaccard_all = np.mean(list(feature_jaccard.values()))

    print(f"\nMean Jaccard (top-10 features): {mean_jaccard_top10:.4f}")
    print(f"Mean Jaccard (all features):    {mean_jaccard_all:.4f}")

    # 7. Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'task': args.task,
        'dataset': args.dataset,
        'top_features_val': top_features_val,
        'top_features_test': top_features_test,
        'feature_jaccard': feature_jaccard,
        'shap_values_val': shap_val,
        'shap_values_test': shap_test,
        'feature_names': feature_names,
        'mean_jaccard_top10': mean_jaccard_top10,
        'mean_jaccard_all': mean_jaccard_all,
        'num_classes': num_classes,
    }

    output_file = output_dir / f'shap_{args.dataset}_{args.task}.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)

    print(f"\n✓ Results saved to {output_file}")

    # 8. Generate plots
    print("\nGenerating plots...")
    from plot_shap_results import plot_shap_summary
    plot_shap_summary(results, output_dir=output_dir)

    print(f"\n{'='*80}")
    print("Feature importance analysis complete!")
    print(f"{'='*80}\n")

    return results


if __name__ == "__main__":
    main()
