"""
Conformalized Quantile Regression (CQR) for Regression Tasks

Implements CQR for regression tasks to complement classification analysis.
Tests same hypothesis: Does feature temporal stability predict coverage degradation?

Based on Romano et al. (2019) "Conformalized Quantile Regression"

Usage:
    python cqr_regression.py --dataset rel-trial --task study-duration --num_seeds 5

Features:
- Quantile regression with LightGBM
- Conformal prediction intervals
- Coverage analysis under distribution shift
- Feature stability (Jaccard) analysis for continuous features
"""

import argparse
import json
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

# Configuration
ALPHA = 0.1  # 90% target coverage
QUANTILES = [0.05, 0.95]  # For 90% prediction intervals
SAMPLE_SIZE = 30000


class CQR:
    """Conformalized Quantile Regression predictor."""

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.quantiles = [alpha/2, 1 - alpha/2]  # e.g., [0.05, 0.95] for 90%
        self.q_low_model = None
        self.q_high_model = None
        self.correction = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_cal: np.ndarray, y_cal: np.ndarray, seed: int = 42):
        """
        Fit CQR model.

        Args:
            X_train: Training features
            y_train: Training targets
            X_cal: Calibration features
            y_cal: Calibration targets
            seed: Random seed
        """
        # Fit quantile regression for lower bound
        params_low = {
            'objective': 'quantile',
            'alpha': self.quantiles[0],
            'metric': 'quantile',
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

        train_data_low = lgb.Dataset(X_train, label=y_train)
        self.q_low_model = lgb.train(params_low, train_data_low, num_boost_round=500)

        # Fit quantile regression for upper bound
        params_high = {
            'objective': 'quantile',
            'alpha': self.quantiles[1],
            'metric': 'quantile',
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

        train_data_high = lgb.Dataset(X_train, label=y_train)
        self.q_high_model = lgb.train(params_high, train_data_high, num_boost_round=500)

        # Calibrate: compute conformity scores
        q_low_cal = self.q_low_model.predict(X_cal)
        q_high_cal = self.q_high_model.predict(X_cal)

        # Conformity scores: max distance from interval
        scores = np.maximum(q_low_cal - y_cal, y_cal - q_high_cal)

        # Compute correction factor
        n = len(scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.correction = np.quantile(scores, min(q_level, 1.0))

        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict conformal intervals.

        Returns:
            (lower_bounds, upper_bounds)
        """
        q_low = self.q_low_model.predict(X)
        q_high = self.q_high_model.predict(X)

        # Apply conformal correction
        lower = q_low - self.correction
        upper = q_high + self.correction

        return lower, upper

    def coverage(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute empirical coverage."""
        lower, upper = self.predict(X)
        covered = (y >= lower) & (y <= upper)
        return np.mean(covered)

    def interval_width(self, X: np.ndarray) -> float:
        """Compute average interval width."""
        lower, upper = self.predict(X)
        return np.mean(upper - lower)


def compute_continuous_jaccard(train_values: np.ndarray, test_values: np.ndarray,
                               n_bins: int = 10) -> float:
    """
    Compute Jaccard similarity for continuous features using binning.

    Args:
        train_values: Values from training set
        test_values: Values from test set
        n_bins: Number of bins for discretization

    Returns:
        Jaccard similarity (0 to 1)
    """
    # Remove NaNs
    train_values = train_values[~np.isnan(train_values)]
    test_values = test_values[~np.isnan(test_values)]

    if len(train_values) == 0 or len(test_values) == 0:
        return 0.0

    # Determine bins from training data
    _, bins = np.histogram(train_values, bins=n_bins)

    # Bin both sets
    train_binned = np.digitize(train_values, bins)
    test_binned = np.digitize(test_values, bins)

    # Compute Jaccard
    train_set = set(train_binned)
    test_set = set(test_binned)

    intersection = len(train_set & test_set)
    union = len(train_set | test_set)

    return intersection / union if union > 0 else 0.0


def run_single_seed(task, task_name: str, seed: int,
                   sample_size: int = SAMPLE_SIZE) -> Dict:
    """
    Run CQR for a single seed.

    Args:
        task: RelBench task object
        task_name: Task name
        seed: Random seed
        sample_size: Training sample size

    Returns:
        Results dictionary
    """
    # Load data
    train_table = task.get_table("train")
    val_table = task.get_table("val")
    test_table = task.get_table("test", mask_input_cols=False)

    dataset = task.dataset
    entity_table = dataset.get_db().table_dict[task.entity_table]
    entity_df = entity_table.df.copy()

    # Merge entity features
    dfs = {}
    for split, table in [("train", train_table), ("val", val_table), ("test", test_table)]:
        entity_df_copy = entity_df.copy()
        left_entity = list(table.fkey_col_to_pkey_table.keys())[0]
        entity_df_copy = entity_df_copy.astype({entity_table.pkey_col: table.df[left_entity].dtype})

        # Remove duplicate columns
        for col in set(entity_df_copy.columns).intersection(set(table.df.columns)):
            if col != entity_table.pkey_col:
                entity_df_copy = entity_df_copy.drop(columns=[col])

        dfs[split] = table.df.merge(
            entity_df_copy,
            how="left",
            left_on=left_entity,
            right_on=entity_table.pkey_col,
        )

    # Subsample training data
    if sample_size and sample_size < len(dfs["train"]):
        np.random.seed(seed)
        idx = np.random.permutation(len(dfs["train"]))[:sample_size]
        dfs["train"] = dfs["train"].iloc[idx].copy()

    target_col = task.target_col

    # Feature engineering
    all_data = pd.concat([dfs["train"], dfs["val"], dfs["test"]], ignore_index=True)
    exclude_cols = [target_col, 'CREATIONTIMESTAMP', 'timestamp']

    # Exclude ID columns
    id_cols = [c for c in all_data.columns if c.endswith('_id') or c.endswith('Id') or c == 'ID']
    exclude_cols.extend(id_cols)

    feature_cols = [c for c in all_data.columns if c not in exclude_cols]

    # Encode categorical features
    label_encoders = {}
    for col in feature_cols:
        if all_data[col].dtype == 'object' or all_data[col].dtype.name == 'category':
            le = LabelEncoder()
            all_data[col] = all_data[col].astype(str).fillna('__MISSING__')
            le.fit(all_data[col])
            label_encoders[col] = le

    # Prepare datasets
    X_data, y_data = {}, {}
    for split, df in dfs.items():
        X = df[feature_cols].copy()

        # Apply label encoding
        for col, le in label_encoders.items():
            X[col] = X[col].astype(str).fillna('__MISSING__')
            X[col] = X[col].apply(lambda x: x if x in le.classes_ else '__MISSING__')
            if '__MISSING__' not in le.classes_:
                le.classes_ = np.append(le.classes_, '__MISSING__')
            X[col] = le.transform(X[col])

        # Convert to numeric
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(-999)

        X_data[split] = X.values.astype(np.float32)
        y_data[split] = df[target_col].values.astype(np.float32)

    # Split validation for calibration
    n_val = len(X_data['val'])
    n_calib = n_val // 2

    X_train = X_data['train']
    y_train = y_data['train']
    X_calib = X_data['val'][:n_calib]
    y_calib = y_data['val'][:n_calib]
    X_val = X_data['val'][n_calib:]
    y_val = y_data['val'][n_calib:]
    X_test = X_data['test']
    y_test = y_data['test']

    # Train CQR
    cqr = CQR(alpha=ALPHA)
    cqr.fit(X_train, y_train, X_calib, y_calib, seed=seed)

    # Compute coverage
    val_coverage = cqr.coverage(X_val, y_val)
    test_coverage = cqr.coverage(X_test, y_test)

    # Compute interval widths
    val_width = cqr.interval_width(X_val)
    test_width = cqr.interval_width(X_test)

    # Compute feature Jaccard (for continuous features)
    feature_jaccard = {}
    for i, col in enumerate(feature_cols):
        train_vals = X_data['train'][:, i]
        test_vals = X_data['test'][:, i]
        jaccard = compute_continuous_jaccard(train_vals, test_vals)
        feature_jaccard[col] = jaccard

    mean_jaccard = np.mean(list(feature_jaccard.values()))

    return {
        'task': task_name,
        'seed': seed,
        'val_coverage': val_coverage,
        'test_coverage': test_coverage,
        'coverage_drop': val_coverage - test_coverage,
        'val_interval_width': val_width,
        'test_interval_width': test_width,
        'mean_jaccard': mean_jaccard,
        'feature_jaccard': feature_jaccard,
        'target_mean': np.mean(y_train),
        'target_std': np.std(y_train),
    }


def run_regression_task(dataset_name: str, task_name: str,
                       num_seeds: int = 5) -> Dict:
    """
    Run CQR experiment for a regression task.

    Args:
        dataset_name: Dataset name (e.g., 'rel-trial')
        task_name: Task name
        num_seeds: Number of seeds to run

    Returns:
        Aggregated results
    """
    from relbench.tasks import get_task

    print(f"\n{'='*80}")
    print(f"Dataset: {dataset_name}, Task: {task_name}")
    print(f"Running {num_seeds} seeds with CQR...")
    print(f"{'='*80}")

    task = get_task(dataset_name, task_name, download=False)

    if task.task_type.name != 'REGRESSION':
        raise ValueError(f"Task {task_name} is not regression (type: {task.task_type})")

    seed_results = []
    for seed in range(42, 42 + num_seeds):
        print(f"  Seed {seed}...", end=" ", flush=True)
        try:
            result = run_single_seed(task, task_name, seed)
            seed_results.append(result)
            print(f"coverage_drop={result['coverage_drop']*100:.1f}%")
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()

    if not seed_results:
        raise RuntimeError("All seeds failed!")

    # Aggregate results
    val_covs = [r['val_coverage'] for r in seed_results]
    test_covs = [r['test_coverage'] for r in seed_results]
    drops = [r['coverage_drop'] for r in seed_results]
    val_widths = [r['val_interval_width'] for r in seed_results]
    test_widths = [r['test_interval_width'] for r in seed_results]
    jaccards = [r['mean_jaccard'] for r in seed_results]

    result = {
        'dataset': dataset_name,
        'task': task_name,
        'num_seeds': len(seed_results),
        'task_type': 'regression',

        # Coverage metrics
        'val_coverage_mean': np.mean(val_covs),
        'val_coverage_std': np.std(val_covs),
        'test_coverage_mean': np.mean(test_covs),
        'test_coverage_std': np.std(test_covs),
        'drop_mean': np.mean(drops),
        'drop_std': np.std(drops),

        # Interval width metrics
        'val_width_mean': np.mean(val_widths),
        'test_width_mean': np.mean(test_widths),

        # Feature stability
        'jaccard_mean': np.mean(jaccards),
        'jaccard_std': np.std(jaccards),

        # Target statistics
        'target_mean': seed_results[0]['target_mean'],
        'target_std': seed_results[0]['target_std'],

        # Raw data
        'seed_results': seed_results,
    }

    # Print summary
    print(f"\nResults for {task_name}:")
    print(f"  Val coverage:   {result['val_coverage_mean']*100:.1f} ± {result['val_coverage_std']*100:.1f}%")
    print(f"  Test coverage:  {result['test_coverage_mean']*100:.1f} ± {result['test_coverage_std']*100:.1f}%")
    print(f"  Drop:           {result['drop_mean']*100:.1f} ± {result['drop_std']*100:.1f}%")
    print(f"  Mean Jaccard:   {result['jaccard_mean']:.3f}")
    print(f"  Val width:      {result['val_width_mean']:.2f}")
    print(f"  Test width:     {result['test_width_mean']:.2f}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run CQR on regression tasks"
    )
    parser.add_argument('--dataset', type=str, default='rel-trial',
                       help='Dataset name (default: rel-trial)')
    parser.add_argument('--task', type=str, required=True,
                       help='Task name')
    parser.add_argument('--num_seeds', type=int, default=5,
                       help='Number of seeds (default: 5)')
    parser.add_argument('--output_dir', type=str,
                       default='papers/conformal_covid/results',
                       help='Output directory')

    args = parser.parse_args()

    # Run experiment
    result = run_regression_task(args.dataset, args.task, args.num_seeds)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"cqr_{args.dataset}_{args.task}.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(result, f)

    print(f"\n✓ Results saved to {output_file}")

    # Also save JSON summary
    json_file = output_dir / f"cqr_{args.dataset}_{args.task}.json"
    json_result = {
        'dataset': result['dataset'],
        'task': result['task'],
        'num_seeds': result['num_seeds'],
        'val_coverage': f"{result['val_coverage_mean']*100:.1f} ± {result['val_coverage_std']*100:.1f}%",
        'test_coverage': f"{result['test_coverage_mean']*100:.1f} ± {result['test_coverage_std']*100:.1f}%",
        'drop': f"{result['drop_mean']*100:.1f} ± {result['drop_std']*100:.1f}%",
        'jaccard': f"{result['jaccard_mean']:.3f} ± {result['jaccard_std']:.3f}",
    }

    with open(json_file, 'w') as f:
        json.dump(json_result, f, indent=2)

    print(f"✓ JSON summary saved to {json_file}")

    return result


if __name__ == "__main__":
    main()
