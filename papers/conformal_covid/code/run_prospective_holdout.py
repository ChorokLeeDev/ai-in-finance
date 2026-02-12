"""
Prospective Holdout Validation: rel-stack badges-class

Tests whether the SHAP concentration diagnostic derived from rel-salt (SAP ERP)
generalizes to a completely different domain (Stack Overflow).

badges-class: 3-class classification (Gold/Silver/Bronze badges)
Temporal split: train < Oct 2020, val = Oct 2020-Jan 2021, test > Jan 2021

Protocol:
1. Train LightGBM on badges features + joined user features
2. Compute SHAP concentration (top-1 / total)
3. Run APS conformal prediction on val/test
4. Measure coverage drop
5. Compare against SALT regression line prediction

Usage:
    PYTHONPATH=/Users/i767700/Github/ai-in-finance:$PYTHONPATH \
    /usr/bin/python3 -u papers/conformal_covid/code/run_prospective_holdout.py

Output:
    papers/conformal_covid/results/prospective_holdout.json
"""

import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from scipy import stats
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

# Configuration
ALPHA = 0.1
SAMPLE_SIZE = 50000
SHAP_SAMPLES = 5000
SEED = 42

RESULTS_DIR = Path(__file__).parent.parent / "results"
OUTPUT_FILE = RESULTS_DIR / "prospective_holdout.json"

# SALT regression line parameters (from mixed-effects model)
# beta_1 = 1.64 (boosting, 3 models): each 1% concentration -> 1.64pp coverage drop
# beta_0 (intercept) from boosting model: we'll compute prediction from the line
# For simpler prediction: use SALT Spearman relationship qualitatively
#   High concentration (>40%) -> predict vulnerable
#   Low concentration (<40%) -> predict robust


def load_task():
    """Load badges-class task from cached rel-stack data."""
    import pooch
    from relbench.datasets.stack import StackDataset
    from relbench.base.task_autocomplete import AutoCompleteTask
    from relbench.base.task_base import TaskType

    cache_dir = os.path.join(str(pooch.os_cache('relbench')), 'rel-stack')
    print(f"Cache dir: {cache_dir}")
    print(f"DB exists: {os.path.exists(os.path.join(cache_dir, 'db'))}")

    dataset = StackDataset(cache_dir=cache_dir)
    task = AutoCompleteTask(
        dataset=dataset,
        task_type=TaskType.MULTICLASS_CLASSIFICATION,
        entity_table='badges',
        target_col='Class',
        remove_columns=[('badges', 'TagBased'), ('badges', 'Name')],
    )
    print(f"Task created: num_classes={task.num_classes}, entity_col={task.entity_col}")
    return task


def prepare_features(task):
    """Prepare feature matrices from task tables."""
    print("\n--- Preparing features ---")

    train_table = task.get_table("train")
    val_table = task.get_table("val")
    test_table = task.get_table("test", mask_input_cols=False)

    print(f"Train: {train_table.df.shape}, Val: {val_table.df.shape}, Test: {test_table.df.shape}")

    dataset = task.dataset
    db = dataset.get_db()
    entity_table = db.table_dict[task.entity_table]
    entity_df = entity_table.df.copy()

    # Join entity features (badges table with Class/TagBased/Name removed -> UserId, Date)
    dfs = {}
    for split, table in [("train", train_table), ("val", val_table), ("test", test_table)]:
        entity_df_copy = entity_df.copy()
        left_entity = list(table.fkey_col_to_pkey_table.keys())[0]
        entity_df_copy = entity_df_copy.astype(
            {entity_table.pkey_col: table.df[left_entity].dtype}
        )
        for col in set(entity_df_copy.columns).intersection(set(table.df.columns)):
            if col != entity_table.pkey_col:
                entity_df_copy = entity_df_copy.drop(columns=[col])
        dfs[split] = table.df.merge(
            entity_df_copy, how="left",
            left_on=left_entity, right_on=entity_table.pkey_col,
        )

    # Also join user table features for richer feature set
    users_table = db.table_dict["users"]
    users_df = users_table.df.copy()
    print(f"Users table columns: {list(users_df.columns)}")

    for split in dfs:
        if 'UserId' in dfs[split].columns:
            # Merge user features
            users_merge = users_df.copy()
            users_merge = users_merge.rename(columns={'Id': 'UserId'})
            # Avoid column conflicts
            for col in set(users_merge.columns).intersection(set(dfs[split].columns)):
                if col != 'UserId':
                    users_merge = users_merge.rename(columns={col: f'{col}_user'})
            dfs[split] = dfs[split].merge(users_merge, on='UserId', how='left')

    print(f"After join - Train columns: {list(dfs['train'].columns)}")
    print(f"After join - Train shape: {dfs['train'].shape}")

    # Subsample training data
    if SAMPLE_SIZE and SAMPLE_SIZE < len(dfs["train"]):
        np.random.seed(SEED)
        idx = np.random.permutation(len(dfs["train"]))[:SAMPLE_SIZE]
        dfs["train"] = dfs["train"].iloc[idx].copy()
        print(f"Subsampled training to {len(dfs['train'])} rows")

    target_col = task.target_col

    # Identify feature columns
    all_data = pd.concat([dfs["train"], dfs["val"], dfs["test"]], ignore_index=True)
    exclude_cols = [target_col]
    # Exclude time columns
    time_cols = [c for c in all_data.columns if 'date' in c.lower() or 'time' in c.lower()]
    exclude_cols.extend(time_cols)
    # Exclude ID columns
    id_cols = [c for c in all_data.columns
               if c.lower().endswith('_id') or c.lower().endswith('id') or c.lower() == 'id']
    exclude_cols.extend(id_cols)
    # Remove duplicates
    exclude_cols = list(set(exclude_cols))

    feature_cols = [c for c in all_data.columns if c not in exclude_cols]
    print(f"Feature columns ({len(feature_cols)}): {feature_cols}")
    print(f"Excluded columns: {exclude_cols}")

    # Always add temporal features for a more realistic model
    if True:
        print("Adding engineered temporal features from Date and UserId")
        for split in dfs:
            df = dfs[split]
            # Extract temporal features from Date column
            if 'Date' in df.columns:
                df['badge_month'] = pd.to_datetime(df['Date']).dt.month
                df['badge_dayofweek'] = pd.to_datetime(df['Date']).dt.dayofweek
                df['badge_hour'] = pd.to_datetime(df['Date']).dt.hour
                df['badge_year'] = pd.to_datetime(df['Date']).dt.year
            if 'CreationDate_user' in df.columns:
                df['user_age_days'] = (
                    pd.to_datetime(df.get('Date', pd.Timestamp.now())) -
                    pd.to_datetime(df['CreationDate_user'])
                ).dt.days
            # User badge count proxy: UserId as numeric (captures user activity level)
            if 'UserId' in df.columns:
                df['user_id_numeric'] = pd.to_numeric(df['UserId'], errors='coerce').fillna(-1)
            dfs[split] = df

        # Recompute feature columns
        all_data = pd.concat([dfs["train"], dfs["val"], dfs["test"]], ignore_index=True)
        feature_cols = [c for c in all_data.columns if c not in exclude_cols]
        print(f"Updated feature columns ({len(feature_cols)}): {feature_cols}")

    # Encode categorical features
    label_encoders = {}
    for col in feature_cols:
        if all_data[col].dtype == 'object' or all_data[col].dtype.name == 'category':
            le = LabelEncoder()
            all_data[col] = all_data[col].astype(str).fillna('__MISSING__')
            le.fit(all_data[col])
            label_encoders[col] = le

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

    num_classes = len(target_le.classes_)
    print(f"\nNum classes: {num_classes}")
    print(f"Target distribution (train): {dict(zip(*np.unique(y_data['train'], return_counts=True)))}")
    print(f"Target distribution (val): {dict(zip(*np.unique(y_data['val'], return_counts=True)))}")
    print(f"Target distribution (test): {dict(zip(*np.unique(y_data['test'], return_counts=True)))}")

    return X_data, y_data, feature_cols, target_le, num_classes


def train_lgb(X_data, y_data, num_classes, feature_cols):
    """Train LightGBM model."""
    print("\n--- Training LightGBM ---")
    t0 = time.time()

    train_set = lgb.Dataset(X_data['train'], label=y_data['train'],
                            feature_name=feature_cols)
    val_set = lgb.Dataset(X_data['val'], label=y_data['val'],
                          feature_name=feature_cols, reference=train_set)

    params = {
        'objective': 'multiclass',
        'num_class': num_classes,
        'metric': 'multi_logloss',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': 6,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': SEED,
        'verbose': -1,
    }

    callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
    model = lgb.train(
        params, train_set,
        num_boost_round=500,
        valid_sets=[val_set],
        callbacks=callbacks,
    )

    elapsed = time.time() - t0
    print(f"Training done in {elapsed:.1f}s, best iteration: {model.best_iteration}")

    # Get predictions
    probs = {}
    for split in ['train', 'val', 'test']:
        probs[split] = model.predict(X_data[split])

    # Accuracy
    for split in ['train', 'val', 'test']:
        preds = np.argmax(probs[split], axis=1)
        acc = np.mean(preds == y_data[split])
        print(f"  {split} accuracy: {acc:.4f}")

    return model, probs, elapsed


def compute_shap_concentration(model, X_data, feature_cols):
    """Compute SHAP concentration metric."""
    print("\n--- Computing SHAP concentration ---")
    t0 = time.time()

    n_samples = min(SHAP_SAMPLES, len(X_data['val']))
    np.random.seed(SEED)
    idx = np.random.permutation(len(X_data['val']))[:n_samples]
    X_sample = X_data['val'][idx]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # shap_values is list of arrays for multiclass: [array_class0, array_class1, ...]
    if isinstance(shap_values, list):
        # Stack: (n_classes, n_samples, n_features) -> mean over samples and classes
        stacked = np.array(shap_values)  # (n_classes, n_samples, n_features)
        mean_abs = np.abs(stacked).mean(axis=(0, 1))  # (n_features,)
    elif shap_values.ndim == 3:
        mean_abs = np.abs(shap_values).mean(axis=(0, 1))
    else:
        mean_abs = np.abs(shap_values).mean(axis=0)

    total_importance = mean_abs.sum()
    top_feature_idx = np.argmax(mean_abs)
    top_feature_importance = mean_abs[top_feature_idx]
    concentration = (top_feature_importance / total_importance * 100) if total_importance > 0 else 0

    elapsed = time.time() - t0
    print(f"SHAP computation done in {elapsed:.1f}s")

    # Feature importance ranking
    sorted_idx = np.argsort(mean_abs)[::-1]
    print(f"\nTop features by mean |SHAP|:")
    for rank, fi in enumerate(sorted_idx[:min(10, len(sorted_idx))]):
        pct = mean_abs[fi] / total_importance * 100
        print(f"  {rank+1}. {feature_cols[fi]}: {mean_abs[fi]:.4f} ({pct:.1f}%)")

    print(f"\nSHAP Concentration (top-1/total): {concentration:.1f}%")
    print(f"Top feature: {feature_cols[top_feature_idx]}")

    return {
        'concentration': round(concentration, 2),
        'top_feature': feature_cols[top_feature_idx],
        'top_feature_importance': round(float(top_feature_importance), 4),
        'total_importance': round(float(total_importance), 4),
        'feature_ranking': {
            feature_cols[fi]: round(float(mean_abs[fi] / total_importance * 100), 2)
            for fi in sorted_idx[:10]
        },
    }


class ConformalClassifier:
    """Adaptive Prediction Sets (APS) for classification."""

    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.quantile = None

    def _compute_scores(self, probs, y_true):
        n = len(y_true)
        scores = np.zeros(n)
        for i in range(n):
            sorted_idx = np.argsort(probs[i])[::-1]
            cumsum = 0.0
            for idx in sorted_idx:
                cumsum += probs[i][idx]
                if idx == y_true[i]:
                    scores[i] = cumsum
                    break
        return scores

    def calibrate(self, probs, y_true):
        scores = self._compute_scores(probs, y_true)
        n = len(scores)
        self.quantile = np.quantile(scores, np.ceil((1 - self.alpha) * (n + 1)) / n, method='higher')
        return scores

    def predict_sets(self, probs):
        set_sizes = np.zeros(len(probs))
        for i in range(len(probs)):
            sorted_idx = np.argsort(probs[i])[::-1]
            cumsum = 0.0
            size = 0
            for idx in sorted_idx:
                cumsum += probs[i][idx]
                size += 1
                if cumsum >= self.quantile:
                    break
            set_sizes[i] = size
        return set_sizes

    def coverage(self, probs, y_true):
        scores = self._compute_scores(probs, y_true)
        return np.mean(scores <= self.quantile) * 100


def run_conformal(probs, y_data):
    """Run APS conformal prediction."""
    print("\n--- Running APS Conformal Prediction ---")

    # Split validation into calibration and evaluation
    n_val = len(y_data['val'])
    np.random.seed(SEED)
    perm = np.random.permutation(n_val)
    n_cal = n_val // 2

    cal_idx = perm[:n_cal]
    eval_idx = perm[n_cal:]

    cal_probs = probs['val'][cal_idx]
    cal_y = y_data['val'][cal_idx]
    eval_probs = probs['val'][eval_idx]
    eval_y = y_data['val'][eval_idx]

    # Calibrate on first half of val
    conformal = ConformalClassifier(alpha=ALPHA)
    conformal.calibrate(cal_probs, cal_y)

    # Coverage on eval part of val
    val_coverage = conformal.coverage(eval_probs, eval_y)
    val_sets = conformal.predict_sets(eval_probs)

    # Coverage on test
    test_coverage = conformal.coverage(probs['test'], y_data['test'])
    test_sets = conformal.predict_sets(probs['test'])

    coverage_drop = val_coverage - test_coverage

    print(f"  Calibration quantile: {conformal.quantile:.4f}")
    print(f"  Val coverage:  {val_coverage:.2f}% (target: {(1-ALPHA)*100:.0f}%)")
    print(f"  Test coverage: {test_coverage:.2f}%")
    print(f"  Coverage drop: {coverage_drop:.2f} pp")
    print(f"  Val mean set size:  {val_sets.mean():.2f}")
    print(f"  Test mean set size: {test_sets.mean():.2f}")

    return {
        'val_coverage': round(val_coverage, 2),
        'test_coverage': round(test_coverage, 2),
        'coverage_drop': round(coverage_drop, 2),
        'val_mean_set_size': round(float(val_sets.mean()), 2),
        'test_mean_set_size': round(float(test_sets.mean()), 2),
        'n_cal': n_cal,
        'n_eval': len(eval_idx),
        'n_test': len(y_data['test']),
    }


def assess_prediction(concentration, coverage_drop):
    """
    Assess whether the SALT-derived diagnostic predicts correctly.

    From SALT data (LGB, n=8):
      - Mixed-effects beta_1 = 1.64 (boosting): 1% concentration -> 1.64pp drop
      - Qualitative threshold: >40% concentration -> vulnerable

    From SALT per-task data:
      - Concentration range: 23.7% - 54.2%
      - Low concentration (<30%): drops 0-11%
      - High concentration (>40%): drops 18-77%
    """
    print("\n--- Diagnostic Assessment ---")

    # Predicted drop from mixed-effects regression line (boosting, 3 models)
    # Drop = beta_0 + beta_1 * concentration
    # From our data: intercept ~= -30 (rough), beta_1 = 1.64
    # Better: use the actual relationship from LGB data
    # Mean concentration across SALT tasks = 39.9%, mean drop = 34.6%
    # beta_1 = 1.64 -> predicted_drop = 1.64 * (concentration - 39.9) + 34.6
    predicted_drop_linear = 1.64 * (concentration - 39.9) + 34.6
    predicted_drop_linear = max(0, predicted_drop_linear)

    # Qualitative threshold prediction
    if concentration > 40:
        threshold_prediction = "vulnerable"
    else:
        threshold_prediction = "robust"

    # Actual result
    if coverage_drop > 15:
        actual_category = "vulnerable"
    elif coverage_drop > 5:
        actual_category = "moderate"
    else:
        actual_category = "robust"

    threshold_correct = (
        (threshold_prediction == "vulnerable" and coverage_drop > 10) or
        (threshold_prediction == "robust" and coverage_drop <= 15)
    )

    print(f"  SHAP concentration: {concentration:.1f}%")
    print(f"  Threshold prediction (>40% = vulnerable): {threshold_prediction}")
    print(f"  Linear prediction (beta=1.64): {predicted_drop_linear:.1f}% drop")
    print(f"  Actual coverage drop: {coverage_drop:.1f}%")
    print(f"  Actual category: {actual_category}")
    print(f"  Threshold correct: {threshold_correct}")

    return {
        'threshold_prediction': threshold_prediction,
        'predicted_drop_linear': round(predicted_drop_linear, 1),
        'actual_category': actual_category,
        'threshold_correct': threshold_correct,
    }


def main():
    print("=" * 80)
    print("Prospective Holdout Validation: rel-stack badges-class")
    print("Domain: Stack Overflow (vs SALT/SAP ERP training domain)")
    print("=" * 80)

    # 1. Load task
    task = load_task()

    # 2. Prepare features
    X_data, y_data, feature_cols, target_le, num_classes = prepare_features(task)

    # 3. Train LightGBM
    model, probs, train_time = train_lgb(X_data, y_data, num_classes, feature_cols)

    # 4. Compute SHAP concentration
    shap_results = compute_shap_concentration(model, X_data, feature_cols)

    # 5. Run APS conformal prediction
    conformal_results = run_conformal(probs, y_data)

    # 6. Assess diagnostic prediction
    assessment = assess_prediction(
        shap_results['concentration'],
        conformal_results['coverage_drop']
    )

    # 7. Compile and save results
    results = {
        'dataset': 'rel-stack',
        'task': 'badges-class',
        'domain': 'Stack Overflow',
        'num_classes': num_classes,
        'n_features': len(feature_cols),
        'feature_cols': feature_cols,
        'n_train': len(y_data['train']),
        'n_val': len(y_data['val']),
        'n_test': len(y_data['test']),
        'train_time_s': round(train_time, 1),
        'shap': shap_results,
        'conformal': conformal_results,
        'assessment': assessment,
        'salt_reference': {
            'mean_concentration': 39.9,
            'mean_drop': 34.6,
            'mixed_effects_beta1': 1.64,
            'threshold': 40.0,
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {OUTPUT_FILE}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Dataset: rel-stack / badges-class ({num_classes} classes)")
    print(f"  Domain: Stack Overflow (different from SALT/SAP)")
    print(f"  SHAP concentration: {shap_results['concentration']:.1f}%")
    print(f"  Top feature: {shap_results['top_feature']}")
    print(f"  Val coverage: {conformal_results['val_coverage']:.1f}%")
    print(f"  Test coverage: {conformal_results['test_coverage']:.1f}%")
    print(f"  Coverage drop: {conformal_results['coverage_drop']:.1f} pp")
    print(f"  Diagnostic prediction: {assessment['threshold_prediction']}")
    print(f"  Prediction correct: {assessment['threshold_correct']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
