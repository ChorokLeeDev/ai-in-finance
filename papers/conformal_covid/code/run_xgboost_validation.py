"""
XGBoost Validation of SHAP Concentration Diagnostic

Tests whether the SHAP concentration -> coverage drop correlation holds for
XGBoost (gradient-boosted trees, structurally similar to LightGBM).

If XGBoost replicates (rho > 0.6), the diagnostic scope becomes "gradient-boosted trees"
rather than "LightGBM-specific". Combined with RF (rho=0.30), this precisely defines scope.

Usage:
    PYTHONPATH=/Users/i767700/Github/ai-in-finance:$PYTHONPATH \
    /usr/bin/python3 -u papers/conformal_covid/code/run_xgboost_validation.py

Output:
    papers/conformal_covid/results/xgboost_validation.json
"""

import json
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import shap
from scipy import stats
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

warnings.filterwarnings('ignore')

# Configuration
ALPHA = 0.1
SAMPLE_SIZE = 30000
SHAP_SAMPLES = 5000
SHAP_SAMPLES_FALLBACK = 2000
SEED = 42

ALL_TASKS = [
    'sales-shipcond', 'sales-group', 'sales-payterms', 'item-plant',
    'item-shippoint', 'sales-incoterms', 'item-incoterms', 'sales-office',
]

LGB_DATA = {
    'sales-shipcond':  {'lgb_concentration': 50.70, 'lgb_coverage_drop': 71.6},
    'sales-group':     {'lgb_concentration': 47.30, 'lgb_coverage_drop': 71.2},
    'sales-payterms':  {'lgb_concentration': 54.18, 'lgb_coverage_drop': 77.1},
    'item-plant':      {'lgb_concentration': 23.90, 'lgb_coverage_drop': 10.6},
    'item-shippoint':  {'lgb_concentration': 48.79, 'lgb_coverage_drop': 18.5},
    'sales-incoterms': {'lgb_concentration': 23.66, 'lgb_coverage_drop': 8.5},
    'item-incoterms':  {'lgb_concentration': 28.93, 'lgb_coverage_drop': 11.3},
    'sales-office':    {'lgb_concentration': 42.65, 'lgb_coverage_drop': 0.1},
}

RESULTS_DIR = Path(__file__).parent.parent / "results"
OUTPUT_FILE = RESULTS_DIR / "xgboost_validation.json"


def load_and_preprocess_data(task, sample_size=SAMPLE_SIZE, seed=SEED):
    """Load task data and preprocess features (same pipeline as RF/LGB)."""
    train_table = task.get_table("train")
    val_table = task.get_table("val")
    test_table = task.get_table("test", mask_input_cols=False)

    dataset = task.dataset
    entity_table = dataset.get_db().table_dict[task.entity_table]
    entity_df = entity_table.df.copy()

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

    if sample_size and sample_size < len(dfs["train"]):
        np.random.seed(seed)
        idx = np.random.permutation(len(dfs["train"]))[:sample_size]
        dfs["train"] = dfs["train"].iloc[idx].copy()

    target_col = task.target_col

    all_data = pd.concat([dfs["train"], dfs["val"], dfs["test"]], ignore_index=True)
    exclude_cols = [target_col, 'CREATIONTIMESTAMP', 'timestamp']
    id_cols = [c for c in all_data.columns
               if c.endswith('_id') or c.endswith('Id') or c == 'ID']
    exclude_cols.extend(id_cols)
    feature_cols = [c for c in all_data.columns if c not in exclude_cols]

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

    target_le = LabelEncoder()
    all_y = np.concatenate([y_data['train'], y_data['val'], y_data['test']])
    target_le.fit(all_y)
    for split in y_data:
        y_data[split] = target_le.transform(y_data[split])

    num_classes = len(target_le.classes_)
    return X_data, y_data, feature_cols, target_le, num_classes


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
        q_level = min(np.ceil((n + 1) * (1 - self.alpha)) / n, 1.0)
        self.quantile = np.quantile(scores, q_level)
        return self

    def predict_sets(self, probs):
        sets = []
        for i in range(len(probs)):
            sorted_idx = np.argsort(probs[i])[::-1]
            pred_set = set()
            cumsum = 0.0
            for idx in sorted_idx:
                pred_set.add(idx)
                cumsum += probs[i][idx]
                if cumsum >= self.quantile:
                    break
            sets.append(pred_set)
        return sets


def compute_xgb_shap_concentration(model, X_val, feature_names, max_samples=SHAP_SAMPLES):
    """Compute SHAP concentration for XGBoost model (native Booster)."""
    if len(X_val) > max_samples:
        np.random.seed(SEED)
        idx = np.random.choice(len(X_val), max_samples, replace=False)
        X_sample = X_val[idx]
    else:
        X_sample = X_val

    print(f"    Computing SHAP on {len(X_sample)} samples...", flush=True)
    t0 = time.time()

    # Use DMatrix for native Booster
    dmatrix = xgb.DMatrix(X_sample, feature_names=feature_names)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(dmatrix)

    # XGBoost multiclass: list of (n_samples, n_features), one per class
    if isinstance(shap_values, list):
        mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    elif shap_values.ndim == 3:
        mean_abs = np.abs(shap_values).mean(axis=(0, 2))
    else:
        mean_abs = np.abs(shap_values).mean(axis=0)

    mean_abs = mean_abs.ravel()
    concentration = float(mean_abs.max() / mean_abs.sum()) * 100.0
    top_idx = int(np.argmax(mean_abs))
    top_feature = feature_names[top_idx] if top_idx < len(feature_names) else f"feature_{top_idx}"

    elapsed = time.time() - t0
    print(f"    SHAP done in {elapsed:.1f}s. Concentration={concentration:.2f}%, top={top_feature}", flush=True)

    return concentration, top_feature, elapsed


def run_single_task(task_name):
    """Run XGBoost validation for a single task."""
    from relbench.tasks import get_task

    print(f"\n{'='*70}", flush=True)
    print(f"Task: {task_name}", flush=True)
    print(f"{'='*70}", flush=True)

    t_start = time.time()

    task = get_task('rel-salt', task_name, download=False)
    X_data, y_data, feature_cols, target_le, num_classes = \
        load_and_preprocess_data(task)

    print(f"  Classes: {num_classes}", flush=True)
    print(f"  Train: {len(X_data['train'])}, Val: {len(X_data['val'])}, Test: {len(X_data['test'])}", flush=True)

    # Train XGBoost (native API to handle non-contiguous class labels from subsampling)
    print(f"  Training XGBoost (n_rounds=500, native API)...", flush=True)
    t_xgb = time.time()

    params = {
        'objective': 'multi:softprob',
        'num_class': num_classes,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'eval_metric': 'mlogloss',
        'seed': SEED,
        'nthread': -1,
        'verbosity': 0,
    }

    dtrain = xgb.DMatrix(X_data['train'], label=y_data['train'], feature_names=feature_cols)
    dval = xgb.DMatrix(X_data['val'], label=y_data['val'], feature_names=feature_cols)
    dtest = xgb.DMatrix(X_data['test'], label=y_data['test'], feature_names=feature_cols)

    model = xgb.train(
        params, dtrain, num_boost_round=500,
        evals=[(dval, 'val')],
        early_stopping_rounds=50,
        verbose_eval=False,
    )
    xgb_time = time.time() - t_xgb
    print(f"  XGBoost trained in {xgb_time:.1f}s", flush=True)

    # Predict probabilities (native API returns full num_class columns)
    val_probs = model.predict(dval)
    test_probs = model.predict(dtest)

    # Split validation 50/50
    np.random.seed(SEED)
    n_val = len(val_probs)
    perm = np.random.permutation(n_val)
    n_calib = n_val // 2

    calib_probs = val_probs[perm[:n_calib]]
    calib_y = y_data['val'][perm[:n_calib]]
    eval_probs = val_probs[perm[n_calib:]]
    eval_y = y_data['val'][perm[n_calib:]]

    # Conformal prediction (APS)
    conf = ConformalClassifier(alpha=ALPHA)
    conf.calibrate(calib_probs, calib_y)

    val_sets = conf.predict_sets(eval_probs)
    test_sets = conf.predict_sets(test_probs)

    val_cov = sum(1 for i, s in enumerate(val_sets) if eval_y[i] in s) / len(val_sets)
    test_cov = sum(1 for i, s in enumerate(test_sets) if y_data['test'][i] in s) / len(test_sets)
    coverage_drop = val_cov - test_cov

    print(f"  Val coverage:  {val_cov*100:.1f}%", flush=True)
    print(f"  Test coverage: {test_cov*100:.1f}%", flush=True)
    print(f"  Coverage drop: {coverage_drop*100:.1f}%", flush=True)

    # SHAP concentration
    shap_samples = SHAP_SAMPLES
    if num_classes > 200:
        shap_samples = SHAP_SAMPLES_FALLBACK
        print(f"  Using reduced SHAP samples ({shap_samples}) for {num_classes}-class task", flush=True)

    concentration, top_feature, shap_time = compute_xgb_shap_concentration(
        model, X_data['val'], feature_cols, max_samples=shap_samples
    )

    total_time = time.time() - t_start

    result = {
        'xgb_concentration': round(concentration, 2),
        'xgb_val_coverage': round(val_cov * 100, 2),
        'xgb_test_coverage': round(test_cov * 100, 2),
        'xgb_coverage_drop': round(coverage_drop * 100, 2),
        'xgb_top_feature': top_feature,
        'num_classes': num_classes,
        'xgb_train_time_s': round(xgb_time, 1),
        'shap_time_s': round(shap_time, 1),
        'total_time_s': round(total_time, 1),
        'lgb_concentration': LGB_DATA[task_name]['lgb_concentration'],
        'lgb_coverage_drop': LGB_DATA[task_name]['lgb_coverage_drop'],
    }

    print(f"\n  RESULT: concentration={concentration:.2f}%, drop={coverage_drop*100:.1f}%", flush=True)
    print(f"  (LGB: concentration={LGB_DATA[task_name]['lgb_concentration']:.2f}%, "
          f"drop={LGB_DATA[task_name]['lgb_coverage_drop']:.1f}%)", flush=True)

    return result


def save_results(task_results, completed_tasks):
    """Save current results to JSON."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output = {
        'tasks': task_results,
        'completed_tasks': completed_tasks,
        'n_tasks_completed': len(completed_tasks),
    }

    if len(completed_tasks) >= 3:
        concs = [task_results[t]['xgb_concentration'] for t in completed_tasks]
        drops = [task_results[t]['xgb_coverage_drop'] for t in completed_tasks]
        rho, p = stats.spearmanr(concs, drops)
        output['xgb_spearman_rho'] = round(float(rho), 3)
        output['xgb_spearman_p'] = round(float(p), 4)
        print(f"\n  Current XGB Spearman rho={rho:.3f}, p={p:.4f} (n={len(completed_tasks)})", flush=True)

    output['lgb_spearman_rho'] = 0.833
    output['lgb_spearman_p'] = 0.010

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"  Saved to {OUTPUT_FILE}", flush=True)


def main():
    print(f"\n{'='*70}", flush=True)
    print("XGBoost Validation of SHAP Concentration Diagnostic", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"Tasks: {len(ALL_TASKS)}", flush=True)
    print(f"Comparing XGBoost vs LightGBM (existing: rho=0.833, p=0.010)", flush=True)
    print(f"{'='*70}\n", flush=True)

    task_results = {}
    completed_tasks = []

    if OUTPUT_FILE.exists():
        try:
            with open(OUTPUT_FILE, 'r') as f:
                existing = json.load(f)
            if 'tasks' in existing:
                task_results = existing['tasks']
                completed_tasks = [t for t in ALL_TASKS if t in task_results]
                print(f"Resuming: found {len(completed_tasks)} completed tasks", flush=True)
        except Exception:
            pass

    t_total = time.time()

    for task_name in ALL_TASKS:
        if task_name in task_results:
            print(f"\nSkipping {task_name} (already completed)", flush=True)
            continue

        try:
            result = run_single_task(task_name)
            if result is not None:
                task_results[task_name] = result
                completed_tasks.append(task_name)
                save_results(task_results, completed_tasks)
        except Exception as e:
            print(f"\nERROR on {task_name}: {e}", flush=True)
            traceback.print_exc()
            task_results[task_name] = {
                'error': str(e),
                'lgb_concentration': LGB_DATA[task_name]['lgb_concentration'],
                'lgb_coverage_drop': LGB_DATA[task_name]['lgb_coverage_drop'],
            }
            save_results(task_results, completed_tasks)

    # Final results
    print(f"\n{'='*70}", flush=True)
    print("FINAL RESULTS", flush=True)
    print(f"{'='*70}", flush=True)

    if len(completed_tasks) >= 3:
        concs = [task_results[t]['xgb_concentration'] for t in completed_tasks]
        drops = [task_results[t]['xgb_coverage_drop'] for t in completed_tasks]
        rho, p = stats.spearmanr(concs, drops)

        lgb_concs = [task_results[t]['lgb_concentration'] for t in completed_tasks]
        lgb_drops = [task_results[t]['lgb_coverage_drop'] for t in completed_tasks]
        lgb_rho, lgb_p = stats.spearmanr(lgb_concs, lgb_drops)

        cross_rho, cross_p = stats.spearmanr(concs, lgb_concs)

        print(f"\nSpearman correlation (n={len(completed_tasks)}):", flush=True)
        print(f"  XGB: rho={rho:.3f}, p={p:.4f}", flush=True)
        print(f"  LGB: rho={lgb_rho:.3f}, p={lgb_p:.4f} (same tasks)", flush=True)
        print(f"  Cross-model concentration: rho={cross_rho:.3f}, p={cross_p:.4f}", flush=True)

        final = {
            'tasks': task_results,
            'completed_tasks': completed_tasks,
            'n_tasks_completed': len(completed_tasks),
            'xgb_spearman_rho': round(float(rho), 3),
            'xgb_spearman_p': round(float(p), 4),
            'lgb_spearman_rho': 0.833,
            'lgb_spearman_p': 0.010,
            'lgb_spearman_rho_same_tasks': round(float(lgb_rho), 3),
            'lgb_spearman_p_same_tasks': round(float(lgb_p), 4),
            'cross_model_concentration_rho': round(float(cross_rho), 3),
            'cross_model_concentration_p': round(float(cross_p), 4),
        }
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(final, f, indent=2)

    # Table
    print(f"\n{'='*70}", flush=True)
    print(f"{'Task':<18} {'LGB Conc':>10} {'XGB Conc':>10} {'LGB Drop':>10} {'XGB Drop':>10}", flush=True)
    print(f"{'-'*70}", flush=True)
    for t in ALL_TASKS:
        if t in task_results and 'xgb_concentration' in task_results[t]:
            r = task_results[t]
            print(f"{t:<18} {r['lgb_concentration']:>9.2f}% {r['xgb_concentration']:>9.2f}% "
                  f"{r['lgb_coverage_drop']:>9.1f}% {r['xgb_coverage_drop']:>9.1f}%", flush=True)

    total_elapsed = time.time() - t_total
    print(f"\nTotal time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
