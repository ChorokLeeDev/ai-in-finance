"""
RAPS (Regularized Adaptive Prediction Sets) Experiment

Compares APS vs RAPS on all 8 SALT tasks to test whether:
1. RAPS changes the vulnerability pattern (coverage drops)
2. SHAP concentration still correlates with RAPS failure
3. The diagnostic is not specific to the APS scoring rule

RAPS adds a regularization penalty: s_RAPS(x,y) = s_APS(x,y) + lambda * max(rank - k_reg, 0)
This penalizes large prediction sets per Angelopoulos et al. (2021).

Usage:
    PYTHONPATH=/Users/i767700/Github/ai-in-finance:$PYTHONPATH \
    /usr/bin/python3 -u papers/conformal_covid/code/run_raps_experiment.py

Output:
    papers/conformal_covid/results/raps_validation.json
"""

import json
import time
import traceback
import warnings
from pathlib import Path
from typing import Dict, List

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

# Configuration
ALPHA = 0.1
SAMPLE_SIZE = 30000
SEED = 42

# RAPS hyperparameters (Angelopoulos et al. 2021 defaults)
LAMBDA_REG = 0.01
K_REG = 5

ALL_TASKS = [
    'sales-shipcond', 'sales-group', 'sales-payterms', 'item-plant',
    'item-shippoint', 'sales-incoterms', 'item-incoterms', 'sales-office',
]

# Existing LightGBM SHAP concentrations for correlation
LGB_CONCENTRATIONS = {
    'sales-shipcond': 50.70, 'sales-group': 47.30, 'sales-payterms': 54.18,
    'item-plant': 23.90, 'item-shippoint': 48.79, 'sales-incoterms': 23.66,
    'item-incoterms': 28.93, 'sales-office': 42.65,
}

RESULTS_DIR = Path(__file__).parent.parent / "results"
OUTPUT_FILE = RESULTS_DIR / "raps_validation.json"


class APSClassifier:
    """Standard Adaptive Prediction Sets."""

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


class RAPSClassifier:
    """Regularized Adaptive Prediction Sets (Angelopoulos et al. 2021)."""

    def __init__(self, alpha=0.1, lambda_reg=0.01, k_reg=5):
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        self.k_reg = k_reg
        self.quantile = None

    def _compute_scores(self, probs, y_true):
        n = len(y_true)
        scores = np.zeros(n)
        for i in range(n):
            sorted_idx = np.argsort(probs[i])[::-1]
            cumsum = 0.0
            for rank, idx in enumerate(sorted_idx):
                cumsum += probs[i][idx]
                reg_penalty = self.lambda_reg * max(rank + 1 - self.k_reg, 0)
                if idx == y_true[i]:
                    scores[i] = cumsum + reg_penalty
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
            for rank, idx in enumerate(sorted_idx):
                pred_set.add(idx)
                cumsum += probs[i][idx]
                reg_penalty = self.lambda_reg * max(rank + 1 - self.k_reg, 0)
                if cumsum + reg_penalty >= self.quantile:
                    break
            sets.append(pred_set)
        return sets


def load_and_preprocess_data(task, seed=SEED):
    """Load task data and preprocess features (same pipeline as 50-seed ensemble)."""
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

    # Subsample training
    if SAMPLE_SIZE < len(dfs["train"]):
        np.random.seed(seed)
        idx = np.random.permutation(len(dfs["train"]))[:SAMPLE_SIZE]
        dfs["train"] = dfs["train"].iloc[idx].copy()

    target_col = task.target_col

    all_data = pd.concat([dfs["train"], dfs["val"], dfs["test"]], ignore_index=True)
    exclude_cols = [target_col, 'CREATIONTIMESTAMP', 'timestamp']
    id_cols = [c for c in all_data.columns if c.endswith('_id') or c.endswith('Id') or c == 'ID']
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
    return X_data, y_data, feature_cols, num_classes


def run_single_task(task_name):
    """Run APS vs RAPS comparison for a single task."""
    from relbench.tasks import get_task

    print(f"\n{'='*70}", flush=True)
    print(f"Task: {task_name}", flush=True)
    print(f"{'='*70}", flush=True)

    t_start = time.time()

    task = get_task('rel-salt', task_name, download=False)
    X_data, y_data, feature_cols, num_classes = load_and_preprocess_data(task)

    print(f"  Classes: {num_classes}", flush=True)
    print(f"  Train: {len(X_data['train'])}, Val: {len(X_data['val'])}, Test: {len(X_data['test'])}", flush=True)

    # Train LightGBM (same as original pipeline)
    print(f"  Training LightGBM...", flush=True)
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
        'seed': SEED,
        'n_jobs': -1,
    }

    train_ds = lgb.Dataset(X_data['train'], label=y_data['train'])
    val_ds = lgb.Dataset(X_data['val'], label=y_data['val'], reference=train_ds)
    model = lgb.train(
        params, train_ds, num_boost_round=500,
        valid_sets=[val_ds],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )

    val_probs = model.predict(X_data['val'])
    test_probs = model.predict(X_data['test'])

    # Split validation 50/50 for calibration
    np.random.seed(SEED)
    n_val = len(val_probs)
    perm = np.random.permutation(n_val)
    n_calib = n_val // 2

    calib_probs = val_probs[perm[:n_calib]]
    calib_y = y_data['val'][perm[:n_calib]]
    eval_probs = val_probs[perm[n_calib:]]
    eval_y = y_data['val'][perm[n_calib:]]

    # --- APS ---
    print("  Running APS...", flush=True)
    aps = APSClassifier(alpha=ALPHA)
    aps.calibrate(calib_probs, calib_y)

    aps_val_sets = aps.predict_sets(eval_probs)
    aps_test_sets = aps.predict_sets(test_probs)

    aps_val_cov = sum(1 for i, s in enumerate(aps_val_sets) if eval_y[i] in s) / len(aps_val_sets)
    aps_test_cov = sum(1 for i, s in enumerate(aps_test_sets) if y_data['test'][i] in s) / len(aps_test_sets)
    aps_val_size = np.mean([len(s) for s in aps_val_sets])
    aps_test_size = np.mean([len(s) for s in aps_test_sets])

    print(f"  APS: val={aps_val_cov*100:.1f}%, test={aps_test_cov*100:.1f}%, "
          f"drop={100*(aps_val_cov-aps_test_cov):.1f}%, "
          f"val_size={aps_val_size:.1f}, test_size={aps_test_size:.1f}", flush=True)

    # --- RAPS (default params) ---
    print(f"  Running RAPS (lambda={LAMBDA_REG}, k={K_REG})...", flush=True)
    raps = RAPSClassifier(alpha=ALPHA, lambda_reg=LAMBDA_REG, k_reg=K_REG)
    raps.calibrate(calib_probs, calib_y)

    raps_val_sets = raps.predict_sets(eval_probs)
    raps_test_sets = raps.predict_sets(test_probs)

    raps_val_cov = sum(1 for i, s in enumerate(raps_val_sets) if eval_y[i] in s) / len(raps_val_sets)
    raps_test_cov = sum(1 for i, s in enumerate(raps_test_sets) if y_data['test'][i] in s) / len(raps_test_sets)
    raps_val_size = np.mean([len(s) for s in raps_val_sets])
    raps_test_size = np.mean([len(s) for s in raps_test_sets])

    print(f"  RAPS: val={raps_val_cov*100:.1f}%, test={raps_test_cov*100:.1f}%, "
          f"drop={100*(raps_val_cov-raps_test_cov):.1f}%, "
          f"val_size={raps_val_size:.1f}, test_size={raps_test_size:.1f}", flush=True)

    # --- RAPS sensitivity (lambda=0.1, more aggressive) ---
    print(f"  Running RAPS (lambda=0.1, k={K_REG})...", flush=True)
    raps_strong = RAPSClassifier(alpha=ALPHA, lambda_reg=0.1, k_reg=K_REG)
    raps_strong.calibrate(calib_probs, calib_y)

    raps_s_val_sets = raps_strong.predict_sets(eval_probs)
    raps_s_test_sets = raps_strong.predict_sets(test_probs)

    raps_s_val_cov = sum(1 for i, s in enumerate(raps_s_val_sets) if eval_y[i] in s) / len(raps_s_val_sets)
    raps_s_test_cov = sum(1 for i, s in enumerate(raps_s_test_sets) if y_data['test'][i] in s) / len(raps_s_test_sets)
    raps_s_val_size = np.mean([len(s) for s in raps_s_val_sets])
    raps_s_test_size = np.mean([len(s) for s in raps_s_test_sets])

    elapsed = time.time() - t_start

    result = {
        'num_classes': num_classes,
        'concentration': LGB_CONCENTRATIONS[task_name],
        # APS results
        'aps_val_coverage': round(aps_val_cov * 100, 2),
        'aps_test_coverage': round(aps_test_cov * 100, 2),
        'aps_coverage_drop': round((aps_val_cov - aps_test_cov) * 100, 2),
        'aps_val_set_size': round(aps_val_size, 2),
        'aps_test_set_size': round(aps_test_size, 2),
        # RAPS (lambda=0.01)
        'raps_val_coverage': round(raps_val_cov * 100, 2),
        'raps_test_coverage': round(raps_test_cov * 100, 2),
        'raps_coverage_drop': round((raps_val_cov - raps_test_cov) * 100, 2),
        'raps_val_set_size': round(raps_val_size, 2),
        'raps_test_set_size': round(raps_test_size, 2),
        # RAPS (lambda=0.1)
        'raps_strong_val_coverage': round(raps_s_val_cov * 100, 2),
        'raps_strong_test_coverage': round(raps_s_test_cov * 100, 2),
        'raps_strong_coverage_drop': round((raps_s_val_cov - raps_s_test_cov) * 100, 2),
        'raps_strong_val_set_size': round(raps_s_val_size, 2),
        'raps_strong_test_set_size': round(raps_s_test_size, 2),
        'elapsed_s': round(elapsed, 1),
    }

    return result


def main():
    print(f"\n{'='*70}", flush=True)
    print("APS vs RAPS Comparison - All 8 SALT Tasks", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"RAPS params: lambda={LAMBDA_REG}, k_reg={K_REG}", flush=True)
    print(f"Also testing lambda=0.1 (aggressive)", flush=True)
    print(f"{'='*70}\n", flush=True)

    task_results = {}
    completed_tasks = []

    # Resume from partial results
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
            task_results[task_name] = result
            completed_tasks.append(task_name)

            # Save incrementally
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            with open(OUTPUT_FILE, 'w') as f:
                json.dump({'tasks': task_results, 'completed_tasks': completed_tasks,
                           'n_completed': len(completed_tasks)}, f, indent=2)
            print(f"  Saved ({len(completed_tasks)}/{len(ALL_TASKS)})", flush=True)

        except Exception as e:
            print(f"\nERROR on {task_name}: {e}", flush=True)
            traceback.print_exc()

    # Final analysis
    print(f"\n{'='*70}", flush=True)
    print("FINAL RESULTS", flush=True)
    print(f"{'='*70}", flush=True)

    if len(completed_tasks) >= 3:
        concs = [task_results[t]['concentration'] for t in completed_tasks]
        aps_drops = [task_results[t]['aps_coverage_drop'] for t in completed_tasks]
        raps_drops = [task_results[t]['raps_coverage_drop'] for t in completed_tasks]
        raps_s_drops = [task_results[t]['raps_strong_coverage_drop'] for t in completed_tasks]

        # Correlations
        aps_rho, aps_p = stats.spearmanr(concs, aps_drops)
        raps_rho, raps_p = stats.spearmanr(concs, raps_drops)
        raps_s_rho, raps_s_p = stats.spearmanr(concs, raps_s_drops)

        # APS-RAPS drop correlation
        aps_raps_rho, aps_raps_p = stats.spearmanr(aps_drops, raps_drops)

        print(f"\nConcentration vs Coverage Drop (n={len(completed_tasks)}):", flush=True)
        print(f"  APS:             rho={aps_rho:.3f}, p={aps_p:.4f}", flush=True)
        print(f"  RAPS (l=0.01):   rho={raps_rho:.3f}, p={raps_p:.4f}", flush=True)
        print(f"  RAPS (l=0.1):    rho={raps_s_rho:.3f}, p={raps_s_p:.4f}", flush=True)
        print(f"\nAPS vs RAPS drop correlation: rho={aps_raps_rho:.3f}, p={aps_raps_p:.4f}", flush=True)

        # Save final
        final = {
            'tasks': task_results,
            'completed_tasks': completed_tasks,
            'n_completed': len(completed_tasks),
            'raps_params': {'lambda_reg': LAMBDA_REG, 'k_reg': K_REG},
            'aps_spearman_rho': round(float(aps_rho), 3),
            'aps_spearman_p': round(float(aps_p), 4),
            'raps_spearman_rho': round(float(raps_rho), 3),
            'raps_spearman_p': round(float(raps_p), 4),
            'raps_strong_spearman_rho': round(float(raps_s_rho), 3),
            'raps_strong_spearman_p': round(float(raps_s_p), 4),
            'aps_vs_raps_drop_rho': round(float(aps_raps_rho), 3),
            'aps_vs_raps_drop_p': round(float(aps_raps_p), 4),
        }
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(final, f, indent=2)

    # Print comparison table
    print(f"\n{'='*70}", flush=True)
    print(f"{'Task':<18} {'Conc':>6} {'APS Drop':>10} {'RAPS Drop':>10} {'RAPS.1 Drop':>12} {'APS Size':>9} {'RAPS Size':>10}", flush=True)
    print(f"{'-'*75}", flush=True)
    for t in ALL_TASKS:
        if t in task_results:
            r = task_results[t]
            print(f"{t:<18} {r['concentration']:>5.1f}% {r['aps_coverage_drop']:>9.1f}% "
                  f"{r['raps_coverage_drop']:>9.1f}% {r['raps_strong_coverage_drop']:>11.1f}% "
                  f"{r['aps_test_set_size']:>8.1f} {r['raps_test_set_size']:>9.1f}", flush=True)

    total_time = time.time() - t_total
    print(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f} min)", flush=True)
    print(f"Results saved to: {OUTPUT_FILE}", flush=True)


if __name__ == "__main__":
    main()
