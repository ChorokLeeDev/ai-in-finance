"""
MLP Neural Network Validation for UAI 2026

Tests whether the SHAP concentration diagnostic generalizes to neural networks
(sklearn MLPClassifier) beyond tree-based models.

Uses the SAME data loading, preprocessing, and conformal prediction as the
LightGBM 50-seed ensemble script, replacing LightGBM with an MLP.

Uses sklearn permutation_importance for feature importance concentration.

Usage:
    python run_mlp_validation.py
    python run_mlp_validation.py --num_seeds 5
    python run_mlp_validation.py --tasks sales-shipcond sales-office

Output:
    results/mlp_validation.json
"""

import argparse
import json
import warnings
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.inspection import permutation_importance

warnings.filterwarnings('ignore')

# Configuration
NUM_SEEDS = 10
SEED_START = 42
ALPHA = 0.1  # 90% target coverage
SAMPLE_SIZE = 30000

ALL_TASKS = [
    'sales-shipcond',
    'sales-group',
    'sales-payterms',
    'item-plant',
    'item-shippoint',
    'sales-incoterms',
    'item-incoterms',
    'sales-office',
]

# Item-level tasks are ~50x slower; use fewer seeds
SLOW_TASKS = {'item-plant', 'item-shippoint', 'item-incoterms'}
SLOW_TASK_SEEDS = 3


class ConformalClassifier:
    """Adaptive Prediction Sets (APS) for classification."""

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.quantile = None

    def _compute_scores(self, probs: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        n = len(y_true)
        scores = np.zeros(n)
        for i in range(n):
            sorted_idx = np.argsort(probs[i])[::-1]
            cumsum = 0
            for j, idx in enumerate(sorted_idx):
                cumsum += probs[i][idx]
                if idx == y_true[i]:
                    scores[i] = cumsum
                    break
        return scores

    def calibrate(self, probs: np.ndarray, y_true: np.ndarray):
        scores = self._compute_scores(probs, y_true)
        n = len(scores)
        q_level = min(np.ceil((n + 1) * (1 - self.alpha)) / n, 1.0)
        self.quantile = np.quantile(scores, q_level)
        return self

    def predict_sets(self, probs: np.ndarray) -> List[set]:
        sets = []
        for i in range(len(probs)):
            sorted_idx = np.argsort(probs[i])[::-1]
            pred_set = set()
            cumsum = 0
            for idx in sorted_idx:
                pred_set.add(idx)
                cumsum += probs[i][idx]
                if cumsum >= self.quantile:
                    break
            sets.append(pred_set)
        return sets


def load_and_prepare_data(task, seed: int, sample_size: int = SAMPLE_SIZE):
    """Load rel-salt task data with same preprocessing as LightGBM script."""

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
        entity_df_copy = entity_df_copy.astype(
            {entity_table.pkey_col: table.df[left_entity].dtype}
        )

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

    # Feature engineering (identical to LightGBM script)
    all_data = pd.concat([dfs["train"], dfs["val"], dfs["test"]], ignore_index=True)
    exclude_cols = [target_col, 'CREATIONTIMESTAMP', 'timestamp']
    id_cols = [c for c in all_data.columns if c.endswith('_id') or c.endswith('Id') or c == 'ID']
    exclude_cols.extend(id_cols)
    feature_cols = [c for c in all_data.columns if c not in exclude_cols]

    # Encode categoricals
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

    return X_data, y_data, num_classes, feature_cols


def run_single_seed(task, task_name: str, seed: int, is_slow: bool = False) -> Dict:
    """Run MLP + conformal prediction + permutation importance for one seed."""

    t0 = time.time()

    # Load data
    X_data, y_data, num_classes, feature_cols = load_and_prepare_data(task, seed)

    # Standardize features (critical for neural networks)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_data['train'])
    X_val = scaler.transform(X_data['val'])
    X_test = scaler.transform(X_data['test'])

    # For slow tasks with many classes, use smaller network and fewer iterations
    if is_slow:
        hidden = (64, 32)
        max_iter = 100
    else:
        hidden = (128, 64)
        max_iter = 200

    # Train MLP
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden,
        activation='relu',
        solver='adam',
        alpha=1e-4,
        batch_size=256,
        learning_rate='adaptive',
        learning_rate_init=1e-3,
        max_iter=max_iter,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=seed,
        verbose=False,
    )

    mlp.fit(X_train, y_data['train'])
    train_time = time.time() - t0

    # Predict probabilities
    val_probs = mlp.predict_proba(X_val)
    test_probs = mlp.predict_proba(X_test)

    # Handle class mismatch: MLP might not see all classes in training
    mlp_classes = set(mlp.classes_)
    all_classes = set(range(num_classes))
    if mlp_classes != all_classes:
        full_val_probs = np.zeros((len(X_val), num_classes))
        full_test_probs = np.zeros((len(X_test), num_classes))
        for i, c in enumerate(mlp.classes_):
            full_val_probs[:, c] = val_probs[:, i]
            full_test_probs[:, c] = test_probs[:, i]
        unseen = all_classes - mlp_classes
        if unseen:
            eps = 1e-10
            for c in unseen:
                full_val_probs[:, c] = eps
                full_test_probs[:, c] = eps
            full_val_probs = full_val_probs / full_val_probs.sum(axis=1, keepdims=True)
            full_test_probs = full_test_probs / full_test_probs.sum(axis=1, keepdims=True)
        val_probs = full_val_probs
        test_probs = full_test_probs

    # Conformal prediction: 50/50 cal/eval split on validation
    n_val = len(val_probs)
    n_calib = n_val // 2
    calib_probs, calib_y = val_probs[:n_calib], y_data['val'][:n_calib]
    eval_probs, eval_y = val_probs[n_calib:], y_data['val'][n_calib:]

    conf = ConformalClassifier(alpha=ALPHA)
    conf.calibrate(calib_probs, calib_y)

    val_sets = conf.predict_sets(eval_probs)
    test_sets = conf.predict_sets(test_probs)

    val_cov = sum(1 for i, s in enumerate(val_sets) if eval_y[i] in s) / len(val_sets)
    test_cov = sum(1 for i, s in enumerate(test_sets) if y_data['test'][i] in s) / len(test_sets)

    val_size = np.mean([len(s) for s in val_sets])
    test_size = np.mean([len(s) for s in test_sets])

    # Permutation importance for concentration
    # Smaller subsample for slow tasks
    perm_n = min(500 if is_slow else 2000, len(X_val))
    perm_repeats = 3 if is_slow else 5
    np.random.seed(seed)
    perm_idx = np.random.permutation(len(X_val))[:perm_n]
    X_perm = X_val[perm_idx]
    y_perm = y_data['val'][perm_idx]

    perm_result = permutation_importance(
        mlp, X_perm, y_perm,
        n_repeats=perm_repeats,
        random_state=seed,
        n_jobs=1,
        scoring='accuracy',
    )

    # Compute concentration: top-1 feature importance share
    importances = perm_result.importances_mean
    importances = np.maximum(importances, 0)
    total_imp = importances.sum()
    if total_imp > 0:
        concentration = importances.max() / total_imp
    else:
        concentration = 0.0

    top_feature_idx = int(np.argmax(importances))
    top_feature_name = feature_cols[top_feature_idx] if top_feature_idx < len(feature_cols) else f"feature_{top_feature_idx}"

    elapsed = time.time() - t0

    return {
        'task': task_name,
        'seed': seed,
        'val_coverage': float(val_cov),
        'test_coverage': float(test_cov),
        'coverage_drop_pp': float((val_cov - test_cov) * 100),
        'val_set_size': float(val_size),
        'test_set_size': float(test_size),
        'num_classes': int(num_classes),
        'concentration_pct': float(concentration * 100),
        'top_feature': top_feature_name,
        'top_feature_importance': float(importances.max()),
        'total_importance': float(total_imp),
        'mlp_train_time_s': float(train_time),
        'total_time_s': float(elapsed),
        'mlp_n_iter': int(mlp.n_iter_),
        'hidden_layers': list(hidden) if is_slow else [128, 64],
    }


def main():
    parser = argparse.ArgumentParser(description="MLP validation for SHAP concentration diagnostic")
    parser.add_argument('--tasks', nargs='+', default=ALL_TASKS)
    parser.add_argument('--num_seeds', type=int, default=NUM_SEEDS)
    parser.add_argument('--slow_seeds', type=int, default=SLOW_TASK_SEEDS)
    args = parser.parse_args()

    print(f"{'='*80}")
    print(f"MLP NEURAL NETWORK VALIDATION")
    print(f"Tasks: {len(args.tasks)}, Seeds: {args.num_seeds} (slow tasks: {args.slow_seeds})")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")

    from relbench.tasks import get_task

    output_dir = Path(__file__).resolve().parent.parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "mlp_validation.json"

    all_task_results = []
    task_summaries = []

    for task_idx, task_name in enumerate(args.tasks):
        is_slow = task_name in SLOW_TASKS
        n_seeds = args.slow_seeds if is_slow else args.num_seeds

        print(f"\n{'='*80}")
        print(f"[{task_idx+1}/{len(args.tasks)}] Task: {task_name} ({'SLOW' if is_slow else 'fast'}, {n_seeds} seeds)")
        print(f"{'='*80}")

        task = get_task('rel-salt', task_name, download=False)

        seed_results = []
        for s_idx, seed in enumerate(range(SEED_START, SEED_START + n_seeds)):
            print(f"  Seed {seed} ({s_idx+1}/{n_seeds})...", end=" ", flush=True)
            try:
                result = run_single_seed(task, task_name, seed, is_slow=is_slow)
                seed_results.append(result)
                print(f"done in {result['total_time_s']:.1f}s | "
                      f"val_cov={result['val_coverage']*100:.1f}% "
                      f"test_cov={result['test_coverage']*100:.1f}% "
                      f"drop={result['coverage_drop_pp']:.1f}pp "
                      f"C={result['concentration_pct']:.1f}%")
            except Exception as e:
                print(f"FAILED: {e}")
                import traceback
                traceback.print_exc()

        if not seed_results:
            print(f"  All seeds failed for {task_name}, skipping.")
            continue

        # Aggregate
        drops = [r['coverage_drop_pp'] for r in seed_results]
        concentrations = [r['concentration_pct'] for r in seed_results]
        val_covs = [r['val_coverage'] for r in seed_results]
        test_covs = [r['test_coverage'] for r in seed_results]

        summary = {
            'task': task_name,
            'num_classes': seed_results[0]['num_classes'],
            'num_seeds': len(seed_results),
            'concentration_mean': float(np.mean(concentrations)),
            'concentration_std': float(np.std(concentrations)),
            'coverage_drop_mean_pp': float(np.mean(drops)),
            'coverage_drop_std_pp': float(np.std(drops)),
            'val_coverage_mean': float(np.mean(val_covs)),
            'test_coverage_mean': float(np.mean(test_covs)),
            'top_feature_mode': max(set(r['top_feature'] for r in seed_results),
                                    key=lambda x: sum(1 for r in seed_results if r['top_feature'] == x)),
            'seed_results': seed_results,
        }
        task_summaries.append(summary)
        all_task_results.extend(seed_results)

        print(f"\n  Summary for {task_name}:")
        print(f"    Concentration: {summary['concentration_mean']:.1f} +/- {summary['concentration_std']:.1f}%")
        print(f"    Coverage drop: {summary['coverage_drop_mean_pp']:.1f} +/- {summary['coverage_drop_std_pp']:.1f}pp")
        print(f"    Val coverage:  {summary['val_coverage_mean']*100:.1f}%")
        print(f"    Test coverage: {summary['test_coverage_mean']*100:.1f}%")
        print(f"    Top feature:   {summary['top_feature_mode']}")

        # Save per-task JSON incrementally
        partial = {
            'model': 'MLPClassifier',
            'hidden_layers': '(64,32) for slow tasks, (128,64) for fast tasks',
            'num_seeds_fast': args.num_seeds,
            'num_seeds_slow': args.slow_seeds,
            'alpha': ALPHA,
            'sample_size': SAMPLE_SIZE,
            'task_summaries': task_summaries,
            'timestamp': datetime.now().isoformat(),
        }
        with open(output_file, 'w') as f:
            json.dump(partial, f, indent=2)
        print(f"  Saved partial results to {output_file}")

    # Final correlation analysis
    print(f"\n{'='*80}")
    print("CORRELATION ANALYSIS")
    print(f"{'='*80}")

    if len(task_summaries) >= 3:
        from scipy import stats

        conc_vals = [s['concentration_mean'] for s in task_summaries]
        drop_vals = [s['coverage_drop_mean_pp'] for s in task_summaries]

        rho, p_val = stats.spearmanr(conc_vals, drop_vals)
        tau, tau_p = stats.kendalltau(conc_vals, drop_vals)

        print(f"\nSpearman rho = {rho:.3f}, p = {p_val:.4f}")
        print(f"Kendall tau  = {tau:.3f}, p = {tau_p:.4f}")
        print(f"\nPer-task results:")
        print(f"  {'Task':<18} {'C (%)':>8} {'Drop (pp)':>10} {'Classes':>8}")
        print(f"  {'-'*48}")
        for s in sorted(task_summaries, key=lambda x: x['coverage_drop_mean_pp'], reverse=True):
            print(f"  {s['task']:<18} {s['concentration_mean']:>7.1f}% {s['coverage_drop_mean_pp']:>9.1f} {s['num_classes']:>8}")

        # Threshold analysis at 40%
        predicted_fail = [s['concentration_mean'] > 40 for s in task_summaries]
        actual_fail = [s['coverage_drop_mean_pp'] > 15 for s in task_summaries]
        tp = sum(p and a for p, a in zip(predicted_fail, actual_fail))
        fp = sum(p and not a for p, a in zip(predicted_fail, actual_fail))
        fn = sum(not p and a for p, a in zip(predicted_fail, actual_fail))
        tn = sum(not p and not a for p, a in zip(predicted_fail, actual_fail))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"\nThreshold analysis (C > 40% predicts failure):")
        print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")
        print(f"  Precision={precision:.2f}, Recall={recall:.2f}, F1={f1:.2f}")

        # Also try lower thresholds since MLP might have different concentration scale
        for thresh in [20, 25, 30, 35, 40, 45, 50]:
            pf = [s['concentration_mean'] > thresh for s in task_summaries]
            af = [s['coverage_drop_mean_pp'] > 15 for s in task_summaries]
            tp2 = sum(p and a for p, a in zip(pf, af))
            fp2 = sum(p and not a for p, a in zip(pf, af))
            fn2 = sum(not p and a for p, a in zip(pf, af))
            tn2 = sum(not p and not a for p, a in zip(pf, af))
            p2 = tp2 / (tp2 + fp2) if (tp2 + fp2) > 0 else 0
            r2 = tp2 / (tp2 + fn2) if (tp2 + fn2) > 0 else 0
            f2 = 2 * p2 * r2 / (p2 + r2) if (p2 + r2) > 0 else 0
            print(f"  Threshold {thresh}%: TP={tp2} FP={fp2} FN={fn2} TN={tn2} F1={f2:.2f}")

        # Save final results
        final = {
            'model': 'MLPClassifier',
            'hidden_layers_fast': [128, 64],
            'hidden_layers_slow': [64, 32],
            'num_seeds_fast': args.num_seeds,
            'num_seeds_slow': args.slow_seeds,
            'alpha': ALPHA,
            'sample_size': SAMPLE_SIZE,
            'spearman_rho': float(rho),
            'spearman_p': float(p_val),
            'kendall_tau': float(tau),
            'kendall_p': float(tau_p),
            'threshold_40pct': {
                'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
            },
            'task_summaries': [
                {k: v for k, v in s.items() if k != 'seed_results'}
                for s in task_summaries
            ],
            'seed_level_results': all_task_results,
            'timestamp': datetime.now().isoformat(),
        }
        with open(output_file, 'w') as f:
            json.dump(final, f, indent=2)
        print(f"\nFinal results saved to {output_file}")
    else:
        print("Not enough tasks completed for correlation analysis.")

    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
