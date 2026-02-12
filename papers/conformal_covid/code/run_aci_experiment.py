"""
ACI (Adaptive Conformal Inference) Multi-Seed Experiment

Tests whether ACI (Gibbs & Candès, 2021) improves coverage under severe
distribution shift. Extends the existing 50-seed ensemble framework with
online quantile updates.

Usage:
    python code/run_aci_experiment.py --task sales-shipcond --num_seeds 10
    python code/run_aci_experiment.py --task sales-shipcond --num_seeds 10 --gammas 0.001 0.01 0.05

Output:
    results/aci/aci_{task}_{num_seeds}seeds.json
"""

import argparse
import json
import warnings
from datetime import datetime
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

# Constants (match run_50seed_ensemble.py)
ALPHA = 0.1
SAMPLE_SIZE = 30000
SEED_START = 42


class ConformalClassifier:
    """Adaptive Prediction Sets (APS) for classification — standard conformal."""

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.quantile = None

    def _compute_scores(self, probs: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Compute conformity scores (cumulative probability until true label)."""
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
        """Calibrate conformal predictor on calibration set."""
        scores = self._compute_scores(probs, y_true)
        n = len(scores)
        q_level = min(np.ceil((n + 1) * (1 - self.alpha)) / n, 1.0)
        self.quantile = np.quantile(scores, q_level)
        return self

    def predict_sets(self, probs: np.ndarray) -> List[set]:
        """Predict conformal sets."""
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

    def evaluate_coverage(self, probs: np.ndarray, y_true: np.ndarray) -> float:
        """Evaluate coverage on a dataset."""
        sets = self.predict_sets(probs)
        return sum(1 for i, s in enumerate(sets) if y_true[i] in s) / len(sets)


class AdaptiveConformalClassifier:
    """
    Adaptive Conformal Inference (ACI) — Gibbs & Candès (2021).

    Updates the target quantile online based on observed coverage errors.
    alpha_{t+1} = alpha_t + gamma * (alpha - err_t)
    where err_t = 1 if y_t not in C_t, else 0.
    """

    def __init__(self, alpha: float = 0.1, gamma: float = 0.01):
        self.alpha = alpha  # target miscoverage rate
        self.gamma = gamma  # learning rate
        self.alpha_t = alpha  # time-varying alpha

    def _compute_score(self, probs: np.ndarray, y_true: int) -> float:
        """Compute conformity score for a single sample."""
        sorted_idx = np.argsort(probs)[::-1]
        cumsum = 0
        for idx in sorted_idx:
            cumsum += probs[idx]
            if idx == y_true:
                return cumsum
        return 1.0

    def _predict_set(self, probs: np.ndarray, quantile: float) -> set:
        """Predict conformal set for a single sample."""
        sorted_idx = np.argsort(probs)[::-1]
        pred_set = set()
        cumsum = 0
        for idx in sorted_idx:
            pred_set.add(idx)
            cumsum += probs[idx]
            if cumsum >= quantile:
                break
        return pred_set

    def run_online(
        self,
        calib_probs: np.ndarray,
        calib_y: np.ndarray,
        test_probs: np.ndarray,
        test_y: np.ndarray,
    ) -> Dict:
        """
        Run ACI: calibrate on calibration set, then update online on test set.

        Returns:
            dict with coverage, alpha trajectory, set sizes
        """
        # Initial calibration (same as standard conformal)
        n_calib = len(calib_y)
        calib_scores = np.array([
            self._compute_score(calib_probs[i], calib_y[i])
            for i in range(n_calib)
        ])

        # Reset alpha_t
        self.alpha_t = self.alpha

        # Online evaluation on test set
        coverages = []
        alpha_trajectory = []
        set_sizes = []

        for t in range(len(test_y)):
            # Current quantile from calibration scores + current alpha_t
            q_level = min(
                np.ceil((n_calib + 1) * (1 - self.alpha_t)) / n_calib, 1.0
            )
            q_level = max(q_level, 0.0)
            current_quantile = np.quantile(calib_scores, min(q_level, 1.0))

            # Predict set
            pred_set = self._predict_set(test_probs[t], current_quantile)
            set_sizes.append(len(pred_set))

            # Check coverage
            covered = test_y[t] in pred_set
            coverages.append(int(covered))

            # ACI update
            err_t = 1 - int(covered)
            self.alpha_t = self.alpha_t + self.gamma * (self.alpha - err_t)
            self.alpha_t = np.clip(self.alpha_t, 0.001, 0.999)
            alpha_trajectory.append(self.alpha_t)

        return {
            'coverage': np.mean(coverages),
            'coverages': coverages,
            'alpha_trajectory': alpha_trajectory,
            'mean_set_size': np.mean(set_sizes),
            'final_alpha': self.alpha_t,
        }


def prepare_data(task, task_name: str, seed: int, sample_size: int = SAMPLE_SIZE):
    """Prepare data for a single seed (same as run_50seed_ensemble.py)."""
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
    np.random.seed(seed)
    if sample_size and sample_size < len(dfs["train"]):
        idx = np.random.permutation(len(dfs["train"]))[:sample_size]
        dfs["train"] = dfs["train"].iloc[idx].copy()

    target_col = task.target_col

    # Feature engineering
    all_data = pd.concat([dfs["train"], dfs["val"], dfs["test"]], ignore_index=True)
    exclude_cols = [target_col, 'CREATIONTIMESTAMP', 'timestamp']
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

    return X_data, y_data, feature_cols, num_classes


def train_model(X_data, y_data, seed: int):
    """Train LightGBM model (same hyperparameters as run_50seed_ensemble.py)."""
    num_classes = len(np.unique(np.concatenate([y_data['train'], y_data['val'], y_data['test']])))

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
        'n_jobs': 1,
    }

    train_data = lgb.Dataset(X_data['train'], label=y_data['train'])
    val_data = lgb.Dataset(X_data['val'], label=y_data['val'], reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    return model


def run_single_seed_aci(task, task_name: str, seed: int, gammas: List[float]) -> Dict:
    """Run standard conformal + ACI for a single seed."""

    X_data, y_data, feature_cols, num_classes = prepare_data(task, task_name, seed)
    model = train_model(X_data, y_data, seed)

    # Predict
    val_probs = model.predict(X_data['val'])
    test_probs = model.predict(X_data['test'])

    # Split validation for calibration (same as 50-seed)
    n_val = len(val_probs)
    n_calib = n_val // 2
    calib_probs = val_probs[:n_calib]
    calib_y = y_data['val'][:n_calib]
    eval_probs = val_probs[n_calib:]
    eval_y = y_data['val'][n_calib:]

    results = {'seed': seed, 'task': task_name, 'num_classes': num_classes}

    # Standard conformal
    conf = ConformalClassifier(alpha=ALPHA)
    conf.calibrate(calib_probs, calib_y)
    results['standard'] = {
        'val_coverage': conf.evaluate_coverage(eval_probs, eval_y),
        'test_coverage': conf.evaluate_coverage(test_probs, y_data['test']),
    }

    # ACI for each gamma
    for gamma in gammas:
        aci = AdaptiveConformalClassifier(alpha=ALPHA, gamma=gamma)
        aci_result = aci.run_online(calib_probs, calib_y, test_probs, y_data['test'])
        results[f'aci_{gamma}'] = {
            'test_coverage': aci_result['coverage'],
            'mean_set_size': aci_result['mean_set_size'],
            'final_alpha': aci_result['final_alpha'],
        }

    return results


def run_seed_wrapper(args):
    """Wrapper for parallel execution."""
    task, task_name, seed, gammas = args
    try:
        result = run_single_seed_aci(task, task_name, seed, gammas)
        print(f"  Seed {seed}: Standard={result['standard']['test_coverage']:.3f}, "
              f"ACI(γ=0.01)={result.get('aci_0.01', {}).get('test_coverage', 'N/A')}")
        return result
    except Exception as e:
        print(f"  ERROR seed {seed}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description='ACI Multi-Seed Experiment')
    parser.add_argument('--task', type=str, default='sales-shipcond',
                        help='Task name (default: sales-shipcond)')
    parser.add_argument('--num_seeds', type=int, default=10,
                        help='Number of seeds (default: 10)')
    parser.add_argument('--gammas', type=float, nargs='+',
                        default=[0.001, 0.01, 0.05],
                        help='ACI learning rates (default: 0.001 0.01 0.05)')
    parser.add_argument('--n_workers', type=int, default=0,
                        help='Number of parallel workers (0=auto)')
    parser.add_argument('--output_dir', type=str, default='results/aci',
                        help='Output directory')
    args = parser.parse_args()

    if args.n_workers == 0:
        args.n_workers = min(cpu_count(), args.num_seeds)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"ACI Experiment: {args.task}")
    print(f"Seeds: {args.num_seeds}, Gammas: {args.gammas}, Workers: {args.n_workers}")
    print("=" * 80)

    from relbench.tasks import get_task
    task = get_task('rel-salt', args.task, download=False)

    seeds = list(range(SEED_START, SEED_START + args.num_seeds))
    task_args = [(task, args.task, seed, args.gammas) for seed in seeds]

    if args.n_workers > 1:
        with Pool(args.n_workers) as pool:
            all_results = pool.map(run_seed_wrapper, task_args)
    else:
        all_results = [run_seed_wrapper(a) for a in task_args]

    # Filter failures
    all_results = [r for r in all_results if r is not None]
    print(f"\nCompleted: {len(all_results)}/{args.num_seeds} seeds")

    # Aggregate
    summary = {
        'task': args.task,
        'num_seeds': len(all_results),
        'gammas': args.gammas,
        'alpha': ALPHA,
        'timestamp': datetime.now().isoformat(),
    }

    # Standard conformal
    std_coverages = [r['standard']['test_coverage'] for r in all_results]
    summary['standard'] = {
        'test_coverage_mean': float(np.mean(std_coverages)),
        'test_coverage_std': float(np.std(std_coverages)),
        'test_coverage_median': float(np.median(std_coverages)),
        'test_coverages': [float(c) for c in std_coverages],
    }

    # ACI per gamma
    for gamma in args.gammas:
        key = f'aci_{gamma}'
        coverages = [r[key]['test_coverage'] for r in all_results if key in r]
        set_sizes = [r[key]['mean_set_size'] for r in all_results if key in r]
        summary[key] = {
            'test_coverage_mean': float(np.mean(coverages)),
            'test_coverage_std': float(np.std(coverages)),
            'test_coverage_median': float(np.median(coverages)),
            'test_coverages': [float(c) for c in coverages],
            'mean_set_size': float(np.mean(set_sizes)),
        }

    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Method':<25} {'Mean Coverage':>15} {'Std':>10} {'Median':>10}")
    print("-" * 60)

    s = summary['standard']
    print(f"{'Standard Conformal':<25} {s['test_coverage_mean']:>14.1%} "
          f"{s['test_coverage_std']:>9.1%} {s['test_coverage_median']:>9.1%}")

    for gamma in args.gammas:
        key = f'aci_{gamma}'
        a = summary[key]
        print(f"{'ACI (γ=' + str(gamma) + ')':<25} {a['test_coverage_mean']:>14.1%} "
              f"{a['test_coverage_std']:>9.1%} {a['test_coverage_median']:>9.1%}")

    # Save
    output_file = output_dir / f"aci_{args.task}_{args.num_seeds}seeds.json"
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {output_file}")

    # Generate LaTeX table snippet
    print("\n--- LaTeX Table Snippet ---")
    print("\\begin{tabular}{lcc}")
    print("\\toprule")
    print("Method & Test Coverage & Std \\\\")
    print("\\midrule")
    print(f"Standard Conformal & {s['test_coverage_mean']:.1%} & $\\pm${s['test_coverage_std']:.1%} \\\\")
    for gamma in args.gammas:
        key = f'aci_{gamma}'
        a = summary[key]
        print(f"ACI ($\\gamma$={gamma}) & {a['test_coverage_mean']:.1%} & $\\pm${a['test_coverage_std']:.1%} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")


if __name__ == "__main__":
    main()
