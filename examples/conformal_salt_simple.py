"""
Simplified Conformal Prediction for rel-salt Dataset (COVID Distribution Shift)

Uses native LightGBM without torch_frame for simplicity.
Analyzes coverage degradation during COVID period.
"""

import argparse
import os
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')


class ConformalClassifier:
    """
    Conformal Prediction for multi-class classification.
    Uses Adaptive Prediction Sets (APS) method.
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.quantile = None

    def _compute_scores(self, probs: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Compute conformity scores: cumulative probability to include true class."""
        n_samples = len(y_true)
        scores = np.zeros(n_samples)

        for i in range(n_samples):
            sorted_indices = np.argsort(probs[i])[::-1]
            sorted_probs = probs[i][sorted_indices]
            true_class = y_true[i]
            cumsum = 0
            for j, idx in enumerate(sorted_indices):
                cumsum += sorted_probs[j]
                if idx == true_class:
                    scores[i] = cumsum
                    break

        return scores

    def calibrate(self, probs: np.ndarray, y_true: np.ndarray) -> 'ConformalClassifier':
        """Calibrate on calibration set."""
        scores = self._compute_scores(probs, y_true)
        n = len(scores)
        quantile_level = min(np.ceil((n + 1) * (1 - self.alpha)) / n, 1.0)
        self.quantile = np.quantile(scores, quantile_level)

        print(f"\nConformal Calibration:")
        print(f"  Target coverage: {100*(1-self.alpha):.1f}%")
        print(f"  Calibration size: {n}")
        print(f"  Quantile: {self.quantile:.4f}")
        return self

    def predict_sets(self, probs: np.ndarray) -> List[set]:
        """Generate prediction sets."""
        if self.quantile is None:
            raise ValueError("Must calibrate first")

        prediction_sets = []
        for i in range(len(probs)):
            sorted_indices = np.argsort(probs[i])[::-1]
            sorted_probs = probs[i][sorted_indices]
            pred_set = set()
            cumsum = 0
            for j, idx in enumerate(sorted_indices):
                pred_set.add(idx)
                cumsum += sorted_probs[j]
                if cumsum >= self.quantile:
                    break
            prediction_sets.append(pred_set)
        return prediction_sets


def compute_coverage_metrics(prediction_sets: List[set], y_true: np.ndarray, alpha: float = 0.1) -> Dict:
    """Compute coverage metrics."""
    n = len(y_true)
    covered = sum(1 for i, ps in enumerate(prediction_sets) if y_true[i] in ps)
    coverage = covered / n
    set_sizes = [len(ps) for ps in prediction_sets]

    return {
        'coverage': coverage,
        'target_coverage': 1 - alpha,
        'coverage_gap': coverage - (1 - alpha),
        'avg_set_size': np.mean(set_sizes),
        'min_set_size': min(set_sizes),
        'max_set_size': max(set_sizes),
    }


def prepare_data(task, sample_size: int = None, seed: int = 42):
    """Prepare data for LightGBM training."""
    from relbench.tasks import get_task

    train_table = task.get_table("train")
    val_table = task.get_table("val")
    test_table = task.get_table("test", mask_input_cols=False)

    # Get entity table
    dataset = task.dataset
    entity_table = dataset.get_db().table_dict[task.entity_table]
    entity_df = entity_table.df.copy()

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

    # Subsample training
    if sample_size and sample_size < len(dfs["train"]):
        np.random.seed(seed)
        idx = np.random.permutation(len(dfs["train"]))[:sample_size]
        dfs["train"] = dfs["train"].iloc[idx].copy()

    return dfs, task.target_col, task.num_classes


def preprocess_features(dfs: Dict[str, pd.DataFrame], target_col: str):
    """Preprocess features for LightGBM."""
    # Combine all data to fit encoders
    all_data = pd.concat([dfs["train"], dfs["val"], dfs["test"]], ignore_index=True)

    # Identify feature columns (exclude target, timestamps, IDs)
    exclude_cols = [target_col, 'CREATIONTIMESTAMP', 'timestamp']
    id_cols = [c for c in all_data.columns if c.endswith('_id') or c.endswith('Id') or c == 'ID']
    exclude_cols.extend(id_cols)

    feature_cols = [c for c in all_data.columns if c not in exclude_cols]

    # Encode categorical columns
    label_encoders = {}
    for col in feature_cols:
        if all_data[col].dtype == 'object' or all_data[col].dtype.name == 'category':
            le = LabelEncoder()
            all_data[col] = all_data[col].astype(str).fillna('__MISSING__')
            le.fit(all_data[col])
            label_encoders[col] = le

    # Process each split
    X_train, X_val, X_test = [], [], []
    y_train, y_val, y_test = [], [], []

    for split, df in dfs.items():
        X = df[feature_cols].copy()

        # Encode categoricals
        for col, le in label_encoders.items():
            X[col] = X[col].astype(str).fillna('__MISSING__')
            # Handle unseen labels
            X[col] = X[col].apply(lambda x: x if x in le.classes_ else '__MISSING__')
            if '__MISSING__' not in le.classes_:
                le.classes_ = np.append(le.classes_, '__MISSING__')
            X[col] = le.transform(X[col])

        # Fill all NaNs and convert to float
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(-999)

        y = df[target_col].values

        # Convert to float array
        X_arr = X.values.astype(np.float32)

        if split == "train":
            X_train, y_train = X_arr, y
        elif split == "val":
            X_val, y_val = X_arr, y
        else:
            X_test, y_test = X_arr, y

    # Encode target
    target_le = LabelEncoder()
    target_le.fit(np.concatenate([y_train, y_val, y_test]))
    y_train = target_le.transform(y_train)
    y_val = target_le.transform(y_val)
    y_test = target_le.transform(y_test)

    return X_train, y_train, X_val, y_val, X_test, y_test, len(target_le.classes_)


def train_lightgbm(X_train, y_train, X_val, y_val, num_classes: int, seed: int = 42):
    """Train LightGBM model."""
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
        valid_sets=[train_data, val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )

    return model


def run_analysis(args):
    """Main analysis."""
    print(f"\n{'='*70}")
    print(f"Conformal Prediction: {args.dataset} / {args.task}")
    print(f"COVID Distribution Shift Analysis")
    print(f"{'='*70}\n")

    from relbench.tasks import get_task

    task = get_task(args.dataset, args.task, download=False)
    dataset = task.dataset

    print(f"Temporal structure:")
    print(f"  Val timestamp (COVID onset): {dataset.val_timestamp}")
    print(f"  Test timestamp (COVID mid): {dataset.test_timestamp}")
    print(f"  Num classes: {task.num_classes}")

    results_dir = Path(f"results/conformal/{args.dataset}/{args.task}")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Prepare data
    print("\nPreparing data...")
    dfs, target_col, num_classes = prepare_data(task, args.sample_size, args.seed)
    print(f"  Train: {len(dfs['train'])}, Val: {len(dfs['val'])}, Test: {len(dfs['test'])}")

    print("\nPreprocessing features...")
    X_train, y_train, X_val, y_val, X_test, y_test, num_classes = preprocess_features(dfs, target_col)
    print(f"  Features: {X_train.shape[1]}, Classes: {num_classes}")

    # Train ensemble
    all_val_probs = []
    all_test_probs = []

    for i, seed in enumerate(range(args.seed, args.seed + args.num_seeds)):
        print(f"\n--- Training model {i+1}/{args.num_seeds} (seed={seed}) ---")
        model = train_lightgbm(X_train, y_train, X_val, y_val, num_classes, seed)
        val_probs = model.predict(X_val)
        test_probs = model.predict(X_test)
        all_val_probs.append(val_probs)
        all_test_probs.append(test_probs)

    # Ensemble average
    val_probs = np.mean(all_val_probs, axis=0)
    test_probs = np.mean(all_test_probs, axis=0)

    print(f"\nEnsemble: {args.num_seeds} models")

    # Split validation for calibration
    n_val = len(val_probs)
    n_calib = n_val // 2

    calib_probs = val_probs[:n_calib]
    calib_labels = y_val[:n_calib]
    eval_probs = val_probs[n_calib:]
    eval_labels = y_val[n_calib:]

    print(f"Calibration: {n_calib}, Evaluation: {n_val - n_calib}")

    # Calibrate
    conformal = ConformalClassifier(alpha=args.alpha)
    conformal.calibrate(calib_probs, calib_labels)

    # Evaluate
    print(f"\n{'='*70}")
    print("COVERAGE ANALYSIS")
    print("="*70)

    val_sets = conformal.predict_sets(eval_probs)
    val_metrics = compute_coverage_metrics(val_sets, eval_labels, args.alpha)

    print(f"\nValidation (COVID onset, Feb-Jul 2020):")
    print(f"  Coverage: {val_metrics['coverage']*100:.2f}% (target: {(1-args.alpha)*100:.0f}%)")
    print(f"  Gap: {val_metrics['coverage_gap']*100:+.2f}%")
    print(f"  Avg set size: {val_metrics['avg_set_size']:.2f}")

    test_sets = conformal.predict_sets(test_probs)
    test_metrics = compute_coverage_metrics(test_sets, y_test, args.alpha)

    print(f"\nTest (COVID peak, Jul 2020+):")
    print(f"  Coverage: {test_metrics['coverage']*100:.2f}% (target: {(1-args.alpha)*100:.0f}%)")
    print(f"  Gap: {test_metrics['coverage_gap']*100:+.2f}%")
    print(f"  Avg set size: {test_metrics['avg_set_size']:.2f}")

    # Analysis
    print(f"\n{'='*70}")
    print("COVID DISTRIBUTION SHIFT ANALYSIS")
    print("="*70)

    coverage_drop = val_metrics['coverage'] - test_metrics['coverage']
    print(f"\nCoverage drop (Val → Test): {coverage_drop*100:.2f}%")

    if coverage_drop > 0.05:
        print(f"\n[!] SIGNIFICANT COVERAGE DROP: {coverage_drop*100:.1f}%")
        print(f"    Conformal guarantee breaks during COVID distribution shift.")
    elif coverage_drop > 0.02:
        print(f"\n[~] MODERATE COVERAGE DROP: {coverage_drop*100:.1f}%")
        print(f"    Some degradation during COVID, but still reasonable.")
    else:
        print(f"\n[✓] COVERAGE MAINTAINED")
        print(f"    Model is robust to COVID distribution shift.")

    # Visualization
    plot_results(val_metrics, test_metrics, args, results_dir)

    # Save results
    results = {
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'coverage_drop': coverage_drop,
        'args': vars(args),
    }
    with open(results_dir / "conformal_results.pkl", 'wb') as f:
        pickle.dump(results, f)

    return val_metrics, test_metrics


def plot_results(val_metrics: Dict, test_metrics: Dict, args, results_dir: Path):
    """Visualize results."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    periods = ['Validation\n(COVID onset)', 'Test\n(COVID peak)']

    # Coverage
    ax = axes[0]
    coverages = [val_metrics['coverage']*100, test_metrics['coverage']*100]
    colors = ['green' if c >= (1-args.alpha)*100-2 else 'orange' if c >= (1-args.alpha)*100-5 else 'red' for c in coverages]
    bars = ax.bar(periods, coverages, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline((1-args.alpha)*100, color='black', linestyle='--', lw=2, label=f'Target ({(1-args.alpha)*100:.0f}%)')
    ax.set_ylabel('Coverage (%)')
    ax.set_title('Coverage by Period')
    ax.set_ylim([0, 100])
    ax.legend()
    for bar, cov in zip(bars, coverages):
        ax.text(bar.get_x()+bar.get_width()/2., bar.get_height()+1, f'{cov:.1f}%', ha='center', fontweight='bold')

    # Set size
    ax = axes[1]
    sizes = [val_metrics['avg_set_size'], test_metrics['avg_set_size']]
    bars = ax.bar(periods, sizes, color=['steelblue', 'coral'], alpha=0.7, edgecolor='black')
    ax.set_ylabel('Avg Set Size')
    ax.set_title('Prediction Set Size')
    for bar, s in zip(bars, sizes):
        ax.text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.05, f'{s:.2f}', ha='center', fontweight='bold')

    # Coverage gap
    ax = axes[2]
    gaps = [val_metrics['coverage_gap']*100, test_metrics['coverage_gap']*100]
    colors = ['green' if g >= -2 else 'orange' if g >= -5 else 'red' for g in gaps]
    bars = ax.bar(periods, gaps, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(0, color='black', lw=1)
    ax.axhline(-5, color='red', linestyle='--', alpha=0.5, label='Threshold (-5%)')
    ax.set_ylabel('Coverage Gap (%)')
    ax.set_title('Gap from Target')
    ax.legend()
    for bar, g in zip(bars, gaps):
        va = 'bottom' if g >= 0 else 'top'
        ax.text(bar.get_x()+bar.get_width()/2., bar.get_height()+(0.5 if g>=0 else -1), f'{g:+.1f}%', ha='center', va=va, fontweight='bold')

    plt.suptitle(f'Conformal Prediction: {args.dataset}/{args.task}\nCOVID Distribution Shift (α={args.alpha})', fontweight='bold')
    plt.tight_layout()

    save_path = results_dir / f"conformal_covid_{args.task}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {save_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="rel-salt")
    parser.add_argument("--task", default="item-plant")
    parser.add_argument("--num_seeds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample_size", type=int, default=30000)
    parser.add_argument("--alpha", type=float, default=0.1)
    args = parser.parse_args()

    run_analysis(args)
