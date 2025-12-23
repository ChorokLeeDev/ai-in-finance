"""
Conformal Prediction for rel-salt Dataset with COVID Distribution Shift Analysis

This script applies conformal prediction to the rel-salt dataset to study:
1. How prediction set coverage changes during COVID (distribution shift)
2. Whether conformal prediction can detect/quantify the shift
3. Comparison of coverage guarantees before vs during COVID

Key dates:
- Train: before 2020-02-01 (pre-COVID)
- Val: 2020-02-01 to 2020-07-01 (COVID onset)
- Test: after 2020-07-01 (COVID mid-period)

Conformal Prediction for Classification:
- Instead of intervals, we generate prediction SETS
- A prediction set {A, B, C} means "true class is one of A, B, or C"
- Coverage guarantee: True class is in the set 90% of the time
"""

import argparse
import json
import os
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch_frame
from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_frame.gbdt import LightGBM
from torch_frame.typing import Metric
from torch_geometric.seed import seed_everything

from relbench.base import TaskType
from relbench.modeling.utils import get_stype_proposal, remove_pkey_fkey
from relbench.tasks import get_task

warnings.filterwarnings('ignore')

os.environ["OMP_NUM_THREADS"] = "8"


class ConformalClassifier:
    """
    Conformal Prediction for multi-class classification.

    Method: Adaptive Prediction Sets (APS) / RAPS
    - Uses softmax scores as conformity measure
    - Generates prediction sets with coverage guarantee
    """

    def __init__(self, alpha: float = 0.1):
        """
        Args:
            alpha: Miscoverage rate (0.1 = 90% coverage)
        """
        self.alpha = alpha
        self.quantile = None

    def _compute_scores(self, probs: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Compute conformity scores using cumulative probability method.

        For each sample, sort classes by probability, then find the cumulative
        probability needed to include the true class.
        """
        n_samples = len(y_true)
        scores = np.zeros(n_samples)

        for i in range(n_samples):
            # Sort classes by decreasing probability
            sorted_indices = np.argsort(probs[i])[::-1]
            sorted_probs = probs[i][sorted_indices]

            # Find cumulative probability to include true class
            true_class = y_true[i]
            cumsum = 0
            for j, idx in enumerate(sorted_indices):
                cumsum += sorted_probs[j]
                if idx == true_class:
                    scores[i] = cumsum
                    break

        return scores

    def calibrate(self, probs: np.ndarray, y_true: np.ndarray) -> 'ConformalClassifier':
        """
        Calibrate conformal predictor using calibration set.

        Args:
            probs: Predicted probabilities (n_samples, n_classes)
            y_true: True labels (n_samples,)
        """
        scores = self._compute_scores(probs, y_true)

        # Compute quantile with finite-sample correction
        n = len(scores)
        quantile_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        quantile_level = min(quantile_level, 1.0)

        self.quantile = np.quantile(scores, quantile_level)

        print(f"\nConformal Calibration (Classification):")
        print(f"  Target coverage: {100*(1-self.alpha):.1f}%")
        print(f"  Calibration set size: {n}")
        print(f"  Computed quantile: {self.quantile:.4f}")

        return self

    def predict_sets(self, probs: np.ndarray) -> List[set]:
        """
        Generate prediction sets with coverage guarantee.

        Args:
            probs: Predicted probabilities (n_samples, n_classes)

        Returns:
            List of sets, each containing predicted classes
        """
        if self.quantile is None:
            raise ValueError("Must call calibrate() first")

        n_samples = probs.shape[0]
        prediction_sets = []

        for i in range(n_samples):
            # Sort classes by decreasing probability
            sorted_indices = np.argsort(probs[i])[::-1]
            sorted_probs = probs[i][sorted_indices]

            # Add classes until cumulative prob >= quantile
            pred_set = set()
            cumsum = 0
            for j, idx in enumerate(sorted_indices):
                pred_set.add(idx)
                cumsum += sorted_probs[j]
                if cumsum >= self.quantile:
                    break

            prediction_sets.append(pred_set)

        return prediction_sets


def compute_coverage_metrics(prediction_sets: List[set], y_true: np.ndarray,
                              alpha: float = 0.1) -> Dict:
    """
    Compute coverage metrics for prediction sets.
    """
    n = len(y_true)

    # Coverage: fraction of times true label is in prediction set
    covered = sum(1 for i, ps in enumerate(prediction_sets) if y_true[i] in ps)
    coverage = covered / n

    # Set sizes
    set_sizes = [len(ps) for ps in prediction_sets]
    avg_set_size = np.mean(set_sizes)

    return {
        'coverage': coverage,
        'target_coverage': 1 - alpha,
        'coverage_gap': coverage - (1 - alpha),
        'avg_set_size': avg_set_size,
        'min_set_size': min(set_sizes),
        'max_set_size': max(set_sizes),
        'median_set_size': np.median(set_sizes),
    }


def train_ensemble_model(task, dataset, args, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Train a single LightGBM model and return predictions.
    """
    seed_everything(seed)
    np.random.seed(seed)

    train_table = task.get_table("train")
    val_table = task.get_table("val")
    test_table = task.get_table("test", mask_input_cols=False)

    # Get entity table
    entity_table = dataset.get_db().table_dict[task.entity_table]
    entity_df = entity_table.df.copy()

    # Get stypes
    stypes_cache_path = Path(f"{args.cache_dir}/{args.dataset}/tasks/{args.task}/stypes.json")
    try:
        with open(stypes_cache_path, "r") as f:
            col_to_stype_dict = json.load(f)
        for table, col_to_stype in col_to_stype_dict.items():
            for col, stype_str in col_to_stype.items():
                col_to_stype[col] = stype(stype_str)
    except FileNotFoundError:
        col_to_stype_dict = get_stype_proposal(dataset.get_db())
        Path(stypes_cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(stypes_cache_path, "w") as f:
            json.dump(col_to_stype_dict, f, indent=2, default=str)

    col_to_stype = col_to_stype_dict[task.entity_table].copy()
    remove_pkey_fkey(col_to_stype, entity_table)

    for col in dataset.remove_columns:
        if col in col_to_stype:
            del col_to_stype[col]

    col_to_stype[task.target_col] = torch_frame.categorical

    # Subsample training data
    if args.sample_size > 0 and args.sample_size < len(train_table):
        np.random.seed(seed)
        sampled_idx = np.random.permutation(len(train_table))[:args.sample_size]
        train_table.df = train_table.df.iloc[sampled_idx].copy()

    # Merge entity features
    dfs = {}
    for split, table in [("train", train_table), ("val", val_table), ("test", test_table)]:
        entity_df_copy = entity_df.copy()
        left_entity = list(table.fkey_col_to_pkey_table.keys())[0]
        entity_df_copy = entity_df_copy.astype({entity_table.pkey_col: table.df[left_entity].dtype})

        for col in set(entity_df_copy.columns).intersection(set(table.df.columns)):
            if col != entity_table.pkey_col:
                entity_df_copy = entity_df_copy.drop(columns=[col])

        dfs[split] = table.df.merge(
            entity_df_copy,
            how="left",
            left_on=left_entity,
            right_on=entity_table.pkey_col,
        )

    # Create torch_frame dataset
    train_dataset = torch_frame.data.Dataset(
        df=dfs["train"],
        col_to_stype=col_to_stype,
        target_col=task.target_col,
    )

    path = Path(f"{args.cache_dir}/{args.dataset}/tasks/{args.task}/materialized/train_seed{seed}_{args.sample_size}.pt")
    path.parent.mkdir(parents=True, exist_ok=True)
    train_dataset = train_dataset.materialize(path=path)

    tf_train = train_dataset.tensor_frame
    tf_val = train_dataset.convert_to_tensor_frame(dfs["val"])
    tf_test = train_dataset.convert_to_tensor_frame(dfs["test"])

    # Train LightGBM
    model = LightGBM(
        task_type=train_dataset.task_type,
        metric=Metric.ACCURACY,
        num_classes=task.num_classes,
    )
    model.tune(tf_train=tf_train, tf_val=tf_val, num_trials=args.num_trials)

    # Get probability predictions
    train_x, _, _ = model._to_lightgbm_input(tf_train)
    val_x, _, _ = model._to_lightgbm_input(tf_val)
    test_x, _, _ = model._to_lightgbm_input(tf_test)

    train_probs = model.model.predict(train_x)
    val_probs = model.model.predict(val_x)
    test_probs = model.model.predict(test_x)

    # Get true labels
    labels = {
        'train': train_table.df[task.target_col].values,
        'val': val_table.df[task.target_col].values,
        'test': test_table.df[task.target_col].values,
    }

    # Get timestamps for temporal analysis
    timestamps = {
        'train': train_table.df['CREATIONTIMESTAMP'].values if 'CREATIONTIMESTAMP' in train_table.df.columns else None,
        'val': val_table.df['CREATIONTIMESTAMP'].values if 'CREATIONTIMESTAMP' in val_table.df.columns else None,
        'test': test_table.df['CREATIONTIMESTAMP'].values if 'CREATIONTIMESTAMP' in test_table.df.columns else None,
    }

    return train_probs, val_probs, test_probs, labels, timestamps


def run_conformal_analysis(args):
    """
    Main conformal prediction analysis with COVID temporal analysis.
    """
    print(f"\n{'='*70}")
    print(f"Conformal Prediction Analysis: {args.dataset} / {args.task}")
    print(f"COVID Distribution Shift Study")
    print(f"{'='*70}\n")

    # Load task
    task = get_task(args.dataset, args.task, download=True)
    dataset = task.dataset

    print(f"Dataset temporal structure:")
    print(f"  Val timestamp (COVID start): {dataset.val_timestamp}")
    print(f"  Test timestamp (COVID mid): {dataset.test_timestamp}")
    print(f"  Number of classes: {task.num_classes}")

    # Results directory
    results_dir = Path(f"results/conformal/{args.dataset}/{args.task}")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Train ensemble or load cached predictions
    all_val_probs = []
    all_test_probs = []
    labels = None
    timestamps = None

    for seed in range(args.seed, args.seed + args.num_seeds):
        cache_path = results_dir / f"predictions_seed{seed}_sample{args.sample_size}.pkl"

        if cache_path.exists() and not args.force_retrain:
            print(f"\nLoading cached predictions for seed {seed}...")
            with open(cache_path, 'rb') as f:
                cached = pickle.load(f)
            all_val_probs.append(cached['val_probs'])
            all_test_probs.append(cached['test_probs'])
            if labels is None:
                labels = cached['labels']
                timestamps = cached.get('timestamps')
        else:
            print(f"\nTraining model with seed {seed}...")
            train_probs, val_probs, test_probs, labels_dict, ts_dict = train_ensemble_model(
                task, dataset, args, seed
            )
            all_val_probs.append(val_probs)
            all_test_probs.append(test_probs)

            if labels is None:
                labels = labels_dict
                timestamps = ts_dict

            # Cache predictions
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'val_probs': val_probs,
                    'test_probs': test_probs,
                    'labels': labels_dict,
                    'timestamps': ts_dict,
                }, f)
            print(f"Cached predictions to {cache_path}")

    # Ensemble averaging
    print(f"\nEnsemble size: {len(all_val_probs)} models")
    val_probs_mean = np.mean(all_val_probs, axis=0)
    test_probs_mean = np.mean(all_test_probs, axis=0)

    # Split validation for calibration
    n_val = len(val_probs_mean)
    n_calib = n_val // 2

    calib_probs = val_probs_mean[:n_calib]
    calib_labels = labels['val'][:n_calib]
    eval_probs = val_probs_mean[n_calib:]
    eval_labels = labels['val'][n_calib:]

    print(f"\nCalibration set: {n_calib} samples")
    print(f"Evaluation set: {n_val - n_calib} samples")
    print(f"Test set: {len(test_probs_mean)} samples")

    # Calibrate conformal predictor
    conformal = ConformalClassifier(alpha=args.alpha)
    conformal.calibrate(calib_probs, calib_labels)

    # Generate prediction sets
    print(f"\n{'='*70}")
    print("COVERAGE ANALYSIS")
    print("="*70)

    # Validation evaluation set
    val_sets = conformal.predict_sets(eval_probs)
    val_metrics = compute_coverage_metrics(val_sets, eval_labels, args.alpha)

    print(f"\nValidation Set (COVID onset period):")
    print(f"  Coverage: {val_metrics['coverage']*100:.2f}% (target: {val_metrics['target_coverage']*100:.0f}%)")
    print(f"  Coverage gap: {val_metrics['coverage_gap']*100:+.2f}%")
    print(f"  Avg set size: {val_metrics['avg_set_size']:.2f}")
    print(f"  Set size range: [{val_metrics['min_set_size']}, {val_metrics['max_set_size']}]")

    # Test set (COVID mid-period)
    test_sets = conformal.predict_sets(test_probs_mean)
    test_metrics = compute_coverage_metrics(test_sets, labels['test'], args.alpha)

    print(f"\nTest Set (COVID mid-period - DISTRIBUTION SHIFT!):")
    print(f"  Coverage: {test_metrics['coverage']*100:.2f}% (target: {test_metrics['target_coverage']*100:.0f}%)")
    print(f"  Coverage gap: {test_metrics['coverage_gap']*100:+.2f}%")
    print(f"  Avg set size: {test_metrics['avg_set_size']:.2f}")
    print(f"  Set size range: [{test_metrics['min_set_size']}, {test_metrics['max_set_size']}]")

    # Key insight
    print(f"\n{'='*70}")
    print("COVID DISTRIBUTION SHIFT ANALYSIS")
    print("="*70)

    coverage_drop = val_metrics['coverage'] - test_metrics['coverage']
    set_size_change = test_metrics['avg_set_size'] - val_metrics['avg_set_size']

    print(f"\nCoverage change (Val → Test): {coverage_drop*100:+.2f}%")
    print(f"Set size change (Val → Test): {set_size_change:+.2f}")

    if coverage_drop > 0.05:
        print(f"\n[!] SIGNIFICANT COVERAGE DROP detected!")
        print(f"    Conformal prediction calibrated on pre/early-COVID data")
        print(f"    fails to maintain coverage during COVID peak.")
        print(f"    This quantifies the distribution shift magnitude.")
    else:
        print(f"\n[✓] Coverage maintained despite distribution shift.")
        print(f"    Either the shift is small or the model is robust.")

    # Visualization
    plot_conformal_results(val_metrics, test_metrics, args, results_dir)

    return val_metrics, test_metrics


def plot_conformal_results(val_metrics: Dict, test_metrics: Dict, args, results_dir: Path):
    """
    Visualize conformal prediction results.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Coverage comparison
    ax = axes[0]
    periods = ['Validation\n(COVID onset)', 'Test\n(COVID peak)']
    coverages = [val_metrics['coverage'] * 100, test_metrics['coverage'] * 100]
    colors = ['green' if c >= (1-args.alpha)*100 - 2 else 'red' for c in coverages]

    bars = ax.bar(periods, coverages, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline((1-args.alpha)*100, color='black', linestyle='--', linewidth=2, label=f'Target ({(1-args.alpha)*100:.0f}%)')
    ax.set_ylabel('Coverage (%)')
    ax.set_title('Coverage by Period')
    ax.set_ylim([0, 100])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    for bar, cov in zip(bars, coverages):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{cov:.1f}%', ha='center', va='bottom', fontweight='bold')

    # 2. Set size comparison
    ax = axes[1]
    set_sizes = [val_metrics['avg_set_size'], test_metrics['avg_set_size']]
    bars = ax.bar(periods, set_sizes, color=['blue', 'orange'], alpha=0.7, edgecolor='black')
    ax.set_ylabel('Average Set Size')
    ax.set_title('Prediction Set Size by Period')
    ax.grid(True, alpha=0.3, axis='y')

    for bar, size in zip(bars, set_sizes):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                f'{size:.2f}', ha='center', va='bottom', fontweight='bold')

    # 3. Coverage gap analysis
    ax = axes[2]
    gaps = [val_metrics['coverage_gap'] * 100, test_metrics['coverage_gap'] * 100]
    colors = ['green' if g >= -2 else 'red' for g in gaps]
    bars = ax.bar(periods, gaps, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(0, color='black', linestyle='-', linewidth=1)
    ax.axhline(-5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Acceptable threshold (-5%)')
    ax.set_ylabel('Coverage Gap (%)')
    ax.set_title('Gap from Target Coverage')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    for bar, gap in zip(bars, gaps):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5 if gap >= 0 else bar.get_height() - 1.5,
                f'{gap:+.1f}%', ha='center', va='bottom' if gap >= 0 else 'top', fontweight='bold')

    plt.suptitle(f'Conformal Prediction: {args.dataset} / {args.task}\n'
                 f'COVID Distribution Shift Analysis (α={args.alpha})', fontsize=12, fontweight='bold')
    plt.tight_layout()

    save_path = results_dir / f"conformal_covid_analysis_alpha{int(args.alpha*100)}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved figure to {save_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conformal Prediction for rel-salt COVID analysis")
    parser.add_argument("--dataset", type=str, default="rel-salt")
    parser.add_argument("--task", type=str, default="item-plant")
    parser.add_argument("--num_seeds", type=int, default=3, help="Number of ensemble members")
    parser.add_argument("--seed", type=int, default=42, help="Starting seed")
    parser.add_argument("--sample_size", type=int, default=50000, help="Training sample size")
    parser.add_argument("--num_trials", type=int, default=5, help="LightGBM tuning trials")
    parser.add_argument("--alpha", type=float, default=0.1, help="Miscoverage rate (0.1 = 90% coverage)")
    parser.add_argument("--cache_dir", type=str, default=os.path.expanduser("~/.cache/relbench_examples"))
    parser.add_argument("--force_retrain", action="store_true", help="Force retraining even if cached")

    args = parser.parse_args()

    run_conformal_analysis(args)
