"""
Temporal Uncertainty Analysis for SALT Dataset
===============================================

**Research Goal**: Measure epistemic uncertainty from ensemble predictions to
detect when distribution has shifted. This is Phase 3 of the UQ-based shift detection research.

**Key Insight**: Predictions themselves are NOT the goal—they are the TOOL we use
to measure uncertainty. When models in an ensemble disagree, it signals that the
data distribution differs from training, indicating a distribution shift.

**Workflow**:
1. Load ensemble predictions (5+ models with different random seeds)
   - Each model trained on same data, different initialization
   - Predictions serve as mechanism to measure uncertainty

2. Calculate epistemic uncertainty (model disagreement)
   - For classification: Mutual Information = H(E[p(y|x)]) - E[H(p(y|x))]
   - Measures how much models disagree on predictions
   - High disagreement = High epistemic uncertainty = Unfamiliar data

3. Track uncertainty evolution across temporal periods
   - Train period (2018-2020): Baseline uncertainty
   - Test period (2020-2021): Post-COVID uncertainty
   - % increase indicates shift severity

4. Validate hypothesis: "Epistemic uncertainty tracks distribution shift"
   - Compare with PSI from Phase 1 (temporal_shift_detection.py)
   - Expected: Higher PSI → Higher uncertainty increase

**Output**: Epistemic uncertainty metrics that will be correlated with PSI
in Phase 4 (compare_shift_uncertainty.py)

**Scientific Contribution**: Demonstrates that UQ can detect distribution shift
without requiring ground truth labels on new data (early warning system).

Author: ChorokLeeDev
Created: 2025-01-19
Last Updated: 2025-01-20
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

from relbench.datasets import get_dataset
from relbench.tasks import get_task

# COVID-19 temporal markers
VAL_TIMESTAMP = pd.Timestamp("2020-02-01")
TEST_TIMESTAMP = pd.Timestamp("2020-07-01")


def load_ensemble_predictions(task_name: str,
                               results_dir: str = "results/ensemble/rel-salt",
                               sample_size: int = 10000) -> Dict[str, np.ndarray]:
    """
    Load ensemble predictions for a task.

    Returns:
        Dictionary with keys: 'train_preds', 'val_preds', 'test_preds'
        Each is array of shape (n_models, n_samples, n_classes)
    """
    task_dir = Path(results_dir) / task_name

    if not task_dir.exists():
        raise FileNotFoundError(f"No ensemble results found for {task_name} at {task_dir}")

    # Find all seed files
    seed_files = sorted(task_dir.glob(f"seed_*_sample_{sample_size}.pkl"))

    if len(seed_files) == 0:
        raise FileNotFoundError(f"No prediction files found in {task_dir}")

    print(f"Loading {len(seed_files)} ensemble members for {task_name}")

    # Load predictions from all seeds
    all_train_preds = []
    all_val_preds = []
    all_test_preds = []

    for seed_file in seed_files:
        with open(seed_file, 'rb') as f:
            data = pickle.load(f)

        all_train_preds.append(data['train_pred'])
        all_val_preds.append(data['val_pred'])
        all_test_preds.append(data['test_pred'])

    # Stack into (n_models, n_samples, n_classes)
    ensemble_preds = {
        'train_preds': np.stack(all_train_preds, axis=0),
        'val_preds': np.stack(all_val_preds, axis=0),
        'test_preds': np.stack(all_test_preds, axis=0),
    }

    print(f"  Train: {ensemble_preds['train_preds'].shape}")
    print(f"  Val: {ensemble_preds['val_preds'].shape}")
    print(f"  Test: {ensemble_preds['test_preds'].shape}")

    return ensemble_preds


def load_timestamps(task_name: str, sample_size: int = 10000) -> Dict[str, pd.Series]:
    """Load timestamps for each split."""
    task = get_task("rel-salt", task_name, download=False)

    # Load task tables
    train_table = task.get_table("train")
    val_table = task.get_table("val")
    test_table = task.get_table("test")

    # Sample if needed (match prediction sampling)
    np.random.seed(42)

    def sample_df(df, size):
        if len(df) > size:
            sampled_idx = np.random.choice(len(df), size=size, replace=False)
            return df.iloc[sampled_idx]
        return df

    train_sampled = sample_df(train_table.df, sample_size)
    val_sampled = sample_df(val_table.df, sample_size)
    test_sampled = sample_df(test_table.df, sample_size)

    timestamps = {
        'train': train_sampled['CREATIONTIMESTAMP'],
        'val': val_sampled['CREATIONTIMESTAMP'],
        'test': test_sampled['CREATIONTIMESTAMP'],
    }

    return timestamps


def calculate_epistemic_uncertainty(predictions: np.ndarray) -> np.ndarray:
    """
    Calculate epistemic uncertainty from ensemble predictions.

    Args:
        predictions: (n_models, n_samples, n_classes) array of probabilities

    Returns:
        (n_samples,) array of epistemic uncertainty values
    """
    # Predictive entropy (total uncertainty)
    mean_probs = predictions.mean(axis=0)  # (n_samples, n_classes)
    predictive_entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-10), axis=1)

    # Expected entropy (aleatoric uncertainty)
    model_entropies = -np.sum(predictions * np.log(predictions + 1e-10), axis=2)  # (n_models, n_samples)
    expected_entropy = model_entropies.mean(axis=0)  # (n_samples,)

    # Epistemic uncertainty = Predictive - Expected (mutual information)
    epistemic_uncertainty = predictive_entropy - expected_entropy

    return epistemic_uncertainty


def calculate_variance_ratio(predictions: np.ndarray) -> np.ndarray:
    """
    Calculate variance ratio (simpler uncertainty metric).

    Variance ratio = 1 - (max vote frequency) = disagreement rate
    """
    # Get predicted class for each model
    predicted_classes = predictions.argmax(axis=2)  # (n_models, n_samples)

    # Count votes for most common class
    n_models = predictions.shape[0]
    n_samples = predictions.shape[1]

    variance_ratios = np.zeros(n_samples)
    for i in range(n_samples):
        votes = predicted_classes[:, i]
        most_common_vote = np.bincount(votes).max()
        variance_ratios[i] = 1 - (most_common_vote / n_models)

    return variance_ratios


def analyze_temporal_uncertainty(task_name: str,
                                 ensemble_preds: Dict[str, np.ndarray],
                                 timestamps: Dict[str, pd.Series]) -> Dict:
    """Analyze uncertainty evolution over temporal periods."""

    results = {
        'task_name': task_name,
        'periods': {},
    }

    # Analyze each split
    for split_name in ['train', 'val', 'test']:
        preds = ensemble_preds[f'{split_name}_preds']
        ts = timestamps[split_name].reset_index(drop=True)

        # Calculate uncertainties
        epistemic_unc = calculate_epistemic_uncertainty(preds)
        variance_ratio = calculate_variance_ratio(preds)

        # Overall statistics
        results['periods'][split_name] = {
            'n_samples': len(epistemic_unc),
            'epistemic_mean': float(epistemic_unc.mean()),
            'epistemic_std': float(epistemic_unc.std()),
            'epistemic_median': float(np.median(epistemic_unc)),
            'variance_ratio_mean': float(variance_ratio.mean()),
            'variance_ratio_std': float(variance_ratio.std()),
        }

        # Monthly breakdown (if timestamps available)
        if split_name == 'train' and len(ts) > 0:
            # Divide train into quarterly bins
            ts_sorted = ts.sort_values()
            quarters = pd.to_datetime(ts_sorted).dt.to_period('Q')

            monthly_stats = []
            for quarter, indices in quarters.groupby(quarters).groups.items():
                indices_list = indices.tolist()
                quarter_unc = epistemic_unc[indices_list]
                monthly_stats.append({
                    'period': str(quarter),
                    'n_samples': len(indices_list),
                    'epistemic_mean': float(quarter_unc.mean()),
                    'epistemic_std': float(quarter_unc.std()),
                })

            results['periods'][split_name]['quarterly_breakdown'] = monthly_stats

    # Calculate shift in uncertainty: train → val → test
    train_unc = results['periods']['train']['epistemic_mean']
    val_unc = results['periods']['val']['epistemic_mean']
    test_unc = results['periods']['test']['epistemic_mean']

    results['uncertainty_shift'] = {
        'train_to_val_increase': float(val_unc - train_unc),
        'train_to_test_increase': float(test_unc - train_unc),
        'val_to_test_increase': float(test_unc - val_unc),
        'train_to_val_pct': float((val_unc - train_unc) / train_unc * 100) if train_unc > 0 else 0.0,
        'train_to_test_pct': float((test_unc - train_unc) / train_unc * 100) if train_unc > 0 else 0.0,
        'val_to_test_pct': float((test_unc - val_unc) / val_unc * 100) if val_unc > 0 else 0.0,
    }

    return results


def plot_uncertainty_evolution(task_name: str,
                               uncertainty_results: Dict,
                               shift_metrics: Dict,
                               output_dir: Path):
    """Plot uncertainty evolution across temporal periods."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Uncertainty by split
    splits = ['train', 'val', 'test']
    split_labels = ['Pre-COVID\n(Train)', 'COVID Onset\n(Val)', 'COVID Impact\n(Test)']
    colors = ['#2ecc71', '#f39c12', '#e74c3c']

    epistemic_means = [uncertainty_results['periods'][s]['epistemic_mean'] for s in splits]
    epistemic_stds = [uncertainty_results['periods'][s]['epistemic_std'] for s in splits]

    axes[0].bar(range(3), epistemic_means, yerr=epistemic_stds,
               color=colors, alpha=0.7, capsize=10, width=0.6)
    axes[0].set_xticks(range(3))
    axes[0].set_xticklabels(split_labels, fontsize=11)
    axes[0].set_ylabel('Epistemic Uncertainty', fontsize=12, fontweight='bold')
    axes[0].set_title(f'Uncertainty Evolution: {task_name}', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')

    # Add percentage increase annotations
    train_val_pct = uncertainty_results['uncertainty_shift']['train_to_val_pct']
    val_test_pct = uncertainty_results['uncertainty_shift']['val_to_test_pct']
    axes[0].text(0.5, max(epistemic_means) * 0.95, f'+{train_val_pct:.1f}%',
                ha='center', fontsize=10, fontweight='bold', color='orange')
    axes[0].text(1.5, max(epistemic_means) * 0.95, f'+{val_test_pct:.1f}%',
                ha='center', fontsize=10, fontweight='bold', color='red')

    # Plot 2: Uncertainty vs Distribution Shift
    psi_value = shift_metrics.get('psi_train_test', 0)
    js_value = shift_metrics.get('js_train_test', 0)
    unc_increase_pct = uncertainty_results['uncertainty_shift']['train_to_test_pct']

    # Show relationship
    axes[1].scatter([psi_value], [unc_increase_pct], s=200, color='#e74c3c', alpha=0.7,
                   edgecolor='black', linewidth=2, zorder=3)
    axes[1].axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    axes[1].axvline(0.1, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='PSI=0.1 (Moderate)')
    axes[1].axvline(0.2, color='red', linestyle='--', linewidth=1, alpha=0.5, label='PSI=0.2 (Significant)')
    axes[1].set_xlabel('Distribution Shift (PSI)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Uncertainty Increase (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('Shift vs Uncertainty Correlation', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    # Annotate point
    axes[1].annotate(f'{task_name}\nPSI={psi_value:.3f}\nJS={js_value:.3f}\nΔUnc={unc_increase_pct:.1f}%',
                    xy=(psi_value, unc_increase_pct), xytext=(10, 10),
                    textcoords='offset points', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    plt.tight_layout()
    output_path = output_dir / f"{task_name}_uncertainty_evolution.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Temporal uncertainty analysis for SALT dataset")
    parser.add_argument("--task", type=str, required=True,
                       help="Task name (e.g., item-plant)")
    parser.add_argument("--results_dir", type=str, default="results/ensemble/rel-salt",
                       help="Directory with ensemble predictions")
    parser.add_argument("--output_dir", type=str, default="chorok/figures/uncertainty_temporal",
                       help="Output directory for figures")
    parser.add_argument("--sample_size", type=int, default=10000,
                       help="Sample size used in ensemble predictions")
    parser.add_argument("--save_json", type=str, default="chorok/results/temporal_uncertainty.json",
                       help="Path to save JSON results")

    args = parser.parse_args()

    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_dir = Path(args.save_json).parent
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"# Temporal Uncertainty Analysis: {args.task}")
    print(f"{'#'*60}\n")

    # Load ensemble predictions
    try:
        ensemble_preds = load_ensemble_predictions(args.task, args.results_dir, args.sample_size)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print(f"\nAvailable tasks:")
        rel_salt_dir = Path(args.results_dir)
        if rel_salt_dir.exists():
            for task_dir in rel_salt_dir.iterdir():
                if task_dir.is_dir():
                    print(f"  - {task_dir.name}")
        return

    # Load timestamps
    print("\nLoading timestamps...")
    timestamps = load_timestamps(args.task, args.sample_size)

    # Load distribution shift metrics
    shift_file = Path("chorok/results/shift_detection.json")
    if shift_file.exists():
        with open(shift_file, 'r') as f:
            all_shift_metrics = json.load(f)
        shift_metrics = all_shift_metrics.get(args.task, {}).get('shift_metrics', {})
    else:
        print("WARNING: Shift detection results not found. Proceeding without shift metrics.")
        shift_metrics = {}

    # Analyze temporal uncertainty
    print("\nAnalyzing temporal uncertainty...")
    uncertainty_results = analyze_temporal_uncertainty(args.task, ensemble_preds, timestamps)

    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"\nEpistemic Uncertainty by Period:")
    for split in ['train', 'val', 'test']:
        period = uncertainty_results['periods'][split]
        print(f"  {split.upper():6s}: {period['epistemic_mean']:.4f} ± {period['epistemic_std']:.4f}")

    print(f"\nUncertainty Shift (Train → Test):")
    shift = uncertainty_results['uncertainty_shift']
    print(f"  Absolute: {shift['train_to_test_increase']:+.4f}")
    print(f"  Relative: {shift['train_to_test_pct']:+.2f}%")

    if 'psi_train_test' in shift_metrics:
        print(f"\nDistribution Shift Metrics:")
        print(f"  PSI: {shift_metrics['psi_train_test']:.4f}")
        print(f"  JS Div: {shift_metrics['js_train_test']:.4f}")

    # Visualize
    print("\nGenerating visualizations...")
    plot_uncertainty_evolution(args.task, uncertainty_results, shift_metrics, output_dir)

    # Save results
    output_data = {
        args.task: {
            'uncertainty_results': uncertainty_results,
            'shift_metrics': shift_metrics,
        }
    }

    # Merge with existing results if file exists
    if Path(args.save_json).exists():
        with open(args.save_json, 'r') as f:
            existing_data = json.load(f)
        existing_data.update(output_data)
        output_data = existing_data

    with open(args.save_json, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)

    print(f"\nResults saved to: {args.save_json}")
    print(f"Figures saved to: {output_dir}")
    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
