"""
Analyze ensemble uncertainty quantification for REGRESSION tasks.

For regression, we use:
- Epistemic uncertainty: std of predictions across models
- Expected Calibration Error (ECE): for binned predictions
- Variance decomposition: total variance = epistemic + aleatoric
"""

import pickle
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def load_ensemble_predictions(results_dir, num_seeds=5, sample_size=50000):
    """Load predictions from multiple seeds."""
    ensemble_data = []
    for seed in range(42, 42 + num_seeds):
        pkl_path = results_dir / f"seed_{seed}_sample_{sample_size}.pkl"
        if pkl_path.exists():
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
                ensemble_data.append(data)
        else:
            print(f"Warning: {pkl_path} not found")

    if len(ensemble_data) == 0:
        raise ValueError("No prediction files found")

    return ensemble_data


def compute_epistemic_uncertainty(predictions):
    """
    Compute epistemic uncertainty for regression.
    predictions: [num_models, num_samples]

    Returns:
        mean_pred: [num_samples] - ensemble mean prediction
        epistemic_unc: [num_samples] - std across models
    """
    predictions = np.array(predictions)  # [num_models, num_samples]
    mean_pred = predictions.mean(axis=0)  # [num_samples]
    epistemic_unc = predictions.std(axis=0)  # [num_samples]
    return mean_pred, epistemic_unc


def compute_regression_metrics(pred_mean, pred_std, y_true):
    """
    Compute regression-specific UQ metrics.

    Args:
        pred_mean: [num_samples] - ensemble mean prediction
        pred_std: [num_samples] - epistemic uncertainty (std)
        y_true: [num_samples] - ground truth

    Returns:
        dict of metrics
    """
    # Squared error for each sample
    squared_errors = (pred_mean - y_true) ** 2

    # Mean Squared Error (MSE)
    mse = squared_errors.mean()
    mae = np.abs(pred_mean - y_true).mean()

    # Correlation between uncertainty and error
    # High uncertainty should correlate with high error
    error_unc_corr = np.corrcoef(np.abs(pred_mean - y_true), pred_std)[0, 1]

    # Calibration: Expected Calibration Error (ECE)
    # Bin samples by predicted std, check if actual errors match
    ece = compute_regression_ece(pred_mean, pred_std, y_true, n_bins=10)

    # Average epistemic uncertainty
    avg_epistemic = pred_std.mean()

    return {
        'mse': mse,
        'mae': mae,
        'rmse': np.sqrt(mse),
        'error_uncertainty_correlation': error_unc_corr,
        'expected_calibration_error': ece,
        'avg_epistemic_uncertainty': avg_epistemic,
        'std_epistemic_uncertainty': pred_std.std(),
    }


def compute_regression_ece(pred_mean, pred_std, y_true, n_bins=10):
    """
    Compute Expected Calibration Error for regression.

    The idea: samples with high predicted uncertainty (std) should have
    higher actual errors.

    We bin samples by predicted std, and check if the average error
    in each bin matches the average predicted std.
    """
    # Absolute errors
    abs_errors = np.abs(pred_mean - y_true)

    # Create bins based on predicted std
    bin_boundaries = np.percentile(pred_std, np.linspace(0, 100, n_bins + 1))
    bin_boundaries[-1] += 1e-8  # Ensure last boundary includes max

    ece = 0.0
    total_samples = len(pred_std)

    for i in range(n_bins):
        # Find samples in this bin
        in_bin = (pred_std >= bin_boundaries[i]) & (pred_std < bin_boundaries[i + 1])

        if in_bin.sum() == 0:
            continue

        # Average predicted uncertainty in this bin
        avg_pred_unc = pred_std[in_bin].mean()

        # Average actual error in this bin
        avg_actual_error = abs_errors[in_bin].mean()

        # Weighted difference
        weight = in_bin.sum() / total_samples
        ece += weight * np.abs(avg_pred_unc - avg_actual_error)

    return ece


def plot_regression_uq(pred_mean, pred_std, y_true, save_path=None):
    """
    Create visualization for regression UQ.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Predicted vs True
    ax = axes[0, 0]
    ax.scatter(y_true, pred_mean, alpha=0.3, s=1)
    min_val = min(y_true.min(), pred_mean.min())
    max_val = max(y_true.max(), pred_mean.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction')
    ax.set_xlabel('True Value')
    ax.set_ylabel('Predicted Value')
    ax.set_title('Predicted vs True')
    ax.legend()

    # 2. Uncertainty vs Absolute Error
    ax = axes[0, 1]
    abs_errors = np.abs(pred_mean - y_true)
    ax.scatter(pred_std, abs_errors, alpha=0.3, s=1)
    ax.set_xlabel('Epistemic Uncertainty (Std)')
    ax.set_ylabel('Absolute Error')
    ax.set_title('Uncertainty vs Error')

    # Add correlation
    corr = np.corrcoef(pred_std, abs_errors)[0, 1]
    ax.text(0.05, 0.95, f'Correlation: {corr:.3f}',
            transform=ax.transAxes, verticalalignment='top')

    # 3. Distribution of Uncertainty
    ax = axes[1, 0]
    ax.hist(pred_std, bins=50, alpha=0.7)
    ax.set_xlabel('Epistemic Uncertainty (Std)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Epistemic Uncertainty')
    ax.axvline(pred_std.mean(), color='r', linestyle='--',
               label=f'Mean: {pred_std.mean():.3f}')
    ax.legend()

    # 4. Calibration plot
    ax = axes[1, 1]
    abs_errors = np.abs(pred_mean - y_true)

    # Bin by predicted uncertainty
    n_bins = 10
    bin_boundaries = np.percentile(pred_std, np.linspace(0, 100, n_bins + 1))
    bin_centers = []
    avg_pred_uncs = []
    avg_actual_errors = []

    for i in range(n_bins):
        in_bin = (pred_std >= bin_boundaries[i]) & (pred_std < bin_boundaries[i + 1])
        if in_bin.sum() == 0:
            continue
        bin_centers.append((bin_boundaries[i] + bin_boundaries[i + 1]) / 2)
        avg_pred_uncs.append(pred_std[in_bin].mean())
        avg_actual_errors.append(abs_errors[in_bin].mean())

    if len(avg_pred_uncs) > 0 and len(avg_actual_errors) > 0:
        ax.plot(avg_pred_uncs, avg_actual_errors, 'o-', label='Actual')
        min_val = min(min(avg_pred_uncs), min(avg_actual_errors))
        max_val = max(max(avg_pred_uncs), max(avg_actual_errors))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect calibration')
        ax.set_xlabel('Average Predicted Uncertainty')
        ax.set_ylabel('Average Actual Error')
        ax.set_title('Calibration: Predicted Uncertainty vs Actual Error')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No calibration data\n(predictions identical)',
                ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    plt.close()


def analyze_task(dataset, task, num_seeds=5, sample_size=50000):
    """Analyze a single task."""
    print(f"\n{'='*60}")
    print(f"Analyzing {dataset} / {task}")
    print(f"{'='*60}\n")

    results_dir = Path(f"results/ensemble/{dataset}/{task}")

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    # Load ensemble predictions
    ensemble_data = load_ensemble_predictions(results_dir, num_seeds, sample_size)
    print(f"Loaded {len(ensemble_data)} models\n")

    # Analyze each split
    for split in ['train', 'val', 'test']:
        print(f"\n{split.upper()} Split:")
        print("-" * 40)

        # Collect predictions from all models
        predictions = []
        y_true = None

        for data in ensemble_data:
            pred = data[f'{split}_pred']
            predictions.append(pred)
            if y_true is None:
                y_true = data[f'{split}_true']

        # Compute ensemble mean and epistemic uncertainty
        pred_mean, pred_std = compute_epistemic_uncertainty(predictions)

        print(f"Prediction mean: {pred_mean.mean():.4f} ± {pred_mean.std():.4f}")
        print(f"True mean: {y_true.mean():.4f} ± {y_true.std():.4f}")
        print(f"Epistemic uncertainty mean: {pred_std.mean():.6f}")
        print(f"Epistemic uncertainty std: {pred_std.std():.6f}")

        # Compute metrics
        metrics = compute_regression_metrics(pred_mean, pred_std, y_true)

        print("\nUQ Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.6f}")

        # Plot for val split (test may have identical predictions in autocomplete tasks)
        if split == 'val':
            plot_path = results_dir / f"regression_uq_analysis_val.png"
            plot_regression_uq(pred_mean, pred_std, y_true, save_path=plot_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="rel-f1")
    parser.add_argument("--task", type=str, default="results-position")
    parser.add_argument("--num_seeds", type=int, default=5)
    parser.add_argument("--sample_size", type=int, default=50000)
    args = parser.parse_args()

    analyze_task(args.dataset, args.task, args.num_seeds, args.sample_size)
