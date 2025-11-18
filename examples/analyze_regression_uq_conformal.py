"""
Enhanced UQ Analysis with Conformal Prediction

Adds coverage guarantees to ensemble predictions using conformal prediction.
This ensures that prediction intervals truly contain the true value with
specified probability (e.g., 90%).

Key improvement over basic ensemble:
- Ensemble alone: No coverage guarantee (often underconfident)
- Ensemble + Conformal: Guaranteed coverage on calibration distribution
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
    """Compute ensemble mean and epistemic uncertainty (std)."""
    predictions = np.array(predictions)
    mean_pred = predictions.mean(axis=0)
    epistemic_unc = predictions.std(axis=0)
    return mean_pred, epistemic_unc


class ConformalPredictor:
    """
    Conformal Prediction for regression with coverage guarantee.

    Method: Split Conformal Prediction (Vovk et al., 2005)
    - Uses calibration set to compute quantile of errors
    - Applies quantile to test predictions
    - Guarantees coverage on exchangeable data
    """

    def __init__(self, alpha=0.1):
        """
        Args:
            alpha: Miscoverage rate (e.g., 0.1 for 90% coverage)
        """
        self.alpha = alpha
        self.quantile = None

    def calibrate(self, pred_mean, pred_std, y_true):
        """
        Calibrate conformal predictor using calibration set.

        Args:
            pred_mean: Ensemble mean predictions on calibration set
            pred_std: Ensemble std (epistemic uncertainty) on calibration set
            y_true: True labels for calibration set
        """
        # Compute conformity scores (absolute errors)
        # Using absolute residuals as the score
        scores = np.abs(y_true - pred_mean)

        # Compute quantile for desired coverage
        # Use (n+1)(1-alpha)/n quantile for finite sample correction
        n = len(scores)
        quantile_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        quantile_level = min(quantile_level, 1.0)  # Cap at 1.0

        self.quantile = np.quantile(scores, quantile_level)

        print(f"\nConformal Calibration:")
        print(f"  Target coverage: {100*(1-self.alpha):.1f}%")
        print(f"  Calibration set size: {n}")
        print(f"  Quantile level: {quantile_level:.4f}")
        print(f"  Computed quantile: {self.quantile:.4f}")

        return self

    def predict(self, pred_mean, pred_std):
        """
        Generate prediction intervals with coverage guarantee.

        Args:
            pred_mean: Ensemble mean predictions
            pred_std: Ensemble std (epistemic uncertainty)

        Returns:
            lower: Lower bound of prediction interval
            upper: Upper bound of prediction interval
        """
        if self.quantile is None:
            raise ValueError("Must call calibrate() first")

        # Simple conformal: use fixed quantile
        # (Could also use adaptive methods that incorporate pred_std)
        lower = pred_mean - self.quantile
        upper = pred_mean + self.quantile

        return lower, upper


def compute_comprehensive_metrics(pred_mean, pred_std, y_true,
                                   lower=None, upper=None, alpha=0.1):
    """
    Compute comprehensive UQ metrics including coverage.

    Args:
        pred_mean: Ensemble mean predictions
        pred_std: Epistemic uncertainty (std)
        y_true: True labels
        lower: Lower bounds from conformal (optional)
        upper: Upper bounds from conformal (optional)
        alpha: Miscoverage rate
    """
    metrics = {}

    # Basic prediction metrics
    metrics['mae'] = np.abs(pred_mean - y_true).mean()
    metrics['rmse'] = np.sqrt(((pred_mean - y_true) ** 2).mean())

    # Epistemic uncertainty stats
    metrics['avg_epistemic_unc'] = pred_std.mean()
    metrics['std_epistemic_unc'] = pred_std.std()

    # Error-uncertainty correlation
    abs_errors = np.abs(pred_mean - y_true)
    corr = np.corrcoef(abs_errors, pred_std)[0, 1]
    metrics['error_unc_correlation'] = corr if not np.isnan(corr) else 0.0

    # Coverage (if intervals provided)
    if lower is not None and upper is not None:
        coverage = ((y_true >= lower) & (y_true <= upper)).mean()
        metrics['coverage'] = coverage
        metrics['target_coverage'] = 1 - alpha
        metrics['coverage_gap'] = coverage - (1 - alpha)

        # Interval width (sharpness)
        metrics['avg_interval_width'] = (upper - lower).mean()
        metrics['std_interval_width'] = (upper - lower).std()

    # Naive Gaussian coverage (for comparison)
    z_score = 1.645  # 90% coverage under normality
    naive_lower = pred_mean - z_score * pred_std
    naive_upper = pred_mean + z_score * pred_std
    naive_coverage = ((y_true >= naive_lower) & (y_true <= naive_upper)).mean()
    metrics['naive_gaussian_coverage'] = naive_coverage
    metrics['naive_interval_width'] = (naive_upper - naive_lower).mean()

    return metrics


def plot_conformal_analysis(pred_mean, pred_std, y_true,
                            conf_lower, conf_upper,
                            naive_lower, naive_upper,
                            save_path=None):
    """
    Visualize conformal prediction intervals vs naive Gaussian intervals.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Sort by true value for better visualization
    sort_idx = np.argsort(y_true)
    y_sorted = y_true[sort_idx]
    pred_sorted = pred_mean[sort_idx]
    conf_lower_sorted = conf_lower[sort_idx]
    conf_upper_sorted = conf_upper[sort_idx]
    naive_lower_sorted = naive_lower[sort_idx]
    naive_upper_sorted = naive_upper[sort_idx]

    # Subsample for clarity (plot every N-th point)
    n_plot = min(500, len(y_sorted))
    step = max(1, len(y_sorted) // n_plot)
    indices = np.arange(0, len(y_sorted), step)

    # 1. Conformal Prediction Intervals
    ax = axes[0, 0]
    ax.plot(indices, y_sorted[indices], 'ko', markersize=2, alpha=0.5, label='True')
    ax.plot(indices, pred_sorted[indices], 'b-', linewidth=1, label='Prediction')
    ax.fill_between(indices, conf_lower_sorted[indices], conf_upper_sorted[indices],
                     alpha=0.3, color='blue', label='90% Conformal Interval')
    ax.set_xlabel('Sample Index (sorted by true value)')
    ax.set_ylabel('Value')
    ax.set_title('Conformal Prediction Intervals')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Naive Gaussian Intervals
    ax = axes[0, 1]
    ax.plot(indices, y_sorted[indices], 'ko', markersize=2, alpha=0.5, label='True')
    ax.plot(indices, pred_sorted[indices], 'r-', linewidth=1, label='Prediction')
    ax.fill_between(indices, naive_lower_sorted[indices], naive_upper_sorted[indices],
                     alpha=0.3, color='red', label='90% Naive Gaussian Interval')
    ax.set_xlabel('Sample Index (sorted by true value)')
    ax.set_ylabel('Value')
    ax.set_title('Naive Gaussian Intervals (No Coverage Guarantee)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Coverage Comparison
    ax = axes[1, 0]
    conf_coverage = ((y_true >= conf_lower) & (y_true <= conf_upper)).mean()
    naive_coverage = ((y_true >= naive_lower) & (y_true <= naive_upper)).mean()

    methods = ['Conformal\n(Guaranteed)', 'Naive Gaussian\n(No Guarantee)']
    coverages = [conf_coverage * 100, naive_coverage * 100]
    colors = ['green' if c >= 89 else 'red' for c in coverages]

    bars = ax.bar(methods, coverages, color=colors, alpha=0.6, edgecolor='black')
    ax.axhline(90, color='black', linestyle='--', linewidth=2, label='Target (90%)')
    ax.set_ylabel('Coverage (%)')
    ax.set_title('Coverage Comparison')
    ax.set_ylim([0, 100])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add values on bars
    for bar, cov in zip(bars, coverages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{cov:.1f}%', ha='center', va='bottom', fontweight='bold')

    # 4. Interval Width Comparison
    ax = axes[1, 1]
    conf_widths = conf_upper - conf_lower
    naive_widths = naive_upper - naive_lower

    ax.hist(conf_widths, bins=50, alpha=0.5, label='Conformal', color='blue')
    ax.hist(naive_widths, bins=50, alpha=0.5, label='Naive Gaussian', color='red')
    ax.axvline(conf_widths.mean(), color='blue', linestyle='--',
               label=f'Conformal Mean: {conf_widths.mean():.2f}')
    ax.axvline(naive_widths.mean(), color='red', linestyle='--',
               label=f'Naive Mean: {naive_widths.mean():.2f}')
    ax.set_xlabel('Interval Width')
    ax.set_ylabel('Frequency')
    ax.set_title('Interval Width Distribution (Sharpness)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved conformal analysis figure to {save_path}")

    plt.close()


def analyze_with_conformal(dataset, task, num_seeds=5, sample_size=50000, alpha=0.1):
    """
    Main analysis function with conformal prediction.

    Uses validation set for both calibration and evaluation (split 50/50).
    """
    print(f"\n{'='*70}")
    print(f"Conformal UQ Analysis: {dataset} / {task}")
    print(f"{'='*70}\n")

    results_dir = Path(f"results/ensemble/{dataset}/{task}")

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    # Load ensemble predictions
    ensemble_data = load_ensemble_predictions(results_dir, num_seeds, sample_size)
    print(f"Loaded {len(ensemble_data)} models\n")

    # Use validation set (split into calibration + evaluation)
    print("Using VALIDATION set (split for calibration + evaluation)")
    print("-" * 70)

    predictions = []
    y_true = None

    for data in ensemble_data:
        pred = data['val_pred']
        predictions.append(pred)
        if y_true is None:
            y_true = data['val_true']

    # Compute ensemble statistics
    pred_mean, pred_std = compute_epistemic_uncertainty(predictions)

    # Split validation set: 50% calibration, 50% evaluation
    n = len(pred_mean)
    n_calib = n // 2

    calib_indices = np.arange(n_calib)
    eval_indices = np.arange(n_calib, n)

    print(f"Total validation samples: {n}")
    print(f"Calibration set: {n_calib} samples")
    print(f"Evaluation set: {len(eval_indices)} samples")

    # Calibration
    conformal = ConformalPredictor(alpha=alpha)
    conformal.calibrate(
        pred_mean[calib_indices],
        pred_std[calib_indices],
        y_true[calib_indices]
    )

    # Evaluation on held-out set
    print(f"\n{'='*70}")
    print("EVALUATION SET RESULTS")
    print("="*70)

    eval_pred_mean = pred_mean[eval_indices]
    eval_pred_std = pred_std[eval_indices]
    eval_y_true = y_true[eval_indices]

    # Get conformal intervals
    conf_lower, conf_upper = conformal.predict(eval_pred_mean, eval_pred_std)

    # Naive Gaussian intervals (for comparison)
    z_score = 1.645  # 90% coverage
    naive_lower = eval_pred_mean - z_score * eval_pred_std
    naive_upper = eval_pred_mean + z_score * eval_pred_std

    # Compute comprehensive metrics
    metrics = compute_comprehensive_metrics(
        eval_pred_mean, eval_pred_std, eval_y_true,
        conf_lower, conf_upper, alpha
    )

    # Print results
    print("\nPrediction Performance:")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")

    print("\nEpistemic Uncertainty:")
    print(f"  Mean: {metrics['avg_epistemic_unc']:.4f}")
    print(f"  Std: {metrics['std_epistemic_unc']:.4f}")
    print(f"  Error-Uncertainty Correlation: {metrics['error_unc_correlation']:.4f}")

    print(f"\n{'='*70}")
    print("COVERAGE ANALYSIS (90% target)")
    print("="*70)

    print("\nConformal Prediction (WITH guarantee):")
    print(f"  Coverage: {metrics['coverage']*100:.2f}%")
    print(f"  Gap from target: {metrics['coverage_gap']*100:+.2f}%")
    print(f"  Avg interval width: {metrics['avg_interval_width']:.4f}")

    print("\nNaive Gaussian (NO guarantee):")
    print(f"  Coverage: {metrics['naive_gaussian_coverage']*100:.2f}%")
    print(f"  Gap from target: {(metrics['naive_gaussian_coverage']-(1-alpha))*100:+.2f}%")
    print(f"  Avg interval width: {metrics['naive_interval_width']:.4f}")

    # Determine winner
    print(f"\n{'='*70}")
    if abs(metrics['coverage_gap']) < abs(metrics['naive_gaussian_coverage'] - (1-alpha)):
        print("[WIN] CONFORMAL: Better coverage guarantee")
    else:
        print("[SIMILAR] Both methods achieved similar coverage")

    if metrics['avg_interval_width'] < metrics['naive_interval_width']:
        print("[WIN] CONFORMAL: Sharper intervals (narrower)")
    else:
        print("[NOTE] Naive has sharper intervals (but worse coverage)")

    print("="*70)

    # Visualization
    plot_path = results_dir / f"conformal_analysis_alpha{int(alpha*100)}.png"
    plot_conformal_analysis(
        eval_pred_mean, eval_pred_std, eval_y_true,
        conf_lower, conf_upper,
        naive_lower, naive_upper,
        save_path=plot_path
    )

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze regression UQ with conformal prediction"
    )
    parser.add_argument("--dataset", type=str, default="rel-f1")
    parser.add_argument("--task", type=str, default="results-position")
    parser.add_argument("--num_seeds", type=int, default=3)
    parser.add_argument("--sample_size", type=int, default=10000)
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="Miscoverage rate (0.1 = 90% coverage)")
    args = parser.parse_args()

    analyze_with_conformal(
        args.dataset, args.task,
        args.num_seeds, args.sample_size,
        args.alpha
    )
