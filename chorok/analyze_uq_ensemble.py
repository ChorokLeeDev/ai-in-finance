"""
Uncertainty Quantification Analysis for LightGBM Ensemble
Analyzes prediction variance across 5 independently trained models.
"""
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# Set style
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    plt.style.use('ggplot')

def load_predictions(model_paths):
    """Load predictions from saved models."""
    all_predictions = []
    test_true = None

    for model_path in model_paths:
        print(f"Loading {model_path}...")
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            # Extract test predictions (key is 'test_pred' not 'test_predictions')
            test_preds = data.get('test_pred', None)
            if test_preds is not None:
                all_predictions.append(test_preds)
                # Save ground truth from first model
                if test_true is None:
                    test_true = data.get('test_true', None)

    predictions = np.array(all_predictions)  # Shape: (n_models, n_samples, n_classes)
    return predictions, test_true

def calculate_uncertainties(predictions):
    """
    Calculate various uncertainty metrics.

    Args:
        predictions: np.array of shape (n_models, n_samples, n_classes)

    Returns:
        Dictionary with uncertainty metrics
    """
    # Mean prediction across models
    mean_pred = np.mean(predictions, axis=0)  # (n_samples, n_classes)

    # Variance across models (aleatoric + epistemic uncertainty)
    variance = np.var(predictions, axis=0)  # (n_samples, n_classes)
    std = np.std(predictions, axis=0)

    # Total uncertainty per sample (averaged over classes)
    total_uncertainty = np.mean(variance, axis=1)

    # Predictive entropy
    epsilon = 1e-10
    entropy = -np.sum(mean_pred * np.log(mean_pred + epsilon), axis=1)

    # Confidence (max probability)
    confidence = np.max(mean_pred, axis=1)

    # Prediction disagreement (coefficient of variation)
    mean_std_per_sample = np.mean(std, axis=1)
    mean_pred_per_sample = np.mean(mean_pred, axis=1)
    disagreement = mean_std_per_sample / (mean_pred_per_sample + epsilon)

    return {
        'mean_prediction': mean_pred,
        'variance': variance,
        'std': std,
        'total_uncertainty': total_uncertainty,
        'entropy': entropy,
        'confidence': confidence,
        'disagreement': disagreement
    }

def analyze_uncertainty_patterns(uncertainties, output_dir):
    """Analyze and visualize uncertainty patterns."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Distribution of uncertainties
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics = [
        ('total_uncertainty', 'Total Uncertainty'),
        ('entropy', 'Predictive Entropy'),
        ('confidence', 'Prediction Confidence'),
        ('disagreement', 'Model Disagreement')
    ]

    for idx, (key, title) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        data = uncertainties[key]

        ax.hist(data, bins=50, alpha=0.7, edgecolor='black')
        ax.set_xlabel(title, fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'Distribution of {title}', fontsize=14)
        ax.axvline(np.mean(data), color='red', linestyle='--',
                   label=f'Mean: {np.mean(data):.4f}')
        ax.axvline(np.median(data), color='green', linestyle='--',
                   label=f'Median: {np.median(data):.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'uncertainty_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Correlation between uncertainty metrics
    fig, ax = plt.subplots(figsize=(10, 8))

    correlation_data = pd.DataFrame({
        'Total Uncertainty': uncertainties['total_uncertainty'],
        'Entropy': uncertainties['entropy'],
        'Confidence': uncertainties['confidence'],
        'Disagreement': uncertainties['disagreement']
    })

    corr_matrix = correlation_data.corr()

    # Create heatmap without seaborn
    im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_matrix.columns)))
    ax.set_yticks(range(len(corr_matrix.columns)))
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
    ax.set_yticklabels(corr_matrix.columns)

    # Add correlation values as text
    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix)):
            text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.3f}',
                          ha='center', va='center', color='black', fontsize=10)

    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title('Correlation Between Uncertainty Metrics', fontsize=16, pad=20)

    plt.tight_layout()
    plt.savefig(output_dir / 'uncertainty_correlations.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Confidence vs Uncertainty scatter
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Confidence vs Total Uncertainty
    ax = axes[0]
    scatter = ax.scatter(uncertainties['confidence'],
                        uncertainties['total_uncertainty'],
                        alpha=0.3, s=10, c=uncertainties['entropy'],
                        cmap='viridis')
    ax.set_xlabel('Prediction Confidence', fontsize=12)
    ax.set_ylabel('Total Uncertainty', fontsize=12)
    ax.set_title('Confidence vs Uncertainty', fontsize=14)
    plt.colorbar(scatter, ax=ax, label='Entropy')
    ax.grid(True, alpha=0.3)

    # Confidence vs Disagreement
    ax = axes[1]
    scatter = ax.scatter(uncertainties['confidence'],
                        uncertainties['disagreement'],
                        alpha=0.3, s=10, c=uncertainties['entropy'],
                        cmap='viridis')
    ax.set_xlabel('Prediction Confidence', fontsize=12)
    ax.set_ylabel('Model Disagreement', fontsize=12)
    ax.set_title('Confidence vs Model Disagreement', fontsize=14)
    plt.colorbar(scatter, ax=ax, label='Entropy')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'confidence_vs_uncertainty.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Summary statistics
    summary_stats = {
        'Metric': [],
        'Mean': [],
        'Std': [],
        'Min': [],
        'Q1': [],
        'Median': [],
        'Q3': [],
        'Max': []
    }

    for key, title in metrics:
        data = uncertainties[key]
        summary_stats['Metric'].append(title)
        summary_stats['Mean'].append(np.mean(data))
        summary_stats['Std'].append(np.std(data))
        summary_stats['Min'].append(np.min(data))
        summary_stats['Q1'].append(np.percentile(data, 25))
        summary_stats['Median'].append(np.median(data))
        summary_stats['Q3'].append(np.percentile(data, 75))
        summary_stats['Max'].append(np.max(data))

    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(output_dir / 'uncertainty_summary.csv', index=False)

    print("\n" + "="*80)
    print("UNCERTAINTY QUANTIFICATION SUMMARY")
    print("="*80)
    print(summary_df.to_string(index=False))
    print("="*80)

    return summary_df

def analyze_high_uncertainty_samples(uncertainties, predictions, output_dir, top_k=100):
    """Analyze samples with highest uncertainty."""
    output_dir = Path(output_dir)

    # Get top-k most uncertain samples
    uncertain_idx = np.argsort(uncertainties['total_uncertainty'])[-top_k:]
    certain_idx = np.argsort(uncertainties['total_uncertainty'])[:top_k]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Compare characteristics
    metrics = [
        ('total_uncertainty', 'Total Uncertainty'),
        ('entropy', 'Entropy'),
        ('confidence', 'Confidence'),
        ('disagreement', 'Disagreement')
    ]

    for idx, (key, title) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]

        data_uncertain = uncertainties[key][uncertain_idx]
        data_certain = uncertainties[key][certain_idx]

        ax.hist([data_certain, data_uncertain], bins=30, label=['Most Certain', 'Most Uncertain'],
                alpha=0.7, edgecolor='black')
        ax.set_xlabel(title, fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'{title}: Certain vs Uncertain Samples', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'certain_vs_uncertain_samples.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Analyze prediction variance for uncertain samples
    fig, ax = plt.subplots(figsize=(12, 6))

    # Select a few uncertain samples to visualize
    sample_indices = uncertain_idx[:20]

    for i, sample_idx in enumerate(sample_indices):
        # Get predictions from all models for this sample
        sample_preds = predictions[:, sample_idx, :]  # (n_models, n_classes)

        # Plot variance across classes
        variance = np.var(sample_preds, axis=0)
        ax.plot(variance, marker='o', alpha=0.5, label=f'Sample {i+1}' if i < 5 else '')

    ax.set_xlabel('Class Index', fontsize=12)
    ax.set_ylabel('Prediction Variance', fontsize=12)
    ax.set_title('Prediction Variance Across Classes (Top 20 Uncertain Samples)', fontsize=14)
    if len(sample_indices) <= 5:
        ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'uncertain_samples_variance.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main analysis pipeline."""
    print("="*80)
    print("UNCERTAINTY QUANTIFICATION ANALYSIS FOR LIGHTGBM ENSEMBLE")
    print("="*80)

    # Define model paths - using actual filenames from lightgbm_autocomplete.py
    results_dir = Path('chorok/results')
    seeds = [42, 123, 456, 789, 1011]
    sample_size = 10000
    model_paths = [results_dir / f'seed_{seed}_sample_{sample_size}.pkl' for seed in seeds]

    # Check if models exist
    missing_models = [p for p in model_paths if not p.exists()]
    if missing_models:
        print(f"\nERROR: Missing models: {missing_models}")
        print(f"\nSearching in directory: {results_dir}")
        if results_dir.exists():
            print(f"Files in directory: {list(results_dir.glob('*.pkl'))}")
        return

    print(f"\nFound {len(model_paths)} models")
    print(f"Seeds: {seeds}")

    # Load predictions
    print("\nLoading predictions...")
    predictions, test_true = load_predictions(model_paths)
    print(f"Predictions shape: {predictions.shape}")
    print(f"  - Number of models: {predictions.shape[0]}")
    print(f"  - Number of samples: {predictions.shape[1]}")
    print(f"  - Number of classes: {predictions.shape[2]}")
    if test_true is not None:
        print(f"Ground truth shape: {test_true.shape}")

    # Calculate uncertainties
    print("\nCalculating uncertainty metrics...")
    uncertainties = calculate_uncertainties(predictions)

    # Create output directory
    output_dir = Path('chorok/uq_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Analyze patterns
    print("\nAnalyzing uncertainty patterns...")
    summary_df = analyze_uncertainty_patterns(uncertainties, output_dir)

    # Analyze high uncertainty samples
    print("\nAnalyzing high uncertainty samples...")
    analyze_high_uncertainty_samples(uncertainties, predictions, output_dir, top_k=100)

    # Save results
    print("\nSaving results...")
    np.savez_compressed(
        output_dir / 'uncertainties.npz',
        mean_prediction=uncertainties['mean_prediction'],
        variance=uncertainties['variance'],
        std=uncertainties['std'],
        total_uncertainty=uncertainties['total_uncertainty'],
        entropy=uncertainties['entropy'],
        confidence=uncertainties['confidence'],
        disagreement=uncertainties['disagreement']
    )

    print(f"\nAll results saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - uncertainty_distributions.png")
    print("  - uncertainty_correlations.png")
    print("  - confidence_vs_uncertainty.png")
    print("  - certain_vs_uncertain_samples.png")
    print("  - uncertain_samples_variance.png")
    print("  - uncertainty_summary.csv")
    print("  - uncertainties.npz")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)

if __name__ == '__main__':
    main()
