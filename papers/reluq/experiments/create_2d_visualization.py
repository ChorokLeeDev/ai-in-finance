"""
Create 2D Importance × Stability Visualization

This is the "hero figure" for the RelUQ paper.

X-axis: FK Importance (permutation importance for accuracy)
Y-axis: FK Uncertainty Contribution (stability)
Color: Domain
Shape: Task type
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

import lightgbm as lgb
from relbench.datasets import get_dataset
from relbench.tasks import get_task


# Domain colors
DOMAIN_COLORS = {
    'rel-f1': '#1f77b4',      # Blue
    'rel-trial': '#2ca02c',   # Green
    'rel-salt': '#ff7f0e',    # Orange
    'rel-avito': '#9467bd',   # Purple
}

# Task type markers
TASK_MARKERS = {
    'regression': 'o',        # Circle
    'classification': 's',    # Square
}


def compute_fk_importance(models, X, fk_to_cols, y, is_classification=False):
    """Compute FK-level permutation importance for ACCURACY (not uncertainty)."""
    # Base performance
    if is_classification:
        base_preds = np.array([m.predict(X) for m in models]).mean(axis=0)
        base_acc = (base_preds.round() == y).mean()
    else:
        base_preds = np.array([m.predict(X) for m in models]).mean(axis=0)
        base_mae = np.abs(base_preds - y).mean()

    fk_importance = {}

    for fk_name, col_indices in fk_to_cols.items():
        if not col_indices:
            continue

        importances = []
        for _ in range(5):
            X_perm = X.copy()
            for col_idx in col_indices:
                X_perm[:, col_idx] = np.random.permutation(X_perm[:, col_idx])

            if is_classification:
                perm_preds = np.array([m.predict(X_perm) for m in models]).mean(axis=0)
                perm_acc = (perm_preds.round() == y).mean()
                imp = (base_acc - perm_acc) / base_acc * 100  # % accuracy drop
            else:
                perm_preds = np.array([m.predict(X_perm) for m in models]).mean(axis=0)
                perm_mae = np.abs(perm_preds - y).mean()
                imp = (perm_mae - base_mae) / base_mae * 100  # % MAE increase

            importances.append(imp)

        fk_importance[fk_name] = np.mean(importances)

    return fk_importance


def load_and_compute_all():
    """Load all datasets and compute importance + uncertainty for each FK."""
    results = []

    # Define all experiments
    experiments = [
        # rel-f1
        ('rel-f1', 'driver-position', 'regression'),
        ('rel-f1', 'driver-dnf', 'classification'),
        ('rel-f1', 'driver-top3', 'classification'),
        # rel-trial
        ('rel-trial', 'study-outcome', 'regression'),
        ('rel-trial', 'study-adverse', 'regression'),
        ('rel-trial', 'site-success', 'regression'),
        # rel-salt
        ('rel-salt', 'item-plant', 'classification'),
        ('rel-salt', 'item-shippoint', 'classification'),
        ('rel-salt', 'sales-payterms', 'classification'),
        # rel-avito
        ('rel-avito', 'ad-ctr', 'regression'),
        ('rel-avito', 'user-clicks', 'classification'),
    ]

    for dataset_name, task_name, task_type in experiments:
        print(f"\nProcessing: {dataset_name} / {task_name}")

        try:
            # Load dataset and task
            dataset = get_dataset(dataset_name, download=False)

            # Handle download parameter
            try:
                task = get_task(dataset_name, task_name, download=False)
            except:
                task = get_task(dataset_name, task_name, download=True)

            # Extract features using the appropriate method
            if dataset_name == 'rel-f1':
                from fk_active_learning import extract_features_with_fk, train_ensemble, ensemble_variance
                X, y, col_to_fk, feature_cols, fk_to_cols, _, _ = extract_features_with_fk(
                    dataset, task, sample_size=2000
                )
                is_classification = task_type == 'classification'

                # Train ensemble
                models = train_ensemble(X, y, n_models=5, seed=42)

                # Compute uncertainty
                base_unc = ensemble_variance(models, X).mean()
                fk_uncertainty = {}
                for fk_name, col_indices in fk_to_cols.items():
                    if not col_indices:
                        continue
                    contributions = []
                    for _ in range(5):
                        X_perm = X.copy()
                        for col_idx in col_indices:
                            X_perm[:, col_idx] = np.random.permutation(X_perm[:, col_idx])
                        perm_unc = ensemble_variance(models, X_perm).mean()
                        contrib = (base_unc - perm_unc) / base_unc * 100
                        contributions.append(contrib)
                    fk_uncertainty[fk_name] = np.mean(contributions)

            elif dataset_name == 'rel-trial':
                from run_trial_validation import extract_features_with_fk as extract_trial
                # Simplified extraction
                from run_trial_validation import (
                    compute_fk_uncertainty_contribution as compute_trial_unc
                )
                from fk_active_learning import extract_features_with_fk, train_ensemble, ensemble_variance

                X, y, col_to_fk, feature_cols, fk_to_cols, _, _ = extract_features_with_fk(
                    dataset, task, sample_size=2000
                )
                is_classification = False

                models = train_ensemble(X, y, n_models=5, seed=42)

                base_unc = ensemble_variance(models, X).mean()
                fk_uncertainty = {}
                for fk_name, col_indices in fk_to_cols.items():
                    if not col_indices:
                        continue
                    contributions = []
                    for _ in range(5):
                        X_perm = X.copy()
                        for col_idx in col_indices:
                            X_perm[:, col_idx] = np.random.permutation(X_perm[:, col_idx])
                        perm_unc = ensemble_variance(models, X_perm).mean()
                        contrib = (base_unc - perm_unc) / base_unc * 100
                        contributions.append(contrib)
                    fk_uncertainty[fk_name] = np.mean(contributions)

            elif dataset_name == 'rel-salt':
                from run_salt_validation import (
                    extract_features_with_fk_classification,
                    train_ensemble_classifier,
                    ensemble_variance_classifier
                )
                X, y, col_to_fk, feature_cols, fk_to_cols, n_classes = extract_features_with_fk_classification(
                    dataset, task, sample_size=2000
                )
                is_classification = True

                models = train_ensemble_classifier(X, y, n_models=5, n_classes=n_classes, seed=42)

                base_unc = ensemble_variance_classifier(models, X).mean()
                fk_uncertainty = {}
                for fk_name, col_indices in fk_to_cols.items():
                    if not col_indices:
                        continue
                    contributions = []
                    for _ in range(5):
                        X_perm = X.copy()
                        for col_idx in col_indices:
                            X_perm[:, col_idx] = np.random.permutation(X_perm[:, col_idx])
                        perm_unc = ensemble_variance_classifier(models, X_perm).mean()
                        contrib = (base_unc - perm_unc) / base_unc * 100
                        contributions.append(contrib)
                    fk_uncertainty[fk_name] = np.mean(contributions)

            elif dataset_name == 'rel-avito':
                from run_avito_validation import (
                    extract_features_generic,
                    train_ensemble,
                    ensemble_variance
                )
                X, y, col_to_fk, feature_cols, fk_to_cols, n_classes, is_classification = extract_features_generic(
                    dataset, task, sample_size=2000
                )

                models = train_ensemble(X, y, n_models=5, n_classes=n_classes,
                                       is_classification=is_classification, seed=42)

                base_unc = ensemble_variance(models, X, is_classification).mean()
                fk_uncertainty = {}
                for fk_name, col_indices in fk_to_cols.items():
                    if not col_indices:
                        continue
                    contributions = []
                    for _ in range(5):
                        X_perm = X.copy()
                        for col_idx in col_indices:
                            X_perm[:, col_idx] = np.random.permutation(X_perm[:, col_idx])
                        perm_unc = ensemble_variance(models, X_perm, is_classification).mean()
                        contrib = (base_unc - perm_unc) / base_unc * 100
                        contributions.append(contrib)
                    fk_uncertainty[fk_name] = np.mean(contributions)

            # Compute importance
            fk_importance = compute_fk_importance(models, X, fk_to_cols, y, is_classification)

            # Store results
            for fk_name in fk_uncertainty.keys():
                if fk_name not in fk_importance:
                    continue

                results.append({
                    'dataset': dataset_name,
                    'task': task_name,
                    'task_type': task_type,
                    'fk': fk_name,
                    'importance': fk_importance[fk_name],
                    'uncertainty': fk_uncertainty[fk_name],
                })

            print(f"  Found {len(fk_uncertainty)} FKs")

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    return results


def create_visualization(results, output_path='figures/importance_stability_2d.png'):
    """Create the 2D visualization."""

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot each FK
    for r in results:
        color = DOMAIN_COLORS.get(r['dataset'], 'gray')
        marker = TASK_MARKERS.get(r['task_type'], 'o')

        ax.scatter(
            r['importance'],
            r['uncertainty'],
            c=color,
            marker=marker,
            s=120,
            alpha=0.7,
            edgecolors='white',
            linewidths=1
        )

        # Add label for notable FKs (high importance or high uncertainty)
        if abs(r['uncertainty']) > 10 or abs(r['importance']) > 10:
            ax.annotate(
                f"{r['fk']}\n({r['task']})",
                (r['importance'], r['uncertainty']),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
                alpha=0.8
            )

    # Add quadrant labels
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    # Quadrant annotations
    ax.text(15, 15, '🟡 Noisy Signal\n(Important + Uncertain)\n→ Collect more data',
            fontsize=10, ha='center', va='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.text(15, -50, '🟢 Stable Signal\n(Important + Stable)\n→ Data sufficient',
            fontsize=10, ha='center', va='center', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    ax.text(-5, 15, '🔴 Pure Noise\n(Not important + Uncertain)\n→ Consider removing',
            fontsize=10, ha='center', va='center', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))

    ax.text(-5, -50, '⚪ Irrelevant\n(Not important + Stable)\n→ Ignore',
            fontsize=10, ha='center', va='center', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

    # Axis labels
    ax.set_xlabel('FK Importance (% accuracy/MAE change when permuted)', fontsize=12)
    ax.set_ylabel('FK Uncertainty Contribution (% variance change when permuted)', fontsize=12)
    ax.set_title('FK-Level Importance × Stability: Data Investment Framework', fontsize=14, fontweight='bold')

    # Legend for domains
    domain_patches = [mpatches.Patch(color=color, label=domain.replace('rel-', '').upper())
                     for domain, color in DOMAIN_COLORS.items()]

    # Legend for task types
    from matplotlib.lines import Line2D
    task_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='Regression'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=10, label='Classification'),
    ]

    # Combine legends
    first_legend = ax.legend(handles=domain_patches, loc='upper left', title='Domain')
    ax.add_artist(first_legend)
    ax.legend(handles=task_handles, loc='lower right', title='Task Type')

    # Grid
    ax.grid(True, alpha=0.3)

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {output_path}")

    # Also save as PDF for paper
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    print(f"PDF saved to: {pdf_path}")

    plt.close()

    return output_path


def main():
    print("="*60)
    print("Creating 2D Importance × Stability Visualization")
    print("="*60)

    # Compute all results
    results = load_and_compute_all()

    print(f"\n\nTotal FK data points: {len(results)}")

    # Save raw data
    output_dir = Path(__file__).parent / 'figures'
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / 'importance_stability_data.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Create visualization
    fig_path = str(output_dir / 'importance_stability_2d.png')
    create_visualization(results, fig_path)

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    print("\nNoisy FKs (uncertainty > +5%):")
    for r in sorted(results, key=lambda x: -x['uncertainty']):
        if r['uncertainty'] > 5:
            print(f"  {r['dataset']}/{r['task']}: {r['fk']} = {r['uncertainty']:+.1f}% unc, {r['importance']:+.1f}% imp")

    print("\nStable FKs (uncertainty < -50%):")
    for r in sorted(results, key=lambda x: x['uncertainty']):
        if r['uncertainty'] < -50:
            print(f"  {r['dataset']}/{r['task']}: {r['fk']} = {r['uncertainty']:+.1f}% unc, {r['importance']:+.1f}% imp")


if __name__ == '__main__':
    main()
