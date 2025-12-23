"""
Causal FK Hypothesis Validation Experiment v2

Uses proper FK extraction from fk_active_learning.py

Hypothesis: "Causal FKs increase epistemic uncertainty, correlational FKs stabilize"
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from fk_active_learning import extract_features_with_fk, train_ensemble, ensemble_variance
from relbench.datasets import get_dataset
from relbench.tasks import get_task


# FK Classification (from domain knowledge)
FK_CLASSIFICATION = {
    'rel-f1': {
        'RESULTS': 'causal',      # Race results directly determine position
        'QUALIFYING': 'correlational',  # Qualifying is correlated, not causal
        'STANDINGS': 'correlational',   # Historical standings are correlational
        'TRAIN': 'unknown',
        'DRIVERS': 'causal',      # Driver identity affects performance
    },
    'rel-trial': {
        'INTERVENTIONS': 'causal',     # Drug directly causes outcome
        'CONDITIONS': 'causal',        # Disease type affects outcome
        'FACILITIES': 'correlational', # Hospital location doesn't cause efficacy
        'SPONSORS': 'correlational',   # Funding source doesn't cause efficacy
        'STUDIES': 'unknown',
    }
}


def compute_fk_uncertainty_contribution(models, X, fk_to_cols, n_permutations=10):
    """
    Compute FK-level uncertainty contribution via permutation.

    Positive value = FK contributes to uncertainty (removing it decreases uncertainty)
    Negative value = FK stabilizes predictions (removing it increases uncertainty)
    """
    base_uncertainty = np.mean(ensemble_variance(models, X))

    fk_contributions = {}

    for fk_name, col_indices in fk_to_cols.items():
        if not col_indices:
            continue

        contributions = []
        for _ in range(n_permutations):
            X_perm = X.copy()
            for col_idx in col_indices:
                X_perm[:, col_idx] = np.random.permutation(X_perm[:, col_idx])

            perm_uncertainty = np.mean(ensemble_variance(models, X_perm))

            # If permuting increases uncertainty, FK was stabilizing (negative contribution)
            # If permuting decreases uncertainty, FK was causing uncertainty (positive contribution)
            contribution = (base_uncertainty - perm_uncertainty) / base_uncertainty * 100
            contributions.append(contribution)

        fk_contributions[fk_name] = {
            'mean': np.mean(contributions),
            'std': np.std(contributions),
            'raw': contributions
        }

    return fk_contributions, base_uncertainty


def run_experiment(dataset_name, task_name, n_models=5, sample_size=3000, seed=42):
    """Run experiment for a single task."""
    print(f"\n{'='*60}")
    print(f"Running: {dataset_name} / {task_name}")
    print(f"{'='*60}")

    # Load data
    print("Loading dataset...")
    dataset = get_dataset(dataset_name, download=True)
    task = get_task(dataset_name, task_name, download=True)

    # Extract features with FK groups
    print("Extracting features...")
    try:
        X, y, col_to_fk, feature_cols, fk_to_cols, fk_value_cols, fk_table_names = \
            extract_features_with_fk(dataset, task, sample_size=sample_size)
    except Exception as e:
        print(f"Feature extraction failed: {e}")
        return None

    print(f"  Data shape: {X.shape}")
    print(f"  FK groups: {list(fk_to_cols.keys())}")
    for fk, cols in fk_to_cols.items():
        print(f"    {fk}: {len(cols)} columns")

    if X.shape[1] < 2:
        print("  ERROR: Not enough features")
        return None

    # Train ensemble
    print(f"Training ensemble ({n_models} models)...")
    models = train_ensemble(X, y, n_models=n_models, seed=seed)

    # Compute FK contributions
    print("Computing FK uncertainty contributions...")
    fk_contributions, base_uncertainty = compute_fk_uncertainty_contribution(
        models, X, fk_to_cols, n_permutations=10
    )

    print(f"\n  Base uncertainty: {base_uncertainty:.4f}")
    print("  FK contributions:")
    for fk_name, contrib in sorted(fk_contributions.items(), key=lambda x: -x[1]['mean']):
        print(f"    {fk_name}: {contrib['mean']:+.2f}% ± {contrib['std']:.2f}%")

    # Classify FKs
    classification = FK_CLASSIFICATION.get(dataset_name, {})
    causal_contribs = []
    correlational_contribs = []

    for fk_name, contrib in fk_contributions.items():
        fk_type = classification.get(fk_name, 'unknown')
        if fk_type == 'causal':
            causal_contribs.append(contrib['mean'])
        elif fk_type == 'correlational':
            correlational_contribs.append(contrib['mean'])

    # Statistical analysis
    stats_result = {}
    if causal_contribs and correlational_contribs:
        stats_result['causal_mean'] = np.mean(causal_contribs)
        stats_result['correlational_mean'] = np.mean(correlational_contribs)
        stats_result['difference'] = stats_result['causal_mean'] - stats_result['correlational_mean']

        # Cohen's d
        pooled_std = np.sqrt((np.var(causal_contribs) + np.var(correlational_contribs)) / 2)
        stats_result['cohens_d'] = stats_result['difference'] / pooled_std if pooled_std > 0 else 0

        # Hypothesis: causal > correlational
        hypothesis_supported = stats_result['difference'] > 0
        stats_result['hypothesis_supported'] = hypothesis_supported

        print(f"\n  Classification results:")
        print(f"    Causal FKs mean:        {stats_result['causal_mean']:+.2f}%")
        print(f"    Correlational FKs mean: {stats_result['correlational_mean']:+.2f}%")
        print(f"    Difference:             {stats_result['difference']:+.2f}%")
        print(f"    Cohen's d:              {stats_result['cohens_d']:.2f}")
        print(f"    Hypothesis supported:   {'YES' if hypothesis_supported else 'NO'}")

    return {
        'dataset': dataset_name,
        'task': task_name,
        'timestamp': datetime.now().isoformat(),
        'data_shape': list(X.shape),
        'fk_groups': {k: len(v) for k, v in fk_to_cols.items()},
        'fk_contributions': {k: v['mean'] for k, v in fk_contributions.items()},
        'fk_contributions_std': {k: v['std'] for k, v in fk_contributions.items()},
        'statistics': stats_result,
        'base_uncertainty': base_uncertainty
    }


def run_multi_seed(dataset_name, task_name, seeds=[42, 43, 44, 45, 46]):
    """Run experiment with multiple seeds for robustness."""
    print(f"\n{'#'*60}")
    print(f"MULTI-SEED VALIDATION: {dataset_name} / {task_name}")
    print(f"{'#'*60}")

    all_results = []
    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        result = run_experiment(dataset_name, task_name, seed=seed)
        if result:
            all_results.append(result)

    if not all_results:
        return None

    # Aggregate
    print(f"\n{'='*60}")
    print("MULTI-SEED SUMMARY")
    print(f"{'='*60}")

    # Get all FK names
    all_fks = set()
    for r in all_results:
        all_fks.update(r['fk_contributions'].keys())

    print("\nFK Contribution Across Seeds:")
    for fk in sorted(all_fks):
        contribs = [r['fk_contributions'].get(fk, 0) for r in all_results]
        print(f"  {fk}: {np.mean(contribs):+.2f}% ± {np.std(contribs):.2f}%")

    # Hypothesis check
    supported_count = sum(1 for r in all_results if r['statistics'].get('hypothesis_supported', False))
    print(f"\nHypothesis supported: {supported_count}/{len(all_results)} seeds")

    return {
        'dataset': dataset_name,
        'task': task_name,
        'seeds': seeds,
        'results': all_results,
        'hypothesis_support_rate': supported_count / len(all_results)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='rel-f1')
    parser.add_argument('--task', type=str, default='driver-position')
    parser.add_argument('--multi-seed', action='store_true')
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()

    if args.all:
        # Run on all available tasks
        experiments = [
            ('rel-f1', 'driver-position'),
            ('rel-f1', 'driver-dnf'),
            ('rel-f1', 'driver-top3'),
        ]

        all_results = []
        for dataset, task in experiments:
            if args.multi_seed:
                result = run_multi_seed(dataset, task)
            else:
                result = run_experiment(dataset, task)
            if result:
                all_results.append(result)

        # Save
        output_path = 'test_results/causal_fk_validation_v2.json'
        os.makedirs('test_results', exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nResults saved to {output_path}")

    elif args.multi_seed:
        run_multi_seed(args.dataset, args.task)
    else:
        run_experiment(args.dataset, args.task)


if __name__ == '__main__':
    main()
