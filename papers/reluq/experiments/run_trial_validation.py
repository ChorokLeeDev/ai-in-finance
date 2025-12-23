"""
Run FK Uncertainty Framework validation on rel-trial dataset
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import json
import os
from datetime import datetime
from fk_active_learning import extract_features_with_fk, train_ensemble, ensemble_variance
from relbench.datasets import get_dataset
from relbench.tasks import get_task

# FK Classification for rel-trial based on domain knowledge
# Causal: directly affects outcome
# Correlational: associated but doesn't cause outcome
TRIAL_FK_CLASSIFICATION = {
    'INTERVENTIONS_STUDIES': 'causal',      # Drug/treatment directly causes outcome
    'CONDITIONS_STUDIES': 'causal',          # Disease type directly affects outcome
    'FACILITIES_STUDIES': 'correlational',   # Hospital location doesn't cause drug efficacy
    'SPONSORS_STUDIES': 'correlational',     # Funding source doesn't cause drug efficacy
    'STUDIES': 'causal',                     # Study design affects outcome
    'OUTCOMES': 'causal',                    # Outcome measures affect reported results
    'DESIGNS': 'causal',                     # Study design is causal
    'ELIGIBILITIES': 'correlational',        # Eligibility criteria are correlational
    'DROP_WITHDRAWALS': 'correlational',     # Dropouts are post-hoc, correlational
    'OUTCOME_ANALYSES': 'correlational',     # Analysis methods are correlational
    'REPORTED_EVENT_TOTALS': 'correlational' # Reported events are observational
}


def compute_fk_uncertainty_contribution(models, X, fk_to_cols, n_permutations=10):
    """Compute FK-level uncertainty contribution via permutation."""
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
            contribution = (base_uncertainty - perm_uncertainty) / base_uncertainty * 100
            contributions.append(contribution)

        fk_contributions[fk_name] = {
            'mean': np.mean(contributions),
            'std': np.std(contributions)
        }

    return fk_contributions, base_uncertainty


def run_experiment(task_name, n_models=5, sample_size=2000, seed=42):
    """Run experiment for a single task."""
    print(f"\n{'='*60}")
    print(f"Running: rel-trial / {task_name}")
    print(f"{'='*60}")

    dataset = get_dataset('rel-trial', download=True)
    task = get_task('rel-trial', task_name, download=True)

    print("Extracting features...")
    try:
        X, y, col_to_fk, feature_cols, fk_to_cols, _, _ = extract_features_with_fk(
            dataset, task, sample_size=sample_size
        )
    except Exception as e:
        print(f"  Error: {e}")
        return None

    print(f"  Data shape: {X.shape}")
    print(f"  FK groups: {list(fk_to_cols.keys())}")

    if X.shape[1] < 2:
        print("  ERROR: Not enough features")
        return None

    print(f"Training ensemble ({n_models} models)...")
    models = train_ensemble(X, y, n_models=n_models, seed=seed)

    print("Computing FK uncertainty contributions...")
    fk_contributions, base_uncertainty = compute_fk_uncertainty_contribution(
        models, X, fk_to_cols
    )

    print(f"\n  Base uncertainty: {base_uncertainty:.6f}")
    print("  FK contributions:")
    for fk_name, contrib in sorted(fk_contributions.items(), key=lambda x: -x[1]['mean']):
        fk_type = TRIAL_FK_CLASSIFICATION.get(fk_name, 'unknown')
        print(f"    {fk_name}: {contrib['mean']:+.2f}% ({fk_type})")

    # Classify
    causal = [c['mean'] for fk, c in fk_contributions.items()
              if TRIAL_FK_CLASSIFICATION.get(fk) == 'causal']
    correlational = [c['mean'] for fk, c in fk_contributions.items()
                    if TRIAL_FK_CLASSIFICATION.get(fk) == 'correlational']

    if causal and correlational:
        causal_mean = np.mean(causal)
        corr_mean = np.mean(correlational)
        diff = causal_mean - corr_mean
        print(f"\n  Causal FKs mean: {causal_mean:+.2f}%")
        print(f"  Correlational FKs mean: {corr_mean:+.2f}%")
        print(f"  Difference: {diff:+.2f}%")
        print(f"  Original hypothesis supported: {'YES' if diff > 0 else 'NO'}")

    return {
        'task': task_name,
        'seed': seed,
        'data_shape': list(X.shape),
        'fk_contributions': {k: v['mean'] for k, v in fk_contributions.items()},
        'base_uncertainty': base_uncertainty
    }


def run_multi_seed(task_name, seeds=[42, 43, 44, 45, 46]):
    """Run with multiple seeds."""
    print(f"\n{'#'*60}")
    print(f"MULTI-SEED: rel-trial / {task_name}")
    print(f"{'#'*60}")

    results = []
    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        r = run_experiment(task_name, seed=seed)
        if r:
            results.append(r)

    if not results:
        return None

    # Aggregate
    print(f"\n{'='*60}")
    print("MULTI-SEED SUMMARY")
    print(f"{'='*60}")

    all_fks = set()
    for r in results:
        all_fks.update(r['fk_contributions'].keys())

    print("\nFK Contribution Across Seeds:")
    fk_summary = {}
    for fk in sorted(all_fks):
        contribs = [r['fk_contributions'].get(fk, 0) for r in results]
        mean_contrib = np.mean(contribs)
        std_contrib = np.std(contribs)
        fk_type = TRIAL_FK_CLASSIFICATION.get(fk, 'unknown')
        print(f"  {fk}: {mean_contrib:+.2f}% ± {std_contrib:.2f}% ({fk_type})")
        fk_summary[fk] = {'mean': mean_contrib, 'std': std_contrib, 'type': fk_type}

    return {
        'task': task_name,
        'seeds': seeds,
        'fk_summary': fk_summary,
        'results': results
    }


def main():
    tasks = ['study-outcome', 'study-adverse', 'site-success']

    all_results = []
    for task in tasks:
        try:
            result = run_multi_seed(task)
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"Error with {task}: {e}")

    # Save results
    os.makedirs('test_results', exist_ok=True)
    with open('test_results/trial_validation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n\nResults saved to test_results/trial_validation_results.json")


if __name__ == '__main__':
    main()
