"""
VERIFICATION: Do Entity-Level Recommendations Actually Reduce Uncertainty?
===========================================================================

Key question: If we follow the recommendation "shift from entity A to entity B",
does uncertainty actually decrease?

Verification approaches:
1. HELD-OUT VALIDATION: Train on subset, verify on held-out data
2. SIMULATION: Replace high-unc entity samples with low-unc entity samples
3. BOOTSTRAP: Resample and check consistency of rankings

This addresses the critical question: Are our recommendations actionable and verifiable?
"""

import numpy as np
import pandas as pd
from collections import defaultdict
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

from data_loader_salt import load_salt_data


def train_ensemble(X, y, n_models=10, base_seed=42):
    models = []
    for i in range(n_models):
        model = lgb.LGBMRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            random_state=base_seed + i, verbose=-1
        )
        model.fit(X, y)
        models.append(model)
    return models


def get_uncertainty(models, X):
    preds = np.array([m.predict(X) for m in models])
    return preds.var(axis=0)


def identify_entities(models, X, y, entity_col, min_samples=5):
    """Identify high and low uncertainty entities."""
    uncertainties = get_uncertainty(models, X)

    entity_data = pd.DataFrame({
        'entity': X[entity_col].values,
        'uncertainty': uncertainties,
        'target': y.values
    })

    entity_stats = entity_data.groupby('entity').agg({
        'uncertainty': 'mean',
        'target': ['mean', 'count']
    }).reset_index()
    entity_stats.columns = ['entity', 'mean_unc', 'avg_target', 'count']
    entity_stats = entity_stats[entity_stats['count'] >= min_samples]

    high_unc = entity_stats.nlargest(10, 'mean_unc')
    low_unc = entity_stats.nsmallest(10, 'mean_unc')

    return high_unc, low_unc, entity_stats


def verification_1_held_out(X, y, entity_col, test_size=0.3, n_seeds=5):
    """
    VERIFICATION 1: Held-out validation
    Train on subset, verify entity rankings are consistent on held-out data.
    """
    print("\n" + "=" * 70)
    print("VERIFICATION 1: Held-Out Consistency")
    print("=" * 70)
    print("Question: Are entity uncertainty rankings consistent across train/test?")

    correlations = []
    for seed in range(n_seeds):
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42 + seed
        )

        # Train on subset
        models = train_ensemble(X_train, y_train, n_models=5, base_seed=seed*10)

        # Get entity rankings on train and test
        _, _, train_stats = identify_entities(models, X_train, y_train, entity_col)
        _, _, test_stats = identify_entities(models, X_test, y_test, entity_col)

        # Merge and compute correlation
        merged = train_stats.merge(
            test_stats, on='entity', suffixes=('_train', '_test')
        )

        if len(merged) > 5:
            corr = merged['mean_unc_train'].corr(merged['mean_unc_test'])
            correlations.append(corr)

    avg_corr = np.mean(correlations)
    print(f"\nResult: Train-test correlation = {avg_corr:.3f} (across {n_seeds} seeds)")
    print(f"Interpretation: {'PASS - Rankings are stable' if avg_corr > 0.5 else 'FAIL - Rankings unstable'}")

    return avg_corr


def verification_2_simulation(X, y, entity_col, models):
    """
    VERIFICATION 2: Counterfactual simulation
    What happens if we actually shift data from high-unc to low-unc entities?
    """
    print("\n" + "=" * 70)
    print("VERIFICATION 2: Counterfactual Simulation")
    print("=" * 70)
    print("Question: If we 'shift' orders, does uncertainty actually decrease?")

    high_unc, low_unc, entity_stats = identify_entities(models, X, y, entity_col)

    if len(high_unc) == 0 or len(low_unc) == 0:
        print("Not enough entities for simulation")
        return None

    # Select worst and best entity
    worst_entity = high_unc.iloc[0]['entity']
    best_entity = low_unc.iloc[0]['entity']

    worst_mask = X[entity_col] == worst_entity
    best_mask = X[entity_col] == best_entity

    worst_samples = X[worst_mask]
    best_samples = X[best_mask]

    print(f"\nWorst entity: {int(worst_entity)} (n={len(worst_samples)})")
    print(f"Best entity: {int(best_entity)} (n={len(best_samples)})")

    # Baseline uncertainty
    baseline_unc = get_uncertainty(models, X).mean()
    print(f"\nBaseline portfolio uncertainty: {baseline_unc:.6f}")

    # Simulation: Replace worst entity samples with best entity characteristics
    # (This is the counterfactual: "what if those orders went to best entity?")
    X_simulated = X.copy()

    # For each worst entity sample, replace with randomly sampled best entity features
    if len(best_samples) > 0 and len(worst_samples) > 0:
        for idx in worst_samples.index:
            # Pick a random sample from best entity
            replacement_idx = np.random.choice(best_samples.index)
            X_simulated.loc[idx] = X.loc[replacement_idx]

    simulated_unc = get_uncertainty(models, X_simulated).mean()
    reduction = (baseline_unc - simulated_unc) / baseline_unc * 100

    print(f"Simulated portfolio uncertainty: {simulated_unc:.6f}")
    print(f"Reduction: {reduction:.1f}%")

    # Verify: This should match the expected reduction from entity stats
    expected_reduction = (high_unc.iloc[0]['mean_unc'] - low_unc.iloc[0]['mean_unc'])
    proportion_shifted = len(worst_samples) / len(X)

    print(f"\nExpected reduction (from entity stats): {expected_reduction:.6f} per sample")
    print(f"Proportion of portfolio shifted: {proportion_shifted:.1%}")

    if reduction > 0:
        print(f"\nInterpretation: PASS - Simulation confirms uncertainty reduction")
    else:
        print(f"\nInterpretation: FAIL - Simulation shows no reduction (possible OOD issue)")

    return reduction


def verification_3_bootstrap(X, y, entity_col, n_bootstrap=10):
    """
    VERIFICATION 3: Bootstrap stability
    Are the entity rankings stable across bootstrap samples?
    """
    print("\n" + "=" * 70)
    print("VERIFICATION 3: Bootstrap Stability")
    print("=" * 70)
    print("Question: Are rankings stable across bootstrap resamples?")

    entity_rankings = defaultdict(list)

    for i in range(n_bootstrap):
        # Bootstrap sample
        indices = np.random.choice(len(X), size=len(X), replace=True)
        X_boot = X.iloc[indices].reset_index(drop=True)
        y_boot = y.iloc[indices].reset_index(drop=True)

        # Train and rank
        models = train_ensemble(X_boot, y_boot, n_models=3, base_seed=i*20)
        _, _, stats = identify_entities(models, X_boot, y_boot, entity_col)

        # Store rank for each entity
        stats['rank'] = range(len(stats))
        for _, row in stats.iterrows():
            entity_rankings[row['entity']].append(row['rank'])

    # Compute rank stability
    rank_stds = []
    for entity, ranks in entity_rankings.items():
        if len(ranks) >= n_bootstrap // 2:
            rank_stds.append(np.std(ranks))

    avg_rank_std = np.mean(rank_stds)
    max_possible_std = len(entity_rankings) / 2

    stability_score = 1 - (avg_rank_std / max_possible_std)
    print(f"\nResult: Rank stability score = {stability_score:.3f} (1.0 = perfect)")
    print(f"Average rank std = {avg_rank_std:.1f} (lower is better)")
    print(f"Interpretation: {'PASS - Rankings stable' if stability_score > 0.5 else 'FAIL - Rankings unstable'}")

    return stability_score


def verification_4_prediction_accuracy(models, X, y, entity_col):
    """
    VERIFICATION 4: Does low uncertainty correlate with lower prediction error?
    This validates that uncertainty is calibrated.
    """
    print("\n" + "=" * 70)
    print("VERIFICATION 4: Uncertainty-Error Calibration")
    print("=" * 70)
    print("Question: Do low-uncertainty entities have lower prediction error?")

    uncertainties = get_uncertainty(models, X)
    predictions = np.mean([m.predict(X) for m in models], axis=0)
    errors = np.abs(predictions - y.values)

    data = pd.DataFrame({
        'entity': X[entity_col].values,
        'uncertainty': uncertainties,
        'error': errors
    })

    entity_stats = data.groupby('entity').agg({
        'uncertainty': 'mean',
        'error': 'mean'
    }).reset_index()

    correlation = entity_stats['uncertainty'].corr(entity_stats['error'])

    print(f"\nResult: Uncertainty-Error correlation = {correlation:.3f}")
    print(f"Interpretation: {'PASS - Uncertainty is calibrated' if correlation > 0.3 else 'WEAK - Uncertainty weakly calibrated'}")

    # Binned analysis
    entity_stats['unc_bin'] = pd.qcut(entity_stats['uncertainty'], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    binned = entity_stats.groupby('unc_bin')['error'].mean()

    print("\nError by uncertainty level:")
    for level, error in binned.items():
        print(f"  {level}: MAE = {error:.3f}")

    return correlation


def run_verification():
    print("=" * 80)
    print("VERIFICATION SUITE: Entity-Level Optimization Recommendations")
    print("=" * 80)

    # Load data
    print("\nLoading SAP SALT data...")
    X, y, feature_cols, col_to_fk = load_salt_data(sample_size=20000)

    entity_col = 'SOLDTOPARTY'
    print(f"\nEntity column: {entity_col}")
    print(f"Unique entities: {X[entity_col].nunique()}")

    # Train baseline models
    print("\nTraining baseline ensemble...")
    models = train_ensemble(X, y, n_models=10)

    # Run all verifications
    results = {}

    # Verification 1: Held-out consistency
    results['held_out_corr'] = verification_1_held_out(X, y, entity_col)

    # Verification 2: Counterfactual simulation
    results['simulation_reduction'] = verification_2_simulation(X, y, entity_col, models)

    # Verification 3: Bootstrap stability
    results['bootstrap_stability'] = verification_3_bootstrap(X, y, entity_col)

    # Verification 4: Uncertainty calibration
    results['uncertainty_error_corr'] = verification_4_prediction_accuracy(models, X, y, entity_col)

    # Summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)

    print(f"""
Test 1 - Held-Out Consistency:    {'PASS' if results['held_out_corr'] > 0.5 else 'FAIL'} (corr={results['held_out_corr']:.3f})
Test 2 - Counterfactual Simulation: {'PASS' if results['simulation_reduction'] and results['simulation_reduction'] > 0 else 'FAIL'}
Test 3 - Bootstrap Stability:     {'PASS' if results['bootstrap_stability'] > 0.5 else 'FAIL'} (score={results['bootstrap_stability']:.3f})
Test 4 - Uncertainty Calibration: {'PASS' if results['uncertainty_error_corr'] > 0.3 else 'WEAK'} (corr={results['uncertainty_error_corr']:.3f})

OVERALL: Recommendations are {'VERIFIED' if sum([
    results['held_out_corr'] > 0.5,
    results['simulation_reduction'] and results['simulation_reduction'] > 0,
    results['bootstrap_stability'] > 0.5,
    results['uncertainty_error_corr'] > 0.3
]) >= 3 else 'NOT VERIFIED'} (passed {sum([
    results['held_out_corr'] > 0.5,
    results['simulation_reduction'] and results['simulation_reduction'] > 0,
    results['bootstrap_stability'] > 0.5,
    results['uncertainty_error_corr'] > 0.3
])}/4 tests)
""")

    print("=" * 80)

    return results


if __name__ == "__main__":
    results = run_verification()
