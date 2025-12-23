"""
Multi-Seed Validation for FK-Guided Active Learning
====================================================

CRITICAL TEST: Determine if +50% efficiency is real or random variation.

Day 2 showed +50% on a single seed - this is suspicious (Day 1 showed 0%).
Need to run multiple seeds to determine true improvement.

Expected outcome: ~15-25% (typical for active learning)

Usage:
    python multi_seed_validation.py --n_seeds 5 --dataset rel-f1 --task driver-position
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import lightgbm as lgb
from relbench.datasets import get_dataset
from relbench.tasks import get_task

# Import functions from fk_active_learning.py
from fk_active_learning import (
    extract_features_with_fk,
    train_ensemble,
    evaluate_mae,
    compute_fk_uncertainty,
    fk_guided_acquisition,
    uncertainty_acquisition,
    random_acquisition,
    ensemble_variance
)


def run_single_seed_experiment(X, y, fk_to_cols, fk_value_cols, fk_table_names,
                                seed, n_iterations=3, n_start_pct=0.2, n_acquire_pct=0.1):
    """
    Run active learning experiment with a specific random seed.

    Returns:
        dict: Results for each strategy (MAEs over iterations)
    """
    print(f"\n{'='*60}")
    print(f"SEED {seed}")
    print(f"{'='*60}")

    # Set seed
    np.random.seed(seed)

    n_total = len(X)
    n_start = int(n_start_pct * n_total)
    n_acquire = int(n_acquire_pct * n_total)

    # Create initial split with this seed
    indices = np.arange(n_total)
    np.random.shuffle(indices)

    strategies = {
        'random': {
            'train_idx': indices[:n_start].copy(),
            'pool_idx': indices[n_start:].copy(),
            'maes': [],
            'samples': []
        },
        'uncertainty': {
            'train_idx': indices[:n_start].copy(),
            'pool_idx': indices[n_start:].copy(),
            'maes': [],
            'samples': []
        },
        'fk_guided_v1': {
            'train_idx': indices[:n_start].copy(),
            'pool_idx': indices[n_start:].copy(),
            'maes': [],
            'samples': []
        }
    }

    for iteration in range(n_iterations):
        print(f"\nIteration {iteration+1}/{n_iterations}")

        for strategy_name, strategy_data in strategies.items():
            train_idx = strategy_data['train_idx']
            pool_idx = strategy_data['pool_idx']

            if len(pool_idx) == 0:
                break

            # Train ensemble
            X_train = X[train_idx]
            y_train = y[train_idx]
            models = train_ensemble(X_train, y_train, n_models=5, seed=seed+iteration)

            # Evaluate on pool
            X_pool = X[pool_idx]
            y_pool = y[pool_idx]
            mae = evaluate_mae(models, X_pool, y_pool)

            strategy_data['maes'].append(mae)
            strategy_data['samples'].append(len(train_idx))

            # Get FK values for pool
            fk_value_cols_pool = {k: v[pool_idx] for k, v in fk_value_cols.items()}

            # Select samples to acquire
            if len(pool_idx) < n_acquire:
                acquire_indices = np.arange(len(pool_idx))
            elif strategy_name == 'random':
                acquire_indices = random_acquisition(X_pool, y_pool, budget=n_acquire)
            elif strategy_name == 'uncertainty':
                acquire_indices = uncertainty_acquisition(X_pool, y_pool, models, budget=n_acquire)
            elif strategy_name == 'fk_guided_v1':
                fk_unc = compute_fk_uncertainty(models, X_pool, fk_to_cols, n_permutations=3)
                acquire_indices = fk_guided_acquisition(
                    X_pool, y_pool, models, fk_to_cols, fk_unc,
                    fk_value_cols_pool, fk_table_names,
                    budget=n_acquire, strategy='v1'
                )

            # Update train/pool
            new_samples = pool_idx[acquire_indices]
            strategy_data['train_idx'] = np.concatenate([train_idx, new_samples])
            strategy_data['pool_idx'] = np.delete(pool_idx, acquire_indices)

    # Extract final results
    results = {
        'seed': seed,
        'strategies': {}
    }

    for strategy_name, strategy_data in strategies.items():
        results['strategies'][strategy_name] = {
            'maes': strategy_data['maes'],
            'final_mae': strategy_data['maes'][-1] if strategy_data['maes'] else None
        }

    return results


def compute_efficiency_gain_from_seed(seed_result):
    """Compute efficiency gain for a single seed."""
    random_maes = seed_result['strategies']['random']['maes']

    if len(random_maes) == 0:
        return {}

    initial_mae = random_maes[0]
    final_mae = random_maes[-1]
    target_mae = final_mae + 0.1 * (initial_mae - final_mae)

    gains = {}

    for strategy_name in ['uncertainty', 'fk_guided_v1']:
        strategy_maes = seed_result['strategies'][strategy_name]['maes']

        # Find iterations to reach target
        random_iter = next((i for i, mae in enumerate(random_maes) if mae <= target_mae), len(random_maes)-1)
        strategy_iter = next((i for i, mae in enumerate(strategy_maes) if mae <= target_mae), len(strategy_maes)-1)

        if random_iter == 0:
            improvement = 0.0
        else:
            improvement = (random_iter - strategy_iter) / random_iter * 100

        gains[strategy_name] = improvement

    return gains


def analyze_multi_seed_results(all_results):
    """Analyze results across multiple seeds."""
    print("\n" + "="*60)
    print("MULTI-SEED ANALYSIS")
    print("="*60)

    # Collect efficiency gains
    efficiency_gains = defaultdict(list)
    final_maes = defaultdict(list)

    for seed_result in all_results:
        gains = compute_efficiency_gain_from_seed(seed_result)

        for strategy_name, gain in gains.items():
            efficiency_gains[strategy_name].append(gain)

        for strategy_name, data in seed_result['strategies'].items():
            if data['final_mae'] is not None:
                final_maes[strategy_name].append(data['final_mae'])

    # Compute statistics
    print("\n" + "="*60)
    print("Final MAE (mean ± std)")
    print("="*60)

    for strategy_name in ['random', 'uncertainty', 'fk_guided_v1']:
        maes = final_maes[strategy_name]
        mean_mae = np.mean(maes)
        std_mae = np.std(maes)
        print(f"{strategy_name:20s}: {mean_mae:.4f} ± {std_mae:.4f}")

    print("\n" + "="*60)
    print("Sample Efficiency Gain over Random (mean ± std)")
    print("="*60)

    stats = {}

    for strategy_name in ['uncertainty', 'fk_guided_v1']:
        gains = efficiency_gains[strategy_name]
        mean_gain = np.mean(gains)
        std_gain = np.std(gains)

        print(f"{strategy_name:20s}: {mean_gain:+6.1f}% ± {std_gain:5.1f}%")
        print(f"  Individual seeds: {[f'{g:+.1f}%' for g in gains]}")

        stats[strategy_name] = {
            'mean_gain_pct': float(mean_gain),
            'std_gain_pct': float(std_gain),
            'all_gains_pct': [float(g) for g in gains],
            'min_gain_pct': float(min(gains)),
            'max_gain_pct': float(max(gains))
        }

    return stats


def main():
    parser = argparse.ArgumentParser(description="Multi-seed validation for active learning")
    parser.add_argument("--dataset", type=str, default="rel-f1")
    parser.add_argument("--task", type=str, default="driver-position")
    parser.add_argument("--sample_size", type=int, default=1500)
    parser.add_argument("--n_iterations", type=int, default=3)
    parser.add_argument("--n_seeds", type=int, default=5, help="Number of random seeds to test")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Specific seeds to use (overrides n_seeds)")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("MULTI-SEED VALIDATION - FK-Guided Active Learning")
    print("="*60)
    print(f"\nDataset: {args.dataset}")
    print(f"Task: {args.task}")
    print(f"Sample size: {args.sample_size}")
    print(f"Iterations: {args.n_iterations}")

    # Determine seeds
    if args.seeds is not None:
        seeds = args.seeds
    else:
        seeds = list(range(42, 42 + args.n_seeds))

    print(f"Seeds: {seeds}")

    # Load data (once)
    print("\nLoading data...")
    dataset = get_dataset(args.dataset, download=True)
    task = get_task(args.dataset, args.task, download=True)

    # Extract features (once, with fixed seed for consistency)
    print("Extracting features...")
    np.random.seed(42)  # Fixed seed for feature extraction
    X, y, col_to_fk, feature_names, fk_to_cols, fk_value_cols, fk_table_names = extract_features_with_fk(
        dataset, task, sample_size=args.sample_size
    )

    print(f"\nFeatures extracted:")
    print(f"  Total features: {len(feature_names)}")
    print(f"  FK groups: {len(fk_to_cols)}")

    # Run experiments for each seed
    all_results = []

    for seed in seeds:
        result = run_single_seed_experiment(
            X, y, fk_to_cols, fk_value_cols, fk_table_names,
            seed=seed,
            n_iterations=args.n_iterations,
            n_start_pct=0.2,
            n_acquire_pct=0.1
        )
        all_results.append(result)

    # Analyze results
    stats = analyze_multi_seed_results(all_results)

    # Verdict
    print("\n" + "="*60)
    print("VERDICT")
    print("="*60)

    mean_unc_gain = stats['uncertainty']['mean_gain_pct']
    std_unc_gain = stats['uncertainty']['std_gain_pct']
    mean_fk_gain = stats['fk_guided_v1']['mean_gain_pct']
    std_fk_gain = stats['fk_guided_v1']['std_gain_pct']

    print(f"\nUncertainty sampling: {mean_unc_gain:+.1f}% ± {std_unc_gain:.1f}%")
    print(f"FK-guided v1:         {mean_fk_gain:+.1f}% ± {std_fk_gain:.1f}%")
    print(f"FK vs Uncertainty:    {(mean_fk_gain - mean_unc_gain):+.1f}%")

    # Decision criteria
    print("\n" + "="*60)
    print("DECISION")
    print("="*60)

    if mean_unc_gain > 25:
        decision = "✅ STRONG - Active learning shows >25% efficiency"
        recommendation = "Include active learning as application direction"
    elif mean_unc_gain > 15:
        decision = "⚠️  MODERATE - Active learning shows 15-25% efficiency"
        recommendation = "Consider including, but not as main contribution"
    elif mean_unc_gain > 10:
        decision = "⚠️  WEAK - Active learning shows 10-15% efficiency"
        recommendation = "Mention in discussion, don't claim as contribution"
    else:
        decision = "❌ FAIL - Active learning shows <10% efficiency"
        recommendation = "Drop active learning direction"

    print(f"\nResult: {decision}")
    print(f"Recommendation: {recommendation}")

    # FK-guided vs uncertainty
    print("\n" + "="*60)
    print("FK-GUIDED vs UNCERTAINTY")
    print("="*60)

    fk_vs_unc = mean_fk_gain - mean_unc_gain

    if abs(fk_vs_unc) < 5:
        fk_verdict = "❌ NO BENEFIT - FK-guided ≈ uncertainty sampling (within 5%)"
        fk_recommendation = "FK information doesn't help acquisition - drop FK-guided claim"
    elif fk_vs_unc > 10:
        fk_verdict = "✅ STRONG BENEFIT - FK-guided beats uncertainty by >10%"
        fk_recommendation = "FK-guided is novel contribution - include in paper"
    elif fk_vs_unc > 5:
        fk_verdict = "⚠️  MARGINAL BENEFIT - FK-guided beats uncertainty by 5-10%"
        fk_recommendation = "FK-guided shows promise but modest gain"
    else:
        fk_verdict = "❌ NEGATIVE - FK-guided worse than uncertainty"
        fk_recommendation = "FK information hurts - use standard uncertainty sampling"

    print(f"\nResult: {fk_verdict}")
    print(f"Recommendation: {fk_recommendation}")

    # Save results
    output_dir = Path(__file__).parent / 'test_results'
    output_dir.mkdir(exist_ok=True)

    output = {
        'test': 'multi_seed_validation',
        'dataset': args.dataset,
        'task': args.task,
        'n_seeds': len(seeds),
        'seeds': seeds,
        'decision': decision,
        'fk_verdict': fk_verdict,
        'recommendation': recommendation,
        'fk_recommendation': fk_recommendation,
        'statistics': stats,
        'all_results': all_results
    }

    output_file = output_dir / 'multi_seed_validation.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    print(f"\nTested {len(seeds)} seeds: {seeds}")
    print(f"\nActive learning efficiency: {mean_unc_gain:+.1f}% ± {std_unc_gain:.1f}%")
    print(f"FK-guided vs uncertainty:   {fk_vs_unc:+.1f}%")

    if mean_unc_gain < 15:
        print(f"\n⚠️  Active learning shows <15% efficiency - likely not worth pursuing")
        print(f"Recommendation: Proceed with 2 directions (SHAP + Decomposition)")
        print(f"NeurIPS probability: 65-70%")
    elif abs(fk_vs_unc) < 5:
        print(f"\n⚠️  FK-guided = uncertainty sampling - no novel contribution")
        print(f"Can mention active learning works, but it's standard uncertainty sampling")
        print(f"NeurIPS probability: 65-70% (same as 2 directions)")
    else:
        print(f"\n✅ Active learning validated AND FK-guided adds value!")
        print(f"Recommendation: Proceed with 3 directions")
        print(f"NeurIPS probability: 75-80%")

    return output


if __name__ == '__main__':
    result = main()
