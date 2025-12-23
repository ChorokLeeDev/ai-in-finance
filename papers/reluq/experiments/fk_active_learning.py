"""
REAL FK-Guided Active Learning Implementation
==============================================

This is the REAL version (not simulated like test_2_active_learning.py).

Key differences from test_2:
- compute_fk_uncertainty() uses permutation-based method (REAL)
- fk_guided_acquisition() targets highest-uncertainty FK (REAL)
- Measures actual improvement (not random uniform)

Usage:
    python fk_active_learning.py --dataset rel-f1 --task driver-position
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


def extract_features_with_fk(dataset, task, sample_size=3000):
    """
    Extract features and track FK groups AND FK values.

    NEW (Day 2): Now tracks which FK VALUES each sample belongs to
    (e.g., raceId=123, constructorId=456), not just FK column groups.
    """
    db = dataset.get_db()
    train_table = task.get_table("train")

    entity_table_name = task.entity_table
    entity_table = db.table_dict[entity_table_name]
    entity_df = entity_table.df.copy()
    train_df = train_table.df.copy()

    if len(train_df) > sample_size:
        train_df = train_df.sample(n=sample_size, random_state=42)

    fk_to_entity = list(train_table.fkey_col_to_pkey_table.keys())[0]
    entity_pkey = entity_table.pkey_col

    merged_df = train_df.merge(entity_df, how='left', left_on=fk_to_entity,
                                right_on=entity_pkey, suffixes=('', '_entity'))

    target_col = task.target_col
    y = merged_df[target_col].values

    col_to_fk = {}
    feature_cols = []
    fk_value_cols = {}  # NEW: Track FK value columns (the ID columns we skipped before)

    # Get TRAIN table features
    for col in train_df.columns:
        if col == target_col:
            continue
        # NEW: Store FK ID columns separately
        if col.endswith('Id') or col.endswith('_id'):
            if col in merged_df.columns:
                fk_value_cols[col] = merged_df[col].values
            continue
        if train_df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            if col in merged_df.columns:
                feature_cols.append(col)
                col_to_fk[col] = 'TRAIN'

    # Get ENTITY table features
    for col in entity_df.columns:
        if col == entity_pkey:
            continue
        if col.endswith('Id') or col.endswith('_id'):
            col_name = col if col in merged_df.columns else f"{col}_entity"
            if col_name in merged_df.columns:
                fk_value_cols[col_name] = merged_df[col_name].values
            continue
        if entity_df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            col_name = col if col in merged_df.columns else f"{col}_entity"
            if col_name in merged_df.columns and col_name not in feature_cols:
                feature_cols.append(col_name)
                col_to_fk[col_name] = entity_table_name.upper()

    # Get FK table features (aggregated)
    fk_table_names = {}  # NEW: Map FK columns to their table names
    for table_name, table in db.table_dict.items():
        if table_name == entity_table_name:
            continue
        if hasattr(table, 'fkey_col_to_pkey_table'):
            for fk_col, ref_table in table.fkey_col_to_pkey_table.items():
                if ref_table == entity_table_name:
                    table_df = table.df
                    numeric_cols = [c for c in table_df.select_dtypes(include=[np.number]).columns
                                   if not c.endswith('Id') and c != fk_col]

                    if numeric_cols:
                        agg_df = table_df.groupby(fk_col)[numeric_cols].mean().reset_index()
                        agg_df.columns = [fk_col] + [f'{table_name}_{c}_mean' for c in numeric_cols]

                        merged_df = merged_df.merge(agg_df, how='left', left_on=fk_to_entity,
                                                    right_on=fk_col, suffixes=('', f'_{table_name}'))

                        for col in agg_df.columns[1:]:
                            if col in merged_df.columns and col not in feature_cols:
                                feature_cols.append(col)
                                col_to_fk[col] = table_name.upper()
                                fk_table_names[table_name.upper()] = fk_col

    X = merged_df[feature_cols].fillna(0).values

    # Create FK column index mapping
    fk_to_cols = defaultdict(list)
    for i, col in enumerate(feature_cols):
        fk_name = col_to_fk[col]
        fk_to_cols[fk_name].append(i)

    return X, y, col_to_fk, feature_cols, fk_to_cols, fk_value_cols, fk_table_names


def train_ensemble(X, y, n_models=5, seed=42):
    """Train simple LightGBM ensemble."""
    models = []
    for i in range(n_models):
        # Bootstrap sample
        idx = np.random.RandomState(seed+i).choice(len(X), int(0.8 * len(X)), replace=True)

        model = lgb.LGBMRegressor(
            n_estimators=50,
            max_depth=6,
            learning_rate=0.1,
            random_state=seed+i,
            verbose=-1
        )
        model.fit(X[idx], y[idx])
        models.append(model)

    return models


def ensemble_variance(models, X):
    """Compute ensemble variance (epistemic uncertainty)."""
    preds = np.array([m.predict(X) for m in models])
    return preds.var(axis=0)


def compute_fk_uncertainty(models, X, fk_to_cols, n_permutations=5):
    """
    Compute REAL FK-level uncertainty using permutation importance.

    This is the REAL version (not simulated!).

    Method:
    1. Compute base uncertainty (ensemble variance)
    2. For each FK, permute its columns
    3. Measure uncertainty increase
    4. FK uncertainty = average increase over permutations

    Returns:
        dict: {fk_name: uncertainty_contribution}
    """
    print("\n" + "="*60)
    print("Computing FK-level uncertainties (REAL - permutation-based)")
    print("="*60)

    # Base uncertainty
    base_unc = ensemble_variance(models, X)
    base_unc_mean = base_unc.mean()

    print(f"\nBase uncertainty (mean): {base_unc_mean:.6f}")

    fk_uncertainties = {}

    for fk_name, col_indices in fk_to_cols.items():
        print(f"\n  Processing FK: {fk_name} ({len(col_indices)} columns)")

        # Run multiple permutations and average
        perm_increases = []

        for perm_i in range(n_permutations):
            # Permute this FK's columns
            X_perm = X.copy()
            for col_idx in col_indices:
                X_perm[:, col_idx] = np.random.permutation(X_perm[:, col_idx])

            # Measure uncertainty increase
            perm_unc = ensemble_variance(models, X_perm)
            increase = (perm_unc - base_unc).mean()
            perm_increases.append(increase)

        # Average across permutations
        avg_increase = np.mean(perm_increases)
        fk_uncertainties[fk_name] = avg_increase

        print(f"    Uncertainty increase: {avg_increase:.6f}")

    # Normalize to sum to 1
    total = sum(fk_uncertainties.values())
    if total > 0:
        fk_uncertainties = {k: v/total for k, v in fk_uncertainties.items()}

    print("\n" + "="*60)
    print("FK Uncertainty Contributions (normalized):")
    print("="*60)
    for fk_name, unc in sorted(fk_uncertainties.items(), key=lambda x: -x[1]):
        print(f"  {fk_name}: {unc*100:.2f}%")

    return fk_uncertainties


def compute_fk_value_uncertainty(models, X_pool, fk_value_cols_pool, top_fk_name, fk_table_names):
    """
    NEW (Day 2): Compute uncertainty at FK VALUE level.

    Instead of "RESULTS FK has high uncertainty",
    compute "raceId=123 has high uncertainty, raceId=456 has low uncertainty".

    Returns:
        dict: {fk_value: mean_uncertainty}
    """
    # Get the FK ID column for this FK group
    # For RESULTS FK, this is 'raceId'; for CONSTRUCTORS FK, this is 'constructorId', etc.
    fk_id_col = None
    for col_name in fk_value_cols_pool.keys():
        # Match FK table name to column
        if top_fk_name.lower() in col_name.lower():
            fk_id_col = col_name
            break

    if fk_id_col is None:
        # Fallback: can't find FK ID column
        return {}

    # Compute per-sample uncertainty
    sample_unc = ensemble_variance(models, X_pool)

    # Group by FK value
    fk_values = fk_value_cols_pool[fk_id_col]
    fk_value_to_unc = defaultdict(list)

    for i, fk_val in enumerate(fk_values):
        if not np.isnan(fk_val):
            fk_value_to_unc[fk_val].append(sample_unc[i])

    # Average uncertainty per FK value
    fk_value_unc = {fk_val: np.mean(uncs) for fk_val, uncs in fk_value_to_unc.items()}

    return fk_value_unc


def fk_guided_acquisition(X_pool, y_pool, models, fk_to_cols, fk_uncertainties,
                          fk_value_cols_pool, fk_table_names, budget=200, strategy='v2'):
    """
    Select samples from highest-uncertainty FK.

    VERSION 2 (Day 2): Target specific FK VALUES within high-uncertainty FK.

    Strategy:
    1. Identify FK with highest uncertainty (e.g., RESULTS)
    2. Compute uncertainty for each FK VALUE (e.g., each race)
    3. Select samples from highest-uncertainty FK values
    4. Within those, pick highest-uncertainty samples

    Args:
        strategy: 'v1' = old (just uncertainty sampling)
                  'v2' = new (FK-value targeted)

    Returns:
        np.ndarray: Indices of samples to acquire
    """
    # Get highest-uncertainty FK
    top_fk = max(fk_uncertainties.items(), key=lambda x: x[1])[0]

    print(f"\n  Target FK: {top_fk} (highest uncertainty)")

    if strategy == 'v1':
        # OLD: Just do uncertainty sampling
        print(f"  Strategy: v1 (uncertainty sampling)")
        sample_unc = ensemble_variance(models, X_pool)
        return np.argsort(-sample_unc)[:budget]

    elif strategy == 'v2':
        # NEW: Target specific FK values
        print(f"  Strategy: v2 (FK-value targeted)")

        # Compute FK-value-level uncertainty
        fk_value_unc = compute_fk_value_uncertainty(
            models, X_pool, fk_value_cols_pool, top_fk, fk_table_names
        )

        if not fk_value_unc:
            # Fallback to v1
            print(f"  WARNING: Could not compute FK-value uncertainty, falling back to v1")
            sample_unc = ensemble_variance(models, X_pool)
            return np.argsort(-sample_unc)[:budget]

        # Find FK ID column
        fk_id_col = None
        for col_name in fk_value_cols_pool.keys():
            if top_fk.lower() in col_name.lower():
                fk_id_col = col_name
                break

        # Get top 3 highest-uncertainty FK values
        top_fk_values = sorted(fk_value_unc.items(), key=lambda x: -x[1])[:3]

        print(f"  Top uncertain FK values:")
        for fk_val, unc in top_fk_values:
            print(f"    {fk_id_col}={fk_val}: unc={unc:.4f}")

        # Select samples from these FK values
        fk_values = fk_value_cols_pool[fk_id_col]
        target_fk_vals = {fk_val for fk_val, _ in top_fk_values}

        candidate_indices = [i for i, val in enumerate(fk_values) if val in target_fk_vals]

        if len(candidate_indices) == 0:
            # Fallback
            sample_unc = ensemble_variance(models, X_pool)
            return np.argsort(-sample_unc)[:budget]

        # Within candidates, pick highest uncertainty
        sample_unc = ensemble_variance(models, X_pool)
        candidate_uncs = [(i, sample_unc[i]) for i in candidate_indices]
        candidate_uncs.sort(key=lambda x: -x[1])

        selected_indices = [i for i, _ in candidate_uncs[:budget]]

        print(f"  Selected {len(selected_indices)} samples from {len(candidate_indices)} candidates")

        return np.array(selected_indices)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def random_acquisition(X_pool, y_pool, budget=200):
    """Random baseline acquisition."""
    return np.random.choice(len(X_pool), min(budget, len(X_pool)), replace=False)


def uncertainty_acquisition(X_pool, y_pool, models, budget=200):
    """Standard uncertainty sampling (no FK information)."""
    sample_unc = ensemble_variance(models, X_pool)
    return np.argsort(-sample_unc)[:budget]


def evaluate_mae(models, X, y):
    """Compute ensemble MAE."""
    preds = np.array([m.predict(X) for m in models])
    mean_pred = preds.mean(axis=0)
    return np.abs(mean_pred - y).mean()


def run_active_learning_experiment(X, y, fk_to_cols, fk_value_cols, fk_table_names,
                                   n_iterations=5, n_start_pct=0.2, n_acquire_pct=0.1):
    """
    Run full active learning experiment with 4 strategies.

    Strategies:
    1. Random: baseline
    2. Uncertainty: standard active learning
    3. FK-guided-v1: our method v1 (targets high-uncertainty FK, but just uncertainty sampling)
    4. FK-guided-v2: our method v2 (targets high-uncertainty FK VALUES)

    Returns:
        dict: Results for each strategy
    """
    print("\n" + "="*60)
    print("Running Active Learning Experiment (REAL - with v1 vs v2)")
    print("="*60)

    n_total = len(X)
    n_start = int(n_start_pct * n_total)
    n_acquire = int(n_acquire_pct * n_total)

    print(f"\nTotal samples: {n_total}")
    print(f"Initial training: {n_start}")
    print(f"Acquire per iteration: {n_acquire}")
    print(f"Iterations: {n_iterations}")

    # Create initial split
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
        },
        'fk_guided_v2': {
            'train_idx': indices[:n_start].copy(),
            'pool_idx': indices[n_start:].copy(),
            'maes': [],
            'samples': []
        }
    }

    for iteration in range(n_iterations):
        print(f"\n{'='*60}")
        print(f"Iteration {iteration+1}/{n_iterations}")
        print(f"{'='*60}")

        for strategy_name, strategy_data in strategies.items():
            print(f"\n  Strategy: {strategy_name}")

            train_idx = strategy_data['train_idx']
            pool_idx = strategy_data['pool_idx']

            if len(pool_idx) == 0:
                print(f"    Pool exhausted!")
                break

            print(f"    Training samples: {len(train_idx)}")
            print(f"    Pool samples: {len(pool_idx)}")

            # Train ensemble
            X_train = X[train_idx]
            y_train = y[train_idx]
            models = train_ensemble(X_train, y_train, n_models=5, seed=42+iteration)

            # Evaluate on pool
            X_pool = X[pool_idx]
            y_pool = y[pool_idx]
            mae = evaluate_mae(models, X_pool, y_pool)

            strategy_data['maes'].append(mae)
            strategy_data['samples'].append(len(train_idx))

            print(f"    MAE on pool: {mae:.4f}")

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
                # V1: FK-guided but just uncertainty sampling
                fk_unc = compute_fk_uncertainty(models, X_pool, fk_to_cols, n_permutations=3)
                acquire_indices = fk_guided_acquisition(
                    X_pool, y_pool, models, fk_to_cols, fk_unc,
                    fk_value_cols_pool, fk_table_names,
                    budget=n_acquire, strategy='v1'
                )
            elif strategy_name == 'fk_guided_v2':
                # V2: FK-guided with FK-value targeting (NEW!)
                fk_unc = compute_fk_uncertainty(models, X_pool, fk_to_cols, n_permutations=3)
                acquire_indices = fk_guided_acquisition(
                    X_pool, y_pool, models, fk_to_cols, fk_unc,
                    fk_value_cols_pool, fk_table_names,
                    budget=n_acquire, strategy='v2'
                )

            # Update train/pool
            new_samples = pool_idx[acquire_indices]
            strategy_data['train_idx'] = np.concatenate([train_idx, new_samples])
            strategy_data['pool_idx'] = np.delete(pool_idx, acquire_indices)

    return strategies


def compute_efficiency_gain(strategies):
    """Compute sample efficiency gain over random baseline."""
    random_maes = strategies['random']['maes']

    if len(random_maes) == 0:
        return {}

    initial_mae = random_maes[0]
    final_mae = random_maes[-1]
    target_mae = final_mae + 0.1 * (initial_mae - final_mae)

    results = {}

    for strategy_name in ['uncertainty', 'fk_guided_v1', 'fk_guided_v2']:
        if strategy_name not in strategies:
            continue

        strategy_maes = strategies[strategy_name]['maes']

        # Find iterations to reach target
        random_iter = next((i for i, mae in enumerate(random_maes) if mae <= target_mae), len(random_maes)-1)
        strategy_iter = next((i for i, mae in enumerate(strategy_maes) if mae <= target_mae), len(strategy_maes)-1)

        if random_iter == 0:
            improvement = 0.0
        else:
            improvement = (random_iter - strategy_iter) / random_iter * 100

        results[strategy_name] = improvement

    return results


def main():
    parser = argparse.ArgumentParser(description="Real FK-guided active learning")
    parser.add_argument("--dataset", type=str, default="rel-f1")
    parser.add_argument("--task", type=str, default="driver-position")
    parser.add_argument("--sample_size", type=int, default=1500)
    parser.add_argument("--n_iterations", type=int, default=5)
    args = parser.parse_args()

    print("\n" + "="*60)
    print("FK-Guided Active Learning - REAL IMPLEMENTATION")
    print("="*60)
    print(f"\nDataset: {args.dataset}")
    print(f"Task: {args.task}")
    print(f"Sample size: {args.sample_size}")

    # Load data
    print("\nLoading data...")
    dataset = get_dataset(args.dataset, download=True)
    task = get_task(args.dataset, args.task, download=True)

    # Extract features
    print("Extracting features...")
    X, y, col_to_fk, feature_names, fk_to_cols, fk_value_cols, fk_table_names = extract_features_with_fk(
        dataset, task, sample_size=args.sample_size
    )

    print(f"\nFeatures extracted:")
    print(f"  Total features: {len(feature_names)}")
    print(f"  FK groups: {len(fk_to_cols)}")
    for fk_name, cols in fk_to_cols.items():
        print(f"    {fk_name}: {len(cols)} features")
    print(f"  FK ID columns tracked: {list(fk_value_cols.keys())}")

    # Run experiment
    strategies = run_active_learning_experiment(
        X, y, fk_to_cols, fk_value_cols, fk_table_names,
        n_iterations=args.n_iterations,
        n_start_pct=0.2,
        n_acquire_pct=0.1
    )

    # Compute improvements
    improvements = compute_efficiency_gain(strategies)

    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)

    print(f"\nFinal MAEs:")
    for strategy_name in ['random', 'uncertainty', 'fk_guided_v1', 'fk_guided_v2']:
        final_mae = strategies[strategy_name]['maes'][-1]
        print(f"  {strategy_name}: {final_mae:.4f}")

    print(f"\nSample efficiency gains over random:")
    for strategy_name in ['uncertainty', 'fk_guided_v1', 'fk_guided_v2']:
        improvement = improvements.get(strategy_name, 0)
        print(f"  {strategy_name}: {improvement:+.1f}%")

    # Verdict
    print("\n" + "="*60)
    print("VERDICT")
    print("="*60)

    v1_improvement = improvements.get('fk_guided_v1', 0)
    v2_improvement = improvements.get('fk_guided_v2', 0)

    print(f"\nFK-guided v1 (uncertainty sampling): {v1_improvement:+.1f}%")
    print(f"FK-guided v2 (FK-value targeting): {v2_improvement:+.1f}%")
    print(f"Improvement from v1→v2: {(v2_improvement - v1_improvement):+.1f}%")

    if v2_improvement > 20:
        verdict = "✅ PASS"
        recommendation = "FK-guided v2 shows strong gains (>20%) - include as main contribution"
    elif v2_improvement > 10:
        verdict = "⚠️  MARGINAL"
        recommendation = "FK-guided v2 shows moderate gains (10-20%) - include as application"
    else:
        verdict = "❌ WEAK"
        recommendation = "FK-guided v2 shows weak gains (<10%) - needs more work or drop"

    print(f"\nResult: {verdict}")
    print(f"\nRecommendation: {recommendation}")

    # Save results
    output_dir = Path(__file__).parent / 'test_results'
    output_dir.mkdir(exist_ok=True)

    output = {
        'test': 'fk_active_learning_real_v2',
        'dataset': args.dataset,
        'task': args.task,
        'verdict': verdict,
        'recommendation': recommendation,
        'results': {
            'fk_v1_improvement_pct': float(v1_improvement),
            'fk_v2_improvement_pct': float(v2_improvement),
            'v1_to_v2_gain_pct': float(v2_improvement - v1_improvement),
            'uncertainty_improvement_pct': float(improvements.get('uncertainty', 0)),
            'n_iterations': args.n_iterations,
            'final_maes': {k: float(v['maes'][-1]) for k, v in strategies.items()},
        }
    }

    output_file = output_dir / 'fk_active_learning_real.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return output


if __name__ == '__main__':
    result = main()
