"""
Data Loader for rel-f1 (Formula 1 Racing)
==========================================

Domain: Formula 1 Racing
Task: driver-position (predict driver finishing position)
Type: Regression

FK Structure:
  Results -> Driver (driverId)
  Results -> Constructor (constructorId)
  Results -> Race (raceId)
  Race -> Circuit (circuitId)

Expected: Error Propagation domain (hierarchical structure)

Author: ChorokLeeDev
Created: 2025-12-08
"""

import numpy as np
import pandas as pd
import pickle
import os
from pathlib import Path

from relbench.datasets import get_dataset
from relbench.tasks import get_task

CACHE_DIR = Path('/Users/i767700/Github/ai-in-finance/chorok/v3_fk_risk_attribution/cache')
CACHE_DIR.mkdir(exist_ok=True)


def load_f1_data(task_name='driver-position', sample_size=3000, use_cache=True):
    """
    Load F1 data for FK attribution experiments.

    Returns:
        X: Feature DataFrame
        y: Target Series
        feature_cols: List of feature column names
        col_to_fk: Dict mapping column -> FK group
    """
    cache_file = CACHE_DIR / f'data_f1_{task_name}_{sample_size}_v3.pkl'

    if use_cache and cache_file.exists():
        print(f"[CACHE] Loading: {cache_file.name}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    print(f"Loading rel-f1 dataset (task: {task_name})...")

    # Load dataset and task
    dataset = get_dataset('rel-f1', download=True)
    task = get_task('rel-f1', task_name, download=True)
    db = dataset.get_db()

    # Get tables
    results_df = db.table_dict['results'].df.copy()
    drivers_df = db.table_dict['drivers'].df.copy()
    constructors_df = db.table_dict['constructors'].df.copy()
    races_df = db.table_dict['races'].df.copy()
    circuits_df = db.table_dict['circuits'].df.copy()

    # Get training data
    # train_df has: date, driverId, position (target)
    train_table = task.get_table('train')
    train_df = train_table.df.copy()
    target_col = task.target_col

    # Get recent race results for each driver (aggregate features)
    # Merge results with races to get date info
    results_with_date = results_df.merge(
        races_df[['raceId', 'date', 'circuitId']].rename(columns={'date': 'race_date'}),
        on='raceId', how='left'
    )

    # Efficient approach: Use cumulative stats per driver
    print("  Computing historical driver stats...")

    # Sort by date for cumulative calculations
    results_with_date = results_with_date.sort_values('race_date')

    # Compute cumulative stats per driver (up to but not including each race)
    results_with_date['cum_grid'] = results_with_date.groupby('driverId')['grid'].expanding().mean().reset_index(level=0, drop=True)
    results_with_date['cum_position'] = results_with_date.groupby('driverId')['positionOrder'].expanding().mean().reset_index(level=0, drop=True)
    results_with_date['cum_points'] = results_with_date.groupby('driverId')['points'].expanding().mean().reset_index(level=0, drop=True)
    results_with_date['cum_laps'] = results_with_date.groupby('driverId')['laps'].expanding().sum().reset_index(level=0, drop=True)
    results_with_date['cum_races'] = results_with_date.groupby('driverId').cumcount() + 1
    results_with_date['cum_best'] = results_with_date.groupby('driverId')['positionOrder'].expanding().min().reset_index(level=0, drop=True)
    results_with_date['cum_worst'] = results_with_date.groupby('driverId')['positionOrder'].expanding().max().reset_index(level=0, drop=True)

    # Get the latest stats per driver per date
    # For each train date, find the most recent race for each driver
    driver_latest = results_with_date.groupby('driverId').agg({
        'cum_grid': 'last',
        'cum_position': 'last',
        'cum_points': 'last',
        'cum_laps': 'last',
        'cum_races': 'last',
        'cum_best': 'last',
        'cum_worst': 'last',
        'constructorId': 'last',
        'circuitId': 'last'
    }).reset_index()

    driver_latest.columns = ['driverId', 'avg_grid', 'avg_position', 'avg_points',
                              'total_laps', 'n_races', 'best_finish', 'worst_finish',
                              'last_constructorId', 'last_circuitId']

    # Merge with train_df
    merged = train_df.merge(driver_latest, on='driverId', how='left')

    # Fill missing values
    for col in ['avg_grid', 'avg_position', 'avg_points', 'total_laps',
                'n_races', 'best_finish', 'worst_finish']:
        if col in merged.columns:
            merged[col] = merged[col].fillna(-999)

    # Merge with drivers
    drivers_df_renamed = drivers_df.add_prefix('DRV_')
    drivers_df_renamed = drivers_df_renamed.rename(columns={'DRV_driverId': 'driverId'})
    merged = merged.merge(drivers_df_renamed, on='driverId', how='left')

    # Merge with constructors (using last_constructorId)
    if 'last_constructorId' in merged.columns:
        constructors_df_renamed = constructors_df.add_prefix('CON_')
        constructors_df_renamed = constructors_df_renamed.rename(columns={'CON_constructorId': 'last_constructorId'})
        merged = merged.merge(constructors_df_renamed, on='last_constructorId', how='left')

    # Merge with circuits (using last_circuitId)
    if 'last_circuitId' in merged.columns:
        circuits_df_renamed = circuits_df.add_prefix('CIR_')
        circuits_df_renamed = circuits_df_renamed.rename(columns={'CIR_circuitId': 'last_circuitId'})
        merged = merged.merge(circuits_df_renamed, on='last_circuitId', how='left')

    print(f"  Merged shape: {merged.shape}")

    # Sample
    if len(merged) > sample_size:
        merged = merged.sample(sample_size, random_state=42)

    # Define FK groups based on feature sources
    col_to_fk = {}

    # Historical results features (aggregated from results table)
    results_cols = ['avg_grid', 'avg_position', 'avg_points', 'total_laps', 'n_races',
                    'best_finish', 'worst_finish']
    for col in results_cols:
        if col in merged.columns:
            col_to_fk[col] = 'HISTORY'

    # Driver features
    for col in merged.columns:
        if col.startswith('DRV_'):
            col_to_fk[col] = 'DRIVER'

    # Constructor features
    for col in merged.columns:
        if col.startswith('CON_'):
            col_to_fk[col] = 'CONSTRUCTOR'

    # Circuit features
    for col in merged.columns:
        if col.startswith('CIR_'):
            col_to_fk[col] = 'CIRCUIT'

    # Exclude non-feature columns
    exclude = {'driverId', 'date', target_col, 'last_constructorId', 'last_circuitId'}
    feature_cols = [c for c in col_to_fk.keys() if c not in exclude and c in merged.columns]

    # Prepare features
    X = merged[feature_cols].copy()

    # Handle categorical/object columns
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype.name == 'category':
            X[col] = X[col].astype('category').cat.codes
        # Handle timedelta
        if pd.api.types.is_timedelta64_dtype(X[col]):
            X[col] = X[col].dt.total_seconds()
        X[col] = pd.to_numeric(X[col], errors='coerce')
        X[col] = X[col].fillna(-999)

    # Prepare target
    y = merged[target_col].copy()
    y = pd.to_numeric(y, errors='coerce')
    y = y.fillna(y.median())

    # Filter col_to_fk to only include valid feature columns
    col_to_fk = {c: fk for c, fk in col_to_fk.items() if c in feature_cols}

    # Check we have enough FK groups
    fk_groups = set(col_to_fk.values())
    print(f"  Samples: {len(X)}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  FK groups: {len(fk_groups)} - {fk_groups}")
    print(f"  Target '{target_col}': min={y.min():.1f}, max={y.max():.1f}, mean={y.mean():.2f}")

    # Cache
    result = (X, y, feature_cols, col_to_fk)
    with open(cache_file, 'wb') as f:
        pickle.dump(result, f)
    print(f"[CACHE] Saved: {cache_file.name}")

    return result


if __name__ == "__main__":
    X, y, feature_cols, col_to_fk = load_f1_data(sample_size=3000)
    print(f"\nFeature columns: {feature_cols[:10]}...")
    print(f"\nFK groups:")
    for fk in set(col_to_fk.values()):
        cols = [c for c, f in col_to_fk.items() if f == fk]
        print(f"  {fk}: {len(cols)} columns")
