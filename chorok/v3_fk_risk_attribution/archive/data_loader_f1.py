"""
Data Loader for FK-Level Risk Attribution (rel-f1)
===================================================

Load rel-f1 data and define FK mappings for regression task.

Theoretical Justification:
- Regression tasks naturally have meaningful uncertainty via ensemble variance
- Reference: Lakshminarayanan et al. 2017 "Simple and Scalable Predictive
  Uncertainty Estimation using Deep Ensembles"
- Unlike classification where softmax can be overconfident (Guo et al. 2017),
  regression ensemble variance directly measures model disagreement
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path

from relbench.datasets import get_dataset
from relbench.tasks import get_task

from cache import cache_data, load_data


# =============================================================================
# FK MAPPING: Column → Source Table
# =============================================================================

FK_MAPPING = {
    # Driver attributes
    'driverRef': 'DRIVER',
    'code': 'DRIVER',
    'forename': 'DRIVER',
    'surname': 'DRIVER',
    'dob': 'DRIVER',
    'nationality': 'DRIVER',

    # Constructor attributes
    'constructorId': 'CONSTRUCTOR',
    'constructorRef': 'CONSTRUCTOR',
    'name': 'CONSTRUCTOR',
    'nationality_constr': 'CONSTRUCTOR',

    # Race attributes
    'raceId': 'RACE',
    'year': 'RACE',
    'round': 'RACE',
    'name_race': 'RACE',
    'date_race': 'RACE',
    'time': 'RACE',

    # Circuit attributes
    'circuitId': 'CIRCUIT',
    'circuitRef': 'CIRCUIT',
    'name_circuit': 'CIRCUIT',
    'location': 'CIRCUIT',
    'country': 'CIRCUIT',
    'lat': 'CIRCUIT',
    'lng': 'CIRCUIT',
    'alt': 'CIRCUIT',

    # Performance features (from results)
    'grid': 'PERFORMANCE',
    'points': 'PERFORMANCE',
    'laps': 'PERFORMANCE',
}


def get_fk_group(col: str) -> str:
    """Get FK group for a column. Returns 'OTHER' if not mapped."""
    return FK_MAPPING.get(col, 'OTHER')


def get_fk_groups() -> Dict[str, List[str]]:
    """Get all FK groups and their columns."""
    groups = {}
    for col, fk in FK_MAPPING.items():
        if fk not in groups:
            groups[fk] = []
        groups[fk].append(col)
    return groups


def load_f1_data(
    task_name: str = 'driver-position',
    sample_size: int = 3000,
    use_cache: bool = True
) -> Tuple[pd.DataFrame, pd.Series, List[str], Dict[str, str]]:
    """
    Load rel-f1 dataset with joined features.

    Args:
        task_name: Task name (e.g., 'driver-position')
        sample_size: Number of samples
        use_cache: Whether to use cache

    Returns:
        X: Feature DataFrame
        y: Target Series (position)
        feature_cols: List of feature column names
        col_to_fk: Column to FK mapping (for available columns)
    """
    cache_key = f"f1_{task_name}"

    # Try cache first
    if use_cache:
        cached = load_data(cache_key, sample_size)
        if cached is not None:
            return (
                pd.DataFrame(cached['X']),
                pd.Series(cached['y']),
                cached['feature_cols'],
                cached['col_to_fk']
            )

    # Load from relbench
    print(f"Loading rel-f1/{task_name}...")
    dataset = get_dataset("rel-f1", download=True)
    db = dataset.get_db()
    task = get_task("rel-f1", task_name, download=True)

    # Get tables
    drivers = db.table_dict['drivers'].df
    results = db.table_dict['results'].df
    races = db.table_dict['races'].df
    constructors = db.table_dict['constructors'].df
    circuits = db.table_dict['circuits'].df

    # Get train table
    train_df = task.get_table('train').df.copy()

    # Sample
    if len(train_df) > sample_size:
        train_df = train_df.sample(sample_size, random_state=42)

    # Join driver info
    df = train_df.merge(drivers, on='driverId', how='left')

    # Get most recent results for each driver (for performance features)
    recent_results = results.sort_values('date').groupby('driverId').last().reset_index()
    recent_results = recent_results[['driverId', 'constructorId', 'raceId', 'grid', 'points', 'laps']]
    df = df.merge(recent_results, on='driverId', how='left', suffixes=('', '_recent'))

    # Join constructor info
    df = df.merge(constructors, on='constructorId', how='left', suffixes=('', '_constr'))

    # Join race info
    df = df.merge(races, on='raceId', how='left', suffixes=('', '_race'))

    # Join circuit info
    df = df.merge(circuits, on='circuitId', how='left', suffixes=('', '_circuit'))

    # Extract target
    target_col = task.target_col
    y = df[target_col].copy()

    # Define feature columns (exclude IDs and target)
    exclude = {'date', target_col, 'driverId'}
    feature_cols = [c for c in df.columns if c not in exclude and not c.endswith('Id')]

    # Build X
    X = df[feature_cols].copy()

    # Encode categorical columns
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = X[col].astype('category').cat.codes
        elif X[col].dtype == 'datetime64[ns]':
            # Convert datetime to numeric (days since epoch)
            X[col] = (X[col] - pd.Timestamp('1970-01-01')).dt.days
        X[col] = X[col].fillna(-999)

    # Build column to FK mapping
    col_to_fk = {}
    for col in feature_cols:
        col_to_fk[col] = get_fk_group(col)

    # Cache
    if use_cache:
        cache_data(cache_key, sample_size, {
            'X': X.to_dict(),
            'y': y.tolist(),
            'feature_cols': feature_cols,
            'col_to_fk': col_to_fk
        })

    return X, y, feature_cols, col_to_fk


def print_fk_summary(col_to_fk: Dict[str, str]):
    """Print FK mapping summary."""
    fk_to_cols = {}
    for col, fk in col_to_fk.items():
        if fk not in fk_to_cols:
            fk_to_cols[fk] = []
        fk_to_cols[fk].append(col)

    print("\n" + "="*60)
    print("FK MAPPING SUMMARY (rel-f1)")
    print("="*60)

    for fk in sorted(fk_to_cols.keys()):
        cols = fk_to_cols[fk]
        print(f"\n{fk} ({len(cols)} columns):")
        for col in sorted(cols):
            print(f"  - {col}")

    print("\n" + "="*60)
    print(f"Total: {len(col_to_fk)} columns, {len(fk_to_cols)} FK groups")
    print("="*60)


if __name__ == "__main__":
    print("Testing rel-f1 data loader...")

    X, y, feature_cols, col_to_fk = load_f1_data(
        task_name='driver-position',
        sample_size=3000,
        use_cache=False  # Force reload for testing
    )

    print(f"\nData shape: X={X.shape}, y={y.shape}")
    print(f"Features: {len(feature_cols)}")
    print(f"Target (position): min={y.min():.1f}, max={y.max():.1f}, mean={y.mean():.2f}")

    print_fk_summary(col_to_fk)

    # Verify non-zero variance in target
    print(f"\nTarget variance: {y.var():.4f} (should be non-zero for regression)")
