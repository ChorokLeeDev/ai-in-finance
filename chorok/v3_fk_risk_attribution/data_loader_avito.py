"""
Data Loader for rel-avito (Online Classifieds)
===============================================

Domain: Avito (Russian classifieds platform)
Task: ad-ctr (predict number of clicks for ads)
Type: Regression (Error Propagation expected)

FK Structure:
  AdsInfo -> Location (LocationID)
  AdsInfo -> Category (CategoryID)

Expected: Error Propagation domain (transactional, master-transaction pattern)

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


def load_avito_data(task_name='ad-ctr', sample_size=3000, use_cache=True):
    """
    Load Avito data for FK attribution experiments.

    Returns:
        X: Feature DataFrame
        y: Target Series
        feature_cols: List of feature column names
        col_to_fk: Dict mapping column -> FK group
    """
    cache_file = CACHE_DIR / f'data_avito_{task_name}_{sample_size}.pkl'

    if use_cache and cache_file.exists():
        print(f"[CACHE] Loading: {cache_file.name}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    print(f"Loading rel-avito dataset (task: {task_name})...")

    # Load dataset and task
    dataset = get_dataset('rel-avito', download=True)
    task = get_task('rel-avito', task_name, download=True)
    db = dataset.get_db()

    # Get tables
    ads_df = db.table_dict['AdsInfo'].df.copy()
    location_df = db.table_dict['Location'].df.copy()
    category_df = db.table_dict['Category'].df.copy()

    # Get training data with targets
    train_table = task.get_table('train')
    train_df = train_table.df.copy()

    # The task table has AdID and num_click (target)
    target_col = task.target_col  # 'num_click'

    # Merge with AdsInfo
    merged = train_df.merge(ads_df, on='AdID', how='left')

    # Merge with Location
    location_df_renamed = location_df.add_prefix('LOC_')
    location_df_renamed = location_df_renamed.rename(columns={'LOC_LocationID': 'LocationID'})
    merged = merged.merge(location_df_renamed, on='LocationID', how='left')

    # Merge with Category
    category_df_renamed = category_df.add_prefix('CAT_')
    category_df_renamed = category_df_renamed.rename(columns={'CAT_CategoryID': 'CategoryID'})
    merged = merged.merge(category_df_renamed, on='CategoryID', how='left')

    print(f"  Merged shape: {merged.shape}")

    # Sample
    if len(merged) > sample_size:
        merged = merged.sample(sample_size, random_state=42)

    # Define FK groups based on column prefixes
    col_to_fk = {}

    # AD features (from AdsInfo directly)
    ad_cols = ['Price', 'IsContext']
    for col in ad_cols:
        if col in merged.columns:
            col_to_fk[col] = 'AD'

    # Location features
    for col in merged.columns:
        if col.startswith('LOC_'):
            col_to_fk[col] = 'LOCATION'

    # Category features
    for col in merged.columns:
        if col.startswith('CAT_'):
            col_to_fk[col] = 'CATEGORY'

    # Exclude non-feature columns
    exclude = {'AdID', 'LocationID', 'CategoryID', 'timestamp', target_col}
    feature_cols = [c for c in col_to_fk.keys() if c not in exclude]

    # Prepare features
    X = merged[feature_cols].copy()

    # Handle categorical columns
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype.name == 'category':
            X[col] = X[col].astype('category').cat.codes
        X[col] = X[col].fillna(-999)

    # Prepare target
    y = merged[target_col].copy()
    y = y.fillna(0)

    # Filter col_to_fk to only include valid feature columns
    col_to_fk = {c: fk for c, fk in col_to_fk.items() if c in feature_cols}

    print(f"  Samples: {len(X)}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  FK groups: {len(set(col_to_fk.values()))}")
    print(f"  FK mapping: {set(col_to_fk.values())}")
    print(f"  Target '{target_col}': min={y.min():.1f}, max={y.max():.1f}, mean={y.mean():.2f}")

    # Cache
    result = (X, y, feature_cols, col_to_fk)
    with open(cache_file, 'wb') as f:
        pickle.dump(result, f)
    print(f"[CACHE] Saved: {cache_file.name}")

    return result


if __name__ == "__main__":
    X, y, feature_cols, col_to_fk = load_avito_data(sample_size=3000)
    print(f"\nFeature columns: {feature_cols}")
    print(f"\nFK groups:")
    for fk in set(col_to_fk.values()):
        cols = [c for c, f in col_to_fk.items() if f == fk]
        print(f"  {fk}: {cols}")
