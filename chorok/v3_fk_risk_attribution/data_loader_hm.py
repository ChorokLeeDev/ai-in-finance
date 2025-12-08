"""
Data Loader for rel-hm (H&M Fashion Retail)
============================================

Domain: H&M (Fashion e-commerce)
Task: item-sales (predict sales count for items)
Type: Regression (Error Propagation expected)

FK Structure:
  transactions -> article (article_id)
  transactions -> customer (customer_id)

Expected: Error Propagation domain (retail, master-transaction pattern)

Author: ChorokLeeDev
Created: 2025-12-08
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path

from relbench.datasets import get_dataset
from relbench.tasks import get_task


CACHE_DIR = Path('/Users/i767700/Github/ai-in-finance/chorok/v3_fk_risk_attribution/cache')
CACHE_DIR.mkdir(exist_ok=True)


def load_hm_data(task_name='item-sales', sample_size=3000, use_cache=True):
    """
    Load H&M data for FK attribution experiments.

    Returns:
        X: Feature DataFrame
        y: Target Series
        feature_cols: List of feature column names
        col_to_fk: Dict mapping column -> FK group
    """
    cache_file = CACHE_DIR / f'data_hm_{task_name}_{sample_size}.pkl'

    if use_cache and cache_file.exists():
        print(f"[CACHE] Loading: {cache_file.name}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    print(f"Loading rel-hm dataset (task: {task_name})...")

    # Load dataset and task
    dataset = get_dataset('rel-hm', download=True)
    task = get_task('rel-hm', task_name, download=True)
    db = dataset.get_db()

    # Get tables
    article_df = db.table_dict['article'].df.copy()

    # Get training data with targets
    train_table = task.get_table('train')
    train_df = train_table.df.copy()

    target_col = task.target_col  # 'sales'

    # Merge with article features
    merged = train_df.merge(article_df, on='article_id', how='left')

    print(f"  Merged shape: {merged.shape}")

    # Sample
    if len(merged) > sample_size:
        merged = merged.sample(sample_size, random_state=42)

    # Define FK groups (semantic grouping within article table)
    # Since H&M only has ARTICLE FK, we create semantic sub-groups
    col_to_fk = {}

    # PRODUCT: product type and group
    product_cols = ['product_type_no', 'product_group_name']
    for col in product_cols:
        if col in merged.columns:
            col_to_fk[col] = 'PRODUCT'

    # APPEARANCE: visual attributes
    appearance_cols = ['graphical_appearance_no', 'colour_group_code',
                       'perceived_colour_value_id', 'perceived_colour_master_id']
    for col in appearance_cols:
        if col in merged.columns:
            col_to_fk[col] = 'APPEARANCE'

    # ORGANIZATION: department and index structure
    org_cols = ['department_no', 'index_code', 'index_group_no',
                'section_no', 'garment_group_no']
    for col in org_cols:
        if col in merged.columns:
            col_to_fk[col] = 'ORGANIZATION'

    # Exclude non-feature columns
    exclude = {'article_id', 'timestamp', target_col, 'prod_name', 'product_type_name',
               'graphical_appearance_name', 'colour_group_name', 'perceived_colour_value_name',
               'perceived_colour_master_name', 'department_name', 'index_name',
               'index_group_name', 'section_name', 'garment_group_name', 'detail_desc'}

    feature_cols = [c for c in col_to_fk.keys() if c not in exclude and c in merged.columns]

    # If too few features, add more
    if len(feature_cols) < 3:
        for col in merged.columns:
            if col not in exclude and col not in feature_cols and merged[col].dtype in ['int64', 'float64']:
                col_to_fk[col] = 'ARTICLE'
                feature_cols.append(col)

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

    # Filter col_to_fk
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
    X, y, feature_cols, col_to_fk = load_hm_data(sample_size=3000)
    print(f"\nFeature columns: {feature_cols}")
    print(f"\nFK groups:")
    for fk in set(col_to_fk.values()):
        cols = [c for c, f in col_to_fk.items() if f == fk]
        print(f"  {fk}: {cols}")
