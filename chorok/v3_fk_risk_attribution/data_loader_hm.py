"""
Data Loader for rel-hm (H&M Fashion Retail)
============================================

Domain: Fashion Retail
Task: item-sales (predict article sales)
Type: Regression

FK Structure:
  Article -> Product Type
  Article -> Product Group
  Article -> Department
  Article -> Section
  Article -> Garment Group

Expected: Error Propagation domain (hierarchical product taxonomy)

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
    cache_file = CACHE_DIR / f'data_hm_{task_name}_{sample_size}_v1.pkl'

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

    # Get training data
    train_table = task.get_table('train')
    train_df = train_table.df.copy()
    target_col = task.target_col

    print(f"  Training samples: {len(train_df)}")

    # Sample first to reduce memory usage
    if len(train_df) > sample_size:
        train_df = train_df.sample(sample_size, random_state=42)

    # Merge with article features
    merged = train_df.merge(article_df, on='article_id', how='left')

    print(f"  Merged shape: {merged.shape}")

    # Define FK groups based on article hierarchy
    col_to_fk = {}

    # Article basic info
    article_cols = ['product_code', 'prod_name', 'graphical_appearance_no',
                    'graphical_appearance_name', 'colour_group_code', 'colour_group_name']
    for col in article_cols:
        if col in merged.columns:
            col_to_fk[col] = 'ARTICLE'

    # Product Type (FK to product type dimension)
    type_cols = ['product_type_no', 'product_type_name']
    for col in type_cols:
        if col in merged.columns:
            col_to_fk[col] = 'PRODUCT_TYPE'

    # Product Group (FK to product group dimension)
    group_cols = ['product_group_name']
    for col in group_cols:
        if col in merged.columns:
            col_to_fk[col] = 'PRODUCT_GROUP'

    # Department (FK to department dimension)
    dept_cols = ['department_no', 'department_name']
    for col in dept_cols:
        if col in merged.columns:
            col_to_fk[col] = 'DEPARTMENT'

    # Section (FK to section dimension)
    section_cols = ['section_no', 'section_name']
    for col in section_cols:
        if col in merged.columns:
            col_to_fk[col] = 'SECTION'

    # Garment Group (FK to garment group dimension)
    garment_cols = ['garment_group_no', 'garment_group_name']
    for col in garment_cols:
        if col in merged.columns:
            col_to_fk[col] = 'GARMENT_GROUP'

    # Index (FK to index dimension - store concept)
    index_cols = ['index_code', 'index_name', 'index_group_no', 'index_group_name']
    for col in index_cols:
        if col in merged.columns:
            col_to_fk[col] = 'INDEX'

    # Detail columns
    detail_cols = ['detail_desc', 'perceived_colour_value_id', 'perceived_colour_value_name',
                   'perceived_colour_master_id', 'perceived_colour_master_name']
    for col in detail_cols:
        if col in merged.columns:
            col_to_fk[col] = 'DETAIL'

    # Exclude non-feature columns
    exclude = {'article_id', 'timestamp', target_col}
    feature_cols = [c for c in col_to_fk.keys() if c not in exclude and c in merged.columns]

    # Prepare features
    X = merged[feature_cols].copy()

    # Handle categorical/object columns
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype.name == 'category':
            X[col] = X[col].astype('category').cat.codes
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
    print(f"  Target '{target_col}': min={y.min():.3f}, max={y.max():.3f}, mean={y.mean():.3f}")

    # Cache
    result = (X, y, feature_cols, col_to_fk)
    with open(cache_file, 'wb') as f:
        pickle.dump(result, f)
    print(f"[CACHE] Saved: {cache_file.name}")

    return result


if __name__ == "__main__":
    X, y, feature_cols, col_to_fk = load_hm_data(sample_size=3000)
    print(f"\nFeature columns: {feature_cols[:10]}...")
    print(f"\nFK groups:")
    for fk in set(col_to_fk.values()):
        cols = [c for c, f in col_to_fk.items() if f == fk]
        print(f"  {fk}: {len(cols)} columns")
