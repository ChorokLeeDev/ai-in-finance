"""
Data Loader for rel-amazon Dataset (FK-Level Risk Attribution)
================================================================

rel-amazon schema:
- product: product_id (pkey), category, brand, title, description, price
- customer: customer_id (pkey), customer_name
- review: customer_id (fkey→customer), product_id (fkey→product),
          review_time, rating, verified, review_text, summary

FK Groups:
- CUSTOMER: customer-related features
- PRODUCT: product-related features
- REVIEW: review-related features (behavioral)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from relbench.datasets import get_dataset
from relbench.tasks import get_task

from cache import save_cache, load_cache


# FK Mapping: column_name -> FK group
FK_MAPPING = {
    # Customer features
    'customer_id': 'CUSTOMER',
    'customer_name': 'CUSTOMER',

    # Product features
    'product_id': 'PRODUCT',
    'category': 'PRODUCT',
    'brand': 'PRODUCT',
    'title': 'PRODUCT',
    'description': 'PRODUCT',
    'price': 'PRODUCT',

    # Review features (behavioral)
    'review_time': 'REVIEW',
    'rating': 'REVIEW',
    'verified': 'REVIEW',
    'review_text': 'REVIEW',
    'summary': 'REVIEW',

    # Aggregated features (will be mapped based on prefix)
    # count_*, mean_*, sum_*, etc.
}


def get_fk_group(col_name: str) -> str:
    """Map column name to FK group."""
    # Direct mapping
    if col_name in FK_MAPPING:
        return FK_MAPPING[col_name]

    # Prefix-based mapping for aggregated features
    col_lower = col_name.lower()

    # Customer aggregates
    if 'customer' in col_lower or 'user' in col_lower:
        return 'CUSTOMER'

    # Product aggregates
    if 'product' in col_lower or 'item' in col_lower or 'price' in col_lower or 'brand' in col_lower:
        return 'PRODUCT'

    # Review aggregates (behavioral patterns)
    if any(x in col_lower for x in ['rating', 'review', 'verified', 'count', 'mean', 'sum', 'std']):
        return 'REVIEW'

    # Default to REVIEW for unknown features (often behavioral)
    return 'REVIEW'


def load_amazon_data(
    task_name: str = 'user-ltv',
    sample_size: int = 3000,
    use_cache: bool = True
) -> Tuple[pd.DataFrame, pd.Series, List[str], Dict[str, str]]:
    """
    Load rel-amazon data for FK-level risk attribution.

    Args:
        task_name: Task name (user-ltv, item-ltv, etc.)
        sample_size: Number of samples to use
        use_cache: Whether to use cached data

    Returns:
        X: Feature DataFrame
        y: Target Series
        feature_cols: List of feature column names
        col_to_fk: Column to FK group mapping
    """
    cache_key = f"data_amazon_{task_name}_{sample_size}"

    if use_cache:
        cached = load_cache(cache_key)
        if cached is not None:
            print(f"[CACHE] Loaded: {cache_key}.pkl")
            return (
                cached['X'],
                cached['y'],
                cached['feature_cols'],
                cached['col_to_fk']
            )

    print(f"[DATA] Loading rel-amazon {task_name}...")

    # Load dataset and task
    dataset = get_dataset('rel-amazon', download=True)
    task = get_task('rel-amazon', task_name, download=True)
    db = dataset.get_db()

    # Get train table
    train_table = task.get_table('train')
    train_df = train_table.df.copy()

    print(f"[DATA] Train table shape: {train_df.shape}")
    print(f"[DATA] Columns: {list(train_df.columns)}")

    # Get target column
    target_col = task.target_col
    print(f"[DATA] Target column: {target_col}")

    # Join with customer table
    customer_df = db.table_dict['customer'].df.copy()
    print(f"[DATA] Customer table: {customer_df.shape}")

    # Join with product info through reviews
    review_df = db.table_dict['review'].df.copy()
    product_df = db.table_dict['product'].df.copy()
    print(f"[DATA] Review table: {review_df.shape}")
    print(f"[DATA] Product table: {product_df.shape}")

    # Create aggregated features per customer
    # Review aggregates
    review_agg = review_df.groupby('customer_id').agg({
        'rating': ['mean', 'std', 'count'],
        'verified': ['mean', 'sum'],
    }).reset_index()
    review_agg.columns = ['customer_id', 'rating_mean', 'rating_std', 'review_count',
                          'verified_ratio', 'verified_count']
    review_agg['rating_std'] = review_agg['rating_std'].fillna(0)

    # Product price aggregates (via reviews)
    review_with_price = review_df.merge(product_df[['product_id', 'price']], on='product_id', how='left')
    price_agg = review_with_price.groupby('customer_id').agg({
        'price': ['mean', 'std', 'sum', 'max', 'min']
    }).reset_index()
    price_agg.columns = ['customer_id', 'price_mean', 'price_std', 'price_sum', 'price_max', 'price_min']
    price_agg['price_std'] = price_agg['price_std'].fillna(0)

    # Merge features
    merged = train_df.merge(review_agg, on='customer_id', how='left')
    merged = merged.merge(price_agg, on='customer_id', how='left')

    # Fill NaN
    for col in merged.columns:
        if merged[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            merged[col] = merged[col].fillna(0)

    # Sample
    if len(merged) > sample_size:
        merged = merged.sample(n=sample_size, random_state=42)

    # Separate features and target
    feature_cols = [c for c in merged.columns if c not in [target_col, 'customer_id', 'timestamp']]

    X = merged[feature_cols].copy()
    y = merged[target_col].copy()

    # Map columns to FK groups
    col_to_fk = {col: get_fk_group(col) for col in feature_cols}

    print(f"\n[DATA] Final shape: X={X.shape}, y={len(y)}")
    print(f"[DATA] FK groups: {set(col_to_fk.values())}")

    # Print FK summary
    print_fk_summary(col_to_fk)

    # Cache
    if use_cache:
        save_cache(cache_key, {
            'X': X,
            'y': y,
            'feature_cols': feature_cols,
            'col_to_fk': col_to_fk
        })

    return X, y, feature_cols, col_to_fk


def print_fk_summary(col_to_fk: Dict[str, str]):
    """Print FK group summary."""
    from collections import defaultdict
    fk_to_cols = defaultdict(list)
    for col, fk in col_to_fk.items():
        fk_to_cols[fk].append(col)

    print("\n--- FK Groups ---")
    for fk, cols in sorted(fk_to_cols.items()):
        print(f"  {fk}: {cols}")


if __name__ == "__main__":
    print("Testing rel-amazon data loading...")

    X, y, feature_cols, col_to_fk = load_amazon_data(
        task_name='user-ltv',
        sample_size=1000,
        use_cache=False
    )

    print(f"\nFeatures: {X.shape}")
    print(f"Target: {y.shape}, mean={y.mean():.2f}, std={y.std():.2f}")
    print(f"\nFeature columns: {feature_cols}")
    print(f"\nFK mapping: {col_to_fk}")
