"""
Data Loader for FK-Level Risk Attribution
==========================================

Load rel-salt data and define FK mappings.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path

from relbench.datasets import get_dataset
from relbench.tasks import get_task

from cache import cache_data, load_data


# =============================================================================
# FK MAPPING: Column → Source Table/Process
# =============================================================================

FK_MAPPING = {
    # Customer-related
    'SOLDTOPARTY': 'CUSTOMER',
    'SHIPTOPARTY': 'CUSTOMER',
    'BILLTOPARTY': 'CUSTOMER',
    'PAYERPARTY': 'CUSTOMER',
    'CUSTOMERPAYMENTTERMS': 'CUSTOMER',
    'CUSTOMERCOUNTRY': 'CUSTOMER',
    'CUSTOMERGROUP': 'CUSTOMER',

    # Sales Organization
    'SALESORGANIZATION': 'SALES_ORG',
    'SALESOFFICE': 'SALES_ORG',
    'ORGANIZATIONDIVISION': 'SALES_ORG',
    'SALESGROUP': 'SALES_ORG',

    # Distribution
    'DISTRIBUTIONCHANNEL': 'DISTRIBUTION',

    # Currency/Pricing
    'TRANSACTIONCURRENCY': 'CURRENCY',
    'PRICINGDATE': 'CURRENCY',

    # Shipping/Logistics
    'SHIPPINGCONDITION': 'SHIPPING',
    'HEADERINCOTERMSCLASSIFICATION': 'SHIPPING',
    'DELIVERYBLOCKSTATUS': 'SHIPPING',

    # Billing
    'BILLINGCOMPANYCODE': 'BILLING',
    'BILLINGBLOCKSTATUS': 'BILLING',

    # Document
    'SALESDOCUMENTTYPE': 'DOCUMENT',
    'SDDOCUMENTCATEGORY': 'DOCUMENT',
    'CREATIONDATE': 'DOCUMENT',

    # Product/Material
    'MATERIALGROUP': 'PRODUCT',
    'MATERIALTYPE': 'PRODUCT',
    'PLANT': 'PRODUCT',
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


def load_salt_data(
    task_name: str = 'sales-group',
    sample_size: int = 500,
    use_cache: bool = True
) -> Tuple[pd.DataFrame, pd.Series, List[str], Dict[str, str]]:
    """
    Load SALT dataset.

    Args:
        task_name: Task name (e.g., 'sales-group')
        sample_size: Number of samples
        use_cache: Whether to use cache

    Returns:
        X: Feature DataFrame
        y: Target Series
        feature_cols: List of feature column names
        col_to_fk: Column to FK mapping (for available columns)
    """
    # Try cache first
    if use_cache:
        cached = load_data(task_name, sample_size)
        if cached is not None:
            return (
                pd.DataFrame(cached['X']),
                pd.Series(cached['y']),
                cached['feature_cols'],
                cached['col_to_fk']
            )

    # Load from relbench
    print(f"Loading rel-salt/{task_name}...")
    task = get_task("rel-salt", task_name, download=False)
    dataset = get_dataset("rel-salt", download=False)
    db = dataset.get_db(upto_test_timestamp=False)

    entity_table = db.table_dict[task.entity_table]
    df = entity_table.df.copy()

    # Sample
    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)

    # Get features and target
    target_col = task.target_col
    exclude = {'CREATIONTIMESTAMP', target_col, entity_table.pkey_col}
    feature_cols = [c for c in df.columns if c not in exclude]

    X = df[feature_cols].copy()

    # Encode categorical columns
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = X[col].astype('category').cat.codes
        X[col] = X[col].fillna(-999)

    y = df[target_col].copy()
    if y.dtype == 'object':
        y = y.astype('category').cat.codes

    # Build column to FK mapping for available columns
    col_to_fk = {}
    for col in feature_cols:
        col_to_fk[col] = get_fk_group(col)

    # Cache
    if use_cache:
        cache_data(task_name, sample_size, {
            'X': X.to_dict(),
            'y': y.tolist(),
            'feature_cols': feature_cols,
            'col_to_fk': col_to_fk
        })

    return X, y, feature_cols, col_to_fk


def print_fk_summary(col_to_fk: Dict[str, str]):
    """Print FK mapping summary."""
    # Group by FK
    fk_to_cols = {}
    for col, fk in col_to_fk.items():
        if fk not in fk_to_cols:
            fk_to_cols[fk] = []
        fk_to_cols[fk].append(col)

    print("\n" + "="*60)
    print("FK MAPPING SUMMARY")
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
    # Test data loading
    print("Testing data loader...")

    X, y, feature_cols, col_to_fk = load_salt_data(
        task_name='sales-group',
        sample_size=500,
        use_cache=False  # Force reload for testing
    )

    print(f"\nData shape: X={X.shape}, y={y.shape}")
    print(f"Features: {len(feature_cols)}")
    print(f"Target classes: {y.nunique()}")

    print_fk_summary(col_to_fk)

    # Check for unmapped columns
    unmapped = [col for col, fk in col_to_fk.items() if fk == 'OTHER']
    if unmapped:
        print(f"\n⚠️  Unmapped columns ({len(unmapped)}):")
        for col in unmapped:
            print(f"  - {col}")
