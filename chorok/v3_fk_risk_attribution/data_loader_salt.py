"""
Data Loader for rel-salt (SAP ERP Data)
=======================================

Real SAP sales data with FK structure:
- salesdocumentitem: 1.9M rows (main entity)
- salesdocument: 411K sales headers
- customer: 140K customers
- address: 1.8M addresses

FK relationships:
- salesdocumentitem → SALESDOCUMENT
- salesdocumentitem → SOLDTOPARTY, SHIPTOPARTY, BILLTOPARTY, PAYERPARTY (customers)
- customer → ADDRESSID

Regression target: PLANT (which plant fulfills the order)
"""

import numpy as np
import pandas as pd
from relbench.datasets import get_dataset
import os
import pickle

CACHE_DIR = '/Users/i767700/Github/ai-in-finance/chorok/v3_fk_risk_attribution/cache'


def load_salt_data(sample_size=5000, use_cache=True, target='PLANT'):
    """
    Load SAP sales data for FK-level uncertainty attribution.

    Returns:
        X: Feature DataFrame
        y: Target series
        feature_cols: List of feature column names
        col_to_fk: Dict mapping column to FK group
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = f'{CACHE_DIR}/data_salt_{target}_{sample_size}.pkl'

    if use_cache and os.path.exists(cache_file):
        print(f"[CACHE] Loading from {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    print("[DATA] Loading rel-salt dataset...")
    ds = get_dataset("rel-salt", download=True)
    db = ds.get_db()

    # Get tables
    items = db.table_dict['salesdocumentitem'].df.copy()
    docs = db.table_dict['salesdocument'].df.copy()
    customers = db.table_dict['customer'].df.copy()
    addresses = db.table_dict['address'].df.copy()

    print(f"  Items: {items.shape}")
    print(f"  Docs: {docs.shape}")
    print(f"  Customers: {customers.shape}")

    # Sample items
    if len(items) > sample_size:
        items = items.sample(n=sample_size, random_state=42)

    # Merge with sales document (header info)
    doc_cols = ['SALESDOCUMENT', 'SALESOFFICE', 'SALESGROUP',
                'CUSTOMERPAYMENTTERMS', 'SHIPPINGCONDITION',
                'HEADERINCOTERMSCLASSIFICATION']
    # Only use columns that exist
    doc_cols = [c for c in doc_cols if c in docs.columns]
    df = items.merge(
        docs[doc_cols],
        on='SALESDOCUMENT',
        how='left',
        suffixes=('', '_doc')
    )

    # Merge with customer (sold-to party)
    customers_renamed = customers.rename(columns={
        'CUSTOMER': 'SOLDTOPARTY',
        'ADDRESSID': 'SOLDTO_ADDRESSID'
    })
    df = df.merge(customers_renamed, on='SOLDTOPARTY', how='left')

    # Merge with customer (ship-to party)
    customers_shipto = customers.rename(columns={
        'CUSTOMER': 'SHIPTOPARTY',
        'ADDRESSID': 'SHIPTO_ADDRESSID'
    })
    df = df.merge(customers_shipto, on='SHIPTOPARTY', how='left')

    print(f"  Merged: {df.shape}")

    # Define FK groups
    col_to_fk = {
        # SALESDOCUMENT FK (header attributes)
        'SALESOFFICE': 'SALESDOCUMENT',
        'SALESGROUP': 'SALESGROUP',
        'CUSTOMERPAYMENTTERMS': 'SALESDOCUMENT',
        'SHIPPINGCONDITION': 'SALESDOCUMENT',
        'HEADERINCOTERMSCLASSIFICATION': 'SALESDOCUMENT',

        # SOLDTOPARTY FK (customer who ordered)
        'SOLDTOPARTY': 'SOLDTOPARTY',
        'SOLDTO_ADDRESSID': 'SOLDTOPARTY',

        # SHIPTOPARTY FK (customer who receives)
        'SHIPTOPARTY': 'SHIPTOPARTY',
        'SHIPTO_ADDRESSID': 'SHIPTOPARTY',

        # ITEM attributes (direct)
        'SHIPPINGPOINT': 'ITEM',
        'ITEMINCOTERMSCLASSIFICATION': 'ITEM',
    }

    # Features
    feature_cols = [c for c in col_to_fk.keys() if c in df.columns and c != target]

    # Filter valid rows
    df = df[df[target].notna()]
    df = df[feature_cols + [target]].dropna()

    X = df[feature_cols].copy()
    y = df[target].astype(float)

    # Update col_to_fk for actual columns
    col_to_fk = {c: col_to_fk[c] for c in feature_cols}

    print(f"\n[DATA] Final: X={X.shape}, y={len(y)}")
    print(f"[DATA] FK groups: {set(col_to_fk.values())}")
    print(f"[DATA] Target '{target}': min={y.min()}, max={y.max()}, mean={y.mean():.2f}")

    # Cache
    result = (X, y, feature_cols, col_to_fk)
    with open(cache_file, 'wb') as f:
        pickle.dump(result, f)

    return result


if __name__ == "__main__":
    X, y, feature_cols, col_to_fk = load_salt_data(sample_size=5000)

    print("\nFeatures by FK group:")
    from collections import defaultdict
    fk_to_cols = defaultdict(list)
    for col, fk in col_to_fk.items():
        fk_to_cols[fk].append(col)

    for fk, cols in fk_to_cols.items():
        print(f"  {fk}: {cols}")
