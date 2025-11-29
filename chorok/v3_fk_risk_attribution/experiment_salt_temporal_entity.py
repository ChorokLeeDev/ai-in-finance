"""
Temporal + FK + Entity Analysis on SAP SALT Data
=================================================

Combines three dimensions:
1. FK structure (SOLDTOPARTY, SHIPTOPARTY, SALESDOCUMENT, ITEM)
2. Entity level (specific customers, shipping points)
3. Temporal dimension (across time periods)

Business Questions:
- "How does uncertainty for Customer X change over time?"
- "Which periods show high uncertainty spikes?"
- "What explains the difference between stable vs volatile entities?"
"""

import numpy as np
import pandas as pd
from collections import defaultdict
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

from relbench.datasets import get_dataset
import os
import pickle

CACHE_DIR = '/Users/i767700/Github/ai-in-finance/chorok/v3_fk_risk_attribution/cache'


def load_salt_temporal(sample_size=20000, use_cache=True):
    """Load SAP SALT data with temporal info preserved."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = f'{CACHE_DIR}/data_salt_temporal_{sample_size}.pkl'

    if use_cache and os.path.exists(cache_file):
        print(f"[CACHE] Loading from {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    print("[DATA] Loading rel-salt dataset with temporal info...")
    ds = get_dataset("rel-salt", download=True)
    db = ds.get_db()

    # Get tables
    items = db.table_dict['salesdocumentitem'].df.copy()
    docs = db.table_dict['salesdocument'].df.copy()
    customers = db.table_dict['customer'].df.copy()

    print(f"  Items: {items.shape}")
    print(f"  Docs: {docs.shape}")

    # Sample items
    if len(items) > sample_size:
        items = items.sample(n=sample_size, random_state=42)

    # Merge with sales document to get CREATIONDATE
    doc_cols = ['SALESDOCUMENT', 'SALESOFFICE', 'SALESGROUP',
                'CUSTOMERPAYMENTTERMS', 'SHIPPINGCONDITION',
                'HEADERINCOTERMSCLASSIFICATION', 'CREATIONDATE']
    doc_cols = [c for c in doc_cols if c in docs.columns]

    df = items.merge(docs[doc_cols], on='SALESDOCUMENT', how='left')

    # Merge with customer
    customers_renamed = customers.rename(columns={
        'CUSTOMER': 'SOLDTOPARTY',
        'ADDRESSID': 'SOLDTO_ADDRESSID'
    })
    df = df.merge(customers_renamed, on='SOLDTOPARTY', how='left')

    # Extract temporal features if CREATIONDATE exists
    if 'CREATIONDATE' in df.columns:
        df['CREATIONDATE'] = pd.to_datetime(df['CREATIONDATE'], errors='coerce')
        df['year'] = df['CREATIONDATE'].dt.year
        df['month'] = df['CREATIONDATE'].dt.month
        df['quarter'] = df['CREATIONDATE'].dt.quarter
        df['year_month'] = df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2)
        has_temporal = True
    else:
        # Use item index as proxy for temporal ordering
        df['year'] = 2020  # dummy
        df['month'] = (df.index % 12) + 1
        df['quarter'] = ((df.index % 12) // 3) + 1
        df['year_month'] = '2020-' + df['month'].astype(str).str.zfill(2)
        has_temporal = False
        print("  Note: CREATIONDATE not available, using synthetic temporal structure")

    # Define target and features
    target = 'PLANT'
    df = df[df[target].notna()]

    # FK groups
    col_to_fk = {
        'SALESOFFICE': 'SALESDOCUMENT',
        'SALESGROUP': 'SALESGROUP',
        'CUSTOMERPAYMENTTERMS': 'SALESDOCUMENT',
        'SHIPPINGCONDITION': 'SALESDOCUMENT',
        'HEADERINCOTERMSCLASSIFICATION': 'SALESDOCUMENT',
        'SOLDTOPARTY': 'SOLDTOPARTY',
        'SOLDTO_ADDRESSID': 'SOLDTOPARTY',
        'SHIPPINGPOINT': 'ITEM',
        'ITEMINCOTERMSCLASSIFICATION': 'ITEM',
    }

    feature_cols = [c for c in col_to_fk.keys() if c in df.columns and c != target]
    temporal_cols = ['year', 'month', 'quarter', 'year_month']

    df = df[feature_cols + [target] + temporal_cols + ['SALESDOCUMENT']].dropna()

    X = df[feature_cols].copy()
    y = df[target].astype(float)
    temporal_info = df[temporal_cols + ['SALESDOCUMENT']].copy()

    col_to_fk = {c: col_to_fk[c] for c in feature_cols}

    print(f"\n[DATA] Final: X={X.shape}, y={len(y)}")
    print(f"[DATA] Temporal range: {temporal_info['year_month'].min()} to {temporal_info['year_month'].max()}")

    result = (X, y, feature_cols, col_to_fk, temporal_info, has_temporal)
    with open(cache_file, 'wb') as f:
        pickle.dump(result, f)

    return result


def train_ensemble(X, y, n_models=10, base_seed=42):
    models = []
    for i in range(n_models):
        model = lgb.LGBMRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            random_state=base_seed + i, verbose=-1
        )
        model.fit(X, y)
        models.append(model)
    return models


def get_uncertainty(models, X):
    preds = np.array([m.predict(X) for m in models])
    return preds.var(axis=0)


def temporal_entity_analysis(models, X, y, temporal_info, entity_col, n_top=5):
    """Analyze how entity uncertainty changes over time."""
    uncertainties = get_uncertainty(models, X)

    # Combine data
    data = pd.DataFrame({
        'entity': X[entity_col].values,
        'uncertainty': uncertainties,
        'year_month': temporal_info['year_month'].values,
        'target': y.values
    })

    # Get top entities by frequency
    top_entities = data.groupby('entity').size().nlargest(n_top).index.tolist()

    results = {}
    for entity in top_entities:
        entity_data = data[data['entity'] == entity]

        temporal_stats = entity_data.groupby('year_month').agg({
            'uncertainty': ['mean', 'std', 'count'],
            'target': 'mean'
        }).reset_index()
        temporal_stats.columns = ['period', 'mean_unc', 'std_unc', 'count', 'avg_target']

        results[entity] = temporal_stats

    return results, data


def entity_stability_analysis(data, min_periods=3):
    """Classify entities by uncertainty stability over time."""
    entity_temporal = data.groupby('entity').agg({
        'year_month': 'nunique',
        'uncertainty': ['mean', 'std']
    })
    entity_temporal.columns = ['n_periods', 'mean_unc', 'std_unc']
    entity_temporal = entity_temporal.reset_index()

    # Filter entities with enough temporal coverage
    entity_temporal = entity_temporal[entity_temporal['n_periods'] >= min_periods]

    # Coefficient of variation = std/mean (normalized volatility)
    entity_temporal['cv'] = entity_temporal['std_unc'] / (entity_temporal['mean_unc'] + 1e-10)

    # Classify
    stable = entity_temporal.nsmallest(5, 'cv')
    volatile = entity_temporal.nlargest(5, 'cv')

    return stable, volatile, entity_temporal


def run_temporal_experiment():
    print("=" * 80)
    print("SAP SALT: Temporal + FK + Entity Analysis")
    print("=" * 80)

    # Load data
    print("\n[1/4] Loading SAP SALT data with temporal structure...")
    X, y, feature_cols, col_to_fk, temporal_info, has_temporal = load_salt_temporal(sample_size=50000)

    print(f"\nData shape: X={X.shape}")
    print(f"Unique periods: {temporal_info['year_month'].nunique()}")

    # Train ensemble
    print("\n[2/4] Training ensemble...")
    models = train_ensemble(X, y, n_models=10)

    # Temporal + entity analysis
    print("\n[3/4] Temporal entity analysis...")

    entity_cols = ['SOLDTOPARTY', 'SALESGROUP', 'SHIPPINGPOINT']
    all_results = {}

    for entity_col in entity_cols:
        if entity_col in X.columns:
            results, data = temporal_entity_analysis(
                models, X, y, temporal_info, entity_col, n_top=3
            )
            all_results[entity_col] = (results, data)

            print(f"\n{entity_col} Temporal Patterns:")
            print("-" * 50)

            for entity, stats in results.items():
                print(f"\n  Entity {int(entity)}:")
                for _, row in stats.tail(5).iterrows():
                    bar = "█" * int(min(row['mean_unc'] * 50, 20))
                    print(f"    {row['period']}: unc={row['mean_unc']:.4f} (n={int(row['count'])}) {bar}")

    # Stability analysis
    print("\n[4/4] Entity stability classification...")
    print("=" * 80)

    for entity_col in entity_cols:
        if entity_col in all_results:
            _, data = all_results[entity_col]
            stable, volatile, all_entity = entity_stability_analysis(data)

            print(f"\n{entity_col} Stability Analysis:")
            print("-" * 50)

            print("\n  STABLE entities (low uncertainty variance over time):")
            for _, row in stable.iterrows():
                print(f"    Entity {int(row['entity'])}: mean_unc={row['mean_unc']:.4f}, "
                      f"CV={row['cv']:.2f}, periods={int(row['n_periods'])}")

            print("\n  VOLATILE entities (high uncertainty variance over time):")
            for _, row in volatile.iterrows():
                print(f"    Entity {int(row['entity'])}: mean_unc={row['mean_unc']:.4f}, "
                      f"CV={row['cv']:.2f}, periods={int(row['n_periods'])}")

    # Business recommendations
    print("\n" + "=" * 80)
    print("TEMPORAL-AWARE RECOMMENDATIONS")
    print("=" * 80)

    print("""
1. ENTITY SELECTION BY STABILITY:
   → Prefer STABLE entities (low uncertainty CV) for critical orders
   → Use VOLATILE entities only when necessary, with larger safety margins

2. TEMPORAL MONITORING:
   → Track uncertainty trends over time for key entities
   → Early warning: uncertainty spike indicates potential fulfillment issues

3. RISK ALLOCATION:
   → Diversify across stable entities during uncertain periods
   → Concentrate on predictable entities when reliability is paramount
""")

    print("=" * 80)

    return all_results


if __name__ == "__main__":
    results = run_temporal_experiment()
