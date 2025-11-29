"""
FK-Level Risk Attribution - Amazon Validation (Lightweight)
============================================================

Loads parquet directly to avoid memory issues.
"""

import numpy as np
import pandas as pd
from collections import defaultdict
import lightgbm as lgb
from scipy import stats
from pathlib import Path

# Cache location
CACHE_DIR = Path.home() / 'Library/Caches/relbench/rel-amazon/db'


def get_fk_group(col_name: str) -> str:
    """Map column name to FK group."""
    col_lower = col_name.lower()

    # Customer aggregates
    if 'customer' in col_lower or 'user' in col_lower:
        return 'CUSTOMER'

    # Product aggregates
    if 'product' in col_lower or 'item' in col_lower or 'price' in col_lower or 'brand' in col_lower:
        return 'PRODUCT'

    # Review aggregates (behavioral)
    return 'REVIEW'


def load_amazon_light(sample_size: int = 3000) -> tuple:
    """Load Amazon data directly from parquet with early sampling."""
    print("[DATA] Loading parquet files...")

    # Load customer parquet
    customer_df = pd.read_parquet(CACHE_DIR / 'customer.parquet')
    print(f"  Customer: {customer_df.shape}")

    # Load product parquet (for price info)
    product_df = pd.read_parquet(CACHE_DIR / 'product.parquet')
    print(f"  Product: {product_df.shape}")

    # Sample customers FIRST to reduce review loading
    sample_customers = customer_df.sample(n=min(sample_size, len(customer_df)), random_state=42)
    customer_ids = set(sample_customers['customer_id'].tolist())
    print(f"  Sampled customers: {len(customer_ids)}")

    # Load review parquet with filter on sampled customers
    print("[DATA] Loading reviews (large file)...")
    review_df = pd.read_parquet(CACHE_DIR / 'review.parquet')
    print(f"  Full reviews: {review_df.shape}")

    # Filter to sampled customers
    review_df = review_df[review_df['customer_id'].isin(customer_ids)]
    print(f"  Filtered reviews: {review_df.shape}")

    # Create features for each customer
    print("[DATA] Creating features...")

    # Review aggregates (REVIEW FK)
    review_agg = review_df.groupby('customer_id').agg({
        'rating': ['mean', 'std', 'count'],
    }).reset_index()
    review_agg.columns = ['customer_id', 'rating_mean', 'rating_std', 'review_count']
    review_agg['rating_std'] = review_agg['rating_std'].fillna(0)

    # Price aggregates (PRODUCT FK)
    review_with_price = review_df.merge(
        product_df[['product_id', 'price']],
        on='product_id',
        how='left'
    )
    price_agg = review_with_price.groupby('customer_id').agg({
        'price': ['mean', 'std', 'sum', 'max', 'min']
    }).reset_index()
    price_agg.columns = ['customer_id', 'price_mean', 'price_std', 'price_sum', 'price_max', 'price_min']
    price_agg['price_std'] = price_agg['price_std'].fillna(0)

    # Customer behavioral features (CUSTOMER FK)
    # Calculate recency and frequency patterns
    review_df['review_time'] = pd.to_datetime(review_df['review_time'], errors='coerce')
    customer_agg = review_df.groupby('customer_id').agg({
        'review_time': ['min', 'max'],
        'product_id': 'nunique'
    }).reset_index()
    customer_agg.columns = ['customer_id', 'first_review', 'last_review', 'unique_products']
    customer_agg['tenure_days'] = (customer_agg['last_review'] - customer_agg['first_review']).dt.days.fillna(0)
    customer_agg = customer_agg[['customer_id', 'unique_products', 'tenure_days']]

    # Merge all features
    merged = review_agg.merge(price_agg, on='customer_id', how='outer')
    merged = merged.merge(customer_agg, on='customer_id', how='outer')

    # Fill NaN
    for col in merged.columns:
        if col != 'customer_id':
            merged[col] = pd.to_numeric(merged[col], errors='coerce').fillna(0)

    # Create target: future LTV = log(price * engagement)
    merged['target'] = np.log1p(merged['price_mean'] * merged['review_count'])

    # Feature columns: 3 FK groups
    feature_cols = [
        # REVIEW: rating patterns
        'rating_mean', 'rating_std',
        # PRODUCT: price patterns
        'price_std', 'price_max', 'price_min',
        # CUSTOMER: behavioral patterns
        'unique_products', 'tenure_days'
    ]

    X = merged[feature_cols].copy()
    y = merged['target'].copy()

    # Map to FK groups
    col_to_fk = {
        'rating_mean': 'REVIEW',
        'rating_std': 'REVIEW',
        'price_std': 'PRODUCT',
        'price_max': 'PRODUCT',
        'price_min': 'PRODUCT',
        'unique_products': 'CUSTOMER',
        'tenure_days': 'CUSTOMER',
    }

    print(f"\n[DATA] Final: X={X.shape}, y={len(y)}")
    print(f"[DATA] Target stats: mean={y.mean():.2f}, std={y.std():.2f}")

    return X, y, feature_cols, col_to_fk


def train_ensemble(X: pd.DataFrame, y: pd.Series, n_models: int = 5, base_seed: int = 42) -> list:
    """Train ensemble of LightGBM models."""
    models = []
    for i in range(n_models):
        model = lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            num_leaves=31,
            random_state=base_seed + i,
            verbose=-1,
            force_col_wise=True
        )
        model.fit(X, y)
        models.append(model)
    return models


def predict_with_uncertainty(models: list, X: pd.DataFrame) -> tuple:
    """Get predictions and epistemic uncertainty."""
    preds = np.array([m.predict(X) for m in models])
    mean_pred = preds.mean(axis=0)
    variance = preds.var(axis=0)
    return mean_pred, variance


def fk_noise_injection(models: list, X: pd.DataFrame, col_to_fk: dict, n_perm: int = 5) -> dict:
    """Compute FK attribution via noise injection."""
    _, base_var = predict_with_uncertainty(models, X)
    base_unc = base_var.mean()

    # Group by FK
    fk_to_cols = defaultdict(list)
    for col, fk in col_to_fk.items():
        if col in X.columns:
            fk_to_cols[fk].append(col)

    # Permute each FK
    fk_deltas = {}
    for fk, cols in fk_to_cols.items():
        deltas = []
        for _ in range(n_perm):
            X_perm = X.copy()
            for col in cols:
                X_perm[col] = np.random.permutation(X_perm[col].values)
            _, perm_var = predict_with_uncertainty(models, X_perm)
            deltas.append(perm_var.mean() - base_unc)
        fk_deltas[fk] = np.mean(deltas)

    # Normalize
    total = sum(max(0, d) for d in fk_deltas.values())
    if total > 0:
        return {fk: max(0, d) / total * 100 for fk, d in fk_deltas.items()}
    return {fk: 0.0 for fk in fk_deltas}


def run_stability_test(X: pd.DataFrame, y: pd.Series, col_to_fk: dict, n_seeds: int = 3) -> tuple:
    """Stability test across seeds."""
    results = []

    for seed in range(n_seeds):
        models = train_ensemble(X, y, n_models=5, base_seed=100*seed)
        attr = fk_noise_injection(models, X, col_to_fk, n_perm=3)
        results.append(attr)

    # Spearman correlation between runs
    fks = list(results[0].keys())
    correlations = []

    for i in range(len(results)):
        for j in range(i+1, len(results)):
            vals_i = [results[i][fk] for fk in fks]
            vals_j = [results[j][fk] for fk in fks]
            rho, _ = stats.spearmanr(vals_i, vals_j)
            correlations.append(rho)

    stability = np.mean(correlations) if correlations else 1.0
    avg_attr = {fk: np.mean([r[fk] for r in results]) for fk in fks}

    return stability, avg_attr


def main():
    print("=" * 60)
    print("FK-Level Risk Attribution: Amazon Validation (Light)")
    print("=" * 60)

    # Load data
    print("\n[1/3] Loading data...")
    X, y, feature_cols, col_to_fk = load_amazon_light(sample_size=3000)

    # FK summary
    fk_to_cols = defaultdict(list)
    for col, fk in col_to_fk.items():
        fk_to_cols[fk].append(col)

    print("\nFK Groups:")
    for fk, cols in sorted(fk_to_cols.items()):
        print(f"  {fk}: {cols}")

    # Decomposition
    print("\n[2/3] FK Decomposition...")
    models = train_ensemble(X, y)
    attr = fk_noise_injection(models, X, col_to_fk, n_perm=5)

    print("\nAttribution:")
    for fk, pct in sorted(attr.items(), key=lambda x: -x[1]):
        print(f"  {fk}: {pct:.1f}%")

    # Stability
    print("\n[3/3] Stability Test...")
    stability, avg_attr = run_stability_test(X, y, col_to_fk, n_seeds=3)

    print(f"\nStability (ρ): {stability:.3f}")
    print("\nAvg Attribution:")
    for fk, pct in sorted(avg_attr.items(), key=lambda x: -x[1]):
        print(f"  {fk}: {pct:.1f}%")

    # Summary
    print("\n" + "=" * 60)
    print("AMAZON VALIDATION SUMMARY")
    print("=" * 60)
    top_fk = max(avg_attr.items(), key=lambda x: x[1])
    print(f"Domain: E-commerce (LTV prediction)")
    print(f"Stability: {stability:.3f}")
    print(f"Top FK: {top_fk[0]} ({top_fk[1]:.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
