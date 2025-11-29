"""
FK-Level Risk Attribution - Amazon Validation Experiment
=========================================================

Validate the framework on rel-amazon (E-commerce domain).
"""

import numpy as np
import pandas as pd
from collections import defaultdict
import lightgbm as lgb
from scipy import stats

from data_loader_amazon import load_amazon_data


def train_ensemble(X: pd.DataFrame, y: pd.Series, n_models: int = 5) -> list:
    """Train ensemble of LightGBM models."""
    models = []
    for i in range(n_models):
        model = lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            num_leaves=31,
            random_state=42 + i,
            verbose=-1,
            force_col_wise=True
        )
        model.fit(X, y)
        models.append(model)
    return models


def predict_with_uncertainty(models: list, X: pd.DataFrame) -> tuple:
    """Get predictions and epistemic uncertainty (variance)."""
    preds = np.array([m.predict(X) for m in models])
    mean_pred = preds.mean(axis=0)
    variance = preds.var(axis=0)
    return mean_pred, variance


def fk_noise_injection(
    models: list,
    X: pd.DataFrame,
    col_to_fk: dict,
    n_permutations: int = 5
) -> dict:
    """Compute FK-level attribution via noise injection."""
    # Baseline uncertainty
    _, base_var = predict_with_uncertainty(models, X)
    base_uncertainty = base_var.mean()

    # Group columns by FK
    fk_to_cols = defaultdict(list)
    for col, fk in col_to_fk.items():
        if col in X.columns:
            fk_to_cols[fk].append(col)

    # Permute each FK group
    fk_deltas = {}
    for fk, cols in fk_to_cols.items():
        deltas = []
        for _ in range(n_permutations):
            X_perm = X.copy()
            for col in cols:
                X_perm[col] = np.random.permutation(X_perm[col].values)
            _, perm_var = predict_with_uncertainty(models, X_perm)
            delta = perm_var.mean() - base_uncertainty
            deltas.append(delta)
        fk_deltas[fk] = np.mean(deltas)

    # Normalize to percentages
    total = sum(max(0, d) for d in fk_deltas.values())
    if total > 0:
        fk_attribution = {fk: max(0, d) / total * 100 for fk, d in fk_deltas.items()}
    else:
        fk_attribution = {fk: 0.0 for fk in fk_deltas}

    return fk_attribution


def run_stability_test(X: pd.DataFrame, y: pd.Series, col_to_fk: dict, n_seeds: int = 3) -> tuple:
    """Run stability test across multiple seeds."""
    results = []

    for seed in range(n_seeds):
        # Train with different seed
        models = []
        for i in range(5):
            model = lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                num_leaves=31,
                random_state=100 * seed + i,
                verbose=-1,
                force_col_wise=True
            )
            model.fit(X, y)
            models.append(model)

        # Get attribution
        attribution = fk_noise_injection(models, X, col_to_fk, n_permutations=3)
        results.append(attribution)

    # Calculate stability (Spearman correlation between runs)
    fks = list(results[0].keys())
    correlations = []

    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            vals_i = [results[i][fk] for fk in fks]
            vals_j = [results[j][fk] for fk in fks]
            rho, _ = stats.spearmanr(vals_i, vals_j)
            correlations.append(rho)

    avg_stability = np.mean(correlations) if correlations else 1.0

    # Average attribution
    avg_attribution = {}
    for fk in fks:
        avg_attribution[fk] = np.mean([r[fk] for r in results])

    return avg_stability, avg_attribution


def main():
    print("=" * 60)
    print("FK-Level Risk Attribution: Amazon Validation")
    print("=" * 60)

    # Load data (smaller sample due to memory)
    print("\n[1/3] Loading rel-amazon data...")
    X, y, feature_cols, col_to_fk = load_amazon_data(
        task_name='user-ltv',
        sample_size=3000,
        use_cache=True
    )

    print(f"\nData shape: {X.shape}")
    print(f"Target mean: {y.mean():.2f}, std: {y.std():.2f}")

    # Print FK summary
    fk_to_cols = defaultdict(list)
    for col, fk in col_to_fk.items():
        fk_to_cols[fk].append(col)

    print("\nFK Groups:")
    for fk, cols in sorted(fk_to_cols.items()):
        print(f"  {fk}: {len(cols)} columns")

    # Run decomposition
    print("\n[2/3] Running FK Decomposition...")
    models = train_ensemble(X, y, n_models=5)
    attribution = fk_noise_injection(models, X, col_to_fk, n_permutations=5)

    print("\nFK Attribution:")
    for fk, pct in sorted(attribution.items(), key=lambda x: -x[1]):
        print(f"  {fk}: {pct:.1f}%")

    # Stability test
    print("\n[3/3] Running Stability Test...")
    stability, avg_attr = run_stability_test(X, y, col_to_fk, n_seeds=3)

    print(f"\nStability (Spearman ρ): {stability:.3f}")

    print("\nAverage Attribution:")
    for fk, pct in sorted(avg_attr.items(), key=lambda x: -x[1]):
        print(f"  {fk}: {pct:.1f}%")

    # Summary
    print("\n" + "=" * 60)
    print("AMAZON VALIDATION SUMMARY")
    print("=" * 60)
    top_fk = max(avg_attr.items(), key=lambda x: x[1])
    print(f"Domain: E-commerce (user LTV prediction)")
    print(f"Stability: {stability:.3f}")
    print(f"Top FK: {top_fk[0]} ({top_fk[1]:.1f}%)")
    print("=" * 60)

    return stability, avg_attr


if __name__ == "__main__":
    main()
