"""
Regression Ensemble for FK-Level Risk Attribution (rel-f1)
==========================================================

Train LightGBM regression ensemble and compute variance-based uncertainty.

Theoretical Justification:
- Lakshminarayanan et al. 2017: Ensemble variance captures epistemic uncertainty
- For regression, variance across ensemble predictions directly measures model disagreement
- Unlike classification entropy (which can be 0 for overconfident models),
  regression variance is naturally non-zero when models are trained with different seeds
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import List, Tuple, Optional
from sklearn.model_selection import train_test_split

from cache import cache_model, load_model
from data_loader_f1 import load_f1_data


def train_regression_ensemble(
    X: pd.DataFrame,
    y: pd.Series,
    n_models: int = 5,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[List[lgb.LGBMRegressor], np.ndarray, np.ndarray]:
    """
    Train LightGBM regression ensemble.

    Args:
        X: Features
        y: Target (continuous)
        n_models: Number of models in ensemble
        test_size: Test set size
        random_state: Random seed

    Returns:
        models: List of trained models
        X_test: Test features
        y_test: Test target
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    models = []
    for i in range(n_models):
        model = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=random_state + i,  # Different seed for each model
            # Add randomization for diversity (Lakshminarayanan et al. 2017)
            subsample=0.8,
            colsample_bytree=0.8,
            subsample_freq=1,
            verbose=-1
        )
        model.fit(X_train, y_train)
        models.append(model)
        print(f"  Model {i+1}/{n_models} trained")

    return models, X_test.values, y_test.values


def compute_regression_uncertainty(
    models: List[lgb.LGBMRegressor],
    X: np.ndarray
) -> np.ndarray:
    """
    Compute prediction uncertainty using ensemble variance.

    This is the theoretically justified approach for regression:
    - Each model predicts a single value
    - Variance across models captures epistemic uncertainty
    - Non-zero by construction (different random seeds → different predictions)

    Reference: Lakshminarayanan et al. 2017

    Args:
        models: Trained regression ensemble
        X: Features (numpy array)

    Returns:
        uncertainty: Per-sample variance across ensemble predictions
    """
    # Get predictions from each model
    all_preds = []
    for model in models:
        pred = model.predict(X)
        all_preds.append(pred)

    all_preds = np.stack(all_preds, axis=0)  # (n_models, n_samples)

    # Variance across models = epistemic uncertainty
    uncertainty = np.var(all_preds, axis=0)

    return uncertainty


def compute_regression_prediction(
    models: List[lgb.LGBMRegressor],
    X: np.ndarray
) -> np.ndarray:
    """
    Get ensemble mean prediction.

    Args:
        models: Trained ensemble
        X: Features

    Returns:
        mean_pred: Mean prediction across ensemble
    """
    all_preds = []
    for model in models:
        pred = model.predict(X)
        all_preds.append(pred)

    all_preds = np.stack(all_preds, axis=0)
    return np.mean(all_preds, axis=0)


def get_regression_ensemble(
    task_name: str = 'driver-position',
    sample_size: int = 3000,
    n_models: int = 5,
    use_cache: bool = True
) -> Tuple[List[lgb.LGBMRegressor], np.ndarray, np.ndarray, pd.DataFrame, List[str], dict]:
    """
    Get or train regression ensemble model.

    Returns:
        models: Trained ensemble
        X_test: Test features (numpy)
        y_test: Test target (numpy)
        X_full: Full feature DataFrame
        feature_cols: Feature column names
        col_to_fk: Column to FK mapping
    """
    cache_key = f"f1_{task_name}"

    # Load data
    X, y, feature_cols, col_to_fk = load_f1_data(task_name, sample_size, use_cache=use_cache)

    # Try cache
    if use_cache:
        cached = load_model(cache_key, sample_size)
        if cached is not None:
            models = cached['models']
            X_test = np.array(cached['X_test'])
            y_test = np.array(cached['y_test'])
            print(f"[ENSEMBLE-REG] Loaded {len(models)} models from cache")
            return models, X_test, y_test, X, feature_cols, col_to_fk

    # Train new ensemble
    print(f"\n[ENSEMBLE-REG] Training {n_models} regression models...")
    models, X_test, y_test = train_regression_ensemble(X, y, n_models=n_models)

    # Cache
    if use_cache:
        cache_model(cache_key, sample_size, {
            'models': models,
            'X_test': X_test.tolist(),
            'y_test': y_test.tolist()
        })

    return models, X_test, y_test, X, feature_cols, col_to_fk


if __name__ == "__main__":
    print("Testing regression ensemble training...")

    # Train ensemble
    models, X_test, y_test, X_full, feature_cols, col_to_fk = get_regression_ensemble(
        task_name='driver-position',
        sample_size=3000,
        n_models=5,
        use_cache=False  # Force retrain for testing
    )

    print(f"\nEnsemble: {len(models)} models")
    print(f"Test set: {X_test.shape}")

    # Compute uncertainty
    uncertainty = compute_regression_uncertainty(models, X_test)
    print(f"\nVariance-based Uncertainty:")
    print(f"  Mean: {uncertainty.mean():.4f}")
    print(f"  Std:  {uncertainty.std():.4f}")
    print(f"  Min:  {uncertainty.min():.4f}")
    print(f"  Max:  {uncertainty.max():.4f}")
    print(f"  Non-zero: {(uncertainty > 0).sum()}/{len(uncertainty)} ({100*(uncertainty > 0).mean():.1f}%)")

    # Compute predictions
    predictions = compute_regression_prediction(models, X_test)
    mae = np.mean(np.abs(predictions - y_test))
    rmse = np.sqrt(np.mean((predictions - y_test)**2))
    print(f"\nPrediction Quality:")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  True range: {y_test.min():.1f} - {y_test.max():.1f}")

    # Test cache
    print("\n--- Testing cache ---")
    models2, _, _, _, _, _ = get_regression_ensemble(
        task_name='driver-position',
        sample_size=3000,
        use_cache=True
    )
    print(f"Cached ensemble: {len(models2)} models")

    print("\nRegression ensemble OK!")
