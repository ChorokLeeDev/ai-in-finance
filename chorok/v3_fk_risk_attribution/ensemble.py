"""
Ensemble Model for FK-Level Risk Attribution
=============================================

Train LightGBM ensemble and compute uncertainty.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import List, Tuple, Optional
from sklearn.model_selection import train_test_split

from cache import cache_model, load_model
from data_loader import load_salt_data


def train_ensemble(
    X: pd.DataFrame,
    y: pd.Series,
    n_models: int = 5,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[List[lgb.LGBMClassifier], np.ndarray, np.ndarray]:
    """
    Train LightGBM ensemble.

    Args:
        X: Features
        y: Target
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
        model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=random_state + i,
            verbose=-1
        )
        model.fit(X_train, y_train)
        models.append(model)
        print(f"  Model {i+1}/{n_models} trained")

    return models, X_test.values, y_test.values


def compute_uncertainty(
    models: List[lgb.LGBMClassifier],
    X: np.ndarray,
    method: str = 'entropy'
) -> np.ndarray:
    """
    Compute prediction uncertainty using ensemble.

    Args:
        models: Trained ensemble models
        X: Features (numpy array)
        method: 'entropy' (mean entropy) or 'variance' (prediction variance)

    Returns:
        uncertainty: Per-sample uncertainty
    """
    from scipy.stats import entropy as sp_entropy

    if method == 'entropy':
        # Mean entropy across ensemble
        entropies = []
        for model in models:
            proba = model.predict_proba(X)
            # Compute entropy for each sample
            ent = sp_entropy(proba.T)  # entropy along class axis
            entropies.append(ent)

        # Mean entropy across models
        uncertainty = np.mean(entropies, axis=0)

    elif method == 'disagreement':
        # Prediction disagreement across models
        preds = []
        for model in models:
            pred = model.predict(X)
            preds.append(pred)

        preds = np.stack(preds, axis=0)  # (n_models, n_samples)

        # Disagreement = proportion of models that disagree with mode
        from scipy.stats import mode
        mode_pred = mode(preds, axis=0, keepdims=False).mode
        disagreement = np.mean(preds != mode_pred, axis=0)
        uncertainty = disagreement

    else:  # variance
        all_preds = []
        for model in models:
            proba = model.predict_proba(X)
            confidence = np.max(proba, axis=1)
            all_preds.append(confidence)

        all_preds = np.stack(all_preds, axis=0)
        uncertainty = np.var(all_preds, axis=0)

    return uncertainty


def get_ensemble(
    task_name: str = 'sales-group',
    sample_size: int = 500,
    n_models: int = 5,
    use_cache: bool = True
) -> Tuple[List[lgb.LGBMClassifier], np.ndarray, np.ndarray, pd.DataFrame, List[str], dict]:
    """
    Get or train ensemble model.

    Returns:
        models: Trained ensemble
        X_test: Test features (numpy)
        y_test: Test target (numpy)
        X_full: Full feature DataFrame
        feature_cols: Feature column names
        col_to_fk: Column to FK mapping
    """
    # Load data
    X, y, feature_cols, col_to_fk = load_salt_data(task_name, sample_size, use_cache=use_cache)

    # Try cache
    if use_cache:
        cached = load_model(task_name, sample_size)
        if cached is not None:
            models = cached['models']
            X_test = np.array(cached['X_test'])
            y_test = np.array(cached['y_test'])
            print(f"[ENSEMBLE] Loaded {len(models)} models from cache")
            return models, X_test, y_test, X, feature_cols, col_to_fk

    # Train new ensemble
    print(f"\n[ENSEMBLE] Training {n_models} models...")
    models, X_test, y_test = train_ensemble(X, y, n_models=n_models)

    # Cache
    if use_cache:
        cache_model(task_name, sample_size, {
            'models': models,
            'X_test': X_test.tolist(),
            'y_test': y_test.tolist()
        })

    return models, X_test, y_test, X, feature_cols, col_to_fk


if __name__ == "__main__":
    print("Testing ensemble training...")

    # Train ensemble
    models, X_test, y_test, X_full, feature_cols, col_to_fk = get_ensemble(
        task_name='sales-group',
        sample_size=500,
        n_models=5,
        use_cache=False  # Force retrain for testing
    )

    print(f"\nEnsemble: {len(models)} models")
    print(f"Test set: {X_test.shape}")

    # Compute uncertainty (different methods)
    for method in ['entropy', 'disagreement', 'variance']:
        uncertainty = compute_uncertainty(models, X_test, method=method)
        print(f"\nUncertainty ({method}):")
        print(f"  Mean: {uncertainty.mean():.4f}")
        print(f"  Std:  {uncertainty.std():.4f}")
        print(f"  Min:  {uncertainty.min():.4f}")
        print(f"  Max:  {uncertainty.max():.4f}")

    # Test with cached
    print("\n--- Testing cache ---")
    models2, _, _, _, _, _ = get_ensemble(
        task_name='sales-group',
        sample_size=500,
        use_cache=True
    )
    print(f"Cached ensemble: {len(models2)} models")

    print("\nEnsemble training OK!")
