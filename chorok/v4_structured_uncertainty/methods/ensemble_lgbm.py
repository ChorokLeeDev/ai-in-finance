"""
LightGBM Ensemble for Uncertainty Quantification
=================================================

Baseline UQ method using ensemble of LightGBM models.
Epistemic uncertainty = variance across ensemble predictions.

Author: ChorokLeeDev
Created: 2025-12-09
"""

import numpy as np
import lightgbm as lgb


class LGBMEnsemble:
    """LightGBM Ensemble for uncertainty estimation."""

    def __init__(self, n_models=5, base_seed=42):
        self.n_models = n_models
        self.base_seed = base_seed
        self.models = []

    def fit(self, X, y, verbose=False):
        """Train ensemble of LightGBM models."""
        self.models = []

        for i in range(self.n_models):
            model = lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.base_seed + i,
                verbose=-1
            )
            model.fit(X, y)
            self.models.append(model)

            if verbose:
                print(f"  Trained model {i+1}/{self.n_models}")

    def predict(self, X):
        """Get predictions from all models."""
        preds = np.array([m.predict(X) for m in self.models])
        return preds  # Shape: (n_models, n_samples)

    def get_uncertainty(self, X):
        """Get epistemic uncertainty (variance of predictions)."""
        preds = self.predict(X)
        return preds.var(axis=0)

    def get_mean_prediction(self, X):
        """Get mean prediction."""
        preds = self.predict(X)
        return preds.mean(axis=0)


def train_lgbm_ensemble(X, y, n_models=5, seed=42):
    """Convenience function to train LightGBM ensemble."""
    ensemble = LGBMEnsemble(n_models=n_models, base_seed=seed)
    ensemble.fit(X, y)
    return ensemble


if __name__ == "__main__":
    print("Testing LightGBM Ensemble...")

    np.random.seed(42)
    X = np.random.randn(1000, 20)
    y = X[:, 0] * 2 + X[:, 1] * 0.5 + np.random.randn(1000) * 0.1

    ensemble = train_lgbm_ensemble(X, y, n_models=5)

    uncertainty = ensemble.get_uncertainty(X[:100])
    print(f"Mean uncertainty: {uncertainty.mean():.4f}")
    print(f"Uncertainty std: {uncertainty.std():.4f}")
    print("LightGBM Ensemble test passed!")
