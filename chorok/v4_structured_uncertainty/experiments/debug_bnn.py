"""
Debug: Is BNN actually learning anything?
"""

import sys
import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from methods.bnn import train_bnn_ensemble
from methods.ensemble_lgbm import train_lgbm_ensemble

CACHE_DIR = Path('/Users/i767700/Github/ai-in-finance/chorok/v3_fk_risk_attribution/cache')


def load_salt_data():
    cache_file = CACHE_DIR / 'data_salt_PLANT_2000.pkl'
    with open(cache_file, 'rb') as f:
        data = pickle.load(f)
    return data[0], data[1]


def main():
    print("="*60)
    print("DEBUG: Comparing Model Quality")
    print("="*60)

    X, y = load_salt_data()
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32).flatten()

    # Split
    n = len(X)
    train_idx = np.random.RandomState(42).permutation(n)[:int(n*0.8)]
    test_idx = np.array([i for i in range(n) if i not in train_idx])

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")
    print(f"y range: [{y.min():.2f}, {y.max():.2f}], mean={y.mean():.2f}, std={y.std():.2f}")

    # LightGBM
    print("\n--- LightGBM Ensemble ---")
    lgbm = train_lgbm_ensemble(X_train, y_train, n_models=5)
    lgbm_pred = lgbm.get_mean_prediction(X_test)
    lgbm_unc = lgbm.get_uncertainty(X_test)

    print(f"MAE: {mean_absolute_error(y_test, lgbm_pred):.4f}")
    print(f"R2: {r2_score(y_test, lgbm_pred):.4f}")
    print(f"Mean Uncertainty: {lgbm_unc.mean():.6f}")
    print(f"Pred range: [{lgbm_pred.min():.2f}, {lgbm_pred.max():.2f}]")

    # BNN
    print("\n--- BNN Ensemble ---")
    bnn = train_bnn_ensemble(X_train, y_train, n_networks=5, mc_samples=10, epochs=100)
    bnn_pred = bnn.get_mean_prediction(X_test)
    bnn_unc = bnn.get_uncertainty(X_test)

    print(f"MAE: {mean_absolute_error(y_test, bnn_pred):.4f}")
    print(f"R2: {r2_score(y_test, bnn_pred):.4f}")
    print(f"Mean Uncertainty: {bnn_unc.mean():.6f}")
    print(f"Pred range: [{bnn_pred.min():.2f}, {bnn_pred.max():.2f}]")

    # BNN with more epochs
    print("\n--- BNN Ensemble (300 epochs) ---")
    bnn2 = train_bnn_ensemble(X_train, y_train, n_networks=5, mc_samples=10, epochs=300)
    bnn2_pred = bnn2.get_mean_prediction(X_test)
    bnn2_unc = bnn2.get_uncertainty(X_test)

    print(f"MAE: {mean_absolute_error(y_test, bnn2_pred):.4f}")
    print(f"R2: {r2_score(y_test, bnn2_pred):.4f}")
    print(f"Mean Uncertainty: {bnn2_unc.mean():.6f}")
    print(f"Pred range: [{bnn2_pred.min():.2f}, {bnn2_pred.max():.2f}]")


if __name__ == "__main__":
    main()
