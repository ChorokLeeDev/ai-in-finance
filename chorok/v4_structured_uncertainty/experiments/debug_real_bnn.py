"""Debug: Is Real BNN learning properly?"""

import sys
import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from methods.bnn_pyro import train_real_bnn
from methods.ensemble_lgbm import train_lgbm_ensemble
from methods.mc_dropout import train_mc_dropout_ensemble

CACHE_DIR = Path('/Users/i767700/Github/ai-in-finance/chorok/v3_fk_risk_attribution/cache')


def load_salt_data():
    cache_file = CACHE_DIR / 'data_salt_PLANT_2000.pkl'
    with open(cache_file, 'rb') as f:
        data = pickle.load(f)
    return np.array(data[0], dtype=np.float32), np.array(data[1], dtype=np.float32).flatten()


def main():
    print("="*60)
    print("DEBUG: Comparing Model Quality (All Three Methods)")
    print("="*60)

    X, y = load_salt_data()

    # Split
    n = len(X)
    np.random.seed(42)
    train_idx = np.random.permutation(n)[:int(n*0.8)]
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

    # MC Dropout
    print("\n--- MC Dropout Ensemble ---")
    mc = train_mc_dropout_ensemble(X_train, y_train, n_networks=5, mc_samples=10, epochs=200)
    mc_pred = mc.get_mean_prediction(X_test)
    mc_unc = mc.get_uncertainty(X_test)
    print(f"MAE: {mean_absolute_error(y_test, mc_pred):.4f}")
    print(f"R2: {r2_score(y_test, mc_pred):.4f}")
    print(f"Mean Uncertainty: {mc_unc.mean():.6f}")

    # Real BNN - different epochs
    for epochs in [500, 1000, 2000]:
        print(f"\n--- Real BNN (Pyro, {epochs} epochs) ---")
        bnn = train_real_bnn(X_train, y_train, hidden_dim=64, epochs=epochs, lr=0.01)
        bnn_pred = bnn.get_mean_prediction(X_test, n_samples=100)
        bnn_unc = bnn.get_uncertainty(X_test, n_samples=100)
        print(f"MAE: {mean_absolute_error(y_test, bnn_pred):.4f}")
        print(f"R2: {r2_score(y_test, bnn_pred):.4f}")
        print(f"Mean Uncertainty: {bnn_unc.mean():.6f}")


if __name__ == "__main__":
    main()
