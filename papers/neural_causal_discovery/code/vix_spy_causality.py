"""
VIX-SPY Relationship: A Real Nonlinear Financial Relationship
=============================================================
The VIX-SPY relationship has KNOWN nonlinearities:
1. Leverage effect: negative returns increase VIX more than positive
2. Volatility convexity: VIX response is convex in absolute returns

This is documented in academic literature (Christie 1982, Black 1976).

Test: Does neural Granger better capture the VIX-SPY causal relationship?
"""
import numpy as np
import pandas as pd
import torch
import yfinance as yf
import sys
sys.path.insert(0, '/Users/i767700/Github/ai-in-finance/papers/neural_causal_discovery/code')

from neural_granger_simple import NeuralGranger, train_neural_granger
from scipy import stats


def test_vix_spy_causality():
    """
    Test if neural methods better detect SPY → VIX causality
    which has known nonlinear characteristics.
    """
    print("=" * 60)
    print("VIX-SPY NONLINEAR CAUSALITY TEST")
    print("=" * 60)
    print("Testing known nonlinear relationship: SPY returns → VIX")
    print("The leverage effect is well-documented in finance literature.")
    print()

    # Download SPY and VIX
    tickers = ['SPY', '^VIX']
    df = yf.download(tickers, start='2010-01-01', end='2024-01-01', progress=False)

    # Get closes
    if 'Close' in df.columns.get_level_values(0):
        spy_close = df['Close']['SPY']
        vix_close = df['Close']['^VIX']
    else:
        spy_close = df.iloc[:, 0]
        vix_close = df.iloc[:, 1]

    # Combine
    data = pd.DataFrame({'SPY': spy_close, 'VIX': vix_close}).dropna()

    # Compute returns for SPY, levels for VIX
    spy_ret = data['SPY'].pct_change()
    vix_change = data['VIX'].pct_change()

    # Combine
    data = pd.DataFrame({
        'SPY_ret': spy_ret,
        'VIX_change': vix_change
    }).dropna()

    print(f"Data: {len(data)} days")

    # Normalize
    data_norm = (data - data.mean()) / (data.std() + 1e-8)
    X = data_norm.values

    # Multiple trials with train/test splits
    n_trials = 10
    neural_r2 = []
    linear_r2 = []

    for trial in range(n_trials):
        seed = 42 + trial
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Random split
        n = len(X)
        train_size = int(0.7 * n)
        idx = np.random.permutation(n)
        train_idx = idx[:train_size]
        test_idx = idx[train_size:]

        train = X[np.sort(train_idx)]  # Keep temporal order
        test = X[np.sort(test_idx)]

        n_lags = 5

        # Linear: Predict VIX_change from lagged SPY_ret
        # Simple linear regression
        X_lin = []
        y_lin = []
        for t in range(n_lags, len(test)):
            X_lin.append(test[t-n_lags:t, 0])  # SPY_ret lags
            y_lin.append(test[t, 1])  # VIX_change

        X_lin = np.array(X_lin)
        y_lin = np.array(y_lin)

        # OLS
        from numpy.linalg import lstsq
        X_lin_bias = np.c_[X_lin, np.ones(len(X_lin))]
        coef, _, _, _ = lstsq(X_lin_bias, y_lin, rcond=None)
        y_pred_lin = X_lin_bias @ coef

        ss_res_lin = np.sum((y_lin - y_pred_lin) ** 2)
        ss_tot = np.sum((y_lin - y_lin.mean()) ** 2)
        r2_lin = 1 - ss_res_lin / ss_tot
        linear_r2.append(max(0, r2_lin))

        # Neural: Train neural Granger
        model = NeuralGranger(n_factors=2, n_lags=n_lags, hidden_dim=16)
        model = train_neural_granger(model, train, n_epochs=30, lr=1e-3)
        model.eval()

        # Predict VIX_change using cross-predictor (SPY → VIX)
        y_pred_neural = []
        with torch.no_grad():
            for t in range(n_lags, len(test)):
                lags = torch.FloatTensor(test[t-n_lags:t, 0])  # SPY lags
                # Use cross predictor [1][0] = SPY→VIX
                pred = model.cross_predictors[1][0](lags).item()
                y_pred_neural.append(pred)

        y_pred_neural = np.array(y_pred_neural)

        ss_res_neural = np.sum((y_lin - y_pred_neural) ** 2)
        r2_neural = 1 - ss_res_neural / ss_tot
        neural_r2.append(max(0, r2_neural))

    # Results
    print("\n" + "=" * 60)
    print("RESULTS: SPY → VIX Prediction (10 trials)")
    print("=" * 60)

    print(f"\nLinear R²: {np.mean(linear_r2):.4f} ± {np.std(linear_r2):.4f}")
    print(f"Neural R²: {np.mean(neural_r2):.4f} ± {np.std(neural_r2):.4f}")

    # Paired t-test
    t_stat, p_val = stats.ttest_rel(neural_r2, linear_r2)
    print(f"\nPaired t-test: t={t_stat:.3f}, p={p_val:.4f}")

    improvement = (np.mean(neural_r2) - np.mean(linear_r2)) / max(np.mean(linear_r2), 0.001) * 100
    print(f"Neural improvement: {improvement:+.1f}%")

    if p_val < 0.05 and np.mean(neural_r2) > np.mean(linear_r2):
        print("\n✅ Neural SIGNIFICANTLY better at capturing VIX-SPY relationship!")
        print("   This validates neural methods on REAL nonlinear financial data!")
    elif np.mean(neural_r2) > np.mean(linear_r2):
        print("\n⚠️ Neural better but not statistically significant")
    else:
        print("\n❌ Neural not better on VIX-SPY relationship")

    return {
        'neural_r2': neural_r2,
        'linear_r2': linear_r2,
        't_stat': t_stat,
        'p_val': p_val
    }


if __name__ == "__main__":
    results = test_vix_spy_causality()
