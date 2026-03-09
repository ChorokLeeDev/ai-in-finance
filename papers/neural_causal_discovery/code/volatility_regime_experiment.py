"""
Volatility Regime Experiment: Testing on High vs Low Volatility Periods
=======================================================================
A cleaner test: Compare neural vs linear during HIGH volatility
vs LOW volatility periods.

Hypothesis: High volatility periods have more nonlinear dynamics
(leverage effect, volatility clustering) where neural should excel.
"""
import numpy as np
import pandas as pd
import torch
import yfinance as yf
import sys
sys.path.insert(0, '/Users/i767700/Github/ai-in-finance/papers/neural_causal_discovery/code')

from neural_granger_simple import NeuralGranger, train_neural_granger
from scipy import stats


def run_volatility_regime_experiment():
    """
    Compare neural vs linear prediction during high/low volatility regimes.
    """
    print("=" * 60)
    print("VOLATILITY REGIME EXPERIMENT")
    print("=" * 60)

    # Download longer history
    tickers = ['SPY', 'QQQ', 'IWM', 'XLF', 'TLT', 'GLD']
    df = yf.download(tickers, start='2010-01-01', end='2024-01-01', progress=False)

    if 'Close' in df.columns.get_level_values(0):
        prices = df['Close']
    else:
        prices = df['Adj Close']

    prices = prices.dropna()
    returns = prices.pct_change().dropna()

    print(f"Data: {len(returns)} days")

    # Compute rolling volatility
    spy_vol = returns['SPY'].rolling(20).std()

    # Define high/low vol thresholds (top/bottom quartile)
    vol_25 = spy_vol.quantile(0.25)
    vol_75 = spy_vol.quantile(0.75)

    low_vol_mask = spy_vol < vol_25
    high_vol_mask = spy_vol > vol_75

    print(f"Low vol days (<{vol_25:.4f}): {low_vol_mask.sum()}")
    print(f"High vol days (>{vol_75:.4f}): {high_vol_mask.sum()}")

    # Normalize data
    returns_norm = (returns - returns.mean()) / (returns.std() + 1e-8)
    data = returns_norm.values
    n_factors = data.shape[1]

    # Run multiple trials
    n_trials = 5
    high_vol_neural = []
    high_vol_linear = []
    low_vol_neural = []
    low_vol_linear = []

    for trial in range(n_trials):
        seed = 42 + trial
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Sample windows from high/low vol periods
        high_vol_idx = np.where(high_vol_mask.values)[0]
        low_vol_idx = np.where(low_vol_mask.values)[0]

        # Use continuous segments
        window_size = 100

        # Find valid high vol windows
        for start_idx in high_vol_idx[::50]:  # Sample every 50 days
            if start_idx + window_size < len(data):
                window = data[start_idx:start_idx+window_size]

                # Split 70/30
                train_size = int(0.7 * len(window))
                train = window[:train_size]
                test = window[train_size:]

                # Linear prediction (simple AR)
                linear_preds = []
                for t in range(5, len(test)):
                    pred = np.mean(test[t-5:t], axis=0)  # Simple moving avg
                    linear_preds.append(pred)
                linear_mse = np.mean((np.array(linear_preds) - test[5:]) ** 2)

                # Neural prediction
                model = NeuralGranger(n_factors=n_factors, n_lags=5, hidden_dim=16)
                model = train_neural_granger(model, train, n_epochs=20, lr=1e-3)
                model.eval()

                with torch.no_grad():
                    x = torch.FloatTensor(test).unsqueeze(0)
                    neural_preds = []
                    for t in range(5, len(test)):
                        pred_t = []
                        for j in range(n_factors):
                            lags = x[0, t-5:t, j]
                            pred_j = model.self_predictors[j](lags).item()
                            pred_t.append(pred_j)
                        neural_preds.append(pred_t)
                    neural_mse = np.mean((np.array(neural_preds) - test[5:]) ** 2)

                high_vol_neural.append(neural_mse)
                high_vol_linear.append(linear_mse)
                break

        # Find valid low vol windows
        for start_idx in low_vol_idx[::50]:
            if start_idx + window_size < len(data):
                window = data[start_idx:start_idx+window_size]

                train_size = int(0.7 * len(window))
                train = window[:train_size]
                test = window[train_size:]

                # Linear prediction
                linear_preds = []
                for t in range(5, len(test)):
                    pred = np.mean(test[t-5:t], axis=0)
                    linear_preds.append(pred)
                linear_mse = np.mean((np.array(linear_preds) - test[5:]) ** 2)

                # Neural prediction
                model = NeuralGranger(n_factors=n_factors, n_lags=5, hidden_dim=16)
                model = train_neural_granger(model, train, n_epochs=20, lr=1e-3)
                model.eval()

                with torch.no_grad():
                    x = torch.FloatTensor(test).unsqueeze(0)
                    neural_preds = []
                    for t in range(5, len(test)):
                        pred_t = []
                        for j in range(n_factors):
                            lags = x[0, t-5:t, j]
                            pred_j = model.self_predictors[j](lags).item()
                            pred_t.append(pred_j)
                        neural_preds.append(pred_t)
                    neural_mse = np.mean((np.array(neural_preds) - test[5:]) ** 2)

                low_vol_neural.append(neural_mse)
                low_vol_linear.append(linear_mse)
                break

        print(f"Trial {trial+1} complete")

    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    if high_vol_neural and low_vol_neural:
        high_neural_mean = np.mean(high_vol_neural)
        high_linear_mean = np.mean(high_vol_linear)
        high_improvement = (high_linear_mean - high_neural_mean) / high_linear_mean * 100

        low_neural_mean = np.mean(low_vol_neural)
        low_linear_mean = np.mean(low_vol_linear)
        low_improvement = (low_linear_mean - low_neural_mean) / low_linear_mean * 100

        print(f"\nHigh Volatility Regime:")
        print(f"  Linear MSE: {high_linear_mean:.4f}")
        print(f"  Neural MSE: {high_neural_mean:.4f}")
        print(f"  Neural improvement: {high_improvement:+.1f}%")

        print(f"\nLow Volatility Regime:")
        print(f"  Linear MSE: {low_linear_mean:.4f}")
        print(f"  Neural MSE: {low_neural_mean:.4f}")
        print(f"  Neural improvement: {low_improvement:+.1f}%")

        diff = high_improvement - low_improvement
        print(f"\nDifference: {diff:+.1f}%")

        if diff > 5:
            print("\n✅ Neural shows LARGER advantage in high-vol regime!")
        elif high_improvement > 0 and high_improvement > low_improvement:
            print("\n⚠️ Neural slightly better in high-vol, but not significantly")
        else:
            print("\n❌ Hypothesis not supported")


if __name__ == "__main__":
    run_volatility_regime_experiment()
