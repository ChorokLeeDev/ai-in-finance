"""
Sophisticated Economic Test: Prediction-Based Trading
=====================================================
Instead of momentum, use actual prediction quality.
"""
import numpy as np
import pandas as pd
import torch
import sys
sys.path.insert(0, '/Users/i767700/Github/ai-in-finance/papers/neural_causal_discovery/code')

from neural_granger_simple import NeuralGranger, train_neural_granger
from baselines import LinearGrangerCausality
from scipy import stats


def fetch_crypto_data():
    """Fetch cryptocurrency data."""
    import yfinance as yf
    tickers = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'SOL-USD', 'ADA-USD']
    data = yf.download(tickers, start='2021-01-01', end='2024-01-01', progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        close_cols = [col for col in data.columns if col[0] == 'Adj Close' or col[0] == 'Close']
        prices = data.loc[:, close_cols]
        prices.columns = [col[1].replace('-USD', '') for col in prices.columns]
    else:
        prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
    return prices.dropna().pct_change().dropna()


def make_predictions(data, adj, lags=5):
    """Make next-day predictions using causal graph."""
    T, n = data.shape
    preds = np.zeros(T)

    for t in range(lags, T-1):
        # For BTC (index 2), predict based on its causes
        btc_idx = 2
        causes = np.where(adj[:, btc_idx] > 0.5)[0]
        if len(causes) > 0:
            # Simple autoregressive prediction using causes
            signal = data[t-lags:t, causes].mean()
            preds[t+1] = np.sign(signal)
        else:
            # Use BTC's own momentum
            preds[t+1] = np.sign(data[t-lags:t, btc_idx].mean())

    return preds


if __name__ == "__main__":
    print("=" * 60)
    print("PREDICTION-BASED TRADING TEST")
    print("=" * 60)

    returns = fetch_crypto_data()
    print(f"Data: {len(returns)} days")

    # Use rolling windows
    window = 250
    n_tests = 30

    neural_mse, linear_mse = [], []
    neural_sharpe, linear_sharpe = [], []

    for i in range(n_tests):
        start = i * 20
        if start + window + 100 > len(returns):
            break

        train_data = returns.iloc[start:start+window].values
        test_data = returns.iloc[start+window:start+window+100]

        np.random.seed(42 + i)
        torch.manual_seed(42 + i)

        # Linear
        gc = LinearGrangerCausality(n_lags=5)
        linear_adj = gc.fit(train_data)

        # Neural
        model = NeuralGranger(n_factors=train_data.shape[1], n_lags=5, hidden_dim=32)
        model = train_neural_granger(model, train_data, n_epochs=15, lr=1e-3)
        x = torch.FloatTensor(train_data).unsqueeze(0)
        model.eval()
        with torch.no_grad():
            neural_adj = (model.compute_granger_adjacency(x) > 0.02).astype(float)

        # Make predictions for BTC
        btc_actual = test_data.iloc[:, 2].values  # BTC column

        linear_preds = make_predictions(test_data.values, linear_adj)
        neural_preds = make_predictions(test_data.values, neural_adj)

        # Direction accuracy
        linear_dir = (np.sign(linear_preds[1:]) == np.sign(btc_actual[:-1])).mean()
        neural_dir = (np.sign(neural_preds[1:]) == np.sign(btc_actual[:-1])).mean()

        # Trading returns
        linear_ret = (linear_preds[1:] * btc_actual[:-1])
        neural_ret = (neural_preds[1:] * btc_actual[:-1])

        linear_sharpe.append(np.sqrt(252) * linear_ret.mean() / (linear_ret.std() + 1e-8))
        neural_sharpe.append(np.sqrt(252) * neural_ret.mean() / (neural_ret.std() + 1e-8))

        print(f"Window {i+1}: Neural dir={neural_dir:.2f}, Linear dir={linear_dir:.2f}")

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    neural_sharpe = np.array(neural_sharpe)
    linear_sharpe = np.array(linear_sharpe)

    print(f"Neural Sharpe: {neural_sharpe.mean():.3f} ± {neural_sharpe.std():.3f}")
    print(f"Linear Sharpe: {linear_sharpe.mean():.3f} ± {linear_sharpe.std():.3f}")

    t, p = stats.ttest_rel(neural_sharpe, linear_sharpe)
    print(f"Paired t-test: t={t:.3f}, p={p:.4f}")
    print(f"Neural wins: {(neural_sharpe > linear_sharpe).sum()}/{len(neural_sharpe)}")
