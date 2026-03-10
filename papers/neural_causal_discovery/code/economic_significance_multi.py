"""
Multiple-run Economic Significance Test
========================================
Run trading simulation 50 times with different random seeds.
"""
import numpy as np
import pandas as pd
import torch
import sys
sys.path.insert(0, '/Users/i767700/Github/ai-in-finance/papers/neural_causal_discovery/code')

from neural_granger_simple import NeuralGranger, train_neural_granger
from baselines import LinearGrangerCausality


def fetch_crypto_data():
    """Fetch cryptocurrency data."""
    import yfinance as yf
    tickers = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'SOL-USD', 'ADA-USD']
    data = yf.download(tickers, start='2022-01-01', end='2024-01-01', progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        close_cols = [col for col in data.columns if col[0] == 'Adj Close' or col[0] == 'Close']
        prices = data.loc[:, close_cols]
        prices.columns = [col[1].replace('-USD', '') for col in prices.columns]
    else:
        prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
    return prices.dropna().pct_change().dropna()


def causal_momentum_strategy(returns, adj_matrix, lookback=5):
    positions = np.zeros((len(returns), returns.shape[1]))
    for t in range(lookback, len(returns)):
        for j in range(returns.shape[1]):
            causes = np.where(adj_matrix[:, j] > 0.5)[0]
            if len(causes) > 0:
                signal = returns.iloc[t-lookback:t, causes].mean().mean()
                positions[t, j] = np.sign(signal)
    return positions


def evaluate_strategy(returns, positions):
    strategy_returns = (positions[1:] * returns.values[1:]).sum(axis=1)
    n_positions = np.maximum(np.abs(positions[1:]).sum(axis=1), 1)
    strategy_returns = strategy_returns / n_positions
    return np.sqrt(252) * strategy_returns.mean() / (strategy_returns.std() + 1e-8)


if __name__ == "__main__":
    print("=" * 60)
    print("ECONOMIC SIGNIFICANCE TEST (Multiple Runs)")
    print("=" * 60)

    returns = fetch_crypto_data()
    print(f"Data: {len(returns)} days, {returns.shape[1]} assets")

    train_end = int(len(returns) * 0.5)

    neural_sharpes, linear_sharpes = [], []
    n_runs = 20

    for run in range(n_runs):
        # Use different train periods (rolling window)
        start = run * 10
        train_data = returns.iloc[start:start+train_end].values
        test_returns = returns.iloc[start+train_end:start+train_end+150]

        if len(test_returns) < 50:
            continue

        seed = 42 + run
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Linear
        gc = LinearGrangerCausality(n_lags=5)
        linear_adj = gc.fit(train_data)
        linear_pos = causal_momentum_strategy(test_returns, linear_adj)
        linear_sharpes.append(evaluate_strategy(test_returns, linear_pos))

        # Neural
        model = NeuralGranger(n_factors=train_data.shape[1], n_lags=5, hidden_dim=32)
        model = train_neural_granger(model, train_data, n_epochs=20, lr=1e-3)
        x = torch.FloatTensor(train_data).unsqueeze(0)
        model.eval()
        with torch.no_grad():
            neural_adj = (model.compute_granger_adjacency(x) > 0.02).astype(float)
        neural_pos = causal_momentum_strategy(test_returns, neural_adj)
        neural_sharpes.append(evaluate_strategy(test_returns, neural_pos))

        print(f"Run {run+1}: Neural Sharpe={neural_sharpes[-1]:.3f}, Linear Sharpe={linear_sharpes[-1]:.3f}")

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    neural_sharpes = np.array(neural_sharpes)
    linear_sharpes = np.array(linear_sharpes)

    print(f"Neural Sharpe: {neural_sharpes.mean():.3f} ± {neural_sharpes.std():.3f}")
    print(f"Linear Sharpe: {linear_sharpes.mean():.3f} ± {linear_sharpes.std():.3f}")

    # Paired t-test
    from scipy import stats
    t, p = stats.ttest_rel(neural_sharpes, linear_sharpes)
    print(f"\nPaired t-test: t={t:.3f}, p={p:.4f}")

    # Win rate
    wins = (neural_sharpes > linear_sharpes).sum()
    print(f"Neural wins: {wins}/{len(neural_sharpes)} ({100*wins/len(neural_sharpes):.1f}%)")
