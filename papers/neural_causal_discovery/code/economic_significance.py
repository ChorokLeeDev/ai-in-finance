"""
Economic Significance Test
==========================
Test if causal edges discovered by neural methods lead to better trading signals.
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
        close_cols = [col for col in data.columns if col[0] == 'Adj Close']
        if close_cols:
            prices = data.loc[:, close_cols]
            prices.columns = [col[1].replace('-USD', '') for col in prices.columns]
        else:
            close_cols = [col for col in data.columns if col[0] == 'Close']
            prices = data.loc[:, close_cols]
            prices.columns = [col[1].replace('-USD', '') for col in prices.columns]
    else:
        prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']

    prices = prices.dropna()
    returns = prices.pct_change().dropna()

    return returns


def causal_momentum_strategy(returns, adj_matrix, lookback=5):
    """
    Trading strategy based on causal edges:
    - If A causes B, and A goes up, predict B goes up next
    - Long B if predicted up, short B if predicted down
    """
    n_assets = returns.shape[1]
    positions = np.zeros((len(returns), n_assets))

    for t in range(lookback, len(returns)):
        for j in range(n_assets):
            # Find all causes of asset j
            causes = np.where(adj_matrix[:, j] > 0.5)[0]

            if len(causes) > 0:
                # Average recent returns of causes
                signal = returns.iloc[t-lookback:t, causes].mean().mean()
                positions[t, j] = np.sign(signal)

    return positions


def evaluate_strategy(returns, positions):
    """Evaluate trading strategy."""
    # Portfolio returns (equal weight across positions)
    strategy_returns = (positions[1:] * returns.values[1:]).sum(axis=1)

    # Normalize by number of positions
    n_positions = np.abs(positions[1:]).sum(axis=1)
    n_positions = np.maximum(n_positions, 1)  # Avoid division by zero
    strategy_returns = strategy_returns / n_positions

    # Metrics
    total_return = (1 + strategy_returns).prod() - 1
    sharpe = np.sqrt(252) * strategy_returns.mean() / (strategy_returns.std() + 1e-8)

    return {
        'total_return': total_return,
        'sharpe': sharpe,
        'mean_daily': strategy_returns.mean(),
        'volatility': strategy_returns.std()
    }


def run_economic_test():
    """Compare trading performance using neural vs linear causal edges."""
    print("=" * 60)
    print("ECONOMIC SIGNIFICANCE TEST")
    print("=" * 60)

    # Fetch data
    returns = fetch_crypto_data()
    print(f"Data: {len(returns)} days, {returns.shape[1]} assets")
    print(f"Assets: {list(returns.columns)}")

    # Split data
    train_end = int(len(returns) * 0.6)
    train_data = returns.iloc[:train_end].values
    test_returns = returns.iloc[train_end:]

    print(f"\nTrain: {train_end} days, Test: {len(test_returns)} days")

    n_factors = train_data.shape[1]

    # Linear Granger
    print("\nFitting Linear Granger...")
    gc = LinearGrangerCausality(n_lags=5)
    linear_adj = gc.fit(train_data)

    # Neural Granger
    print("Fitting Neural Granger...")
    np.random.seed(42)
    torch.manual_seed(42)
    model = NeuralGranger(n_factors=n_factors, n_lags=5, hidden_dim=32)
    model = train_neural_granger(model, train_data, n_epochs=30, lr=1e-3)

    x = torch.FloatTensor(train_data).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        neural_adj = model.compute_granger_adjacency(x)

    # Threshold neural adjacency - use lower threshold
    neural_adj_binary = (neural_adj > 0.01).astype(float)  # Lower threshold

    print(f"\nLinear edges: {int(linear_adj.sum())}")
    print(f"Neural edges: {int(neural_adj_binary.sum())}")

    # Run strategies on test data
    linear_positions = causal_momentum_strategy(test_returns, linear_adj)
    neural_positions = causal_momentum_strategy(test_returns, neural_adj_binary)

    # Random baseline (buy and hold equally)
    random_positions = np.ones((len(test_returns), n_factors))

    # Evaluate
    linear_metrics = evaluate_strategy(test_returns, linear_positions)
    neural_metrics = evaluate_strategy(test_returns, neural_positions)
    random_metrics = evaluate_strategy(test_returns, random_positions)

    print("\n" + "=" * 60)
    print("OUT-OF-SAMPLE TRADING RESULTS")
    print("=" * 60)
    print(f"{'Strategy':<20} {'Return':<12} {'Sharpe':<12}")
    print("-" * 60)
    print(f"{'Buy & Hold':<20} {random_metrics['total_return']*100:>8.2f}%    {random_metrics['sharpe']:.3f}")
    print(f"{'Linear Causal':<20} {linear_metrics['total_return']*100:>8.2f}%    {linear_metrics['sharpe']:.3f}")
    print(f"{'Neural Causal':<20} {neural_metrics['total_return']*100:>8.2f}%    {neural_metrics['sharpe']:.3f}")

    # Statistical test (bootstrap)
    print("\n" + "=" * 60)
    print("BOOTSTRAP TEST (Neural vs Linear, 1000 resamples)")
    print("=" * 60)

    linear_returns = (linear_positions[1:] * test_returns.values[1:]).sum(axis=1)
    neural_returns = (neural_positions[1:] * test_returns.values[1:]).sum(axis=1)

    diff = neural_returns - linear_returns

    n_bootstrap = 1000
    bootstrap_diffs = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(diff), len(diff), replace=True)
        bootstrap_diffs.append(diff[idx].mean())

    ci_low = np.percentile(bootstrap_diffs, 2.5)
    ci_high = np.percentile(bootstrap_diffs, 97.5)
    mean_diff = np.mean(bootstrap_diffs)

    # One-sided test: is neural better than linear?
    p_value = (np.array(bootstrap_diffs) < 0).mean()

    print(f"Mean return difference: {mean_diff*100:.4f}% daily")
    print(f"95% CI: [{ci_low*100:.4f}%, {ci_high*100:.4f}%]")
    print(f"P-value (neural > linear): {1-p_value:.4f}")

    return {
        'linear': linear_metrics,
        'neural': neural_metrics,
        'random': random_metrics,
        'p_value': 1 - p_value
    }


if __name__ == "__main__":
    results = run_economic_test()
