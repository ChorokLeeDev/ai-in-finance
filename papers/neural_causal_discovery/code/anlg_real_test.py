"""
Test ANLG on Real Financial Data
================================
Key question: Does ANLG's adaptive selection help on real data?
"""
import numpy as np
import pandas as pd
import torch
import yfinance as yf
import sys
sys.path.insert(0, '/Users/i767700/Github/ai-in-finance/papers/neural_causal_discovery/code')

from anlg import AdaptiveNeuralLinearGranger
from neural_granger_simple import NeuralGranger, train_neural_granger
from baselines import LinearGrangerCausality


def test_anlg_real_data():
    """Test ANLG on real ETF data."""
    print("=" * 60)
    print("ANLG ON REAL FINANCIAL DATA")
    print("=" * 60)

    # Download data
    tickers = ['SPY', 'QQQ', 'IWM', 'XLF', 'TLT', 'GLD']
    df = yf.download(tickers, start='2015-01-01', end='2024-01-01', progress=False)

    if 'Close' in df.columns.get_level_values(0):
        prices = df['Close']
    else:
        prices = df['Adj Close']

    prices = prices.dropna()
    returns = prices.pct_change().dropna()

    print(f"Data: {len(returns)} days, {returns.shape[1]} assets")

    # Normalize
    data = (returns - returns.mean()) / (returns.std() + 1e-8)
    data = data.values

    # Split train/test
    train_size = int(0.7 * len(data))
    train = data[:train_size]
    test = data[train_size:]

    print(f"Train: {len(train)} days, Test: {len(test)} days")

    # Fit ANLG
    print("\nFitting ANLG...")
    anlg = AdaptiveNeuralLinearGranger(n_lags=5, nonlinearity_threshold=0.3)
    anlg_adj = anlg.fit(train)

    # Get interpretation
    interp = anlg.get_interpretation()
    print(f"\nANLG Method Selection:")
    print(f"  Neural edges: {interp['n_neural_edges']}")
    print(f"  Linear edges: {interp['n_linear_edges']}")
    print(f"  Pct Neural: {interp['pct_neural']:.1f}%")

    # Show nonlinearity scores
    print(f"\nNonlinearity scores (top 5):")
    scores = interp['nonlinearity_scores']
    n = scores.shape[0]
    flat_scores = []
    for i in range(n):
        for j in range(n):
            if i != j:
                flat_scores.append((i, j, scores[i, j]))
    flat_scores.sort(key=lambda x: x[2], reverse=True)
    for i, j, s in flat_scores[:5]:
        print(f"  {tickers[i]} → {tickers[j]}: {s:.3f}")

    # Compare prediction MSE on test set
    print("\n" + "=" * 60)
    print("PREDICTION MSE COMPARISON ON TEST SET")
    print("=" * 60)

    # Simple prediction: use fitted adjacency to weight sources
    def compute_mse(adj, test_data, n_lags=5):
        """Compute prediction MSE using adjacency as weights."""
        n_factors = test_data.shape[1]
        preds = []

        for t in range(n_lags, len(test_data)):
            pred_t = []
            for j in range(n_factors):
                # Weighted prediction from all sources
                pred_j = 0
                total_weight = 0
                for i in range(n_factors):
                    if adj[i, j] > 0.1:
                        # Use lagged value weighted by adjacency
                        pred_j += adj[i, j] * test_data[t-1, i]
                        total_weight += adj[i, j]
                if total_weight > 0:
                    pred_j /= total_weight
                else:
                    pred_j = test_data[t-1, j]  # Fallback to own lag
                pred_t.append(pred_j)
            preds.append(pred_t)

        preds = np.array(preds)
        actuals = test_data[n_lags:]
        return np.mean((preds - actuals) ** 2)

    # ANLG MSE
    anlg_mse = compute_mse(anlg_adj, test, n_lags=5)
    print(f"ANLG MSE: {anlg_mse:.6f}")

    # Linear MSE
    linear_gc = LinearGrangerCausality(n_lags=5)
    linear_adj = linear_gc.fit(train)
    linear_mse = compute_mse(linear_adj, test, n_lags=5)
    print(f"Linear MSE: {linear_mse:.6f}")

    # Neural MSE
    np.random.seed(42)
    torch.manual_seed(42)
    neural_model = NeuralGranger(n_factors=6, n_lags=5, hidden_dim=16)
    neural_model = train_neural_granger(neural_model, train, n_epochs=30, lr=1e-3)
    x = torch.FloatTensor(train).unsqueeze(0)
    neural_model.eval()
    with torch.no_grad():
        neural_adj = neural_model.compute_granger_adjacency(x)
    neural_mse = compute_mse(neural_adj, test, n_lags=5)
    print(f"Neural MSE: {neural_mse:.6f}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    best_mse = min(anlg_mse, linear_mse, neural_mse)
    if anlg_mse == best_mse:
        print("🎯 ANLG achieves best prediction on real data!")
    elif anlg_mse < max(linear_mse, neural_mse):
        print("✅ ANLG competitive with best method")
    else:
        print("⚠️ ANLG not best on this real data")

    return {
        'anlg_mse': anlg_mse,
        'linear_mse': linear_mse,
        'neural_mse': neural_mse,
        'pct_neural': interp['pct_neural']
    }


if __name__ == "__main__":
    results = test_anlg_real_data()
