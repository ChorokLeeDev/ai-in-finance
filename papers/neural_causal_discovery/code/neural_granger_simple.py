"""
Quick Neural Granger Test
========================
Implement component-wise neural Granger (Tank et al. style) directly.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralGranger(nn.Module):
    """
    Simple neural Granger: separate MLP per (source, target) pair.
    Edge weight = how much source improves target prediction.
    """

    def __init__(self, n_factors: int, n_lags: int = 5, hidden_dim: int = 16):
        super().__init__()
        self.n_factors = n_factors
        self.n_lags = n_lags

        # Per-factor MLP (predicts from own lags)
        self.self_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_lags, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            ) for _ in range(n_factors)
        ])

        # Cross-factor MLP (predicts from other factor's lags)
        self.cross_predictors = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(n_lags, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1)
                ) if i != j else None
                for i in range(n_factors)
            ]) for j in range(n_factors)
        ])

    def compute_granger_adjacency(self, x):
        """
        Compute Granger-style adjacency based on prediction improvement.
        adj[i,j] = how much i helps predict j beyond j's own lags.
        """
        batch_size, seq_len, n_factors = x.shape

        adj = np.zeros((n_factors, n_factors))

        for j in range(n_factors):
            target = x[:, self.n_lags:, j]  # (batch, T-n_lags)

            # Baseline: predict j from j's own lags
            baseline_pred = []
            for t in range(self.n_lags, seq_len):
                x_j_lagged = x[:, t-self.n_lags:t, j]
                p = self.self_predictors[j](x_j_lagged)
                baseline_pred.append(p)
            baseline_pred = torch.cat(baseline_pred, dim=1)
            baseline_error = F.mse_loss(baseline_pred, target).item()

            for i in range(n_factors):
                if i == j:
                    continue

                # Predict j from i's lags
                cross_pred = []
                for t in range(self.n_lags, seq_len):
                    x_i_lagged = x[:, t-self.n_lags:t, i]
                    p = self.cross_predictors[j][i](x_i_lagged)
                    cross_pred.append(p)
                cross_pred = torch.cat(cross_pred, dim=1)
                cross_error = F.mse_loss(cross_pred, target).item()

                # Granger score: improvement over baseline
                # Higher = i helps predict j more
                improvement = max(0, baseline_error - cross_error)
                adj[i, j] = improvement / (baseline_error + 1e-8)

        return adj


def train_neural_granger(model, data, n_epochs=30, lr=1e-3):
    """Train neural Granger model."""
    x = torch.FloatTensor(data).unsqueeze(0)  # (1, T, n_factors)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    seq_len = x.shape[1]
    n_factors = x.shape[2]
    n_lags = model.n_lags

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0

        for j in range(n_factors):
            target = x[:, n_lags:, j]

            # Train self predictor
            for t in range(n_lags, seq_len):
                x_j_lagged = x[:, t-n_lags:t, j]
                pred = model.self_predictors[j](x_j_lagged)
                loss = F.mse_loss(pred.squeeze(), target[:, t-n_lags].squeeze())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Train cross predictors
            for i in range(n_factors):
                if i == j:
                    continue
                for t in range(n_lags, seq_len):
                    x_i_lagged = x[:, t-n_lags:t, i]
                    pred = model.cross_predictors[j][i](x_i_lagged)
                    loss = F.mse_loss(pred.squeeze(), target[:, t-n_lags].squeeze())

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}")

    return model


def test_neural_granger():
    """Test neural Granger approach."""
    import sys
    sys.path.insert(0, '/Users/i767700/Github/ai-in-finance/papers/neural_causal_discovery/code')
    from nonlinear_experiments import generate_nonlinear_causal_data
    from baselines import LinearGrangerCausality, VARModel, evaluate_causal_discovery

    print("=" * 60)
    print("Neural Granger Test (Tank et al. style)")
    print("=" * 60)

    np.random.seed(42)
    torch.manual_seed(42)

    # Generate nonlinear data
    data, true_adj = generate_nonlinear_causal_data(n_factors=6, T=400, seed=42)
    print(f"Data: {data.shape}, True edges: {(true_adj > 0).sum()}")
    print(f"True adjacency:\n{true_adj}")

    # Linear Granger baseline
    gc = LinearGrangerCausality(n_lags=5)
    gc_adj = gc.fit(data)
    gc_m = evaluate_causal_discovery(true_adj, gc_adj)
    print(f"\nLinear Granger F1: {gc_m['f1']:.3f}")

    # VAR baseline
    var = VARModel(n_lags=5)
    var_adj = var.fit(data)
    var_m = evaluate_causal_discovery(true_adj, var_adj, threshold=0.25)
    print(f"VAR F1: {var_m['f1']:.3f}")

    # Neural Granger
    print("\nTraining Neural Granger...")
    model = NeuralGranger(n_factors=6, n_lags=5, hidden_dim=16)
    model = train_neural_granger(model, data, n_epochs=20, lr=1e-3)

    # Compute Granger adjacency
    x = torch.FloatTensor(data).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        neural_adj = model.compute_granger_adjacency(x)

    print(f"\nNeural Granger adjacency:\n{np.round(neural_adj, 3)}")

    # Evaluate at different thresholds
    print("\nNeural Granger results:")
    for thresh in [0.05, 0.1, 0.15, 0.2]:
        m = evaluate_causal_discovery(true_adj, neural_adj, threshold=thresh)
        print(f"  Threshold {thresh}: F1={m['f1']:.3f}, P={m['precision']:.3f}, R={m['recall']:.3f}")


if __name__ == "__main__":
    test_neural_granger()
