"""
Simplified RANCD for faster experiments
Uses shared encoder instead of per-factor LSTMs
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimplifiedRANCD(nn.Module):
    """Faster version of RANCD with shared encoder."""

    def __init__(self, n_factors: int, hidden_dim: int = 32,
                 n_regimes: int = 3, n_lags: int = 5):
        super().__init__()
        self.n_factors = n_factors
        self.n_regimes = n_regimes
        self.n_lags = n_lags

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(n_factors, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Regime classifier
        self.regime_head = nn.Linear(hidden_dim, n_regimes)

        # Edge predictor (simplified)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + n_regimes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Predictors
        self.predictors = nn.ModuleList([
            nn.Linear(n_factors * n_lags, 1)
            for _ in range(n_factors)
        ])

    def forward(self, x, return_all=False):
        batch_size, seq_len, n_factors = x.shape

        # Encode
        h = self.encoder(x)  # (batch, seq, hidden)

        # Regime
        regime_logits = self.regime_head(h)
        regime_probs = F.softmax(regime_logits, dim=-1)
        regime_pooled = regime_probs.mean(dim=1)

        # Factor embeddings (use mean hidden state)
        factor_emb = torch.zeros(batch_size, n_factors, h.size(-1), device=x.device)
        for i in range(n_factors):
            factor_emb[:, i] = h.mean(dim=1)  # Simplified

        # Edge prediction
        adj = torch.zeros(batch_size, n_factors, n_factors, device=x.device)
        for i in range(n_factors):
            for j in range(n_factors):
                if i == j:
                    continue
                edge_input = torch.cat([
                    factor_emb[:, i],
                    factor_emb[:, j],
                    regime_pooled
                ], dim=-1)
                adj[:, i, j] = torch.sigmoid(self.edge_mlp(edge_input).squeeze(-1))

        # Predictions
        predictions = []
        for t in range(self.n_lags, seq_len):
            x_lagged = x[:, t-self.n_lags:t, :].reshape(batch_size, -1)
            pred_t = []
            for j in range(n_factors):
                pred_j = self.predictors[j](x_lagged)
                pred_t.append(pred_j)
            pred_t = torch.cat(pred_t, dim=-1)
            predictions.append(pred_t)
        predictions = torch.stack(predictions, dim=1)

        if return_all:
            return predictions, adj, regime_probs, factor_emb
        return predictions

    def compute_loss(self, x):
        pred, adj, regime_probs, _ = self(x, return_all=True)
        target = x[:, self.n_lags:, :]

        # Prediction loss
        pred_loss = F.mse_loss(pred, target)

        # DAG constraint (simplified)
        W = adj.mean(dim=0)
        dag_loss = (W * W).sum()

        # Sparsity
        sparse_loss = adj.abs().mean()

        # Regime smoothness
        regime_diff = (regime_probs[:, 1:] - regime_probs[:, :-1]).abs()
        regime_loss = regime_diff.mean()

        total = pred_loss + 0.01 * dag_loss + 0.01 * sparse_loss + 0.1 * regime_loss

        return total, {
            'total': total.item(),
            'pred': pred_loss.item(),
            'dag': dag_loss.item()
        }


def quick_rancd_test():
    """Quick test with simplified model."""
    import sys
    sys.path.insert(0, '/Users/i767700/Github/ai-in-finance/papers/neural_causal_discovery/code')
    from data_loader import SyntheticCausalData, create_data_loader
    from baselines import LinearGrangerCausality, VARModel, evaluate_causal_discovery

    print("=" * 50)
    print("Quick RANCD Test (Simplified)")
    print("=" * 50)

    np.random.seed(42)
    torch.manual_seed(42)

    # Generate data
    synth = SyntheticCausalData(n_factors=6, regime_lengths=[200, 200], seed=42)
    data, true_adj, _ = synth.generate()
    true_adj_eval = true_adj[0]

    print(f"Data: {data.shape}")
    print(f"True edges: {(true_adj_eval > 0).sum()}")

    # Baselines
    gc = LinearGrangerCausality(n_lags=5)
    gc_m = evaluate_causal_discovery(true_adj_eval, gc.fit(data))
    print(f"Granger F1: {gc_m['f1']:.3f}")

    var = VARModel(n_lags=5)
    var_m = evaluate_causal_discovery(true_adj_eval, var.fit(data), threshold=0.25)
    print(f"VAR F1: {var_m['f1']:.3f}")

    # Simplified RANCD
    print("\nTraining Simplified RANCD...")
    loader = create_data_loader(data, window_size=50, batch_size=16)
    model = SimplifiedRANCD(n_factors=6, hidden_dim=32, n_regimes=2, n_lags=5)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(20):
        model.train()
        for batch in loader:
            optimizer.zero_grad()
            loss, _ = model.compute_loss(batch)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/20")

    # Get graph
    model.eval()
    test_data = torch.FloatTensor(data[:50]).unsqueeze(0)
    with torch.no_grad():
        _, adj, _, _ = model(test_data, return_all=True)
        rancd_adj = adj.cpu().numpy().mean(axis=0)

    print(f"\nRANCD Adjacency:")
    print(np.round(rancd_adj, 2))

    for thresh in [0.3, 0.4, 0.5]:
        m = evaluate_causal_discovery(true_adj_eval, rancd_adj, threshold=thresh)
        print(f"Threshold {thresh}: F1={m['f1']:.3f}")

    return gc_m, var_m


if __name__ == "__main__":
    quick_rancd_test()
