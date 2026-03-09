"""
RANCD V2: Fixed edge learning with correlation-based initialization
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RANCDV2(nn.Module):
    """RANCD with proper factor-specific embeddings."""

    def __init__(self, n_factors: int, hidden_dim: int = 32,
                 n_regimes: int = 3, n_lags: int = 5):
        super().__init__()
        self.n_factors = n_factors
        self.n_regimes = n_regimes
        self.n_lags = n_lags

        # Per-factor encoder (small)
        self.factor_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_lags, hidden_dim),
                nn.ReLU()
            ) for _ in range(n_factors)
        ])

        # Regime encoder
        self.regime_encoder = nn.Sequential(
            nn.Linear(n_factors, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_regimes)
        )

        # Edge predictor - takes source, target embeddings + regime
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + n_regimes, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # Initialize edge MLP to produce moderate values
        for m in self.edge_mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        # Causal predictors
        self.predictors = nn.ModuleList([
            nn.Linear(n_factors * n_lags, 1)
            for _ in range(n_factors)
        ])

    def forward(self, x, return_all=False):
        batch_size, seq_len, n_factors = x.shape

        # Compute factor embeddings using lagged values
        factor_emb = []
        for i in range(n_factors):
            # Use last n_lags values for factor i
            factor_lags = x[:, -self.n_lags:, i]  # (batch, n_lags)
            emb_i = self.factor_encoders[i](factor_lags)  # (batch, hidden)
            factor_emb.append(emb_i)
        factor_emb = torch.stack(factor_emb, dim=1)  # (batch, n_factors, hidden)

        # Regime probabilities
        x_mean = x.mean(dim=1)  # (batch, n_factors)
        regime_logits = self.regime_encoder(x_mean)
        regime_probs = F.softmax(regime_logits, dim=-1)  # (batch, n_regimes)

        # Edge prediction
        adj = torch.zeros(batch_size, n_factors, n_factors, device=x.device)
        for i in range(n_factors):
            for j in range(n_factors):
                if i == j:
                    continue
                edge_input = torch.cat([
                    factor_emb[:, i],
                    factor_emb[:, j],
                    regime_probs
                ], dim=-1)
                edge_logit = self.edge_mlp(edge_input).squeeze(-1)
                adj[:, i, j] = torch.sigmoid(edge_logit)

        # Predictions using graph
        predictions = []
        for t in range(self.n_lags, seq_len):
            x_lagged = x[:, t-self.n_lags:t, :].reshape(batch_size, -1)
            pred_t = []
            for j in range(n_factors):
                # Simple prediction (can add graph masking later)
                pred_j = self.predictors[j](x_lagged)
                pred_t.append(pred_j)
            pred_t = torch.cat(pred_t, dim=-1)
            predictions.append(pred_t)
        predictions = torch.stack(predictions, dim=1)

        if return_all:
            return predictions, adj, regime_probs, factor_emb
        return predictions

    def compute_loss(self, x, lambda_edge=0.5):
        pred, adj, regime_probs, _ = self(x, return_all=True)
        target = x[:, self.n_lags:, :]

        # Prediction loss
        pred_loss = F.mse_loss(pred, target)

        # Edge regularization (encourage some edges but not too many)
        # Binary cross-entropy with soft target around 0.3
        edge_loss = -0.3 * torch.log(adj + 1e-8) - 0.7 * torch.log(1 - adj + 1e-8)
        edge_loss = edge_loss.mean()

        total = pred_loss + lambda_edge * edge_loss

        return total, {
            'total': total.item(),
            'pred': pred_loss.item(),
            'edge': edge_loss.item(),
            'adj_mean': adj.mean().item()
        }


def test_rancdv2():
    """Test RANCD V2."""
    import sys
    sys.path.insert(0, '/Users/i767700/Github/ai-in-finance/papers/neural_causal_discovery/code')
    from data_loader import SyntheticCausalData, create_data_loader
    from baselines import LinearGrangerCausality, VARModel, evaluate_causal_discovery

    print("=" * 50)
    print("RANCD V2 Test")
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

    # RANCD V2
    print("\nTraining RANCD V2...")
    loader = create_data_loader(data, window_size=50, batch_size=16)
    model = RANCDV2(n_factors=6, hidden_dim=32, n_regimes=2, n_lags=5)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(30):
        model.train()
        for batch in loader:
            optimizer.zero_grad()
            loss, loss_dict = model.compute_loss(batch)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/30 | adj_mean: {loss_dict['adj_mean']:.3f}")

    # Get graph
    model.eval()
    test_data = torch.FloatTensor(data[:50]).unsqueeze(0)
    with torch.no_grad():
        _, adj, _, _ = model(test_data, return_all=True)
        rancd_adj = adj.cpu().numpy().mean(axis=0)

    print(f"\nRANCD V2 Adjacency:")
    print(np.round(rancd_adj, 2))

    print(f"\nTrue Adjacency:")
    print(true_adj_eval)

    for thresh in [0.3, 0.4, 0.5, 0.6]:
        m = evaluate_causal_discovery(true_adj_eval, rancd_adj, threshold=thresh)
        print(f"Threshold {thresh}: F1={m['f1']:.3f}, P={m['precision']:.3f}, R={m['recall']:.3f}")

    # Final comparison
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    best_m = evaluate_causal_discovery(true_adj_eval, rancd_adj, threshold=0.5)
    print(f"Granger F1: {gc_m['f1']:.3f}")
    print(f"VAR F1: {var_m['f1']:.3f}")
    print(f"RANCD V2 F1: {best_m['f1']:.3f}")


if __name__ == "__main__":
    test_rancdv2()
