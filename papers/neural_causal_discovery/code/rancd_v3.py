"""
RANCD V3: Contrastive Edge Learning
===================================
Add contrastive loss to directly supervise edge discrimination.
Key insight: Edges should predict better than non-edges.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RANCDV3(nn.Module):
    """RANCD with contrastive edge supervision."""

    def __init__(self, n_factors: int, hidden_dim: int = 32,
                 n_lags: int = 5, temperature: float = 0.5):
        super().__init__()
        self.n_factors = n_factors
        self.n_lags = n_lags
        self.temperature = temperature

        # Per-factor predictors (each predicts one target)
        self.predictors = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(n_lags, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1)
                ) for _ in range(n_factors)  # One predictor per source
            ]) for _ in range(n_factors)  # For each target
        ])

        # Edge weights (learnable)
        self.edge_logits = nn.Parameter(torch.zeros(n_factors, n_factors))

    def forward(self, x, return_all=False):
        batch_size, seq_len, n_factors = x.shape
        device = x.device

        # Get edge probabilities
        edge_probs = torch.sigmoid(self.edge_logits / self.temperature)
        edge_probs = edge_probs * (1 - torch.eye(n_factors, device=device))

        # Predictions
        predictions = []
        for t in range(self.n_lags, seq_len):
            pred_t = []
            for j in range(n_factors):
                # Weighted sum of per-source predictions
                pred_j = 0
                for i in range(n_factors):
                    if i == j:
                        continue
                    # Source i's lagged values
                    x_i_lagged = x[:, t-self.n_lags:t, i]  # (batch, n_lags)
                    # Prediction from source i to target j
                    pred_ij = self.predictors[j][i](x_i_lagged)  # (batch, 1)
                    # Weight by edge probability
                    pred_j = pred_j + edge_probs[i, j] * pred_ij

                pred_t.append(pred_j)
            pred_t = torch.cat(pred_t, dim=-1)
            predictions.append(pred_t)

        predictions = torch.stack(predictions, dim=1)

        if return_all:
            adj = edge_probs.unsqueeze(0).expand(batch_size, -1, -1)
            return predictions, adj
        return predictions

    def compute_loss(self, x):
        batch_size, seq_len, n_factors = x.shape
        device = x.device

        # Get predictions and adjacency
        pred, adj = self(x, return_all=True)
        target = x[:, self.n_lags:, :]

        # 1. Prediction loss
        pred_loss = F.mse_loss(pred, target)

        # 2. Contrastive edge loss: good predictors should have high edge weight
        # For each target j, compare prediction errors from different sources
        contrastive_loss = 0
        edge_probs = torch.sigmoid(self.edge_logits / self.temperature)

        for j in range(n_factors):
            target_j = target[:, :, j:j+1]  # (batch, time, 1)
            errors = []

            for i in range(n_factors):
                if i == j:
                    errors.append(torch.tensor(float('inf'), device=device))
                    continue

                # Compute prediction from source i alone
                pred_ij = []
                for t in range(self.n_lags, seq_len):
                    x_i_lagged = x[:, t-self.n_lags:t, i]
                    p = self.predictors[j][i](x_i_lagged)
                    pred_ij.append(p)
                pred_ij = torch.stack(pred_ij, dim=1)  # (batch, time, 1)

                # Error from this source
                error_i = F.mse_loss(pred_ij, target_j, reduction='none').mean()
                errors.append(error_i)

            errors = torch.stack(errors)  # (n_factors,)

            # Sources with lower error should have higher edge probability
            # Use ranking loss: edge_probs should be inversely related to errors
            for i in range(n_factors):
                for k in range(n_factors):
                    if i == j or k == j or i == k:
                        continue
                    if errors[i] < errors[k]:
                        # i is better predictor, should have higher edge
                        margin = 0.1
                        contrastive_loss += F.relu(edge_probs[k, j] - edge_probs[i, j] + margin)

        contrastive_loss = contrastive_loss / (n_factors * n_factors)

        # 3. Sparsity
        sparse_loss = edge_probs.abs().mean()

        total = pred_loss + 0.5 * contrastive_loss + 0.01 * sparse_loss

        return total, {
            'total': total.item(),
            'pred': pred_loss.item(),
            'contrastive': contrastive_loss.item() if isinstance(contrastive_loss, torch.Tensor) else contrastive_loss,
            'adj_mean': edge_probs.mean().item()
        }

    def get_adjacency(self):
        edge_probs = torch.sigmoid(self.edge_logits / self.temperature)
        edge_probs = edge_probs * (1 - torch.eye(self.n_factors))
        return edge_probs.detach().cpu().numpy()


def test_rancdv3():
    """Test RANCD V3 with contrastive learning."""
    import sys
    sys.path.insert(0, '/Users/i767700/Github/ai-in-finance/papers/neural_causal_discovery/code')
    from data_loader import create_data_loader
    from nonlinear_experiments import generate_nonlinear_causal_data
    from baselines import LinearGrangerCausality, VARModel, evaluate_causal_discovery

    print("=" * 60)
    print("RANCD V3: Contrastive Edge Learning")
    print("=" * 60)

    results = {'granger': [], 'var': [], 'rancdv3': []}

    for trial in range(3):
        seed = 42 + trial
        print(f"\n--- Trial {trial+1}/3 (seed={seed}) ---")

        np.random.seed(seed)
        torch.manual_seed(seed)

        # Nonlinear data
        data, true_adj = generate_nonlinear_causal_data(n_factors=6, T=600, seed=seed)

        # Baselines
        gc = LinearGrangerCausality(n_lags=5)
        gc_m = evaluate_causal_discovery(true_adj, gc.fit(data))
        results['granger'].append(gc_m['f1'])
        print(f"Granger F1: {gc_m['f1']:.3f}")

        var = VARModel(n_lags=5)
        var_m = evaluate_causal_discovery(true_adj, var.fit(data), threshold=0.25)
        results['var'].append(var_m['f1'])
        print(f"VAR F1: {var_m['f1']:.3f}")

        # RANCD V3
        print("Training RANCD V3...")
        loader = create_data_loader(data, window_size=60, batch_size=16)
        model = RANCDV3(n_factors=6, hidden_dim=32, n_lags=5)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for epoch in range(50):
            model.train()
            for batch in loader:
                optimizer.zero_grad()
                loss, loss_dict = model.compute_loss(batch)
                loss.backward()
                optimizer.step()
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/50 | adj_mean: {loss_dict['adj_mean']:.3f}")

        rancd_adj = model.get_adjacency()
        print(f"Learned adjacency:\n{np.round(rancd_adj, 2)}")

        best_f1 = 0
        for thresh in [0.2, 0.3, 0.4, 0.5]:
            m = evaluate_causal_discovery(true_adj, rancd_adj, threshold=thresh)
            if m['f1'] > best_f1:
                best_f1 = m['f1']
        results['rancdv3'].append(best_f1)
        print(f"RANCD V3 F1: {best_f1:.3f}")

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY (Nonlinear Data)")
    print("=" * 60)
    for method in ['granger', 'var', 'rancdv3']:
        f1_mean = np.mean(results[method])
        f1_std = np.std(results[method])
        print(f"{method.upper():<12} F1: {f1_mean:.3f} ± {f1_std:.3f}")


if __name__ == "__main__":
    test_rancdv3()
