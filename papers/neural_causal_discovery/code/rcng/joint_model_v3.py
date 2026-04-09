"""
Joint RCNG v3: MDL-Guided Regime Discovery

Key insight: Use Minimum Description Length principle.
A regime should explain the data with the SIMPLEST causal graph possible.

For each timestep t:
- Compute effective cost = prediction_error + graph_complexity
- Assign to regime with lowest effective cost

This encourages:
1. Sparse causal graphs (simpler explanation)
2. Different sparsity patterns per regime (structural diversity)
3. Time points grouped by the structure that best explains them
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple


class SparseGraphPredictor(nn.Module):
    """
    Predictor with learnable sparse adjacency.
    Uses straight-through Gumbel-softmax for differentiable edge selection.
    """
    def __init__(self, n_factors: int, n_lags: int, hidden_dim: int):
        super().__init__()
        self.n_factors = n_factors
        self.n_lags = n_lags

        # Edge existence logits
        self.edge_logits = nn.Parameter(torch.zeros(n_factors, n_factors))
        self.register_buffer('diag_mask', 1 - torch.eye(n_factors))

        # Edge weight magnitudes (separate from existence)
        self.edge_weights = nn.Parameter(torch.ones(n_factors, n_factors) * 0.5)

        # Simple linear predictor per target
        self.predictors = nn.ModuleList([
            nn.Linear(n_lags * 2, 1)
            for _ in range(n_factors)
        ])

    def get_adj(self, temperature: float = 1.0, hard: bool = False) -> torch.Tensor:
        """
        Get adjacency matrix with Gumbel-softmax for differentiable edge selection.
        """
        # Mask diagonal
        logits = self.edge_logits * self.diag_mask

        # Gumbel-softmax for each edge (binary: exist or not)
        if self.training:
            # During training: soft with noise
            u = torch.rand_like(logits).clamp(1e-8, 1-1e-8)
            gumbel = -torch.log(-torch.log(u))
            soft = torch.sigmoid((logits + gumbel) / temperature)
        else:
            # During eval: hard threshold
            soft = torch.sigmoid(logits / temperature)

        if hard:
            hard_adj = (soft > 0.5).float()
            # Straight-through gradient
            adj = hard_adj - soft.detach() + soft
        else:
            adj = soft

        # Multiply by learned weights
        adj = adj * torch.sigmoid(self.edge_weights) * self.diag_mask

        return adj

    def get_sparsity(self) -> torch.Tensor:
        """Get expected number of edges (L0 proxy)."""
        probs = torch.sigmoid(self.edge_logits) * self.diag_mask
        return probs.sum()

    def forward(self, x_lagged: torch.Tensor, temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_lagged: (batch, time, n_lags, n_factors)
        Returns:
            predictions: (batch, time, n_factors)
            adj: (n_factors, n_factors)
        """
        batch, time, n_lags, n_factors = x_lagged.shape
        adj = self.get_adj(temperature)

        predictions = []
        for j in range(n_factors):
            self_lags = x_lagged[:, :, :, j]
            cross_lags = (x_lagged * adj[:, j].view(1, 1, 1, n_factors)).sum(dim=-1)
            inputs = torch.cat([self_lags, cross_lags], dim=-1)
            pred_j = self.predictors[j](inputs)
            predictions.append(pred_j)

        predictions = torch.cat(predictions, dim=-1)
        return predictions, adj


class JointRCNGv3(nn.Module):
    """
    Joint RCNG v3: MDL-Guided Regime Discovery

    Regime assignment based on:
    cost_k(t) = prediction_error_k(t) + β * edge_complexity_k

    Intuition: If regime k can explain time t with a simpler graph,
    then t should belong to regime k.

    This creates structural diversity because:
    - If all regimes had the same graph, they'd have the same cost
    - Diverse graphs → different cost profiles → meaningful regime separation
    """

    def __init__(
        self,
        n_factors: int,
        n_lags: int = 5,
        n_regimes: int = 3,
        hidden_dim: int = 32,
        beta: float = 0.1,  # Complexity penalty weight
        temperature: float = 0.5,
    ):
        super().__init__()

        self.n_factors = n_factors
        self.n_lags = n_lags
        self.n_regimes = n_regimes
        self.beta = beta
        self.temperature = temperature

        # Per-regime sparse predictors
        self.regime_predictors = nn.ModuleList([
            SparseGraphPredictor(n_factors, n_lags, hidden_dim)
            for _ in range(n_regimes)
        ])

    def create_lagged_data(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, time, n_factors = x.shape
        lagged_list = []
        for lag in range(1, self.n_lags + 1):
            lagged_list.append(x[:, self.n_lags - lag:-lag, :])
        x_lagged = torch.stack(lagged_list, dim=2)
        y_target = x[:, self.n_lags:, :]
        return x_lagged, y_target

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x_lagged, y_target = self.create_lagged_data(x)
        batch, time, _, _ = x_lagged.shape

        regime_preds = []
        regime_adjs = []
        regime_costs = []

        for k in range(self.n_regimes):
            pred_k, adj_k = self.regime_predictors[k](x_lagged, self.temperature)
            regime_preds.append(pred_k)
            regime_adjs.append(adj_k)

            # Per-timestep MSE
            error_k = ((pred_k - y_target) ** 2).mean(dim=-1)  # (batch, time)

            # Graph complexity (shared across time)
            complexity_k = self.regime_predictors[k].get_sparsity()

            # Total cost per timestep
            cost_k = error_k + self.beta * complexity_k  # (batch, time)
            regime_costs.append(cost_k)

        regime_preds = torch.stack(regime_preds, dim=0)
        regime_adjs = torch.stack(regime_adjs, dim=0)
        regime_costs = torch.stack(regime_costs, dim=-1)  # (batch, time, K)

        # Regime assignment: softmin over costs
        regime_probs = F.softmax(-regime_costs / self.temperature, dim=-1)

        # Final prediction
        predictions = torch.einsum('kbtf,btk->btf', regime_preds, regime_probs)

        return {
            'predictions': predictions,
            'regime_probs': regime_probs,
            'adj': regime_adjs,
            'regime_costs': regime_costs,
        }

    def compute_loss(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.forward(x)
        predictions = out['predictions']
        regime_probs = out['regime_probs']
        adj = out['adj']

        _, y_target = self.create_lagged_data(x)

        # Prediction loss
        L_pred = F.mse_loss(predictions, y_target)

        # Total sparsity across all regimes
        L_sparse = sum(self.regime_predictors[k].get_sparsity() for k in range(self.n_regimes))
        L_sparse = L_sparse / self.n_regimes

        # Diversity loss: encourage different sparsity PATTERNS (not just amounts)
        L_diverse = torch.tensor(0.0, device=adj.device)
        n_pairs = 0
        for k1 in range(self.n_regimes):
            for k2 in range(k1 + 1, self.n_regimes):
                # XOR-like: reward edges that exist in one but not other
                p1 = torch.sigmoid(self.regime_predictors[k1].edge_logits)
                p2 = torch.sigmoid(self.regime_predictors[k2].edge_logits)
                # Maximize: sum of |p1 - p2| (different edge patterns)
                diff = (p1 - p2).abs().sum()
                L_diverse = L_diverse - diff
                n_pairs += 1
        if n_pairs > 0:
            L_diverse = L_diverse / n_pairs

        # Entropy: encourage regime usage
        avg_prob = regime_probs.mean(dim=(0, 1))
        L_entropy = ((avg_prob - 1/self.n_regimes) ** 2).sum()

        # Confidence: encourage confident assignments
        per_t_entropy = -(regime_probs * (regime_probs + 1e-8).log()).sum(dim=-1).mean()
        L_confidence = per_t_entropy

        total = L_pred + 0.01 * L_sparse + 0.1 * L_diverse + 0.1 * L_entropy + 0.05 * L_confidence

        return {
            'total': total,
            'pred': L_pred,
            'sparse': L_sparse,
            'diverse': L_diverse,
        }

    def get_adjacency_matrices(self) -> np.ndarray:
        adjs = []
        for k in range(self.n_regimes):
            adj_k = self.regime_predictors[k].get_adj(self.temperature, hard=True)
            adjs.append(adj_k.detach().cpu().numpy())
        return np.stack(adjs, axis=0)

    def get_regime_assignments(self, x: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            out = self.forward(x)
            return out['regime_probs'].argmax(dim=-1).cpu().numpy()


def train_joint_rcng_v3(
    model: JointRCNGv3,
    data: np.ndarray,
    n_epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 32,
    window_size: int = 100,
    verbose: bool = True,
) -> Dict[str, list]:
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    T, n_factors = data.shape
    n_windows = T - window_size + 1

    windows = np.array([data[i:i+window_size] for i in range(n_windows)])
    windows_tensor = torch.tensor(windows, dtype=torch.float32, device=device)

    history = {'total': [], 'pred': [], 'sparse': [], 'diverse': []}

    for epoch in range(n_epochs):
        perm = torch.randperm(len(windows_tensor))
        epoch_losses = {k: [] for k in history.keys()}

        for i in range(0, len(windows_tensor), batch_size):
            batch = windows_tensor[perm[i:i+batch_size]]

            optimizer.zero_grad()
            losses = model.compute_loss(batch)
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            for k, v in losses.items():
                epoch_losses[k].append(v.item())

        for k in history.keys():
            history[k].append(np.mean(epoch_losses[k]))

        if verbose and (epoch + 1) % 10 == 0:
            # Also print edge counts per regime
            edge_counts = []
            for kk in range(model.n_regimes):
                adj = model.regime_predictors[kk].get_adj(model.temperature)
                edge_counts.append(f"{(adj > 0.5).sum().item():.0f}")
            print(f"Epoch {epoch+1}/{n_epochs} | "
                  f"Pred: {history['pred'][-1]:.4f} | "
                  f"Edges: [{'/'.join(edge_counts)}]")

    return history


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')

    from synthetic_data import RegimeSwitchingDGP
    from sklearn.metrics import adjusted_rand_score

    print("Testing JointRCNGv3 (MDL-guided)...")

    dgp = RegimeSwitchingDGP(seed=42)
    data, true_regimes, true_adj = dgp.generate(T=1500)

    print(f"True regime proportions: {dict(zip(range(3), [(true_regimes==k).mean() for k in range(3)]))}")
    print(f"True edge counts: {[true_adj[k].sum() for k in range(3)]}")

    model = JointRCNGv3(
        n_factors=6, n_lags=5, n_regimes=3,
        hidden_dim=32,
        beta=0.05,  # Complexity penalty
        temperature=0.3,
    )

    history = train_joint_rcng_v3(model, data, n_epochs=100, lr=1e-3, verbose=True)

    x_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
    pred_regimes = model.get_regime_assignments(x_tensor).flatten()

    n_lags = model.n_lags
    true_aligned = true_regimes[n_lags:]

    ari = adjusted_rand_score(true_aligned, pred_regimes)
    print(f"\nARI: {ari:.4f}")

    print("\nConfusion (true rows, pred cols):")
    for tk in range(3):
        row = [((true_aligned == tk) & (pred_regimes == pk)).sum() for pk in range(3)]
        print(f"  True {tk}: {row}")

    print("\nLearned adjacencies (binarized at 0.5):")
    adj = model.get_adjacency_matrices()
    for k in range(3):
        print(f"\nRegime {k} ({(adj[k] > 0.5).sum()} edges):")
        print((adj[k] > 0.5).astype(int))
