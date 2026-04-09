"""
Joint RCNG v2: Prediction-Guided Regime Discovery

Key insight: Instead of learning regime assignment separately,
use PREDICTION ERROR as the signal for regime membership.

If regime k's causal model explains the data well at time t,
then t should belong to regime k.

This creates a direct link between causal structure and regime assignment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple


class PerRegimePredictor(nn.Module):
    """
    Per-regime predictor: Each regime has its own causal graph and predictor.
    """
    def __init__(self, n_factors: int, n_lags: int, hidden_dim: int):
        super().__init__()
        self.n_factors = n_factors
        self.n_lags = n_lags

        # Learnable adjacency matrix (logits)
        self.adj_logits = nn.Parameter(torch.randn(n_factors, n_factors) * 0.1)
        self.register_buffer('diag_mask', 1 - torch.eye(n_factors))

        # Per-target predictors
        # Input: n_lags (self) + n_lags (weighted cross from one parent)
        self.predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_lags * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
            for _ in range(n_factors)
        ])

    def get_adj(self) -> torch.Tensor:
        """Get soft adjacency matrix."""
        adj = torch.sigmoid(self.adj_logits) * self.diag_mask
        return adj

    def forward(self, x_lagged: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_lagged: (batch, time, n_lags, n_factors)
        Returns:
            predictions: (batch, time, n_factors)
            adj: (n_factors, n_factors)
        """
        batch, time, n_lags, n_factors = x_lagged.shape
        adj = self.get_adj()

        predictions = []
        for j in range(n_factors):
            # Self lags
            self_lags = x_lagged[:, :, :, j]  # (batch, time, n_lags)

            # Cross lags weighted by adjacency
            # Weighted sum of all parents' contributions
            cross_lags = (x_lagged * adj[:, j].view(1, 1, 1, n_factors)).sum(dim=-1)  # (batch, time, n_lags)

            # Concatenate
            inputs = torch.cat([self_lags, cross_lags], dim=-1)  # (batch, time, n_lags*2)

            # Predict
            pred_j = self.predictors[j](inputs)  # (batch, time, 1)
            predictions.append(pred_j)

        predictions = torch.cat(predictions, dim=-1)  # (batch, time, n_factors)
        return predictions, adj


class JointRCNGv2(nn.Module):
    """
    Joint RCNG v2: Prediction-Guided Regime Discovery

    Key idea: Regime assignment is based on which regime's causal model
    best explains the current observation.

    π_t^k ∝ exp(-λ * MSE_k(t))

    This creates a DIRECT link between causal structure and regime assignment:
    - If regime k's graph explains t well → high π_t^k
    - Different causal structures → different time points assigned
    """

    def __init__(
        self,
        n_factors: int,
        n_lags: int = 5,
        n_regimes: int = 3,
        hidden_dim: int = 32,
        lambda_sparse: float = 0.01,
        lambda_diverse: float = 0.1,
        temperature: float = 1.0,
    ):
        super().__init__()

        self.n_factors = n_factors
        self.n_lags = n_lags
        self.n_regimes = n_regimes
        self.lambda_sparse = lambda_sparse
        self.lambda_diverse = lambda_diverse
        self.temperature = temperature

        # Per-regime predictors (each learns its own causal graph)
        self.regime_predictors = nn.ModuleList([
            PerRegimePredictor(n_factors, n_lags, hidden_dim)
            for _ in range(n_regimes)
        ])

        # Optional: Small regime prior network (helps break symmetry)
        self.regime_prior = nn.Sequential(
            nn.Linear(n_factors, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_regimes),
        )

    def create_lagged_data(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create lagged features and targets."""
        batch, time, n_factors = x.shape

        lagged_list = []
        for lag in range(1, self.n_lags + 1):
            lagged_list.append(x[:, self.n_lags - lag:-lag, :])

        x_lagged = torch.stack(lagged_list, dim=2)
        y_target = x[:, self.n_lags:, :]

        return x_lagged, y_target

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with prediction-guided regime assignment.
        """
        x_lagged, y_target = self.create_lagged_data(x)
        batch, time, _, _ = x_lagged.shape

        # Get predictions from each regime
        regime_preds = []
        regime_adjs = []
        regime_errors = []

        for k in range(self.n_regimes):
            pred_k, adj_k = self.regime_predictors[k](x_lagged)
            regime_preds.append(pred_k)
            regime_adjs.append(adj_k)

            # Per-timestep MSE
            error_k = ((pred_k - y_target) ** 2).mean(dim=-1)  # (batch, time)
            regime_errors.append(error_k)

        regime_preds = torch.stack(regime_preds, dim=0)  # (K, batch, time, n_factors)
        regime_adjs = torch.stack(regime_adjs, dim=0)    # (K, n_factors, n_factors)
        regime_errors = torch.stack(regime_errors, dim=-1)  # (batch, time, K)

        # Regime assignment based on prediction error (softmin)
        # Lower error → higher probability
        regime_probs = F.softmax(-regime_errors / self.temperature, dim=-1)  # (batch, time, K)

        # Add prior (helps break symmetry initially)
        # Use local statistics as prior input
        x_trimmed = x[:, self.n_lags:, :]  # Align with predictions
        prior_logits = self.regime_prior(x_trimmed)  # (batch, time, K)
        prior_probs = F.softmax(prior_logits, dim=-1)

        # Combine error-based and prior-based (weighted)
        alpha = 0.8  # Weight on error-based assignment
        regime_probs = alpha * regime_probs + (1 - alpha) * prior_probs

        # Final predictions: weighted average
        predictions = torch.einsum('kbtf,btk->btf', regime_preds, regime_probs)

        return {
            'predictions': predictions,
            'regime_probs': regime_probs,
            'adj': regime_adjs,
            'regime_preds': regime_preds,
            'regime_errors': regime_errors,
        }

    def compute_loss(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute loss."""
        out = self.forward(x)
        predictions = out['predictions']
        regime_probs = out['regime_probs']
        adj = out['adj']
        regime_errors = out['regime_errors']

        _, y_target = self.create_lagged_data(x)

        # L_pred: MSE
        L_pred = F.mse_loss(predictions, y_target)

        # L_sparse: Edge sparsity
        L_sparse = adj.abs().mean()

        # L_diverse: Different graphs across regimes
        L_diverse = torch.tensor(0.0, device=adj.device)
        n_pairs = 0
        for k1 in range(self.n_regimes):
            for k2 in range(k1 + 1, self.n_regimes):
                diff = (adj[k1] - adj[k2]).pow(2).sum().sqrt()
                L_diverse = L_diverse - diff
                n_pairs += 1
        if n_pairs > 0:
            L_diverse = L_diverse / n_pairs

        # L_entropy: Encourage regime usage
        avg_prob = regime_probs.mean(dim=(0, 1))
        target = 1.0 / self.n_regimes
        L_entropy = ((avg_prob - target) ** 2).sum()

        # L_confidence: Encourage confident assignments (low entropy per timestep)
        per_t_entropy = -(regime_probs * (regime_probs + 1e-8).log()).sum(dim=-1).mean()
        L_confidence = per_t_entropy  # We want to MINIMIZE this

        total_loss = (
            L_pred
            + self.lambda_sparse * L_sparse
            + self.lambda_diverse * L_diverse
            + 0.1 * L_entropy
            + 0.05 * L_confidence
        )

        return {
            'total': total_loss,
            'pred': L_pred,
            'sparse': L_sparse,
            'diverse': L_diverse,
        }

    def get_adjacency_matrices(self) -> np.ndarray:
        """Get learned adjacency matrices."""
        adjs = []
        for k in range(self.n_regimes):
            adj_k = self.regime_predictors[k].get_adj().detach().cpu().numpy()
            adjs.append(adj_k)
        return np.stack(adjs, axis=0)

    def get_regime_assignments(self, x: torch.Tensor) -> np.ndarray:
        """Get hard regime assignments."""
        with torch.no_grad():
            out = self.forward(x)
            regime_probs = out['regime_probs']
            assignments = regime_probs.argmax(dim=-1)
        return assignments.cpu().numpy()


def train_joint_rcng_v2(
    model: JointRCNGv2,
    data: np.ndarray,
    n_epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 32,
    window_size: int = 100,
    verbose: bool = True,
) -> Dict[str, list]:
    """Train Joint RCNG v2."""
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    T, n_factors = data.shape
    n_windows = T - window_size + 1

    windows = []
    for i in range(n_windows):
        windows.append(data[i:i+window_size])
    windows = np.array(windows)
    windows_tensor = torch.tensor(windows, dtype=torch.float32, device=device)

    history = {'total': [], 'pred': [], 'sparse': [], 'diverse': []}

    for epoch in range(n_epochs):
        perm = torch.randperm(len(windows_tensor))
        epoch_losses = {k: [] for k in history.keys()}

        for i in range(0, len(windows_tensor), batch_size):
            batch_idx = perm[i:i+batch_size]
            batch = windows_tensor[batch_idx]

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
            print(f"Epoch {epoch+1}/{n_epochs} | "
                  f"Total: {history['total'][-1]:.4f} | "
                  f"Pred: {history['pred'][-1]:.4f} | "
                  f"Diverse: {history['diverse'][-1]:.4f}")

    return history


if __name__ == "__main__":
    # Quick test
    import sys
    sys.path.insert(0, '.')

    from synthetic_data import RegimeSwitchingDGP

    print("Testing JointRCNGv2...")

    dgp = RegimeSwitchingDGP(seed=42)
    data, true_regimes, true_adj = dgp.generate(T=1000)

    print(f"Data shape: {data.shape}")
    print(f"Regime proportions: {dict(zip(range(3), [(true_regimes==k).mean() for k in range(3)]))}")

    model = JointRCNGv2(n_factors=6, n_lags=5, n_regimes=3, hidden_dim=32, temperature=0.5)

    history = train_joint_rcng_v2(model, data, n_epochs=50, verbose=True)

    # Check results
    x_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
    pred_regimes = model.get_regime_assignments(x_tensor).flatten()

    # Align lengths (predictions start from n_lags)
    n_lags = model.n_lags
    true_regimes_aligned = true_regimes[n_lags:]

    print(f"\nPred regime proportions: {dict(zip(range(3), [(pred_regimes==k).mean() for k in range(3)]))}")

    # Confusion matrix
    print("\nConfusion (true rows, pred cols):")
    for tk in range(3):
        row = [((true_regimes_aligned == tk) & (pred_regimes == pk)).sum() for pk in range(3)]
        print(f"  True {tk}: {row}")

    print("\nLearned adjacencies:")
    adj = model.get_adjacency_matrices()
    for k in range(3):
        print(f"\nRegime {k}:")
        print(np.round(adj[k], 2))
