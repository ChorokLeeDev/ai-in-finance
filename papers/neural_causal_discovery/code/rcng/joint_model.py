"""
Joint Regime-Conditional Neural Granger (RCNG) Model

End-to-end joint learning of:
1. Regime discovery (soft assignment)
2. Per-regime causal graphs
3. Regime-weighted prediction

Key novelty: L_diverse loss encourages regimes to have DIFFERENT causal structures,
so regimes are defined by causal differences, not just distributional differences.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional


class RegimeEncoder(nn.Module):
    """
    Encodes time series into soft regime assignments.
    Uses GRU + attention for temporal context.
    Also uses local volatility features to help distinguish regimes.
    """
    def __init__(self, n_factors: int, hidden_dim: int, n_regimes: int):
        super().__init__()
        self.n_regimes = n_regimes

        # Input: n_factors + n_factors (rolling volatility features)
        input_dim = n_factors * 2

        # GRU for temporal encoding
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=True)

        # Project to regime logits
        self.regime_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_regimes)
        )

        # Temperature for softmax (learnable)
        self.temperature = nn.Parameter(torch.ones(1) * 0.5)  # Start sharper

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, time, n_factors)
        Returns:
            regime_probs: (batch, time, n_regimes) - soft regime assignments
        """
        # Compute local volatility features (rolling window std)
        # Use a simple exponential moving average of squared values
        x_sq = x ** 2
        # Simple cumulative moving average approximation
        vol_features = torch.zeros_like(x)
        alpha = 0.1
        vol_features[:, 0, :] = x_sq[:, 0, :]
        for t in range(1, x.shape[1]):
            vol_features[:, t, :] = alpha * x_sq[:, t, :] + (1 - alpha) * vol_features[:, t-1, :]
        vol_features = torch.sqrt(vol_features + 1e-6)

        # Concatenate original features with volatility
        x_aug = torch.cat([x, vol_features], dim=-1)  # (batch, time, n_factors*2)

        # Temporal encoding
        h, _ = self.gru(x_aug)  # (batch, time, hidden*2)

        # Regime logits
        logits = self.regime_proj(h)  # (batch, time, n_regimes)

        # Soft assignment with temperature
        regime_probs = F.softmax(logits / self.temperature.clamp(min=0.1), dim=-1)

        return regime_probs


class CausalGraphLearner(nn.Module):
    """
    Learns K causal adjacency matrices, one per regime.
    Each A^(k) represents the Granger-causal structure in regime k.
    """
    def __init__(self, n_factors: int, n_regimes: int):
        super().__init__()
        self.n_factors = n_factors
        self.n_regimes = n_regimes

        # Learnable adjacency matrices (logits)
        # Shape: (n_regimes, n_factors, n_factors)
        self.adj_logits = nn.Parameter(torch.randn(n_regimes, n_factors, n_factors) * 0.1)

        # Mask diagonal (no self-loops in causal graph)
        self.register_buffer('diag_mask', 1 - torch.eye(n_factors))

    def forward(self) -> torch.Tensor:
        """
        Returns:
            adj: (n_regimes, n_factors, n_factors) - soft adjacency matrices in [0,1]
        """
        # Sigmoid to get [0,1] edge probabilities
        adj = torch.sigmoid(self.adj_logits)

        # Zero out diagonal
        adj = adj * self.diag_mask.unsqueeze(0)

        return adj


class NeuralGrangerPredictor(nn.Module):
    """
    Per-regime neural Granger predictor.
    For each regime k, predicts x_t using lagged values weighted by A^(k).
    """
    def __init__(self, n_factors: int, n_lags: int, hidden_dim: int, n_regimes: int):
        super().__init__()
        self.n_factors = n_factors
        self.n_lags = n_lags
        self.n_regimes = n_regimes

        # Per-regime predictors: each is an MLP that takes [self_lags, weighted_cross_lags]
        # Input: n_lags (self) + n_lags * n_factors (cross, weighted by adj)
        input_dim = n_lags + n_lags * n_factors

        self.predictors = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim, 1)
                )
                for _ in range(n_factors)  # One predictor per target factor
            ])
            for _ in range(n_regimes)  # One set per regime
        ])

    def forward(self, x_lagged: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_lagged: (batch, time, n_lags, n_factors) - lagged observations
            adj: (n_regimes, n_factors, n_factors) - adjacency matrices
        Returns:
            predictions: (n_regimes, batch, time, n_factors)
        """
        batch, time, n_lags, n_factors = x_lagged.shape

        predictions = []

        for k in range(self.n_regimes):
            regime_preds = []

            for j in range(n_factors):  # Target factor
                # Self lags: x_lagged[:, :, :, j]
                self_lags = x_lagged[:, :, :, j]  # (batch, time, n_lags)

                # Cross lags weighted by adjacency
                # adj[k, i, j] = edge weight from i to j in regime k
                cross_lags = x_lagged * adj[k, :, j].view(1, 1, 1, n_factors)  # weighted
                cross_lags = cross_lags.reshape(batch, time, -1)  # (batch, time, n_lags*n_factors)

                # Concatenate self and cross
                inputs = torch.cat([self_lags, cross_lags], dim=-1)  # (batch, time, n_lags + n_lags*n_factors)

                # Predict
                pred_j = self.predictors[k][j](inputs)  # (batch, time, 1)
                regime_preds.append(pred_j)

            regime_preds = torch.cat(regime_preds, dim=-1)  # (batch, time, n_factors)
            predictions.append(regime_preds)

        predictions = torch.stack(predictions, dim=0)  # (n_regimes, batch, time, n_factors)
        return predictions


class JointRCNG(nn.Module):
    """
    Joint Regime-Conditional Neural Granger Model

    End-to-end learning of:
    1. Soft regime assignments: π_t^k = P(regime=k | X)
    2. Per-regime causal graphs: A^(k)
    3. Regime-weighted predictions: ŷ_t = Σ_k π_t^k · f_k(x, A^(k))

    Loss = L_pred + λ_sparse * L_sparse + λ_smooth * L_smooth + λ_diverse * L_diverse

    Key novelty: L_diverse encourages different regimes to have different causal structures.
    """
    def __init__(
        self,
        n_factors: int,
        n_lags: int = 5,
        n_regimes: int = 3,
        hidden_dim: int = 32,
        lambda_sparse: float = 0.01,
        lambda_smooth: float = 0.1,
        lambda_diverse: float = 0.1,
    ):
        super().__init__()

        self.n_factors = n_factors
        self.n_lags = n_lags
        self.n_regimes = n_regimes

        # Hyperparameters
        self.lambda_sparse = lambda_sparse
        self.lambda_smooth = lambda_smooth
        self.lambda_diverse = lambda_diverse

        # Components
        self.regime_encoder = RegimeEncoder(n_factors, hidden_dim, n_regimes)
        self.graph_learner = CausalGraphLearner(n_factors, n_regimes)
        self.predictor = NeuralGrangerPredictor(n_factors, n_lags, hidden_dim, n_regimes)

    def create_lagged_data(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create lagged features and targets.

        Args:
            x: (batch, time, n_factors)
        Returns:
            x_lagged: (batch, time-n_lags, n_lags, n_factors)
            y_target: (batch, time-n_lags, n_factors)
        """
        batch, time, n_factors = x.shape

        # Create lagged tensor
        lagged_list = []
        for lag in range(1, self.n_lags + 1):
            lagged_list.append(x[:, self.n_lags - lag:-lag, :])

        x_lagged = torch.stack(lagged_list, dim=2)  # (batch, time-n_lags, n_lags, n_factors)
        y_target = x[:, self.n_lags:, :]  # (batch, time-n_lags, n_factors)

        return x_lagged, y_target

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: (batch, time, n_factors)
        Returns:
            dict with:
                - predictions: (batch, time-n_lags, n_factors)
                - regime_probs: (batch, time, n_regimes)
                - adj: (n_regimes, n_factors, n_factors)
        """
        # Get regime probabilities
        regime_probs = self.regime_encoder(x)  # (batch, time, n_regimes)

        # Get causal graphs
        adj = self.graph_learner()  # (n_regimes, n_factors, n_factors)

        # Create lagged data
        x_lagged, _ = self.create_lagged_data(x)  # (batch, time-n_lags, n_lags, n_factors)

        # Get per-regime predictions
        regime_preds = self.predictor(x_lagged, adj)  # (n_regimes, batch, time-n_lags, n_factors)

        # Trim regime_probs to match prediction length
        regime_probs_trimmed = regime_probs[:, self.n_lags:, :]  # (batch, time-n_lags, n_regimes)

        # Regime-weighted prediction: ŷ_t = Σ_k π_t^k · f_k(x, A^(k))
        # regime_preds: (n_regimes, batch, time-n_lags, n_factors)
        # regime_probs_trimmed: (batch, time-n_lags, n_regimes)
        predictions = torch.einsum('kbtf,btk->btf', regime_preds, regime_probs_trimmed)

        return {
            'predictions': predictions,
            'regime_probs': regime_probs,
            'adj': adj,
            'regime_preds': regime_preds,
        }

    def compute_loss(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute total loss with all components.

        Loss = L_pred + λ_sparse * L_sparse + λ_smooth * L_smooth + λ_diverse * L_diverse + λ_entropy * L_entropy
        """
        # Forward pass
        out = self.forward(x)
        predictions = out['predictions']
        regime_probs = out['regime_probs']
        adj = out['adj']
        regime_preds = out['regime_preds']

        # Get target
        _, y_target = self.create_lagged_data(x)

        # L_pred: MSE prediction loss
        L_pred = F.mse_loss(predictions, y_target)

        # L_sparse: L1 regularization on edges
        L_sparse = adj.abs().mean()

        # L_smooth: Regime smoothness (discourage rapid switching)
        regime_diff = regime_probs[:, 1:, :] - regime_probs[:, :-1, :]
        L_smooth = (regime_diff ** 2).mean()

        # L_diverse: Encourage different causal structures across regimes (KEY NOVELTY)
        # Negative pairwise distance between adjacency matrices
        L_diverse = torch.tensor(0.0, device=adj.device)
        n_pairs = 0
        for k1 in range(self.n_regimes):
            for k2 in range(k1 + 1, self.n_regimes):
                # Frobenius norm of difference
                diff = (adj[k1] - adj[k2]).pow(2).sum().sqrt()
                L_diverse = L_diverse - diff  # Negative because we want to MAXIMIZE diversity
                n_pairs += 1
        if n_pairs > 0:
            L_diverse = L_diverse / n_pairs

        # L_entropy: Encourage balanced regime usage (prevent collapse to single regime)
        # Average regime probability across time
        regime_probs_trimmed = regime_probs[:, self.n_lags:, :]
        avg_regime_prob = regime_probs_trimmed.mean(dim=(0, 1))  # (n_regimes,)
        # Target: uniform distribution
        target_prob = 1.0 / self.n_regimes
        # KL-like penalty for deviating from uniform
        L_entropy = ((avg_regime_prob - target_prob) ** 2).sum()

        # Total loss
        total_loss = (
            L_pred
            + self.lambda_sparse * L_sparse
            + self.lambda_smooth * L_smooth
            + self.lambda_diverse * L_diverse
            + 0.1 * L_entropy  # Encourage balanced regimes
        )

        return {
            'total': total_loss,
            'pred': L_pred,
            'sparse': L_sparse,
            'smooth': L_smooth,
            'diverse': L_diverse,
        }

    def get_adjacency_matrices(self) -> np.ndarray:
        """Get learned adjacency matrices as numpy array."""
        with torch.no_grad():
            adj = self.graph_learner()
        return adj.cpu().numpy()

    def get_regime_assignments(self, x: torch.Tensor) -> np.ndarray:
        """Get hard regime assignments."""
        with torch.no_grad():
            regime_probs = self.regime_encoder(x)
            assignments = regime_probs.argmax(dim=-1)
        return assignments.cpu().numpy()


def train_joint_rcng(
    model: JointRCNG,
    data: np.ndarray,
    n_epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 32,
    window_size: int = 100,
    verbose: bool = True,
) -> Dict[str, list]:
    """
    Train Joint RCNG model.

    Args:
        model: JointRCNG model
        data: (time, n_factors) numpy array
        n_epochs: number of training epochs
        lr: learning rate
        batch_size: batch size
        window_size: sliding window size for creating sequences
        verbose: print progress

    Returns:
        history: dict of loss histories
    """
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Create sliding windows
    T, n_factors = data.shape
    n_windows = T - window_size + 1

    windows = []
    for i in range(n_windows):
        windows.append(data[i:i+window_size])
    windows = np.array(windows)  # (n_windows, window_size, n_factors)

    # Convert to tensor
    windows_tensor = torch.tensor(windows, dtype=torch.float32, device=device)

    history = {'total': [], 'pred': [], 'sparse': [], 'smooth': [], 'diverse': []}

    for epoch in range(n_epochs):
        # Shuffle windows
        perm = torch.randperm(len(windows_tensor))

        epoch_losses = {k: [] for k in history.keys()}

        for i in range(0, len(windows_tensor), batch_size):
            batch_idx = perm[i:i+batch_size]
            batch = windows_tensor[batch_idx]

            optimizer.zero_grad()
            losses = model.compute_loss(batch)
            losses['total'].backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            for k, v in losses.items():
                epoch_losses[k].append(v.item())

        # Record epoch losses
        for k in history.keys():
            history[k].append(np.mean(epoch_losses[k]))

        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs} | "
                  f"Total: {history['total'][-1]:.4f} | "
                  f"Pred: {history['pred'][-1]:.4f} | "
                  f"Diverse: {history['diverse'][-1]:.4f}")

    return history


# Utility functions for evaluation

def binarize_adjacency(adj: np.ndarray, threshold: float = 0.3) -> np.ndarray:
    """Binarize soft adjacency matrix."""
    return (adj > threshold).astype(float)


def compute_regime_f1(pred_adj: np.ndarray, true_adj: np.ndarray, threshold: float = 0.3) -> Dict[str, float]:
    """
    Compute F1 score per regime.

    Args:
        pred_adj: (n_regimes, n_factors, n_factors) predicted adjacency
        true_adj: (n_regimes, n_factors, n_factors) true adjacency
        threshold: binarization threshold

    Returns:
        dict with per-regime and average F1
    """
    n_regimes = pred_adj.shape[0]

    pred_binary = binarize_adjacency(pred_adj, threshold)
    true_binary = (true_adj > 0).astype(float)

    results = {}
    f1_scores = []

    for k in range(n_regimes):
        pred_k = pred_binary[k].flatten()
        true_k = true_binary[k].flatten()

        tp = ((pred_k == 1) & (true_k == 1)).sum()
        fp = ((pred_k == 1) & (true_k == 0)).sum()
        fn = ((pred_k == 0) & (true_k == 1)).sum()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        results[f'regime_{k}_precision'] = precision
        results[f'regime_{k}_recall'] = recall
        results[f'regime_{k}_f1'] = f1
        f1_scores.append(f1)

    results['macro_f1'] = np.mean(f1_scores)

    return results
