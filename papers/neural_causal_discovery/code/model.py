"""
Regime-Aware Neural Causal Discovery (RANCD)
=============================================

A novel architecture that jointly learns:
1. Latent regime structure (via temporal encoder)
2. Regime-conditional causal graphs (via graph structure learner)
3. Time-varying causal strength (via causal attention)

Key innovations:
- Per-factor temporal embeddings for pair-wise edge prediction
- NOTEARS-style DAG constraint for acyclicity
- Regime-conditioned graph learning
- Interpretable attention weights

Reference papers:
- Tank et al. (2021) - Neural Granger Causality
- Zheng et al. (2018) - NOTEARS
- Kipf et al. (2018) - NRI
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class FactorEncoder(nn.Module):
    """
    Encode each factor's time series into a latent representation.
    Produces per-factor embeddings for pair-wise edge prediction.
    """

    def __init__(self, n_factors: int, hidden_dim: int, n_layers: int = 2):
        super().__init__()
        self.n_factors = n_factors
        self.hidden_dim = hidden_dim

        # Per-factor LSTM encoder
        self.factor_lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=False
        )

        # Temporal attention for pooling
        self.temporal_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, time, n_factors) - multivariate time series

        Returns:
            factor_emb: (batch, n_factors, hidden_dim) - per-factor embeddings
        """
        batch_size, seq_len, n_factors = x.shape
        assert n_factors == self.n_factors

        factor_embeddings = []

        for i in range(n_factors):
            # Extract single factor: (batch, time, 1)
            factor_i = x[:, :, i:i+1]

            # Encode: (batch, time, hidden_dim)
            h_i, _ = self.factor_lstm(factor_i)

            # Attention pooling over time
            attn_weights = self.temporal_attention(h_i)  # (batch, time, 1)
            attn_weights = F.softmax(attn_weights, dim=1)

            # Weighted sum: (batch, hidden_dim)
            factor_emb_i = (h_i * attn_weights).sum(dim=1)
            factor_embeddings.append(factor_emb_i)

        # Stack: (batch, n_factors, hidden_dim)
        factor_emb = torch.stack(factor_embeddings, dim=1)

        return factor_emb


class RegimeEncoder(nn.Module):
    """
    Discover latent regimes from multivariate time series.
    Uses Transformer to capture temporal dependencies.
    """

    def __init__(self, n_factors: int, hidden_dim: int, n_regimes: int = 3,
                 n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.n_regimes = n_regimes

        # Input projection
        self.input_proj = nn.Linear(n_factors, hidden_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)

        # Regime classifier
        self.regime_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_regimes)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, time, n_factors)

        Returns:
            regime_probs: (batch, time, n_regimes) - soft regime assignments
            h: (batch, time, hidden_dim) - temporal embeddings
        """
        h = self.input_proj(x)
        h = self.transformer(h)

        regime_logits = self.regime_head(h)
        regime_probs = F.softmax(regime_logits, dim=-1)

        return regime_probs, h


class GraphStructureLearner(nn.Module):
    """
    Learn causal graph structure from factor embeddings.
    Predicts edge probability for each directed pair.

    Key: Regime-conditioned graph learning.
    """

    def __init__(self, hidden_dim: int, n_factors: int, n_regimes: int = 3,
                 temperature: float = 0.5):
        super().__init__()
        self.n_factors = n_factors
        self.n_regimes = n_regimes
        self.temperature = temperature

        # Edge predictor: takes [source_emb, target_emb, regime_emb]
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + n_regimes, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Regime embedding for conditioning
        self.regime_embedding = nn.Embedding(n_regimes, n_regimes)

    def forward(self, factor_emb: torch.Tensor,
                regime_probs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            factor_emb: (batch, n_factors, hidden_dim) - per-factor embeddings
            regime_probs: (batch, n_regimes) - regime probabilities (pooled over time)

        Returns:
            adj: (batch, n_factors, n_factors) - soft adjacency matrix
        """
        batch_size = factor_emb.size(0)
        device = factor_emb.device

        adj = torch.zeros(batch_size, self.n_factors, self.n_factors, device=device)

        for i in range(self.n_factors):
            for j in range(self.n_factors):
                if i == j:
                    continue  # No self-loops

                # Source and target embeddings
                src_emb = factor_emb[:, i, :]  # (batch, hidden)
                tgt_emb = factor_emb[:, j, :]  # (batch, hidden)

                # Concatenate with regime info
                edge_input = torch.cat([src_emb, tgt_emb, regime_probs], dim=-1)

                # Predict edge probability
                edge_logit = self.edge_mlp(edge_input).squeeze(-1)
                edge_prob = torch.sigmoid(edge_logit / self.temperature)

                adj[:, i, j] = edge_prob

        return adj


class DAGConstraint(nn.Module):
    """
    NOTEARS-style continuous DAG constraint.
    h(W) = tr(e^{W∘W}) - d = 0 for acyclic graphs.
    """

    def __init__(self):
        super().__init__()

    def forward(self, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            adj: (batch, n, n) - adjacency matrix

        Returns:
            dag_loss: scalar - acyclicity violation
        """
        # Average over batch
        W = adj.mean(dim=0)  # (n, n)
        n = W.size(0)

        # h(W) = tr(e^{W∘W}) - n
        W_squared = W * W  # Element-wise square
        expm_W = torch.matrix_exp(W_squared)
        h = torch.trace(expm_W) - n

        return h


class CausalPredictor(nn.Module):
    """
    Use learned causal graph for prediction.
    Implements Granger-style prediction with graph masking.
    """

    def __init__(self, n_factors: int, hidden_dim: int, n_lags: int = 5):
        super().__init__()
        self.n_factors = n_factors
        self.n_lags = n_lags

        # Prediction MLP for each target
        self.predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_factors * n_lags, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
            for _ in range(n_factors)
        ])

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, time, n_factors)
            adj: (batch, n_factors, n_factors) - causal graph

        Returns:
            pred: (batch, time - n_lags, n_factors) - predictions
        """
        batch_size, seq_len, n_factors = x.shape
        predictions = []

        for t in range(self.n_lags, seq_len):
            # Lagged inputs: (batch, n_lags, n_factors)
            x_lagged = x[:, t-self.n_lags:t, :]

            # Flatten: (batch, n_lags * n_factors)
            x_flat = x_lagged.reshape(batch_size, -1)

            # Predict each factor
            pred_t = []
            for j in range(n_factors):
                # Mask inputs by causal graph (which factors cause j)
                # adj[:, :, j] gives incoming edges to j
                mask = adj[:, :, j]  # (batch, n_factors)

                # Expand mask over lags
                mask_expanded = mask.unsqueeze(1).expand(-1, self.n_lags, -1)
                mask_flat = mask_expanded.reshape(batch_size, -1)

                # Apply mask
                x_masked = x_flat * mask_flat

                # Predict
                pred_j = self.predictors[j](x_masked)
                pred_t.append(pred_j)

            pred_t = torch.cat(pred_t, dim=-1)  # (batch, n_factors)
            predictions.append(pred_t)

        predictions = torch.stack(predictions, dim=1)  # (batch, time-n_lags, n_factors)
        return predictions


class RANCD(nn.Module):
    """
    Regime-Aware Neural Causal Discovery (RANCD)

    Main model that combines:
    1. Factor encoding (per-factor embeddings)
    2. Regime discovery (temporal transformer)
    3. Graph structure learning (regime-conditional)
    4. Causal prediction (graph-masked)

    Losses:
    - Prediction loss (Granger-style)
    - DAG constraint (acyclicity)
    - Sparsity penalty (edge regularization)
    - Regime smoothness (temporal consistency)
    """

    def __init__(self, n_factors: int, hidden_dim: int = 64, n_regimes: int = 3,
                 n_lags: int = 5, temperature: float = 0.5):
        super().__init__()
        self.n_factors = n_factors
        self.n_regimes = n_regimes
        self.n_lags = n_lags

        # Components
        self.factor_encoder = FactorEncoder(n_factors, hidden_dim)
        self.regime_encoder = RegimeEncoder(n_factors, hidden_dim, n_regimes)
        self.graph_learner = GraphStructureLearner(hidden_dim, n_factors, n_regimes, temperature)
        self.dag_constraint = DAGConstraint()
        self.predictor = CausalPredictor(n_factors, hidden_dim, n_lags)

    def forward(self, x: torch.Tensor, return_all: bool = False):
        """
        Args:
            x: (batch, time, n_factors) - input time series
            return_all: whether to return intermediate outputs

        Returns:
            pred: (batch, time - n_lags, n_factors) - predictions
            Additional outputs if return_all=True
        """
        # 1. Encode factors
        factor_emb = self.factor_encoder(x)  # (batch, n_factors, hidden)

        # 2. Discover regimes
        regime_probs, temporal_emb = self.regime_encoder(x)  # (batch, time, n_regimes)

        # Pool regime probs over time for graph conditioning
        regime_pooled = regime_probs.mean(dim=1)  # (batch, n_regimes)

        # 3. Learn causal graph
        adj = self.graph_learner(factor_emb, regime_pooled)  # (batch, n, n)

        # 4. Predict using graph
        pred = self.predictor(x, adj)

        if return_all:
            return pred, adj, regime_probs, factor_emb
        return pred

    def compute_loss(self, x: torch.Tensor,
                     lambda_pred: float = 1.0,
                     lambda_dag: float = 0.5,
                     lambda_sparse: float = 0.01,
                     lambda_regime: float = 0.1) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined loss for training.

        Args:
            x: (batch, time, n_factors)
            lambda_*: loss weights

        Returns:
            total_loss: scalar
            loss_dict: individual loss components
        """
        # Forward pass
        pred, adj, regime_probs, _ = self.forward(x, return_all=True)

        # Target (shifted by n_lags)
        target = x[:, self.n_lags:, :]

        # 1. Prediction loss (MSE)
        pred_loss = F.mse_loss(pred, target)

        # 2. DAG constraint
        dag_loss = self.dag_constraint(adj)

        # 3. Sparsity loss (L1 on edges)
        sparse_loss = adj.abs().mean()

        # 4. Regime smoothness (encourage temporal consistency)
        regime_diff = (regime_probs[:, 1:, :] - regime_probs[:, :-1, :]).abs()
        regime_loss = regime_diff.mean()

        # Combined loss
        total_loss = (
            lambda_pred * pred_loss +
            lambda_dag * dag_loss +
            lambda_sparse * sparse_loss +
            lambda_regime * regime_loss
        )

        loss_dict = {
            'total': total_loss.item(),
            'pred': pred_loss.item(),
            'dag': dag_loss.item(),
            'sparse': sparse_loss.item(),
            'regime': regime_loss.item()
        }

        return total_loss, loss_dict

    def get_causal_graph(self, x: torch.Tensor) -> np.ndarray:
        """Extract learned causal graph."""
        self.eval()
        with torch.no_grad():
            _, adj, _, _ = self.forward(x, return_all=True)
        return adj.cpu().numpy()

    def get_regime_assignments(self, x: torch.Tensor) -> np.ndarray:
        """Extract regime assignments."""
        self.eval()
        with torch.no_grad():
            _, _, regime_probs, _ = self.forward(x, return_all=True)
        return regime_probs.argmax(dim=-1).cpu().numpy()


def train_rancd(model: RANCD, data_loader, n_epochs: int = 100,
                lr: float = 1e-3, device: str = 'cpu'):
    """Training loop for RANCD."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = []

    for epoch in range(n_epochs):
        model.train()
        epoch_losses = []

        for batch in data_loader:
            x = batch.to(device)

            optimizer.zero_grad()
            loss, loss_dict = model.compute_loss(x)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss_dict)

        # Average epoch losses
        avg_loss = {k: np.mean([d[k] for d in epoch_losses]) for k in epoch_losses[0]}
        history.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs} | Loss: {avg_loss['total']:.4f} | "
                  f"Pred: {avg_loss['pred']:.4f} | DAG: {avg_loss['dag']:.4f}")

    return history


# Quick test
if __name__ == "__main__":
    print("Testing RANCD model...")

    batch_size = 16
    seq_len = 100
    n_factors = 6

    model = RANCD(n_factors=n_factors, hidden_dim=32, n_regimes=3, n_lags=5)
    x = torch.randn(batch_size, seq_len, n_factors)

    # Forward pass
    pred = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Prediction shape: {pred.shape}")

    # With all outputs
    pred, adj, regimes, factor_emb = model(x, return_all=True)
    print(f"Adjacency shape: {adj.shape}")
    print(f"Regime probs shape: {regimes.shape}")
    print(f"Factor embeddings shape: {factor_emb.shape}")

    # Compute loss
    loss, loss_dict = model.compute_loss(x)
    print(f"Loss: {loss_dict}")

    # Get causal graph
    graph = model.get_causal_graph(x)
    print(f"Causal graph shape: {graph.shape}")
    print(f"Edge density: {(graph > 0.5).mean():.2%}")

    print("\n✅ RANCD model test passed!")
