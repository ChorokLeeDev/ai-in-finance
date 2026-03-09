"""
Neural Causal Discovery Model
=============================

Regime-aware neural causal discovery for financial networks.

Architecture:
1. Temporal Encoder - Learn regime representations
2. Graph Structure Learner - Discover causal edges
3. Causal Attention - Time-varying edge weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TemporalEncoder(nn.Module):
    """Encode temporal patterns to discover latent regimes."""

    def __init__(self, input_dim, hidden_dim, n_layers=2, n_heads=4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)

    def forward(self, x):
        """
        Args:
            x: (batch, time, n_factors)
        Returns:
            h: (batch, time, hidden_dim)
        """
        h = self.input_proj(x)
        h = self.transformer(h)
        return h


class GraphStructureLearner(nn.Module):
    """Learn causal graph structure from temporal embeddings."""

    def __init__(self, hidden_dim, n_factors, temperature=0.5):
        super().__init__()
        self.n_factors = n_factors
        self.temperature = temperature

        # Edge probability predictor
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, h):
        """
        Args:
            h: (batch, time, hidden_dim) - per-factor embeddings needed
        Returns:
            adj: (batch, n_factors, n_factors) - adjacency matrix
        """
        batch_size = h.size(0)

        # Pool temporal dimension
        h_pooled = h.mean(dim=1)  # (batch, hidden_dim)

        # For each pair, predict edge probability
        # This is simplified - real implementation needs per-factor embeddings
        adj = torch.zeros(batch_size, self.n_factors, self.n_factors)

        # TODO: Implement proper pair-wise edge prediction

        return adj


class CausalAttention(nn.Module):
    """Time-varying causal attention over graph edges."""

    def __init__(self, hidden_dim, n_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_dim, n_heads, dropout=0.1, batch_first=True
        )

    def forward(self, h, adj_matrix):
        """
        Args:
            h: (batch, time, hidden_dim)
            adj_matrix: (batch, n_factors, n_factors)
        Returns:
            causal_weights: (batch, time, n_factors, n_factors)
        """
        # Use attention to compute time-varying causal strength
        attn_output, attn_weights = self.attention(h, h, h)
        return attn_output, attn_weights


class NeuralCausalDiscovery(nn.Module):
    """
    Main model: Regime-aware Neural Causal Discovery

    Learns:
    1. Latent regime structure (via temporal encoder)
    2. Causal graph structure (via graph learner)
    3. Time-varying causal strength (via causal attention)
    """

    def __init__(self, n_factors, hidden_dim=64, n_regimes=3):
        super().__init__()
        self.n_factors = n_factors
        self.n_regimes = n_regimes
        self.hidden_dim = hidden_dim

        # Components
        self.temporal_encoder = TemporalEncoder(n_factors, hidden_dim)
        self.graph_learner = GraphStructureLearner(hidden_dim, n_factors)
        self.causal_attention = CausalAttention(hidden_dim)

        # Regime classifier
        self.regime_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_regimes)
        )

        # Prediction head (for Granger-style prediction loss)
        self.prediction_head = nn.Linear(hidden_dim, n_factors)

    def forward(self, x, return_graph=False):
        """
        Args:
            x: (batch, time, n_factors) - factor returns
            return_graph: whether to return causal graph

        Returns:
            predictions: (batch, time, n_factors)
            regimes: (batch, time, n_regimes) - regime probabilities
            adj_matrix: (batch, n_factors, n_factors) - if return_graph
        """
        # 1. Temporal encoding
        h = self.temporal_encoder(x)  # (batch, time, hidden)

        # 2. Learn causal graph
        adj_matrix = self.graph_learner(h)

        # 3. Time-varying causal attention
        causal_h, causal_weights = self.causal_attention(h, adj_matrix)

        # 4. Regime classification
        regime_logits = self.regime_head(h)
        regimes = F.softmax(regime_logits, dim=-1)

        # 5. Prediction (for training)
        predictions = self.prediction_head(causal_h)

        if return_graph:
            return predictions, regimes, adj_matrix, causal_weights
        return predictions, regimes

    def get_causal_graph(self, x):
        """Extract learned causal graph for interpretation."""
        with torch.no_grad():
            _, _, adj, weights = self.forward(x, return_graph=True)
        return adj.numpy(), weights.numpy()


# Loss functions
class CausalDiscoveryLoss(nn.Module):
    """Combined loss for causal discovery training."""

    def __init__(self, lambda_pred=1.0, lambda_sparse=0.1, lambda_dag=0.5):
        super().__init__()
        self.lambda_pred = lambda_pred
        self.lambda_sparse = lambda_sparse
        self.lambda_dag = lambda_dag

    def forward(self, predictions, targets, adj_matrix):
        """
        Args:
            predictions: (batch, time, n_factors)
            targets: (batch, time, n_factors) - shifted by 1
            adj_matrix: (batch, n_factors, n_factors)
        """
        # Prediction loss (Granger-style)
        pred_loss = F.mse_loss(predictions[:, :-1], targets[:, 1:])

        # Sparsity loss (L1 on edges)
        sparse_loss = adj_matrix.abs().mean()

        # DAG constraint (acyclicity) - NOTEARS style
        # tr(e^A) - n should equal 0 for DAG
        n = adj_matrix.size(-1)
        dag_loss = torch.trace(torch.matrix_exp(adj_matrix.mean(0))) - n

        total_loss = (
            self.lambda_pred * pred_loss +
            self.lambda_sparse * sparse_loss +
            self.lambda_dag * dag_loss
        )

        return total_loss, {
            'pred_loss': pred_loss.item(),
            'sparse_loss': sparse_loss.item(),
            'dag_loss': dag_loss.item()
        }


if __name__ == "__main__":
    # Quick test
    batch_size = 32
    seq_len = 100
    n_factors = 6

    model = NeuralCausalDiscovery(n_factors=n_factors)
    x = torch.randn(batch_size, seq_len, n_factors)

    pred, regimes = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Prediction shape: {pred.shape}")
    print(f"Regimes shape: {regimes.shape}")

    # With graph
    pred, regimes, adj, weights = model(x, return_graph=True)
    print(f"Adjacency shape: {adj.shape}")
    print(f"Attention weights shape: {weights.shape}")
