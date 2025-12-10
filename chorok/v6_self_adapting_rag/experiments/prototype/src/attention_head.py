"""
MoE-RAG Attention Heads

Simple learned attention heads for RAG context weighting.
Each head can specialize on different query types.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional


class AttentionHead(nn.Module):
    """
    Simple attention head that learns to weight passages given a query.

    Unlike transformer attention, this is a standalone module that:
    - Takes query embedding and passage embeddings
    - Outputs soft attention weights over passages
    """

    def __init__(self, embed_dim: int = 384, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim

        # Project query and keys to attention space
        self.query_proj = nn.Linear(embed_dim, hidden_dim)
        self.key_proj = nn.Linear(embed_dim, hidden_dim)

        # Learnable temperature for softmax sharpness
        self.temperature = nn.Parameter(torch.ones(1))

        self.scale = hidden_dim ** -0.5

    def forward(
        self,
        query_emb: torch.Tensor,
        passage_embs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute attention weights over passages.

        Args:
            query_emb: [D] or [1, D] - single query embedding
            passage_embs: [N, D] - N passage embeddings

        Returns:
            weights: [N] - attention weights (sum to 1)
        """
        # Ensure query is 2D
        if query_emb.dim() == 1:
            query_emb = query_emb.unsqueeze(0)

        # Project
        q = self.query_proj(query_emb)  # [1, H]
        k = self.key_proj(passage_embs)  # [N, H]

        # Attention scores
        scores = torch.matmul(q, k.T) * self.scale  # [1, N]

        # Temperature-scaled softmax
        weights = F.softmax(scores / self.temperature.abs().clamp(min=0.1), dim=-1)

        return weights.squeeze(0)  # [N]

    def get_attention_entropy(
        self,
        query_emb: torch.Tensor,
        passage_embs: torch.Tensor
    ) -> float:
        """
        Compute entropy of attention distribution.
        High entropy = spread attention, Low entropy = focused attention.
        """
        weights = self.forward(query_emb, passage_embs)
        # Add small epsilon to avoid log(0)
        entropy = -torch.sum(weights * torch.log(weights + 1e-10))
        return entropy.item()


class MoEAttention(nn.Module):
    """
    Mixture of Attention Experts.

    Multiple attention heads, each potentially specialized.
    Router determines which heads to activate for each query.
    """

    def __init__(
        self,
        num_heads: int = 4,
        embed_dim: int = 384,
        head_names: Optional[List[str]] = None
    ):
        super().__init__()

        self.num_heads = num_heads
        self.embed_dim = embed_dim

        # Head names for interpretability
        self.head_names = head_names or [f"head_{i}" for i in range(num_heads)]

        # Create attention heads
        self.heads = nn.ModuleList([
            AttentionHead(embed_dim) for _ in range(num_heads)
        ])

        # Router: query -> head weights
        self.router = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, num_heads),
        )

    def route(self, query_emb: torch.Tensor, hard: bool = False) -> torch.Tensor:
        """
        Route query to heads.

        Args:
            query_emb: [D] or [1, D]
            hard: If True, return one-hot (only top head)

        Returns:
            weights: [num_heads] - soft or hard routing weights
        """
        if query_emb.dim() == 1:
            query_emb = query_emb.unsqueeze(0)

        logits = self.router(query_emb)  # [1, num_heads]

        if hard:
            # Hard routing: only top head
            idx = logits.argmax(dim=-1)
            weights = F.one_hot(idx, self.num_heads).float()
        else:
            # Soft routing
            weights = F.softmax(logits, dim=-1)

        return weights.squeeze(0)  # [num_heads]

    def forward(
        self,
        query_emb: torch.Tensor,
        passage_embs: torch.Tensor,
        hard_routing: bool = False,
        return_details: bool = False
    ) -> torch.Tensor:
        """
        Compute weighted attention over passages using MoE.

        Args:
            query_emb: [D] query embedding
            passage_embs: [N, D] passage embeddings
            hard_routing: Use only top head
            return_details: Also return per-head attentions and routing weights

        Returns:
            final_attention: [N] - combined attention weights
            (optional) details: dict with per-head info
        """
        # Get routing weights
        routing_weights = self.route(query_emb, hard=hard_routing)  # [num_heads]

        # Compute attention from each head
        head_attentions = []
        for head in self.heads:
            attn = head(query_emb, passage_embs)  # [N]
            head_attentions.append(attn)

        head_attentions = torch.stack(head_attentions, dim=0)  # [num_heads, N]

        # Weighted combination
        final_attention = torch.einsum('h,hn->n', routing_weights, head_attentions)

        if return_details:
            details = {
                'routing_weights': routing_weights,
                'head_attentions': head_attentions,
                'head_names': self.head_names,
                'top_head': self.head_names[routing_weights.argmax().item()],
                'routing_entropy': self._entropy(routing_weights).item(),
            }
            return final_attention, details

        return final_attention

    def _entropy(self, probs: torch.Tensor) -> torch.Tensor:
        """Compute entropy of probability distribution."""
        return -torch.sum(probs * torch.log(probs + 1e-10))

    def get_head(self, name_or_idx) -> AttentionHead:
        """Get a specific head by name or index."""
        if isinstance(name_or_idx, str):
            idx = self.head_names.index(name_or_idx)
        else:
            idx = name_or_idx
        return self.heads[idx]


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for training attention heads.

    Query should attend more to its source passage than random passages.
    """

    def __init__(self, margin: float = 0.2, temperature: float = 0.1):
        super().__init__()
        self.margin = margin
        self.temperature = temperature

    def forward(
        self,
        head: AttentionHead,
        query_emb: torch.Tensor,
        positive_emb: torch.Tensor,
        negative_embs: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            head: Attention head to train
            query_emb: [D] query embedding
            positive_emb: [D] source passage embedding
            negative_embs: [K, D] random passage embeddings

        Returns:
            loss: scalar
        """
        # Combine positive and negatives
        all_passages = torch.cat([
            positive_emb.unsqueeze(0),  # [1, D]
            negative_embs               # [K, D]
        ], dim=0)  # [K+1, D]

        # Get attention weights
        attention = head(query_emb, all_passages)  # [K+1]

        # Positive is index 0
        positive_score = attention[0]
        negative_scores = attention[1:]

        # InfoNCE-style loss
        logits = torch.cat([
            positive_score.unsqueeze(0),
            negative_scores
        ]) / self.temperature

        labels = torch.zeros(1, dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits.unsqueeze(0), labels)

        return loss


class DiversityRegularizer(nn.Module):
    """
    Regularizer to prevent head collapse while allowing overlap.

    Key insight: We don't want orthogonal heads (can't combine).
    We want specialized but combinable heads.
    """

    def __init__(self, collapse_threshold: float = 0.95):
        super().__init__()
        self.collapse_threshold = collapse_threshold

    def forward(
        self,
        head_attentions: torch.Tensor
    ) -> torch.Tensor:
        """
        Only penalize if heads are TOO similar (near collapse).

        Args:
            head_attentions: [num_heads, N] - attention patterns from each head

        Returns:
            loss: scalar - 0 if diverse enough, positive if too similar
        """
        num_heads = head_attentions.shape[0]

        # Normalize attention patterns
        normed = F.normalize(head_attentions, dim=-1)

        # Pairwise cosine similarity
        similarity_matrix = torch.matmul(normed, normed.T)  # [H, H]

        # Only penalize similarities above threshold (near collapse)
        # Mask diagonal
        mask = 1 - torch.eye(num_heads, device=similarity_matrix.device)
        similarities = similarity_matrix * mask

        # ReLU: only penalize if > threshold
        excess_similarity = F.relu(similarities - self.collapse_threshold)

        # Mean excess similarity
        loss = excess_similarity.sum() / (num_heads * (num_heads - 1))

        return loss


class LoadBalanceLoss(nn.Module):
    """
    Encourage all heads to be used across queries.

    Prevents router from always choosing same head.
    """

    def __init__(self, target_balance: float = None):
        super().__init__()
        self.target_balance = target_balance  # If None, use uniform

    def forward(
        self,
        routing_weights_batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            routing_weights_batch: [B, num_heads] - routing weights for batch

        Returns:
            loss: scalar
        """
        # Average usage across batch
        avg_usage = routing_weights_batch.mean(dim=0)  # [num_heads]

        num_heads = avg_usage.shape[0]
        target = self.target_balance or (1.0 / num_heads)

        # Variance from target
        loss = ((avg_usage - target) ** 2).mean()

        return loss
