"""
MoE Attention Core

Mixture-of-Experts attention that replaces retrieval + reranking.
Each head specializes on different query patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Tuple


class AttentionHead(nn.Module):
    """Single attention head that learns to weight passages."""

    def __init__(self, embed_dim: int = 384, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim

        self.query_proj = nn.Linear(embed_dim, hidden_dim)
        self.key_proj = nn.Linear(embed_dim, hidden_dim)
        self.temperature = nn.Parameter(torch.ones(1))
        self.scale = hidden_dim ** -0.5

    def forward(self, query_emb: torch.Tensor, passage_embs: torch.Tensor) -> torch.Tensor:
        """
        Compute attention weights over passages.

        Args:
            query_emb: [D] or [1, D] query embedding
            passage_embs: [N, D] passage embeddings

        Returns:
            weights: [N] attention weights (sum to 1)
        """
        if query_emb.dim() == 1:
            query_emb = query_emb.unsqueeze(0)

        q = self.query_proj(query_emb)
        k = self.key_proj(passage_embs)

        scores = torch.matmul(q, k.T) * self.scale
        weights = F.softmax(scores / self.temperature.abs().clamp(min=0.1), dim=-1)

        return weights.squeeze(0)


class MoEAttention(nn.Module):
    """
    Mixture of Attention Experts.

    Multiple heads, each specialized. Router selects which to use.
    This replaces: retrieval + reranking + prompt weighting.
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
        self.head_names = head_names or [f"head_{i}" for i in range(num_heads)]

        # Attention heads
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

    def forward(
        self,
        query_emb: torch.Tensor,
        passage_embs: torch.Tensor,
        return_details: bool = False
    ) -> torch.Tensor:
        """
        Compute weighted attention over passages.

        Args:
            query_emb: [D] query embedding
            passage_embs: [N, D] passage embeddings
            return_details: Return per-head info

        Returns:
            attention: [N] combined attention weights
            (optional) details: dict with routing info
        """
        if query_emb.dim() == 1:
            query_emb = query_emb.unsqueeze(0)

        # Route
        routing_logits = self.router(query_emb)
        routing_weights = F.softmax(routing_logits, dim=-1).squeeze(0)

        # Get attention from each head
        head_attentions = torch.stack([
            head(query_emb, passage_embs) for head in self.heads
        ])  # [num_heads, N]

        # Weighted combination
        final_attention = torch.einsum('h,hn->n', routing_weights, head_attentions)

        if return_details:
            top_idx = routing_weights.argmax().item()
            return final_attention, {
                'routing_weights': routing_weights,
                'head_attentions': head_attentions,
                'top_head': self.head_names[top_idx],
                'top_head_weight': routing_weights[top_idx].item(),
            }

        return final_attention


class ContrastiveLoss(nn.Module):
    """Query should attend more to source passage than random passages."""

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        head: AttentionHead,
        query_emb: torch.Tensor,
        positive_emb: torch.Tensor,
        negative_embs: torch.Tensor
    ) -> torch.Tensor:
        all_passages = torch.cat([positive_emb.unsqueeze(0), negative_embs], dim=0)
        attention = head(query_emb, all_passages)

        logits = torch.log(attention + 1e-10) / self.temperature
        labels = torch.zeros(1, dtype=torch.long, device=logits.device)

        return F.cross_entropy(logits.unsqueeze(0), labels)


class DiversityLoss(nn.Module):
    """Prevent heads from collapsing to same pattern."""

    def __init__(self, threshold: float = 0.95):
        super().__init__()
        self.threshold = threshold

    def forward(self, head_attentions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            head_attentions: [num_heads, N]
        """
        normed = F.normalize(head_attentions, dim=-1)
        similarity = torch.matmul(normed, normed.T)

        num_heads = head_attentions.shape[0]
        mask = 1 - torch.eye(num_heads, device=similarity.device)

        excess = F.relu(similarity * mask - self.threshold)
        return excess.sum() / (num_heads * (num_heads - 1))
