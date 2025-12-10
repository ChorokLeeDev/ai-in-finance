#!/usr/bin/env python
"""
ToneNorm: Instance Normalization for Text Embeddings

Key insight: Informal/conversational text has "tone features" that create
spurious similarity with informal queries. We need to normalize these away,
keeping only semantic content - like how InstanceNorm separates style from content.

Approach:
1. Learn a "tone subspace" from paired formal/informal text
2. Project embeddings onto the orthogonal complement (content space)
3. Compare in content space for retrieval
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass


@dataclass
class ToneNormConfig:
    """Configuration for ToneNorm."""
    tone_dims: int = 32  # Dimensions to attribute to tone
    method: str = "learned"  # "learned", "statistical", or "adversarial"


class ToneNorm(nn.Module):
    """
    Normalize away tone/style from text embeddings.

    Like InstanceNorm for images, but for text:
    - Images: InstanceNorm removes style (texture, color distribution)
    - Text: ToneNorm removes tone (formality, conversational markers)

    Methods:
    1. Statistical: Remove dimensions with high variance across tone pairs
    2. Learned: Learn a projection that maximizes content similarity
    3. Adversarial: Train to fool a tone classifier
    """

    def __init__(
        self,
        embedding_dim: int = 384,
        tone_dims: int = 32,
        method: str = "statistical"
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.tone_dims = tone_dims
        self.method = method

        # Tone subspace basis vectors (learned or computed)
        self.register_buffer(
            "tone_basis",
            torch.zeros(tone_dims, embedding_dim)
        )

        # For learned method: projection network
        if method == "learned":
            self.content_projector = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, embedding_dim),
            )

        # For adversarial method: tone classifier to fool
        if method == "adversarial":
            self.tone_classifier = nn.Sequential(
                nn.Linear(embedding_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 2),  # formal vs informal
            )

        self.is_fitted = False

    def fit_statistical(
        self,
        formal_embeddings: torch.Tensor,
        informal_embeddings: torch.Tensor
    ):
        """
        Fit by finding dimensions that vary most between formal/informal.

        Intuition: If the same content expressed formally vs informally
        differs mainly in certain dimensions, those are "tone dimensions".
        """
        # Compute difference vectors (tone signal)
        diff = informal_embeddings - formal_embeddings  # [N, D]

        # PCA on differences to find tone subspace
        diff_centered = diff - diff.mean(dim=0)

        # SVD to get principal tone directions
        U, S, Vh = torch.linalg.svd(diff_centered, full_matrices=False)

        # Top-k directions are the tone subspace
        self.tone_basis = Vh[:self.tone_dims]  # [tone_dims, D]

        self.is_fitted = True

    def fit_learned(
        self,
        formal_embeddings: torch.Tensor,
        informal_embeddings: torch.Tensor,
        epochs: int = 100,
        lr: float = 1e-3
    ):
        """
        Learn a projection that makes formal/informal pairs similar.

        Objective: proj(formal) ≈ proj(informal) for same content
        """
        optimizer = torch.optim.Adam(self.content_projector.parameters(), lr=lr)

        for epoch in range(epochs):
            # Project both
            formal_proj = self.content_projector(formal_embeddings)
            informal_proj = self.content_projector(informal_embeddings)

            # Normalize
            formal_proj = torch.nn.functional.normalize(formal_proj, dim=-1)
            informal_proj = torch.nn.functional.normalize(informal_proj, dim=-1)

            # Loss: paired samples should be similar
            similarity = (formal_proj * informal_proj).sum(dim=-1)
            loss = -similarity.mean()  # Maximize similarity

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.is_fitted = True

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Normalize away tone, keeping content.

        Args:
            embeddings: [batch_size, embedding_dim]

        Returns:
            content_embeddings: [batch_size, embedding_dim] with tone removed
        """
        if self.method == "learned" and hasattr(self, 'content_projector'):
            return self.content_projector(embeddings)

        elif self.method == "statistical" and self.is_fitted:
            # Project out the tone subspace
            # content = emb - sum_i (emb · tone_i) * tone_i
            tone_components = embeddings @ self.tone_basis.T  # [B, tone_dims]
            tone_reconstruction = tone_components @ self.tone_basis  # [B, D]
            content = embeddings - tone_reconstruction
            return content

        else:
            # Not fitted, return as-is
            return embeddings

    def normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """NumPy interface for normalize."""
        with torch.no_grad():
            emb_tensor = torch.tensor(embeddings, dtype=torch.float32)
            normalized = self.forward(emb_tensor)
            return normalized.numpy()


class ToneNormRAG:
    """
    RAG with ToneNorm - normalize tone before similarity computation.

    Usage:
        rag = ToneNormRAG.from_texts(documents)
        rag.fit_tone_norm(formal_texts, informal_texts)  # Optional calibration
        results = rag.retrieve("how do i reset my pwd??")
    """

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        tone_dims: int = 32,
        method: str = "statistical"
    ):
        from sentence_transformers import SentenceTransformer

        self.encoder = SentenceTransformer(embedding_model)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()

        self.tone_norm = ToneNorm(
            embedding_dim=self.embedding_dim,
            tone_dims=tone_dims,
            method=method
        )

        self.documents: List[str] = []
        self.doc_embeddings: Optional[np.ndarray] = None
        self.doc_embeddings_normalized: Optional[np.ndarray] = None

    @classmethod
    def from_texts(cls, texts: List[str], **kwargs) -> "ToneNormRAG":
        """Create RAG from list of texts."""
        rag = cls(**kwargs)
        rag.index(texts)
        return rag

    def index(self, texts: List[str]):
        """Index documents."""
        self.documents = texts
        self.doc_embeddings = self.encoder.encode(texts)
        # Will be set after fit_tone_norm
        self.doc_embeddings_normalized = None

    def fit_tone_norm(
        self,
        content_pairs: List[Tuple[str, str]],
        epochs: int = 100
    ):
        """
        Fit the tone normalizer with formal/informal pairs.

        Args:
            content_pairs: List of (formal_text, informal_text) pairs
                          expressing the same content differently
        """
        formal_texts = [p[0] for p in content_pairs]
        informal_texts = [p[1] for p in content_pairs]

        formal_emb = torch.tensor(self.encoder.encode(formal_texts))
        informal_emb = torch.tensor(self.encoder.encode(informal_texts))

        if self.tone_norm.method == "statistical":
            self.tone_norm.fit_statistical(formal_emb, informal_emb)
        elif self.tone_norm.method == "learned":
            self.tone_norm.fit_learned(formal_emb, informal_emb, epochs=epochs)

        # Re-normalize document embeddings
        self.doc_embeddings_normalized = self.tone_norm.normalize(self.doc_embeddings)

    def auto_fit_tone_norm(self):
        """
        Auto-fit using synthetic formal/informal pairs.

        Uses common patterns to generate training data.
        """
        # Synthetic pairs covering common tone variations
        pairs = [
            # Question formality
            ("How do I reset my password?", "how do i reset my password"),
            ("What is the API rate limit?", "whats the api limit"),
            ("How can I contact support?", "how do i contact support??"),

            # Request formality
            ("Please provide the pricing information.", "give me pricing info"),
            ("I would like to cancel my subscription.", "i want to cancel"),
            ("Could you explain the refund policy?", "explain refund policy pls"),

            # Statement formality
            ("The system is not working correctly.", "system not working"),
            ("I am experiencing an error.", "im getting an error"),
            ("The login page is not loading.", "login page wont load"),

            # Urgency markers
            ("I need assistance with my account.", "need help with account asap!!"),
            ("There is a problem with billing.", "billing problem urgent pls help"),
            ("The service is unavailable.", "service down!!!"),

            # Typos and shortcuts
            ("password", "pasword"),
            ("authentication", "auth"),
            ("configuration", "config"),
            ("information", "info"),
            ("please", "pls"),
            ("thanks", "thx"),

            # Conversational noise
            ("Reset password procedure", "hey anyone know how to reset password?"),
            ("API documentation", "where are the api docs lol"),
            ("Contact support team", "who do i contact for help"),
        ]

        self.fit_tone_norm(pairs)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        use_tone_norm: bool = True
    ) -> List[Dict]:
        """
        Retrieve documents for query.

        Args:
            query: Search query
            top_k: Number of results
            use_tone_norm: Whether to apply tone normalization

        Returns:
            List of {text, score, rank} dicts
        """
        query_emb = self.encoder.encode([query])[0]

        if use_tone_norm and self.tone_norm.is_fitted:
            query_emb = self.tone_norm.normalize(query_emb.reshape(1, -1))[0]
            doc_emb = self.doc_embeddings_normalized
        else:
            doc_emb = self.doc_embeddings

        # Cosine similarity
        query_norm = query_emb / np.linalg.norm(query_emb)
        doc_norms = doc_emb / np.linalg.norm(doc_emb, axis=1, keepdims=True)

        similarities = doc_norms @ query_norm
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices):
            results.append({
                "text": self.documents[idx],
                "score": float(similarities[idx]),
                "rank": rank + 1,
            })

        return results


# =============================================================================
# Synthetic Tone Pair Generator
# =============================================================================

def generate_tone_pairs(
    base_texts: List[str],
    num_variations: int = 3
) -> List[Tuple[str, str]]:
    """
    Generate formal/informal pairs from base texts.

    Applies various "informalization" transforms.
    """
    import random

    transforms = [
        # Remove punctuation
        lambda s: s.replace(".", "").replace(",", "").replace("?", ""),
        # Lowercase
        lambda s: s.lower(),
        # Add urgency
        lambda s: s + " asap" if random.random() > 0.5 else s + "!!",
        # Add filler
        lambda s: "hey " + s if random.random() > 0.5 else s + " lol",
        # Common typos
        lambda s: s.replace("password", "pasword").replace("the", "teh"),
        # Abbreviations
        lambda s: s.replace("please", "pls").replace("thanks", "thx").replace("information", "info"),
    ]

    pairs = []
    for text in base_texts:
        formal = text

        for _ in range(num_variations):
            informal = formal
            # Apply 2-4 random transforms
            for transform in random.sample(transforms, random.randint(2, 4)):
                informal = transform(informal)
            pairs.append((formal, informal))

    return pairs
