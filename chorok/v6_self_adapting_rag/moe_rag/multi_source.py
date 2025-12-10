#!/usr/bin/env python
"""
Multi-Source RAG with Style Normalization

Key insight: Enterprise knowledge lives in heterogeneous sources:
- Slack threads (conversational, typos, informal)
- Wiki pages (structured, formal)
- Runbooks (technical jargon, steps)
- Email threads (mixed formality)
- PDFs (OCR errors, formatting artifacts)

Each source has a "style" that creates spurious similarity within-source
and reduces cross-source matching. ToneNorm learns to normalize each
source type into a common content space.

This can OUTPERFORM single-embedder approaches when:
1. Content is spread across many source types
2. The "right answer" might be in ANY source type
3. Query style doesn't match answer source style
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class SourceType(Enum):
    """Known source types with characteristic styles."""
    SLACK = "slack"           # Conversational, abbreviations
    WIKI = "wiki"             # Formal, structured
    RUNBOOK = "runbook"       # Technical, step-based
    EMAIL = "email"           # Mixed formality
    PDF = "pdf"               # OCR artifacts, formatting
    CODE = "code"             # Technical, syntax-heavy
    STACKOVERFLOW = "stackoverflow"  # Q&A format
    UNKNOWN = "unknown"


@dataclass
class SourceDocument:
    """Document with source type annotation."""
    text: str
    source_type: SourceType
    doc_id: str
    metadata: Dict = None


class SourceStyleNormalizer(nn.Module):
    """
    Learn a normalization per source type.

    Each source type has its own projection that maps to common content space.
    This is like having multiple InstanceNorm layers, one per "domain".
    """

    def __init__(self, embedding_dim: int = 384, num_source_types: int = 8):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Per-source projections to common space
        self.source_projectors = nn.ModuleDict({
            st.value: nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.LayerNorm(embedding_dim),
            )
            for st in SourceType
        })

        # Query projector (learns to match content space)
        self.query_projector = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def normalize_document(self, emb: torch.Tensor, source_type: SourceType) -> torch.Tensor:
        """Project document to common content space."""
        return self.source_projectors[source_type.value](emb)

    def normalize_query(self, emb: torch.Tensor) -> torch.Tensor:
        """Project query to content space."""
        return self.query_projector(emb)


class SourceTypeDetector:
    """
    Auto-detect source type from text patterns.

    Uses heuristics to classify documents without manual labeling.
    """

    def __init__(self):
        self.patterns = {
            SourceType.SLACK: [
                r"@\w+",              # @mentions
                r"\[\d+:\d+\s*(AM|PM)?\]",  # timestamps
                r"(lol|lmao|omg)",    # informal
                r"thx|pls|btw",       # abbreviations
            ],
            SourceType.WIKI: [
                r"==.*==",            # Wiki headers
                r"\[\[.*\]\]",        # Wiki links
                r"Last updated:",
                r"Tags:",
            ],
            SourceType.RUNBOOK: [
                r"RUNBOOK:",
                r"SEVERITY:",
                r"PRE-REQS:",
                r"STEPS:",
                r"\d+\.",             # Numbered steps
            ],
            SourceType.EMAIL: [
                r"From:.*To:",
                r"Subject:",
                r"RE:|FW:",
                r"Best,|Regards,",
            ],
            SourceType.PDF: [
                r"\[EXTRACTED FROM",
                r"OCR",
                r"Page \d+ of \d+",
                r"\.{3,}",            # OCR artifacts like dots
            ],
            SourceType.STACKOVERFLOW: [
                r"Q:|A:",
                r"\d+ upvotes",
                r"accepted",
                r"```",               # Code blocks
            ],
            SourceType.CODE: [
                r"```\w+",
                r"def |class |function",
                r"import |from |require",
            ],
        }

    def detect(self, text: str) -> SourceType:
        """Detect source type from text patterns."""
        import re

        scores = {st: 0 for st in SourceType}

        for source_type, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    scores[source_type] += 1

        # Return type with most matches
        best_type = max(scores, key=scores.get)
        if scores[best_type] > 0:
            return best_type
        return SourceType.UNKNOWN


class MultiSourceRAG:
    """
    RAG optimized for heterogeneous source types.

    Key innovation: Learns separate normalizations per source type,
    enabling better cross-source retrieval.

    Usage:
        rag = MultiSourceRAG()
        rag.add_documents(docs)
        rag.train()  # Learn source-specific normalizations
        results = rag.retrieve("how do i reset my password")
    """

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        auto_detect_source: bool = True
    ):
        from sentence_transformers import SentenceTransformer

        self.encoder = SentenceTransformer(embedding_model)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()

        self.normalizer = SourceStyleNormalizer(self.embedding_dim)
        self.source_detector = SourceTypeDetector() if auto_detect_source else None

        self.documents: List[SourceDocument] = []
        self.doc_embeddings: Optional[np.ndarray] = None
        self.doc_embeddings_normalized: Optional[np.ndarray] = None

        self.is_trained = False

    def add_documents(
        self,
        texts: List[str],
        source_types: Optional[List[SourceType]] = None,
        doc_ids: Optional[List[str]] = None
    ):
        """Add documents with optional source type annotations."""
        for i, text in enumerate(texts):
            # Auto-detect source type if not provided
            if source_types:
                st = source_types[i]
            elif self.source_detector:
                st = self.source_detector.detect(text)
            else:
                st = SourceType.UNKNOWN

            doc_id = doc_ids[i] if doc_ids else f"doc_{len(self.documents)}"

            self.documents.append(SourceDocument(
                text=text,
                source_type=st,
                doc_id=doc_id,
            ))

        # Compute embeddings
        self.doc_embeddings = self.encoder.encode([d.text for d in self.documents])
        self.is_trained = False

    def _create_cross_source_pairs(self) -> List[Tuple[int, int, int]]:
        """
        Create contrastive training triplets: (anchor, positive, negative).

        - Positive: Same topic, different source type
        - Negative: Different topic, any source type

        This preserves discrimination while aligning cross-source content.
        """
        triplets = []

        # Group by topic (from metadata if available, else use high similarity)
        by_topic = {}
        for i, doc in enumerate(self.documents):
            # Check if doc has topic metadata
            topic = doc.metadata.get("topic") if doc.metadata else None

            if topic is None:
                # Infer topic from highest similarity cluster
                topic = f"cluster_{i}"  # Default: own cluster

            if topic not in by_topic:
                by_topic[topic] = []
            by_topic[topic].append(i)

        # Create triplets
        for topic, indices in by_topic.items():
            if len(indices) < 2:
                continue

            # Get negative indices (different topics)
            negative_indices = []
            for other_topic, other_indices in by_topic.items():
                if other_topic != topic:
                    negative_indices.extend(other_indices)

            if not negative_indices:
                continue

            # For each pair in same topic with different sources, create triplet
            for i, idx_i in enumerate(indices):
                for idx_j in indices[i+1:]:
                    # Only pair across different source types
                    if self.documents[idx_i].source_type != self.documents[idx_j].source_type:
                        # Random negative
                        neg_idx = negative_indices[np.random.randint(len(negative_indices))]
                        triplets.append((idx_i, idx_j, neg_idx))
                        triplets.append((idx_j, idx_i, neg_idx))  # Symmetric

        return triplets

    def _infer_topics(self, similarity_threshold: float = 0.7):
        """Infer topic clusters from embedding similarity."""
        n = len(self.documents)
        visited = [False] * n
        topic_id = 0

        for i in range(n):
            if visited[i]:
                continue

            # Start new cluster
            cluster = [i]
            visited[i] = True

            # Find all similar documents
            for j in range(i + 1, n):
                if visited[j]:
                    continue

                sim = np.dot(self.doc_embeddings[i], self.doc_embeddings[j]) / (
                    np.linalg.norm(self.doc_embeddings[i]) * np.linalg.norm(self.doc_embeddings[j])
                )

                if sim > similarity_threshold:
                    cluster.append(j)
                    visited[j] = True

            # Assign topic to cluster
            for idx in cluster:
                if self.documents[idx].metadata is None:
                    self.documents[idx].metadata = {}
                self.documents[idx].metadata["topic"] = f"topic_{topic_id}"

            topic_id += 1

    def train(self, epochs: int = 50, lr: float = 1e-3, margin: float = 0.3, verbose: bool = True):
        """
        Train source normalizers with contrastive triplet loss.

        Objective:
        - Same topic, different source → should be similar
        - Different topic → should be dissimilar

        Uses triplet margin loss to preserve discrimination.
        """
        if len(self.documents) < 2:
            if verbose:
                print("Not enough documents to train")
            return

        # Infer topics if not provided
        self._infer_topics(similarity_threshold=0.6)

        # Create contrastive triplets
        triplets = self._create_cross_source_pairs()

        if verbose:
            # Show source distribution
            source_counts = {}
            topic_counts = {}
            for doc in self.documents:
                st = doc.source_type.value
                source_counts[st] = source_counts.get(st, 0) + 1
                topic = doc.metadata.get("topic") if doc.metadata else "unknown"
                topic_counts[topic] = topic_counts.get(topic, 0) + 1

            print(f"Source distribution: {source_counts}")
            print(f"Topic distribution: {topic_counts}")
            print(f"Training with {len(triplets)} contrastive triplets")

        if len(triplets) < 2:
            if verbose:
                print("Not enough triplets. Need cross-source same-topic pairs.")
            self._normalize_documents()
            return

        optimizer = torch.optim.Adam(self.normalizer.parameters(), lr=lr)
        triplet_loss_fn = nn.TripletMarginLoss(margin=margin)

        doc_emb_tensor = torch.tensor(self.doc_embeddings, dtype=torch.float32)

        for epoch in range(epochs):
            total_loss = 0
            np.random.shuffle(triplets)

            for anchor_idx, pos_idx, neg_idx in triplets:
                anchor_doc = self.documents[anchor_idx]
                pos_doc = self.documents[pos_idx]
                neg_doc = self.documents[neg_idx]

                # Get embeddings
                anchor_emb = doc_emb_tensor[anchor_idx].unsqueeze(0)
                pos_emb = doc_emb_tensor[pos_idx].unsqueeze(0)
                neg_emb = doc_emb_tensor[neg_idx].unsqueeze(0)

                # Normalize by source type
                anchor_norm = self.normalizer.normalize_document(anchor_emb, anchor_doc.source_type)
                pos_norm = self.normalizer.normalize_document(pos_emb, pos_doc.source_type)
                neg_norm = self.normalizer.normalize_document(neg_emb, neg_doc.source_type)

                # L2 normalize for cosine-like distance
                anchor_norm = nn.functional.normalize(anchor_norm, dim=-1)
                pos_norm = nn.functional.normalize(pos_norm, dim=-1)
                neg_norm = nn.functional.normalize(neg_norm, dim=-1)

                # Triplet loss: anchor closer to positive than negative
                loss = triplet_loss_fn(anchor_norm, pos_norm, neg_norm)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if verbose and epoch % 10 == 0:
                avg_loss = total_loss / len(triplets)
                print(f"  Epoch {epoch}: triplet_loss = {avg_loss:.4f}")

        self._normalize_documents()
        self.is_trained = True

        if verbose:
            print("Training complete")

    def _normalize_documents(self):
        """Apply learned normalization to all documents."""
        normalized = []

        with torch.no_grad():
            for i, doc in enumerate(self.documents):
                emb = torch.tensor(self.doc_embeddings[i], dtype=torch.float32).unsqueeze(0)
                norm_emb = self.normalizer.normalize_document(emb, doc.source_type)
                normalized.append(norm_emb.squeeze(0).numpy())

        self.doc_embeddings_normalized = np.array(normalized)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        source_filter: Optional[List[SourceType]] = None
    ) -> List[Dict]:
        """
        Retrieve documents for query.

        Args:
            query: Search query
            top_k: Number of results
            source_filter: Only return from these source types

        Returns:
            List of results with text, score, source_type, doc_id
        """
        query_emb = self.encoder.encode([query])[0]

        if self.is_trained:
            with torch.no_grad():
                q_tensor = torch.tensor(query_emb, dtype=torch.float32).unsqueeze(0)
                query_emb = self.normalizer.normalize_query(q_tensor).squeeze(0).numpy()
            doc_emb = self.doc_embeddings_normalized
        else:
            doc_emb = self.doc_embeddings

        # Cosine similarity
        query_norm = query_emb / np.linalg.norm(query_emb)
        doc_norms = doc_emb / np.linalg.norm(doc_emb, axis=1, keepdims=True)

        similarities = doc_norms @ query_norm

        # Apply source filter
        if source_filter:
            for i, doc in enumerate(self.documents):
                if doc.source_type not in source_filter:
                    similarities[i] = -1

        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices):
            doc = self.documents[idx]
            results.append({
                "text": doc.text,
                "score": float(similarities[idx]),
                "rank": rank + 1,
                "source_type": doc.source_type.value,
                "doc_id": doc.doc_id,
            })

        return results


# =============================================================================
# Convenience function
# =============================================================================

def create_multi_source_rag(
    documents: List[Dict],
    text_key: str = "text",
    source_key: str = "source_type",
    id_key: str = "id"
) -> MultiSourceRAG:
    """
    Create MultiSourceRAG from list of document dicts.

    Args:
        documents: List of dicts with text and optional source_type
        text_key: Key for document text
        source_key: Key for source type (optional)
        id_key: Key for document ID (optional)
    """
    rag = MultiSourceRAG()

    texts = [d[text_key] for d in documents]
    doc_ids = [d.get(id_key, f"doc_{i}") for i, d in enumerate(documents)]

    source_types = None
    if source_key in documents[0]:
        source_types = [
            SourceType(d[source_key]) if isinstance(d[source_key], str)
            else d[source_key]
            for d in documents
        ]

    rag.add_documents(texts, source_types, doc_ids)

    return rag
