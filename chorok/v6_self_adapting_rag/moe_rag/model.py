"""
MoERAG: The Main Interface

This is the only thing users need to interact with.
Everything else is implementation details.

Usage:
    rag = MoERAG.from_directory("./docs/")
    rag.train()
    results = rag.retrieve("How do I reset my password?")
"""

import os
import torch
from typing import List, Dict, Optional, Union
from dataclasses import dataclass

from .attention import MoEAttention
from .encoder import Encoder
from .chunker import auto_chunk, chunk_documents, load_directory, load_files, Chunk
from .query_gen import generate_training_queries, get_query_type_names
from .trainer import Trainer


@dataclass
class RetrievalResult:
    """A single retrieval result."""
    text: str
    score: float
    chunk_id: str
    source: Optional[str] = None
    head_used: Optional[str] = None


class MoERAG:
    """
    MoE-RAG: Replace RAG pipelines with learned attention.

    No chunking config. No embedding selection. No vector DB.
    Just train and use.
    """

    def __init__(
        self,
        num_heads: int = 4,
        device: str = "cpu"
    ):
        """
        Initialize MoE-RAG.

        Args:
            num_heads: Number of attention heads (default: 4, one per query type)
            device: Device to use (cpu/cuda)
        """
        self.device = device
        self.num_heads = num_heads

        # Initialize encoder (embedding model)
        self.encoder = Encoder()

        # Initialize MoE attention
        self.moe = MoEAttention(
            num_heads=num_heads,
            embed_dim=self.encoder.dim,
            head_names=get_query_type_names()[:num_heads]
        )

        # Storage
        self.chunks: List[Dict] = []
        self.chunk_embeddings: Optional[torch.Tensor] = None

        # State
        self._trained = False

    @classmethod
    def from_directory(cls, path: str, **kwargs) -> "MoERAG":
        """
        Create MoE-RAG from a directory of documents.

        Args:
            path: Path to directory containing documents
            **kwargs: Additional arguments for MoERAG

        Returns:
            MoERAG instance with documents loaded
        """
        rag = cls(**kwargs)
        rag.add_directory(path)
        return rag

    @classmethod
    def from_texts(cls, texts: List[str], **kwargs) -> "MoERAG":
        """
        Create MoE-RAG from a list of texts.

        Args:
            texts: List of document texts
            **kwargs: Additional arguments for MoERAG

        Returns:
            MoERAG instance with documents loaded
        """
        rag = cls(**kwargs)
        rag.add_texts(texts)
        return rag

    @classmethod
    def from_files(cls, paths: List[str], **kwargs) -> "MoERAG":
        """
        Create MoE-RAG from specific files.

        Args:
            paths: List of file paths
            **kwargs: Additional arguments for MoERAG

        Returns:
            MoERAG instance with documents loaded
        """
        rag = cls(**kwargs)
        rag.add_files(paths)
        return rag

    def add_directory(self, path: str) -> None:
        """Add all documents from a directory."""
        documents = load_directory(path)
        print(f"Found {len(documents)} documents in {path}")

        for doc in documents:
            chunks = auto_chunk(doc['text'], source=doc['source'])
            for chunk in chunks:
                self.chunks.append({
                    'id': chunk.id,
                    'text': chunk.text,
                    'source': chunk.source,
                })

        print(f"Total chunks: {len(self.chunks)}")

    def add_texts(self, texts: List[str], sources: Optional[List[str]] = None) -> None:
        """Add texts directly."""
        sources = sources or [f"text_{i}" for i in range(len(texts))]

        for text, source in zip(texts, sources):
            chunks = auto_chunk(text, source=source)
            for chunk in chunks:
                self.chunks.append({
                    'id': chunk.id,
                    'text': chunk.text,
                    'source': chunk.source,
                })

        print(f"Added {len(texts)} texts, total chunks: {len(self.chunks)}")

    def add_files(self, paths: List[str]) -> None:
        """Add specific files."""
        documents = load_files(paths)

        for doc in documents:
            chunks = auto_chunk(doc['text'], source=doc['source'])
            for chunk in chunks:
                self.chunks.append({
                    'id': chunk.id,
                    'text': chunk.text,
                    'source': chunk.source,
                })

        print(f"Added {len(documents)} files, total chunks: {len(self.chunks)}")

    def train(
        self,
        epochs: int = 10,
        verbose: bool = True
    ) -> Dict:
        """
        Train MoE-RAG on loaded documents.

        Self-supervised training - no labels needed.

        Args:
            epochs: Number of training epochs
            verbose: Print progress

        Returns:
            Training results
        """
        if not self.chunks:
            raise ValueError("No documents loaded. Use add_directory(), add_texts(), or add_files() first.")

        if verbose:
            print(f"\nTraining MoE-RAG on {len(self.chunks)} chunks...")

        # Generate synthetic queries
        if verbose:
            print("Generating synthetic queries...")

        queries = generate_training_queries(self.chunks, queries_per_type=1)
        if verbose:
            print(f"Generated {len(queries)} queries")

        # Prepare trainer
        trainer = Trainer(self.moe, self.encoder, self.device)

        # Prepare data
        data = trainer.prepare_data(self.chunks, queries)

        # Store chunk embeddings for retrieval
        self.chunk_embeddings = data['chunk_embeddings']

        # Train
        results = trainer.train(
            data,
            head_epochs=epochs,
            router_epochs=epochs * 2,
            finetune_epochs=epochs // 2,
            verbose=verbose
        )

        self._trained = True

        if verbose:
            print("\nTraining complete!")

        return results

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        return_scores: bool = True
    ) -> List[Union[RetrievalResult, str]]:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: The query string
            top_k: Number of results to return
            return_scores: Return RetrievalResult objects (True) or just text (False)

        Returns:
            List of results
        """
        if not self._trained:
            raise ValueError("Model not trained. Call train() first.")

        # Encode query
        query_emb = self.encoder.encode(query).to(self.device)

        # Get MoE attention over all chunks
        self.moe.eval()
        with torch.no_grad():
            chunk_embs = self.chunk_embeddings.to(self.device)
            attention, details = self.moe(query_emb, chunk_embs, return_details=True)

        # Get top-k
        top_k = min(top_k, len(self.chunks))
        scores, indices = torch.topk(attention, top_k)

        results = []
        for score, idx in zip(scores.tolist(), indices.tolist()):
            chunk = self.chunks[idx]

            if return_scores:
                results.append(RetrievalResult(
                    text=chunk['text'],
                    score=score,
                    chunk_id=chunk['id'],
                    source=chunk.get('source'),
                    head_used=details['top_head'],
                ))
            else:
                results.append(chunk['text'])

        return results

    def save(self, path: str) -> None:
        """
        Save trained model.

        Args:
            path: Path to save to (e.g., "my_rag.pt")
        """
        if not self._trained:
            raise ValueError("Model not trained. Call train() first.")

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        torch.save({
            'moe_state': self.moe.state_dict(),
            'chunks': self.chunks,
            'chunk_embeddings': self.chunk_embeddings,
            'num_heads': self.num_heads,
            'embed_dim': self.encoder.dim,
        }, path)

        print(f"Saved to {path}")

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "MoERAG":
        """
        Load a trained model.

        Args:
            path: Path to saved model
            device: Device to use

        Returns:
            Loaded MoERAG instance
        """
        checkpoint = torch.load(path, map_location=device)

        rag = cls(
            num_heads=checkpoint['num_heads'],
            device=device
        )

        rag.moe.load_state_dict(checkpoint['moe_state'])
        rag.chunks = checkpoint['chunks']
        rag.chunk_embeddings = checkpoint['chunk_embeddings']
        rag._trained = True

        print(f"Loaded model with {len(rag.chunks)} chunks")

        return rag

    def __repr__(self):
        status = "trained" if self._trained else "untrained"
        return f"MoERAG(chunks={len(self.chunks)}, heads={self.num_heads}, {status})"
