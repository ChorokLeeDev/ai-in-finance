"""
Document and Query Encoding

Wraps sentence-transformers for embedding.
User doesn't need to choose - we use a good default.
"""

import torch
from typing import List, Union


class Encoder:
    """
    Embedding encoder with sensible defaults.

    User never sees this - it's an implementation detail.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "MoE-RAG requires sentence-transformers. "
                "Install with: pip install sentence-transformers"
            )

        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        self._model_name = model_name

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> torch.Tensor:
        """
        Encode texts to embeddings.

        Args:
            texts: Single text or list of texts
            batch_size: Batch size for encoding
            show_progress: Show progress bar

        Returns:
            embeddings: [N, D] tensor
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_tensor=True,
            show_progress_bar=show_progress
        )

        return embeddings

    def __repr__(self):
        return f"Encoder({self._model_name}, dim={self.dim})"
