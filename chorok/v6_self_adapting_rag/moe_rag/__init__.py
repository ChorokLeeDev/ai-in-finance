"""
MoE-RAG: Replace RAG Pipelines with Learned Attention

No chunking config. No embedding selection. No vector DB. No prompt engineering.
Just train and use.

Usage:
    from moe_rag import MoERAG

    rag = MoERAG.from_directory("./docs/")
    rag.train()
    results = rag.retrieve("your question here")

CLI:
    $ moe-rag train ./docs/ -o model.pt
    $ moe-rag retrieve model.pt "your question"
"""

from .model import MoERAG

__version__ = "0.1.0"
__all__ = ["MoERAG"]
