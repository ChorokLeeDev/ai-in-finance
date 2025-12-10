"""
Auto-Chunking

No configuration needed. Sensible defaults that work.
User never has to think about chunk size, overlap, etc.
"""

import os
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class Chunk:
    """A chunk of text with metadata."""
    id: str
    text: str
    source: Optional[str] = None
    start_char: Optional[int] = None
    end_char: Optional[int] = None


def auto_chunk(
    text: str,
    source: Optional[str] = None,
    chunk_size: int = 512,
    overlap: int = 50
) -> List[Chunk]:
    """
    Chunk text into overlapping segments.

    Uses simple fixed-size chunking with overlap.
    Works well for most documents without any tuning.

    Args:
        text: Document text
        source: Source identifier (e.g., filename)
        chunk_size: Characters per chunk (default works well)
        overlap: Overlap between chunks

    Returns:
        List of Chunk objects
    """
    if not text or not text.strip():
        return []

    text = text.strip()
    chunks = []

    # Simple sentence-aware chunking
    # Try to break at sentence boundaries when possible
    sentences = _split_sentences(text)

    current_chunk = ""
    current_start = 0
    chunk_idx = 0

    for sentence in sentences:
        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            # Save current chunk
            chunks.append(Chunk(
                id=f"{source or 'doc'}_{chunk_idx}",
                text=current_chunk.strip(),
                source=source,
                start_char=current_start,
                end_char=current_start + len(current_chunk)
            ))
            chunk_idx += 1

            # Start new chunk with overlap
            overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
            current_chunk = overlap_text + sentence
            current_start = current_start + len(current_chunk) - len(overlap_text) - len(sentence)
        else:
            current_chunk += sentence

    # Don't forget the last chunk
    if current_chunk.strip():
        chunks.append(Chunk(
            id=f"{source or 'doc'}_{chunk_idx}",
            text=current_chunk.strip(),
            source=source,
            start_char=current_start,
            end_char=current_start + len(current_chunk)
        ))

    return chunks


def _split_sentences(text: str) -> List[str]:
    """Simple sentence splitting."""
    import re
    # Split on sentence endings, keeping the delimiter
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p + " " for p in parts if p.strip()]


def chunk_documents(documents: List[str], sources: Optional[List[str]] = None) -> List[Chunk]:
    """
    Chunk multiple documents.

    Args:
        documents: List of document texts
        sources: Optional list of source identifiers

    Returns:
        List of all chunks from all documents
    """
    if sources is None:
        sources = [f"doc_{i}" for i in range(len(documents))]

    all_chunks = []
    for doc, source in zip(documents, sources):
        chunks = auto_chunk(doc, source=source)
        all_chunks.extend(chunks)

    return all_chunks


def load_directory(path: str, extensions: Optional[List[str]] = None) -> List[Dict]:
    """
    Load all documents from a directory.

    Args:
        path: Directory path
        extensions: File extensions to include (default: .txt, .md)

    Returns:
        List of dicts with 'text' and 'source' keys
    """
    extensions = extensions or ['.txt', '.md', '.rst']

    documents = []

    for root, dirs, files in os.walk(path):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        text = f.read()
                    documents.append({
                        'text': text,
                        'source': filepath
                    })
                except Exception as e:
                    print(f"Warning: Could not read {filepath}: {e}")

    return documents


def load_files(paths: List[str]) -> List[Dict]:
    """
    Load specific files.

    Args:
        paths: List of file paths

    Returns:
        List of dicts with 'text' and 'source' keys
    """
    documents = []

    for path in paths:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
            documents.append({
                'text': text,
                'source': path
            })
        except Exception as e:
            print(f"Warning: Could not read {path}: {e}")

    return documents
