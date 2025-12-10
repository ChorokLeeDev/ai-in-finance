"""
Synthetic Query Generation

Generates diverse queries from passages for self-supervised training.
No labels needed - the passage IS the ground truth.
"""

import random
from typing import List, Dict
from dataclasses import dataclass


@dataclass
class QueryType:
    name: str
    templates: List[str]


# Query types that cover different retrieval needs
QUERY_TYPES = [
    QueryType(
        name="factual",
        templates=[
            "What is {term}?",
            "Define {term}",
            "Explain {term}",
            "What does {term} mean?",
        ]
    ),
    QueryType(
        name="procedural",
        templates=[
            "How do I use {term}?",
            "How to {term}?",
            "Steps for {term}",
            "How can I {term}?",
        ]
    ),
    QueryType(
        name="causal",
        templates=[
            "Why is {term} important?",
            "Why use {term}?",
            "What causes {term}?",
            "Why does {term} happen?",
        ]
    ),
    QueryType(
        name="comparative",
        templates=[
            "How does {term} compare?",
            "{term} vs alternatives",
            "When to use {term}?",
            "Difference between {term} and others",
        ]
    ),
]


def extract_key_terms(text: str, max_terms: int = 5) -> List[str]:
    """
    Extract key terms from text for query generation.

    Simple heuristic: capitalized words and long words.
    """
    words = text.split()

    # Find potentially important terms
    candidates = []
    for word in words:
        clean = word.strip('.,!?()[]{}":;')
        if len(clean) < 3:
            continue

        # Capitalized (likely entity/concept)
        if clean[0].isupper() and len(clean) > 3:
            candidates.append(clean)
        # Long word (likely technical term)
        elif len(clean) > 8:
            candidates.append(clean.lower())

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for c in candidates:
        if c.lower() not in seen:
            seen.add(c.lower())
            unique.append(c)

    # If no candidates, fall back to random content words
    if not unique:
        content_words = [w.strip('.,!?()[]{}":;') for w in words
                        if len(w.strip('.,!?()[]{}":;')) > 4]
        unique = content_words[:max_terms]

    return unique[:max_terms]


def generate_queries_for_chunk(
    chunk_id: str,
    chunk_text: str,
    queries_per_type: int = 1
) -> List[Dict]:
    """
    Generate synthetic queries for a chunk.

    Args:
        chunk_id: Chunk identifier
        chunk_text: Chunk text
        queries_per_type: How many queries per type

    Returns:
        List of query dicts
    """
    terms = extract_key_terms(chunk_text)
    if not terms:
        return []

    queries = []

    for query_type in QUERY_TYPES:
        for _ in range(queries_per_type):
            term = random.choice(terms)
            template = random.choice(query_type.templates)
            query_text = template.format(term=term)

            queries.append({
                'query': query_text,
                'chunk_id': chunk_id,
                'query_type': query_type.name,
            })

    return queries


def generate_training_queries(
    chunks: List[Dict],
    queries_per_type: int = 1
) -> List[Dict]:
    """
    Generate training queries for all chunks.

    Args:
        chunks: List of chunk dicts with 'id' and 'text'
        queries_per_type: Queries per type per chunk

    Returns:
        List of all generated queries
    """
    all_queries = []

    for chunk in chunks:
        chunk_id = chunk.get('id', chunk.get('chunk_id', 'unknown'))
        chunk_text = chunk.get('text', '')

        queries = generate_queries_for_chunk(
            chunk_id=chunk_id,
            chunk_text=chunk_text,
            queries_per_type=queries_per_type
        )
        all_queries.extend(queries)

    return all_queries


def get_query_type_names() -> List[str]:
    """Get list of query type names."""
    return [qt.name for qt in QUERY_TYPES]
