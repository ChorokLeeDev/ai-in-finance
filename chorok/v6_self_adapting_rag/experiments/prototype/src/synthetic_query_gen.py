"""
Synthetic Query Generation for MoE-RAG Training

Generates diverse query types from passages without human labels.
Uses LLM to create queries that would naturally need different attention patterns.
"""

import json
import os
from typing import List, Dict, Optional
from dataclasses import dataclass
import hashlib

# Will use OpenAI for cheap/fast generation
# Can swap for local LLM later


@dataclass
class QueryType:
    """Definition of a query type."""
    name: str
    description: str
    prompt_template: str
    examples: List[str]


# Define query type spectrum
QUERY_TYPES = {
    "factual": QueryType(
        name="factual",
        description="Questions asking what something IS (definitions, facts)",
        prompt_template="""Given this passage, generate a factual question that asks "what is" or "what are" about a key concept.

Passage: {passage}

Generate a natural factual question (just the question, no explanation):""",
        examples=[
            "What is a neural network?",
            "What are the main components of a cell?",
            "What is the capital of France?"
        ]
    ),

    "procedural": QueryType(
        name="procedural",
        description="Questions asking HOW to do something (steps, methods)",
        prompt_template="""Given this passage, generate a how-to question that asks about a process, method, or procedure mentioned.

Passage: {passage}

Generate a natural procedural question (just the question, no explanation):""",
        examples=[
            "How do I create a list in Python?",
            "How can I reset my password?",
            "How do I calculate the area of a circle?"
        ]
    ),

    "causal": QueryType(
        name="causal",
        description="Questions asking WHY something happens (causes, reasons)",
        prompt_template="""Given this passage, generate a "why" question that asks about causes, reasons, or explanations.

Passage: {passage}

Generate a natural causal question (just the question, no explanation):""",
        examples=[
            "Why does water boil at 100°C?",
            "Why do leaves change color in fall?",
            "Why is the sky blue?"
        ]
    ),

    "comparative": QueryType(
        name="comparative",
        description="Questions asking about differences or similarities between things",
        prompt_template="""Given this passage, generate a comparison question that asks about differences or similarities between concepts mentioned.

Passage: {passage}

Generate a natural comparative question (just the question, no explanation):""",
        examples=[
            "What's the difference between a list and a tuple?",
            "How does TCP compare to UDP?",
            "Which is faster, Python or C++?"
        ]
    ),
}


def get_query_types() -> Dict[str, QueryType]:
    """Get all defined query types."""
    return QUERY_TYPES


def generate_synthetic_query_openai(
    passage: str,
    query_type: str,
    model: str = "gpt-3.5-turbo",
    api_key: Optional[str] = None
) -> str:
    """
    Generate a synthetic query using OpenAI API.

    Args:
        passage: Source passage
        query_type: One of QUERY_TYPES keys
        model: OpenAI model to use
        api_key: OpenAI API key (or from env)

    Returns:
        Generated query string
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("pip install openai")

    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")

    client = OpenAI(api_key=api_key)

    qt = QUERY_TYPES[query_type]
    prompt = qt.prompt_template.format(passage=passage[:1500])  # Truncate long passages

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You generate natural questions based on passages. Output only the question, nothing else."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100,
        temperature=0.7
    )

    query = response.choices[0].message.content.strip()

    # Clean up: remove quotes if present
    query = query.strip('"\'')

    return query


def generate_synthetic_query_local(
    passage: str,
    query_type: str,
) -> str:
    """
    Generate synthetic query using templates (no LLM needed).
    Faster but lower quality. Good for initial testing.
    """
    import random

    qt = QUERY_TYPES[query_type]

    # Extract key terms from passage (simple heuristic)
    words = passage.split()
    # Find capitalized words (likely entities) or long words (likely concepts)
    key_terms = [w.strip('.,!?') for w in words
                 if (w[0].isupper() and len(w) > 3) or len(w) > 8][:5]

    if not key_terms:
        key_terms = [words[i] for i in range(0, len(words), len(words)//5 + 1)][:3]

    term = random.choice(key_terms) if key_terms else "this"

    templates = {
        "factual": [
            f"What is {term}?",
            f"What are the characteristics of {term}?",
            f"Can you explain {term}?",
        ],
        "procedural": [
            f"How do I use {term}?",
            f"What are the steps to {term}?",
            f"How can I implement {term}?",
        ],
        "causal": [
            f"Why is {term} important?",
            f"Why does {term} happen?",
            f"What causes {term}?",
        ],
        "comparative": [
            f"How does {term} compare to alternatives?",
            f"What's the difference between {term} and others?",
            f"When should I use {term} vs other options?",
        ],
    }

    return random.choice(templates[query_type])


def generate_queries_for_passage(
    passage: str,
    passage_id: str,
    query_types: List[str] = None,
    use_llm: bool = True,
    model: str = "gpt-3.5-turbo"
) -> List[Dict]:
    """
    Generate queries of each type for a passage.

    Args:
        passage: Source passage text
        passage_id: Unique identifier for passage
        query_types: List of query types to generate (default: all)
        use_llm: Use LLM for generation (vs templates)
        model: LLM model to use

    Returns:
        List of dicts with query info
    """
    query_types = query_types or list(QUERY_TYPES.keys())
    results = []

    for qtype in query_types:
        try:
            if use_llm:
                query = generate_synthetic_query_openai(passage, qtype, model)
            else:
                query = generate_synthetic_query_local(passage, qtype)

            results.append({
                "passage_id": passage_id,
                "query_type": qtype,
                "query": query,
                "passage": passage,
            })
        except Exception as e:
            print(f"Error generating {qtype} query for {passage_id}: {e}")
            continue

    return results


def generate_dataset(
    passages: List[Dict],  # [{"id": str, "text": str}, ...]
    output_path: str,
    query_types: List[str] = None,
    use_llm: bool = True,
    model: str = "gpt-3.5-turbo",
    max_passages: int = None,
    cache_dir: str = None
) -> List[Dict]:
    """
    Generate full synthetic query dataset.

    Args:
        passages: List of passage dicts with id and text
        output_path: Where to save results
        query_types: Query types to generate
        use_llm: Use LLM vs templates
        model: LLM model
        max_passages: Limit number of passages
        cache_dir: Cache individual queries (resume support)

    Returns:
        List of all generated query dicts
    """
    query_types = query_types or list(QUERY_TYPES.keys())

    if max_passages:
        passages = passages[:max_passages]

    all_queries = []
    cache = {}

    # Load cache if exists
    if cache_dir and os.path.exists(os.path.join(cache_dir, "cache.json")):
        with open(os.path.join(cache_dir, "cache.json")) as f:
            cache = json.load(f)

    for i, passage_data in enumerate(passages):
        pid = passage_data["id"]
        ptext = passage_data["text"]

        # Check cache
        cache_key = hashlib.md5(f"{pid}_{ptext[:100]}".encode()).hexdigest()

        if cache_key in cache:
            all_queries.extend(cache[cache_key])
            continue

        # Generate
        queries = generate_queries_for_passage(
            passage=ptext,
            passage_id=pid,
            query_types=query_types,
            use_llm=use_llm,
            model=model
        )

        all_queries.extend(queries)

        # Update cache
        if cache_dir:
            cache[cache_key] = queries

        # Progress
        if (i + 1) % 10 == 0:
            print(f"Generated queries for {i + 1}/{len(passages)} passages")

            # Save intermediate
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
                with open(os.path.join(cache_dir, "cache.json"), "w") as f:
                    json.dump(cache, f)

    # Save final dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_queries, f, indent=2)

    print(f"Saved {len(all_queries)} queries to {output_path}")

    # Print stats
    type_counts = {}
    for q in all_queries:
        type_counts[q["query_type"]] = type_counts.get(q["query_type"], 0) + 1
    print("Query type distribution:")
    for qtype, count in type_counts.items():
        print(f"  {qtype}: {count}")

    return all_queries


def load_synthetic_queries(path: str) -> List[Dict]:
    """Load synthetic query dataset."""
    with open(path) as f:
        return json.load(f)


def split_by_query_type(queries: List[Dict]) -> Dict[str, List[Dict]]:
    """Split queries by type for per-head training."""
    by_type = {}
    for q in queries:
        qtype = q["query_type"]
        if qtype not in by_type:
            by_type[qtype] = []
        by_type[qtype].append(q)
    return by_type


# Example usage
if __name__ == "__main__":
    # Test with a sample passage
    test_passage = """
    Python lists are mutable sequences, typically used to store collections
    of homogeneous items. Lists may be constructed in several ways: using a
    pair of square brackets to denote the empty list [], using square brackets
    with items separated by commas [a, b, c], or using the list() constructor.
    Lists implement all of the common sequence operations.
    """

    print("Testing synthetic query generation...")
    print(f"\nPassage: {test_passage[:100]}...")

    for qtype in QUERY_TYPES:
        # Use local (template) generation for testing
        query = generate_synthetic_query_local(test_passage, qtype)
        print(f"\n{qtype.upper()}: {query}")
