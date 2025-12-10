"""
Multi-Strategy Passage Generator

Creates passages that reward different retrieval strategies:
1. Keyword-optimized: Dense with searchable terms
2. Semantic-optimized: Descriptive, conceptual

This tests whether MoE heads can specialize on different strategies.
"""

import json
import os
import random
from typing import List, Dict, Tuple


# Topics with both keyword and semantic descriptions
TOPICS = {
    "python_list": {
        "keywords": ["list", "append", "extend", "insert", "remove", "pop", "index", "slice", "mutable", "sequence"],
        "concepts": ["ordered collection", "dynamic sizing", "element access", "modification", "iteration"],
    },
    "http_methods": {
        "keywords": ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS", "REST", "API", "endpoint"],
        "concepts": ["resource retrieval", "data submission", "state modification", "client-server", "stateless"],
    },
    "database_index": {
        "keywords": ["B-tree", "hash", "index", "primary key", "foreign key", "clustered", "query optimization", "cardinality"],
        "concepts": ["faster lookups", "trade-off with writes", "data organization", "search efficiency", "storage overhead"],
    },
    "neural_network": {
        "keywords": ["layer", "neuron", "weights", "bias", "activation", "ReLU", "sigmoid", "backpropagation", "gradient"],
        "concepts": ["learning patterns", "feature extraction", "nonlinear transformation", "optimization", "generalization"],
    },
    "git_workflow": {
        "keywords": ["commit", "branch", "merge", "rebase", "pull", "push", "checkout", "stash", "HEAD", "origin"],
        "concepts": ["version tracking", "collaboration", "history management", "parallel development", "code integration"],
    },
    "encryption": {
        "keywords": ["AES", "RSA", "SHA", "key", "cipher", "plaintext", "ciphertext", "symmetric", "asymmetric", "hash"],
        "concepts": ["data protection", "confidentiality", "integrity verification", "secure communication", "authentication"],
    },
    "docker_container": {
        "keywords": ["image", "container", "Dockerfile", "layer", "volume", "port", "network", "compose", "registry", "daemon"],
        "concepts": ["isolation", "reproducibility", "lightweight virtualization", "dependency management", "deployment"],
    },
    "machine_learning": {
        "keywords": ["training", "validation", "test", "overfitting", "regularization", "hyperparameter", "cross-validation", "loss"],
        "concepts": ["pattern recognition", "generalization", "model selection", "bias-variance", "performance evaluation"],
    },
    "sql_join": {
        "keywords": ["INNER", "LEFT", "RIGHT", "OUTER", "JOIN", "ON", "WHERE", "table", "foreign key", "cartesian"],
        "concepts": ["combining tables", "relationship traversal", "data integration", "query composition", "relational algebra"],
    },
    "api_authentication": {
        "keywords": ["OAuth", "JWT", "token", "bearer", "API key", "refresh", "scope", "authorization", "header", "expiry"],
        "concepts": ["identity verification", "access control", "session management", "security delegation", "permission scoping"],
    },
}


def generate_keyword_passage(topic: str, topic_data: dict) -> str:
    """
    Generate keyword-dense passage.

    Strategy: List terms, use exact terminology, structured format.
    Best retrieved by: Term matching, keyword overlap.
    """
    keywords = topic_data["keywords"]

    templates = [
        # Template 1: Definition + keyword list
        f"{topic.replace('_', ' ').title()}: Key terms include {', '.join(keywords[:5])}. "
        f"Related concepts: {', '.join(keywords[5:])}. "
        f"Common operations involve {keywords[0]}, {keywords[1]}, and {keywords[2]}.",

        # Template 2: Technical specification style
        f"Technical reference for {topic.replace('_', ' ')}: "
        f"{keywords[0].upper()} - primary operation. "
        f"{keywords[1].upper()} - secondary operation. "
        f"Supports: {', '.join(keywords[2:6])}. "
        f"See also: {', '.join(keywords[6:])}.",

        # Template 3: Glossary style
        f"{topic.replace('_', ' ').title()} glossary: "
        + " | ".join([f"{kw}: [{kw} definition]" for kw in keywords[:6]]),
    ]

    return random.choice(templates)


def generate_semantic_passage(topic: str, topic_data: dict) -> str:
    """
    Generate semantic-rich passage.

    Strategy: Conceptual explanation, few exact keywords, natural language.
    Best retrieved by: Semantic similarity, concept matching.
    """
    concepts = topic_data["concepts"]

    templates = [
        # Template 1: Conceptual explanation
        f"When working with {topic.replace('_', ' ')}, the fundamental idea is {concepts[0]}. "
        f"This approach enables {concepts[1]} while maintaining {concepts[2]}. "
        f"Practitioners often focus on {concepts[3]} to achieve {concepts[4]}.",

        # Template 2: Problem-solution framing
        f"The challenge of {concepts[0]} in software systems is addressed through "
        f"techniques that emphasize {concepts[1]}. "
        f"By focusing on {concepts[2]}, developers can improve {concepts[3]}. "
        f"The ultimate goal is {concepts[4]}.",

        # Template 3: Narrative explanation
        f"Understanding {topic.replace('_', ' ')} requires grasping how {concepts[0]} relates to "
        f"{concepts[1]}. In practice, this means prioritizing {concepts[2]} "
        f"while being mindful of {concepts[3]}. Success depends on {concepts[4]}.",
    ]

    return random.choice(templates)


def generate_keyword_query(topic: str, topic_data: dict) -> str:
    """
    Generate query that benefits from keyword matching.

    Direct term lookup, specific terminology.
    """
    keywords = topic_data["keywords"]
    kw = random.choice(keywords[:5])  # Use prominent keywords

    templates = [
        f"What is {kw}?",
        f"Define {kw}",
        f"{kw} syntax",
        f"How to use {kw}?",
        f"{kw} in {topic.replace('_', ' ')}",
    ]

    return random.choice(templates)


def generate_semantic_query(topic: str, topic_data: dict) -> str:
    """
    Generate query that benefits from semantic understanding.

    Conceptual question, no exact keyword match.
    """
    concepts = topic_data["concepts"]
    concept = random.choice(concepts)

    # Rephrase concepts to avoid exact match
    rephrases = {
        "ordered collection": "maintaining item sequence",
        "dynamic sizing": "flexible capacity",
        "resource retrieval": "fetching data from server",
        "data submission": "sending information",
        "faster lookups": "improving search speed",
        "learning patterns": "recognizing regularities",
        "version tracking": "managing code history",
        "data protection": "keeping information secure",
        "isolation": "separating environments",
        "pattern recognition": "finding structure in data",
        "combining tables": "merging related data",
        "identity verification": "confirming who someone is",
    }

    rephrased = rephrases.get(concept, f"achieving {concept}")

    templates = [
        f"How can I approach {rephrased}?",
        f"What's the best way to handle {rephrased}?",
        f"Why is {rephrased} important?",
        f"Explain the concept of {rephrased}",
        f"How does {topic.replace('_', ' ')} help with {rephrased}?",
    ]

    return random.choice(templates)


def generate_dataset(
    num_passages_per_type: int = 50,
    queries_per_passage: int = 2,
    output_dir: str = "data"
) -> Tuple[List[Dict], List[Dict]]:
    """
    Generate full dataset with keyword and semantic passages/queries.

    Args:
        num_passages_per_type: Passages per strategy type
        queries_per_passage: Queries to generate per passage
        output_dir: Where to save

    Returns:
        (passages, queries)
    """
    os.makedirs(output_dir, exist_ok=True)

    passages = []
    queries = []

    topics = list(TOPICS.items())

    # Generate keyword-optimized passages
    for i in range(num_passages_per_type):
        topic, topic_data = topics[i % len(topics)]

        passage_id = f"keyword_{i}"
        passage_text = generate_keyword_passage(topic, topic_data)

        passages.append({
            "id": passage_id,
            "text": passage_text,
            "strategy": "keyword",
            "topic": topic,
        })

        # Generate matching queries
        for j in range(queries_per_passage):
            query_text = generate_keyword_query(topic, topic_data)
            queries.append({
                "passage_id": passage_id,
                "query": query_text,
                "query_type": "keyword",
                "topic": topic,
            })

    # Generate semantic-optimized passages
    for i in range(num_passages_per_type):
        topic, topic_data = topics[i % len(topics)]

        passage_id = f"semantic_{i}"
        passage_text = generate_semantic_passage(topic, topic_data)

        passages.append({
            "id": passage_id,
            "text": passage_text,
            "strategy": "semantic",
            "topic": topic,
        })

        # Generate matching queries
        for j in range(queries_per_passage):
            query_text = generate_semantic_query(topic, topic_data)
            queries.append({
                "passage_id": passage_id,
                "query": query_text,
                "query_type": "semantic",
                "topic": topic,
            })

    # Save
    passages_path = os.path.join(output_dir, "multi_strategy_passages.json")
    with open(passages_path, "w") as f:
        json.dump(passages, f, indent=2)

    queries_path = os.path.join(output_dir, "multi_strategy_queries.json")
    with open(queries_path, "w") as f:
        json.dump(queries, f, indent=2)

    print(f"Generated {len(passages)} passages ({num_passages_per_type} keyword, {num_passages_per_type} semantic)")
    print(f"Generated {len(queries)} queries")
    print(f"Saved to {output_dir}")

    return passages, queries


def load_multi_strategy_data(data_dir: str) -> Tuple[List[Dict], List[Dict]]:
    """Load generated dataset."""
    passages_path = os.path.join(data_dir, "multi_strategy_passages.json")
    queries_path = os.path.join(data_dir, "multi_strategy_queries.json")

    with open(passages_path) as f:
        passages = json.load(f)
    with open(queries_path) as f:
        queries = json.load(f)

    return passages, queries


# Quick test
if __name__ == "__main__":
    print("=" * 60)
    print("Multi-Strategy Data Generator")
    print("=" * 60)

    # Show examples
    topic = "python_list"
    topic_data = TOPICS[topic]

    print(f"\nTopic: {topic}")
    print(f"\nKeyword passage:")
    print(f"  {generate_keyword_passage(topic, topic_data)}")
    print(f"\nSemantic passage:")
    print(f"  {generate_semantic_passage(topic, topic_data)}")
    print(f"\nKeyword query:")
    print(f"  {generate_keyword_query(topic, topic_data)}")
    print(f"\nSemantic query:")
    print(f"  {generate_semantic_query(topic, topic_data)}")

    # Generate small test set
    print("\n" + "-" * 40)
    print("Generating test dataset...")
    passages, queries = generate_dataset(
        num_passages_per_type=10,
        queries_per_passage=2,
        output_dir="test_data"
    )
