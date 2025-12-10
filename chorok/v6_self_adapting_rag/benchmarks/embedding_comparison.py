#!/usr/bin/env python
"""
What is a "good embedding"?

Compare different embedding models on our test scenarios to show
which ones are actually "good" and why.
"""

import numpy as np
from typing import List, Dict, Tuple


# Test cases that reveal embedding quality
TEST_PAIRS = [
    # Semantic equivalence (should be HIGH similarity)
    ("How do I reset my password?", "password reset steps", "semantic"),
    ("What's the API rate limit?", "rate limiting for API calls", "semantic"),
    ("Cancel my subscription", "how to unsubscribe", "semantic"),

    # Typos/informal (should STILL be HIGH - good embeddings handle this)
    ("How do I reset my password?", "how do i reset pasword", "typo"),
    ("What's the API rate limit?", "whats the api limit", "informal"),
    ("Cancel my subscription", "cancel subscription pls", "informal"),

    # Different topic (should be LOW similarity)
    ("How do I reset my password?", "What's the pricing?", "different"),
    ("API rate limit", "shipping policy", "different"),
    ("Cancel subscription", "create new account", "different"),

    # Keyword overlap but different meaning (should be LOW - tests robustness)
    ("How do I reset my password?", "Password strength tips for security", "misleading"),
    ("API rate limit", "Rate your experience with our API", "misleading"),
]


def test_embedding_model(model_name: str) -> Dict:
    """Test a single embedding model."""
    from sentence_transformers import SentenceTransformer

    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"{'='*60}")

    try:
        model = SentenceTransformer(model_name)
    except Exception as e:
        print(f"  Failed to load: {e}")
        return None

    results = {"semantic": [], "typo": [], "informal": [], "different": [], "misleading": []}

    for text1, text2, category in TEST_PAIRS:
        emb1 = model.encode([text1])[0]
        emb2 = model.encode([text2])[0]

        sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        results[category].append(sim)

        print(f"\n  [{category}] {sim:.3f}")
        print(f"    '{text1[:40]}...' vs '{text2[:40]}...'")

    # Summary
    summary = {
        "model": model_name,
        "semantic_avg": np.mean(results["semantic"]),
        "typo_avg": np.mean(results["typo"]),
        "informal_avg": np.mean(results["informal"]),
        "different_avg": np.mean(results["different"]),
        "misleading_avg": np.mean(results["misleading"]),
    }

    # Quality score: high semantic/typo/informal, low different/misleading
    summary["quality_score"] = (
        summary["semantic_avg"] +
        summary["typo_avg"] +
        summary["informal_avg"] -
        summary["different_avg"] -
        summary["misleading_avg"]
    )

    return summary


def run_comparison():
    """Compare popular embedding models."""

    # Models to compare (from worst to best typically)
    models = [
        # Older/smaller models
        "sentence-transformers/all-MiniLM-L6-v2",      # 22M params, fast
        "sentence-transformers/all-mpnet-base-v2",     # 110M params, better

        # Modern high-quality models
        "BAAI/bge-small-en-v1.5",                      # 33M params, very good
        "BAAI/bge-base-en-v1.5",                       # 110M params, excellent

        # Instruction-tuned (best for retrieval)
        # "BAAI/bge-large-en-v1.5",                    # 335M params (skip if slow)
    ]

    print("="*60)
    print("EMBEDDING MODEL COMPARISON")
    print("What makes an embedding 'good' for RAG?")
    print("="*60)

    print("""
A 'good' embedding model should:
1. HIGH similarity for semantically equivalent text
2. HIGH similarity despite typos/informal writing
3. LOW similarity for different topics
4. LOW similarity for misleading keyword overlap

Let's test...
""")

    results = []
    for model_name in models:
        result = test_embedding_model(model_name)
        if result:
            results.append(result)

    # Summary table
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    print(f"\n{'Model':<45} {'Semantic':<10} {'Typo':<10} {'Informal':<10} {'Different':<10} {'Misleading':<10} {'Score':<8}")
    print("-"*103)

    for r in sorted(results, key=lambda x: x["quality_score"], reverse=True):
        model_short = r["model"].split("/")[-1][:40]
        print(f"{model_short:<45} {r['semantic_avg']:<10.3f} {r['typo_avg']:<10.3f} {r['informal_avg']:<10.3f} {r['different_avg']:<10.3f} {r['misleading_avg']:<10.3f} {r['quality_score']:<8.3f}")

    # Interpretation
    print("\n" + "-"*60)
    print("INTERPRETATION")
    print("-"*60)

    best = max(results, key=lambda x: x["quality_score"])
    worst = min(results, key=lambda x: x["quality_score"])

    print(f"""
Best model: {best['model'].split('/')[-1]}
  - Semantic similarity: {best['semantic_avg']:.3f} (want HIGH)
  - Handles typos: {best['typo_avg']:.3f} (want HIGH)
  - Handles informal: {best['informal_avg']:.3f} (want HIGH)
  - Rejects different topics: {best['different_avg']:.3f} (want LOW)
  - Rejects misleading: {best['misleading_avg']:.3f} (want LOW)

Worst model: {worst['model'].split('/')[-1]}
  - Quality score: {worst['quality_score']:.3f} vs {best['quality_score']:.3f}

Key insight:
  Even the 'worst' model (MiniLM) handles typos/informal well ({worst['typo_avg']:.3f}).
  This is why our ToneNorm didn't help - the invariance is already built-in.
""")


if __name__ == "__main__":
    run_comparison()
