#!/usr/bin/env python
"""
ToneNorm Experiment

Test the hypothesis: Normalizing tone/style improves retrieval on messy data.

Like InstanceNorm removes style from images, ToneNorm should remove
conversational tone from text, leaving only semantic content.
"""

import os
import sys
import numpy as np
from typing import List, Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.messy_real_world import create_messy_corpus, evaluate_retrieval


def run_simple_similarity(documents: List[Dict], queries: List[Dict]) -> Dict:
    """Baseline: Simple embedding similarity."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    texts = [d["text"] for d in documents]
    doc_embeddings = model.encode(texts)

    results = []
    for q in queries:
        query_emb = model.encode([q["query"]])[0]

        similarities = np.dot(doc_embeddings, query_emb) / (
            np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(query_emb)
        )
        top_indices = np.argsort(similarities)[::-1][:5]
        retrieved_ids = [documents[i]["id"] for i in top_indices]
        top_scores = [similarities[i] for i in top_indices]

        metrics = evaluate_retrieval(retrieved_ids, q["correct_doc_id"])
        metrics["query"] = q["query"]
        metrics["top_retrieved"] = retrieved_ids[:3]
        metrics["top_scores"] = top_scores[:3]
        results.append(metrics)

    return {
        "method": "Simple Similarity",
        "mrr": np.mean([r["mrr"] for r in results]),
        "recall@1": np.mean([r["recall@1"] for r in results]),
        "recall@3": np.mean([r["recall@3"] for r in results]),
        "details": results,
    }


def run_tone_norm(documents: List[Dict], queries: List[Dict], auto_fit: bool = True) -> Dict:
    """ToneNorm: Normalize tone before similarity."""
    from moe_rag.tone_norm import ToneNormRAG

    texts = [d["text"] for d in documents]

    rag = ToneNormRAG.from_texts(texts, method="statistical", tone_dims=32)

    if auto_fit:
        # Auto-fit with synthetic pairs
        rag.auto_fit_tone_norm()

    print(f"\n  ToneNorm fitted: {rag.tone_norm.is_fitted}")

    results = []
    for q in queries:
        retrieved = rag.retrieve(q["query"], top_k=5, use_tone_norm=True)

        retrieved_ids = []
        for r in retrieved:
            for d in documents:
                if r["text"][:80] in d["text"] or d["text"][:80] in r["text"]:
                    retrieved_ids.append(d["id"])
                    break

        metrics = evaluate_retrieval(retrieved_ids, q["correct_doc_id"])
        metrics["query"] = q["query"]
        metrics["top_retrieved"] = retrieved_ids[:3]
        metrics["top_scores"] = [r["score"] for r in retrieved[:3]]
        results.append(metrics)

    return {
        "method": "ToneNorm",
        "mrr": np.mean([r["mrr"] for r in results]),
        "recall@1": np.mean([r["recall@1"] for r in results]),
        "recall@3": np.mean([r["recall@3"] for r in results]),
        "details": results,
    }


def run_tone_norm_custom_pairs(documents: List[Dict], queries: List[Dict]) -> Dict:
    """ToneNorm with custom domain-specific pairs."""
    from moe_rag.tone_norm import ToneNormRAG

    texts = [d["text"] for d in documents]

    rag = ToneNormRAG.from_texts(texts, method="statistical", tone_dims=48)

    # Domain-specific pairs from our messy corpus
    domain_pairs = [
        # Password reset variations
        ("How do I reset my password?", "how do i reset password"),
        ("Password reset procedure", "hey anyone know how to reset a users password??"),
        ("Reset password in admin portal", "whats the admin way to reset a users pwd?"),
        ("Force password reset for user", "reset pwd for user"),

        # Help desk tone
        ("I need assistance", "need help asap"),
        ("Unable to login", "cant login"),
        ("The system is not responding", "system being weird"),
        ("Please provide steps", "what are the steps"),

        # Documentation vs chat
        ("Navigate to Settings", "go to settings"),
        ("Click the menu option", "click the three dots menu"),
        ("Select the desired action", "theres an option there"),
        ("Confirm your selection", "ok found it thx!!"),

        # Formal vs informal requests
        ("API rate limit information", "API rate limit how many requests"),
        ("Pricing tier details", "how much does pro cost"),
        ("Subscription cancellation", "cancel my subscription"),
        ("Refund policy inquiry", "can i get a refund"),

        # Technical vs casual
        ("Authentication error 429", "getting 429 errors"),
        ("User provisioning process", "add team members"),
        ("Credential management", "reset creds"),

        # Support vs runbook
        ("Verify user identity via ticket", "check the ticket first"),
        ("Navigate to Users section", "go to users"),
        ("Select force password reset", "click force reset"),
    ]

    rag.fit_tone_norm(domain_pairs)

    print(f"\n  ToneNorm (custom) fitted: {rag.tone_norm.is_fitted}")

    results = []
    for q in queries:
        retrieved = rag.retrieve(q["query"], top_k=5, use_tone_norm=True)

        retrieved_ids = []
        for r in retrieved:
            for d in documents:
                if r["text"][:80] in d["text"] or d["text"][:80] in r["text"]:
                    retrieved_ids.append(d["id"])
                    break

        metrics = evaluate_retrieval(retrieved_ids, q["correct_doc_id"])
        metrics["query"] = q["query"]
        metrics["top_retrieved"] = retrieved_ids[:3]
        metrics["top_scores"] = [r["score"] for r in retrieved[:3]]
        results.append(metrics)

    return {
        "method": "ToneNorm (domain)",
        "mrr": np.mean([r["mrr"] for r in results]),
        "recall@1": np.mean([r["recall@1"] for r in results]),
        "recall@3": np.mean([r["recall@3"] for r in results]),
        "details": results,
    }


def run_tone_norm_sweep(documents: List[Dict], queries: List[Dict]) -> List[Dict]:
    """Sweep over tone_dims to find optimal value."""
    from moe_rag.tone_norm import ToneNormRAG

    texts = [d["text"] for d in documents]
    results = []

    for tone_dims in [8, 16, 32, 64, 128]:
        rag = ToneNormRAG.from_texts(texts, method="statistical", tone_dims=tone_dims)
        rag.auto_fit_tone_norm()

        query_results = []
        for q in queries:
            retrieved = rag.retrieve(q["query"], top_k=5, use_tone_norm=True)

            retrieved_ids = []
            for r in retrieved:
                for d in documents:
                    if r["text"][:80] in d["text"] or d["text"][:80] in r["text"]:
                        retrieved_ids.append(d["id"])
                        break

            metrics = evaluate_retrieval(retrieved_ids, q["correct_doc_id"])
            query_results.append(metrics)

        results.append({
            "tone_dims": tone_dims,
            "mrr": np.mean([r["mrr"] for r in query_results]),
            "recall@1": np.mean([r["recall@1"] for r in query_results]),
        })

    return results


def analyze_tone_space(documents: List[Dict], queries: List[Dict]):
    """Analyze what the tone space captures."""
    from moe_rag.tone_norm import ToneNormRAG
    from sentence_transformers import SentenceTransformer
    import torch

    print("\n" + "=" * 60)
    print("Tone Space Analysis")
    print("=" * 60)

    texts = [d["text"] for d in documents]
    rag = ToneNormRAG.from_texts(texts, method="statistical", tone_dims=32)
    rag.auto_fit_tone_norm()

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Compare formal vs informal embeddings
    test_pairs = [
        ("How do I reset my password?", "how do i reset password"),
        ("What is the API rate limit?", "whats the api limit"),
        ("Please help me cancel", "help me cancel asap!!"),
    ]

    print("\nSimilarity before/after ToneNorm:")
    print("-" * 50)

    for formal, informal in test_pairs:
        formal_emb = model.encode([formal])[0]
        informal_emb = model.encode([informal])[0]

        # Original similarity
        orig_sim = np.dot(formal_emb, informal_emb) / (
            np.linalg.norm(formal_emb) * np.linalg.norm(informal_emb)
        )

        # After ToneNorm
        formal_norm = rag.tone_norm.normalize(formal_emb.reshape(1, -1))[0]
        informal_norm = rag.tone_norm.normalize(informal_emb.reshape(1, -1))[0]

        norm_sim = np.dot(formal_norm, informal_norm) / (
            np.linalg.norm(formal_norm) * np.linalg.norm(informal_norm)
        )

        print(f"\n'{formal[:30]}...' vs '{informal[:30]}...'")
        print(f"  Original:  {orig_sim:.4f}")
        print(f"  ToneNorm:  {norm_sim:.4f}")
        print(f"  Change:    {norm_sim - orig_sim:+.4f}")


def run_experiment():
    """Run ToneNorm experiment on messy real-world data."""
    print("=" * 60)
    print("TONENORM EXPERIMENT")
    print("Hypothesis: Normalizing tone improves messy retrieval")
    print("=" * 60)

    documents, queries = create_messy_corpus()

    print(f"\nDocuments: {len(documents)}")
    print(f"Queries: {len(queries)}")
    for q in queries:
        print(f"  '{q['query']}' → {q['correct_doc_id']}")

    # Run methods
    print("\n" + "-" * 40)
    print("Running Simple Similarity (baseline)...")
    simple_result = run_simple_similarity(documents, queries)

    print("\n" + "-" * 40)
    print("Running ToneNorm (auto-fit)...")
    tone_result = run_tone_norm(documents, queries, auto_fit=True)

    print("\n" + "-" * 40)
    print("Running ToneNorm (domain pairs)...")
    tone_custom_result = run_tone_norm_custom_pairs(documents, queries)

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\n{'Method':<25} {'MRR':<10} {'R@1':<10} {'R@3':<10}")
    print("-" * 55)
    for r in [simple_result, tone_result, tone_custom_result]:
        print(f"{r['method']:<25} {r['mrr']:<10.4f} {r['recall@1']:<10.2%} {r['recall@3']:<10.2%}")

    # Per-query breakdown
    print("\n" + "-" * 40)
    print("Per-Query Results")
    print("-" * 40)

    for i, q in enumerate(queries):
        simple_rank = simple_result["details"][i]["rank"]
        tone_rank = tone_result["details"][i]["rank"]
        tone_custom_rank = tone_custom_result["details"][i]["rank"]

        winner = "→" if tone_custom_rank < simple_rank else ("=" if tone_custom_rank == simple_rank else "←")

        print(f"\n'{q['query'][:40]}...'")
        print(f"  Correct: {q['correct_doc_id']}")
        print(f"  Simple rank: {simple_rank}  |  ToneNorm: {tone_rank}  |  Domain: {tone_custom_rank}  {winner}")

    # Analysis
    print("\n" + "-" * 40)
    print("ANALYSIS")
    print("-" * 40)

    simple_mrr = simple_result["mrr"]
    tone_mrr = tone_custom_result["mrr"]

    if tone_mrr > simple_mrr:
        improvement = (tone_mrr - simple_mrr) / simple_mrr * 100
        print(f"\n✓ ToneNorm IMPROVES by {improvement:.1f}% MRR!")
        print("  Removing tone helps with messy/informal queries.")
    elif tone_mrr == simple_mrr:
        print("\n= ToneNorm matches baseline.")
        print("  Tone normalization neither helps nor hurts.")
    else:
        degradation = (simple_mrr - tone_mrr) / simple_mrr * 100
        print(f"\n✗ ToneNorm DEGRADES by {degradation:.1f}% MRR")
        print("  Tone normalization removes useful signal.")

    # Tone space analysis
    analyze_tone_space(documents, queries)

    # Sweep tone_dims
    print("\n" + "-" * 40)
    print("Tone Dims Sweep")
    print("-" * 40)
    sweep_results = run_tone_norm_sweep(documents, queries)
    print(f"\n{'Dims':<10} {'MRR':<10} {'R@1':<10}")
    for r in sweep_results:
        print(f"{r['tone_dims']:<10} {r['mrr']:<10.4f} {r['recall@1']:<10.2%}")


if __name__ == "__main__":
    run_experiment()
