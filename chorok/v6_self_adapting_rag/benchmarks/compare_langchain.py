#!/usr/bin/env python
"""
Benchmark: MoE-RAG vs LangChain

Compare:
1. Setup complexity (lines of code, config decisions)
2. Accuracy (retrieval quality)
3. Time (training/indexing, inference)

This demonstrates the "pipeline killer" value proposition.
"""

import os
import sys
import time
import json
from typing import List, Dict, Tuple
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class BenchmarkResult:
    method: str
    setup_lines: int
    config_decisions: int
    indexing_time: float
    query_time: float
    mrr: float
    recall_at_1: float
    recall_at_3: float


def create_test_corpus(num_docs: int = 100) -> Tuple[List[Dict], List[Dict]]:
    """
    Create test corpus with known answers for evaluation.

    Returns:
        (documents, test_queries) where each query has a known answer doc
    """
    topics = [
        ("Password Reset", "To reset your password: Go to Settings > Security > Reset Password. Enter your current password, then your new password twice. Click Save."),
        ("API Authentication", "Our API uses JWT tokens. Register at developer.example.com, include the token in Authorization header. Tokens expire after 24 hours."),
        ("Billing", "We accept credit cards, PayPal, and bank transfers. Monthly billing on the 1st. Annual billing saves 20%."),
        ("Account Settings", "Access account settings from the profile menu. You can update email, notification preferences, and privacy settings."),
        ("Data Export", "Export your data from Settings > Privacy > Export Data. Supported formats: JSON, CSV, PDF. Export may take up to 24 hours."),
        ("Two-Factor Auth", "Enable 2FA in Security settings. We support authenticator apps and SMS. Backup codes are provided during setup."),
        ("Team Management", "Add team members from Admin > Team. Assign roles: Admin, Editor, Viewer. Each role has different permissions."),
        ("Integrations", "Connect third-party apps in Settings > Integrations. We support Slack, GitHub, Jira, and 50+ other services."),
        ("Search", "Use the search bar to find content. Filters available: date, type, author. Advanced search supports boolean operators."),
        ("Notifications", "Configure notifications in Settings > Notifications. Choose email, push, or in-app. Set quiet hours to pause alerts."),
    ]

    documents = []
    test_queries = []

    for i in range(num_docs):
        topic_idx = i % len(topics)
        title, content = topics[topic_idx]

        # Add some variation
        variation = f" Document #{i}. Last updated: 2024-{(i % 12) + 1:02d}-01."

        doc_id = f"doc_{i}"
        documents.append({
            "id": doc_id,
            "title": title,
            "content": content + variation,
            "text": f"{title}\n\n{content}{variation}"
        })

        # Create a query for every 10th document
        if i % 10 == 0:
            query_templates = [
                f"How do I {title.lower()}?",
                f"What is {title.lower()}?",
                f"Help with {title.lower()}",
            ]
            test_queries.append({
                "query": query_templates[i % len(query_templates)],
                "answer_doc_id": doc_id,
                "topic": title,
            })

    return documents, test_queries


def evaluate_retrieval(
    retrieved_ids: List[str],
    correct_id: str
) -> Dict[str, float]:
    """Compute retrieval metrics."""
    if not retrieved_ids:
        return {"mrr": 0, "recall@1": 0, "recall@3": 0}

    # MRR
    mrr = 0
    for i, rid in enumerate(retrieved_ids):
        if rid == correct_id:
            mrr = 1.0 / (i + 1)
            break

    # Recall
    recall_1 = 1.0 if correct_id in retrieved_ids[:1] else 0.0
    recall_3 = 1.0 if correct_id in retrieved_ids[:3] else 0.0

    return {"mrr": mrr, "recall@1": recall_1, "recall@3": recall_3}


# =============================================================================
# MoE-RAG Implementation
# =============================================================================

def run_moe_rag(documents: List[Dict], queries: List[Dict]) -> BenchmarkResult:
    """
    MoE-RAG benchmark.

    Setup complexity: ~5 lines
    Config decisions: 0
    """
    from moe_rag import MoERAG

    # === SETUP CODE (count these lines) ===
    texts = [d["text"] for d in documents]           # Line 1
    rag = MoERAG.from_texts(texts)                   # Line 2
    rag.train(epochs=5, verbose=False)               # Line 3
    # === END SETUP ===

    setup_lines = 3
    config_decisions = 0  # No decisions needed

    # Measure indexing time (training)
    start = time.time()
    # Already done above, but for fair comparison:
    rag2 = MoERAG.from_texts(texts)
    rag2.train(epochs=5, verbose=False)
    indexing_time = time.time() - start

    # Evaluate
    mrrs, r1s, r3s = [], [], []
    query_times = []

    for q in queries:
        start = time.time()
        results = rag.retrieve(q["query"], top_k=5)
        query_times.append(time.time() - start)

        # Map back to doc IDs (chunks have source info)
        retrieved_ids = []
        for r in results:
            # Find matching doc by text prefix
            for d in documents:
                if d["text"][:50] in r.text or r.text[:50] in d["text"]:
                    retrieved_ids.append(d["id"])
                    break

        metrics = evaluate_retrieval(retrieved_ids, q["answer_doc_id"])
        mrrs.append(metrics["mrr"])
        r1s.append(metrics["recall@1"])
        r3s.append(metrics["recall@3"])

    return BenchmarkResult(
        method="MoE-RAG",
        setup_lines=setup_lines,
        config_decisions=config_decisions,
        indexing_time=indexing_time,
        query_time=sum(query_times) / len(query_times),
        mrr=sum(mrrs) / len(mrrs),
        recall_at_1=sum(r1s) / len(r1s),
        recall_at_3=sum(r3s) / len(r3s),
    )


# =============================================================================
# LangChain Implementation
# =============================================================================

def run_langchain(documents: List[Dict], queries: List[Dict]) -> BenchmarkResult:
    """
    LangChain RAG benchmark.

    Setup complexity: ~15 lines
    Config decisions: 5+ (chunking, embedding, vector store, retriever, k)
    """
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
        from langchain.docstore.document import Document
    except ImportError:
        print("LangChain not installed. Run: pip install langchain langchain-community faiss-cpu")
        return None

    # === SETUP CODE (count these lines) ===
    # Decision 1: Chunking strategy
    text_splitter = RecursiveCharacterTextSplitter(    # Line 1
        chunk_size=500,                                 # Line 2 - Decision 2: chunk size
        chunk_overlap=50                                # Line 3 - Decision 3: overlap
    )

    # Prepare documents
    docs = [Document(                                   # Line 4
        page_content=d["text"],                        # Line 5
        metadata={"id": d["id"]}                       # Line 6
    ) for d in documents]

    splits = text_splitter.split_documents(docs)       # Line 7

    # Decision 4: Embedding model
    embeddings = HuggingFaceEmbeddings(                # Line 8
        model_name="sentence-transformers/all-MiniLM-L6-v2"  # Line 9
    )

    # Decision 5: Vector store
    vectorstore = FAISS.from_documents(splits, embeddings)  # Line 10

    # Decision 6: Retriever config
    retriever = vectorstore.as_retriever(              # Line 11
        search_kwargs={"k": 5}                         # Line 12 - Decision 7: k value
    )
    # === END SETUP ===

    setup_lines = 12
    config_decisions = 7

    # Measure indexing time
    start = time.time()
    splits2 = text_splitter.split_documents(docs)
    vectorstore2 = FAISS.from_documents(splits2, embeddings)
    indexing_time = time.time() - start

    # Evaluate
    mrrs, r1s, r3s = [], [], []
    query_times = []

    for q in queries:
        start = time.time()
        results = retriever.get_relevant_documents(q["query"])
        query_times.append(time.time() - start)

        retrieved_ids = [r.metadata.get("id") for r in results if r.metadata.get("id")]

        metrics = evaluate_retrieval(retrieved_ids, q["answer_doc_id"])
        mrrs.append(metrics["mrr"])
        r1s.append(metrics["recall@1"])
        r3s.append(metrics["recall@3"])

    return BenchmarkResult(
        method="LangChain",
        setup_lines=setup_lines,
        config_decisions=config_decisions,
        indexing_time=indexing_time,
        query_time=sum(query_times) / len(query_times),
        mrr=sum(mrrs) / len(mrrs),
        recall_at_1=sum(r1s) / len(r1s),
        recall_at_3=sum(r3s) / len(r3s),
    )


# =============================================================================
# Simple Embedding Baseline (no framework)
# =============================================================================

def run_simple_embedding(documents: List[Dict], queries: List[Dict]) -> BenchmarkResult:
    """
    Simple embedding similarity baseline.

    No MoE, no framework. Just embed and compare.
    """
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError:
        print("sentence-transformers not installed")
        return None

    # === SETUP CODE ===
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  # Line 1
    texts = [d["text"] for d in documents]                                  # Line 2
    doc_embeddings = model.encode(texts)                                    # Line 3
    # === END SETUP ===

    setup_lines = 3
    config_decisions = 1  # Just embedding model choice

    # Measure indexing time
    start = time.time()
    doc_embeddings2 = model.encode(texts)
    indexing_time = time.time() - start

    # Evaluate
    mrrs, r1s, r3s = [], [], []
    query_times = []

    for q in queries:
        start = time.time()
        query_emb = model.encode([q["query"]])[0]

        # Cosine similarity
        similarities = np.dot(doc_embeddings, query_emb) / (
            np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(query_emb)
        )
        top_indices = np.argsort(similarities)[::-1][:5]
        query_times.append(time.time() - start)

        retrieved_ids = [documents[i]["id"] for i in top_indices]

        metrics = evaluate_retrieval(retrieved_ids, q["answer_doc_id"])
        mrrs.append(metrics["mrr"])
        r1s.append(metrics["recall@1"])
        r3s.append(metrics["recall@3"])

    return BenchmarkResult(
        method="Simple Embedding",
        setup_lines=setup_lines,
        config_decisions=config_decisions,
        indexing_time=indexing_time,
        query_time=sum(query_times) / len(query_times),
        mrr=sum(mrrs) / len(mrrs),
        recall_at_1=sum(r1s) / len(r1s),
        recall_at_3=sum(r3s) / len(r3s),
    )


# =============================================================================
# Main Benchmark
# =============================================================================

def run_benchmark(num_docs: int = 100):
    """Run full benchmark comparison."""
    print("=" * 60)
    print(f"RAG Benchmark: MoE-RAG vs LangChain vs Simple Embedding")
    print(f"Corpus size: {num_docs} documents")
    print("=" * 60)

    # Create test data
    print("\nCreating test corpus...")
    documents, queries = create_test_corpus(num_docs)
    print(f"Created {len(documents)} documents, {len(queries)} test queries")

    results = []

    # Run benchmarks
    print("\n" + "-" * 40)
    print("Running MoE-RAG...")
    print("-" * 40)
    moe_result = run_moe_rag(documents, queries)
    if moe_result:
        results.append(moe_result)
        print(f"  MRR: {moe_result.mrr:.4f}")

    print("\n" + "-" * 40)
    print("Running LangChain...")
    print("-" * 40)
    lc_result = run_langchain(documents, queries)
    if lc_result:
        results.append(lc_result)
        print(f"  MRR: {lc_result.mrr:.4f}")

    print("\n" + "-" * 40)
    print("Running Simple Embedding...")
    print("-" * 40)
    simple_result = run_simple_embedding(documents, queries)
    if simple_result:
        results.append(simple_result)
        print(f"  MRR: {simple_result.mrr:.4f}")

    # Print comparison table
    print("\n" + "=" * 60)
    print("RESULTS COMPARISON")
    print("=" * 60)

    print("\n### Setup Complexity ###")
    print(f"{'Method':<20} {'Lines':<10} {'Decisions':<10}")
    print("-" * 40)
    for r in results:
        print(f"{r.method:<20} {r.setup_lines:<10} {r.config_decisions:<10}")

    print("\n### Accuracy ###")
    print(f"{'Method':<20} {'MRR':<10} {'R@1':<10} {'R@3':<10}")
    print("-" * 40)
    for r in results:
        print(f"{r.method:<20} {r.mrr:<10.4f} {r.recall_at_1:<10.2%} {r.recall_at_3:<10.2%}")

    print("\n### Speed ###")
    print(f"{'Method':<20} {'Index (s)':<12} {'Query (ms)':<12}")
    print("-" * 40)
    for r in results:
        print(f"{r.method:<20} {r.indexing_time:<12.2f} {r.query_time*1000:<12.2f}")

    # Save results
    output = {
        "num_docs": num_docs,
        "num_queries": len(queries),
        "results": [
            {
                "method": r.method,
                "setup_lines": r.setup_lines,
                "config_decisions": r.config_decisions,
                "indexing_time": r.indexing_time,
                "query_time": r.query_time,
                "mrr": r.mrr,
                "recall_at_1": r.recall_at_1,
                "recall_at_3": r.recall_at_3,
            }
            for r in results
        ]
    }

    output_path = os.path.join(os.path.dirname(__file__), "benchmark_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_docs", type=int, default=100)
    args = parser.parse_args()

    run_benchmark(args.num_docs)
