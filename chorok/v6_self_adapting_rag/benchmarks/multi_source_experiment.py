#!/usr/bin/env python
"""
Multi-Source RAG Experiment

Key hypothesis: When content is spread across heterogeneous sources
(Slack, Wiki, Runbook, etc.), source-aware normalization OUTPERFORMS
single-embedder approaches.

This is the "text multi-modality" angle:
- LLMs see multi-modality as: text + images + audio
- We see text multi-modality as: Slack + Wiki + Runbook + Email + Code

Each source type is like a different "modality" - same content, different style.
"""

import os
import sys
import numpy as np
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# Multi-Source Corpus - Same content across different source types
# =============================================================================

MULTI_SOURCE_CORPUS = [
    # Topic 1: Password Reset - Same info, 4 different source styles
    {
        "id": "pwd_wiki",
        "source_type": "wiki",
        "topic": "password_reset",
        "has_answer": True,
        "text": """Password Reset Guide
Last updated: 2024-01-15

== Overview ==
This document describes the password reset process for administrators.

== Steps ==
1. Navigate to admin.internal.com/users
2. Search for the user by email address
3. Click the user row to open details
4. Select "Actions" dropdown menu
5. Choose "Force Password Reset"
6. Select method: Email (default), SMS, or Manual
7. Document the action in the support ticket

== Notes ==
- Admin role required
- Reset link expires in 15 minutes
- See AUTH-001 runbook for escalations

Tags: password, reset, admin, authentication
""",
    },
    {
        "id": "pwd_slack",
        "source_type": "slack",
        "topic": "password_reset",
        "has_answer": True,
        "text": """#support-escalations
@john.doe [10:32 AM]
hey anyone know how to reset a users password?? the portal is being weird

@mike.chen [10:40 AM]
go to admin.internal.com/users, search the email, click the three dots menu, theres a "reset pwd" option there

@sarah.smith [10:42 AM]
btw mike u need admin role for that, regular support cant do it

@mike.chen [10:43 AM]
oh right yeah. also the reset link only lasts like 15min so tell the user to check email quick

@john.doe [10:45 AM]
found it thx!! didnt realize i needed admin access lol
""",
    },
    {
        "id": "pwd_runbook",
        "source_type": "runbook",
        "topic": "password_reset",
        "has_answer": True,
        "text": """RUNBOOK: AUTH-001 - Password Reset Procedure (PROD)

SEVERITY: P3
ONCALL: security-oncall@company.com

PRE-REQS:
- VPN connected
- Admin credentials
- MFA device ready

STEPS:
1. Verify user identity via Zendesk ticket
2. Access admin portal: https://admin.internal.com
3. Navigate: Users > Search > [enter email]
4. Click user row > Actions dropdown > "Force Password Reset"
5. Select reset method:
   - Email (default, 15min expiry)
   - SMS (requires phone on file)
   - Manual (generates temp pwd, 1hr expiry)
6. Document in ticket: reset initiated, method used
7. Close ticket after user confirms access

ROLLBACK:
If user locked out after reset, see RUNBOOK: AUTH-002
""",
    },
    {
        "id": "pwd_email",
        "source_type": "email",
        "topic": "password_reset",
        "has_answer": False,  # Just complaining, no actual steps
        "text": """From: angry.user@gmail.com
To: support@company.com
Subject: RE: RE: RE: STILL CANT LOGIN!!!

This is ridiculous. I've been trying to reset my password for 3 DAYS.

Your website says "click forgot password" but when I click it NOTHING HAPPENS.
I've tried Chrome, Firefox, even IE. Nothing works.

I need access to my account ASAP for work. This is urgent!!!

---Previous message---
From: support@company.com

Hi, please try clearing your browser cache and trying again.
If that doesn't work, we can manually reset from our admin portal.
""",
    },

    # Topic 2: API Rate Limits - Same info, different sources
    {
        "id": "api_wiki",
        "source_type": "wiki",
        "topic": "api_limits",
        "has_answer": True,
        "text": """API Rate Limiting

== Rate Limits by Tier ==

| Tier     | Per Minute | Per Hour  | Per Day    |
|----------|------------|-----------|------------|
| Free     | 100        | 1,000     | 10,000     |
| Pro      | 1,000      | 10,000    | 100,000    |
| Ent.     | 10,000     | unlimited | unlimited  |

== Error Handling ==
When rate limited, API returns HTTP 429.
Response includes Retry-After header.

== Best Practices ==
- Implement exponential backoff
- Cache responses where possible
- Use bulk endpoints for batch operations

Related: API Authentication, API Errors
""",
    },
    {
        "id": "api_stackoverflow",
        "source_type": "stackoverflow",
        "topic": "api_limits",
        "has_answer": True,
        "text": """Q: Rate limit exceeded error 429 - what's the limit?? [closed as duplicate]

I keep getting 429 errors from the API. Documentation doesn't say what the limit is. Anyone know?

---
A1: (accepted, 45 upvotes)
The limits are:
- Free: 100/min, 1000/hr, 10000/day
- Pro: 1000/min, 10000/hr

But honestly the docs are terrible, I had to find this by trial and error lol

---
A2: (12 upvotes)
Try adding exponential backoff:
```python
import time
def api_call_with_retry(func):
    for i in range(5):
        try:
            return func()
        except RateLimitError:
            time.sleep(2 ** i)
```
""",
    },
    {
        "id": "api_slack",
        "source_type": "slack",
        "topic": "api_limits",
        "has_answer": True,
        "text": """#dev-help
@dev.jane [2:15 PM]
anyone know the api rate limits? getting 429s

@senior.dev [2:18 PM]
free tier is 100/min, 1000/hr. pro is 1000/min. enterprise is basically unlimited

@dev.jane [2:19 PM]
ah ok. we're on free tier rn. guess need to add backoff

@senior.dev [2:20 PM]
ya exponential backoff is the move. or upgrade to pro lol

@dev.jane [2:21 PM]
thx!
""",
    },

    # Topic 3: Pricing - Different sources
    {
        "id": "pricing_wiki",
        "source_type": "wiki",
        "topic": "pricing",
        "has_answer": True,
        "text": """Pricing Tiers

== Plans ==

=== Basic Plan - $9.99/month ===
- 5 users maximum
- 10GB storage
- Email support only
- Basic analytics

=== Pro Plan - $29.99/month ===
- Unlimited users
- 100GB storage
- Priority support
- API access (1000 req/min)
- Advanced analytics

=== Enterprise - Contact Sales ===
- Custom user limits
- Unlimited storage
- Dedicated support
- Custom SLA
- SSO/SAML

== Discounts ==
- Annual billing: 20% off
- Non-profit: 50% off
- Education: Free for students

Contact sales@company.com for custom pricing.
""",
    },
    {
        "id": "pricing_pdf",
        "source_type": "pdf",
        "topic": "pricing",
        "has_answer": True,
        "text": """[EXTRACTED FROM PRICING.PDF - OCR MAY HAVE ERRORS]

PRICING TIERS 2024

BasicPlan $9.99/mo
- 5 users max
- 10G8 storage (note: might be 10GB, OCR unclear)
- Email support only

Pro Plan.......$29.99/mo
- Unlimited users
- 100GB stoarge
- Priority supoprt
- API access

Enterprise.....Contact Sales
- Custom everything
- SLA included
- Dedicated CSM

*Prices in USD. Annual discount: 20% off
**Subject to change without notice

Page 1 of 3
[NEXT PAGE MISSING]
""",
    },
    {
        "id": "pricing_email",
        "source_type": "email",
        "topic": "pricing",
        "has_answer": True,
        "text": """From: sales@company.com
To: prospect@client.com
Subject: Re: Pricing inquiry

Hi,

Thanks for your interest! Here's our pricing breakdown:

Basic: $9.99/month (5 users, 10GB, email support)
Pro: $29.99/month (unlimited users, 100GB, priority support, API)
Enterprise: Custom pricing (contact us!)

We also offer:
- 20% off for annual billing
- 50% off for non-profits
- Free tier for students

Let me know if you have any questions!

Best,
Sales Team
""",
    },
]


# =============================================================================
# Test Queries - From different "perspectives"
# =============================================================================

MULTI_SOURCE_QUERIES = [
    # Informal queries (should match Slack despite formal Wiki having answer)
    {
        "query": "how do i reset a users password",
        "topic": "password_reset",
        "expected_best": ["pwd_runbook", "pwd_wiki"],  # Has exact steps
    },
    {
        "query": "pwd reset steps for admin?",
        "topic": "password_reset",
        "expected_best": ["pwd_runbook", "pwd_wiki"],
    },

    # Technical queries (should find runbook)
    {
        "query": "AUTH-001 password reset procedure",
        "topic": "password_reset",
        "expected_best": ["pwd_runbook"],
    },

    # API queries
    {
        "query": "api rate limit free tier",
        "topic": "api_limits",
        "expected_best": ["api_wiki", "api_stackoverflow"],
    },
    {
        "query": "getting 429 errors whats the limit",
        "topic": "api_limits",
        "expected_best": ["api_stackoverflow", "api_wiki"],
    },

    # Pricing queries
    {
        "query": "how much does pro cost",
        "topic": "pricing",
        "expected_best": ["pricing_wiki", "pricing_email"],
    },
    {
        "query": "pricing tiers",
        "topic": "pricing",
        "expected_best": ["pricing_wiki", "pricing_pdf"],
    },
]


def evaluate_cross_source(
    results: List[Dict],
    query_topic: str,
    expected_best: List[str],
    corpus: List[Dict]
) -> Dict:
    """Evaluate cross-source retrieval."""
    # Check if any expected doc is in top 3
    retrieved_ids = [r["doc_id"] for r in results[:3]]

    hit_at_1 = results[0]["doc_id"] in expected_best if results else False
    hit_at_3 = any(rid in expected_best for rid in retrieved_ids)

    # Check source diversity - are we getting docs from multiple sources?
    source_types = set(r["source_type"] for r in results[:3])

    # Find rank of best expected
    best_rank = -1
    for i, r in enumerate(results):
        if r["doc_id"] in expected_best:
            best_rank = i + 1
            break

    return {
        "hit@1": hit_at_1,
        "hit@3": hit_at_3,
        "best_rank": best_rank,
        "source_diversity": len(source_types),
        "retrieved_sources": list(source_types),
    }


def run_baseline(corpus: List[Dict], queries: List[Dict]) -> Dict:
    """Single-embedder baseline."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    texts = [d["text"] for d in corpus]
    doc_embeddings = model.encode(texts)

    all_results = []
    for q in queries:
        query_emb = model.encode([q["query"]])[0]

        similarities = np.dot(doc_embeddings, query_emb) / (
            np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(query_emb)
        )
        top_indices = np.argsort(similarities)[::-1][:5]

        results = [
            {
                "doc_id": corpus[i]["id"],
                "source_type": corpus[i]["source_type"],
                "score": similarities[i],
            }
            for i in top_indices
        ]

        metrics = evaluate_cross_source(results, q["topic"], q["expected_best"], corpus)
        metrics["query"] = q["query"]
        metrics["results"] = results
        all_results.append(metrics)

    return {
        "method": "Single Embedder",
        "hit@1": np.mean([r["hit@1"] for r in all_results]),
        "hit@3": np.mean([r["hit@3"] for r in all_results]),
        "avg_source_diversity": np.mean([r["source_diversity"] for r in all_results]),
        "details": all_results,
    }


def run_multi_source(corpus: List[Dict], queries: List[Dict], epochs: int = 50) -> Dict:
    """Multi-source RAG with style normalization."""
    from moe_rag.multi_source import MultiSourceRAG, SourceType, SourceDocument

    rag = MultiSourceRAG()

    texts = [d["text"] for d in corpus]
    source_types = [SourceType(d["source_type"]) for d in corpus]
    doc_ids = [d["id"] for d in corpus]

    rag.add_documents(texts, source_types, doc_ids)

    # Inject topic metadata from corpus (ground truth)
    for i, doc in enumerate(rag.documents):
        doc.metadata = {"topic": corpus[i]["topic"]}

    rag.train(epochs=epochs, verbose=True)

    all_results = []
    for q in queries:
        results = rag.retrieve(q["query"], top_k=5)

        metrics = evaluate_cross_source(results, q["topic"], q["expected_best"], corpus)
        metrics["query"] = q["query"]
        metrics["results"] = results
        all_results.append(metrics)

    return {
        "method": f"Multi-Source ({epochs} epochs)",
        "hit@1": np.mean([r["hit@1"] for r in all_results]),
        "hit@3": np.mean([r["hit@3"] for r in all_results]),
        "avg_source_diversity": np.mean([r["source_diversity"] for r in all_results]),
        "details": all_results,
    }


def analyze_source_bias(corpus: List[Dict], queries: List[Dict]):
    """Analyze which sources the baseline prefers."""
    from sentence_transformers import SentenceTransformer

    print("\n" + "=" * 60)
    print("Source Bias Analysis")
    print("Which sources does single-embedder prefer?")
    print("=" * 60)

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    texts = [d["text"] for d in corpus]
    doc_embeddings = model.encode(texts)

    source_rank_sums = {}
    source_counts = {}

    for q in queries:
        query_emb = model.encode([q["query"]])[0]

        similarities = np.dot(doc_embeddings, query_emb) / (
            np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(query_emb)
        )
        ranked_indices = np.argsort(similarities)[::-1]

        for rank, idx in enumerate(ranked_indices):
            st = corpus[idx]["source_type"]
            if st not in source_rank_sums:
                source_rank_sums[st] = 0
                source_counts[st] = 0
            source_rank_sums[st] += rank + 1
            source_counts[st] += 1

    print("\nAverage rank by source type (lower = more preferred):")
    print("-" * 40)

    source_avg_ranks = {
        st: source_rank_sums[st] / source_counts[st]
        for st in source_rank_sums
    }

    for st, avg_rank in sorted(source_avg_ranks.items(), key=lambda x: x[1]):
        print(f"  {st:<15} {avg_rank:.2f}")


def run_experiment():
    """Run multi-source experiment."""
    print("=" * 60)
    print("MULTI-SOURCE RAG EXPERIMENT")
    print("Hypothesis: Source-aware normalization improves cross-source retrieval")
    print("=" * 60)

    corpus = MULTI_SOURCE_CORPUS
    queries = MULTI_SOURCE_QUERIES

    # Show corpus stats
    source_counts = {}
    for d in corpus:
        st = d["source_type"]
        source_counts[st] = source_counts.get(st, 0) + 1

    print(f"\nCorpus: {len(corpus)} documents")
    print("Source distribution:")
    for st, count in sorted(source_counts.items()):
        print(f"  {st}: {count}")

    print(f"\nQueries: {len(queries)}")

    # Analyze source bias
    analyze_source_bias(corpus, queries)

    # Run methods
    print("\n" + "-" * 40)
    print("Running Single Embedder baseline...")
    baseline_result = run_baseline(corpus, queries)

    print("\n" + "-" * 40)
    print("Running Multi-Source RAG...")
    multi_result = run_multi_source(corpus, queries, epochs=50)

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\n{'Method':<25} {'Hit@1':<10} {'Hit@3':<10} {'Source Div':<12}")
    print("-" * 57)
    for r in [baseline_result, multi_result]:
        print(f"{r['method']:<25} {r['hit@1']:<10.2%} {r['hit@3']:<10.2%} {r['avg_source_diversity']:<12.2f}")

    # Per-query breakdown
    print("\n" + "-" * 40)
    print("Per-Query Results")
    print("-" * 40)

    for i, q in enumerate(queries):
        base_rank = baseline_result["details"][i]["best_rank"]
        multi_rank = multi_result["details"][i]["best_rank"]

        base_sources = baseline_result["details"][i]["retrieved_sources"]
        multi_sources = multi_result["details"][i]["retrieved_sources"]

        winner = "→" if multi_rank < base_rank else ("=" if multi_rank == base_rank else "←")

        print(f"\n'{q['query'][:40]}...'")
        print(f"  Expected: {q['expected_best']}")
        print(f"  Baseline rank: {base_rank}, sources: {base_sources}")
        print(f"  Multi-Source rank: {multi_rank}, sources: {multi_sources}  {winner}")

    # Analysis
    print("\n" + "-" * 40)
    print("ANALYSIS")
    print("-" * 40)

    baseline_hit3 = baseline_result["hit@3"]
    multi_hit3 = multi_result["hit@3"]

    if multi_hit3 > baseline_hit3:
        improvement = (multi_hit3 - baseline_hit3) / baseline_hit3 * 100
        print(f"\n✓ Multi-Source RAG IMPROVES by {improvement:.1f}% Hit@3!")
        print("  Source-aware normalization helps cross-source retrieval.")
        print("  This is the 'text multi-modality' advantage.")
    elif multi_hit3 == baseline_hit3:
        print("\n= Multi-Source RAG matches baseline.")
        print("  Sources may be too similar or corpus too small.")
    else:
        degradation = (baseline_hit3 - multi_hit3) / baseline_hit3 * 100
        print(f"\n✗ Multi-Source RAG DEGRADES by {degradation:.1f}% Hit@3")
        print("  Need more cross-source training data.")

    # Source diversity
    baseline_div = baseline_result["avg_source_diversity"]
    multi_div = multi_result["avg_source_diversity"]

    if multi_div > baseline_div:
        print(f"\n✓ Multi-Source has HIGHER source diversity ({multi_div:.2f} vs {baseline_div:.2f})")
        print("  Retrieves from more diverse sources.")


if __name__ == "__main__":
    run_experiment()
