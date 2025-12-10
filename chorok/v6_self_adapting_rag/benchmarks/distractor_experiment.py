#!/usr/bin/env python
"""
Distractor Experiment

Test whether MoE-RAG can learn to distinguish correct answers from
high-similarity distractors.

This is the key test: if MoE can't beat simple similarity here,
the approach is fundamentally limited.

Setup:
- 10 topics, 5 docs per topic (1 correct, 4 distractors)
- Distractors share vocabulary but have wrong/missing key info
- Compare MoE vs simple similarity
"""

import os
import sys
import time
import json
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# Distractor Corpus Generation
# =============================================================================

TOPICS = [
    {
        "name": "password_reset",
        "query": "What are the exact steps to reset my password?",
        "correct": """Password Reset Instructions

To reset your password, follow these exact steps:
1. Click the "Forgot Password" link on the login page
2. Enter your registered email address
3. Check your inbox for the reset link (valid for 24 hours)
4. Click the link and enter your new password
5. Password must be at least 12 characters with one number
6. Click "Confirm" to save your new password

Your account will be temporarily locked after 3 failed attempts.""",

        "distractors": [
            """Password Security Best Practices

Strong passwords are essential for account security. We recommend:
- Using a password manager to generate random passwords
- Never reusing passwords across different sites
- Enabling two-factor authentication when available
- Changing passwords every 90 days

For more security tips, visit our security center.""",

            """Account Recovery Options

If you're locked out of your account, you have several options:
- Password reset via email
- SMS verification to your registered phone
- Security questions (if enabled)
- Contact customer support

Our support team is available 24/7 to assist with account issues.""",

            """Password Policy Update - January 2024

We've updated our password requirements. New passwords must:
- Be at least 8 characters long
- Include uppercase and lowercase letters
- Not contain your username or email

Existing passwords remain valid until your next password change.""",

            """Login Troubleshooting Guide

Having trouble logging in? Common issues include:
- Caps Lock enabled (passwords are case-sensitive)
- Browser cache issues - try clearing cookies
- Account may be locked after failed attempts
- Check if the service is experiencing downtime

For persistent issues, please reset your password or contact support.""",
        ]
    },
    {
        "name": "refund_policy",
        "query": "How many days do I have to request a refund?",
        "correct": """Refund Policy

You have exactly 30 days from the date of purchase to request a full refund.
To be eligible:
- Product must be unused and in original packaging
- You must have proof of purchase (receipt or order confirmation)
- Digital products are non-refundable after download

After 30 days, we offer store credit for an additional 60 days.
Refunds are processed within 5-7 business days.""",

        "distractors": [
            """Return Shipping Instructions

To return a product:
1. Print the prepaid shipping label from your account
2. Pack the item securely in its original box
3. Drop off at any authorized shipping location

Tracking information will be emailed once the package is scanned.""",

            """Warranty Information

All products come with a 1-year manufacturer warranty covering:
- Manufacturing defects
- Hardware malfunctions
- Component failures

Warranty does not cover physical damage, water damage, or normal wear.""",

            """Exchange Policy

Looking to exchange for a different size or color? Easy!
- Request exchange within 14 days of delivery
- Original item must be unworn with tags attached
- No additional shipping cost for first exchange

Exchanges are processed faster than refunds.""",

            """Order Cancellation

Need to cancel an order? Time is important:
- Orders can be cancelled within 2 hours of placement
- After 2 hours, orders enter processing and cannot be cancelled
- You can refuse delivery and return for refund instead

Check your order status in your account dashboard.""",
        ]
    },
    {
        "name": "api_rate_limit",
        "query": "What is the API rate limit for free tier users?",
        "correct": """API Rate Limits

Free tier rate limits:
- 100 requests per minute
- 1,000 requests per hour
- 10,000 requests per day

When you exceed the limit, you'll receive a 429 error.
Rate limits reset at the top of each period.

Upgrade to Pro for 10x higher limits.""",

        "distractors": [
            """API Authentication Guide

To authenticate API requests:
1. Generate an API key in your dashboard
2. Include the key in the Authorization header
3. Keys can be rotated at any time

Never share your API keys or commit them to version control.""",

            """API Pricing Plans

We offer flexible pricing for all needs:
- Free: Best for testing and small projects
- Pro ($29/month): For production applications
- Enterprise: Custom pricing for large scale

All plans include 99.9% uptime SLA.""",

            """API Error Codes

Common error codes you might encounter:
- 400: Bad Request - check your parameters
- 401: Unauthorized - verify your API key
- 404: Not Found - endpoint doesn't exist
- 500: Server Error - try again later

Full error documentation available in our API reference.""",

            """API Changelog

Recent API updates:
- v2.3: Added batch processing endpoint
- v2.2: Improved response times by 40%
- v2.1: New webhook support

Breaking changes are announced 30 days in advance.""",
        ]
    },
    {
        "name": "shipping_time",
        "query": "How long does standard shipping take?",
        "correct": """Shipping Options and Times

Standard Shipping: 5-7 business days
- Free for orders over $50
- $4.99 for orders under $50

Express Shipping: 2-3 business days
- $9.99 flat rate

Overnight Shipping: Next business day
- $24.99, order by 2pm local time

Business days exclude weekends and holidays.""",

        "distractors": [
            """International Shipping

We ship to over 50 countries worldwide.
International orders:
- Customs fees may apply
- Delivery times vary by destination
- Tracking available for all shipments

Some items may be restricted in certain countries.""",

            """Shipping Address Updates

Need to change your shipping address?
- Before shipment: Update in your account
- After shipment: Contact carrier directly
- PO Boxes: Not available for all shipping methods

Always verify your address before checkout.""",

            """Package Tracking

Track your order status:
1. Log into your account
2. Go to "Order History"
3. Click "Track Package"

Tracking updates may take 24 hours after shipment.""",

            """Delivery Instructions

Special delivery options:
- Leave at door: Select during checkout
- Signature required: For high-value items
- Hold at location: Pick up at carrier facility

Contact us within 24 hours for delivery issues.""",
        ]
    },
    {
        "name": "subscription_cancel",
        "query": "How do I cancel my subscription?",
        "correct": """Cancel Your Subscription

To cancel your subscription:
1. Go to Account Settings
2. Click "Subscription" tab
3. Select "Cancel Subscription"
4. Choose your cancellation reason
5. Confirm cancellation

Your access continues until the end of the current billing period.
You will not be charged again after cancellation.
You can reactivate anytime.""",

        "distractors": [
            """Subscription Plans Comparison

Choose the plan that fits your needs:
- Basic: 5 projects, 10GB storage
- Pro: Unlimited projects, 100GB storage
- Team: Everything in Pro + collaboration features

All plans include email support.""",

            """Billing FAQ

Common billing questions:
- When am I charged? On your signup anniversary date
- Can I change plans? Yes, anytime in settings
- Is there a contract? No, month-to-month

View your billing history in Account > Billing.""",

            """Pause Subscription

Need a break but don't want to lose your data?
Pause your subscription for up to 3 months:
- Your data is preserved
- No charges while paused
- Resume anytime

Pause option available in subscription settings.""",

            """Subscription Benefits

Your subscription includes:
- Access to all premium features
- Priority customer support
- Early access to new features
- Exclusive member discounts

Thank you for being a subscriber!""",
        ]
    },
]


def create_distractor_corpus() -> Tuple[List[Dict], List[Dict]]:
    """
    Create corpus with correct answers and distractors.

    Returns:
        (documents, queries) where each query has one correct doc
    """
    documents = []
    queries = []

    for topic in TOPICS:
        topic_name = topic["name"]

        # Add correct document
        correct_id = f"{topic_name}_correct"
        documents.append({
            "id": correct_id,
            "text": topic["correct"],
            "topic": topic_name,
            "is_correct": True,
        })

        # Add distractors
        for i, distractor in enumerate(topic["distractors"]):
            doc_id = f"{topic_name}_distractor_{i}"
            documents.append({
                "id": doc_id,
                "text": distractor,
                "topic": topic_name,
                "is_correct": False,
            })

        # Create query
        queries.append({
            "query": topic["query"],
            "correct_doc_id": correct_id,
            "topic": topic_name,
        })

    return documents, queries


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_retrieval(retrieved_ids: List[str], correct_id: str) -> Dict:
    """Compute metrics."""
    if not retrieved_ids:
        return {"mrr": 0, "recall@1": 0, "recall@3": 0, "rank": -1}

    rank = -1
    for i, rid in enumerate(retrieved_ids):
        if rid == correct_id:
            rank = i + 1
            break

    mrr = 1.0 / rank if rank > 0 else 0
    recall_1 = 1.0 if rank == 1 else 0.0
    recall_3 = 1.0 if 0 < rank <= 3 else 0.0

    return {"mrr": mrr, "recall@1": recall_1, "recall@3": recall_3, "rank": rank}


# =============================================================================
# Methods
# =============================================================================

def run_simple_similarity(documents: List[Dict], queries: List[Dict]) -> Dict:
    """Simple embedding similarity baseline."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    texts = [d["text"] for d in documents]
    doc_embeddings = model.encode(texts)

    results = []
    for q in queries:
        query_emb = model.encode([q["query"]])[0]

        # Cosine similarity
        similarities = np.dot(doc_embeddings, query_emb) / (
            np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(query_emb)
        )
        top_indices = np.argsort(similarities)[::-1][:5]
        retrieved_ids = [documents[i]["id"] for i in top_indices]

        metrics = evaluate_retrieval(retrieved_ids, q["correct_doc_id"])
        metrics["query"] = q["query"]
        metrics["retrieved"] = retrieved_ids[:3]
        results.append(metrics)

    return {
        "method": "Simple Similarity",
        "mrr": np.mean([r["mrr"] for r in results]),
        "recall@1": np.mean([r["recall@1"] for r in results]),
        "recall@3": np.mean([r["recall@3"] for r in results]),
        "details": results,
    }


def run_moe_rag(documents: List[Dict], queries: List[Dict], epochs: int = 10) -> Dict:
    """MoE-RAG with training."""
    from moe_rag import MoERAG

    texts = [d["text"] for d in documents]

    # Train
    start = time.time()
    rag = MoERAG.from_texts(texts)
    rag.train(epochs=epochs, verbose=False)
    train_time = time.time() - start

    results = []
    for q in queries:
        retrieved = rag.retrieve(q["query"], top_k=5)

        # Match retrieved chunks back to doc IDs
        retrieved_ids = []
        for r in retrieved:
            for d in documents:
                if r.text[:100] in d["text"] or d["text"][:100] in r.text:
                    retrieved_ids.append(d["id"])
                    break

        metrics = evaluate_retrieval(retrieved_ids, q["correct_doc_id"])
        metrics["query"] = q["query"]
        metrics["retrieved"] = retrieved_ids[:3]
        results.append(metrics)

    return {
        "method": "MoE-RAG",
        "mrr": np.mean([r["mrr"] for r in results]),
        "recall@1": np.mean([r["recall@1"] for r in results]),
        "recall@3": np.mean([r["recall@3"] for r in results]),
        "train_time": train_time,
        "details": results,
    }


def run_moe_rag_longer(documents: List[Dict], queries: List[Dict]) -> Dict:
    """MoE-RAG with more training."""
    return run_moe_rag(documents, queries, epochs=20)


# =============================================================================
# Analysis
# =============================================================================

def analyze_similarity_scores(documents: List[Dict], queries: List[Dict]):
    """
    Analyze why simple similarity might fail or succeed.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("\n" + "=" * 60)
    print("Similarity Analysis: Correct vs Distractors")
    print("=" * 60)

    for q in queries:
        query_emb = model.encode([q["query"]])[0]

        # Find correct doc and distractors
        correct_doc = None
        distractors = []

        for d in documents:
            if d["topic"] == q["topic"]:
                if d["is_correct"]:
                    correct_doc = d
                else:
                    distractors.append(d)

        if not correct_doc:
            continue

        # Compute similarities
        correct_emb = model.encode([correct_doc["text"]])[0]
        correct_sim = np.dot(query_emb, correct_emb) / (
            np.linalg.norm(query_emb) * np.linalg.norm(correct_emb)
        )

        distractor_sims = []
        for d in distractors:
            d_emb = model.encode([d["text"]])[0]
            d_sim = np.dot(query_emb, d_emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(d_emb)
            )
            distractor_sims.append(d_sim)

        max_distractor = max(distractor_sims) if distractor_sims else 0
        gap = correct_sim - max_distractor

        print(f"\nQuery: {q['query'][:50]}...")
        print(f"  Correct doc similarity:   {correct_sim:.4f}")
        print(f"  Best distractor similarity: {max_distractor:.4f}")
        print(f"  Gap (correct - distractor): {gap:+.4f}")

        if gap < 0:
            print("  *** DISTRACTOR WINS - This is a hard case! ***")
        elif gap < 0.05:
            print("  ** Close call - distractor nearly wins **")


# =============================================================================
# Main
# =============================================================================

def run_experiment():
    """Run the full distractor experiment."""
    print("=" * 60)
    print("Distractor Experiment")
    print("Can MoE-RAG distinguish correct answers from distractors?")
    print("=" * 60)

    # Create corpus
    documents, queries = create_distractor_corpus()
    print(f"\nCreated {len(documents)} documents ({len(queries)} topics)")
    print(f"Each topic: 1 correct + 4 distractors")

    # Analyze similarity landscape
    analyze_similarity_scores(documents, queries)

    # Run methods
    print("\n" + "-" * 40)
    print("Running Simple Similarity...")
    simple_result = run_simple_similarity(documents, queries)

    print("\n" + "-" * 40)
    print("Running MoE-RAG (10 epochs)...")
    moe_result = run_moe_rag(documents, queries, epochs=10)

    print("\n" + "-" * 40)
    print("Running MoE-RAG (20 epochs)...")
    moe_longer_result = run_moe_rag(documents, queries, epochs=20)

    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\n{'Method':<25} {'MRR':<10} {'R@1':<10} {'R@3':<10}")
    print("-" * 55)

    for r in [simple_result, moe_result, moe_longer_result]:
        print(f"{r['method']:<25} {r['mrr']:<10.4f} {r['recall@1']:<10.2%} {r['recall@3']:<10.2%}")

    # Per-query analysis
    print("\n" + "-" * 40)
    print("Per-Query Results (Rank of correct doc, lower is better)")
    print("-" * 40)

    print(f"\n{'Query':<40} {'Simple':<10} {'MoE-RAG':<10}")
    print("-" * 60)

    for i, q in enumerate(queries):
        simple_rank = simple_result["details"][i]["rank"]
        moe_rank = moe_result["details"][i]["rank"]

        simple_str = str(simple_rank) if simple_rank > 0 else "miss"
        moe_str = str(moe_rank) if moe_rank > 0 else "miss"

        # Highlight wins/losses
        if simple_rank > 0 and moe_rank > 0:
            if moe_rank < simple_rank:
                moe_str += " *"  # MoE wins
            elif simple_rank < moe_rank:
                simple_str += " *"  # Simple wins

        print(f"{q['query'][:40]:<40} {simple_str:<10} {moe_str:<10}")

    # Conclusion
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)

    simple_mrr = simple_result["mrr"]
    moe_mrr = moe_result["mrr"]

    if moe_mrr > simple_mrr + 0.05:
        print("\nMoE-RAG WINS: Learns to distinguish correct from distractors!")
        print("This validates the approach for distractor-heavy settings.")
    elif moe_mrr > simple_mrr:
        print("\nMarginal improvement. MoE-RAG slightly better but not significant.")
        print("Need more training or harder distractors to see clear benefit.")
    elif abs(moe_mrr - simple_mrr) < 0.02:
        print("\nTIE: MoE-RAG provides no benefit over simple similarity.")
        print("The learned attention doesn't help with distractors.")
    else:
        print("\nSimple similarity WINS. MoE-RAG training hurts performance.")
        print("Need to investigate why - possibly overfitting to training queries.")

    return {
        "simple": simple_result,
        "moe": moe_result,
        "moe_longer": moe_longer_result,
    }


if __name__ == "__main__":
    results = run_experiment()
