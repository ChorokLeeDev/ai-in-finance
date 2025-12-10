#!/usr/bin/env python
"""
Adversarial Distractor Experiment

Create distractors that MAXIMIZE similarity to query while being WRONG.
This is the true test - if simple similarity fails here, can MoE-RAG help?

Strategy: Distractors repeat key query terms but give wrong/incomplete answers.
"""

import os
import sys
import numpy as np
from typing import List, Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# Adversarial Corpus - Distractors designed to have HIGH similarity
# =============================================================================

ADVERSARIAL_TOPICS = [
    {
        "name": "password_reset",
        "query": "What are the exact steps to reset my password?",
        "correct": """To reset your password, follow these exact steps:
1. Click "Forgot Password" on the login page
2. Enter your email address
3. Check your inbox for the reset link
4. Click the link and enter your new password
5. Confirm and save""",

        # Adversarial distractors - repeat "reset password" and "steps" but wrong info
        "distractors": [
            """Password Reset Steps - Coming Soon
We're currently updating our password reset steps. The new password reset process
will include enhanced security steps. Check back for the exact steps to reset
your password. Reset password functionality is temporarily unavailable.""",

            """Why Reset Your Password? Understanding Password Reset Steps
Before you reset your password, consider these steps: Think about why you need
to reset. The steps to reset your password should be taken seriously.
Password reset steps vary by account type. Reset password responsibly.""",

            """Password Reset Steps for Admins Only
These are the exact steps to reset your password if you're an administrator.
Admin password reset steps differ from regular steps. To reset admin password,
contact IT. Regular users cannot use these reset password steps.""",

            """Password Reset - Steps Not Required
Did you know? In some cases you don't need steps to reset your password.
Auto-reset your password without following steps. Reset password automatically.
No exact steps needed - our AI handles the reset password process.""",
        ]
    },
    {
        "name": "refund_days",
        "query": "How many days do I have to request a refund?",
        "correct": """Refund Policy: You have exactly 30 days from purchase to request a refund.
After 30 days, refunds are not available. Request your refund within this period.""",

        # Adversarial - repeat "days" and "refund" but wrong/ambiguous numbers
        "distractors": [
            """Refund Processing Days
Once you request a refund, it takes 5-7 business days to process.
Refund days vary by payment method. Request refund early to account for
processing days. How many days for refund depends on your bank.""",

            """Days to Refund: Multiple Policies
Different products have different refund days. Some items have 14 days,
others have 60 days for refund. The days you have to request a refund
depends on the category. Check product page for refund days.""",

            """Refund Request Days - Holiday Schedule
During holidays, refund request days may be extended. How many days for
holiday refunds? Typically extra days are added. Request refund with
holiday days in mind. Days for refund processing increase during peak.""",

            """How Many Days to Request Refund - FAQ
Q: How many days for refund? A: Days vary. Q: Request refund when?
A: Within the allowed days. Q: How many business days? A: Multiple days.
Refund days are explained in our terms. Request refund during business days.""",
        ]
    },
    {
        "name": "api_limit",
        "query": "What is the API rate limit for free tier users?",
        "correct": """API Rate Limits for Free Tier:
- 100 requests per minute
- 1,000 requests per hour
- 10,000 requests per day
Exceeding limits returns 429 error.""",

        # Adversarial - repeat "API rate limit" and "free tier" but wrong numbers
        "distractors": [
            """API Rate Limit Information
API rate limits protect our free tier users. The API rate limit varies.
Free tier API rate limits are designed for testing. API rate limit
exceeded? Upgrade from free tier. Rate limit API calls responsibly.""",

            """Free Tier API Rate Limit Comparison
How does our free tier API rate limit compare? Other services' API rate
limits for free tier vary. Free tier rate limit is competitive. API
rate limit free tier users fairly. Compare free tier API rate limits.""",

            """API Rate Limit Changes - Free Tier Update
We're updating API rate limits for free tier users. New API rate limit
coming soon. Free tier rate limit improvements planned. What API rate
limit for free tier? Stay tuned for rate limit announcements.""",

            """Understanding API Rate Limit - Free Tier Guide
API rate limits explained for free tier users. Rate limit your API calls.
Free tier API rate limit basics. What is rate limiting? API rate limit
protects servers. Free tier users should understand rate limit concepts.""",
        ]
    },
    {
        "name": "shipping_days",
        "query": "How long does standard shipping take?",
        "correct": """Standard Shipping: 5-7 business days
Orders ship within 24 hours. Tracking provided via email.
Business days exclude weekends and holidays.""",

        # Adversarial - repeat "standard shipping" and "days" but wrong info
        "distractors": [
            """Standard Shipping Days May Vary
How long does standard shipping take? It depends. Standard shipping days
vary by location. Shipping standard times differ. Long shipping? Choose
express. Standard shipping take what you expect - or does it?""",

            """Standard Shipping vs Express - Days Comparison
Standard shipping takes longer than express. How long? More days.
Standard shipping: more days. Express: fewer days. Shipping days for
standard: variable. Take standard shipping if days don't matter.""",

            """How Long Until Standard Shipping Improves?
We're working on standard shipping times. How long until faster? Days
away. Standard shipping take priority? Soon. Long-term shipping improvements.
Does standard shipping take too long? Feedback on days appreciated.""",

            """Standard Shipping Time - Days Processing
Before standard shipping, processing takes days. How long for processing?
1-3 days. Then standard shipping begins. Shipping take additional days.
Long process: processing days + standard shipping days = total days.""",
        ]
    },
    {
        "name": "cancel_subscription",
        "query": "How do I cancel my subscription?",
        "correct": """To cancel your subscription:
1. Go to Account Settings
2. Click "Subscription"
3. Select "Cancel Subscription"
4. Confirm cancellation
Access continues until billing period ends.""",

        # Adversarial - repeat "cancel subscription" but don't give steps
        "distractors": [
            """Before You Cancel Subscription
Thinking about cancel subscription? Consider: Why cancel? Cancel subscription
means losing benefits. Before you cancel subscription, try pause. Cancel
subscription only if sure. How do I keep you from cancel subscription?""",

            """Cancel Subscription? Read This First
Don't cancel subscription yet! Why cancel? Cancel subscription alternatives:
downgrade, pause, transfer. How do I convince you not to cancel subscription?
Cancel subscription = lose everything. I want to cancel subscription? Wait.""",

            """Cancel Subscription Policy
Our cancel subscription policy is fair. Cancel subscription anytime.
When you cancel subscription, some rules apply. Cancel subscription
terms vary. How do I understand cancel subscription? Read our policy.""",

            """Cancel Subscription Feedback
Did you cancel subscription? How was cancel subscription process?
Cancel subscription survey. I want to cancel subscription because...
Why cancel subscription? Help us improve. Cancel subscription reasons.""",
        ]
    },
]


def create_adversarial_corpus() -> Tuple[List[Dict], List[Dict]]:
    """Create corpus with adversarial distractors."""
    documents = []
    queries = []

    for topic in ADVERSARIAL_TOPICS:
        topic_name = topic["name"]

        # Add correct document
        correct_id = f"{topic_name}_correct"
        documents.append({
            "id": correct_id,
            "text": topic["correct"],
            "topic": topic_name,
            "is_correct": True,
        })

        # Add adversarial distractors
        for i, distractor in enumerate(topic["distractors"]):
            doc_id = f"{topic_name}_adversarial_{i}"
            documents.append({
                "id": doc_id,
                "text": distractor,
                "topic": topic_name,
                "is_correct": False,
            })

        queries.append({
            "query": topic["query"],
            "correct_doc_id": correct_id,
            "topic": topic_name,
        })

    return documents, queries


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


def analyze_adversarial_similarity(documents: List[Dict], queries: List[Dict]):
    """Analyze if adversarial distractors actually have high similarity."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("\n" + "=" * 60)
    print("Adversarial Similarity Analysis")
    print("Goal: Distractors should have SIMILAR scores to correct docs")
    print("=" * 60)

    total_gap = 0
    adversarial_wins = 0
    close_calls = 0

    for q in queries:
        query_emb = model.encode([q["query"]])[0]

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
            distractor_sims.append((d_sim, d["id"]))

        distractor_sims.sort(reverse=True)
        best_dist_sim, best_dist_id = distractor_sims[0]
        gap = correct_sim - best_dist_sim
        total_gap += gap

        print(f"\nQuery: {q['query'][:50]}...")
        print(f"  Correct similarity:     {correct_sim:.4f}")
        print(f"  Best distractor:        {best_dist_sim:.4f} ({best_dist_id.split('_')[-1]})")
        print(f"  Gap:                    {gap:+.4f}")

        if gap < 0:
            print("  *** ADVERSARIAL WINS! Simple similarity would fail ***")
            adversarial_wins += 1
        elif gap < 0.05:
            print("  ** Close call - within 0.05 **")
            close_calls += 1

    print("\n" + "-" * 40)
    print(f"Average gap: {total_gap/len(queries):.4f}")
    print(f"Adversarial wins (gap < 0): {adversarial_wins}/{len(queries)}")
    print(f"Close calls (gap < 0.05): {close_calls}/{len(queries)}")

    if adversarial_wins == 0 and close_calls == 0:
        print("\nDIAGNOSIS: Adversarial distractors aren't fooling embeddings.")
        print("Simple similarity will still win easily.")
    elif adversarial_wins > 0:
        print(f"\nDIAGNOSIS: {adversarial_wins} cases where simple similarity fails!")
        print("This is where MoE-RAG could potentially help.")


def run_simple_similarity(documents: List[Dict], queries: List[Dict]) -> Dict:
    """Simple embedding similarity."""
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

        metrics = evaluate_retrieval(retrieved_ids, q["correct_doc_id"])
        metrics["query"] = q["query"]
        results.append(metrics)

    return {
        "method": "Simple Similarity",
        "mrr": np.mean([r["mrr"] for r in results]),
        "recall@1": np.mean([r["recall@1"] for r in results]),
        "recall@3": np.mean([r["recall@3"] for r in results]),
        "details": results,
    }


def run_moe_rag(documents: List[Dict], queries: List[Dict], epochs: int = 10) -> Dict:
    """MoE-RAG."""
    from moe_rag import MoERAG

    texts = [d["text"] for d in documents]

    rag = MoERAG.from_texts(texts)
    rag.train(epochs=epochs, verbose=False)

    results = []
    for q in queries:
        retrieved = rag.retrieve(q["query"], top_k=5)

        retrieved_ids = []
        for r in retrieved:
            for d in documents:
                if r.text[:80] in d["text"] or d["text"][:80] in r.text:
                    retrieved_ids.append(d["id"])
                    break

        metrics = evaluate_retrieval(retrieved_ids, q["correct_doc_id"])
        metrics["query"] = q["query"]
        results.append(metrics)

    return {
        "method": f"MoE-RAG ({epochs} epochs)",
        "mrr": np.mean([r["mrr"] for r in results]),
        "recall@1": np.mean([r["recall@1"] for r in results]),
        "recall@3": np.mean([r["recall@3"] for r in results]),
        "details": results,
    }


def run_experiment():
    """Run adversarial experiment."""
    print("=" * 60)
    print("Adversarial Distractor Experiment")
    print("Testing if keyword-stuffed distractors can fool embeddings")
    print("=" * 60)

    documents, queries = create_adversarial_corpus()
    print(f"\nCreated {len(documents)} documents ({len(queries)} topics)")

    # Analyze similarity landscape
    analyze_adversarial_similarity(documents, queries)

    # Run methods
    print("\n" + "-" * 40)
    print("Running Simple Similarity...")
    simple_result = run_simple_similarity(documents, queries)

    print("\n" + "-" * 40)
    print("Running MoE-RAG...")
    moe_result = run_moe_rag(documents, queries, epochs=15)

    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\n{'Method':<25} {'MRR':<10} {'R@1':<10} {'R@3':<10}")
    print("-" * 55)

    for r in [simple_result, moe_result]:
        print(f"{r['method']:<25} {r['mrr']:<10.4f} {r['recall@1']:<10.2%} {r['recall@3']:<10.2%}")

    # Analysis
    print("\n" + "-" * 40)
    print("Per-Query Ranks (lower is better)")
    print("-" * 40)

    for i, q in enumerate(queries):
        simple_rank = simple_result["details"][i]["rank"]
        moe_rank = moe_result["details"][i]["rank"]
        print(f"{q['topic']:<20} Simple: {simple_rank}  MoE: {moe_rank}")


if __name__ == "__main__":
    run_experiment()
