#!/usr/bin/env python
"""
Messy Real-World Experiment

Simulate actual enterprise RAG chaos:
- Documents: Mixed formatting, typos, inconsistent structure, noise
- Queries: Informal, misspelled, incomplete, context-dependent

This is the true test of "dump and use" - can it handle real mess?
"""

import os
import sys
import numpy as np
from typing import List, Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# Messy Enterprise Documents - Real chaos
# =============================================================================

MESSY_DOCUMENTS = [
    # Document 1: Slack thread export (messy conversation)
    {
        "id": "slack_thread_123",
        "text": """#support-escalations
@john.doe [10:32 AM]
hey anyone know how to reset a users password?? the portal is being weird

@sarah.smith [10:35 AM]
which portal lol we have like 5

@john.doe [10:36 AM]
the main one... admin portal

@mike.chen [10:40 AM]
go to admin.internal.com/users, search the email, click the three dots menu, theres a "reset pwd" option there

@john.doe [10:41 AM]
oh found it thx!!

@sarah.smith [10:42 AM]
btw mike u need admin role for that, regular support cant do it

@mike.chen [10:43 AM]
oh right yeah u need admin access. @john.doe do u have admin?

@john.doe [10:45 AM]
nope lol. ok ill ask my manager
""",
        "topic": "password_reset",
        "has_answer": True,  # The actual process IS here, buried in chat
    },

    # Document 2: Wiki page with outdated info mixed in
    {
        "id": "wiki_password_old",
        "text": """Password Management Guide
Last updated: 2019-03-15 (!!NEEDS UPDATE!!)

== Overview ==
This page describes the password reset proccess.

== For End Users ==
1. Click "Forgot Password" on login page
2. Check email for reset link
3. Follow link to set new pasword

== For Admins == (DEPRECATED - see new admin portal docs)
Old method: SSH to server, run reset_pwd.sh script
New method: Use admin portal (see separate doc)

== Troubleshooting ==
Q: Reset email not received?
A: Check spam folder. If still missing, contact IT@company.com

NOTE: This doc may be out of date. The new system was deployed in 2022.
Contact security team for questions.

Tags: password, reset, admin, users, authentication
Related: SSO Setup, MFA Configuration, User Provisioning
""",
        "topic": "password_reset",
        "has_answer": True,  # Partial answer with noise
    },

    # Document 3: Meeting notes (rambling, off-topic)
    {
        "id": "meeting_notes_q3",
        "text": """Q3 Planning Meeting Notes - 2024-09-15

Attendees: Product, Engineering, Support

1. Revenue update (skip - see finance deck)

2. Support tickets review
   - Password issues up 30% this quarter
   - Main complaint: reset process too slow
   - ACTION: Engineering to look at faster reset flow
   - NOTE: Current reset takes 24hrs due to security review

3. Product roadmap
   - New dashboard launching Q4
   - Mobile app delayed (resourcing issues)

4. Misc
   - Holiday schedule reminder
   - Office snacks survey results (pizza won)
   - Parking lot discussion: should we change password policy?

Next meeting: Oct 15

---
Random notes from sidebar convos:
- Jim mentioned the API is slow
- Need to update the onboarding doc
- Sara's birthday is next week
""",
        "topic": "password_reset",
        "has_answer": False,  # Mentions password but no reset steps
    },

    # Document 4: Technical runbook with jargon
    {
        "id": "runbook_auth",
        "text": """RUNBOOK: AUTH-001 - Password Reset Procedure (PROD)

SEVERITY: P3
ONCALL: security-oncall@company.com
LAST INCIDENT: INC-4521

PRE-REQS:
- VPN connected
- Admin creds (see vault: /secret/admin/portal)
- MFA device ready

STEPS:
1. Verify user identity via Zendesk ticket
2. Access admin portal: https://admin.internal.company.com
3. Navigate: Users > Search > [enter email]
4. Click user row > Actions dropdown > "Force Password Reset"
5. Select reset method:
   - Email (default, 15min expiry)
   - SMS (requires phone on file)
   - Manual (generates temp pwd, expires 1hr)
6. Document in ticket: reset initiated, method used
7. Close ticket after user confirms access

ROLLBACK:
If user locked out after reset, see RUNBOOK: AUTH-002

METRICS:
- SLA: 4 hours for P3
- Avg resolution: 45 min

CHANGELOG:
2024-01-15: Added SMS option
2023-06-01: Removed legacy SSH method
""",
        "topic": "password_reset",
        "has_answer": True,  # This is the REAL answer but buried in jargon
    },

    # Document 5: Customer email thread
    {
        "id": "email_thread_pw",
        "text": """From: angry.customer@gmail.com
To: support@company.com
Subject: RE: RE: RE: STILL CANT LOGIN!!!

This is ridiculous. I've been trying to reset my password for 3 DAYS.

Your website says "click forgot password" but when I click it NOTHING HAPPENS.
I've tried Chrome, Firefox, even IE. Nothing works.

I need access to my account ASAP for work. This is urgent!!!

---Previous message---
From: support@company.com
To: angry.customer@gmail.com

Hi,

Thank you for contacting support. To reset your password:
1. Go to company.com/login
2. Click "Forgot Password"
3. Enter your email
4. Check inbox (and spam)

If you don't receive the email within 10 minutes, please try again
or reply to this thread.

Best,
Support Team

---Previous message---
From: angry.customer@gmail.com

how do i reset my passwrod? cant login and need help asap

my email is angry.customer@gmail.com
""",
        "topic": "password_reset",
        "has_answer": True,  # Support reply has the steps
    },

    # Document 6: Completely irrelevant but keyword-heavy
    {
        "id": "blog_security",
        "text": """5 Tips for Strong Passwords in 2024

Passwords are the first line of defense for your accounts. Here's how to
stay secure:

1. Use long passwords (16+ characters)
   - "correct horse battery staple" > "P@ssw0rd!"

2. Never reuse passwords
   - One breach = all accounts compromised

3. Use a password manager
   - Bitwarden, 1Password, LastPass (had breach tho)

4. Enable MFA everywhere
   - Even if password stolen, they need your phone

5. Reset passwords after breaches
   - Check haveibeenpwned.com regularly

Remember: The strongest password is the one you don't have to remember!

#cybersecurity #passwords #infosec #tips
""",
        "topic": "password_reset",
        "has_answer": False,  # About passwords but NOT how to reset
    },

    # More messy docs for other topics...
    {
        "id": "pricing_pdf_extract",
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
- API access (100 req/min)

Enterprise.....Contact Sales
- Custom everything
- SLA included
- Dedicated CSM

*Prices in USD. Annual discount: 20% off
**Subject to change without notice

Page 1 of 3
[NEXT PAGE MISSING]
""",
        "topic": "pricing",
        "has_answer": True,
    },

    {
        "id": "api_stackoverflow",
        "text": """Q: Rate limit exceeded error 429 - what's the limit?? [closed as duplicate]

I keep getting 429 errors from the API. Documentation doesn't say what the
limit is. Anyone know?

---
A1: (accepted, 45 upvotes)
The limits are:
- Free: 100/min, 1000/hr
- Pro: 1000/min, 10000/hr

But honestly the docs are terrible, I had to find this by trial and error lol

---
A2: (3 upvotes)
Try adding exponential backoff. Here's what worked for me:
```python
import time
def api_call_with_retry(func):
    for i in range(5):
        try:
            return func()
        except RateLimitError:
            time.sleep(2 ** i)
```

---
A3: (-2 downvotes)
Just upgrade to pro, problem solved
""",
        "topic": "api_limits",
        "has_answer": True,
    },
]

# =============================================================================
# Messy Real-World Queries - How people actually ask
# =============================================================================

MESSY_QUERIES = [
    {
        "query": "how do i reset password",  # No punctuation, terse
        "correct_doc_id": "runbook_auth",  # The runbook has the real process
        "topic": "password_reset",
    },
    {
        "query": "cant login need to change pasword asap help",  # Typo, urgent
        "correct_doc_id": "runbook_auth",
        "topic": "password_reset",
    },
    {
        "query": "whats the admin way to reset a users pwd?",  # Slang
        "correct_doc_id": "runbook_auth",
        "topic": "password_reset",
    },
    {
        "query": "API rate limit how many requests",
        "correct_doc_id": "api_stackoverflow",
        "topic": "api_limits",
    },
    {
        "query": "how much does pro cost",
        "correct_doc_id": "pricing_pdf_extract",
        "topic": "pricing",
    },
]


def create_messy_corpus() -> Tuple[List[Dict], List[Dict]]:
    """Create messy real-world corpus."""
    return MESSY_DOCUMENTS, MESSY_QUERIES


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


def run_simple_similarity(documents: List[Dict], queries: List[Dict]) -> Dict:
    """Simple embedding similarity."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    texts = [d["text"] for d in documents]
    doc_embeddings = model.encode(texts)

    print("\nSimple Similarity Results:")
    print("-" * 60)

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

        print(f"\nQuery: '{q['query']}'")
        print(f"  Correct doc: {q['correct_doc_id']}")
        print(f"  Retrieved:")
        for doc_id, score in zip(retrieved_ids[:3], top_scores[:3]):
            marker = "✓" if doc_id == q["correct_doc_id"] else " "
            print(f"    {marker} [{score:.3f}] {doc_id}")
        print(f"  Rank: {metrics['rank']}")

    return {
        "method": "Simple Similarity",
        "mrr": np.mean([r["mrr"] for r in results]),
        "recall@1": np.mean([r["recall@1"] for r in results]),
        "recall@3": np.mean([r["recall@3"] for r in results]),
        "details": results,
    }


def run_moe_rag(documents: List[Dict], queries: List[Dict], epochs: int = 15) -> Dict:
    """MoE-RAG."""
    from moe_rag import MoERAG

    texts = [d["text"] for d in documents]

    rag = MoERAG.from_texts(texts)
    rag.train(epochs=epochs, verbose=False)

    print(f"\nMoE-RAG ({epochs} epochs) Results:")
    print("-" * 60)

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

        print(f"\nQuery: '{q['query']}'")
        print(f"  Correct doc: {q['correct_doc_id']}")
        print(f"  Retrieved: {retrieved_ids[:3]}")
        print(f"  Head used: {retrieved[0].head_used if retrieved else 'N/A'}")
        print(f"  Rank: {metrics['rank']}")

    return {
        "method": f"MoE-RAG ({epochs} epochs)",
        "mrr": np.mean([r["mrr"] for r in results]),
        "recall@1": np.mean([r["recall@1"] for r in results]),
        "recall@3": np.mean([r["recall@3"] for r in results]),
        "details": results,
    }


def run_experiment():
    """Run messy real-world experiment."""
    print("=" * 60)
    print("MESSY REAL-WORLD EXPERIMENT")
    print("Testing with actual enterprise chaos")
    print("=" * 60)

    documents, queries = create_messy_corpus()

    print(f"\nDocuments: {len(documents)}")
    for d in documents:
        has_ans = "✓" if d.get("has_answer") else "✗"
        print(f"  [{has_ans}] {d['id']}: {d['text'][:50]}...")

    print(f"\nQueries: {len(queries)}")
    for q in queries:
        print(f"  '{q['query']}' → {q['correct_doc_id']}")

    # Run methods
    simple_result = run_simple_similarity(documents, queries)
    moe_result = run_moe_rag(documents, queries, epochs=15)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"\n{'Method':<25} {'MRR':<10} {'R@1':<10} {'R@3':<10}")
    print("-" * 55)
    for r in [simple_result, moe_result]:
        print(f"{r['method']:<25} {r['mrr']:<10.4f} {r['recall@1']:<10.2%} {r['recall@3']:<10.2%}")

    # Analysis
    print("\n" + "-" * 40)
    print("KEY OBSERVATIONS:")
    print("-" * 40)

    if simple_result["mrr"] < 0.5:
        print("• Simple similarity struggles with messy data!")
        print("  This could be where MoE-RAG helps.")
    else:
        print("• Simple similarity still handles messy data well.")

    if moe_result["mrr"] > simple_result["mrr"]:
        print(f"• MoE-RAG WINS by {moe_result['mrr'] - simple_result['mrr']:.4f} MRR!")
    else:
        print(f"• Simple similarity still better by {simple_result['mrr'] - moe_result['mrr']:.4f} MRR")


if __name__ == "__main__":
    run_experiment()
