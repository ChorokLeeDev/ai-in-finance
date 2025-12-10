#!/usr/bin/env python
"""
Does "One Size Fits All" Chunking Work?

SAP AI Core uses fixed character chunking. Is this actually bad?
Let's test different chunking strategies on different document types.

Hypothesis: The "right" chunking depends on document type:
- Code: semantic (function/class boundaries)
- Docs: paragraph/section boundaries
- Chat: message boundaries
- Tables: row/column boundaries
"""

import numpy as np
from typing import List, Dict, Tuple
import re


# =============================================================================
# Different Document Types
# =============================================================================

DOCUMENTS = {
    "code": '''def reset_password(user_id: str, method: str = "email") -> bool:
    """Reset a user's password.

    Args:
        user_id: The user's unique identifier
        method: Reset method - "email", "sms", or "manual"

    Returns:
        True if reset initiated successfully
    """
    user = get_user(user_id)
    if not user:
        raise UserNotFoundError(f"User {user_id} not found")

    if method == "email":
        token = generate_reset_token(user_id)
        send_email(user.email, "Password Reset", f"Click here: {token}")
        return True
    elif method == "sms":
        if not user.phone:
            raise ValueError("No phone number on file")
        code = generate_sms_code()
        send_sms(user.phone, f"Your reset code: {code}")
        return True
    elif method == "manual":
        temp_password = generate_temp_password()
        set_temp_password(user_id, temp_password, expiry_hours=1)
        return temp_password
    else:
        raise ValueError(f"Unknown method: {method}")


def get_user(user_id: str) -> Optional[User]:
    """Fetch user from database."""
    return db.users.find_one({"id": user_id})


def generate_reset_token(user_id: str, expiry_minutes: int = 15) -> str:
    """Generate a secure reset token."""
    token = secrets.token_urlsafe(32)
    db.tokens.insert({"user_id": user_id, "token": token, "expires": time.time() + expiry_minutes * 60})
    return token
''',

    "documentation": '''# Password Reset Guide

## Overview

This document describes how to reset user passwords in our system. There are three methods available: email, SMS, and manual reset.

## Prerequisites

Before resetting a password, ensure you have:
- Admin access to the user management portal
- The user's email address or phone number
- A valid support ticket for the reset request

## Method 1: Email Reset

The most common and secure method. Steps:

1. Navigate to admin.company.com/users
2. Search for the user by email
3. Click the user row to open details
4. Select "Actions" → "Force Password Reset"
5. Choose "Email" as the method
6. The user receives a reset link valid for 15 minutes

## Method 2: SMS Reset

Use when the user cannot access email. Requirements:
- Phone number must be on file
- User must have SMS capability

Steps are the same as email, but select "SMS" in step 5.

## Method 3: Manual Reset

Last resort when other methods fail:
1. Generate a temporary password
2. Communicate securely to user
3. Password expires in 1 hour
4. User must change on first login

## Troubleshooting

**Q: User didn't receive reset email?**
A: Check spam folder. If still missing, try SMS method.

**Q: Reset link expired?**
A: Generate a new one. Links are only valid for 15 minutes.
''',

    "chat": '''#support-escalations

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

@mike.chen [10:47 AM]
also heads up the reset link only lasts 15 mins so tell them to check email quick

@sarah.smith [10:48 AM]
or use sms if they cant get email. thats option 2 in the dropdown
''',
}


# =============================================================================
# Chunking Strategies
# =============================================================================

def chunk_fixed_chars(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """SAP AI Core style: fixed character chunking."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return [c.strip() for c in chunks if c.strip()]


def chunk_by_paragraph(text: str) -> List[str]:
    """Split by paragraphs/double newlines."""
    chunks = re.split(r'\n\s*\n', text)
    return [c.strip() for c in chunks if c.strip()]


def chunk_by_section(text: str) -> List[str]:
    """Split by markdown headers."""
    # Split on ## headers
    chunks = re.split(r'\n(?=##?\s)', text)
    return [c.strip() for c in chunks if c.strip()]


def chunk_by_message(text: str) -> List[str]:
    """Split by chat messages (@ mentions with timestamps)."""
    chunks = re.split(r'\n(?=@\w+\s*\[)', text)
    return [c.strip() for c in chunks if c.strip()]


def chunk_by_function(text: str) -> List[str]:
    """Split by function definitions."""
    chunks = re.split(r'\n(?=def\s)', text)
    return [c.strip() for c in chunks if c.strip()]


def chunk_semantic(text: str, doc_type: str) -> List[str]:
    """Choose chunking based on document type."""
    if doc_type == "code":
        return chunk_by_function(text)
    elif doc_type == "documentation":
        return chunk_by_section(text)
    elif doc_type == "chat":
        return chunk_by_message(text)
    else:
        return chunk_by_paragraph(text)


# =============================================================================
# Evaluation
# =============================================================================

TEST_QUERIES = {
    "code": [
        ("how to reset password", "reset_password"),  # Should find the function
        ("send sms reset code", "sms"),
        ("generate token", "generate_reset_token"),
    ],
    "documentation": [
        ("how to reset password", "Overview"),
        ("email reset steps", "Method 1"),
        ("troubleshooting reset", "Troubleshooting"),
    ],
    "chat": [
        ("how to reset password", "admin.internal.com"),
        ("admin access needed", "admin role"),
        ("reset link expiry", "15 mins"),
    ],
}


def evaluate_chunking(doc_type: str, chunks: List[str], queries: List[Tuple[str, str]]) -> Dict:
    """Evaluate if correct chunk is retrieved for each query."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    chunk_embeddings = model.encode(chunks)
    results = []

    for query, expected_content in queries:
        query_emb = model.encode([query])[0]

        similarities = np.dot(chunk_embeddings, query_emb) / (
            np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(query_emb)
        )

        top_idx = np.argmax(similarities)
        top_chunk = chunks[top_idx]

        # Check if expected content is in top chunk
        found = expected_content.lower() in top_chunk.lower()

        results.append({
            "query": query,
            "expected": expected_content,
            "found": found,
            "top_chunk_preview": top_chunk[:100],
            "similarity": similarities[top_idx],
        })

    return {
        "accuracy": np.mean([r["found"] for r in results]),
        "details": results,
    }


def run_experiment():
    """Compare chunking strategies across document types."""
    print("="*70)
    print("DOES CHUNKING STRATEGY MATTER?")
    print("Testing: Fixed chars (SAP style) vs Semantic chunking")
    print("="*70)

    strategies = {
        "fixed_500": lambda t, dt: chunk_fixed_chars(t, 500, 50),
        "fixed_1000": lambda t, dt: chunk_fixed_chars(t, 1000, 100),
        "semantic": chunk_semantic,
    }

    all_results = {}

    for doc_type, text in DOCUMENTS.items():
        print(f"\n{'='*60}")
        print(f"Document Type: {doc_type.upper()}")
        print(f"{'='*60}")
        print(f"Document length: {len(text)} chars")

        queries = TEST_QUERIES.get(doc_type, [])
        if not queries:
            continue

        doc_results = {}

        for strategy_name, chunk_fn in strategies.items():
            chunks = chunk_fn(text, doc_type)
            print(f"\n  {strategy_name}: {len(chunks)} chunks")

            # Show chunk previews
            for i, chunk in enumerate(chunks[:3]):
                preview = chunk[:60].replace('\n', ' ')
                print(f"    Chunk {i}: '{preview}...'")

            result = evaluate_chunking(doc_type, chunks, queries)
            doc_results[strategy_name] = result

            print(f"\n  Accuracy: {result['accuracy']:.0%}")
            for r in result["details"]:
                status = "✓" if r["found"] else "✗"
                print(f"    {status} '{r['query']}' → found '{r['expected']}'? {r['found']}")

        all_results[doc_type] = doc_results

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print(f"\n{'Doc Type':<15} {'Fixed 500':<12} {'Fixed 1000':<12} {'Semantic':<12}")
    print("-"*51)

    for doc_type, results in all_results.items():
        fixed_500 = results["fixed_500"]["accuracy"]
        fixed_1000 = results["fixed_1000"]["accuracy"]
        semantic = results["semantic"]["accuracy"]

        # Highlight winner
        scores = {"fixed_500": fixed_500, "fixed_1000": fixed_1000, "semantic": semantic}
        winner = max(scores, key=scores.get)

        row = f"{doc_type:<15}"
        for name, acc in scores.items():
            marker = "*" if name == winner else " "
            row += f"{acc:.0%}{marker:<10}"
        print(row)

    # Analysis
    print("\n" + "-"*60)
    print("ANALYSIS")
    print("-"*60)

    # Calculate overall
    overall = {s: [] for s in strategies}
    for doc_type, results in all_results.items():
        for strategy, result in results.items():
            overall[strategy].append(result["accuracy"])

    print("\nOverall accuracy:")
    for strategy, accs in overall.items():
        print(f"  {strategy}: {np.mean(accs):.0%}")

    semantic_better = np.mean(overall["semantic"]) > np.mean(overall["fixed_500"])

    if semantic_better:
        improvement = (np.mean(overall["semantic"]) - np.mean(overall["fixed_500"])) * 100
        print(f"\n→ Semantic chunking is {improvement:.0f}% better overall!")
        print("  Document-aware chunking DOES matter.")
    else:
        print("\n→ Fixed chunking is competitive!")
        print("  'One size fits all' may be acceptable for many use cases.")


if __name__ == "__main__":
    run_experiment()
