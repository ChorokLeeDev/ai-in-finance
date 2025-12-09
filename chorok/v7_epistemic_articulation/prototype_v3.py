#!/usr/bin/env python
"""
Epistemic Articulation Prototype v3

Simpler approach: Just measure FIRST TOKEN entropy.
This tells us how uncertain the model is about the immediate continuation.
"""

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_first_token_entropy(model, tokenizer, prompt: str, top_k: int = 10):
    """
    Get entropy of first predicted token distribution.

    Returns entropy and top-k predictions.
    """
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        next_token_logits = outputs.logits[0, -1, :]

    # Softmax
    probs = F.softmax(next_token_logits, dim=-1)

    # Entropy (only over significant probability mass to avoid NaN)
    # Filter to top 1000 tokens
    top_probs, top_indices = torch.topk(probs, k=1000)
    entropy = -torch.sum(top_probs * torch.log(top_probs + 1e-10)).item()

    # Get top-k tokens
    top_k_probs, top_k_indices = torch.topk(probs, k=top_k)
    top_tokens = [(tokenizer.decode(idx), prob.item()) for idx, prob in zip(top_k_indices, top_k_probs)]

    return {
        "entropy": entropy,
        "top_tokens": top_tokens,
        "top_1_prob": top_tokens[0][1],
    }


def main():
    print("=" * 70)
    print("Epistemic Articulation Prototype v3")
    print("Measuring first-token entropy as uncertainty signal")
    print("=" * 70)

    print("\nLoading GPT-2...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.eval()

    test_cases = [
        # Should be LOW entropy (single dominant answer)
        ("The capital of France is", "low"),
        ("1 + 1 equals", "low"),
        ("Water is composed of hydrogen and", "low"),
        ("The sun rises in the", "low"),

        # Should be HIGH entropy (many valid options)
        ("The best movie is", "high"),
        ("I think that", "high"),
        ("The meaning of life is", "high"),
        ("Tomorrow I will", "high"),

        # Medium (some constraints but multiple options)
        ("The president of the United States", "medium"),
        ("Python is a programming", "low"),
    ]

    results = []

    print("\n" + "-" * 70)
    print(f"{'Prompt':<40} {'Exp':<8} {'Entropy':<10} {'Top-1 Prob':<10} {'Top-1 Token'}")
    print("-" * 70)

    for prompt, expected in test_cases:
        result = get_first_token_entropy(model, tokenizer, prompt)

        top_token = result["top_tokens"][0][0].strip()
        top_prob = result["top_1_prob"]
        entropy = result["entropy"]

        print(f"{prompt:<40} {expected:<8} {entropy:<10.3f} {top_prob:<10.3f} '{top_token}'")

        results.append({
            "prompt": prompt,
            "expected": expected,
            "entropy": entropy,
            "top_prob": top_prob,
            "top_token": top_token,
        })

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    low_entropy = [r for r in results if r["expected"] == "low"]
    high_entropy = [r for r in results if r["expected"] == "high"]

    low_mean = np.mean([r["entropy"] for r in low_entropy])
    high_mean = np.mean([r["entropy"] for r in high_entropy])
    low_prob = np.mean([r["top_prob"] for r in low_entropy])
    high_prob = np.mean([r["top_prob"] for r in high_entropy])

    print(f"\n'Should be LOW entropy' cases:")
    print(f"  Mean entropy: {low_mean:.3f}")
    print(f"  Mean top-1 prob: {low_prob:.3f}")

    print(f"\n'Should be HIGH entropy' cases:")
    print(f"  Mean entropy: {high_mean:.3f}")
    print(f"  Mean top-1 prob: {high_prob:.3f}")

    separation = high_mean - low_mean
    print(f"\nEntropy separation: {separation:.3f}")

    if separation > 0.5:
        print("\n✓ GOOD SEPARATION: Entropy distinguishes certain vs uncertain prompts!")
    else:
        print(f"\n⚠ WEAK SEPARATION: Only {separation:.3f} difference")

    # Detailed view of high-entropy cases
    print("\n" + "-" * 70)
    print("Top predictions for HIGH-entropy prompts:")
    print("-" * 70)

    for prompt, expected in test_cases:
        if expected != "high":
            continue

        result = get_first_token_entropy(model, tokenizer, prompt)
        print(f"\n'{prompt}'")
        print(f"  Entropy: {result['entropy']:.3f}")
        print(f"  Top predictions:")
        for tok, prob in result["top_tokens"][:5]:
            print(f"    {prob:.3f} '{tok.strip()}'")

    # Show how to use for hedging
    print("\n" + "=" * 70)
    print("APPLICATION: Entropy-Based Hedging")
    print("=" * 70)

    print("""
The key insight:
- LOW entropy (< 4.0) → Model is confident → No hedging needed
- MEDIUM entropy (4-6) → Some uncertainty → "I believe..."
- HIGH entropy (> 6.0) → Very uncertain → "I'm not sure, but..."

Training approach:
1. Generate response
2. Compute entropy at each token
3. If entropy spikes, reward hedging language at that point
4. Train end-to-end to learn this mapping
""")

    # Demo
    print("\nDemo hedging decisions:")
    print("-" * 50)

    for r in results:
        entropy = r["entropy"]
        prompt = r["prompt"]

        if entropy < 4.0:
            hedge = "[confident]"
        elif entropy < 6.0:
            hedge = "[I believe]"
        else:
            hedge = "[I'm not sure]"

        print(f"  {entropy:.2f} {hedge:<15} → '{prompt}' ...")


if __name__ == "__main__":
    main()
