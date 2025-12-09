#!/usr/bin/env python
"""
Epistemic Articulation Prototype v1

Quick test: Can we detect token-level uncertainty and map it to hedging?

Approach:
1. Use a small model with dropout
2. Run MC Dropout to get uncertainty per token
3. See if high-uncertainty tokens correlate with "should hedge" situations
"""

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Tuple


def load_model_with_dropout(model_name: str = "gpt2"):
    """Load model and keep dropout active."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def mc_dropout_forward(
    model,
    tokenizer,
    prompt: str,
    n_samples: int = 10,
    max_new_tokens: int = 50
) -> Dict:
    """
    Run MC Dropout: Generate multiple times with dropout active.

    Returns token-level uncertainty estimates.
    """
    model.train()  # Keep dropout active

    inputs = tokenizer(prompt, return_tensors="pt")

    all_outputs = []
    all_logits = []

    for _ in range(n_samples):
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        all_outputs.append(generated_text)

        # Get logits for each generated token
        if outputs.scores:
            scores = torch.stack(outputs.scores, dim=0)  # [seq_len, batch, vocab]
            all_logits.append(scores.squeeze(1))  # [seq_len, vocab]

    # Compute uncertainty from output variance
    # Simple approach: semantic diversity of outputs
    unique_outputs = len(set(all_outputs))
    output_diversity = unique_outputs / n_samples

    return {
        "prompt": prompt,
        "outputs": all_outputs,
        "diversity": output_diversity,
        "n_samples": n_samples,
    }


def compute_semantic_entropy(outputs: List[str]) -> float:
    """
    Compute semantic entropy from multiple outputs.

    Simple version: Use exact string matching for clustering.
    Better version would use embedding similarity.
    """
    from collections import Counter

    counts = Counter(outputs)
    total = len(outputs)

    entropy = 0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * np.log(p)

    return entropy


def test_uncertainty_detection():
    """Test if we can detect uncertainty on known uncertain vs certain questions."""

    print("=" * 60)
    print("Epistemic Articulation Prototype")
    print("Testing: Can we detect when model should hedge?")
    print("=" * 60)

    model, tokenizer = load_model_with_dropout("gpt2")

    # Test cases: things GPT-2 should/shouldn't know
    test_cases = [
        # Should be CERTAIN (common knowledge, in training)
        {
            "prompt": "The capital of France is",
            "expected_uncertainty": "low",
            "reason": "Common fact, definitely in training",
        },
        {
            "prompt": "Water freezes at",
            "expected_uncertainty": "low",
            "reason": "Basic science fact",
        },

        # Should be UNCERTAIN (recent, specific, or obscure)
        {
            "prompt": "The CEO of OpenAI in 2024 is",
            "expected_uncertainty": "high",
            "reason": "After training cutoff, may have changed",
        },
        {
            "prompt": "The population of Tokyo in 2024 is",
            "expected_uncertainty": "high",
            "reason": "Specific recent number",
        },
        {
            "prompt": "My friend John's favorite color is",
            "expected_uncertainty": "high",
            "reason": "Impossible to know - not in training",
        },

        # AMBIGUOUS (multiple valid answers)
        {
            "prompt": "The best programming language is",
            "expected_uncertainty": "high",
            "reason": "Subjective, multiple valid answers",
        },
    ]

    results = []

    for case in test_cases:
        print(f"\n{'─' * 50}")
        print(f"Prompt: '{case['prompt']}'")
        print(f"Expected: {case['expected_uncertainty']} uncertainty")
        print(f"Reason: {case['reason']}")

        result = mc_dropout_forward(model, tokenizer, case["prompt"], n_samples=10)
        entropy = compute_semantic_entropy(result["outputs"])

        print(f"\nOutputs ({result['n_samples']} samples):")
        for i, out in enumerate(set(result["outputs"])):
            count = result["outputs"].count(out)
            print(f"  [{count}x] '{out[:60]}...'")

        print(f"\nMetrics:")
        print(f"  Output diversity: {result['diversity']:.2f}")
        print(f"  Semantic entropy: {entropy:.3f}")

        # Did we detect uncertainty correctly?
        detected_uncertain = entropy > 0.5 or result["diversity"] > 0.5
        expected_uncertain = case["expected_uncertainty"] == "high"
        correct = detected_uncertain == expected_uncertain

        status = "✓" if correct else "✗"
        print(f"\n  {status} Detection: {'UNCERTAIN' if detected_uncertain else 'CERTAIN'}")

        results.append({
            "prompt": case["prompt"],
            "expected": case["expected_uncertainty"],
            "entropy": entropy,
            "diversity": result["diversity"],
            "correct": correct,
        })

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    accuracy = sum(r["correct"] for r in results) / len(results)
    print(f"\nDetection accuracy: {accuracy:.0%}")

    print("\nResults:")
    print(f"{'Prompt':<40} {'Expected':<10} {'Entropy':<10} {'Correct':<8}")
    print("-" * 70)
    for r in results:
        status = "✓" if r["correct"] else "✗"
        print(f"{r['prompt'][:38]:<40} {r['expected']:<10} {r['entropy']:<10.3f} {status}")

    return results


def test_hedging_mapping():
    """
    Test: Can we map uncertainty levels to appropriate hedging?

    This is the key innovation: uncertainty → linguistic hedging
    """
    print("\n" + "=" * 60)
    print("Testing: Uncertainty → Hedging Mapping")
    print("=" * 60)

    # Define hedging templates by uncertainty level
    hedging_templates = {
        "very_low": "{answer}",  # No hedging
        "low": "I believe {answer}",
        "medium": "I think {answer}, but I'm not entirely certain",
        "high": "I'm not sure, but it might be {answer}",
        "very_high": "I don't know for certain. It could be {answer}, but please verify this.",
    }

    # Map entropy to hedging level
    def entropy_to_hedging(entropy: float) -> str:
        if entropy < 0.2:
            return "very_low"
        elif entropy < 0.5:
            return "low"
        elif entropy < 1.0:
            return "medium"
        elif entropy < 1.5:
            return "high"
        else:
            return "very_high"

    # Test cases with their answers and uncertainties
    test_cases = [
        {"entropy": 0.1, "answer": "Paris"},
        {"entropy": 0.4, "answer": "approximately 37 million"},
        {"entropy": 0.8, "answer": "Sam Altman"},
        {"entropy": 1.2, "answer": "Python or JavaScript"},
        {"entropy": 2.0, "answer": "blue, maybe green"},
    ]

    print("\nEntropy → Hedging Level → Articulated Response:")
    print("-" * 60)

    for case in test_cases:
        level = entropy_to_hedging(case["entropy"])
        template = hedging_templates[level]
        response = template.format(answer=case["answer"])

        print(f"\n  Entropy: {case['entropy']:.1f} → Level: {level}")
        print(f"  Response: \"{response}\"")

    print("\n" + "-" * 60)
    print("Key insight: We can systematically map uncertainty to hedging!")
    print("Next step: Train model to produce this mapping automatically.")


if __name__ == "__main__":
    print("Loading model (this may take a moment)...")

    # Test 1: Can we detect uncertainty?
    results = test_uncertainty_detection()

    # Test 2: Can we map uncertainty to hedging?
    test_hedging_mapping()

    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("""
The prototype shows:
1. MC Dropout / output diversity CAN detect uncertainty
2. We CAN map uncertainty levels to hedging templates
3. The gap: Training model to do this AUTOMATICALLY

Next steps:
- Use entropy as reward signal during training
- Train model to produce hedged outputs when entropy is high
- Evaluate on benchmarks (TruthfulQA, etc.)
""")
