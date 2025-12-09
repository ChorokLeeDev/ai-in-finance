#!/usr/bin/env python
"""
Training Design for Epistemic Articulation

Goal: Train model to produce hedging language when its entropy is high.

Approach: Self-supervised reward modeling
1. Generate response from base model
2. Compute token-level entropy during generation
3. Check if hedging language appears at high-entropy points
4. Reward alignment between entropy and hedging

This file designs and tests the training pipeline components.
"""

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Tuple
import re


# =============================================================================
# Hedging Detection
# =============================================================================

HEDGING_PHRASES = [
    # Uncertainty markers
    "I think", "I believe", "I'm not sure", "I'm uncertain",
    "possibly", "probably", "maybe", "perhaps",
    "might be", "could be", "may be",
    "it seems", "it appears", "it looks like",

    # Qualification
    "generally", "typically", "usually", "often",
    "in most cases", "as far as I know",

    # Explicit uncertainty
    "I don't know", "I'm not certain", "I cannot be sure",
    "this is uncertain", "this may not be accurate",

    # Source hedging (for RAG)
    "according to", "based on", "the document says",
    "I found that", "it's reported that",
]

CONFIDENT_PHRASES = [
    # Strong assertions
    "definitely", "certainly", "absolutely", "clearly",
    "obviously", "undoubtedly", "without doubt",
    "the answer is", "it is", "this is",
]


def detect_hedging(text: str) -> Dict:
    """Detect hedging and confident phrases in text."""
    text_lower = text.lower()

    hedging_found = []
    confident_found = []

    for phrase in HEDGING_PHRASES:
        if phrase.lower() in text_lower:
            hedging_found.append(phrase)

    for phrase in CONFIDENT_PHRASES:
        if phrase.lower() in text_lower:
            confident_found.append(phrase)

    hedging_score = len(hedging_found) - len(confident_found)

    return {
        "hedging_phrases": hedging_found,
        "confident_phrases": confident_found,
        "hedging_score": hedging_score,
        "has_hedging": len(hedging_found) > 0,
    }


# =============================================================================
# Entropy Computation
# =============================================================================

def compute_response_entropy(
    model,
    tokenizer,
    prompt: str,
    response: str
) -> List[float]:
    """
    Compute entropy at each token position of the response.

    Returns list of entropies, one per response token.
    """
    full_text = prompt + response
    inputs = tokenizer(full_text, return_tensors="pt")
    prompt_length = len(tokenizer(prompt)["input_ids"])

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0]  # [seq_len, vocab]

    # Compute entropy for response tokens
    entropies = []
    for i in range(prompt_length - 1, len(logits) - 1):
        token_logits = logits[i]
        probs = F.softmax(token_logits, dim=-1)
        # Use top-k for numerical stability
        top_probs, _ = torch.topk(probs, k=1000)
        entropy = -torch.sum(top_probs * torch.log(top_probs + 1e-10)).item()
        entropies.append(entropy)

    return entropies


def categorize_entropy(entropies: List[float]) -> Dict:
    """Categorize response based on entropy profile."""
    if not entropies:
        return {"category": "empty", "mean": 0, "max": 0}

    mean_entropy = np.mean(entropies)
    max_entropy = np.max(entropies)

    if mean_entropy < 3.0:
        category = "confident"
    elif mean_entropy < 5.0:
        category = "moderate"
    else:
        category = "uncertain"

    return {
        "category": category,
        "mean": mean_entropy,
        "max": max_entropy,
        "profile": entropies,
    }


# =============================================================================
# Reward Computation
# =============================================================================

def compute_alignment_reward(
    entropy_category: str,
    hedging_detected: Dict
) -> float:
    """
    Compute reward based on alignment between entropy and hedging.

    Good alignment:
    - High entropy + hedging present → positive reward
    - Low entropy + no hedging → positive reward

    Bad alignment:
    - High entropy + confident language → negative reward
    - Low entropy + excessive hedging → negative reward
    """
    has_hedging = hedging_detected["has_hedging"]
    hedging_score = hedging_detected["hedging_score"]

    if entropy_category == "uncertain":
        # Should have hedging
        if has_hedging:
            return 1.0 + 0.2 * hedging_score  # Bonus for more hedging
        else:
            return -1.0  # Penalty for no hedging when uncertain

    elif entropy_category == "confident":
        # Should NOT have hedging
        if has_hedging:
            return -0.5  # Slight penalty for unnecessary hedging
        else:
            return 0.5  # Reward for confident when appropriate

    else:  # moderate
        # Either is acceptable
        return 0.0 + 0.1 * hedging_score


# =============================================================================
# Dataset Generation
# =============================================================================

def generate_training_examples(
    model,
    tokenizer,
    prompts: List[str],
    n_samples_per_prompt: int = 5
) -> List[Dict]:
    """
    Generate training examples with entropy and hedging labels.

    For each prompt:
    1. Generate multiple responses
    2. Compute entropy for each
    3. Detect hedging
    4. Compute alignment reward
    5. Select best/worst for contrastive training
    """
    examples = []

    for prompt in prompts:
        prompt_examples = []

        for _ in range(n_samples_per_prompt):
            # Generate response
            inputs = tokenizer(prompt, return_tensors="pt")

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.8,
                    pad_token_id=tokenizer.pad_token_id,
                )

            response = tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )

            # Compute entropy
            entropies = compute_response_entropy(model, tokenizer, prompt, response)
            entropy_info = categorize_entropy(entropies)

            # Detect hedging
            hedging_info = detect_hedging(response)

            # Compute reward
            reward = compute_alignment_reward(
                entropy_info["category"],
                hedging_info
            )

            prompt_examples.append({
                "prompt": prompt,
                "response": response,
                "entropy": entropy_info,
                "hedging": hedging_info,
                "reward": reward,
            })

        # Sort by reward
        prompt_examples.sort(key=lambda x: x["reward"], reverse=True)

        # Keep best and worst for contrastive learning
        if len(prompt_examples) >= 2:
            examples.append({
                "prompt": prompt,
                "chosen": prompt_examples[0],  # Best alignment
                "rejected": prompt_examples[-1],  # Worst alignment
            })

    return examples


# =============================================================================
# Training Loop (Pseudo-code for DPO)
# =============================================================================

def create_dpo_dataset(examples: List[Dict]) -> List[Dict]:
    """
    Format examples for Direct Preference Optimization.

    DPO format:
    {
        "prompt": str,
        "chosen": str,  # Response with good entropy-hedging alignment
        "rejected": str,  # Response with bad alignment
    }
    """
    dpo_data = []

    for ex in examples:
        dpo_data.append({
            "prompt": ex["prompt"],
            "chosen": ex["chosen"]["response"],
            "rejected": ex["rejected"]["response"],
            "chosen_reward": ex["chosen"]["reward"],
            "rejected_reward": ex["rejected"]["reward"],
        })

    return dpo_data


# =============================================================================
# Testing
# =============================================================================

def test_pipeline():
    """Test the training pipeline components."""

    print("=" * 70)
    print("Testing Epistemic Articulation Training Pipeline")
    print("=" * 70)

    print("\n1. Loading model...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    # Test hedging detection
    print("\n2. Testing hedging detection...")
    test_texts = [
        "Paris is the capital of France.",
        "I think Paris might be the capital, but I'm not certain.",
        "The answer is definitely Paris, without any doubt.",
    ]

    for text in test_texts:
        result = detect_hedging(text)
        print(f"\n  '{text[:50]}...'")
        print(f"    Hedging: {result['hedging_phrases']}")
        print(f"    Confident: {result['confident_phrases']}")
        print(f"    Score: {result['hedging_score']}")

    # Test entropy computation
    print("\n3. Testing entropy computation...")
    prompts = [
        "The capital of France is",
        "The meaning of life is",
    ]

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        entropies = compute_response_entropy(model, tokenizer, prompt, response)
        entropy_info = categorize_entropy(entropies)

        print(f"\n  Prompt: '{prompt}'")
        print(f"  Response: '{response[:50]}...'")
        print(f"  Entropy: mean={entropy_info['mean']:.2f}, max={entropy_info['max']:.2f}")
        print(f"  Category: {entropy_info['category']}")

    # Test reward computation
    print("\n4. Testing reward computation...")
    test_cases = [
        ("uncertain", {"has_hedging": True, "hedging_score": 2}),
        ("uncertain", {"has_hedging": False, "hedging_score": -1}),
        ("confident", {"has_hedging": False, "hedging_score": 0}),
        ("confident", {"has_hedging": True, "hedging_score": 1}),
    ]

    for entropy_cat, hedging in test_cases:
        reward = compute_alignment_reward(entropy_cat, hedging)
        print(f"  {entropy_cat:12s} + hedging={str(hedging['has_hedging']):5s} → reward={reward:+.2f}")

    # Test dataset generation
    print("\n5. Testing dataset generation...")
    test_prompts = [
        "The president of the United States is",
        "The best way to learn programming is",
    ]

    examples = generate_training_examples(model, tokenizer, test_prompts, n_samples_per_prompt=3)

    for ex in examples:
        print(f"\n  Prompt: '{ex['prompt']}'")
        print(f"  Chosen (reward={ex['chosen']['reward']:.2f}):")
        print(f"    '{ex['chosen']['response'][:60]}...'")
        print(f"  Rejected (reward={ex['rejected']['reward']:.2f}):")
        print(f"    '{ex['rejected']['response'][:60]}...'")

    # Create DPO dataset
    print("\n6. Creating DPO dataset...")
    dpo_data = create_dpo_dataset(examples)

    print(f"  Generated {len(dpo_data)} preference pairs")

    print("\n" + "=" * 70)
    print("PIPELINE TEST COMPLETE")
    print("=" * 70)
    print("""
Summary:
1. Hedging detection ✓
2. Entropy computation ✓
3. Reward computation ✓
4. Dataset generation ✓
5. DPO formatting ✓

Next steps:
- Run DPO training with trl library
- Scale to larger model (Llama-7B)
- Evaluate on TruthfulQA
""")


if __name__ == "__main__":
    test_pipeline()
