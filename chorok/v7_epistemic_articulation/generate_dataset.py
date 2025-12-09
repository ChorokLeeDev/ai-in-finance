#!/usr/bin/env python
"""
Generate Hedged Output Dataset from Entropy Signals

Creates a dataset of (prompt, chosen, rejected) pairs for DPO training.
- chosen: Response with good entropy-hedging alignment
- rejected: Response with bad alignment

Uses diverse prompts to capture different uncertainty scenarios.
"""

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict
import json
from pathlib import Path
from tqdm import tqdm

# Import from training_design
from training_design import (
    detect_hedging,
    compute_response_entropy,
    categorize_entropy,
    compute_alignment_reward,
)


# =============================================================================
# Prompt Templates
# =============================================================================

# Factual prompts (should be LOW entropy for well-known facts)
FACTUAL_PROMPTS = [
    "The capital of France is",
    "The capital of Japan is",
    "The capital of Germany is",
    "Water is composed of",
    "The speed of light is approximately",
    "The chemical symbol for gold is",
    "The largest planet in our solar system is",
    "The year World War II ended was",
    "The author of Romeo and Juliet is",
    "The atomic number of carbon is",
]

# Subjective prompts (should be HIGH entropy - multiple valid answers)
SUBJECTIVE_PROMPTS = [
    "The best programming language is",
    "The meaning of life is",
    "The most beautiful city is",
    "The best way to learn is",
    "The most important quality in a leader is",
    "The future of AI will be",
    "The ideal career choice is",
    "The key to happiness is",
    "The most influential person in history is",
    "The best approach to problem solving is",
]

# Ambiguous prompts (could go either way)
AMBIGUOUS_PROMPTS = [
    "The president of the United States",
    "The CEO of Apple",
    "The population of Tokyo",
    "The stock price of Tesla",
    "Tomorrow's weather will be",
    "The winner of the next election",
    "The release date of the next iPhone",
    "The score of the game tonight",
    "My friend's favorite color is",
    "What I had for breakfast was",
]

# Knowledge boundary prompts (tests model's limits)
KNOWLEDGE_BOUNDARY_PROMPTS = [
    "The capital of the newly formed country",
    "The latest discovery in quantum physics",
    "The president elected in 2030",
    "The recipe my grandmother used for",
    "The exact number of stars in the galaxy",
    "What happens after death is",
    "The cure for cancer will be discovered",
    "The true meaning of this ancient text",
    "What the future holds for humanity",
    "The solution to consciousness is",
]


def generate_response_with_entropy(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.8,
) -> Dict:
    """Generate a response and compute its entropy profile."""
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
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

    # Compute alignment reward
    reward = compute_alignment_reward(entropy_info["category"], hedging_info)

    return {
        "prompt": prompt,
        "response": response,
        "entropy": entropy_info,
        "hedging": hedging_info,
        "reward": reward,
    }


def generate_dataset(
    model,
    tokenizer,
    n_samples_per_prompt: int = 10,
    output_path: str = "dpo_dataset.json",
) -> List[Dict]:
    """
    Generate full dataset for DPO training.

    For each prompt category, generate multiple samples and select
    the best (chosen) and worst (rejected) for contrastive training.
    """
    all_prompts = (
        FACTUAL_PROMPTS +
        SUBJECTIVE_PROMPTS +
        AMBIGUOUS_PROMPTS +
        KNOWLEDGE_BOUNDARY_PROMPTS
    )

    print(f"Generating dataset from {len(all_prompts)} prompts...")
    print(f"Samples per prompt: {n_samples_per_prompt}")

    dataset = []

    for prompt in tqdm(all_prompts, desc="Processing prompts"):
        samples = []

        for _ in range(n_samples_per_prompt):
            try:
                sample = generate_response_with_entropy(model, tokenizer, prompt)
                samples.append(sample)
            except Exception as e:
                print(f"Error generating for '{prompt}': {e}")
                continue

        if len(samples) < 2:
            continue

        # Sort by reward (best alignment first)
        samples.sort(key=lambda x: x["reward"], reverse=True)

        # Create preference pair
        dataset.append({
            "prompt": prompt,
            "chosen": samples[0]["response"],
            "rejected": samples[-1]["response"],
            "chosen_reward": samples[0]["reward"],
            "rejected_reward": samples[-1]["reward"],
            "chosen_entropy": samples[0]["entropy"]["mean"],
            "rejected_entropy": samples[-1]["entropy"]["mean"],
            "chosen_hedging": samples[0]["hedging"]["hedging_phrases"],
            "rejected_hedging": samples[-1]["hedging"]["hedging_phrases"],
        })

    # Save dataset
    output_path = Path(output_path)
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"\nSaved {len(dataset)} preference pairs to {output_path}")

    return dataset


def analyze_dataset(dataset: List[Dict]):
    """Analyze the generated dataset."""
    print("\n" + "=" * 70)
    print("DATASET ANALYSIS")
    print("=" * 70)

    rewards = [d["chosen_reward"] - d["rejected_reward"] for d in dataset]
    print(f"\nReward margin (chosen - rejected):")
    print(f"  Mean: {np.mean(rewards):.3f}")
    print(f"  Min:  {np.min(rewards):.3f}")
    print(f"  Max:  {np.max(rewards):.3f}")

    # Count hedging presence
    chosen_with_hedging = sum(1 for d in dataset if len(d["chosen_hedging"]) > 0)
    rejected_with_hedging = sum(1 for d in dataset if len(d["rejected_hedging"]) > 0)

    print(f"\nHedging presence:")
    print(f"  Chosen responses with hedging:   {chosen_with_hedging}/{len(dataset)}")
    print(f"  Rejected responses with hedging: {rejected_with_hedging}/{len(dataset)}")

    # Entropy stats
    chosen_entropy = [d["chosen_entropy"] for d in dataset]
    rejected_entropy = [d["rejected_entropy"] for d in dataset]

    print(f"\nEntropy:")
    print(f"  Chosen mean:   {np.mean(chosen_entropy):.3f}")
    print(f"  Rejected mean: {np.mean(rejected_entropy):.3f}")

    # Show some examples
    print("\n" + "-" * 70)
    print("EXAMPLE PAIRS")
    print("-" * 70)

    for i, d in enumerate(dataset[:5]):
        print(f"\n[{i+1}] Prompt: '{d['prompt']}'")
        print(f"    Chosen (r={d['chosen_reward']:.2f}): '{d['chosen'][:60]}...'")
        print(f"    Rejected (r={d['rejected_reward']:.2f}): '{d['rejected'][:60]}...'")


def main():
    print("=" * 70)
    print("Epistemic Articulation Dataset Generation")
    print("=" * 70)

    print("\n1. Loading model...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    print("\n2. Generating dataset...")
    dataset = generate_dataset(
        model,
        tokenizer,
        n_samples_per_prompt=10,
        output_path="dpo_dataset.json",
    )

    print("\n3. Analyzing dataset...")
    analyze_dataset(dataset)

    print("\n" + "=" * 70)
    print("DATASET GENERATION COMPLETE")
    print("=" * 70)
    print("""
Next steps:
1. Run DPO training: python train_dpo.py
2. Evaluate on TruthfulQA
3. Human evaluation study
""")


if __name__ == "__main__":
    main()
