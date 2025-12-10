#!/usr/bin/env python
"""
Generate DPO Dataset for Llama 3.1

Uses Llama's chat template and generates higher-quality preference pairs.
Designed for A100 GPU with 60+ GB VRAM.
"""

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict
import json
from pathlib import Path
from tqdm import tqdm

# Hedging detection
HEDGING_PHRASES = [
    "I think", "I believe", "I'm not sure", "I'm uncertain",
    "possibly", "probably", "maybe", "perhaps",
    "might be", "could be", "may be",
    "it seems", "it appears", "it looks like",
    "generally", "typically", "usually", "often",
    "in most cases", "as far as I know",
    "I don't know", "I'm not certain", "I cannot be sure",
    "this is uncertain", "this may not be accurate",
    "according to", "based on",
]

CONFIDENT_PHRASES = [
    "definitely", "certainly", "absolutely", "clearly",
    "obviously", "undoubtedly", "without doubt",
    "the answer is", "it is", "this is",
]


def detect_hedging(text: str) -> Dict:
    """Detect hedging and confident phrases."""
    text_lower = text.lower()
    hedging_found = [p for p in HEDGING_PHRASES if p.lower() in text_lower]
    confident_found = [p for p in CONFIDENT_PHRASES if p.lower() in text_lower]

    return {
        "hedging_phrases": hedging_found,
        "confident_phrases": confident_found,
        "hedging_score": len(hedging_found) - len(confident_found),
        "has_hedging": len(hedging_found) > 0,
    }


# =============================================================================
# Prompts designed for instruction-tuned models
# =============================================================================

PROMPTS = {
    "factual_certain": [
        "What is the capital of France?",
        "What is 2 + 2?",
        "Who wrote Romeo and Juliet?",
        "What is the chemical symbol for water?",
        "In what year did World War II end?",
    ],
    "factual_uncertain": [
        "What is the exact population of Tokyo right now?",
        "Who will win the next presidential election?",
        "What will the stock market do tomorrow?",
        "What is the cure for cancer?",
        "What happened to Malaysia Airlines Flight 370?",
    ],
    "subjective": [
        "What is the best programming language?",
        "What is the meaning of life?",
        "What is the best movie ever made?",
        "Is pineapple good on pizza?",
        "What career should I pursue?",
    ],
    "impossible": [
        "What is my favorite color?",
        "What did I have for breakfast?",
        "What is my mother's name?",
        "What am I thinking right now?",
        "What will I do next week?",
    ],
    "controversial": [
        "Is God real?",
        "What is the best political system?",
        "Should abortion be legal?",
        "Is capitalism better than socialism?",
        "Are humans causing climate change?",
    ],
}


def format_chat_prompt(tokenizer, question: str) -> str:
    """Format question using Llama chat template."""
    messages = [
        {"role": "user", "content": question}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def compute_response_entropy(model, tokenizer, prompt: str, response: str) -> Dict:
    """Compute entropy of response tokens."""
    full_text = prompt + response
    inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
    prompt_length = len(tokenizer(prompt)["input_ids"])

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0]

    entropies = []
    for i in range(prompt_length - 1, len(logits) - 1):
        probs = F.softmax(logits[i], dim=-1)
        top_probs, _ = torch.topk(probs, k=min(1000, probs.shape[0]))
        entropy = -torch.sum(top_probs * torch.log(top_probs + 1e-10)).item()
        entropies.append(entropy)

    if not entropies:
        return {"mean": 0, "max": 0, "category": "empty"}

    mean_entropy = np.mean(entropies)
    category = "confident" if mean_entropy < 3.0 else ("moderate" if mean_entropy < 5.0 else "uncertain")

    return {"mean": mean_entropy, "max": np.max(entropies), "category": category}


def compute_alignment_reward(entropy_category: str, hedging: Dict) -> float:
    """Compute reward for entropy-hedging alignment."""
    has_hedging = hedging["has_hedging"]
    hedging_score = hedging["hedging_score"]

    if entropy_category == "uncertain":
        return 1.0 + 0.2 * hedging_score if has_hedging else -1.0
    elif entropy_category == "confident":
        return -0.5 if has_hedging else 0.5
    else:
        return 0.1 * hedging_score


def generate_dataset(
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    n_samples: int = 5,
    output_path: str = "dpo_dataset_llama.json",
):
    """Generate preference dataset using Llama."""
    print(f"Loading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    dataset = []

    for category, questions in PROMPTS.items():
        print(f"\nProcessing {category}...")

        for question in tqdm(questions, desc=category):
            samples = []
            prompt = format_chat_prompt(tokenizer, question)

            for _ in range(n_samples):
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=150,
                        do_sample=True,
                        temperature=0.8,
                        top_p=0.9,
                        pad_token_id=tokenizer.pad_token_id,
                    )

                response = tokenizer.decode(
                    outputs[0][inputs.input_ids.shape[1]:],
                    skip_special_tokens=True
                )

                entropy = compute_response_entropy(model, tokenizer, prompt, response)
                hedging = detect_hedging(response)
                reward = compute_alignment_reward(entropy["category"], hedging)

                samples.append({
                    "response": response,
                    "entropy": entropy,
                    "hedging": hedging,
                    "reward": reward,
                })

            # Sort by reward and create preference pair
            samples.sort(key=lambda x: x["reward"], reverse=True)

            if len(samples) >= 2 and samples[0]["reward"] != samples[-1]["reward"]:
                dataset.append({
                    "prompt": question,
                    "category": category,
                    "chosen": samples[0]["response"],
                    "rejected": samples[-1]["response"],
                    "chosen_reward": samples[0]["reward"],
                    "rejected_reward": samples[-1]["reward"],
                    "chosen_entropy": samples[0]["entropy"]["mean"],
                    "rejected_entropy": samples[-1]["entropy"]["mean"],
                })

    # Save dataset
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"\nSaved {len(dataset)} preference pairs to {output_path}")

    # Analysis
    print("\n" + "=" * 50)
    print("DATASET ANALYSIS")
    print("=" * 50)

    for cat in PROMPTS.keys():
        cat_items = [d for d in dataset if d["category"] == cat]
        if cat_items:
            avg_margin = np.mean([d["chosen_reward"] - d["rejected_reward"] for d in cat_items])
            print(f"{cat:20s}: {len(cat_items):3d} pairs, avg margin: {avg_margin:.3f}")

    return dataset


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--output", default="dpo_dataset_llama.json")
    args = parser.parse_args()

    generate_dataset(args.model, args.samples, args.output)
