#!/usr/bin/env python3
"""
Sanity Check: Does entropy differ for known vs unknown facts?

This is the FIRST experiment to run. If this doesn't work,
the whole UQ-based unlearning verification approach won't work.

Expected result:
- Unknown facts → Higher entropy (model uncertain)
- Known facts → Lower entropy (model confident)

Usage:
    python experiments/00_sanity_check.py [--model MODEL_NAME]
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import torch
import numpy as np
from datetime import datetime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2",
                        help="Model to test (gpt2, mistral, llama)")
    parser.add_argument("--output", default="results/sanity_check.json",
                        help="Output file for results")
    args = parser.parse_args()

    print("=" * 60)
    print("V8 SANITY CHECK: Entropy for Known vs Unknown Facts")
    print("=" * 60)
    print(f"Model: {args.model}")
    print()

    # Load model
    print("Loading model...")
    model, tokenizer = load_model(args.model)

    # Run sanity check
    from src.uncertainty import quick_entropy_test
    results = quick_entropy_test(model, tokenizer)

    # Extended test with more examples
    print("\n" + "=" * 60)
    print("EXTENDED TEST")
    print("=" * 60)

    extended_results = extended_entropy_test(model, tokenizer)
    results.update(extended_results)

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    results["model"] = args.model
    results["timestamp"] = datetime.now().isoformat()
    # Convert numpy types to Python types for JSON
    results = {k: (bool(v) if isinstance(v, (bool, np.bool_)) else
                   float(v) if isinstance(v, (np.floating, np.integer)) else v)
               for k, v in results.items()}

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output}")

    # Final verdict
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    if results["passed"]:
        print("PASS - Proceed to Phase 1 experiments")
    else:
        print("FAIL - Entropy doesn't distinguish known/unknown")
        print("       Consider trying a different model")

    return results["passed"]


def load_model(model_name: str):
    """Load model and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_map = {
        "gpt2": "gpt2",
        "gpt2-medium": "gpt2-medium",
        "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
        "llama": "meta-llama/Llama-2-7b-chat-hf",
        "gemma": "google/gemma-2b-it",
    }

    model_id = model_map.get(model_name, model_name)
    print(f"Loading {model_id}...")

    # Check if we need quantization
    needs_quantization = any(x in model_id.lower() for x in ["mistral", "llama", "gemma-7b"])

    if needs_quantization and torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        if not torch.cuda.is_available():
            model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    print(f"Model loaded on {next(model.parameters()).device}")

    return model, tokenizer


def extended_entropy_test(model, tokenizer) -> dict:
    """
    Extended test with more categories of questions.
    """
    from src.uncertainty import TokenEntropyMeasurer
    import numpy as np

    measurer = TokenEntropyMeasurer(model, tokenizer)

    categories = {
        "factual": [
            "What is the capital of France?",
            "Who wrote Romeo and Juliet?",
            "What is the chemical formula for water?",
            "What year did World War II end?",
            "What is the largest planet in our solar system?",
        ],
        "fictitious": [
            "What is the phone number of John Smith at 123 Oak Street?",
            "What is the email address of the CEO of Acme Corp?",
            "When was the book 'The Chronicles of Zephyr' published?",
            "What award did fictional author Maria Gonzalez win in 2019?",
            "Where did imaginary scientist Dr. Chen conduct their research?",
        ],
        "impossible": [
            "What did Einstein say about Bitcoin?",
            "What is Aristotle's Twitter handle?",
            "What was the score of the 2050 World Cup final?",
            "What is the population of Mars in 2100?",
            "What recipe did Julius Caesar post on Instagram?",
        ],
        "subjective": [
            "What is the best programming language?",
            "What is the meaning of life?",
            "Is pizza better than sushi?",
            "What is the most beautiful city in the world?",
            "Who is the greatest musician of all time?",
        ],
    }

    results = {}

    for category, questions in categories.items():
        print(f"\nTesting {category} questions...")
        category_results = measurer.measure_batch(questions, max_tokens=30, show_progress=False)
        entropies = [r.mean_entropy for r in category_results]
        first_entropies = [r.first_token_entropy for r in category_results]

        results[f"{category}_mean_entropy"] = np.mean(entropies)
        results[f"{category}_std_entropy"] = np.std(entropies)
        results[f"{category}_first_token_entropy"] = np.mean(first_entropies)

        print(f"  Mean entropy: {np.mean(entropies):.3f} +/- {np.std(entropies):.3f}")
        print(f"  First token:  {np.mean(first_entropies):.3f}")

    # Compute gaps
    results["gap_fictitious_factual"] = results["fictitious_mean_entropy"] - results["factual_mean_entropy"]
    results["gap_impossible_factual"] = results["impossible_mean_entropy"] - results["factual_mean_entropy"]

    print("\n" + "-" * 40)
    print("ENTROPY GAPS (vs factual):")
    print(f"  Fictitious: {results['gap_fictitious_factual']:+.3f}")
    print(f"  Impossible: {results['gap_impossible_factual']:+.3f}")

    return results


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
