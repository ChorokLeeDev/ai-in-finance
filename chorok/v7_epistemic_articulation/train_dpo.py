#!/usr/bin/env python
"""
DPO Training for Epistemic Articulation

Train a model to align linguistic hedging with internal uncertainty.

Uses Direct Preference Optimization (DPO) to train on preference pairs where:
- Chosen: Response with good entropy-hedging alignment
- Rejected: Response with bad alignment

Requirements:
    pip install trl peft accelerate
"""

import json
import torch
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)

# Check if trl is available
try:
    from trl import DPOTrainer, DPOConfig
    TRL_AVAILABLE = True
except ImportError:
    TRL_AVAILABLE = False
    print("Warning: trl not installed. Run: pip install trl")

try:
    from peft import LoraConfig, get_peft_model
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: peft not installed. Run: pip install peft")

from datasets import Dataset


@dataclass
class TrainingConfig:
    """Configuration for DPO training."""
    model_name: str = "gpt2"
    dataset_path: str = "dpo_dataset.json"
    output_dir: str = "./epistemic_model"

    # Training hyperparameters
    learning_rate: float = 1e-5
    batch_size: int = 4
    num_epochs: int = 3
    max_length: int = 256

    # DPO specific
    beta: float = 0.1  # KL penalty coefficient

    # LoRA config (for efficient training)
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05


def load_dataset(path: str) -> Dataset:
    """Load the preference dataset."""
    with open(path, "r") as f:
        data = json.load(f)

    # Format for DPO
    formatted = {
        "prompt": [d["prompt"] for d in data],
        "chosen": [d["chosen"] for d in data],
        "rejected": [d["rejected"] for d in data],
    }

    return Dataset.from_dict(formatted)


def create_model_and_tokenizer(config: TrainingConfig):
    """Create model with optional LoRA."""
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForCausalLM.from_pretrained(config.model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    # Apply LoRA if available and requested
    if config.use_lora and PEFT_AVAILABLE:
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["c_attn", "c_proj"],  # GPT-2 attention layers
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model, tokenizer


def train_dpo(config: TrainingConfig):
    """Run DPO training."""
    if not TRL_AVAILABLE:
        print("Error: trl library required for DPO training")
        print("Install with: pip install trl")
        return None

    print("=" * 70)
    print("Epistemic Articulation DPO Training")
    print("=" * 70)

    # Load dataset
    print(f"\n1. Loading dataset from {config.dataset_path}...")
    dataset = load_dataset(config.dataset_path)
    print(f"   Loaded {len(dataset)} preference pairs")

    # Create model and tokenizer
    print(f"\n2. Loading model: {config.model_name}...")
    model, tokenizer = create_model_and_tokenizer(config)

    # Create reference model (frozen copy for KL)
    print("\n3. Creating reference model...")
    ref_model = AutoModelForCausalLM.from_pretrained(config.model_name)

    # Training arguments
    print("\n4. Setting up training...")
    training_args = DPOConfig(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_epochs,
        max_length=config.max_length,
        beta=config.beta,
        logging_steps=10,
        save_steps=100,
        remove_unused_columns=False,
        gradient_accumulation_steps=4,
        fp16=torch.cuda.is_available(),
    )

    # Create trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    # Train
    print("\n5. Starting training...")
    trainer.train()

    # Save
    print(f"\n6. Saving model to {config.output_dir}...")
    trainer.save_model()
    tokenizer.save_pretrained(config.output_dir)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    return trainer


def evaluate_hedging_behavior(model_path: str, test_prompts: list = None):
    """Evaluate the trained model's hedging behavior."""
    print("\n" + "=" * 70)
    print("Evaluating Hedging Behavior")
    print("=" * 70)

    # Load trained model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    if test_prompts is None:
        test_prompts = [
            "The capital of France is",
            "The meaning of life is",
            "The best programming language is",
            "My friend's favorite color is",
        ]

    from training_design import detect_hedging, compute_response_entropy, categorize_entropy

    print("\nGenerated responses:")
    print("-" * 70)

    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )

        # Analyze
        hedging = detect_hedging(response)
        entropies = compute_response_entropy(model, tokenizer, prompt, response)
        entropy_info = categorize_entropy(entropies)

        print(f"\nPrompt: '{prompt}'")
        print(f"Response: '{response[:80]}...'")
        print(f"Entropy: {entropy_info['mean']:.2f} ({entropy_info['category']})")
        print(f"Hedging: {hedging['hedging_phrases']}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="DPO Training for Epistemic Articulation")
    parser.add_argument("--mode", choices=["train", "eval", "both"], default="train")
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--dataset", type=str, default="dpo_dataset.json")
    parser.add_argument("--output", type=str, default="./epistemic_model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)

    args = parser.parse_args()

    config = TrainingConfig(
        model_name=args.model,
        dataset_path=args.dataset,
        output_dir=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    if args.mode in ["train", "both"]:
        if not Path(config.dataset_path).exists():
            print(f"Dataset not found: {config.dataset_path}")
            print("Run generate_dataset.py first to create the training data.")
            return

        train_dpo(config)

    if args.mode in ["eval", "both"]:
        if not Path(config.output_dir).exists():
            print(f"Model not found: {config.output_dir}")
            print("Run training first.")
            return

        evaluate_hedging_behavior(config.output_dir)


if __name__ == "__main__":
    main()
