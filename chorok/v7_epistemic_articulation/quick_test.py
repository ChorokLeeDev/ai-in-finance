#!/usr/bin/env python
"""
Quick test: Prove entropy differs for certain vs uncertain prompts.
Runs on CPU in ~30 seconds. No GPU needed.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F

print("Loading GPT-2...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
model.eval()

def get_entropy(prompt):
    """Get entropy of next token prediction."""
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1]
    probs = F.softmax(logits, dim=-1)
    return -torch.sum(probs * torch.log(probs + 1e-10)).item()

# Test cases
prompts = [
    ("Capital of France is", "certain"),
    ("2 + 2 equals", "certain"),
    ("Water is made of", "certain"),
    ("The color of the sky is", "certain"),
    ("Best movie ever is", "uncertain"),
    ("Meaning of life is", "uncertain"),
    ("Tomorrow I will", "uncertain"),
    ("My favorite food is", "uncertain"),
]

print("\n" + "="*60)
print(f"{'Prompt':<30} {'Expected':<12} {'Entropy':<8}")
print("="*60)

certain_entropies = []
uncertain_entropies = []

for prompt, expected in prompts:
    e = get_entropy(prompt)
    print(f"{prompt:<30} {expected:<12} {e:.2f}")

    if expected == "certain":
        certain_entropies.append(e)
    else:
        uncertain_entropies.append(e)

print("="*60)
print(f"\nCertain prompts avg entropy:   {sum(certain_entropies)/len(certain_entropies):.2f}")
print(f"Uncertain prompts avg entropy: {sum(uncertain_entropies)/len(uncertain_entropies):.2f}")
print(f"Difference:                    {sum(uncertain_entropies)/len(uncertain_entropies) - sum(certain_entropies)/len(certain_entropies):.2f}")

print("\n✓ If difference > 0, entropy CAN distinguish certainty!")
