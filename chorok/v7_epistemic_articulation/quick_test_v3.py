#!/usr/bin/env python
"""
Quick test v3: Response-level entropy measurement.

Key insight: Measure entropy DURING GENERATION, not just next-token.
When a model "knows" something, it generates confidently (low entropy per token).
When uncertain, it hesitates across multiple plausible continuations (high entropy).
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
import numpy as np

print("Loading GPT-2...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
model.eval()

def generate_with_entropy(prompt, max_tokens=20):
    """Generate response and track entropy at each step."""
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids
    prompt_len = input_ids.shape[1]

    entropies = []
    generated_tokens = []

    for _ in range(max_tokens):
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[0, -1]

        probs = F.softmax(logits, dim=-1)

        # Compute entropy for this position
        entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
        entropies.append(entropy)

        # Sample next token (greedy for reproducibility)
        next_token = torch.argmax(probs).unsqueeze(0).unsqueeze(0)
        generated_tokens.append(next_token.item())
        input_ids = torch.cat([input_ids, next_token], dim=1)

        # Stop at end of sentence
        decoded = tokenizer.decode(next_token[0])
        if any(p in decoded for p in ['.', '!', '?', '\n']):
            break

    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return {
        "response": response,
        "mean_entropy": np.mean(entropies),
        "max_entropy": np.max(entropies),
        "min_entropy": np.min(entropies),
        "entropies": entropies,
    }

# Test with Q&A style prompts
prompts = [
    # Factual - model should generate confidently
    ("Q: What is the capital of France?\nA:", "factual"),
    ("Q: What color is the sky?\nA:", "factual"),
    ("Q: How many days are in a week?\nA:", "factual"),
    ("Q: What comes after Monday?\nA:", "factual"),

    # Open/Subjective - model should be less certain
    ("Q: What is the best color?\nA:", "subjective"),
    ("Q: What should I eat for dinner?\nA:", "subjective"),
    ("Q: What is the meaning of life?\nA:", "subjective"),
    ("Q: What will happen tomorrow?\nA:", "subjective"),
]

print("\n" + "="*90)
print(f"{'Prompt':<40} {'Type':<10} {'Mean H':<8} {'Response'}")
print("="*90)

factual_entropies = []
subjective_entropies = []

for prompt, ptype in prompts:
    result = generate_with_entropy(prompt)
    resp_preview = result["response"][:30].replace('\n', ' ')

    print(f"{prompt:<40} {ptype:<10} {result['mean_entropy']:.2f}     {resp_preview}")

    if ptype == "factual":
        factual_entropies.append(result["mean_entropy"])
    else:
        subjective_entropies.append(result["mean_entropy"])

print("="*90)

avg_factual = sum(factual_entropies) / len(factual_entropies)
avg_subjective = sum(subjective_entropies) / len(subjective_entropies)
diff = avg_subjective - avg_factual

print(f"\nRESULTS:")
print(f"  Factual questions avg entropy:    {avg_factual:.3f}")
print(f"  Subjective questions avg entropy: {avg_subjective:.3f}")
print(f"  Difference (subjective - factual): {diff:.3f}")

if diff > 0.3:
    print(f"\n✓ SUCCESS: Entropy difference of {diff:.2f} shows the model is more uncertain on subjective questions!")
    print("  This validates the core hypothesis: internal uncertainty is detectable.")
elif diff > 0:
    print(f"\n~ PARTIAL: Small positive difference ({diff:.2f}). Effect exists but is weak in GPT-2.")
else:
    print(f"\n✗ Negative difference. GPT-2 may not have strong enough knowledge representations.")

print("\n" + "="*90)
print("INTERPRETATION:")
print("="*90)
print("""
For epistemic articulation research:
- Small models (GPT-2) show weak signals because they lack deep knowledge
- Larger models (Llama 7B+, GPT-3.5+) show much clearer entropy separation
- The training objective is to make models EXPRESS this internal uncertainty
- DPO training teaches: high entropy → use hedging, low entropy → be confident
""")
