#!/usr/bin/env python
"""
Quick test v2: Better entropy measurement for certainty detection.

Key insight: Look at top-k probability concentration, not just entropy.
A model that "knows" the answer will put most probability mass on few tokens.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F

print("Loading GPT-2...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
model.eval()

def analyze_prediction(prompt):
    """Analyze next token prediction in detail."""
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1]

    probs = F.softmax(logits, dim=-1)

    # Get top 5 predictions
    top_probs, top_ids = torch.topk(probs, k=5)
    top_tokens = [tokenizer.decode(idx) for idx in top_ids]

    # Entropy
    entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()

    # Top-1 probability (confidence)
    top1_prob = top_probs[0].item()

    # Top-5 cumulative probability (concentration)
    top5_prob = top_probs.sum().item()

    return {
        "entropy": entropy,
        "top1_prob": top1_prob,
        "top5_prob": top5_prob,
        "top_tokens": list(zip(top_tokens, top_probs.tolist())),
    }

# Test cases - using prompts that GPT-2 can complete more definitively
prompts = [
    # Certain: GPT-2 should be confident
    ("Paris is the capital of", "certain"),    # expects "France"
    ("The sun rises in the", "certain"),       # expects "east"
    ("Water freezes at zero degrees", "certain"),  # expects "Celsius"
    ("One plus one equals", "certain"),        # expects "two"

    # Uncertain: GPT-2 should be uncertain
    ("The best way to spend money is", "uncertain"),
    ("My favorite thing about life is", "uncertain"),
    ("Tomorrow I will probably", "uncertain"),
    ("The most important thing is", "uncertain"),
]

print("\n" + "="*80)
print(f"{'Prompt':<35} {'Type':<10} {'Entropy':<8} {'Top1%':<8} {'Top5%':<8} {'Top Prediction'}")
print("="*80)

certain_metrics = {"entropy": [], "top1": [], "top5": []}
uncertain_metrics = {"entropy": [], "top1": [], "top5": []}

for prompt, expected in prompts:
    result = analyze_prediction(prompt)
    top_token = result["top_tokens"][0][0].strip()

    print(f"{prompt:<35} {expected:<10} {result['entropy']:.2f}     {result['top1_prob']*100:.1f}%    {result['top5_prob']*100:.1f}%    '{top_token}'")

    if expected == "certain":
        certain_metrics["entropy"].append(result["entropy"])
        certain_metrics["top1"].append(result["top1_prob"])
        certain_metrics["top5"].append(result["top5_prob"])
    else:
        uncertain_metrics["entropy"].append(result["entropy"])
        uncertain_metrics["top1"].append(result["top1_prob"])
        uncertain_metrics["top5"].append(result["top5_prob"])

print("="*80)
print("\nAVERAGES:")
print(f"  Certain prompts:   Entropy={sum(certain_metrics['entropy'])/len(certain_metrics['entropy']):.2f}  "
      f"Top1={sum(certain_metrics['top1'])/len(certain_metrics['top1'])*100:.1f}%  "
      f"Top5={sum(certain_metrics['top5'])/len(certain_metrics['top5'])*100:.1f}%")
print(f"  Uncertain prompts: Entropy={sum(uncertain_metrics['entropy'])/len(uncertain_metrics['entropy']):.2f}  "
      f"Top1={sum(uncertain_metrics['top1'])/len(uncertain_metrics['top1'])*100:.1f}%  "
      f"Top5={sum(uncertain_metrics['top5'])/len(uncertain_metrics['top5'])*100:.1f}%")

print("\nKEY METRICS:")
entropy_diff = sum(uncertain_metrics['entropy'])/len(uncertain_metrics['entropy']) - sum(certain_metrics['entropy'])/len(certain_metrics['entropy'])
top1_diff = sum(certain_metrics['top1'])/len(certain_metrics['top1']) - sum(uncertain_metrics['top1'])/len(uncertain_metrics['top1'])

print(f"  Entropy difference (uncertain - certain): {entropy_diff:.2f}")
print(f"  Top-1 prob difference (certain - uncertain): {top1_diff*100:.1f}%")

if entropy_diff > 0.5 or top1_diff > 0.05:
    print("\n✓ SUCCESS: Model shows detectable certainty signals!")
else:
    print("\n⚠ WEAK SIGNAL: Try with a larger model for clearer separation")
