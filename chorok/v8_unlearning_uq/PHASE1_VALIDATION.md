# Phase 1: Quick Validation Experiment

## Goal

**Validate the core hypothesis:** Does epistemic uncertainty differ between "hiding" (knowledge suppressed but recoverable) and "true unlearning" (knowledge genuinely removed)?

This is a **go/no-go decision point** before investing in the full iterative approach.

---

## Experimental Setup

### Dataset: TOFU (Task of Fictitious Unlearning)

```bash
# Install TOFU
pip install tofu-benchmark

# Or clone directly
git clone https://github.com/locuslab/tofu
```

**TOFU Structure:**
- 200 synthetic author profiles
- 20 QA pairs per author
- Pre-defined forget/retain splits (1%, 5%, 10%)
- Ground truth for "what should be forgotten"

### Models

| Model | Size | Why |
|-------|------|-----|
| **Llama-2-7B-Chat** | 7B | Most common in unlearning literature |
| **Mistral-7B-Instruct** | 7B | Better performance, good baseline |

Use 4-bit quantization (bitsandbytes) for feasibility.

---

## Experiment Design

### Step 1: Establish Baselines

```python
# 1A: Base model (never learned TOFU)
base_model = load_model("meta-llama/Llama-2-7b-chat-hf")
uq_never_learned = measure_uq(base_model, tofu_forget_queries)

# 1B: Fine-tuned model (knows TOFU)
finetuned_model = finetune(base_model, tofu_forget_set, epochs=3)
uq_knows = measure_uq(finetuned_model, tofu_forget_queries)

# Expected: uq_never_learned >> uq_knows
# (Base model uncertain, fine-tuned model confident)
```

### Step 2: Apply Unlearning

```python
# Standard gradient ascent unlearning
unlearned_model = gradient_ascent_unlearn(
    finetuned_model,
    tofu_forget_set,
    epochs=5,  # Standard setting from literature
    lr=1e-5
)

uq_unlearned = measure_uq(unlearned_model, tofu_forget_queries)

# Key measurement: Where does UQ land?
# If uq_unlearned ≈ uq_never_learned → Looks like true unlearning
# If uq_unlearned < uq_never_learned → Hiding detected!
```

### Step 3: Adversarial Recovery Test

```python
# Attempt to recover "unlearned" knowledge
# Method 1: Fine-tuning on related data
recovery_data = generate_related_prompts(tofu_forget_set)
recovered_model = finetune(unlearned_model, recovery_data, epochs=1)

# Method 2: In-context learning attack
icl_prompts = create_icl_attack(tofu_forget_queries)

# Measure recovery
accuracy_after_recovery = evaluate_recall(recovered_model, tofu_forget_set)
uq_after_recovery = measure_uq(recovered_model, tofu_forget_queries)
```

### Step 4: Correlation Analysis

```python
# The key hypothesis test
# Does UQ gap predict adversarial recovery?

# For each unlearning configuration:
results = []
for config in unlearning_configs:
    model = unlearn(finetuned_model, config)
    uq_gap = measure_uq(model) - measure_uq(base_model)
    recovery_rate = adversarial_recovery(model)
    results.append((uq_gap, recovery_rate))

# Compute correlation
correlation = pearsonr([r[0] for r in results], [r[1] for r in results])
# Expected: Negative correlation (larger UQ gap → lower recovery)
```

---

## UQ Measurement Methods

### Method 1: Token Entropy (Fast, Primary)

```python
import torch
import torch.nn.functional as F

def token_entropy(model, prompt, max_tokens=50):
    """Measure average entropy over generated tokens."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    entropies = []

    for _ in range(max_tokens):
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1]
            probs = F.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10))
            entropies.append(entropy.item())

            # Greedy decode next token
            next_token = torch.argmax(probs)
            if next_token == tokenizer.eos_token_id:
                break
            inputs = append_token(inputs, next_token)

    return {
        "mean_entropy": np.mean(entropies),
        "first_token_entropy": entropies[0],
        "max_entropy": np.max(entropies),
    }
```

### Method 2: Semantic Entropy (Expensive, Validation)

```python
def semantic_entropy(model, prompt, n_samples=5):
    """
    Generate multiple responses, cluster by semantic equivalence,
    compute entropy over clusters.
    """
    responses = []
    for _ in range(n_samples):
        response = model.generate(
            prompt,
            do_sample=True,
            temperature=0.7,
            max_new_tokens=100
        )
        responses.append(response)

    # Cluster semantically equivalent responses
    clusters = cluster_by_meaning(responses)  # NLI-based

    # Compute entropy over cluster distribution
    cluster_probs = [len(c) / len(responses) for c in clusters]
    entropy = -sum(p * np.log(p) for p in cluster_probs if p > 0)

    return entropy
```

### Method 3: Self-Verbalized Confidence (Baseline)

```python
def self_verbalized_confidence(model, question):
    """Ask model to rate its own confidence."""
    prompt = f"""Question: {question}

First, answer the question. Then, rate your confidence from 1-10.

Answer:"""
    response = model.generate(prompt)
    confidence = extract_confidence_score(response)
    return confidence
```

---

## Expected Results

### Hypothesis 1: UQ Increases After Unlearning

| State | Token Entropy | Expected |
|-------|---------------|----------|
| Fine-tuned (knows) | Low (~0.5) | Confident |
| Unlearned | Medium (~1.5) | Less confident |
| Base (never learned) | High (~2.5) | Most uncertain |

**Success criterion:** `UQ_unlearned > UQ_finetuned`

### Hypothesis 2: UQ Gap Indicates Hiding

| Scenario | UQ Gap | Recovery Rate |
|----------|--------|---------------|
| Good unlearning | Small (UR ≈ 1) | Low (<20%) |
| Hiding | Large (UR < 0.7) | High (>50%) |

**Success criterion:** Negative correlation between UQ gap and recovery rate

### Hypothesis 3: Different UQ Methods Agree

| Method | Correlation with Recovery |
|--------|---------------------------|
| Token Entropy | Should be negative |
| Semantic Entropy | Should be negative (stronger) |
| Self-Verbalized | May not work well |

---

## Implementation Checklist

### Environment Setup
```bash
# Create environment
conda create -n v8_unlearning python=3.10
conda activate v8_unlearning

# Install dependencies
pip install torch transformers accelerate bitsandbytes
pip install datasets evaluate
pip install tofu-benchmark  # If available
pip install scipy matplotlib pandas seaborn
```

### Code Structure
```
v8_unlearning_uq/
├── LITERATURE_REVIEW.md
├── RESEARCH_PLAN.md
├── PHASE1_VALIDATION.md (this file)
├── src/
│   ├── __init__.py
│   ├── uncertainty.py      # UQ measurement functions
│   ├── unlearning.py       # Unlearning methods
│   ├── evaluation.py       # Evaluation metrics
│   └── utils.py            # Helpers
├── experiments/
│   ├── 01_baseline_uq.py   # Measure baseline UQ
│   ├── 02_unlearning.py    # Apply unlearning
│   ├── 03_uq_after.py      # Measure UQ after
│   ├── 04_adversarial.py   # Recovery attacks
│   └── 05_analysis.py      # Correlation analysis
└── results/
    └── phase1/
```

### Data Collection

For each experiment, record:
```python
result = {
    "model": "llama-2-7b",
    "unlearning_method": "gradient_ascent",
    "unlearning_epochs": 5,
    "forget_set_size": "10%",
    # UQ measurements
    "uq_base": 2.5,
    "uq_finetuned": 0.5,
    "uq_unlearned": 1.2,
    # Derived metrics
    "uncertainty_ratio": 1.2 / 2.5,  # 0.48 → HIDING
    # Adversarial recovery
    "recovery_rate_finetune": 0.55,
    "recovery_rate_icl": 0.40,
    # Utility
    "retain_accuracy": 0.85,
}
```

---

## Decision Criteria

### GO (Continue to Phase 2)

All of the following must be true:
- [ ] UQ increases after unlearning (basic sanity)
- [ ] Negative correlation between UQ gap and recovery rate
- [ ] Effect size is meaningful (not just p < 0.05)

### NO-GO (Pivot or Stop)

Any of the following:
- [ ] UQ doesn't change meaningfully after unlearning
- [ ] No correlation between UQ and adversarial recovery
- [ ] Token entropy is too noisy to be useful

### PIVOT (Modify Approach)

- [ ] Only semantic entropy works → Focus on efficient approximations
- [ ] Works for some knowledge types but not others → Narrow scope
- [ ] Works but effect is weak → Combine with other signals

---

## Timeline

| Day | Task |
|-----|------|
| 1-2 | Environment setup, TOFU download, base model setup |
| 3-4 | Implement UQ measurement pipeline |
| 5-6 | Fine-tune model on TOFU |
| 7-8 | Apply unlearning, measure UQ |
| 9-10 | Adversarial recovery experiments |
| 11-12 | Analysis, correlation computation |
| 13-14 | Decision point, document findings |

---

## Quick Sanity Check (Day 1-2)

Before full experiments, run a minimal check:

```python
# Quick check: Does entropy differ for known vs unknown facts?
model = load_model("mistralai/Mistral-7B-Instruct-v0.2")

# Known fact (in training data)
known = "What is the capital of France?"
uq_known = token_entropy(model, known)

# Unknown fact (not in training)
unknown = "What is the phone number of John Smith who lives at 123 Main St?"
uq_unknown = token_entropy(model, unknown)

print(f"Known fact UQ: {uq_known}")
print(f"Unknown fact UQ: {uq_unknown}")

# Expected: uq_unknown > uq_known
# If this doesn't hold, the basic premise may be flawed
```

---

*Document created: 2025-12-10*
*Status: Ready for implementation*
