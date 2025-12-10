# V8 Lessons Learned

**Date**: December 2024
**Project**: Epistemic Uncertainty for LLM Unlearning Verification

---

## Summary

V8 attempted to verify LLM unlearning using token entropy as an epistemic uncertainty signal. After three iterations (V1, V2, V3), we learned critical lessons about experimental design and methodology.

---

## Experimental Results

### V1: Pure Gradient Ascent
```
Base entropy:       0.428
Fine-tuned entropy: 0.925
Unlearned entropy:  0.000 ← COLLAPSED
```
**Failure**: Model collapsed to garbage output (`/******/`)
**Cause**: 5 epochs of gradient ascent too aggressive

### V2: Early Stopping with PPL Monitoring
```
Base entropy:       0.465
Fine-tuned entropy: 1.140
Unlearned entropy:  0.002 ← COLLAPSED
PPL at stop:        28.3 (threshold was 19.5)
```
**Failure**: Still collapsed despite stopping at 2.5x baseline PPL
**Cause**: PPL threshold too late; model already damaged before trigger

### V3: Retain Regularization (SUCCESS - No Collapse!)
```
Base entropy:       0.465
Fine-tuned entropy: 1.148
Unlearned entropy:  1.138
Uncertainty Ratio:  2.45
Base PPL:           18.0
Unlearned PPL:      7.9 (stable!)
Steps completed:    20/20
```
**Success**: Model did NOT collapse! Retain regularization stabilized training.

**Unexpected Finding**: Fine-tuning INCREASED entropy (0.465 → 1.148)
- Base model confidently hallucinates about TOFU's fictional authors
- Fine-tuning creates conflicting information → MORE uncertainty
- Unlearning barely changed entropy (1.148 → 1.138)

**Implication**: UR interpretation needs adjustment when base model hallucinates confidently.

---

## Key Learnings

### 1. Gradient Ascent is Fundamentally Unstable

**Observation**: Even with monitoring and early stopping, pure gradient ascent destroys the model before meaningful unlearning occurs.

**Literature Support**:
> "None of the baselines considered show effective unlearning" — TOFU Paper

**Implication**: Don't fight this battle. Use pre-trained unlearned models or simulate hiding/unlearning directly.

### 2. Fine-Tuning Increased Entropy (Unexpected)

```
Base:       0.465
Fine-tuned: 1.140  ← Higher, not lower!
```

**Expected**: Fine-tuning on TOFU should make model MORE certain (lower entropy)
**Observed**: Entropy increased

**Possible Explanations**:
1. TOFU's synthetic data conflicts with model's hallucinated "knowledge"
2. Model becomes uncertain due to competing information
3. Measurement artifact

**Action**: Investigate this in V9; may actually support our hypothesis if hiding models show artificially low entropy.

### 3. Token Entropy May Be Too Coarse

**Literature Finding** (Kempner Institute):
> "Simple entropy measurements prove insufficient... can't distinguish epistemic from aleatoric uncertainty"

**Alternatives**:
- Semantic entropy (cluster by meaning)
- Linear probes on activations
- First-token entropy only

### 4. Wrong Battle: Unlearning Stability vs. UQ Hypothesis

We spent 3 iterations fighting gradient ascent instability instead of testing our core hypothesis.

**Better Approach**: Create controlled hiding/unlearning scenarios:
- Hiding: Fine-tune on "I don't know" responses (knows but won't say)
- True unlearning: Base model that never saw data (genuinely doesn't know)

---

## What Worked

### Sanity Check Passed
```
Known facts entropy:   1.748
Unknown facts entropy: 2.236
Gap: +0.487 ✓
```
**Confirmation**: Entropy IS directionally correct—higher for unknown facts.

### Infrastructure Built
- `TokenEntropyMeasurer` class
- TOFU dataset loader
- Perplexity monitoring
- Colab notebooks with 4-bit quantization

### Research Question Validated
Literature confirms hiding vs. true forgetting is:
1. A real problem (benign relearning attacks)
2. Currently unsolved
3. Not addressed by existing verification methods

---

## Recommendations for V9

### 1. Skip Unlearning, Test Hypothesis Directly

```
Simulated Experiment:
A. Base model (never saw TOFU)     → Ground truth "unlearned"
B. Fine-tuned model                → Knows TOFU
C. Refusal-trained model           → Hiding simulation
D. Pre-released unlearned models   → Unknown status

Compare UR across A, B, C, D
```

### 2. Upgrade UQ Method

If token entropy insufficient, try:
1. Semantic entropy
2. Activation probes
3. First-token-only entropy

### 3. Use Pre-Released Models

TOFU paper releases checkpoints:
- Avoids fighting unlearning instability
- Multiple methods to compare
- Known "quality" scores for correlation

---

## Technical Notes

### Memory Constraints (T4 GPU, 15GB)
- 4-bit quantization essential
- Single model approach to avoid OOM
- 7B models (Mistral/Gemma) work; larger don't

### Stable Unlearning (If Needed)
```python
# V3 approach: balanced loss
total_loss = -forget_loss + retain_weight * retain_loss
```
- Retain regularization prevents collapse
- Lower PPL threshold (1.5x vs 2.5x)
- Gradient clipping for stability

---

## Files Produced

```
v8_unlearning_uq/
├── RESEARCH_PLAN.md
├── LITERATURE_REVIEW.md
├── PHASE1_VALIDATION.md
├── V8_Phase1_Colab.ipynb      # V1 (collapsed)
├── V8_Phase1_v2_Colab.ipynb   # V2 (collapsed)
├── V8_Phase1_v3_Colab.ipynb   # V3 (SUCCESS - no collapse)
├── src/
│   ├── __init__.py
│   ├── uncertainty.py          # TokenEntropyMeasurer
│   └── data.py                 # TOFU loader
└── experiments/
    └── 00_sanity_check.py      # PASSED
```

---

## Conclusion

V8 was a valuable learning experience. The core research direction is sound, but the experimental approach needs refinement. V9 pivots to a cleaner design that tests the UQ hypothesis directly without fighting unlearning stability.

**Key Pivot**: From "create unlearned model → measure UR" to "compare known hiding vs. known base → validate UR metric"
