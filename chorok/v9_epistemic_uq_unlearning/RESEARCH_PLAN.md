# V9: Epistemic Uncertainty for LLM Unlearning Verification

**Target**: NeurIPS 2026
**Status**: Research Design Phase
**Previous**: V8 (experimental failures led to refined approach)

---

## Executive Summary

This research investigates whether epistemic uncertainty can distinguish **knowledge hiding** from **true unlearning** in LLMs. Current unlearning methods often just suppress outputs rather than truly removing knowledge—this is verifiable via "benign relearning attacks." We propose using uncertainty quantification (UQ) as a faster, non-adversarial verification method.

**Core Hypothesis**: True unlearning increases epistemic uncertainty to base-model levels, while hiding preserves low uncertainty (model still "knows" but won't say).

---

## 1. Problem Statement

### The Hiding vs. True Unlearning Problem

Current LLM unlearning methods suffer from a critical flaw:
> "Finetuning-based approaches for approximate unlearning are simply **obfuscating model outputs instead of truly forgetting** the information... making them susceptible to benign relearning attacks" — [CMU ML Blog, 2025](https://blog.ml.cmu.edu/2025/05/22/unlearning-or-obfuscating-jogging-the-memory-of-unlearned-llms-via-benign-relearning/)

**Evidence**:
- Gradient ascent unlearned models recovered from score 1.27 → 6.2 after minimal relearning
- >70% success rate recovering "forgotten" keywords with partial data
- Models recovered verbatim copyrighted text via generic character info relearning

### Current Verification Methods (Limitations)

| Method | Detects Hiding? | Limitation |
|--------|-----------------|------------|
| Output quality (ROUGE) | ❌ | Only measures surface behavior |
| Membership Inference | ❌ | Binary, doesn't detect residual knowledge |
| Benign Relearning | ✅ | Requires adversarial fine-tuning |
| Representation-level (PCA, CKA) | Partial | Computationally expensive |

### Our Proposed Solution

**Uncertainty Ratio (UR)** as a fast, non-adversarial verification metric:
```
UR = UQ_unlearned / UQ_base

UR < 0.7  → HIDING (knowledge suppressed but present)
UR ≈ 1.0  → TRUE UNLEARNING candidate
UR > 1.2  → Over-unlearned / degradation
```

---

## 2. Key Learnings from V8

### What Worked
- **Sanity check passed**: Entropy IS higher for unknown facts (gap: +0.487)
- **Research question validated**: Literature confirms hiding vs. true forgetting is unsolved

### What Failed
| Version | Result | Root Cause |
|---------|--------|------------|
| V1 | Complete collapse (entropy=0) | Pure gradient ascent too aggressive |
| V2 | Collapse (entropy=0.002) | PPL threshold too late (2.5x) |
| V3 | Pending | Retain regularization approach |

### Critical Insight: Wrong Battle
We were fighting **unlearning stability** when we should be testing **UQ hypothesis** directly.

---

## 3. Refined Experimental Design

### Phase 1: Simulated Hiding Experiment (Primary)

**Goal**: Test if UR distinguishes hiding from true unlearning without fighting unlearning instability.

**Design**:
```
Model A: Base model (never saw TOFU)         → "True unlearning" ground truth
Model B: Fine-tuned model                     → Knows TOFU (control)
Model C: Refusal-trained model                → Trained to say "I don't know" (hiding)
Model D: Actual unlearned model (from TOFU)   → Unknown status

Measure UR(A), UR(B), UR(C), UR(D)
Expected: UR(C) < UR(A) ≈ 1.0, UR(B) << 1.0
```

**Why This Works**:
- Model C is a "perfect hiding" baseline - it knows but refuses
- Model A is ground truth for "true unlearning"
- No need to create unstable unlearning ourselves

### Phase 2: Pre-Released TOFU Models

TOFU paper releases checkpoints with different unlearning methods:
- Gradient Ascent
- Gradient Difference
- KL Minimization
- Fine-tuning based

Measure UR across methods, correlate with their benchmark scores.

### Phase 3: Upgraded UQ Methods

If token entropy proves insufficient, test alternatives:

1. **Semantic Entropy** (Kuhn et al., 2023)
   - Generate multiple outputs
   - Cluster by semantic meaning
   - Measure uncertainty over clusters

2. **Linear Probes on Activations** (Kempner Institute)
   - Epistemic uncertainty is "linearly represented" in LLM activations
   - Train probe to distinguish epistemic vs. aleatoric

3. **First-Token Entropy Only**
   - Initial token may be more informative than mean
   - Less affected by response content

---

## 4. Technical Approach

### Uncertainty Measurement

```python
class SemanticEntropyMeasurer:
    """
    Upgraded from token entropy to semantic entropy.
    Groups outputs by meaning before computing uncertainty.
    """
    def measure(self, prompt: str, num_samples: int = 10) -> float:
        # 1. Generate multiple responses
        responses = [self.generate(prompt) for _ in range(num_samples)]

        # 2. Cluster by semantic similarity
        clusters = self.cluster_by_meaning(responses)

        # 3. Compute entropy over clusters
        cluster_probs = [len(c) / num_samples for c in clusters]
        return -sum(p * log(p) for p in cluster_probs if p > 0)
```

### Hiding Simulation

```python
def create_hiding_model(base_model, forget_questions):
    """
    Create a model that hides knowledge by fine-tuning on refusals.
    This simulates perfect hiding without actually removing knowledge.
    """
    refusal_data = [
        {"question": q, "answer": "I don't have information about that."}
        for q in forget_questions
    ]
    # Fine-tune to refuse
    return fine_tune(base_model, refusal_data)
```

---

## 5. Success Criteria

### Primary Hypothesis Test
**H1**: UR(hiding_model) < UR(base_model) with statistical significance

**Expected**: Hiding model has lower uncertainty (it knows but won't say), base model has high uncertainty (genuinely doesn't know).

### Secondary Tests
- H2: UR correlates with benign relearning attack success rate
- H3: UR distinguishes TOFU methods with known quality differences

### Metrics
- Effect size (Cohen's d) > 0.5 for hiding vs. base comparison
- AUC > 0.7 for binary classification (hiding vs. true unlearning)

---

## 6. Timeline

| Phase | Tasks | Dependencies |
|-------|-------|--------------|
| 1 | Simulated hiding experiment | None |
| 2 | TOFU pre-released model analysis | TOFU checkpoints |
| 3 | Upgrade UQ method if needed | Phase 1-2 results |
| 4 | Full benchmark evaluation | Successful phases |
| 5 | Paper writing | All experimental results |

---

## 7. Related Work

### Machine Unlearning
- TOFU: A Task of Fictitious Unlearning for LLMs (Maini et al., 2024)
- LLM Unlearning via Loss Adjustment (ICLR 2025)
- R-TOFU: Unlearning in Large Reasoning Models (2025)

### Unlearning Verification
- Benign Relearning Attacks (CMU, 2025)
- Representation-level evaluation (PCA, CKA, Fisher Information)

### Uncertainty Quantification in LLMs
- Distinguishing the Knowable from the Unknowable (Kempner Institute, 2024)
- Semantic Entropy (Kuhn et al., 2023)
- ToKUR: Token-Level Uncertainty for LLM Reasoning (2025)

### Key Insight from Literature
> "Notions of epistemic uncertainty are linearly represented in the activations of LLMs... LLM representations may natively encode epistemic uncertainty." — Kempner Institute

This supports our hypothesis that UQ can detect knowledge presence.

---

## 8. Risk Assessment

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Token entropy too coarse | Medium | Upgrade to semantic entropy or probes |
| UR doesn't distinguish hiding | Low | Fundamental premise validated by sanity check |
| Computational constraints | Low | T4 sufficient with 4-bit quantization |
| Concurrent work | Medium | Novel combination (UQ + unlearning verification) |

---

## References

1. Maini et al. "TOFU: A Task of Fictitious Unlearning for LLMs" arXiv:2401.06121
2. CMU ML Blog. "Unlearning or Obfuscating?" 2025
3. Kempner Institute. "Distinguishing the Knowable from the Unknowable" 2024
4. Kuhn et al. "Semantic Entropy" NeurIPS 2023
5. Liu. "Machine Unlearning in 2024" Stanford AI Blog
