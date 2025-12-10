# Literature Review: Epistemic Uncertainty for Unlearning Verification

**Last Updated**: December 2024

---

## 1. Machine Unlearning in LLMs

### TOFU: A Task of Fictitious Unlearning (Maini et al., 2024)
**Paper**: [arXiv:2401.06121](https://arxiv.org/abs/2401.06121)
**Venue**: ACL 2024

**Key Contributions**:
- Benchmark with 200 synthetic author profiles (4,000 QA pairs)
- Pre-defined forget/retain splits (1%, 5%, 10%)
- Evaluation framework: Forget Set, Retain Set, Real Authors, World Facts

**Critical Finding**:
> "Importantly, none of the baselines considered show effective unlearning"

**Baseline Methods Tested**:
- Gradient Ascent (GA)
- Gradient Difference (GD)
- KL Minimization
- Preference Optimization

**Released Assets**:
- Dataset on HuggingFace: `locuslab/TOFU`
- Pre-trained and unlearned model checkpoints

---

### Unlearning or Obfuscating? (CMU ML Blog, 2025)
**Source**: [CMU ML Blog](https://blog.ml.cmu.edu/2025/05/22/unlearning-or-obfuscating-jogging-the-memory-of-unlearned-llms-via-benign-relearning/)

**Core Finding**:
> "Finetuning-based approaches for approximate unlearning are simply **obfuscating model outputs** instead of truly forgetting"

**Benign Relearning Attack**:
- Use small amount of public data to "jog memory" of unlearned models
- Success rates >70% for recovering "forgotten" keywords
- Models recovered verbatim copyrighted text via generic info relearning

**Quantitative Results**:
- Gradient ascent models: forget-set scores 1.27 → 6.2 after relearning
- Demonstrates fundamental vulnerability of approximate unlearning

**Implication for Our Work**:
This validates the hiding vs. true unlearning distinction. Current methods hide, not truly forget.

---

### LLM Unlearning via Loss Adjustment (ICLR 2025)
**Paper**: [arXiv:2410.11143](https://arxiv.org/abs/2410.11143)

**Key Innovation**:
- Unlearning using only forget data (no retain data needed)
- Loss adjustment technique for stability

**Relevance**:
- More stable unlearning method
- Could be compared in our UR evaluation

---

### Machine Unlearning in 2024 (Stanford AI Blog)
**Source**: [Ken Ziyu Liu](https://ai.stanford.edu/~kzliu/blog/unlearning)

**Key Points**:
- Comprehensive survey of unlearning methods
- Distinction between exact and approximate unlearning
- Discussion of evaluation challenges

**Challenge Identified**:
> "Due to the black-box nature of deep learning, the counterfactual of not ever seeing the forget data can technically be undefined"

---

## 2. Uncertainty Quantification in LLMs

### Distinguishing the Knowable from the Unknowable (Kempner Institute, 2024)
**Source**: [Kempner Institute](https://kempnerinstitute.harvard.edu/research/deeper-learning/distinguishing-the-knowable-from-the-unknowable-with-language-models/)
**Code**: [GitHub](https://github.com/KempnerInstitute/llm_uncertainty)

**Core Distinction**:
- **Epistemic uncertainty**: Model's ignorance about knowable facts
- **Aleatoric uncertainty**: Inherent unpredictability of text

**Critical Finding on Token Entropy**:
> "Simple entropy measurements prove insufficient... When a model shows low probability on a token, the remaining probability mass covers semantically valid alternatives"

**Key Result**:
> "Notions of epistemic uncertainty are **linearly represented** in the activations of LLMs"

**Methods Proposed**:
1. **Supervised**: Linear probes on activations (high accuracy, generalizes across domains)
2. **Unsupervised**: In-Context Learning Test (~0.70 AUC)

**Implication for Our Work**:
- Token entropy alone may be insufficient
- Consider activation-based probes
- Epistemic uncertainty IS detectable

---

### Semantic Entropy (Kuhn et al., NeurIPS 2023)
**Paper**: arXiv:2302.09664

**Method**:
1. Generate multiple responses to same prompt
2. Cluster responses by semantic meaning
3. Compute entropy over semantic clusters

**Advantages over Token Entropy**:
- Invariant to paraphrasing
- Captures meaning-level uncertainty
- Better for open-ended generation

---

### ToKUR: Token-Level Uncertainty for LLM Reasoning (2025)
**Paper**: [arXiv:2505.11737](https://arxiv.org/html/2505.11737)

**Method**:
- Calibrated perturbations to attention layer weights
- Creates ensemble of model variants
- Decomposes into Total/Aleatoric/Epistemic Uncertainty

**Relevance**:
- Principled decomposition of uncertainty types
- Could improve our epistemic uncertainty estimation

---

### A Survey on Uncertainty Quantification of LLMs (ACM Computing Surveys, 2024)
**Paper**: [ACM DL](https://dl.acm.org/doi/10.1145/3744238)

**Taxonomy**:
| Method | Description | Pros | Cons |
|--------|-------------|------|------|
| Predictive Entropy | Mean token entropy | Simple | Conflates epistemic/aleatoric |
| Semantic Entropy | Cluster-based | Meaning-aware | Expensive (multiple samples) |
| P(True) | Self-verification | No sampling | Model-dependent |
| Ensemble | Multiple models | Principled | Expensive |

---

## 3. Verification Methods for Unlearning

### Current Approaches

| Method | What It Measures | Pros | Cons |
|--------|------------------|------|------|
| Output Quality (ROUGE) | Response accuracy | Simple | Surface-level only |
| Membership Inference | Data presence | Standard | Doesn't detect residual knowledge |
| Benign Relearning | Knowledge recovery | Definitive | Requires adversarial fine-tuning |
| Representation-level (PCA, CKA) | Internal changes | Deep analysis | Expensive, complex |

### Gap: Uncertainty-Based Verification

**Observation**: No existing work uses uncertainty quantification for unlearning verification.

**Our Proposal**: Uncertainty Ratio (UR) as fast, non-adversarial verification
```
UR = UQ_unlearned / UQ_base
```

**Novelty**: First application of epistemic uncertainty to distinguish hiding from true unlearning.

---

## 4. Key Papers by Relevance

### Must-Cite
1. **TOFU** (Maini et al., 2024) - Benchmark we use
2. **Kempner uncertainty paper** - Validates epistemic uncertainty is detectable
3. **CMU benign relearning** - Validates hiding problem

### Should-Cite
4. **Semantic entropy** (Kuhn et al., 2023) - Alternative UQ method
5. **Machine unlearning survey** (Liu, 2024) - Context
6. **ICLR 2025 loss adjustment** - Better unlearning method

### Nice-to-Cite
7. **ToKUR** - Decomposition method
8. **ACM UQ Survey** - Comprehensive background

---

## 5. Open Questions from Literature

1. **Does UR actually distinguish hiding from true unlearning?**
   - Not yet tested (our contribution)

2. **Which UQ method is best for unlearning verification?**
   - Token entropy: simple but may conflate uncertainty types
   - Semantic entropy: more principled but expensive
   - Activation probes: promising but requires training

3. **Can we verify unlearning without adversarial methods?**
   - Current: Benign relearning (adversarial)
   - Our proposal: UQ-based (non-adversarial)

---

## 6. Conclusion

The literature supports our research direction:

| Aspect | Literature Support |
|--------|-------------------|
| Problem exists | ✅ CMU confirms hiding problem |
| UQ can detect knowledge | ✅ Kempner shows epistemic uncertainty is linearly represented |
| Method is novel | ✅ No prior UQ-based unlearning verification |
| Token entropy may be insufficient | ⚠️ May need semantic entropy or probes |

**Key Gap We Address**: No existing work uses uncertainty quantification to verify LLM unlearning quality.
