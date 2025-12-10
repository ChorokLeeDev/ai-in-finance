# V8: Epistemic Uncertainty for LLM Unlearning Verification

## Literature Review - December 10, 2025

---

## 1. Problem Background

### What is Machine Unlearning?
Selectively removing the influence of specific training data from a model without retraining from scratch.

**Why it matters:**
- GDPR "right to be forgotten"
- Removing copyrighted content
- Erasing dangerous knowledge (biosecurity, cyberweapons)
- Privacy compliance / data ownership

### The Verification Problem
> "How can model providers convincingly demonstrate that the targeted data has been thoroughly and irreversibly removed?"

This is the **central unresolved question** in the field.

---

## 2. Current Benchmarks and Their Weaknesses

### TOFU (Task of Fictitious Unlearning)
- **Source:** [CMU, COLM 2024](https://locuslab.github.io/tofu/)
- **Setup:** 200 synthetic author profiles, 20 QA pairs each
- **Metrics:** Forget Quality + Model Utility
- **Problem:** Independently tests forget/retain queries

### WMDP (Weapons of Mass Destruction Proxy)
- **Focus:** Dangerous knowledge (bio, cyber, chemical)
- **Format:** 4000+ multiple-choice questions
- **Problem:** Sensitive to simple input perturbations

### CMU Critique (April 2025)
**Source:** [ML@CMU Blog](https://blog.ml.cmu.edu/2025/04/18/llm-unlearning-benchmarks-are-weak-measures-of-progress/)

**Key findings:**
1. **TOFU flaw:** Concatenating forget + retain queries → model answers BOTH correctly
2. **WMDP flaw:** Swapping one MCQ option with forget keyword → 28% accuracy drop
3. **Root cause:** Benchmarks don't test dependencies between forget/retain data

**Quote:**
> "The majority of popular evaluation benchmarks (including TOFU and WMDP) are weak measures of progress."

---

## 3. The "Hiding vs True Unlearning" Problem

### Fundamental Limits of Unlearning

**Key Paper:** "The Fundamental Limits of LLM Unlearning" (2024-2025)

**Core Finding:** Exact unlearning verification is **coNP-Hard**
- Cannot be efficiently verified in general case
- Current methods "hide rather than erase"

**Adversarial Recovery Results:**
- ~55.2% of "unlearned" knowledge recoverable via adversarial attacks
- Knowledge persists in weight space even when output changes
- Jailbreaking techniques can resurrect supposedly forgotten information

### The Two Types of "Unlearning"

```
HIDING (what current methods do):
├── Output changes: ✓ (looks unlearned)
├── Weights: Knowledge still encoded
├── Uncertainty: LOW (model "knows" but won't say)
└── Adversarial recovery: POSSIBLE (~55%)

TRUE UNLEARNING (the goal):
├── Output changes: ✓
├── Weights: Knowledge actually removed
├── Uncertainty: HIGH (model genuinely doesn't know)
└── Adversarial recovery: NOT POSSIBLE
```

---

## 4. Current Unlearning Methods

| Method | Description | Problems |
|--------|-------------|----------|
| **Gradient Ascent (GA)** | Maximize loss on forget data | Catastrophic collapse, no convergence |
| **Negative Preference Optimization (NPO)** | Alignment-inspired loss | Slower collapse than GA |
| **LoRA fine-tuning** | Parameter-efficient unlearning | Lacks plasticity |
| **Gradient Descent + Random Labels** | "Confuse" the model | Incomplete forgetting |

### The Stopping Criterion Problem

**Critical Issue:** No well-defined convergence point

> "Since gradient ascent-based unlearning lacks a well-defined point of convergence, it is inherently difficult to determine when to stop the unlearning process."

**Current "solutions":**
- Fixed number of epochs (arbitrary)
- Perplexity thresholds (can cause collapse)
- Validation set accuracy (doesn't guarantee true unlearning)

**None address whether knowledge is actually removed vs hidden.**

---

## 5. SAE-Based Unlearning (Crowded Space)

### Methods in 2025
| Method | Approach | Status |
|--------|----------|--------|
| **CRISP** | Contrastive SAE features | Published |
| **SAUCE** | SAE + alignment for concepts | Published |
| **SAEmnesia** | Directly ablate SAE neurons | Published |
| **DSG** | Dynamic sparse graphs | Published |
| **SSPU** | Sparse subspace pursuit | Published |

### Why This Direction Is Saturated
- All focus on **localization** (where is knowledge?)
- None address **verification** (is it truly gone?)
- Polysemanticity remains a challenge
- Surgical precision is limited

---

## 6. Epistemic Uncertainty for Unlearning

### Foundational Paper
**"Evaluating Machine Unlearning via Epistemic Uncertainty"** - [Becker & Liebig, 2022](https://arxiv.org/abs/2208.10836)

**Key insight:**
> Data that should no longer influence predictions should demonstrate **increased epistemic uncertainty**.

**Approach:**
- Use epistemic uncertainty as proxy for measuring unlearning success
- First general evaluation metric for machine unlearning

**Limitation:** Developed for traditional ML (image classification), NOT LLMs.

### LLM Uncertainty Quantification Methods

| Method | Mechanism | Pros | Cons |
|--------|-----------|------|------|
| **Semantic Entropy** | Cluster equivalent meanings | State-of-art accuracy | N samples needed |
| **Token Entropy** | Next-token distribution | Single pass | Doesn't capture meaning |
| **Ensemble Disagreement** | Multiple models | Well-calibrated | Expensive |
| **MC Dropout** | Dropout at inference | Cheap | Underestimates uncertainty |
| **Self-Verbalized** | Ask model its confidence | Easy | Often miscalibrated |

### UQ-Robustness Correlation (Key Finding)

From Harvard NeurIPS 2024 workshop paper:
> "There exists a **slight correlation** between robust accuracy and robust uncertainty."

**Implication:** Epistemic uncertainty may distinguish hiding from true unlearning.

---

## 7. Gap Analysis: The Research Opportunity

### What Exists:
| Area | Status |
|------|--------|
| SAE-based unlearning | Crowded (5+ methods in 2025) |
| Behavioral benchmarks | Weak (CMU critique) |
| Epistemic UQ for ML unlearning | Exists (Becker & Liebig) |
| LLM uncertainty methods | Well-developed |
| Stopping criterion research | Acknowledged problem, no solution |

### What's Missing:

| Gap | Description | Impact |
|-----|-------------|--------|
| **LLM-specific UQ verification** | Becker & Liebig not extended to LLMs | High |
| **UQ as stopping criterion** | No work uses UQ to decide "when done" | Very High |
| **Hiding vs true unlearning distinction** | No metric reliably separates them | Critical |
| **Iterative UQ-guided unlearning** | No feedback loop approach | Novel |

### The Opportunity

**Nobody is using epistemic uncertainty as:**
1. A diagnostic for hiding vs true unlearning
2. A feedback signal during iterative unlearning
3. A principled stopping criterion

---

## 8. Key Hypothesis

> **If knowledge is truly unlearned (removed from weights), the model should exhibit epistemic uncertainty similar to a base model that was never trained on that data.**

### Testable Predictions:

| Scenario | Expected UQ | Adversarial Recovery |
|----------|-------------|---------------------|
| Before unlearning | Low (confident) | N/A |
| After hiding | Low (still knows) | Possible |
| After true unlearning | High (like base model) | Not possible |

### Why This Should Work:

1. **Hiding preserves knowledge** → internal states allow confident generation → low UQ
2. **True removal eliminates knowledge** → model must rely on priors → high UQ
3. **Base model comparison** → provides natural "never learned" baseline

---

## 9. Supporting Evidence

### Evidence FOR the hypothesis:

1. **UQ-Robustness correlation exists** (Harvard NeurIPS 2024)
   - Uncertainty correlates with adversarial robustness
   - Suggests UQ captures something about knowledge persistence

2. **Stopping criterion is acknowledged problem**
   - Community recognizes need for better metrics
   - UQ offers principled information-theoretic approach

3. **Epistemic UQ worked for classical ML** (Becker & Liebig)
   - Same intuition should transfer to LLMs
   - Just needs LLM-specific uncertainty estimation

### Potential Challenges:

1. **Calibration** - LLM uncertainty may be miscalibrated
2. **Confounders** - Other factors affect uncertainty
3. **Measurement** - Semantic entropy is expensive, simpler methods may not work

---

## 10. Related Active Research

### SemEval 2025 Challenge
- [LLM Unlearning Challenge](https://llmunlearningsemeval2025.github.io/)
- ~100 submissions from 26 teams
- Shows active interest in verification

### Recent Publications
- Nature Machine Intelligence review (Feb 2025)
- ICLR 2025: "LLM Unlearning via Loss Adjustment"
- ACL 2024: "Machine Unlearning of Pre-trained LLMs"
- NeurIPS 2024: Multiple unlearning papers

---

## 11. Sources

### Core Papers
- [TOFU Benchmark](https://locuslab.github.io/tofu/) - COLM 2024
- [Epistemic UQ for Unlearning](https://arxiv.org/abs/2208.10836) - Becker & Liebig 2022
- [Fundamental Limits of Unlearning](https://arxiv.org/abs/2406.10952) - 2024
- [Survey on LLM Unlearning](https://arxiv.org/html/2503.01854v2) - 2025

### Critical Analysis
- [CMU: Benchmarks are Weak](https://blog.ml.cmu.edu/2025/04/18/llm-unlearning-benchmarks-are-weak-measures-of-progress/) - April 2025

### SAE-Based Methods
- [Neuronpedia](https://www.neuronpedia.org/) - SAE feature visualization
- [CRISP, SAUCE, SAEmnesia papers] - See GitHub awesome-llm-unlearning

### Resource Collections
- [Awesome LLM Unlearning](https://github.com/chrisliu298/awesome-llm-unlearning)
- [Stanford: Unlearning in 2024](https://ai.stanford.edu/~kzliu/blog/unlearning)

---

*Review completed: 2025-12-10*
*Direction: Epistemic UQ as iterative stopping criterion for unlearning*
