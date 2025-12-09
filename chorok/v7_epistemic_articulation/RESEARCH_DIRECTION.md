# V7: Epistemic Articulation in Language Models

**Date:** 2025-12-10
**Status:** Research direction exploration
**Target:** NeurIPS 2026

---

## The Core Problem

LLM-generated text feels "robotic" because it lacks **epistemic humility** - the ability to:
- Know what it doesn't know
- Express uncertainty naturally (not as probability scores)
- Distinguish between what's grounded vs inferred
- Avoid over-generalization

### Human Expert vs LLM

| Human Expert | LLM |
|--------------|-----|
| "I'm not sure about the 2024 numbers, but in 2023..." | States 2024 numbers confidently (possibly wrong) |
| "That's outside my expertise, but I think..." | Answers everything with equal authority |
| "It depends - what specifically do you mean by X?" | Picks one interpretation and runs with it |
| "I've seen cases where... but your situation might differ" | Generalizes as if universal truth |
| Pauses, hedges, qualifies | Fluent, confident, complete |

**Key insight:** Humans constantly signal their epistemic state. LLMs flatten this into uniform confident prose. This is what makes LLM text distinguishable from human text.

---

## Proposed Approach: MC Dropout + Epistemic Reward

### Training Scheme

1. **MC Dropout at inference** → Get uncertainty estimate for each part of response
2. **Mask high-uncertainty parts** → Create controlled "don't know" situations
3. **Reward for articulating uncertainty** → Train model to express hedging for masked/uncertain parts

```
Training signal:
- Input: "What is the population of Tokyo in 2024?"
- MC Dropout shows high variance on "2024 figure"
- Good response: "I'm not certain about the exact 2024 figure, but as of 2023..."
- Bad response: "The population is 14.2 million" (confident about uncertain content)

Reward = alignment(MC dropout uncertainty, epistemic language used)
```

### Why This Works

- **Controllable ground truth** - Masking creates known "don't know" situations
- **Self-supervised** - No human labeling needed
- **Calibrated** - Uncertainty magnitude maps to hedging intensity

---

## Extension to RAG: Provenance-Aware Articulation

The framework extends naturally to RAG systems:

| Uncertainty Source | Desired Behavior |
|-------------------|------------------|
| MC Dropout high variance | "I'm not certain about X..." |
| RAG: nothing found | "I couldn't find this in the documents, but generally..." |
| RAG: conflicting sources | "Source A says X, Source B says Y..." |
| RAG + World knowledge mix | "The documents don't mention this, but it's commonly understood that..." |
| Ambiguous question | "If you mean X, then... but if Y, then..." |

### Key Innovation: Provenance Tracking

Not just "confident vs uncertain" but **where did this come from**:
- What's from retrieved documents (grounded)
- What's from world knowledge (possibly outdated)
- What's the model filling in (inference)
- Where sources conflict (disagreement)

---

## Training Signals

| Signal | Ground Truth | Reward |
|--------|--------------|--------|
| MC Dropout variance | Token-level uncertainty | Express hedging on high-variance tokens |
| RAG retrieval score | Document relevance | Cite when grounded, hedge when not |
| RAG source agreement | Conflict detection | Articulate disagreement, don't pick arbitrary side |
| Temperature=0 mode | "Be precise" instruction | Decompose answer into grounded vs inferred |

---

## What Makes This Novel

Current work does:
- Hallucination detection (post-hoc, not during generation)
- RAG citation (binary: cited or not)
- Calibration (outputs probability numbers, not natural language)
- Uncertainty quantification (measures uncertainty, doesn't articulate it)

**Nobody combines:**
1. Uncertainty quantification (MC dropout)
2. Provenance tracking (RAG retrieval scores)
3. Linguistic articulation (natural hedging, not confidence scores)

Into a unified training objective.

---

## Paper Framing Options

**Title candidates:**
1. "Epistemic Articulation: Teaching Language Models to Express What They Know and Don't Know"
2. "Beyond Calibration: Training LLMs for Natural Uncertainty Expression"
3. "Provenance-Aware Generation: Grounding LLM Confidence in Source Attribution"

**Core claim:**
> "We propose epistemic articulation training, which combines uncertainty quantification with linguistic hedging to produce LLM responses that naturally express the model's epistemic state - what it knows, what it's uncertain about, and where its information comes from."

---

## Research Questions

1. **Measurement:** How do you measure "epistemic language" in output?
   - Classifier for hedging phrases?
   - Semantic similarity to uncertainty expressions?
   - Human evaluation?

2. **Masking strategy:** What to mask during training?
   - Random tokens?
   - High-perplexity tokens?
   - Factual claims specifically?
   - Named entities and numbers?

3. **Generalization:** Does training on masked content generalize to true OOD?

4. **RAG integration:** How to combine retrieval scores with MC dropout?

5. **Evaluation:** What benchmarks test epistemic articulation?
   - TruthfulQA?
   - Custom benchmark needed?

---

## Connection to Previous Work (v6)

From v6_self_adapting_rag, we learned:
- Simple embeddings already handle style/tone variation
- Learned retrieval transformations don't help
- The problem isn't retrieval accuracy - it's **what to do when uncertain**

This motivates v7: Instead of improving retrieval, focus on **expressing uncertainty about retrieved (or not retrieved) content**.

---

## Next Steps

1. **Literature review:** What exists in epistemic uncertainty for LLMs?
2. **Experimental design:** Define the training loop precisely
3. **Baseline:** Current LLM behavior on uncertainty-requiring tasks
4. **Prototype:** MC dropout + hedging reward on small model

---

*Research direction documented on 2025-12-10*
