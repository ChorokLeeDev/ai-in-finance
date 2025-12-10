# MoE-RAG & Style Normalization Benchmark Findings

**Date:** 2025-12-09
**Status:** Negative results - neither MoE attention nor style normalization improve retrieval

---

## Executive Summary

We tested multiple hypotheses for improving RAG retrieval:

1. **MoE Attention**: Can learned mixture-of-experts attention beat simple similarity? **No.**
2. **ToneNorm (Style Normalization)**: Can normalizing conversational tone help messy queries? **No.**
3. **Multi-Source Fusion**: Can source-aware normalization help heterogeneous corpora? **No.**

| Experiment | Simple Similarity | Our Approach |
|------------|------------------|--------------|
| MoE-RAG (general) | 26% MRR | 10% MRR |
| MoE-RAG (adversarial) | 100% R@1 | 40% R@1 |
| ToneNorm (messy) | 61.7% MRR | 61.7% MRR |
| Multi-Source | 100% Hit@3 | 28.6% Hit@3 |

**Key finding:** Modern sentence embeddings are **extremely robust**. They already capture:
- Semantic similarity despite typos and informality
- Content similarity across different writing styles
- Topic structure across heterogeneous sources

Any learned transformation on top of these embeddings **destroys discriminative information** without adding value.

---

## Experiments Conducted

### Experiment 1: MoE-RAG General Benchmark (100 docs)

- **Setup:** 100 synthetic documents, 10 test queries
- **Result:** Simple similarity MRR 0.26, MoE-RAG MRR 0.10
- **Conclusion:** MoE-RAG significantly worse

### Experiment 2: MoE-RAG Distractor Corpus (25 docs)

- **Setup:** 5 topics × (1 correct + 4 topically similar distractors)
- **Similarity gap:** 0.17 - 0.33 (correct much higher than distractors)
- **Result:** Simple = 100%, MoE-RAG = 20%
- **Conclusion:** Distractors too easy, simple similarity handles perfectly

### Experiment 3: MoE-RAG Adversarial Corpus (25 docs)

- **Setup:** Distractors keyword-stuffed with query terms
- **Similarity gap:** 0.01 - 0.13 (much closer!)
- **Result:** Simple = 100%, MoE-RAG = 40%
- **Conclusion:** Even with gap < 0.05, simple similarity wins

### Experiment 4: ToneNorm on Messy Real-World Data

- **Setup:** 8 docs (Slack threads, wikis, runbooks, emails, PDFs), 5 informal queries with typos
- **Hypothesis:** Normalizing "tone" (formality, conversational markers) will help match informal queries to formal docs
- **Result:** Simple = 61.7% MRR, ToneNorm = 61.7% MRR (identical!)
- **Analysis:** ToneNorm DOES increase similarity between formal/informal pairs (+0.1073 for "API limit" query), but baseline already captures this without normalization
- **Conclusion:** The tone signal is useful information, not noise

### Experiment 5: Multi-Source Fusion (10 docs, 3 topics, 5 source types)

- **Setup:** Password reset info in Wiki/Slack/Runbook/Email, API limits in Wiki/SO/Slack, Pricing in Wiki/PDF/Email
- **Hypothesis:** Learning per-source normalization enables better cross-source matching
- **Result:** Simple = 100% Hit@3, Multi-Source = 28.6% Hit@3
- **Analysis:** Contrastive training on small corpus overfits, loses discrimination
- **Conclusion:** Baseline already handles cross-source retrieval perfectly

---

## Why Do All These Approaches Fail?

### The Core Realization

Modern sentence transformers (MiniLM, BGE, E5, etc.) are **already trained to be invariant to**:
- Writing style (formal/informal)
- Typos and misspellings
- Surface-level keyword overlap without semantic match
- Source type differences (chat vs documentation)

This is **exactly what we were trying to learn**. But it's already baked into the embeddings through:
- Massive pretraining on diverse text
- Contrastive learning on paraphrase pairs
- Hard negative mining during training

### Why MoE-RAG Fails

1. **Embeddings already capture what we want**: Different "heads" for different query types is unnecessary when the embedding already represents semantic content well.

2. **Training destroys information**: Our self-supervised training either:
   - Overfits on synthetic queries (hurts generalization)
   - Collapses the embedding space (loses discrimination)

3. **No room for improvement**: When baseline achieves 100% on adversarial/multi-source tasks, learned components can only hurt.

### Why ToneNorm Fails

1. **Tone IS information**: Informal queries often match informal docs better because they share context (user asking, Slack has user conversations).

2. **Embeddings already handle tone**: MiniLM was trained on diverse corpora including informal text. It doesn't need explicit normalization.

3. **Statistical normalization removes signal**: PCA on formal/informal differences captures variation, not noise.

### Why Multi-Source Fails

1. **No ground truth supervision**: Inferring topic clusters from similarity creates circular reasoning.

2. **Small corpus overfitting**: 10 docs with 14 triplets → loss converges to 0, all embeddings become identical.

3. **Baseline already wins**: 100% Hit@3 means perfect cross-source retrieval without any learning.

---

## What Would Actually Help?

For learned retrieval transformations to provide value, we need tasks where:

1. **Embeddings fundamentally fail** (not just close calls)
2. **Ground truth labels are available** (not self-supervised)
3. **Corpus is large enough** to prevent overfitting

### Potential Directions

| Direction | Why Embeddings Might Fail | Required Scale |
|-----------|--------------------------|----------------|
| Cross-lingual | Different semantic spaces | 10K+ doc pairs |
| Multi-modal (image+text) | Different modality encoders | 50K+ examples |
| Very long docs (10K+ tokens) | Fixed embedding dimension | 1K+ long docs |
| Structured data (tables) | Layout/structure not captured | 5K+ tables |
| Domain-specific (medical, legal) | Out-of-distribution | Domain corpus |

### The Honest Answer

For **standard text RAG on English documents**, the answer is:

> **Use simple cosine similarity on good embeddings. That's it.**

Learned attention/normalization adds:
- ❌ No accuracy improvement
- ❌ Training overhead
- ❌ Complexity
- ❌ Overfitting risk

For NeurIPS 2026, consider pivoting to:
1. A domain where embeddings genuinely struggle
2. A "negative results" paper documenting why learned retrieval doesn't help
3. A different research direction entirely

---

## What We Built & Tested

### Code Artifacts

1. **`moe_rag/`** - Complete MoE-RAG package
   - `model.py`: Main API (`MoERAG.from_texts()`, `.train()`, `.retrieve()`)
   - `attention.py`: MoE attention with soft/hard routing
   - `tone_norm.py`: Style normalization (ToneNorm)
   - `multi_source.py`: Source-aware multi-source RAG

2. **`benchmarks/`** - Comprehensive experiments
   - `compare_langchain.py`: LangChain comparison
   - `distractor_experiment.py`: Topical distractors
   - `adversarial_experiment.py`: Keyword-stuffed adversarial
   - `messy_real_world.py`: Enterprise chaos simulation
   - `tone_norm_experiment.py`: Style normalization test
   - `multi_source_experiment.py`: Cross-source retrieval

---

## Paper Angle (if publishing)

**Title:** "When Does Learned Retrieval Help RAG? A Comprehensive Negative Result"

**Contribution:**
1. Systematic study of 3 learned retrieval approaches (MoE attention, style normalization, multi-source fusion)
2. Demonstrate that modern sentence embeddings are sufficient for diverse scenarios
3. Identify why learned transformations fail (overfitting, information destruction)
4. Provide conditions where learned retrieval might genuinely help

**Honest claim:**
> "We find that for standard RAG retrieval across diverse scenarios (adversarial distractors, informal queries, heterogeneous sources), learned retrieval transformations provide no benefit over cosine similarity. Modern sentence embeddings are sufficiently robust that additional learned components only add overhead, complexity, and overfitting risk without improving accuracy."

---

## Implications

1. **For practitioners:** Use simple RAG pipelines with good embeddings (BGE, E5, OpenAI). Don't add learned components unless you have a specific failure case.

2. **For researchers:** The "learned retrieval for text" direction is likely exhausted. Pivot to:
   - Cross-lingual retrieval
   - Multi-modal (image + text)
   - Structured data (tables, code)
   - Domains where embeddings genuinely fail

3. **For this project (NeurIPS 2026 goal):**
   - Option A: Negative results paper (valuable but harder to publish)
   - Option B: Find a domain where embeddings fail, then learned retrieval helps
   - Option C: Pivot to generation/synthesis instead of retrieval

---

## Files Created

```
v6_self_adapting_rag/
├── moe_rag/
│   ├── __init__.py
│   ├── model.py
│   ├── attention.py
│   ├── encoder.py
│   ├── chunker.py
│   ├── query_gen.py
│   ├── trainer.py
│   ├── cli.py
│   ├── tone_norm.py          # NEW: Style normalization
│   └── multi_source.py       # NEW: Multi-source fusion
└── benchmarks/
    ├── compare_langchain.py
    ├── distractor_experiment.py
    ├── adversarial_experiment.py
    ├── messy_real_world.py
    ├── tone_norm_experiment.py    # NEW
    ├── multi_source_experiment.py # NEW
    └── FINDINGS.md               # This document
```

---

*Generated from benchmark experiments on 2025-12-09*
*Research conducted as exploration for NeurIPS 2026 submission*
