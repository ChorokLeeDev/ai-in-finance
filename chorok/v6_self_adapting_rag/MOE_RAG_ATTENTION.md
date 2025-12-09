# MoE-RAG: Mixture of Attention Experts for Zero-Config RAG

**Created**: 2025-12-09
**Status**: Core Research Direction
**Target**: NeurIPS 2026

---

## One-Liner

> **"Learn diverse attention heads on documents alone, route queries to relevant heads at inference. Zero configuration required."**

---

## The Problem

### Standard RAG Limitation

```
Query → Retrieve (embedding similarity) → Concatenate → Generate

Problems:
1. All retrieved passages treated equally
2. No learned attention over retrieval
3. Can't pre-learn attention without knowing queries
4. Requires per-domain configuration
```

### The Core Challenge

```
Training time: We have documents, NO queries
Inference time: Query arrives, need to attend

How do you learn attention without knowing what to attend FOR?
```

---

## The Insight

**Queries cluster into types. Different types need different attention patterns.**

```
Query types:              Attention pattern:
─────────────────────────────────────────────────
"What is X?"              → Focus on definitions
"How do I do X?"          → Focus on procedures/steps
"X vs Y?"                 → Focus on both, comparisons
"Why did X fail?"         → Focus on error patterns
"What if X?"              → Focus on conditionals/edge cases
```

**Key insight:** We can pre-learn attention PATTERNS without knowing specific queries.

---

## MoE-RAG Architecture

### Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Query                                │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                    Router                               │
│         Classify query → head weights                   │
│         [0.7, 0.2, 0.1, 0.0, ...]                       │
└─────────────────────────────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │  Head 1  │    │  Head 2  │    │  Head K  │
    │ Factual  │    │Procedural│    │Diagnostic│
    └──────────┘    └──────────┘    └──────────┘
          │               │               │
          └───────────────┼───────────────┘
                          ▼
┌─────────────────────────────────────────────────────────┐
│              Weighted Attention Combination             │
│         final_attn = Σ weight_k × head_k(query, docs)   │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                    Generation                           │
│         Generate answer using attended context          │
└─────────────────────────────────────────────────────────┘
```

### Components

```python
class MoERAGAttention:
    """Mixture of Attention Experts for RAG"""

    def __init__(self, num_heads=8):
        self.heads = [AttentionHead() for _ in range(num_heads)]
        self.router = QueryRouter()

    def forward(self, query, retrieved_docs):
        # Route query to heads
        head_weights = self.router(query)  # [K]

        # Each head computes attention
        attentions = []
        for k, head in enumerate(self.heads):
            if head_weights[k] > threshold:
                attn = head.attend(query, retrieved_docs)
                attentions.append(head_weights[k] * attn)

        # Combine weighted attentions
        final_attention = sum(attentions)

        return final_attention, head_weights
```

---

## Self-Supervised Training

### The Key: Query-Type Spectrum

```python
# Query types (can be predefined or discovered)
QUERY_TYPES = [
    "factual",      # What is X?
    "procedural",   # How to do X?
    "causal",       # Why does X happen?
    "comparative",  # X vs Y?
    "conditional",  # What if X?
    "aggregative",  # List all X
    "temporal",     # When did X?
    "diagnostic",   # What's wrong with X?
]
```

### Training Pipeline

```
Step 1: Synthetic Query Generation
────────────────────────────────────
For each document:
  For each query_type:
    Generate synthetic query of that type

Example:
  Document: "Python lists are mutable sequences..."

  Factual:     "What is a Python list?"
  Procedural:  "How do I create a list in Python?"
  Comparative: "What's the difference between list and tuple?"
  Conditional: "What happens if I modify a list while iterating?"

Step 2: Train Heads on Type-Specific Data
────────────────────────────────────
Head k trains ONLY on queries of type k

  head["factual"].train(factual_queries, docs)
  head["procedural"].train(procedural_queries, docs)
  ...

This guarantees diversity by construction.

Step 3: Train Router
────────────────────────────────────
Router learns: query → query_type probabilities

  router.train(synthetic_queries, query_type_labels)

Step 4: Add Diversity Regularization (Optional)
────────────────────────────────────
Mild loss to prevent head collapse:

  diversity_loss = cosine_similarity(head_i, head_j) for i≠j
```

---

## Ensuring Head Diversity

### Approach A: Diversity by Construction (Recommended)

```
Each head trains on different query type
→ Heads see different data
→ Heads learn different patterns
→ Diversity guaranteed

No loss function tuning needed.
```

### Approach B: Loss Function Regularization

```python
def diversity_loss(heads, query, docs):
    # Compute attention patterns
    patterns = [h.attend(query, docs) for h in heads]

    # Penalize similar patterns
    loss = 0
    for i in range(len(heads)):
        for j in range(i+1, len(heads)):
            loss += cosine_similarity(patterns[i], patterns[j])

    return loss / (len(heads) * (len(heads)-1) / 2)
```

### Approach C: Hybrid (Best)

```python
# 1. Train on type-specific data (diversity by construction)
# 2. Add mild regularization (insurance against collapse)

total_loss = (
    task_loss                           # Generate correct answer
    + λ1 * diversity_loss               # Heads should differ
    + λ2 * load_balance_loss            # All heads should be used
    + λ3 * specialization_loss          # Each head should be confident
)
```

---

## Full Training Algorithm

```python
def train_moe_rag(corpus, num_heads=8):
    """Self-supervised training of MoE-RAG"""

    # Step 1: Define or discover query types
    query_types = discover_query_types(corpus)  # or use predefined
    assert len(query_types) == num_heads

    # Step 2: Generate synthetic training data
    training_data = {}
    for query_type in query_types:
        training_data[query_type] = []
        for doc in corpus:
            query = generate_synthetic_query(doc, query_type)
            training_data[query_type].append((query, doc))

    # Step 3: Train heads (diversity by construction)
    heads = [AttentionHead() for _ in range(num_heads)]
    for k, query_type in enumerate(query_types):
        for query, doc in training_data[query_type]:
            heads[k].train_step(query, doc)

    # Step 4: Train router
    router = QueryRouter()
    all_queries = []
    all_labels = []
    for k, query_type in enumerate(query_types):
        for query, doc in training_data[query_type]:
            all_queries.append(query)
            all_labels.append(k)
    router.train(all_queries, all_labels)

    # Step 5: Fine-tune with diversity regularization (optional)
    for epoch in range(fine_tune_epochs):
        for query, doc in sample_data:
            head_weights = router(query)
            attentions = [h.attend(query, doc) for h in heads]

            task_loss = compute_task_loss(query, doc, attentions, head_weights)
            div_loss = compute_diversity_loss(attentions)

            loss = task_loss + λ * div_loss
            loss.backward()
            optimizer.step()

    return MoERAG(heads, router)
```

---

## Inference (Zero-Config)

```python
def inference(moe_rag, query, retriever, generator):
    """Zero-config inference"""

    # Standard retrieval
    retrieved_docs = retriever.retrieve(query, top_k=10)

    # MoE attention (the novel part)
    attention_weights, head_activations = moe_rag(query, retrieved_docs)

    # Weight documents by attention
    weighted_context = apply_attention(retrieved_docs, attention_weights)

    # Generate
    answer = generator(query, weighted_context)

    # Interpretability: which heads fired?
    active_heads = get_active_heads(head_activations)

    return answer, active_heads
```

---

## Why This Enables "Dump and Use"

```
Traditional RAG setup:
  1. Choose retriever          ← requires expertise
  2. Set chunk size            ← requires tuning
  3. Tune top-k                ← requires tuning
  4. Design prompt             ← requires expertise
  5. Configure per domain      ← repeated effort

MoE-RAG setup:
  1. Dump documents
  2. Run self-supervised training
  3. Use

Why zero-config?
  - Query types discovered automatically (or generic)
  - Attention patterns learned automatically
  - Router learned automatically
  - No per-domain tuning (types are domain-agnostic)
```

---

## Comparison to Existing Work

| Method | Attention | Learned? | Diverse? | Zero-Config? |
|--------|-----------|----------|----------|--------------|
| Standard RAG | None (equal weight) | No | N/A | Needs tuning |
| AttentionRAG | LLM internal | No (heuristic) | No | Needs tuning |
| RankRAG | Hard ranking | Yes (supervised) | N/A | Needs labels |
| FiD | Cross-attention | Yes (implicit) | No | Needs tuning |
| SEER | Evidence extraction | Yes | No | Needs config |
| **MoE-RAG** | **Soft attention** | **Yes (self-sup)** | **Yes (by design)** | **Yes** |

---

## Research Questions

### Technical

1. **How many heads?**
   - Fixed (8)? Discovered from data? Adaptive?

2. **Predefined vs discovered query types?**
   - Predefined: Easier, interpretable
   - Discovered: More general, data-driven

3. **How to generate good synthetic queries?**
   - LLM prompting per type
   - Template-based
   - Mix of both

4. **Soft vs hard routing?**
   - Soft: Activate multiple heads (robust)
   - Hard: One head per query (efficient)

5. **Granularity?**
   - Document-level attention?
   - Passage-level?
   - Sentence-level?

### Empirical

1. Does MoE attention beat single attention?
2. Does diversity regularization help?
3. Do heads actually specialize?
4. Does it transfer across domains?
5. What's the compute overhead?

---

## Experiments Plan

### Datasets

| Dataset | Task | Why |
|---------|------|-----|
| Natural Questions | Open QA | Standard benchmark |
| TriviaQA | Open QA | Multi-passage |
| HotpotQA | Multi-hop QA | Tests attention over multiple docs |
| FEVER | Fact verification | Conflict resolution |
| MS MARCO | Passage ranking | Relevance judgment |

### Baselines

1. Standard RAG (no attention)
2. AttentionRAG (heuristic attention)
3. RankRAG (learned ranking)
4. Single learned attention (no MoE)

### Ablations

1. Number of heads: 2, 4, 8, 16
2. With/without diversity loss
3. Predefined vs discovered query types
4. Soft vs hard routing
5. With/without router (random head selection)

### Metrics

- Accuracy / F1 / EM (task performance)
- Head utilization (are all heads used?)
- Head specialization (do heads differ?)
- Attention entropy (is attention focused?)
- Inference time (efficiency)

---

## Paper Structure

```
Title: "MoE-RAG: Mixture of Attention Experts for
        Zero-Configuration Retrieval-Augmented Generation"

1. Introduction
   - RAG needs per-domain configuration
   - Core problem: can't learn attention without queries
   - Our insight: learn attention PATTERNS, route queries to patterns

2. Related Work
   - RAG improvements (Self-RAG, RankRAG, AttentionRAG)
   - Mixture of Experts
   - Self-supervised retrieval

3. Method
   - MoE-RAG architecture
   - Query-type spectrum
   - Self-supervised training
   - Diversity mechanisms

4. Experiments
   - Datasets and baselines
   - Main results
   - Ablation studies
   - Analysis of learned heads

5. Analysis
   - What do heads learn?
   - When does MoE help most?
   - Failure cases

6. Conclusion
   - Zero-config RAG is possible via MoE attention
   - Patterns can be learned without queries
```

---

## Novel Contributions

1. **Apply MoE to RAG attention** (not just FFN layers)
2. **Self-supervised head training** via query-type spectrum
3. **Diversity by construction** + regularization
4. **Zero-config deployment** (no per-domain tuning)
5. **Interpretable attention** (which heads activated?)

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Heads collapse to same pattern | Diversity by construction + regularization |
| Synthetic queries don't match real | Diverse generation + evaluation on real queries |
| Overhead of multiple heads | Sparse routing (only activate top-k heads) |
| Doesn't beat simpler methods | Strong ablations to show when MoE helps |
| Query types don't generalize | Test cross-domain transfer |

---

## Next Steps

### Immediate (Week 1-2)

1. [ ] Implement basic MoE-RAG architecture
2. [ ] Implement synthetic query generation (8 types)
3. [ ] Test on small dataset (Natural Questions subset)
4. [ ] Verify heads learn different patterns

### Short-term (Week 3-6)

1. [ ] Full training pipeline
2. [ ] Diversity regularization experiments
3. [ ] Benchmark on NQ, TriviaQA, HotpotQA
4. [ ] Compare to baselines

### Medium-term (Week 7-12)

1. [ ] Ablation studies
2. [ ] Cross-domain transfer experiments
3. [ ] Analysis of learned heads
4. [ ] Paper writing

---

## Connection to V6 Vision

```
V6 Vision: "Dump data, get working Q&A. No configuration."

MoE-RAG achieves this by:
  1. Learning attention patterns on documents (no queries needed)
  2. Routing queries to relevant patterns (no manual config)
  3. Diverse heads cover different information needs
  4. Works across domains without retuning
```

---

*This document defines the MoE-RAG Attention research direction.*
*Core insight: Pre-learn diverse attention patterns, route queries at inference.*
