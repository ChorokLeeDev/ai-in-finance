# V6: MoE-RAG Attention

> **"Learn diverse attention heads on documents alone, route queries to relevant heads at inference. Zero configuration required."**

---

## Status

| Aspect | Status |
|--------|--------|
| Vision | Documented |
| Core Idea | **MoE-RAG Attention** |
| Technical Approach | Detailed |
| Literature Validation | Done (gap confirmed) |
| Prototype | Not started |
| Feasibility | Promising |

---

## The Problem

```
Training time: We have documents, NO queries
Inference time: Query arrives, need to attend

How do you learn attention without knowing what to attend FOR?
```

**Insight:** Queries cluster into types. Different types need different attention patterns. We can pre-learn patterns without knowing specific queries.

---

## The Solution: MoE-RAG

```
Standard RAG:
  Query → Retrieve → Concatenate (equal weight) → Generate

MoE-RAG:
  Query → Retrieve → Route to attention heads → Weighted attention → Generate
                     ↑
                     Heads pre-learned on documents
                     Each head specializes on query type
```

---

## Documents

| Document | Purpose |
|----------|---------|
| [MOE_RAG_ATTENTION.md](MOE_RAG_ATTENTION.md) | **Core research direction** |
| [VISION.md](VISION.md) | Original motivation |
| [TECHNICAL_SKETCH.md](TECHNICAL_SKETCH.md) | Earlier technical approach |
| [VALIDATION_PLAN.md](VALIDATION_PLAN.md) | Validation strategy |
| [LITERATURE_TODO.md](LITERATURE_TODO.md) | Literature analysis |

---

## Key Technical Ideas

1. **Mixture of Attention Experts**: Multiple attention heads, each specialized
2. **Query-type spectrum**: Factual, procedural, causal, comparative, etc.
3. **Self-supervised training**: Generate synthetic queries per type
4. **Diversity by construction**: Each head trains on different query type
5. **Learned routing**: Query → relevant head(s)

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
4. [ ] Compare to baselines (AttentionRAG, RankRAG)

### Medium-term (Week 7-12)

1. [ ] Ablation studies (# heads, diversity loss, routing)
2. [ ] Cross-domain transfer experiments
3. [ ] Analysis of learned heads
4. [ ] Paper writing

---

## Literature Gap (Confirmed)

| Existing Work | What It Does | Our Gap |
|---------------|--------------|---------|
| AttentionRAG | Uses LLM's built-in attention | Not learned, not diverse |
| RankRAG | Hard ranking of contexts | Not soft attention |
| FiD | Implicit cross-attention | Not explicit, not interpretable |
| SEER | Evidence extraction | Binary, not weighted |
| SimRAG | Self-supervised QA | No attention mechanism |
| ALoFTRAG | Auto fine-tuning | No MoE, no attention heads |

**Our contribution:** MoE applied to RAG attention with self-supervised diverse heads.

---

## Connection to Previous Work

| Previous | Lesson | Applied in V6 |
|----------|--------|---------------|
| V3 FK Attribution | Structure matters | Learn attention patterns |
| V5 Shrinkage | Don't claim existing theory | Novel MoE application |
| Workshop Paper | Hand-crafted works but doesn't scale | Self-supervised replaces manual |
| Quant Research | Apply methods to new domains | MoE → RAG attention |

---

## Target

- **Venue**: NeurIPS 2026
- **Timeline**: ~4 months to submission-ready
- **Paper title**: "MoE-RAG: Mixture of Attention Experts for Zero-Config RAG"

---

## Novel Contributions

1. Apply MoE to RAG attention (not FFN)
2. Self-supervised head training via query-type spectrum
3. Diversity by construction + regularization
4. Zero-config deployment
5. Interpretable attention (which heads activated?)

---

## The Bet

**If this works**: Zero-config RAG that adapts via learned attention heads.

**If this fails**: We learn whether query-type clustering is the right inductive bias.

---

*Created: 2025-12-09*
*Updated: 2025-12-09 (MoE-RAG focus)*
*Author: Chorok Lee*
