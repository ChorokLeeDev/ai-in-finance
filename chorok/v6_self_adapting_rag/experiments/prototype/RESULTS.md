# MoE-RAG Prototype Results

**Date:** 2025-12-09
**Status:** Initial prototype validated, performance gap identified

---

## Experiment Setup

- **Passages:** 100 simple synthetic passages (8 topics: Python, ML, Database, Neural Network, API, Cloud, Version Control, Testing)
- **Queries:** 400 template-generated queries (100 per type: factual, procedural, causal, comparative)
- **Model:** 4 attention heads (one per query type), MiniLM-L6 embeddings (384d)
- **Training:** 10 epochs heads, 20 epochs router, 5 epochs fine-tuning
- **Device:** CPU

---

## Results Summary

### MoE-RAG Performance

| Metric | Value |
|--------|-------|
| Task Accuracy | 66.75% |
| Routing Accuracy | 92.50% |
| MRR | 0.8242 |
| Recall@1 | 67.75% |
| Recall@5 | 100.00% |
| Head Diversity | 0.059 |
| Soft vs Hard Delta | +0.0098 |

### Per-Type Task Accuracy

| Query Type | Accuracy |
|------------|----------|
| Factual | 56.00% |
| Procedural | 80.00% |
| Causal | 76.00% |
| Comparative | 56.00% |

### Baseline Comparisons

| Method | MRR | Recall@1 |
|--------|-----|----------|
| Single Attention | **0.8756** | **76.25%** |
| MoE-RAG (Ours) | 0.8242 | 67.75% |
| Random Routing | 0.8201 | 68.20% |

**Gap:** Single attention beats MoE by **-0.05 MRR**

---

## Analysis

### What Works

1. **Router learns perfectly** - 92.5% top-1, 100% accuracy during training
2. **Soft routing helps** - +0.01 MRR over hard routing (multi-head cooperation works)
3. **Pipeline is functional** - end-to-end training completes successfully

### What Doesn't Work

1. **Single attention baseline wins** - MoE overhead not justified on simple data
2. **Head diversity is low** (0.059) - heads learn nearly identical patterns
3. **Task accuracy varies wildly** - factual/comparative at 56%, procedural/causal at 76-80%

### Root Cause Analysis

| Problem | Cause | Evidence |
|---------|-------|----------|
| Low diversity | Passages too similar | 8 topics × repetitive variations |
| Single attention wins | No need for specialization | All queries retrievable with single pattern |
| Routing perfect but useless | Template queries have obvious type signals | "What is X" vs "How to X" trivially separable |

---

## Key Insight

**The current experiment cannot validate MoE-RAG's value proposition.**

MoE-RAG hypothesis: Different query types need different attention patterns.

Current data reality: All queries need the same attention pattern (similarity to source passage).

---

## Next Steps to Validate

### Option A: Natural Questions Dataset
- Real diverse passages from Wikipedia
- Naturally varied query styles
- Established benchmark

### Option B: Multi-Domain Dataset
- Mix passages from different domains (code, legal, medical, wiki)
- Each domain may need different attention (keyword vs semantic vs structural)
- Closer to real RAG use case

### Option C: Adversarial Setup
- Create passages that require different strategies
- E.g., one passage has answer in first sentence, another has it buried
- Forces heads to specialize

### Hyperparameter Experiments
- Increase diversity regularization weight
- Lower collapse threshold (currently 0.95)
- More training epochs (current: 10 head, 20 router, 5 finetune)

---

## Files

```
outputs/run_20251209_223029/
├── moe_rag.pt                 # Trained model
├── training_results.json      # Training metrics
├── evaluation_results.json    # Full evaluation
└── data/
    ├── simple_passages.json   # 100 passages
    └── synthetic_queries.json # 400 queries
```

---

## Conclusion

**Prototype validated, hypothesis not yet tested.**

The MoE-RAG architecture works mechanically, but we haven't created conditions where multiple specialized heads would outperform a single head. Need data where query-type-specific attention patterns emerge naturally.

---

*Next: Design experiment with data that requires diverse attention patterns.*
