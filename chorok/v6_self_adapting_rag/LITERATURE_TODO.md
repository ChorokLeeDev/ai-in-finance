# V6: Literature Search TODO

**Created**: 2025-12-09
**Purpose**: Validate the research gap before committing

---

## Critical Questions

### Q1: Does "zero-config RAG" exist?

**Search terms**:
- "Zero-shot RAG"
- "Automatic RAG configuration"
- "Self-configuring retrieval"
- "No-tuning RAG"

**What to look for**:
- Systems that work without per-domain tuning
- Automatic chunk size / retrieval parameter selection
- Self-adapting to new domains

### Q2: Does "self-supervised RAG attention" exist?

**Search terms**:
- "Self-supervised retrieval attention"
- "Learned RAG attention"
- "Unsupervised retrieval weighting"
- "Contrastive learning retrieval"

**What to look for**:
- Learning attention weights without labeled data
- Self-training for document relevance
- Synthetic query generation for RAG

### Q3: What's the state of enterprise RAG?

**Search terms**:
- "Enterprise knowledge base QA"
- "Corporate RAG systems"
- "Internal documentation QA"
- "Enterprise LLM deployment"

**What to look for**:
- How companies deploy RAG
- What engineering is required
- Pain points in enterprise RAG

---

## Known Related Work (from earlier search)

### RAG Improvements

| Paper | What It Does | Difference from V6 |
|-------|--------------|-------------------|
| AttentionRAG (2025) | Attention-guided pruning | Post-hoc, not learned per-domain |
| Self-RAG (ICLR 2024) | Self-reflective retrieval | Decides when to retrieve, not what to attend |
| RankRAG (NeurIPS 2024) | Context ranking | Ranking, not self-adapting |
| REPLUG (2023) | Black-box LLM retrieval | Requires retrieval tuning |

### Self-Supervised Retrieval

| Paper | What It Does | Difference from V6 |
|-------|--------------|-------------------|
| Contriever (2022) | Unsupervised dense retrieval | For retrieval, not attention |
| SimCSE (2021) | Contrastive embeddings | General embeddings, not RAG-specific |
| LaPraDoR (2022) | Unsupervised passage retrieval | Passage ranking, not attention |

### Knowledge Conflict

| Paper | What It Does | Difference from V6 |
|-------|--------------|-------------------|
| Probing Latent Conflict | Detect conflicts | Detection only, not resolution |
| Astute RAG | Reliability estimation | Requires training data |
| FaithfulRAG | Conflict resolution | Fact-level, not attention-level |

---

## Deep Dive Papers to Read

### Priority 1 (Must Read)

1. **AttentionRAG (2025)** - Closest to our attention approach
   - URL: https://arxiv.org/html/2503.10720v1
   - Question: How different is their attention from learned attention?

2. **Self-RAG (ICLR 2024)** - Self-supervised RAG
   - Question: Can their self-training approach be adapted?

3. **Contriever** - Unsupervised retrieval
   - Question: Can we use similar contrastive approach for attention?

### Priority 2 (Should Read)

4. **RankRAG (NeurIPS 2024)** - Context ranking
5. **REPLUG** - Black-box retrieval
6. **Selective Attention (NeurIPS 2024)** - Attention for transformers

### Priority 3 (Nice to Have)

7. Synthetic query generation papers
8. Enterprise RAG case studies
9. Knowledge distillation for retrieval

---

## Search Strategy

### Venues to Check

- NeurIPS 2024, 2023
- ICML 2024, 2023
- ICLR 2024, 2025
- ACL/EMNLP 2024
- SIGIR 2024
- KDD 2024

### Arxiv Searches

```
"self-supervised" AND "RAG"
"zero-shot" AND "retrieval augmented"
"automatic" AND "RAG" AND "configuration"
"enterprise" AND "RAG"
"synthetic query" AND "retrieval"
```

### Google Scholar Searches

```
"self-adapting RAG" OR "adaptive RAG"
"zero-configuration retrieval"
"learned attention retrieval"
```

---

## Gap Analysis Template

After reading each paper, fill in:

```
Paper: [Title]
What they do: [Brief summary]
What they don't do: [Gap]
How V6 differs: [Our differentiation]
```

---

## Expected Outcome

### If Gap Exists

- Proceed with V6 as planned
- Position paper against related work
- Clear contribution story

### If Gap is Small

- Find differentiation (enterprise focus? specific domain?)
- Adjust claims to be more specific
- May need to pivot angle

### If Gap Doesn't Exist

- Identify what's different about our approach
- Consider empirical comparison paper
- Or pivot to different problem

---

## Timeline

| Day | Task |
|-----|------|
| Day 1 | Read AttentionRAG, Self-RAG |
| Day 2 | Read Contriever, RankRAG |
| Day 3 | Arxiv/Scholar deep search |
| Day 4 | Gap analysis summary |
| Day 5 | Go/no-go decision |

---

*Complete this literature validation before building prototype.*
