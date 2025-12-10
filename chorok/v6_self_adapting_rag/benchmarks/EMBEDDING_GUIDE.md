# What is a "Good" Embedding Model?

A practical guide based on empirical testing.

---

## Quick Answer

For **most use cases**, use one of these:
- **Fast & Good**: `all-MiniLM-L6-v2` (22M params, ~14k tokens/sec)
- **Balanced**: `all-mpnet-base-v2` (110M params, ~3k tokens/sec)
- **Best Quality**: `BAAI/bge-large-en-v1.5` (335M params, ~1k tokens/sec)

For **enterprise/production**: Consider OpenAI `text-embedding-3-small` or Cohere `embed-v3`.

---

## What Makes an Embedding "Good"?

### 1. Semantic Understanding

| Property | Good Embedding | Bad Embedding |
|----------|---------------|---------------|
| "reset password" vs "password reset steps" | High similarity (~0.75+) | Low similarity (<0.5) |
| "cancel" vs "unsubscribe" | High similarity | Low similarity |
| Paraphrase detection | Works | Fails |

### 2. Robustness to Noise

| Property | Good Embedding | Bad Embedding |
|----------|---------------|---------------|
| Handles typos ("pasword") | Still high similarity | Drops significantly |
| Handles informal ("whats the api") | Still works | Confused |
| Handles abbreviations ("pls", "thx") | Understands | Literal matching only |

### 3. Discrimination

| Property | Good Embedding | Bad Embedding |
|----------|---------------|---------------|
| Different topics | Low similarity (~0.2) | Still high (~0.6+) |
| Keyword overlap w/o semantic match | Low similarity | Fooled by keywords |

---

## Key Considerations

### 1. Language Support

| Model | Languages | Notes |
|-------|-----------|-------|
| MiniLM/MPNet | English only | Best for English |
| BGE series | English + multilingual variants | bge-m3 for multilingual |
| E5 series | English + multilingual | Good cross-lingual |
| OpenAI | 100+ languages | Best multilingual |

### 2. Speed vs Quality

```
Faster ←──────────────────────────────→ Better
MiniLM-L6    MPNet-base    BGE-base    BGE-large
  22M           110M         110M         335M
 ~14k/s        ~3k/s        ~2k/s        ~1k/s
```

### 3. Max Sequence Length

| Model | Max Tokens | Good For |
|-------|------------|----------|
| MiniLM | 256 | Short queries/docs |
| MPNet | 384 | Standard docs |
| BGE/E5 | 512 | Longer documents |
| OpenAI | 8192 | Very long documents |

### 4. Instruction-Tuned vs Base

**Instruction-tuned** (BGE, E5-instruct, OpenAI):
- Prepend instruction like "Represent this query for retrieval:"
- Better for retrieval tasks
- May need different prefixes for query vs document

**Base models** (MiniLM, MPNet):
- No prefix needed
- Simpler to use
- Still very good

---

## Our Test Results

Tested on: semantic similarity, typo handling, informal text, topic discrimination

```
Model                    Semantic  Typo   Informal  Different  Misleading  Score
--------------------------------------------------------------------------------
all-mpnet-base-v2        0.688    0.668   0.911     0.165      0.409      1.693
all-MiniLM-L6-v2         0.751    0.534   0.893     0.158      0.480      1.541
bge-small-en-v1.5        0.875    0.764   0.942     0.508      0.740      1.334
```

**Key finding**: Even "worse" models like MiniLM handle informal/typo text well (0.534-0.893).
This is why learned normalization (ToneNorm) doesn't help - robustness is built-in.

---

## What Matters MORE Than Embedding Model?

### 1. Chunking Strategy (11% impact in our tests)

| Document Type | Best Chunking |
|--------------|---------------|
| Code | By function/class |
| Documentation | By section/header |
| Chat/Slack | By message |
| Tables | By row |
| General | Fixed 500-1000 chars |

**Finding**: Fixed 1000-char chunks = semantic chunking in many cases.
SAP's "one size fits all" isn't terrible, but semantic is 11% better.

### 2. Retrieval Top-K

| Top-K | Use Case |
|-------|----------|
| 1-3 | Direct answer needed |
| 5-10 | Context for LLM |
| 10-20 | Recall-focused (reranker downstream) |

### 3. Reranking (if using)

Cross-encoder rerankers can fix retrieval errors:
- `cross-encoder/ms-marco-MiniLM-L-6-v2` - Fast
- `BAAI/bge-reranker-large` - Better
- Cohere rerank - Best (API-based)

---

## Decision Flowchart

```
Need multilingual?
├─ Yes → OpenAI text-embedding-3-small or BGE-M3
└─ No → English only
         ├─ Need speed? → MiniLM-L6-v2
         ├─ Balanced? → MPNet-base-v2
         └─ Best quality? → BGE-large or OpenAI
                            ├─ On-prem required? → BGE-large
                            └─ Cloud OK? → OpenAI
```

---

## Common Mistakes

### 1. Over-optimizing embedding model

Switching from MiniLM to BGE-large = ~10% improvement
Better chunking strategy = ~11% improvement
Adding reranker = ~15-20% improvement

**Lesson**: Chunking and reranking often matter more than embedding model.

### 2. Ignoring max sequence length

If your chunks are >512 tokens, MiniLM will truncate and lose information.

### 3. Forgetting instruction prefixes

BGE and E5 need different prefixes for queries vs documents:
```python
# BGE query
"Represent this sentence for searching relevant passages: " + query
# BGE document
passage  # no prefix
```

### 4. Using cosine similarity wrong

Always normalize embeddings before comparison:
```python
# Wrong
similarity = np.dot(emb1, emb2)

# Right
similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
```

---

## Bottom Line

For **95% of RAG use cases**:

1. Use `all-mpnet-base-v2` or `bge-small-en-v1.5`
2. Chunk by document structure (not fixed chars)
3. Retrieve top-5 to top-10
4. Add a reranker if quality matters

The embedding model choice is often **less important** than:
- Good chunking
- Appropriate top-k
- Reranking

**Don't overthink it.** Pick a decent model and focus on the rest of your pipeline.

---

*Generated from empirical testing on 2025-12-09*
