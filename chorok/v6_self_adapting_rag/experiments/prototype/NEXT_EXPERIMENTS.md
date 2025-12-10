# Next Experiments: Validating MoE-RAG

**Core Question:** Under what conditions do multiple specialized attention heads outperform a single head?

---

## The Real Problem

Current setup fails because:
```
Simple passages + Template queries = Single attention pattern sufficient
```

We need:
```
Diverse passages + Diverse queries = Multiple attention patterns required
```

---

## Hypothesis Refinement

**Original:** Different query types need different attention patterns.

**Refined:** Different query types need different attention patterns **when passages contain information accessible through different retrieval strategies**.

---

## Experiment Designs

### Experiment 1: Multi-Strategy Passages

**Idea:** Create passages that reward different attention strategies.

```
Passage Type A (Keyword-heavy):
"Python list methods: append(), extend(), insert(), remove(), pop(),
clear(), index(), count(), sort(), reverse(), copy()"
→ Best retrieved by keyword matching

Passage Type B (Semantic-heavy):
"When you need to add items to a collection in Python, you have several
options depending on whether you want to add one item or many..."
→ Best retrieved by semantic similarity

Passage Type C (Structural):
"Step 1: Initialize empty list. Step 2: Use append() for single items.
Step 3: Use extend() for multiple items. Step 4: Verify with print()."
→ Best retrieved by structure matching
```

**Test:** Does head trained on keyword queries attend differently than head trained on semantic queries?

### Experiment 2: Answer Position Variation

**Idea:** Vary where the answer appears in passages.

```
Early-answer passage:
"Python is a programming language. [rest is context]"

Late-answer passage:
"[context about history, uses, comparisons] Python is a programming language."

Distributed-answer passage:
"Python [context] is [context] a programming [context] language."
```

**Test:** Do heads learn to attend to different positions?

### Experiment 3: Domain Mixing

**Idea:** Mix passages from domains with different retrieval needs.

| Domain | Retrieval Need |
|--------|---------------|
| Code documentation | Exact function names, parameters |
| Legal text | Specific clause numbers, definitions |
| Scientific papers | Concepts, relationships |
| How-to guides | Sequential steps |

**Test:** Does routing correctly identify domain and activate appropriate head?

### Experiment 4: Noise Injection

**Idea:** Add distractors that different strategies handle differently.

```
Clean passage: "The capital of France is Paris."

With semantic noise: "The capital of France is Paris. Cities are
important cultural centers. Many capitals have historical significance."

With keyword noise: "The capital of France is Paris. Capital gains tax
applies to investments. France is known for wine. Paris Hilton is famous."
```

**Test:** Do specialized heads better filter noise relevant to their query type?

---

## Minimum Viable Experiment

**Goal:** Prove heads CAN specialize (before scaling up).

**Setup:**
1. Create 50 "keyword-optimized" passages
2. Create 50 "semantic-optimized" passages
3. Generate queries that clearly need one strategy or the other
4. Train 2-head MoE
5. Verify heads develop different attention patterns

**Success Criteria:**
- Head diversity > 0.3 (currently 0.06)
- Correct head beats wrong head > 80% (currently 67%)
- MoE beats single attention (currently loses by 0.05)

---

## Data Generation Plan

### Option A: Synthetic (Fast, Controlled)

```python
def create_keyword_passage(topic):
    """Dense with searchable terms."""
    return f"{topic}: {', '.join(get_keywords(topic))}"

def create_semantic_passage(topic):
    """Descriptive, few keywords."""
    return f"When working with {topic}, one typically considers..."

def create_keyword_query(passage):
    """Direct term lookup."""
    return f"What is {extract_main_term(passage)}?"

def create_semantic_query(passage):
    """Conceptual question."""
    return f"How would you describe the purpose of {topic}?"
```

### Option B: Real Data (Slower, More Valid)

1. **Natural Questions** - Use document structure (tables vs text vs lists)
2. **MS MARCO** - Different query types naturally present
3. **HotpotQA** - Multi-hop vs single-hop queries

### Option C: Hybrid

1. Start with synthetic to prove concept
2. Validate on real data subset
3. Scale to full benchmark

---

## Implementation Priority

1. **[Now]** Implement Experiment 1 (Multi-Strategy Passages)
2. **[If works]** Run on Natural Questions subset
3. **[If works]** Full benchmark comparison
4. **[If fails]** Revisit hypothesis - maybe query types aren't the right axis

---

## Questions to Answer

1. **What makes attention patterns different?**
   - Position bias?
   - Term frequency weighting?
   - Semantic clustering?

2. **How much diversity is enough?**
   - Current: 0.06
   - Target: ???
   - Need ablation study

3. **When does soft routing help?**
   - Current: +0.01
   - When should it help more?
   - Ambiguous queries that need multiple perspectives

---

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Heads collapse despite diverse data | Stronger diversity loss, lower threshold |
| Router overfits to surface patterns | Regularization, dropout |
| No natural query-type clusters exist | Rethink axis of specialization |
| Computational overhead not worth it | Focus on interpretability value |

---

## Success Definition

**Minimum:** MoE beats single attention on controlled experiment
**Good:** MoE beats single attention on real benchmark
**Great:** Clear interpretable specialization (head X = factual, head Y = procedural)

---

*Decision needed: Which experiment to run first?*
