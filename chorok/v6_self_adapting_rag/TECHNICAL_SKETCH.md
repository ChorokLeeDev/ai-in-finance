# V6: Technical Sketch

**Created**: 2025-12-09
**Status**: Initial Design
**Purpose**: Technical approach for self-adapting RAG

---

## Problem Formalization

### Current RAG

```
Input: Query q, Retrieved documents D = {d_1, d_2, ..., d_k}
Output: Answer a = LLM(q, concat(D))

Problems:
- All documents treated equally
- Noise in D hurts performance
- Conflicts in D confuse LLM
- Requires manual tuning of retrieval
```

### Self-Adapting RAG

```
Input: Query q, Raw corpus C (unprocessed)
Output: Answer a, Sources S, Confidence c

Key difference:
- No manual retrieval tuning
- Learned attention over corpus
- Self-calibrated confidence
- Source attribution
```

---

## Architecture

### Overview

```
┌─────────────┐     ┌─────────────────┐     ┌──────────────┐
│  Raw Corpus │ ──▶ │  Self-Training  │ ──▶ │ Adapted RAG  │
│      C      │     │    Pipeline     │     │    System    │
└─────────────┘     └─────────────────┘     └──────────────┘
                           │
                    ┌──────┴──────┐
                    │             │
              ┌─────▼─────┐ ┌─────▼─────┐
              │ Structure │ │ Attention │
              │ Discovery │ │  Learner  │
              └───────────┘ └───────────┘
```

### Component 1: Structure Discovery

**Goal**: Automatically discover latent structure in raw corpus

```python
class StructureDiscovery:
    def __init__(self, corpus: List[Document]):
        self.corpus = corpus

    def discover(self) -> CorpusStructure:
        # 1. Document clustering (topic modeling)
        topics = self.cluster_by_topic()

        # 2. Entity extraction (files, functions, configs)
        entities = self.extract_entities()

        # 3. Relationship discovery (doc-doc, entity-doc)
        relationships = self.discover_relationships()

        # 4. Build knowledge graph
        graph = self.build_graph(topics, entities, relationships)

        return CorpusStructure(topics, entities, graph)
```

**Techniques**:
- Topic modeling: LDA, BERTopic, or clustering on embeddings
- Entity extraction: NER + rule-based for code entities
- Relationship discovery: Co-occurrence, hyperlinks, imports

### Component 2: Synthetic Query Generation

**Goal**: Create training data without human labels

```python
class SyntheticQueryGenerator:
    def __init__(self, corpus: List[Document], structure: CorpusStructure):
        self.corpus = corpus
        self.structure = structure

    def generate_queries(self, n: int) -> List[TrainingPair]:
        pairs = []
        for _ in range(n):
            # Strategy 1: Document → Question
            doc = random.choice(self.corpus)
            query = self.doc_to_question(doc)
            relevant = [doc]

            # Strategy 2: Entity → Question
            entity = random.choice(self.structure.entities)
            query = self.entity_to_question(entity)
            relevant = self.get_entity_docs(entity)

            # Strategy 3: Relationship → Question
            rel = random.choice(self.structure.relationships)
            query = self.relationship_to_question(rel)
            relevant = self.get_relationship_docs(rel)

            pairs.append(TrainingPair(query, relevant))

        return pairs
```

**Query generation strategies**:
- Extract key sentences, mask entities, form questions
- Use LLM to generate questions about document content
- Template-based: "How does X work?", "What is Y?", "How to do Z?"

### Component 3: Attention Learner

**Goal**: Learn query → document relevance without manual labels

```python
class AttentionLearner:
    def __init__(self, corpus: List[Document]):
        self.corpus = corpus
        self.encoder = DocumentEncoder()  # Embedding model
        self.attention = AttentionNetwork()  # Learned attention

    def train(self, synthetic_pairs: List[TrainingPair]):
        for query, relevant_docs in synthetic_pairs:
            # Encode query and all documents
            q_emb = self.encoder(query)
            d_embs = [self.encoder(d) for d in self.corpus]

            # Compute attention weights
            attention_weights = self.attention(q_emb, d_embs)

            # Loss: high attention on relevant, low on irrelevant
            loss = self.contrastive_loss(
                attention_weights,
                relevant_docs,
                self.corpus
            )

            loss.backward()
            self.optimizer.step()

    def get_attention(self, query: str) -> Dict[Document, float]:
        q_emb = self.encoder(query)
        d_embs = [self.encoder(d) for d in self.corpus]
        weights = self.attention(q_emb, d_embs)
        return {d: w for d, w in zip(self.corpus, weights)}
```

**Training objective**:
- Contrastive loss: Relevant docs should have high attention
- Could use InfoNCE, triplet loss, or margin-based loss

### Component 4: Confidence Estimator

**Goal**: Know when the system is uncertain

```python
class ConfidenceEstimator:
    def __init__(self, attention_learner: AttentionLearner):
        self.attention = attention_learner

    def estimate_confidence(self, query: str) -> float:
        weights = self.attention.get_attention(query)

        # Signal 1: Attention entropy (spread = uncertain)
        entropy = self.compute_entropy(weights.values())

        # Signal 2: Top attention weight (low = uncertain)
        max_weight = max(weights.values())

        # Signal 3: Answer consistency (if multiple generations differ)
        consistency = self.check_consistency(query)

        # Combine signals
        confidence = self.calibrate(entropy, max_weight, consistency)

        return confidence
```

**Confidence signals**:
- Low attention entropy → focused → confident
- High max attention weight → clear winner → confident
- Answer consistency across samples → confident

### Component 5: Generation with Attribution

**Goal**: Generate answer with source citations

```python
class AttributedGenerator:
    def __init__(self, attention_learner: AttentionLearner, llm: LLM):
        self.attention = attention_learner
        self.llm = llm

    def generate(self, query: str) -> Response:
        # Get attention weights
        weights = self.attention.get_attention(query)

        # Select top-k by attention (not similarity)
        top_docs = sorted(weights.items(), key=lambda x: -x[1])[:k]

        # Generate with weighted context
        context = self.format_context(top_docs)
        answer = self.llm(query, context)

        # Extract source attributions from attention
        sources = [(doc.path, doc.line, weight)
                   for doc, weight in top_docs if weight > threshold]

        # Estimate confidence
        confidence = self.confidence.estimate(query)

        return Response(answer, sources, confidence)
```

---

## Training Pipeline

### End-to-End Flow

```
Raw Corpus
    │
    ▼
┌───────────────────────────┐
│ 1. Structure Discovery    │
│    - Cluster documents    │
│    - Extract entities     │
│    - Build graph          │
└───────────────────────────┘
    │
    ▼
┌───────────────────────────┐
│ 2. Synthetic Query Gen    │
│    - Doc → Question       │
│    - Entity → Question    │
│    - 1000s of pairs       │
└───────────────────────────┘
    │
    ▼
┌───────────────────────────┐
│ 3. Attention Training     │
│    - Contrastive learning │
│    - Query → Doc weights  │
│    - Self-supervised      │
└───────────────────────────┘
    │
    ▼
┌───────────────────────────┐
│ 4. Confidence Calibration │
│    - Held-out validation  │
│    - Calibrate estimates  │
└───────────────────────────┘
    │
    ▼
Ready for Queries
```

### Training Data Requirements

| Component | Data Needed | Source |
|-----------|-------------|--------|
| Structure Discovery | Raw corpus | Team dump |
| Synthetic Queries | None (generated) | Automatic |
| Attention Training | Synthetic pairs | Generated |
| Calibration | Held-out pairs | 10% of synthetic |

**Key: No human labels required.**

---

## Comparison: Old vs New

### Old Approach (Your 5 Months)

```
Human Expert
    │
    ▼
Hand-craft JSON schemas
    │
    ▼
Define MECE dimensions
    │
    ▼
Enumerate edge cases
    │
    ▼
Write validation rules
    │
    ▼
Test and iterate
    │
    ▼
Working system (90.3% accuracy)

Time: 5 months
Maintenance: Ongoing
Scalability: Doesn't scale
```

### New Approach (V6)

```
Team Data Dump
    │
    ▼
Self-training pipeline (automatic)
    │
    ▼
Working system (target: 80-90%)

Time: Hours to days
Maintenance: Minimal (retrain on new data)
Scalability: Any team can use
```

---

## Experiments Design

### Experiment 1: Proof of Concept

**Question**: Can self-trained attention match hand-crafted structure?

```
Dataset: SQL migration (your existing work)

Setup:
- Baseline A: Your hand-crafted JSON (90.3% accuracy)
- Baseline B: Raw dump + standard RAG (estimate: 40-50%)
- Test: Raw dump + self-trained attention

Metric: Test coverage (% functions reaching 100% validation)

Success: Test > 75% (at least 75% of hand-crafted quality)
```

### Experiment 2: Ablation Study

**Question**: Which components matter most?

```
Ablations:
- No structure discovery (random clustering)
- No synthetic queries (use document titles as queries)
- No attention learning (use embedding similarity)
- No confidence estimation (always confident)

Measure: Impact on accuracy from each ablation
```

### Experiment 3: Cross-Domain Transfer

**Question**: Does the approach generalize across teams?

```
Teams:
- Team A: SQL migration (your team)
- Team B: API documentation (another SAP team)
- Team C: Config management (another SAP team)

Setup:
- Train self-adapting system on each team's raw data
- Measure accuracy on held-out queries per team

Success: Works on all teams without modification
```

### Experiment 4: Setup Time Comparison

**Question**: How much engineering time is saved?

```
Compare:
- Traditional RAG: Time to configure, tune, validate
- Self-adapting: Time to dump data and train

Metric: Hours/days to working system
```

---

## Technical Challenges

### Challenge 1: Synthetic Query Quality

**Problem**: Generated queries might not match real user queries

**Mitigation**:
- Multiple generation strategies (doc-based, entity-based, template)
- Validate on small set of real queries if available
- Iterative refinement of generation

### Challenge 2: Cold Start

**Problem**: Small corpus might not have enough signal

**Mitigation**:
- Transfer learning from general domain
- Minimum corpus size requirements
- Bootstrap with LLM-generated augmentation

### Challenge 3: Specialized Jargon

**Problem**: Team-specific terminology might not be in embeddings

**Mitigation**:
- Fine-tune embeddings on team corpus
- Entity-aware encoding (treat jargon as entities)
- Acronym/abbreviation expansion

### Challenge 4: Evaluation Without Ground Truth

**Problem**: No labeled data to measure accuracy

**Mitigation**:
- Use synthetic queries as proxy
- User feedback loop (optional)
- Compare to hand-tuned baseline where available

---

## Implementation Plan

### Phase 1: Minimal Prototype (Week 1-2)

```python
# Simplest possible version

def prototype():
    # 1. Load raw corpus
    corpus = load_documents("team_data/")

    # 2. Simple clustering (BERTopic)
    topics = cluster_documents(corpus)

    # 3. Generate synthetic queries (LLM-based)
    queries = generate_questions(corpus, n=1000)

    # 4. Train simple attention (linear layer over embeddings)
    attention = train_attention(queries, corpus)

    # 5. Evaluate on held-out
    accuracy = evaluate(attention, held_out_queries)

    return accuracy
```

### Phase 2: Full System (Week 3-6)

- Implement all components properly
- Add confidence estimation
- Add source attribution
- Optimize performance

### Phase 3: Evaluation (Week 7-10)

- Run all experiments
- Analyze results
- Identify failure modes

### Phase 4: Paper (Week 11-16)

- Write up methodology
- Document results
- Position against related work

---

## Related Work to Study

### Self-Supervised Learning for Retrieval

- Contriever (Izacard et al., 2022) - Unsupervised dense retrieval
- LaPraDoR (Xu et al., 2022) - Unsupervised passage retrieval
- SimCSE (Gao et al., 2021) - Contrastive sentence embeddings

### RAG Improvements

- Self-RAG (Asai et al., 2024) - Self-reflective retrieval
- REPLUG (Shi et al., 2023) - Black-box LLM retrieval
- RankRAG (Yu et al., 2024) - Context ranking

### Attention Mechanisms

- AttentionRAG (2025) - Attention-guided pruning
- Selective Attention (Zhang et al., NeurIPS 2024)

### Knowledge Distillation

- Document structure learning
- Query generation from documents

---

## Open Questions for Research

1. **What's the minimum corpus size for self-training?**
   - Hypothesis: 100+ documents for meaningful clusters

2. **How domain-specific should the attention be?**
   - Fully general vs. per-domain training

3. **Can we use user feedback to improve?**
   - Online learning from query-click patterns

4. **How to handle dynamic/updating corpora?**
   - Incremental training vs. full retrain

5. **What's the right granularity?**
   - Document-level vs. chunk-level vs. sentence-level

---

*This document outlines the technical approach for V6.*
*Next step: Implement minimal prototype and validate core hypothesis.*
