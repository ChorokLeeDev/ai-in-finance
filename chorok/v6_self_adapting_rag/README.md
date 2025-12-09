# V6: Self-Adapting RAG

> **"Dump your data, get a working Q&A system. No configuration required."**

---

## Status

| Aspect | Status |
|--------|--------|
| Vision | Documented |
| Technical Approach | Sketched |
| Literature Validation | TODO |
| Prototype | Not started |
| Feasibility | Unknown (needs validation) |

---

## The Problem

Every SAP team builds their own RAG. Even with shared infrastructure, each team spends weeks on:
- Prompt engineering
- Context curation
- RAG tuning
- Validation setup

**This doesn't scale.**

---

## The Vision

A self-adapting RAG system that:
1. Takes raw data dump (no structure required)
2. Trains itself to understand the domain
3. Answers questions with source attribution
4. No per-team engineering needed

---

## Documents

| Document | Purpose |
|----------|---------|
| [VISION.md](VISION.md) | Full vision and motivation |
| [TECHNICAL_SKETCH.md](TECHNICAL_SKETCH.md) | Technical architecture and approach |
| [VALIDATION_PLAN.md](VALIDATION_PLAN.md) | How to validate this is achievable |
| [LITERATURE_TODO.md](LITERATURE_TODO.md) | Literature search to confirm gap |

---

## Key Technical Ideas

1. **Self-supervised attention**: Learn what to focus on without labels
2. **Synthetic query generation**: Create training data from documents
3. **Automatic structure discovery**: Find patterns in raw data
4. **Confidence calibration**: Know when to say "I don't know"

---

## Next Steps

1. [ ] Complete literature search (confirm gap)
2. [ ] Quick validation tests (embeddings, clustering)
3. [ ] Minimal prototype on SQL migration data
4. [ ] Decide go/no-go based on results

---

## Connection to Previous Work

| Previous | Lesson | Applied in V6 |
|----------|--------|---------------|
| V3 FK Attribution | Structure matters | Learn structure automatically |
| V5 Shrinkage | Don't claim existing theory | Focus on application, not theory |
| Workshop Paper | Hand-crafted works but doesn't scale | Use hand-crafted as training signal |
| Quant Research | Apply methods to new domains | Self-supervised learning → RAG |

---

## Target

- **Venue**: NeurIPS 2026 (or KDD if more applied)
- **Timeline**: ~4 months to submission-ready
- **Success metric**: Match 80% of hand-crafted quality with zero configuration

---

## The Bet

**If this works**: Every team can have working RAG in hours, not weeks.

**If this fails**: We learn what human input is truly irreplaceable.

---

*Created: 2025-12-09*
*Author: Chorok Lee*
