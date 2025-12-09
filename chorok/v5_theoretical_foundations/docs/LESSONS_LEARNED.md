# Lessons Learned: v5 Theoretical Foundations

**Created**: 2025-12-09
**Outcome**: Research direction abandoned after discovering no novel gap

---

## The Core Lesson

> **Different vocabulary ≠ Different problem**

We spent significant effort developing what we thought was a novel research direction, only to discover we were rediscovering well-established concepts with different terminology.

---

## Timeline of Discovery

### Stage 1: Initial Confidence
- Proposed "Hierarchical Bayesian Intervention Analysis"
- Claimed novelty in variance reduction bounds
- **Reality**: Textbook material (Gelman, Murphy)

### Stage 2: First Pivot
- Pivoted to "unknown unknowns detection"
- Claimed novelty in identifying what's missing
- **Reality**: Lakkaraju et al. AAAI 2017

### Stage 3: Second Pivot
- Pivoted to "shrinkage behavior under regime change"
- Claimed Dynamic Shrinkage handles smooth variation, not discrete shifts
- **Reality**: "Discrete vs gradual" distinction is arbitrary and post-facto

### Stage 4: Final Realization
- Recognized this is just **transfer learning** with Bayesian vocabulary
- Searched for "negative transfer Bayesian"
- **Found**: Finkel & Manning 2009, Power Priors, extensive literature

---

## The Vocabulary Mapping

| Our Term | Established Term | Field |
|----------|------------------|-------|
| Prior becomes stale | Source-target mismatch | Transfer Learning |
| Shrinkage hurts after shift | Negative transfer | Domain Adaptation |
| When to stop borrowing | Transfer parameter α | Power Prior Methods |
| Temporal neighborhoods | Domain adaptation | Bayesian Transfer Learning |
| Hierarchical borrowing | Multi-task learning | MTL Literature |

---

## What We Should Have Done

### 1. Broader Initial Search
Before committing to any direction, search across fields:
- Statistics terminology
- Machine learning terminology
- Bayesian terminology
- Domain adaptation terminology

### 2. Ask "What is this really?"
When proposing something "novel," ask:
- What is the fundamental problem being solved?
- Who else might care about this problem?
- What would they call it?

### 3. Check Wikipedia First
Many "gaps" are actually well-documented:
- Transfer learning: https://en.wikipedia.org/wiki/Transfer_learning
- Domain adaptation: https://en.wikipedia.org/wiki/Domain_adaptation
- Negative transfer: mentioned in transfer learning literature

### 4. Search for the Opposite
If claiming "X doesn't exist," search for:
- "X"
- Synonyms of X
- The problem X would solve
- Who would benefit from X

---

## Red Flags We Missed

1. **"This seems obvious"** - If it seems obvious, someone probably did it
2. **"Just different vocabulary"** - User correctly identified this early
3. **"Nobody studies shrinkage under shift"** - Too broad a claim
4. **"60 years of shrinkage theory but nobody thought of this?"** - Unlikely

---

## What Would Be Actually Novel

After this experience, genuine novelty requires:

### Structural Novelty
- New problem setting (not just new domain)
- New constraints (privacy, fairness, causality)
- New data structure (not hierarchical, graph, or time series)

### Algorithmic Novelty
- Provably better than existing methods
- New computational approach
- New theoretical guarantees

### Application Novelty (Lower Bar)
- First application to specific domain
- Engineering contribution
- Empirical findings in new setting

---

## Salvageable Outcomes

Despite no novel theory, we gained:

1. **Deep understanding** of hierarchical Bayesian methods
2. **Literature map** of shrinkage, transfer learning, negative transfer
3. **Working code** for shrinkage experiments
4. **Clear documentation** of what doesn't work

### Potential Application Paper
Could still write application paper:
- "Applying Power Priors to COVID Distribution Shift in SALT"
- Lower novelty, but valid contribution
- Venue: KDD Applied, Workshop papers

---

## Key References Discovered

### Transfer Learning / Negative Transfer
1. Finkel & Manning (2009) - Hierarchical Bayesian Domain Adaptation
2. Power Priors (Ibrahim & Chen, 2000)
3. arXiv:2502.19796 (2025) - Principled Bayesian Transfer Learning
4. arXiv:2105.01445 (2021) - Online Transfer Learning

### Shrinkage Fundamentals
5. Efron & Morris (1971, 1973, 1975) - Shrinkage theory
6. Kowal et al. (2019) - Dynamic Shrinkage Processes
7. Local EB (2024) - Neighborhood-based shrinkage
8. Individual Shrinkage (2023) - Individual accuracy

---

## Conclusion

**Failed research directions are still valuable** - they prevent wasted effort on papers that would be rejected for lack of novelty.

The lesson: **Literature search is not optional.** Do it thoroughly, across multiple fields, with multiple vocabularies, before committing to a direction.

---

*This document records lessons from the v5 research direction.*
*Outcome: Direction abandoned, but lessons learned.*
