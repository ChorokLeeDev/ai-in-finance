# Why Quant Research Found Novelty While Chorok Struggles

**Created**: 2025-12-09
**Purpose**: Understand what works vs what doesn't in finding research novelty

---

## The Quant Research That Works

### Factor Crowding (KDD 2026)

**Claim**: "Not all factors crowd equally - mechanical factors decay faster than judgment factors"

```
Model: α(t) = K/(1 + λt)  (hyperbolic decay)

Findings:
- Momentum (mechanical): R² = 0.65, fits well
- Value (judgment): R² = poor, doesn't fit
- Taxonomy: Mechanical vs Judgment explains difference
```

**Why it works:**
1. **Doesn't claim new theory** - game-theoretic models exist
2. **Applies existing theory** to specific problem (factor crowding)
3. **Empirical discovery** - the taxonomy is new observation
4. **Narrow gap** - nobody categorized factors by "signal clarity"
5. **Strong data** - 60 years of Fama-French data

### Conformal Prediction for Finance (ICML 2026)

**Claim**: "Apply conformal prediction to factor crowding detection"

**Why it works:**
1. **Method exists** (conformal prediction)
2. **Domain is new** (factor crowding)
3. **Clear contribution** - distribution-free UQ for finance
4. **No bold theoretical claims**

---

## The Chorok Research That Fails

### V5: Shrinkage Under Regime Change

**Attempted claim**: "Shrinkage behavior changes under temporal distribution shift"

**Why it failed**:
- This is **Bayesian transfer learning** + **negative transfer**
- Fully covered by existing literature (Finkel 2009, power priors)

### V3: FK = Causal DAG

**Attempted claim**: "FK structure encodes causal DAG in transactional domains"

**Why it failed**:
- **Relational Causal Discovery** literature exists precisely because FK ≠ Causal
- RPC, RCD, RelFCI algorithms wouldn't exist if FK gave causality for free
- Database schema is for storage, not causality

---

## The Key Difference

| Aspect | Quant (Works) | Chorok (Fails) |
|--------|---------------|----------------|
| **Claim boldness** | Medium | **Too bold** |
| **Approach** | Apply existing methods to new domain | Claim new theory |
| **Gap size** | Narrow, specific | Broad, turns out covered |
| **Verification** | Literature check confirms gap | Literature check reveals coverage |
| **Contribution type** | Empirical discovery | Attempted theoretical claim |

---

## What Makes Quant Research Work

### 1. Domain-Specific Application

```
Quant approach:
  Existing method + New domain = Novel contribution

Examples:
  - Game theory + Factor crowding = Alpha decay model
  - Conformal prediction + Crowding detection = Calibrated UQ
  - LSTM + Global factors = Cross-region prediction
```

### 2. Empirical-First Discovery

```
Quant workflow:
  1. Get data (60 years of factor returns)
  2. Observe patterns (momentum decays, value doesn't)
  3. Propose explanation (mechanical vs judgment taxonomy)
  4. Validate empirically (R², out-of-sample)

NOT:
  1. Claim bold theory
  2. Search literature
  3. Find it's covered
  4. Abandon
```

### 3. Narrow, Defensible Gaps

```
Quant gap: "Nobody has done game-theoretic modeling of factor alpha decay"
  - Narrow enough to actually check
  - If someone had, they'd have the model
  - Easy to verify novelty

Chorok gap: "Nobody studies shrinkage under temporal shift"
  - Too broad
  - Different vocabulary (transfer learning, negative transfer)
  - Hard to verify, easy to miss existing work
```

### 4. No Overclaiming

```
Quant claim: "We observe that mechanical factors crowd faster"
  - Empirical observation
  - Doesn't claim to explain WHY deeply
  - Proposes taxonomy as explanation

Chorok claim: "FK structure IS the causal DAG"
  - Theoretical claim
  - If true, would upend a research field
  - Too bold, turns out wrong
```

---

## What Chorok Should Do Differently

### Option A: Follow Quant Approach

```
Instead of: "FK = Causal DAG" (theoretical claim)

Do: "FK groups show high Attribution-Error correlation in transactional
     domains but not in social domains" (empirical observation)

Then: Propose taxonomy (transactional vs social) as explanation
      Without claiming WHY at a deep theoretical level
```

### Option B: Apply Existing Methods to New Domain

```
Instead of: Invent new theory for relational UQ

Do: Apply conformal prediction to relational data
    Apply power priors to COVID shift in SALT
    Apply game-theoretic crowding models to something else
```

### Option C: Find Narrower Gap

```
Instead of: "Shrinkage under distribution shift" (broad, covered)

Find: Specific narrow gap that nobody has addressed
      - Specific method X on specific domain Y
      - "Nobody has done conformal prediction for relational autocomplete"
      - Easy to verify, hard to have been done before
```

---

## The Meta-Lesson

**The quant research asks: "What can we apply existing methods to?"**
**The chorok research asks: "What new theory can we prove?"**

Proving new theory is hard:
- If it's obvious, someone did it
- If nobody did it, there's often a reason (it's wrong)
- The more mature the field, the harder to find gaps

Applying methods is easier:
- New domains constantly emerge
- Existing methods can be adapted
- Contribution is clear: "first to apply X to Y"

---

## Concrete Recommendation for Chorok

### Don't:
- Claim FK = Causal DAG
- Claim theoretical novelty for shrinkage/transfer learning
- Try to prove new bounds or theorems

### Do:
- Accept V3 as empirical/applied contribution
- Frame as: "First application of grouped uncertainty attribution to relational databases"
- Propose domain taxonomy (transactional vs social) as empirical finding
- Target KDD Applied Track or VLDB

### Or:
- Start fresh with quant-style approach
- Pick existing method (conformal prediction, etc.)
- Apply to new finance/relational domain
- Don't claim new theory

---

## Summary

| Research Style | Quant | Chorok (Failed) | Chorok (Should Be) |
|----------------|-------|-----------------|-------------------|
| Theory | Use existing | Claim new | Use existing |
| Domain | Narrow (finance) | Broad (relational) | Narrow |
| Gap | Specific application | Broad theory | Specific application |
| Verification | Easy | Hard | Easy |
| Risk | Low | High | Low |

**The quant research works because it's humble about theory but ambitious about application.**

**The chorok research fails because it's ambitious about theory in a mature field.**

---

*Key insight: Stop trying to prove new theory. Start applying existing methods to new problems.*
