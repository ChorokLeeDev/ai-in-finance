# Why "FK = Causal DAG" Is NOT a Valid Claim

**Created**: 2025-12-09
**Purpose**: Document why the bold theoretical claim we considered is wrong

---

## The Claim We Considered

> "In transactional domains, FK structure encodes the causal DAG."

**This claim is wrong.** Here's why:

---

## Reason 1: Database Schema ≠ Causal Model

The literature explicitly states:

> "Causal relationships cannot be explicitly modeled in current database systems, which offer no specific support for such queries."
> — [UMass Causal Relationships in Databases](https://people.cs.umass.edu/~ameli/projects/causality/)

**Schema tells you:**
- What tables exist
- How they're connected (FK relationships)
- Data constraints (referential integrity)

**Schema does NOT tell you:**
- Which variables causally affect which
- The direction of causal effects
- What happens under intervention

---

## Reason 2: Referential Integrity ≠ Causality

| FK Referential Integrity | Causal Relationship |
|--------------------------|---------------------|
| "Order references Customer" | "Customer causes Order's outcome" |
| Structural constraint | Effect of intervention |
| Ensures valid references | Explains WHY outcomes occur |
| Built into RDBMS | Requires inference/discovery |

**FK says:** "This row points to that row"
**Causality asks:** "Does changing X affect Y?"

These are fundamentally different questions.

---

## Reason 3: The Existence of Relational Causal Discovery Proves FK ≠ Causal

If FK structure encoded causality, why would these exist?

| Algorithm | Paper | Purpose |
|-----------|-------|---------|
| RPC | [Maier et al., AAAI 2010](https://ojs.aaai.org/index.php/AAAI/article/view/7695) | Learn causal structure from relational data |
| RCD | [Maier et al., UAI 2013](https://dl.acm.org/doi/10.5555/3023638.3023676) | Sound and complete relational causal discovery |
| RelFCI | [Negro et al., 2025](https://arxiv.org/abs/2507.01700) | Handle latent confounders in relational data |
| CARL | [Salimi et al., SIGMOD 2020](https://dl.acm.org/doi/10.1145/3318464.3389759) | Causal inference on relational data |

**The entire research field of Relational Causal Discovery exists because FK ≠ Causal.**

If FK gave you the causal DAG for free, you wouldn't need algorithms to discover it.

---

## Reason 4: FK Direction Can Be Opposite to Causal Direction

```
Example: E-commerce database

FK structure:
  Order.customer_id → Customer.id
  Order.product_id → Product.id

FK direction: Order → Customer (order references customer)

Causal question: Does Customer cause Order outcomes?
  - Customer's purchase history → Order's delivery time? (maybe)
  - Customer's location → Order's shipping cost? (yes)
  - But FK points Order → Customer, not Customer → Order

The FK arrow is about DATA ORGANIZATION, not causal flow.
```

---

## Reason 5: Propositionalization Causes Problems

From [Maier et al.](https://groups.cs.umass.edu/jensen/wp-content/uploads/sites/17/2022/03//maier-et-al-aaai2010.pdf):

> "Propositionalization is largely inadequate for effective causal learning in relational domains."
>
> "Methods for propositionalizing relational data frequently create implicit conditioning that can produce cases of Berkson's paradox."

When you join tables along FK relationships (which is what FK-based attribution implicitly does), you can introduce **spurious associations** that don't reflect causal relationships.

---

## What Our Empirical Finding Actually Shows

We found ρ ≈ 0.9 Attribution-Error correlation in SALT and Trial.

**This does NOT show:**
- FK structure = Causal DAG
- FK attribution = Causal attribution

**This MIGHT show:**
- In transactional domains, FK groups capture "decision units"
- The model's reliance on FK features is genuine (not spurious)
- Permuting FK groups has predictable effects on predictions

But this is an **empirical observation**, not a causal claim.

---

## What Is the Actual Contribution Then?

### NOT Theoretical
We cannot claim:
- "FK = Causal DAG"
- Any formal connection to do-calculus
- Theoretical guarantees

### Empirical/Applied
We CAN claim:
- FK structure provides **semantically meaningful** grouping
- Attribution-Error validation is a **novel validation method**
- **Domain-dependent** effectiveness (transactional vs social)
- **Practical actionability** (entity drill-down)

---

## Revised Position

| Original Claim | Revised Claim |
|----------------|---------------|
| "FK encodes causal structure" | "FK provides meaningful grouping" |
| "FK attribution = Causal attribution" | "FK attribution is validated by error correlation" |
| "Theoretical contribution" | "Empirical/methodological contribution" |

---

## Implications for Publication

### Venue Fit

| Venue | Fit | Rationale |
|-------|-----|-----------|
| NeurIPS Theory | **Weak** | No theoretical contribution |
| NeurIPS Applications | Medium | Novel application |
| ICML | Medium | Empirical ML |
| KDD | **Strong** | Applied, business-relevant |
| VLDB | **Strong** | Relational database focus |

### Positioning

**Don't claim:**
- Theoretical novelty
- Causal interpretation
- Formal guarantees

**Do claim:**
- Novel validation method (Attribution-Error)
- Practical framework (3-level drill-down)
- Domain-dependent findings
- Actionability for practitioners

---

## Key References That Contradict "FK = Causal"

1. **Maier et al. (AAAI 2010)**: "Learning Causal Models of Relational Domains"
   - Proves you need to LEARN causal structure, not read it from schema

2. **RCD (UAI 2013)**: Sound and complete algorithm
   - If FK = Causal, why do we need a discovery algorithm?

3. **UMass Causality Project**: "Causal relationships cannot be explicitly modeled in current database systems"

4. **DAPER Model**: Extends ER with probabilistic/causal semantics
   - Shows ER/FK alone is insufficient for causal reasoning

---

## Lessons Learned

1. **"Obvious" claims that haven't been made usually have a reason**
2. **Always check why a claim hasn't been made, not just whether it exists**
3. **Relational Causal Discovery is a mature field** - they've thought about this
4. **Schema is for storage, not causality**

---

*This document explains why the "FK = Causal DAG" claim is invalid.*
*The bold theoretical claim we considered cannot be made.*
