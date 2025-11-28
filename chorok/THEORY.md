# Theoretical Foundation: FK-Causal-UQ

**Phase 2: Formal Theory for Interventional Uncertainty Attribution**

---

## 1. Notation and Setup

### 1.1 Relational Schema

A relational schema R = (T, FK) consists of:
- **Tables**: T = {T₁, T₂, ..., Tₙ}
- **Foreign Keys**: FK = {fk₁, fk₂, ..., fkₘ}

Each table Tᵢ has:
- Primary key: pkey(Tᵢ)
- Attributes: Attr(Tᵢ) = {A₁, A₂, ..., Aₖ}

Each foreign key fkⱼ: Tₛ → Tₜ links:
- Source table Tₛ (child)
- Target table Tₜ (parent)

### 1.2 Example: SALT Dataset

```
Tables:
  - SalesItem (prediction target)
  - SalesDocument
  - Product
  - Customer
  - BusinessPartner

Foreign Keys:
  - SalesItem → SalesDocument (SALESDOCUMENT)
  - SalesItem → Product (PRODUCT)
  - SalesDocument → Customer (SOLDTOPARTY)
  - Customer → BusinessPartner (PAYERPARTY)
```

---

## 2. FK → Causal DAG Mapping

### 2.1 Definition: FK-Induced DAG

**Definition 1** (FK-Induced DAG). Given schema R = (T, FK), the FK-induced DAG G_FK = (V, E) is:
- **Vertices**: V = T (each table is a node)
- **Edges**: E = {(Tₜ → Tₛ) | fk: Tₛ → Tₜ ∈ FK}

Note: Edge direction is **reversed** from FK direction.
- FK from child → parent
- Causal edge from parent → child

### 2.2 Causal Interpretation

**Proposition 1** (FK as Causal Prior). In G_FK:
- Parent table Tₜ causally affects child table Tₛ
- Attributes of Tₜ are potential causes of attributes in Tₛ

**Justification**:
- FK constraints encode entity relationships
- Parent entities (e.g., Customer) exist before child entities (e.g., Order)
- Parent attributes can influence child outcomes

### 2.3 Example: SALT Causal DAG

```
BusinessPartner → Customer → SalesDocument → SalesItem ← Product
                                    ↑
                              (other FKs)
```

Prediction task: Predict attribute Y in SalesItem
Causal parents: All tables reachable via FK paths

---

## 3. Observational vs Interventional Attribution

### 3.1 Observational Attribution (Current LOO)

**Definition 2** (Leave-One-Out Attribution). For FK fk with features F_fk:

```
Attr_LOO(fk) = U(M_{-fk}, X) - U(M_full, X)
```

Where:
- M_full: Model trained on all features
- M_{-fk}: Model trained without F_fk
- U(·): Uncertainty measure (entropy, variance)

**Problem**: This is **observational** — we condition on absence, not intervene.

### 3.2 Interventional Attribution (Target)

**Definition 3** (Interventional UQ Attribution). For FK fk:

```
Attr_do(fk) = E[U(Y) | do(X_fk = x_fk)] - E[U(Y)]
```

Where do(·) is Pearl's intervention operator.

**Key Difference**:
- LOO: What happens when we **don't observe** FK features?
- Interventional: What happens when we **intervene** on FK values?

### 3.3 When Are They Different?

**Proposition 2** (LOO ≠ Interventional). Attr_LOO(fk) ≠ Attr_do(fk) when:
1. FK features are correlated with other features
2. There exist confounders between FK tables
3. The FK relationship has collider structure

**Example**:
- Product → SalesItem ← Customer
- Intervening on Product attributes ≠ removing Product features
- Removing features breaks correlations, intervention preserves causal structure

---

## 4. Identification Theorem

### 4.1 Assumptions

**Assumption 1** (Causal Sufficiency within FK Path).
For any FK path fk₁ → fk₂ → ... → fkₙ, there are no unmeasured confounders between adjacent tables.

**Assumption 2** (FK Direction = Causal Direction).
The FK constraint direction reflects the temporal/causal ordering of entities.

**Assumption 3** (Acyclicity).
The FK-induced DAG G_FK is acyclic.

### 4.2 Identification Result

**Theorem 1** (Identification of FK-Causal-UQ). Under Assumptions 1-3, the interventional uncertainty attribution Attr_do(fk) is identifiable from observational data via:

```
Attr_do(fk) = Σ_z P(Z=z) · [U(Y | X_fk, Z=z) - U(Y | Z=z)]
```

Where Z = Pa(fk) ∪ NonDesc(fk) is the adjustment set (parents and non-descendants of fk in G_FK).

**Proof Sketch**:
1. By Assumption 3, G_FK is a valid causal DAG
2. By backdoor criterion, conditioning on Z blocks all backdoor paths
3. By Assumption 1, no unmeasured confounders, so adjustment is valid
4. The interventional distribution is identified by backdoor adjustment

### 4.3 When Identification Fails

**Proposition 3** (Non-Identification). Attr_do(fk) is NOT identifiable when:
1. Unmeasured confounders exist between FK-linked tables
2. FK direction is reversed (child → parent in reality)
3. Cyclic dependencies exist (feedback loops)

---

## 5. Algorithm: FK-Causal-UQ

### 5.1 Input/Output

**Input**:
- Relational schema R = (T, FK)
- Trained model M
- Test instance x
- Uncertainty measure U

**Output**:
- Attribution scores {Attr_do(fk) : fk ∈ FK}

### 5.2 Algorithm

```
Algorithm: FK-Causal-UQ

1. Construct G_FK from schema R
2. Verify G_FK is acyclic (else warn)
3. For each fk ∈ FK:
   a. Compute adjustment set Z_fk = Pa(fk) ∪ NonDesc(fk)
   b. Estimate Attr_do(fk) via backdoor adjustment:
      - Sample z values from P(Z_fk)
      - For each z: compute U(Y | X_fk, Z=z) - U(Y | Z=z)
      - Average over z samples
4. Return {Attr_do(fk) : fk ∈ FK}
```

### 5.3 Computational Complexity

- **LOO**: O(|FK| · C_train) — requires retraining per FK
- **FK-Causal-UQ**: O(|FK| · N · C_predict) — no retraining!

Where:
- C_train: Cost of training model
- C_predict: Cost of prediction
- N: Number of adjustment samples

**Key Advantage**: Our method avoids retraining, using causal adjustment instead.

---

## 6. Comparison: LOO vs FK-Causal-UQ

| Aspect | LOO (Baseline) | FK-Causal-UQ (Ours) |
|--------|----------------|---------------------|
| Type | Observational | Interventional |
| Question | "What if we don't have FK?" | "What if we intervene on FK?" |
| Retraining | Required (O(|FK|) times) | Not required |
| Confounders | Ignores | Adjusts for |
| Guarantees | None | Identification theorem |
| Validity | Correlational | Causal |

---

## 7. Validation Strategy

### 7.1 Synthetic Data (Ground Truth)

Generate data where:
1. True causal graph is known
2. True interventional effects computable
3. Compare: LOO vs FK-Causal-UQ vs Ground Truth

**Metrics**:
- Rank correlation with ground truth
- MSE of attribution scores

### 7.2 Semi-Synthetic (Injected Shifts)

Use real FK structure but inject:
1. Known distribution shifts in specific FKs
2. Measure which method detects the injected shift

### 7.3 Real Data (Qualitative)

COVID-19 natural experiment:
- Known external shock
- Method should attribute uncertainty to plausibly affected FKs
- Compare with domain knowledge

---

## 8. Open Questions

1. **Partial Identification**: What bounds can we derive when Assumption 1 fails?
2. **Sensitivity Analysis**: How sensitive is Attr_do to unmeasured confounding?
3. **Multi-hop FKs**: How to handle indirect FK relationships (A → B → C)?
4. **Conformal Wrapper**: Can we add coverage guarantees to FK-Causal-UQ?

---

## 9. Summary: Core Contribution

**Theorem 1** (restated): Under FK-induced DAG structure, interventional uncertainty attribution is identifiable without retraining the model.

This provides:
1. **Theoretical novelty**: First identification result for relational UQ
2. **Practical benefit**: No retraining required (LOO requires |FK| retrainings)
3. **Causal validity**: Answers interventional question, not just correlational

---

**Next Steps**:
1. [ ] Formalize proof of Theorem 1
2. [ ] Implement FK-Causal-UQ algorithm
3. [ ] Generate synthetic data with known ground truth
4. [ ] Compare LOO vs FK-Causal-UQ empirically

---

**Last Updated**: 2025-11-29
