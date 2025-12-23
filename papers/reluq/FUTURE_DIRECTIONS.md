# Future Research Directions for RelUQ

**Status**: Open research ideas that could significantly strengthen novelty for top venues

---

## 1. Schema-Aware Uncertainty Decomposition

**Core Idea**: Theoretically decompose epistemic vs aleatoric uncertainty through FK paths, not just aggregate importance.

### Background
Current approach aggregates feature importance by FK origin. This tells us *which table* contributes most but not *why*. A principled decomposition would separate:
- **Aleatoric uncertainty**: Irreducible noise inherent in the data (e.g., measurement error, stochastic outcomes)
- **Epistemic uncertainty**: Reducible uncertainty from limited data or model capacity

### Research Questions
1. Can we attribute aleatoric vs epistemic uncertainty to different FK sources?
2. Do upstream tables (further from target) contribute more epistemic uncertainty?
3. Does join cardinality (1:1 vs 1:N vs N:M) affect uncertainty type?

### Proposed Approach

#### Theoretical Framework
For a prediction on entity $e$ with FK path $P = \{t_1 \to t_2 \to ... \to t_k\}$:

$$\text{Var}[f(x)] = \underbrace{\text{Var}_{\text{model}}[f(x)]}_{\text{epistemic}} + \underbrace{\mathbb{E}_{\text{model}}[\text{Var}[y|x]]}_{\text{aleatoric}}$$

**Key insight**: FK paths create a causal graph where uncertainty propagates:
- Tables with sparse coverage → epistemic (fixable with more data)
- Tables with inherent noise → aleatoric (irreducible)

#### Decomposition Method
```
For each FK group g:
    1. Compute ensemble variance (total uncertainty)
    2. Estimate aleatoric via:
       - Heteroscedastic regression (predict variance)
       - Or: repeated measurements if available
    3. Epistemic = Total - Aleatoric
    4. Attribute to FK origin
```

### Validation Experiments
1. **Synthetic injection**: Add known aleatoric noise to one FK, epistemic (sparse sampling) to another
2. **Data augmentation**: If epistemic, adding data should reduce uncertainty for that FK
3. **Noise vs sparsity**: Distinguish dense-noisy FK from sparse-clean FK

### Why This Matters
- **Actionability**: Epistemic → collect more data; Aleatoric → improve measurement
- **Theoretical grounding**: Not just "which table" but "what kind of problem"
- **Novel contribution**: First principled uncertainty decomposition for relational data

### Key References
- Kendall & Gal (2017) - "What Uncertainties Do We Need in Bayesian Deep Learning?"
- Depeweg et al. (2018) - Decomposition of uncertainty in BNNs
- Law of total variance for hierarchical decomposition

---

## 2. Causal Attribution Through FK Paths

**Core Idea**: Use relational structure for causal discovery, not just correlational importance.

### Background
Current FK attribution shows correlation: "RESULTS table features are most predictive." But this doesn't answer:
- Is RESULTS *causally* responsible for errors?
- Or is it confounded by an upstream table?
- What's the causal mechanism of error propagation?

### Research Questions
1. Can FK paths encode causal relationships?
2. Can we identify which FK introduces *causal* vs *confounded* uncertainty?
3. Does intervention on causal FK reduce uncertainty more than confounded FK?

### Proposed Approach

#### Causal Graph from Schema
FK relationships often encode temporal/causal structure:
```
DRIVERS → QUALIFYING → RESULTS → STANDINGS
         (practice)   (race)    (season)
```

This forms a DAG where:
- Parents are potential causes
- Children are potential effects
- Confounders create backdoor paths

#### Causal Attribution Method
```
1. Construct causal DAG from FK schema
2. For each FK group:
   a. Compute observational importance (current method)
   b. Compute interventional importance:
      - do(X) intervention (replace with population mean)
      - Measure causal effect on prediction uncertainty
3. Compare: Causal effect > Observational effect → true cause
             Causal effect < Observational effect → confounded
```

#### Theoretical Framework
Using do-calculus notation:

- **Observational**: $P(Y | \text{high uncertainty from FK}_i)$
- **Interventional**: $P(Y | do(\text{FK}_i = \text{mean}))$

If these differ significantly, confounding exists.

### Validation Experiments
1. **Known causal structure**: Use datasets with documented causal relationships
2. **Synthetic confounding**: Inject confounders, verify method identifies them
3. **Intervention vs observation**: Compare "fix FK" intervention with observational importance

### Why This Matters
- **Root cause analysis**: Find the *actual* source of uncertainty, not just correlates
- **Actionable interventions**: Intervening on confounded FK won't help
- **Novel contribution**: First causal uncertainty attribution for relational data

### Key References
- Pearl (2009) - Causality
- Peters et al. (2017) - Elements of Causal Inference
- Janzing et al. (2013) - Quantifying causal contributions

---

## 3. Active Learning with FK-Level Uncertainty

**Core Idea**: Use FK-level uncertainty to guide which *tables* need more data, not just which samples.

### Background
Standard active learning: "Which sample should I label next?"
Relational active learning: "Which table/FK should I collect more data for?"

This is more actionable in enterprise settings where:
- Data comes from different systems/teams
- Collection costs vary by source
- Some sources are easier to improve than others

### Research Questions
1. Can FK-level uncertainty guide data collection strategy?
2. Does acquiring data for high-uncertainty FK reduce overall uncertainty?
3. How does this compare to sample-level active learning?

### Proposed Approach

#### FK-Level Acquisition Function
```
For each FK group g:
    1. Compute average epistemic uncertainty for samples using g
    2. Estimate "coverage" of g (% of FK values seen in training)
    3. Acquisition score = epistemic_uncertainty(g) × (1 - coverage(g))
    4. Prioritize FK with highest score
```

#### Acquisition Strategies
1. **Uniform FK expansion**: Add samples uniformly from highest-uncertainty FK
2. **Rare value targeting**: Target rare FK values (low coverage in training)
3. **Uncertainty-weighted sampling**: Sample from FK weighted by uncertainty contribution

#### Simulation Framework
```
1. Start with sparse training set
2. Iterate:
   a. Train model, compute FK-level uncertainty
   b. Select FK to expand using acquisition function
   c. Add N samples from that FK
   d. Retrain, measure uncertainty reduction
3. Compare: FK-guided vs random vs uncertainty sampling
```

### Validation Experiments
1. **Simulated acquisition**: Start sparse, measure uncertainty reduction per added sample
2. **Cost-aware acquisition**: Weight FKs by collection cost, optimize uncertainty/cost ratio
3. **Comparison baselines**:
   - Random sampling
   - Standard uncertainty sampling (sample-level)
   - Diversity sampling

### Why This Matters
- **Practical impact**: Enterprises care about *where* to invest in data quality
- **Novel framing**: Active learning at schema level, not sample level
- **Cost efficiency**: Potentially much cheaper than sample-level acquisition

### Key References
- Settles (2012) - Active Learning literature survey
- Konyushkova et al. (2017) - Learning active learning
- Ren et al. (2021) - A survey of deep active learning

---

## Comparison: Which Direction to Pursue?

| Direction | Novelty | Difficulty | Impact | Paper Fit |
|-----------|---------|------------|--------|-----------|
| **Uncertainty Decomposition** | High | High | High | Theory-focused venue (NeurIPS, ICML) |
| **Causal Attribution** | Very High | Very High | Very High | Causality venue (UAI, CLeaR) or top ML |
| **Active Learning** | Medium-High | Medium | Very High | Applied venue (KDD, SIGMOD) |

### Recommended Priority
1. **Uncertainty Decomposition** - Most natural extension, clear experimental validation path
2. **Active Learning** - High practical impact, easier to validate, good for applied venues
3. **Causal Attribution** - Highest novelty but requires careful theoretical work

### Integration Opportunity
These directions are complementary:
- Decomposition tells you *what type* of uncertainty
- Causal attribution tells you *where it comes from*
- Active learning tells you *what to do about it*

A unified framework could address all three:
> "Schema-aware uncertainty decomposition enables causal root cause analysis and FK-level active learning for relational ML"

---

## Implementation Notes

### Data Requirements
- Need datasets with:
  - Rich FK structure (≥4 FK groups)
  - Known causal relationships (for validation)
  - Ability to simulate data acquisition (for active learning)

### Suitable Datasets
- **rel-salt**: Complex ERP schema, clear temporal structure
- **rel-hm**: Rich product hierarchy, customer-article-transaction chain
- **Synthetic**: Full control for validation experiments

### Computational Requirements
- Decomposition: 2x current (need heteroscedastic models)
- Causal: 3-5x current (multiple intervention experiments)
- Active Learning: 10x current (iterative retraining)

---

*Created: 2025-12-23*
*Status: Open research directions for future work*
