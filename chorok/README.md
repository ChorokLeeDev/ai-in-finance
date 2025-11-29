# Uncertainty Quantification for Relational Data

## Research Journey (2025-11-29)

This project explores uncertainty quantification and attribution for relational databases.

---

## Folder Structure

```
chorok/
├── v1_fk_causal_uq/     # DEPRECATED: FK as causal prior
├── v2_aggregation_uq/   # PAUSED: Aggregation-aware calibration
├── v3_fk_risk_attribution/  # CURRENT: FK-level risk attribution
├── cache/               # Cached datasets
├── figures/             # Generated plots
├── results/             # JSON results
└── archive/             # Old files
```

---

## Research Directions

### V1: FK-Causal-UQ (DEPRECATED)

**Idea**: Use FK structure as causal DAG for interventional uncertainty attribution.

**Result**: Same as Causal SHAP (Heskes 2020). No advantage from FK structure.

**Lesson**: Permutation-based methods are equivalent regardless of DAG knowledge.

→ See [v1_fk_causal_uq/](v1_fk_causal_uq/)

---

### V2: Aggregation-Aware UQ (PAUSED)

**Idea**: Models are miscalibrated for aggregation uncertainty (1/√n).

**Evidence**: 12% error difference between low/high cardinality (p=0.039), but model uncertainty is same.

**Problem**: Real phenomenon, but contribution is weak.

→ See [v2_aggregation_uq/](v2_aggregation_uq/)

---

### V3: Process Risk Attribution (CURRENT)

**Idea**: Decompose prediction uncertainty by data source (FK relationship) for process risk assessment.

**Goal**: "Which process/relationship contributes most to prediction risk?"

**Novel Contribution**:
- Relational data has natural structure (FK = data source = process step)
- Uncertainty can be decomposed along this structure
- Enables counterfactual risk analysis: "What if we change supplier X?"
- Actionable: tells you which process to investigate, not which feature to change

**Related but Different**:
- Existing work (InfoSHAP, SHAP): Feature-level, single table, model explanation
- Our focus: Relationship-level, multi-table, process risk management

→ See [v3_fk_risk_attribution/](v3_fk_risk_attribution/)

---

## Key Literature

| Paper | Relevance |
|-------|-----------|
| [InfoSHAP (NeurIPS 2023)](https://arxiv.org/abs/2306.05724) | Feature-level uncertainty attribution |
| [Causal SHAP (NeurIPS 2020)](https://arxiv.org/abs/2011.01625) | Interventional attribution |
| [Depeweg (ICML 2018)](https://proceedings.mlr.press/v80/depeweg18a.html) | Epistemic/Aleatoric decomposition |
| [Risk Attribution via Shapley](https://academic.oup.com/rof/article/20/3/1189/2461317) | Shapley for risk decomposition |

---

## Current Focus

**FK-Level Risk Attribution**:
1. Decompose uncertainty by FK relationship
2. Enable process risk assessment
3. Support counterfactual questions ("what if we change supplier?")
4. More actionable than feature-level attribution

---

*Last Updated: 2025-11-29*
