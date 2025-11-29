# V3: FK-Level Risk Attribution (CURRENT)

**Status**: ACTIVE (2025-11-29)
**Goal**: FK-level uncertainty decomposition for process risk assessment

---

## Research Question

> "Which FK relationship (process/data source) contributes most to prediction uncertainty?"

## Motivation

**Not just data quality improvement, but RISK ASSESSMENT:**

1. **Process Bottleneck Identification**: Which relationship is the risk bottleneck?
2. **Counterfactual Analysis**: "If we change supplier X, how does uncertainty change?"
3. **Decision Support**: Actionable insights at relationship level, not feature level

## Novel Contribution

**Process Risk Attribution for Relational Data**

| Aspect | Existing Work | Our Focus |
|--------|---------------|-----------|
| Granularity | Feature-level | **Relationship-level** |
| Data type | Single table | **Multi-table (FK structure)** |
| Question | "Which feature causes uncertainty?" | "Which process/data source causes risk?" |
| Goal | Model explanation | **Risk management & decision support** |
| Action | "Change feature X" | "Investigate/improve relationship X" |

**Key Insight**: FK structure = natural decomposition for process risk

## Key Insight

**FK grouping is not arbitrary**:
- FK = Data source / Process step
- FK-level attribution = "Which process contributes to risk"
- More actionable than feature-level

## Files

| File | Description |
|------|-------------|
| `fk_uncertainty_attribution.py` | LOO-based FK attribution |
| `shap_attribution.py` | SHAP baseline |
| `permutation_attribution.py` | Permutation importance baseline |
| `vfa_attribution.py` | Variance-based attribution |
| `compare_attribution_methods.py` | Method comparison |
| `covid_timeline_analysis.py` | COVID case study |
| `statistical_significance.py` | Statistical tests |

## Literature

**Closest competitor**: [InfoSHAP (Watson et al., NeurIPS 2023)](https://arxiv.org/abs/2306.05724)
- Information-theoretic Shapley for uncertainty attribution
- Feature-level, single table

**Our extension**: Relational data, FK-level grouping

## Next Steps

1. Formalize FK-level uncertainty decomposition
2. Compare with InfoSHAP on RelBench data
3. Show FK-level is more actionable via case study
4. Counterfactual analysis framework

---

*Last Updated: 2025-11-29*
