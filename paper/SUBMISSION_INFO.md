# RelUQ: NeurIPS 2026 Submission Guide

**Last Updated:** 2025-12-08

---

## Target Venue

| Item | Value |
|------|-------|
| **Conference** | NeurIPS 2026 |
| **Track** | Main Conference |
| **Deadline** | ~May 2026 (abstract: 4 days before full submission) |
| **Format** | 8 pages + unlimited references/appendix |
| **Review** | Double-blind |

---

## Subject Areas

### Primary Area (select ONE)

| Option | Rationale |
|--------|-----------|
| **`Probabilistic Methods`** | Causal inference foundation (do-calculus), Error Propagation Hypothesis |
| `General Machine Learning` | Novel attribution framework for uncertainty |

**Recommendation:** `Probabilistic Methods` - stronger differentiation from SHAP/interpretability papers

### Secondary Areas (select multiple)

| Area | Relevance |
|------|-----------|
| `Social/Economic aspects of ML` | Interpretability, actionability for practitioners |
| `Applications` | ERP, clinical trials, supply chain |
| `Theory` | Error Propagation Structure theorems |

---

## Paper Positioning

### Core Contribution
**FK-level uncertainty attribution** with **Error Propagation Hypothesis**

### Key Differentiators

| vs. | RelUQ Advantage |
|-----|-----------------|
| Feature-level SHAP | Stable (0.93 vs 0.96 with 5x fewer groups) + Actionable |
| Correlation clustering | Fixed hierarchy (schema-defined) + Interpretable |
| Random grouping | Meaningful (ρ=0.93 vs ρ=-0.40) |

### Validated Scope

| Domain Type | Applicability | Evidence |
|-------------|---------------|----------|
| ERP / Supply Chain | **VALIDATED** | SALT: ρ=0.90 |
| Clinical Trials | **VALIDATED** | Trial: ρ=0.94 |
| Retail / E-commerce | Expected | (H&M planned) |
| Online Classifieds | Expected | (Avito planned) |
| Social Networks | **NOT APPLICABLE** | Stack: ρ=-0.50 |
| Content Platforms | NOT APPLICABLE | - |

---

## Submission Checklist

### Required Components

- [ ] Main paper (8 pages, NeurIPS template)
- [ ] Abstract submitted 4 days before deadline
- [ ] Supplementary material (proofs, additional results)
- [ ] Code availability statement
- [ ] Ethics statement (if applicable)

### Recommended Additions

- [ ] Open-source code release (GitHub)
- [ ] Reproducibility checklist
- [ ] Broader impact statement

---

## Current Limitations (Honest Assessment)

### Methodological

| Limitation | Status | Plan |
|------------|--------|------|
| Only ensemble variance for UQ | Gap | Add MC Dropout, Conformal comparison |
| Only permutation for attribution | Gap | Add SHAP baseline comparison |
| Only LightGBM | Gap | Add MLP validation |

### Domain

| Limitation | Status | Plan |
|------------|--------|------|
| Only 2 strong validation domains | Gap | Add Avito, H&M |
| Not applicable to social/content | **By Design** | Error Propagation Hypothesis scope |

### Theory

| Limitation | Status | Plan |
|------------|--------|------|
| Proof sketches, not rigorous | Acceptable | Full proofs in appendix |
| τ threshold not specified | Gap | Empirical recommendation (τ=0.7) |

---

## Reviewer Anticipated Questions

| Question | Prepared Response |
|----------|-------------------|
| "How does this compare to SHAP?" | Run TreeSHAP comparison; expect similar stability, faster, actionable |
| "Does this work with other UQ methods?" | Run MC Dropout/Conformal; expect similar FK rankings |
| "Why only 2 domains validated?" | Add 2 more (Avito, H&M); Stack is negative result by design |
| "Why not use causal discovery?" | FK schema IS the causal structure in EP domains |
| "What about denormalized data?" | Limitation: requires manual FK definition; future work |

---

## Timeline to Submission

| Month | Week | Milestone |
|-------|------|-----------|
| Dec 2025 | 1-4 | SHAP baseline implementation + comparison |
| Jan 2026 | 5-8 | Avito, H&M domain validation + MC Dropout |
| Feb 2026 | 9-12 | Paper writing + open-source package |
| Mar 2026 | 13-16 | Complete draft + advisor review |
| Apr 2026 | 17-20 | Final revisions + peer review |
| May 2026 | 21-22 | Submit |

---

## Files Reference

| File | Purpose |
|------|---------|
| `paper/main.tex` | Main paper draft |
| `paper/THEORY_ERROR_PROPAGATION.md` | Theoretical foundation (integrate into main.tex) |
| `chorok/v3_fk_risk_attribution/paper/main.tex` | Alternative paper location |
| `chorok/v3_fk_risk_attribution/paper/figures/` | Generated figures (8 PDFs) |

---

*Generated: 2025-12-08*
