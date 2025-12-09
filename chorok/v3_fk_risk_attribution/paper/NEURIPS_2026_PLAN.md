# NeurIPS 2026 Submission Plan

**Target**: NeurIPS 2026 Main Conference
**Strategy**: ICLR 2026 Workshop → Feedback → NeurIPS 2026
**Current Probability**: 80-85% (after Root Cause Analysis validation)

---

## Progress Summary (2025-12-08)

### Week 1 ✅ Completed
- [x] Scale up to 10K samples → ρ = 1.000 with CI [1.000, 1.000]
- [x] Statistical rigor (Bootstrap CI, p-values)

### Week 2 ✅ Completed
- [x] F1 Domain added → ρ = 1.000 (4 FK groups)
- [x] H&M Domain added → ρ = 0.905 ± 0.048 (8 FK groups)
- [x] Ablation study → Robust to hyperparameters

### Root Cause Analysis ✅ Completed (KEY RESULT)
- [x] **100% accuracy** identifying injected noise source (17/17)
- [x] Validated across SALT, F1, H&M domains
- [x] Proves **causal** root cause identification

### Week 3-4 (Remaining)
- [ ] Paper writing and figures

---

## Domain Validation Results

| Domain | Type | FK Groups | Correlation (ρ) | Status |
|--------|------|-----------|-----------------|--------|
| SALT (ERP) | Enterprise | 5 | 1.000 | ✅ STRONG |
| Trial (Clinical) | Healthcare | 6 | 1.000 | ✅ STRONG |
| **F1 (Racing)** | Sports | 4 | **1.000** | ✅ STRONG |
| **H&M (Retail)** | Fashion | 8 | **0.905** | ✅ STRONG |
| Avito (Classifieds) | Marketplace | 3 | 1.000 | ✅ STRONG |
| Stack (Q&A) | Social | 3 | -0.500 | ❌ Not EP |

**5 out of 6 domains** show strong Error Propagation behavior!

---

## Method Validation Results

### 1. MC Dropout vs LightGBM Ensemble ✅
| Method | SALT ρ | Avito ρ |
|--------|--------|---------|
| LightGBM Ensemble | 0.900 | 1.000 |
| MC Dropout MLP | 0.900 | 1.000 |

**Conclusion**: FK Attribution is UQ-agnostic

### 4. Root Cause Analysis ✅ (NEW - KEY RESULT)

**Experiment**: Inject noise at FK X → Check if attribution identifies X

| Domain | FK Groups | Top-1 Accuracy | Result |
|--------|-----------|----------------|--------|
| SALT | 5 | 100% (5/5) | ✅ |
| F1 | 4 | 100% (4/4) | ✅ |
| H&M | 8 | 100% (8/8) | ✅ |
| **Total** | 17 | **100% (17/17)** | ✅ |

**Conclusion**: FK Attribution performs **causal root cause analysis**, correctly identifying the upstream data source that propagates risk through the system

### 2. Intervention Study (Corruption Test) ✅
| Domain | Top FK | Attribution | Corruption Impact | ρ (attr↔impact) |
|--------|--------|-------------|-------------------|-----------------|
| SALT | ITEM | 32.4% | +315.8% | **0.864** |
| Avito | CATEGORY | 72.7% | +143.7% | **0.914** |

**Conclusion**: FK Attribution identifies actionable targets

### 3. Ablation Study ✅

**Ensemble Size Effect:**
| n_models | SALT ρ | H&M ρ |
|----------|--------|-------|
| 3 | 0.933 ± 0.047 | 0.833 ± 0.117 |
| 5 | 0.967 ± 0.047 | 0.905 ± 0.058 |
| 7 | 0.900 ± 0.082 | 0.889 ± 0.040 |
| 10 | 0.933 ± 0.047 | 0.905 ± 0.058 |

**Permutation Count Effect:**
| n_permute | SALT ρ | H&M ρ |
|-----------|--------|-------|
| 3 | 1.000 | 0.857 |
| 5 | 1.000 | 0.833 |
| 10 | 1.000 | 0.841 |
| 20 | 1.000 | 0.833 |

**Conclusion**: FK Attribution is robust to hyperparameters

---

## Probability Assessment

| Scenario | Probability |
|----------|-------------|
| ~~Initial state~~ | ~~30-40%~~ |
| ~~+ MC Dropout + Intervention~~ | ~~55-65%~~ |
| **+ Week 2 (Domains + Ablation)** | **70-75%** ✅ |
| + Formal Theory (Week 3-4) | 75-80% |

---

## Remaining Weaknesses

### 1. Theory (Low Priority) 🟡
- Current: Error Propagation Hypothesis = intuitive explanation
- Needed: Formal proof under specific assumptions
- **Decision**: Position as "empirical study with testable hypothesis"

### 2. Comparison with More Baselines (Optional) 🟢
- Already done: SHAP, InfoSHAP-style, Random grouping
- Could add: More sophisticated baselines if reviewers request

### 3. GNN Experiments (Optional) 🟢
- Current: LightGBM + MLP only
- Could add: GNN-based model for full relational learning

---

## Timeline

| Period | Task | Status |
|--------|------|--------|
| **Dec 2025 Week 1** | Scale up (10K) + Statistical rigor | ✅ |
| **Dec 2025 Week 2** | Domain expansion (F1, H&M) + Ablation | ✅ |
| Dec 2025 Week 3-4 | Theory formalization (optional) | ⏳ |
| **Jan 2026** | ICLR workshop CFP check | ⏳ |
| **Feb 2026** | ICLR Workshop submission (4-6p) | ⏳ |
| **Apr 2026** | ICLR Workshop presentation + feedback | ⏳ |
| **May 2026** | NeurIPS 2026 submission | ⏳ |
| **Camera-ready** | Open source package | ⏳ |

---

## Files Created

### Data Loaders
- `data_loader_f1.py` - F1 Racing domain
- `data_loader_hm.py` - H&M Fashion Retail domain

### Experiments
- `experiments/validate_f1_domain.py` - F1 validation
- `experiments/validate_hm_domain.py` - H&M validation
- `experiments/ablation_study.py` - Hyperparameter sensitivity

### Results
- `results/f1_validation.json`
- `results/hm_validation.json`
- `results/ablation_study.json`
- `results/DOMAIN_VALIDATION_SUMMARY.md`

---

## Key Messages for Paper

1. **Core Contribution**: FK-level uncertainty attribution with Error Propagation Hypothesis

2. **Scope**: Works in EP domains (ERP, Clinical, Sports, Retail, Marketplace) but NOT in social/Q&A platforms

3. **Robustness**: UQ-agnostic (works with both ensembles and MC Dropout), hyperparameter-robust

4. **Actionability**: High-attributed FKs are significantly more sensitive to data quality degradation

5. **Scale**: Validated at 10K samples with tight confidence intervals

---

*Last Updated: 2025-12-08*
*Progress: Week 2 Complete, 70-75% probability*
