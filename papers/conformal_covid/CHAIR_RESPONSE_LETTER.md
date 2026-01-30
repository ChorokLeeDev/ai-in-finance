# Response to Conference Chair Review
**Paper**: Conformal Prediction Under Distribution Shift: A COVID-19 Natural Experiment
**Author**: Chorok Lee (KAIST)
**Date**: December 27, 2025

---

Dear Conference Chair,

Thank you for the thorough and constructive review (Rating: 8/10, Strong Accept). We greatly appreciate the detailed feedback and have carefully addressed all **critical issues** and **required revisions** identified in your report.

## Summary of Changes

### 1. **Statistical Claims Corrected** ✅
**Issue**: Abstract/conclusion claimed quarterly "outperforms" monthly, but p=0.24 is not statistically significant.

**Resolution**:
- Rewrote Abstract, Introduction, Section 5.2, and Conclusion to clarify: "While quarterly achieves higher mean coverage than monthly (41% vs 32%), this difference is **not statistically significant** (p=0.24)"
- Shifted narrative to **cost-effectiveness** (3× fewer retrains, lower variance, no coverage collapses) rather than statistical superiority
- Now explicitly states both quarterly AND monthly significantly beat baseline

**Impact**: Eliminates misleading claims while preserving valid cost-benefit finding.

---

### 2. **Sensitivity Analysis Added** ✅
**Issue**: With n=8 tasks, one outlier (sales-office) could affect correlation significance.

**Resolution**:
- Added sensitivity analysis in Section 4.4: "Excluding this outlier strengthens the correlation (Spearman **ρ=0.89, p=0.007**, n=7), confirming robustness"
- Explained outlier mechanism: stable secondary feature (SALESORGANIZATION, Jaccard=0.61) provides protection despite 42.6% concentration
- Added statistical caveat: "With n=8 tasks, statistical power is limited. The 40% threshold should be treated as **preliminary guidance** requiring validation"

**Impact**: Demonstrates correlation is robust (p=0.007) while transparently acknowledging limitations.

---

### 3. **Scope & Limitations Section Added** ✅
**Issue**: Generalizability limitations not adequately discussed.

**Resolution**:
- Added comprehensive **Section 7.2: Scope and Limitations** covering:
  - **Domain scope**: 8 supply chain tasks (rel-salt), limited validation on clinical trials/motorsports
  - **Feature types**: Analysis uses categorical features; may differ for continuous, high-dimensional, structured features
  - **Statistical power**: n=8 limitation explicitly acknowledged
  - **Temporal scope**: 11 months (Feb-Dec 2020); longer-term dynamics (2021-2022) remain open
  - **Model class**: LightGBM focus; deep learning may differ
  - **Practical implications**: Findings applicable within scope but require validation for other domains

**Impact**: Transparent about limitations while clarifying where findings are immediately applicable.

---

### 4. **Extended SHAP Analysis to Intermediate Tasks** ✅
**Issue**: Figure 2 shows only extremes (catastrophic vs robust). What about intermediate cases?

**Resolution**:
- Added analysis of i-plant (23.9% concentration, 10.6% drop) and i-incoterms (28.9% concentration, 11.3% drop) in Section 4.4
- Validated mechanism holds across full spectrum: "These moderate-concentration tasks show intermediate coverage degradation, consistent with the hypothesis: their top features show 3-5× importance increases (between catastrophic's 4.5× and robust's distributed pattern)"

**Impact**: Demonstrates mechanism is not cherry-picked from extremes but holds across full range.

---

### 5. **Computational Cost Quantified** ✅
**Issue**: Abstract mentions "save cost" but no concrete numbers.

**Resolution**:
- Added in Section 5.2: "Training a single LightGBM model on sales-shipcond requires **~2 minutes** on standard CPU (8 cores, 8GB RAM). Quarterly retraining costs **~6 CPU-minutes/year** (3 retrains × 2 min) vs **~20 CPU-minutes/year** for monthly (10 retrains × 2 min), making quarterly **3.3× more cost-effective**"

**Impact**: Concrete numbers enable practitioners to make informed cost-benefit decisions.

---

### 6. **Minor Issues Fixed** ✅
- **Table 1 caption**: Added explicit coefficient of variation threshold definition (>50%)
- **Table 4 caption**: Added test direction (two-tailed), corrected std values
- **Section 7.1**: Enhanced knife-edge task guidance with preliminary diagnostic (entropy + rare classes)
- **Section 7.2**: Added stopping criteria hypothesis for retraining (2-3 quarters stability)
- **Throughout**: Specified "Wilcoxon signed-rank test" instead of just "Wilcoxon"

---

## Responses to Specific Questions

### Q1: "With sales-office removed, is correlation still significant?"
**A**: Yes. Spearman ρ=0.89, p=0.007 (n=7). Correlation strengthens without outlier, confirming robustness.

### Q2: "Can you clarify quarterly vs monthly interpretation?"
**A**: Both quarterly and monthly significantly beat baseline. Quarterly shows higher mean but difference is not significant (p=0.24). **Key advantage is cost-effectiveness** (3× fewer retrains) and **stability** (lower variance, no coverage collapses to 0.6%).

### Q3: "Can you validate on continuous features?"
**A**: Not in current submission (requires new experiments). Acknowledged as limitation in Section 7.2. Presented as future work.

### Q4: "Can you predict knife-edge tasks before training 50 seeds?"
**A**: Partial progress. Added preliminary diagnostic in Section 7.1: tasks with high class imbalance (entropy < 1.5) AND many rare classes (e.g., s-group with 459 classes) appear at higher risk. Noted as requiring further investigation.

### Q5: "What happens after Dec 2020?"
**A**: Not tested (requires 2021-2022 data). Acknowledged in Section 7.2. Added stopping criteria hypothesis: monitor coverage stability for 2-3 quarters without retraining.

### Q6: "How confident in 40% threshold generalization?"
**A**: Moderate confidence within scope (categorical features, supply chain/operational data). Added explicit caveat in Section 4.4: "With n=8 tasks, statistical power is limited. The 40% threshold should be treated as **preliminary guidance requiring validation** on additional domains."

### Q7: "Could confounders explain variance better than SHAP concentration?"
**A**: We acknowledge this limitation. Section 7.2 notes that broader validation is needed. The intermediate task analysis (i-plant, i-incoterms) strengthens the mechanism, but we cannot fully rule out confounders with n=8.

---

## Verification

✅ **PDF compiles** without errors (10 pages)
✅ **All p-values** accurately reported
✅ **No significance claims** where p>0.05
✅ **Scope transparently** communicated
✅ **All required revisions** addressed

---

## Changes Summary by Location

| Section | Change Type | Status |
|---------|-------------|--------|
| **Abstract** | Fix quarterly vs monthly claim | ✅ |
| **Introduction** | Fix bullet point 3 | ✅ |
| **Section 4.4** | Add sensitivity analysis | ✅ |
| **Section 4.4** | Extend to intermediate tasks | ✅ |
| **Section 5.2** | Rewrite catastrophic task results | ✅ |
| **Section 5.2** | Add computational costs | ✅ |
| **Section 7.1** | Enhance knife-edge guidance | ✅ |
| **NEW Section 7.2** | Add Scope & Limitations | ✅ |
| **Section 8** | Fix conclusion claims | ✅ |
| **Table 1** | Add CoV threshold | ✅ |
| **Table 4** | Correct values, add test details | ✅ |

---

## Files Submitted

1. **main.pdf** - Revised manuscript (10 pages, 1.5 MB)
2. **REVIEW_RESPONSE_SUMMARY.md** - Detailed change log
3. **CHAIR_RESPONSE_LETTER.md** - This letter

---

## Conclusion

All **critical issues** identified in your review have been comprehensively addressed:

1. ✅ Statistical claims are now **scientifically rigorous** (no significance claims where p>0.05)
2. ✅ **Sensitivity analysis** confirms correlation robustness (ρ=0.89, p=0.007)
3. ✅ **Scope & Limitations** section transparently communicates generalizability bounds
4. ✅ **Intermediate task analysis** validates mechanism across full spectrum
5. ✅ **Computational costs** quantified with concrete numbers
6. ✅ **Minor issues** corrected (captions, test names, thresholds)

The paper maintains its **strong practical impact** (actionable decision framework, cost-benefit analysis, negative result on ACI) while improving scientific rigor and transparency.

We believe these revisions fully satisfy the conditions for acceptance and look forward to presenting this work at the conference.

Thank you again for the constructive feedback that significantly strengthened the paper.

Sincerely,
**Chorok Lee**
Korea Advanced Institute of Science and Technology (KAIST)
choroklee@kaist.ac.kr

---

**Appendix: Optional Future Work (Not Required for Acceptance)**

The chair identified these as "encouraged but optional":
- Validation on continuous features (requires new experiments)
- Extend temporal analysis to 2021-2022 (requires additional data)
- Add alternative baselines (weighted/online conformal)
- Deeper mechanistic analysis of knife-edge tasks

These are noted for follow-up work and journal extensions but are not blocking acceptance of the current submission.
