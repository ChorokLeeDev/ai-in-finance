# Priority 1 Text Revisions for UAI 2026 Paper

**Date**: 2025-12-27
**Purpose**: Address reviewer concerns by softening causal language and adding statistical evidence

---

## 1. Statistical Test Results (COMPLETED ✓)

### Retraining Comparison - Key Findings:

**Quarterly vs No Retrain**: +18.9%, p=0.0357 → SIGNIFICANT ✓
**Quarterly vs Monthly**: +9.0%, p=0.2367 → NOT SIGNIFICANT ⚠️

**Implication**: Must soften claim about "outperforms"

---

## 2. Causal Language Revisions Needed

### Location 1: Abstract (Line 35)

**CURRENT (OVERCLAIMED)**:
> "catastrophic failures **stem from** single-feature dependence, not feature instability."

**ISSUES**:
- "stem from" implies causality
- Based on n=2 tasks only
- Needs hedging

**REVISED**:
> "catastrophic failures **appear to be associated with** single-feature dependence rather than feature instability."

**OR (STRONGER BUT STILL HEDGED)**:
> "analysis of contrasting task pairs suggests catastrophic failures **arise from** single-feature dependence rather than feature instability."

---

### Location 2: Abstract (Line 36-37) - Retraining Claim

**CURRENT (OVERCLAIMED)**:
> "quarterly retraining restores catastrophic task coverage from 22% to 41% (+19 percentage points), **outperforming** monthly retraining which suffers from noise overfitting."

**ISSUES**:
- "outperforming" implies significant difference
- Statistical test shows p=0.24 (NOT significant)
- "suffers from" is too definitive

**REVISED**:
> "quarterly retraining restores catastrophic task coverage from 22% to 41% (+19 percentage points, p=0.04 vs no retraining), achieving numerically higher mean coverage than monthly retraining (32%), though the difference is not statistically significant (p=0.24). Monthly retraining shows higher variance consistent with noise overfitting."

**ALTERNATIVE (SHORTER)**:
> "quarterly retraining significantly restores catastrophic task coverage from 22% to 41% (p=0.04 vs baseline) with lower variance than monthly retraining."

---

### Location 3: Introduction (Line 69) - Mechanism Discovery

**CURRENT (OVERCLAIMED)**:
> "Retraining experiments show quarterly retraining restores catastrophic task coverage by 19 percentage points, **outperforming** monthly retraining which suffers from noise overfitting"

**REVISED**:
> "Retraining experiments show quarterly retraining significantly restores catastrophic task coverage by 19 percentage points (p=0.04), achieving higher mean coverage than monthly retraining though the difference is not statistically significant (p=0.24)."

---

### Location 4: Section 4.4 (Line 272) - Mechanism Insight

**CURRENT (OVERCLAIMED)**:
> "**Mechanism insight:** Catastrophic failure **occurs when** models develop single-feature dependence that breaks under distribution shift."

**ISSUES**:
- "occurs when" is deterministic causal claim
- Based on 2 tasks

**REVISED**:
> "**Mechanism insight:** Analysis of contrasting task pairs suggests catastrophic failure is **associated with** single-feature dependence that breaks under distribution shift."

**OR (ALTERNATIVE)**:
> "**Mechanism hypothesis:** Catastrophic failure may **occur when** models develop single-feature dependence that breaks under distribution shift. This pattern is observed in sales-shipcond (catastrophic) vs sales-office (robust), and warrants validation across additional tasks."

---

### Location 5: Section 5.2 (Line 309) - Retraining Results

**CURRENT (OVERCLAIMED)**:
> "Surprisingly, quarterly retraining **outperforms** monthly (41.1% vs 32.0% mean coverage)"

**REVISED**:
> "Quarterly retraining achieves highest mean coverage (41.1% vs 32.0% for monthly, p=0.24), though this difference is not statistically significant."

**OR (INLINE HEDGE)**:
> "Quarterly retraining shows numerically higher mean coverage than monthly (41.1% vs 32.0%), though statistical testing reveals this difference is not significant (Wilcoxon p=0.24)."

---

### Location 6: Section 5.2 (Line 311-312) - Noise Overfitting

**CURRENT (OVERCLAIMED)**:
> "Monthly retraining **suffers from** higher variance (std: 23.4% vs 28.3%) and occasional coverage collapse (min: 0.6%), suggesting overfitting to recent noise."

**ISSUES**:
- "suggesting" is OK but could be softer
- Only one experiment

**REVISED**:
> "Monthly retraining exhibits higher variance (std: 28.3% vs 23.4% for quarterly) and occasional coverage collapse (min: 0.6%), **consistent with** overfitting to recent noise, though alternative explanations cannot be ruled out."

**OR (KEEP MOSTLY AS IS)**:
> "Monthly retraining shows higher variance (std: 28.3% vs 23.4%) and occasional coverage collapse (min: 0.6%), **consistent with** overfitting to recent noise."

---

### Location 7: Conclusion (Line 485) - Mechanism Discovery

**CURRENT (OVERCLAIMED)**:
> "**Mechanism discovery**: Catastrophic failures **stem from** single-feature dependence (4.5× importance explosion), not feature instability."

**REVISED**:
> "**Mechanism discovery**: Catastrophic failures **appear to stem from** single-feature dependence (4.5× importance explosion observed in sales-shipcond) rather than feature instability, based on analysis of contrasting task pairs."

**OR (STRONGER WITH CAVEAT)**:
> "**Mechanism discovery**: Analysis suggests catastrophic failures **stem from** single-feature dependence (e.g., 4.5× importance explosion in sales-shipcond) rather than feature instability."

---

### Location 8: Conclusion (Line 487) - Retraining Solution

**CURRENT (OVERCLAIMED)**:
> "Quarterly retraining restores catastrophic task coverage by 19 percentage points, **outperforming** monthly retraining (noise overfitting) and bi-annual retraining (insufficient adaptation)"

**REVISED**:
> "Quarterly retraining significantly restores catastrophic task coverage by 19 percentage points (p=0.04), achieving higher mean coverage than monthly retraining (though p=0.24) and bi-annual retraining (p=0.22)."

**OR (SIMPLER)**:
> "Quarterly retraining significantly improves catastrophic task coverage (+19 percentage points, p=0.04) with lower variance than more frequent retraining."

---

## 3. New Table 4 (With Statistical Tests)

**ACTION**: Replace current Table 4 with the updated version that includes statistical significance indicators.

**FILE**: `papers/conformal_covid/results/retraining/table4_updated_with_stats.tex`

**Changes**:
- Add caption explaining Wilcoxon test
- Bold quarterly row with footnote markers
- Add footnotes: "† Quarterly vs no retrain: p=0.04. ‡ Quarterly vs monthly: not significant (p=0.24)."

---

## 4. Add Statistical Methods to Methodology Section

**NEW PARAGRAPH** (After line 155, in Section 3.4):

```latex
\subsection{Statistical Significance Testing}

For retraining experiments (Section 5.2), we test statistical significance using
Wilcoxon signed-rank test, a non-parametric paired test appropriate for non-normal
distributions. We compare coverage values at the same time points (11 months) across
different retraining frequencies. Significance levels: * $p<0.05$, ** $p<0.01$,
*** $p<0.001$.
```

---

## 5. SHAP Concentration - Pending Full Analysis

**STATUS**: Running analysis on all 8 tasks (in progress)

**Once complete, need to**:
1. Add new table/figure showing concentration for all 8 tasks
2. Validate whether 40% threshold holds
3. Update text to cite full 8-task analysis instead of n=2

**Locations to update after SHAP results**:
- Line 274: "40\% threshold is derived empirically from our data"
  - Add: "validated across all 8 tasks (see Table X)"
- Section 4.4: Add reference to full SHAP concentration table
- Add new Table/Figure in results section

---

## 6. Summary of Changes

| Location | Current Claim | Issue | Revision Strategy |
|----------|---------------|-------|-------------------|
| Abstract line 35 | "stem from" | Causal, n=2 | → "appear to be associated with" |
| Abstract line 36 | "outperforming" | Not significant | → "numerically higher (p=0.24)" |
| Intro line 69 | "outperforming" | Not significant | → Add p-value |
| Section 4.4 line 272 | "occurs when" | Causal, n=2 | → "is associated with" |
| Section 5.2 line 309 | "outperforms" | Not significant | → "highest mean (p=0.24 n.s.)" |
| Section 5.2 line 311 | "suffers from" | Too strong | → "consistent with" |
| Conclusion line 485 | "stem from" | Causal, n=2 | → "appear to stem from" |
| Conclusion line 487 | "outperforming" | Not significant | → "higher mean (p=0.24)" |
| Table 4 | No stats | Missing | → Add Wilcoxon p-values |

---

## Next Steps

1. ✅ **DONE**: Run statistical tests on retraining data
2. ✅ **DONE**: Generate updated Table 4 with p-values
3. 🔄 **IN PROGRESS**: Run SHAP concentration on all 8 tasks
4. ⏳ **TODO**: Apply text revisions to main.tex
5. ⏳ **TODO**: Add statistical methods subsection
6. ⏳ **TODO**: Add SHAP concentration table/figure (after analysis completes)
7. ⏳ **TODO**: Final consistency check

---

## Estimated Impact on Acceptance Probability

**Current state**: 70-75% (with overclaiming risk)

**After Priority 1 fixes**:
- Remove overclaiming → reduces rejection risk
- Add statistical tests → increases rigor
- Complete SHAP analysis → validates mechanism

**Projected state**: 85-88% (strong accept territory)

---

**Document prepared**: 2025-12-27
**Status**: Revisions drafted, awaiting application to manuscript
