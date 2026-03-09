# MethodCritic Final Review

**Paper**: "Diagnosing Conformal Prediction Failures Under Distribution Shift: A COVID-19 Case Study"
**File**: `/Users/i767700/Github/ai-in-finance/papers/conformal_covid/uai_2026/main.tex`
**Date**: 2026-02-22

---

## Executive Summary

The paper is methodologically mature with extensive self-qualification, transparent reporting of limitations, and independently verifiable core statistics. The primary correlation ($\rho = 0.853$, $p < 0.001$, $n = 16$) and all subsidiary claims I could recompute are numerically correct. The theorem is mathematically valid under stated assumptions. I identify one moderate issue (an undiscussed confound), several minor issues, and no fatal flaws. The paper would benefit from acknowledging validation coverage as a predictor of failure severity alongside the existing class-cardinality confound analysis.

---

## Fatal Issues

None identified.

---

## Major Issues

None identified.

---

## Moderate Issues

### 1. Validation coverage is a significant undiscussed confound (Section 2, lines 78; Section 5.3, lines 298--306)

The paper extensively discusses class cardinality as a confound (Section 5.3, Table 4) but does not examine validation coverage itself. Within SALT ($n = 8$), validation coverage correlates significantly with coverage drop: $\rho = -0.762$, $p = 0.028$. Tasks with sub-nominal validation coverage (s-group at 83.6%, s-payterms at 90.8%) are the same tasks that fail catastrophically under shift, raising the question of whether pre-shift miscalibration predicts post-shift failure independently of SHAP concentration.

Critically, the partial correlations show that concentration retains significance after controlling for validation coverage ($\rho_{\text{partial}} = 0.768$, $p = 0.026$), while validation coverage becomes marginal ($\rho_{\text{partial}} = -0.661$, $p = 0.075$). So concentration does carry incremental signal. But the paper does not report this analysis.

Additionally, Section 2 (line 78) states that "ID validation coverage is near-uniform across tasks" to contrast with the Miller et al. (2021) accuracy-on-the-line finding. The actual validation coverages range from 83.6% to 99.9%---a 16.3 percentage-point span. This is not "near-uniform." The rhetorical claim that validation performance cannot distinguish tasks is contradicted by the data: validation coverage alone is a significant predictor of failure severity at $n = 8$.

**Recommendation**: Add validation coverage to the diagnostic comparison in Table 4 as an additional confound check, reporting the partial correlations. Soften or remove the "near-uniform" characterization in Section 2. At $n = 16$, validation coverages for external datasets are not reported in the paper, so this analysis cannot be extended to the primary endpoint without additional data disclosure.

### 2. s-group sub-nominal validation coverage not explicitly discussed (Table 1, line 202)

s-group has validation coverage of 83.6% [81.7, 85.5], entirely below the 90% target. This means the conformal predictor is already failing to achieve nominal coverage *before* any distribution shift occurs. The 71.2 pp "drop" therefore confounds pre-existing calibration failure with shift-induced degradation. The paper notes "high model variance" for s-group but does not flag the sub-nominal baseline as methodologically distinct from the other tasks where the starting point is near-nominal.

With 459 classes and non-randomized APS, sub-nominal validation coverage is expected (Ding et al. 2023 discuss this exact phenomenon). But the paper should acknowledge that for s-group, part of the measured "vulnerability" may reflect a fundamentally miscalibrated starting point rather than shift-induced degradation in the same sense as the other tasks.

**Recommendation**: Add a sentence noting that s-group's validation coverage is sub-nominal and that its large drop partly reflects calibration difficulty with $K = 459$, not purely shift-induced failure. This strengthens the paper by pre-empting the obvious reviewer objection.

---

## Minor Issues

### 3. Partial correlation value discrepancy (Section 5.3, line 306)

The paper reports $\rho_{\text{partial}}(\text{conc.}) = 0.771$ ($p = 0.0008$) controlling for $\log K$ at $n = 16$. I reproduce $\rho = 0.771$ using the Pearson-on-ranks-after-OLS-residualization method. However, the $p$-value I obtain is $p = 0.0005$ rather than $p = 0.0008$. The difference is small and does not affect any conclusion, but the exact $p$ should be verified. This may depend on degrees-of-freedom conventions in the software used.

### 4. Bootstrap CI seed sensitivity (Section 5.3, line 249; Section 1, line 57)

The paper reports bootstrap 95% CI $[0.50, 0.96]$ at $n = 16$ and $[0.30, 1.00]$ at $n = 8$ using 10,000 resamples with the percentile method. My replication with seed 42 gives $[0.51, 0.96]$ and $[0.31, 1.00]$ respectively. The differences are at the rounding boundary and are negligible in practice, but this confirms that the reported CIs are seed-dependent at the second decimal place. The paper does not report which random seed was used for bootstrap resampling, which is a minor reproducibility gap.

### 5. Theorem A1 assumption gap is well-documented but could be more precise (Section 4, lines 148--150)

The footnote on A1 (line 150) correctly acknowledges that TreeExplainer computes SHAP values in log-odds space, not probability space, making A1 an idealization. The paper calls this "directional intuition" rather than a formal guarantee. This is appropriate. However, the relationship between log-odds-space SHAP and the probability-space additive decomposition could be quantified: for small perturbations, the log-odds-to-probability mapping is approximately linear, so A1 holds approximately when SHAP values are small relative to the base rate. For the catastrophic tasks where concentration is high and SHAP values are large, the approximation degrades---precisely where the theorem matters most. This tension could be noted.

### 6. External dataset shift mechanisms are heterogeneous and sometimes undocumented (Section 3.1, line 87; Appendix A.7, Table 2)

The paper acknowledges that "shift mechanisms for 5 UCI pre-defined-split datasets are not separately documented" (line 259). For datasets like Satimage (random 60/20/20 split) and Avila (50/50 split), the "shift" is simply random sampling variation, not a genuine distribution shift. Including datasets with no actual shift as "robust" examples creates a favorable comparison: of course low-concentration models on randomly-split data maintain coverage. The external validation would be more convincing if restricted to datasets with documented, meaningful shift mechanisms (Covertype geographic, Gas Sensor temporal, KDDCup99 attack distribution).

### 7. RAPS calibration split inconsistency (Section 5.2, lines 340)

The RAPS experiments use a random 50/50 calibration split while the main APS experiments use a deterministic first-half/second-half split. The paper notes this in the RAPS table footnote (Appendix F, line 690) and states within-experiment comparisons are unaffected. This is correct, but it means APS drop values in Table 8 (10-seed RAPS comparison) differ from Table 1 (50-seed main results) for two reasons: different seed counts *and* different calibration splits. The footnote could be clearer that both factors contribute to the discrepancy.

### 8. Retraining analysis is single-seed (Section 5.4, lines 347--348)

The retraining results (+18.9 pp, $p = 0.036$ unadjusted, Holm-corrected $p = 0.11$) are explicitly flagged as single-seed and non-significant after correction. The paper handles this responsibly. The Holm correction arithmetic is verified: $3 \times 0.036 = 0.108 \approx 0.11$.

### 9. "9 domains" counting (Section 1, various)

The paper counts SALT as 1 domain and each external dataset as a separate domain, yielding 9 domains for $n = 16$ tasks. This is defensible but asymmetric: SALT contributes 8 tasks from 1 domain while external contributes 8 tasks from 8 domains. The correlation could be driven disproportionately by the 8 SALT tasks sharing the same temporal shift. The within-SALT $\rho = 0.833$ and the cross-domain extension to $\rho = 0.853$ suggest the external data adds confirmatory value, but the domain-imbalance should be noted. A reviewer might ask for the external-only correlation, which would be computable from the 8 external tasks alone.

### 10. Table 3 footnote: "6/8 jackknife significant" (line 280)

My replication confirms that exactly 2 of 8 LOO samples produce $p = 0.052$ (dropping s-shipcond or s-payterms), so 6/8 are significant at $\alpha = 0.05$. This matches the paper's claim.

---

## Theorem Correctness Assessment

Theorem 1 (Section 4, lines 147--177) is mathematically correct under the stated assumptions.

- **Part (i)**: The counting bound $|B| \leq K - 1$ and the substitution of A1--A2 into the APS score definition are valid. The bound direction (lower bound on $s$) is correct.
- **Part (ii)**: Taking expectations and applying A3 (residual exchangeability) is valid.
- **Part (iii)**: The derivative $d(\text{bound})/dC = (K-1)(\bar{h} - \varepsilon) > 0$ when $\bar{h} > \varepsilon$ is correct. The sufficient condition $\bar{h} \geq 1/K$ combined with A2 ($\varepsilon < 1/K$) gives $\varepsilon < \bar{h}$.
- **Part (iv)**: $T'(C) = [(1 - \hat{q}_\alpha)/(K-1) - \varepsilon]/(1-C)^2 > 0$ under the stated condition. The coverage bound is non-increasing in $C$.
- **Conservative bound verification** (Appendix E, line 621): Using $\varepsilon = 0$, $\bar{h} = 1/K$, the bound $C + (1-C)/K$ is verified on all 5 tasks. Values match: e.g., s-shipcond: $0.507 + 0.493/45 = 0.518$ vs. observed 0.98.

The A1 gap (probability space vs. log-odds space) is appropriately disclosed. The theorem functions as intended: providing directional intuition rather than a tight quantitative bound.

---

## Numerical Verification Summary

| Claim | Paper Value | My Value | Status |
|-------|-------------|----------|--------|
| $n = 16$ Spearman $\rho$ | 0.853 | 0.853 | Verified |
| $n = 16$ $p$-value | $< 0.001$ | 0.000027 | Verified |
| $n = 16$ Kendall $\tau$ | 0.667 | 0.667 | Verified |
| $n = 8$ Spearman $\rho$ | 0.833 | 0.833 | Verified |
| $n = 8$ $p$-value | 0.010 | 0.010 | Verified |
| $n = 8$ Kendall $\tau$ | 0.714 | 0.714 | Verified |
| $n = 17$ Spearman $\rho$ | 0.654 | 0.654 | Verified |
| Bootstrap CI ($n = 16$) | [0.50, 0.96] | [0.51, 0.96] | Seed-dependent boundary |
| Bootstrap CI ($n = 8$) | [0.30, 1.00] | [0.31, 1.00] | Seed-dependent boundary |
| Partial $\rho$(conc\|logK) at $n = 16$ | 0.771 | 0.771 | Verified |
| Partial $\rho$(logK\|conc) at $n = 16$ | -0.010 | -0.009 | Verified (rounding) |
| LOO range | [0.75, 0.96] | [0.750, 0.964] | Verified |
| LOO non-significant count | 2/8 at $p = 0.052$ | 2/8 at $p = 0.052$ | Verified |
| Design effect | 34.1 | 34.1 | Verified |
| $n_{\text{eff}}$ | 11.7 | 11.7 | Verified |
| Holm correction (retraining) | 0.11 | 0.108 | Verified |
| Threshold precision/recall at 40% | 0.83/0.83 | 0.83/0.83 | Verified |
| Theorem bounds (5 tasks) | All verified | All verified | Verified |

---

## Reproducibility Score

**7/10**

Strengths: 50-seed ensemble with explicit seed range, all hyperparameters documented, calibration protocol specified, bootstrap method stated, statistical tests named with exact values. The figure-generation code (`/Users/i767700/Github/ai-in-finance/papers/conformal_covid/code/generate_n16_figure.py`) hardcodes the data points and is fully reproducible.

Gaps: No bootstrap random seed reported. External dataset validation coverages not tabulated. No requirements.txt or environment lock file. Software versions stated (Python 3.9, LightGBM 3.3, SHAP 0.41) but not pinned. Data pipeline code is not packaged as a single reproducible script. The partial Spearman method is not specified (Pearson on ranks after OLS residualization vs. other implementations).

---

## Recommended Actions (Priority Order)

1. **Add validation coverage as a confound** in the diagnostic comparison (Table 4 or a new table), reporting partial correlations controlling for val coverage within SALT. This pre-empts the obvious reviewer question and strengthens the case since concentration remains significant.

2. **Soften the "near-uniform" characterization** of validation coverage in Section 2 (line 78). Replace with something like "validation coverages range from 83.6% to 99.9%, with no significant predictive relationship after controlling for concentration."

3. **Note s-group's sub-nominal validation coverage** explicitly in Table 1 or Section 5.1. A short parenthetical noting that $K = 459$ produces expected sub-nominal coverage per Ding et al. (2023) would suffice.

4. **Report the external-only correlation** ($\rho$ for the 8 external multiclass tasks alone) as a supplementary robustness check. This addresses domain-imbalance concerns.

5. **Specify bootstrap random seed** or state that results are averaged over multiple bootstrap runs.

---

## Verdict

**MINOR REVISION REQUIRED**

The core statistical claims are numerically verified, the theorem is correct, and the paper exhibits unusually thorough self-qualification for its limitations. The undiscussed validation-coverage confound (Moderate Issue 1) is the most substantive gap, but the partial correlation analysis shows concentration retains significance after controlling for it---so addressing this issue strengthens rather than weakens the paper. The remaining issues are genuinely minor. The paper is in strong shape for submission.
