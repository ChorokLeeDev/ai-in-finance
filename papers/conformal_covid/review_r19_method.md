# MethodCritic R19 Final Review

**Paper**: `/Users/i767700/Github/ai-in-finance/papers/conformal_covid/uai_2026/main.tex`
**Bib**: `/Users/i767700/Github/ai-in-finance/papers/conformal_covid/uai_2026/references.bib`
**Date**: 2026-02-22
**Focus**: Final pre-submission audit across all six dimensions. Genuine issues only.

---

## Executive Summary

The paper is in strong shape after 18 rounds of revision. The statistical claims are well-hedged, the theorem's idealization is disclosed, and internal consistency is high. I find no fatal issues. I identify 1 major issue (PSI footnote placement creates a misleading citation context), 3 moderate issues (a theorem assumption gap, a calibration protocol concern, and a threshold table arithmetic check), and 4 minor issues. The paper is suitable for submission.

---

## Fatal Issues

None.

---

## Major Issues

### M1. PSI footnote is attached to MMD, not PSI (Line 59)

**Location**: Section 1, Contribution 2, line 59.

The footnote marker appears after "MMD," but the footnote text defines PSI:

```latex
Standard shift indicators (MMD,\footnote{Population Stability Index (PSI): ...} C2ST, PSI)
```

This creates two problems: (1) a reader clicking the footnote from "MMD" finds an unrelated definition of PSI; (2) a reader at "PSI" later in the parenthetical finds no footnote. The footnote should be attached to "PSI" instead.

**Severity**: MAJOR -- this is a factual misattribution that a reviewer will notice immediately. Simple fix: move `\footnote{...}` from after "MMD," to after "PSI)".

---

## Moderate Issues

### Mo1. Theorem A1 gap: "each term in L is < p_hat(y*)" is not always true for APS

**Location**: Appendix H (line 617), proof details.

The proof states: "Since each term in $L$ is $<\hat p(y^*)$ and $|\mathcal{B}| \le K-1$, $L \le (K-1)\hat p(y^*)$."

This is the correct definition of APS: $\mathcal{B}$ is the set of classes ranked *strictly below* $y^*$, and by definition each such class has $\hat{p}(y) < \hat{p}(y^*)$ only under strict ordering. However, APS allows ties: if multiple classes share the same predicted probability as $y^*$, the inequality becomes $\hat{p}(y) \leq \hat{p}(y^*)$, and the strict-inequality counting bound $L < (K-1)\hat{p}(y^*)$ becomes $L \leq (K-1)\hat{p}(y^*)$. The bound still holds with $\leq$, so the theorem result is unaffected, but the proof sketch's use of strict inequality is imprecise.

The main text (line 141) says "The bound holds regardless of randomized tie-breaking conventions in APS implementations, since $\mathcal{B}$ uses strict inequality." This is slightly misleading: the strict inequality in $\mathcal{B}$'s definition means tied classes are *not* in $\mathcal{B}$ (they are included in $s$), so the bound is correct. But it would be clearer to say: "classes ranked strictly below $y^*$, i.e., with $\hat{p}(y) < \hat{p}(y^*)$ (ties are included in $s$, not in $\mathcal{B}$)."

**Severity**: MODERATE -- the bound is correct but the reasoning could confuse a reviewer familiar with APS tie-breaking. Not a mathematical error, but a clarity gap.

### Mo2. Deterministic calibration split introduces a systematic bias risk

**Location**: Appendix A (line 399).

The calibration split is described as "deterministic first-half/second-half split of the validation set (i.e., the first floor(n_val/2) records form the calibration set)." The paper notes this "preserves temporal order within the validation period."

This is a deliberate design choice but creates a concern: if the validation period (Feb-Jul 2020) exhibits within-period temporal drift (early COVID vs. peak COVID), the first-half calibration set systematically differs from the second-half evaluation set. This means the calibration quantile may be biased relative to what a random split would produce. The paper acknowledges the split is uniform across tasks (so it does not affect the *relative* concentration-drop correlation), but it could affect the *absolute* coverage values in Table 1.

The RAPS experiments (Appendix I, line 688) used "random 50/50 calibration splits," creating a protocol inconsistency. The paper notes this does not affect within-experiment comparisons, which is correct. However, the APS drop values in Table 7 (10-seed) differ from Table 1 (50-seed) and the paper attributes this to "smaller seed range (10 vs. 50) and high variance" -- but the calibration split difference is an additional confound that should be mentioned alongside.

**Severity**: MODERATE -- the correlation analysis is unaffected, but the absolute coverage numbers have an unacknowledged systematic component from the deterministic split. A reviewer may question whether the 77.1% maximum drop would change under random calibration splits.

### Mo3. Threshold sensitivity table: verify TP+FP+FN+TN=16

**Location**: Appendix D, Table 4 (lines 482-493).

At 40%: TP=5, FP=1, FN=1. Precision=5/6=0.833, Recall=5/6=0.833. This implies 6 at-risk tasks and 10 not-at-risk tasks. Check: 5+1+1+(16-5-1-1)=5+1+1+9=16. TN=9.

Cross-reference with Table 5 (lines 507-532): At-risk tasks (drop >15 pp) among the 16 multiclass tasks:
- SALT: s-shipcond (71.6), s-group (71.2), s-payterms (77.1), i-shippoint (18.5) = 4 at-risk
- External: Covertype (81.8), KDDCup99 (15.9) = 2 at-risk
- Total at-risk = 6. Total not-at-risk = 10.

TP=5 at 40% threshold: the 5 flagged at-risk tasks are s-shipcond, s-group, s-payterms, i-shippoint, Covertype (all have C>40%). FN=1: KDDCup99 (C=21.1%, at-risk). FP=1: s-office (C=42.6%, not at-risk). This checks out.

At 50%: TP=2, FP=0, FN=4. The 2 TPs must be tasks with C>50% that are at-risk: s-payterms (54.2%) and s-shipcond (50.7%). i-shippoint (48.8%), s-group (47.3%), and Covertype (49.8%) all fall below 50%, becoming FN along with KDDCup99. That gives FN=4 (i-shippoint, s-group, Covertype, KDDCup99). Recall = 2/6 = 0.333. This checks out.

At 45%: TP=5, FP=0, FN=1. Tasks with C>45%: s-payterms (54.2), s-shipcond (50.7), Covertype (49.8), i-shippoint (48.8), s-group (47.3) = 5 tasks, all at-risk. s-office (42.6%) drops below 45%, so FP=0. FN=1 (KDDCup99). Precision=5/5=1.00. Recall=5/6=0.833. This checks out.

**Verdict**: Table 4 arithmetic is internally consistent. No issue.

---

## Minor Issues

### m1. Abstract length

The abstract is 207 words, which is within most conference limits but dense. UAI 2026 may have a word limit; verify before submission.

### m2. "9 domains" counting ambiguity persists in one location

**Location**: Section 5.5 (line 348), "rho=0.853, p<0.001 (Kendall tau=0.667) across n=16 multiclass tasks."

The preceding sentence says "8 external multiclass datasets." Adding 8 SALT + 8 external = 16 is correct. However, the phrase "9 domains" appears in the abstract (line 44) and Section 1 (line 57) but not here -- this is fine, just noting the counting is correct.

### m3. Gibbs et al. 2025 journal reference

**Location**: `references.bib`, lines 263-271.

The entry gives `journal={Journal of the Royal Statistical Society: Series B}` with volume 87, number 4, pages 1100-1126, year 2025. This is a very recent publication. Verify that these publication details (volume/number/pages) are accurate; JRSS-B 2025 volume 87 issue 4 would be published late 2025. If the paper was only available as a preprint at submission time, the arXiv number (2305.12616) should be added as a note for verifiability.

### m4. Single-author study with no independent replication

The paper is a single-author study (line 369-371). While this is not a methodological flaw per se, reviewers at UAI may note the absence of independent verification of the computational results. The code availability claim (line 375) partially mitigates this. Ensure the repository actually exists and is public before submission.

---

## Internal Consistency Checks

### Numbers cross-referenced across paper

| Claim | Abstract | Section 1 | Results | Appendix | Consistent? |
|-------|----------|-----------|---------|----------|-------------|
| rho=0.853, n=16 | Line 44 | Line 57 | Line 255 | Line 613 | YES |
| p<0.001 | Line 44 | Line 57 | Line 255 | Line 613 | YES |
| Kendall tau=0.667 | Line 44 | Line 57 | Line 255 | --- | YES |
| Boot CI [0.50, 0.96] | Line 44 | Line 57 | --- | --- | YES |
| rho=0.833, n=8 SALT | --- | Line 57 | Line 245 | Line 613 | YES |
| p=0.010 for SALT | --- | Line 57 | Line 245 | --- | YES |
| Boot CI [0.30, 1.00] at n=8 | --- | --- | Line 245 | --- | YES |
| LOO: rho in [0.75, 0.96] | --- | Line 57 | Line 245 | --- | YES |
| Coverage range 0.1%-77.1% | Line 44 | Line 52 | Line 185 | --- | YES |
| Covertype: 81.8 pp drop | Line 45 | --- | Line 255 | Line 522 | YES |
| Covertype C=49.8% | Line 45 | --- | Line 255 | Line 522 | YES |
| KDDCup99: C=21.1% | --- | --- | Line 302 | Line 526 | YES |
| Stack Overflow: C=48.9% | --- | --- | --- | Line 530 | YES |
| Mixed-effects beta1=1.64 | --- | --- | --- | Line 718 | YES |
| Wald p=0.0006 | --- | --- | --- | Line 718 | YES |
| MMD/C2ST rho<=0.19 | Line 44 | Line 59 | Line 340 | --- | YES |
| Retraining +19 pp | Line 45 | Line 61 | Line 344 | --- | Discrepancy (see below) |

### Retraining number discrepancy

Abstract (line 45): "+19 pp, p=0.036"
Contribution 3 (line 61): "+19 pp, p=0.036"
Section 5.4 (line 344): "+18.9 pp (p=0.036)"

The abstract and Contribution 3 round 18.9 to 19, which is acceptable. No issue.

### Table 1 vs Table 5 consistency

Table 1 reports 50-seed means. Table 5 reports concentration values and At-risk labels using the same means. Cross-checking:

- s-shipcond: Drop 71.6% (Table 1) -> At-risk (Table 5). Conc 50.7% -> VULN. CONSISTENT.
- i-shippoint: Drop 18.5% (Table 1, mean) -> At-risk in Table 5 (mean > 15 pp). But Table 1 labels it ROB with dagger footnote. Table 5 labels it "At-risk*" with "High variance" note. The footnote in Table 1 explains the dagger = "meets At-risk criterion under decision framework." CONSISTENT (dual-labeling system explained).
- s-office: Drop 0.1% (Table 1) -> ROB (Table 5). Conc 42.6% -> VULN (Table 5). Protective factor noted. CONSISTENT.

### Table 2 (stratified correlation) consistency

- Row "Multiclass (SALT)": n=8, rho=0.833, tau=0.714, p=0.010, CI [0.30, 1.00] -- matches Section 4.3 text. CONSISTENT.
- Row "Multiclass (9 dom.)": n=16, rho=0.853, tau=0.667, p<0.001, CI [0.50, 0.96] -- matches abstract and Section 1. CONSISTENT.
- Row "Combined (10 dom.)": n=17, rho=0.654, p=0.004 -- matches Appendix F (line 613). CONSISTENT.

### Tau values

At n=8: tau=0.714 (Table 2). At n=16: tau=0.667 (Table 2, abstract). Note tau *decreases* from n=8 to n=16 while rho increases (0.833 to 0.853). This is plausible: Kendall tau and Spearman rho measure different aspects of rank concordance, and adding 8 external tasks with different rank profiles can change them differently. No issue.

---

## Theorem Verification

### Theorem 1, Part (iii): Monotonicity condition

The bound in Eq. 5 is:
$$E[s_{\text{test}}] \geq C(1-(K-1)\varepsilon) + (1-C)(1-(K-1)\bar{h})$$

Differentiate with respect to C:
$$\frac{\partial}{\partial C} = (1-(K-1)\varepsilon) - (1-(K-1)\bar{h}) = (K-1)(\bar{h} - \varepsilon)$$

This is positive when $\bar{h} > \varepsilon$, confirming Part (iii). The sufficient condition $\bar{h} \geq 1/K$ with $\varepsilon < 1/K$ (from A2) gives $\bar{h} > \varepsilon$. CORRECT.

### Theorem 1, Part (iv): Coverage degradation

From Eq. 4: $s(x^{\text{test}}, y^*) \geq 1 - (K-1)[C\varepsilon + (1-C)h(y^* | x_{\setminus 1}^{\text{test}})]$

For $y^*$ to be in the prediction set: $s(x^{\text{test}}, y^*) \leq \hat{q}_\alpha$

Combining: $1 - (K-1)[C\varepsilon + (1-C)h] \leq \hat{q}_\alpha$

Rearranging: $(K-1)[C\varepsilon + (1-C)h] \geq 1 - \hat{q}_\alpha$

$(1-C)h \geq \frac{1 - \hat{q}_\alpha}{K-1} - C\varepsilon$

$h \geq \frac{(1 - \hat{q}_\alpha)/(K-1) - C\varepsilon}{1-C} = T(C)$

So $\mathbb{P}(y^* \in \mathcal{C}(x^{\text{test}})) \leq \mathbb{P}(h(y^*_{\text{cal}}) \geq T(C))$ using (A3). CORRECT.

$T'(C) = \frac{-(1-C)\varepsilon + [(1-\hat{q}_\alpha)/(K-1) - C\varepsilon]}{(1-C)^2}$

Numerator: $-\varepsilon + C\varepsilon + (1-\hat{q}_\alpha)/(K-1) - C\varepsilon = (1-\hat{q}_\alpha)/(K-1) - \varepsilon$

$T'(C) > 0$ when $\varepsilon < (1-\hat{q}_\alpha)/(K-1)$.

Paper states: "satisfied at $\varepsilon = 0$." CORRECT.

### Conservative bound verification (Appendix H, line 619)

With $\varepsilon=0$, $\bar{h}=1/K$, Eq. 5 gives:
$$E[s] \geq C \cdot 1 + (1-C)(1 - (K-1)/K) = C + (1-C)/K$$

For s-shipcond: $K=45$, $C=0.507$. Bound = $0.507 + 0.493/45 = 0.507 + 0.011 = 0.518$. Paper says 0.518. CORRECT.

For s-payterms: $K=137$, $C=0.542$. Bound = $0.542 + 0.458/137 = 0.542 + 0.003 = 0.545$. Paper says 0.545. CORRECT.

For s-group: $K=459$, $C=0.473$. Bound = $0.473 + 0.527/459 = 0.473 + 0.001 = 0.474$. Paper says 0.474. CORRECT.

For i-plant: $K=35$, $C=0.239$. Bound = $0.239 + 0.761/35 = 0.239 + 0.022 = 0.261$. Paper says 0.261. CORRECT.

For s-incoterms: $K=13$, $C=0.237$. Bound = $0.237 + 0.763/13 = 0.237 + 0.059 = 0.296$. Paper says 0.296. CORRECT.

All five bounds verified. The theorem is mathematically correct under its stated assumptions.

---

## Dimension-by-Dimension Assessment

### Dimension 1: Internal Validity

The paper correctly frames itself as an observational study, not a causal experiment. SHAP concentration is presented as "associated with" and "correlates with," not as causing coverage failure. The theorem establishes a directional mechanism under idealized assumptions (A1-A3), and the A1 idealization is disclosed in the footnote. The placebo test (Appendix B) provides a within-design contrast. No unacknowledged confounders beyond what is discussed (class cardinality, addressed with partial correlations).

### Dimension 2: External Validity

The paper is transparent about scope: gradient-boosted models only, multiclass (K>=4) only, tabular data only. External validation spans 9 domains but all use the same LightGBM pipeline. The model-sensitivity analysis (Appendix J) documents the boundary. The protective-factor rule is acknowledged as derived from n=1 (line 249). The paper does not overclaim generalization.

### Dimension 3: Statistical Rigor

- Power analysis provided: n=8 exploratory (power~0.76), n=16 confirmatory (power>0.99). Appropriately hedged.
- Multiple comparisons: Holm-Bonferroni for 5 concentration metrics (adjusted p=0.050). Acknowledged as boundary.
- Bootstrap CIs: percentile method noted; BCa alternative acknowledged (line 245).
- Mixed-effects: KR correction caveat provided (line 718).
- Retraining: Holm correction applied (p=0.036 -> p=0.11). Honestly reported.
- Effect sizes and CIs consistently reported alongside p-values.

### Dimension 4: Measurement Quality

SHAP concentration stability quantified: CV < 1%, 95% CI within +/-1 pp (line 245). ICC analysis for task independence (Appendix F). Seed stability protocol for external datasets (Appendix E, line 544).

### Dimension 5: Reproducibility

Code availability claimed (line 375, 391). Software versions listed (line 413). Seeds specified (42-91). Hyperparameters fixed (not tuned). Calibration protocol documented. External dataset protocols in Table 6. The deterministic calibration split is documented. This scores well.

### Dimension 6: Logical Soundness

Conclusions are well-calibrated to the evidence. The abstract says "strongest association" rather than "only" or "best." The framework is labeled "exploratory." Limitations are embedded throughout (n=1 protective factor, Holm-corrected retraining result, model-specificity). The paper does not conflate correlation with causation.

---

## Reproducibility Score

**8/10**. Code claimed public, seeds specified, hyperparameters fixed, software versions listed. Deductions: (1) cannot verify repository exists without checking URL; (2) deterministic calibration split is documented but unusual -- most CP papers use random splits, and a reader would need the exact data ordering to reproduce.

---

## Recommended Actions (Priority Order)

1. **[MAJOR]** Move the PSI footnote from after "MMD," to after "PSI" in Contribution 2 (line 59). One-line fix.
2. **[MODERATE]** In Appendix I (RAPS, line 688), add that the calibration split difference (random vs. deterministic) is an additional reason the 10-seed APS drops differ from Table 1, alongside the seed-count difference already mentioned.
3. **[MODERATE]** In the proof sketch (Appendix H, line 617), change "Since each term in $L$ is $<\hat p(y^*)$" to "Since each term in $L$ is $\leq \hat p(y^*)$" to handle ties correctly. The bound holds either way.
4. **[MINOR]** Verify the Gibbs et al. 2025 JRSS-B volume/issue/pages are final publication metadata, not preprint placeholders.
5. **[MINOR]** Verify the GitHub repository URL is live and public before submission.

---

## Verdict

**MINOR REVISION REQUIRED**

The paper has one major issue (misplaced footnote -- trivial fix) and two moderate issues (calibration split documentation, proof sketch precision). All are fixable in under 30 minutes. No fatal flaws, no statistical overclaims, no internal inconsistencies in numbers. The theorem is mathematically correct. The paper is ready for submission after these minor fixes.
