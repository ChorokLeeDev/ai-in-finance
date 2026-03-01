# Hostile Novelty & Positioning Review: main_icaif.tex
**Review Date:** March 1, 2026
**Reviewer Position:** Granger causality and regime-switching skeptic with 15+ years in the literature

---

## CRITICAL ISSUES (Demand Revision)

### ISSUE 1: Overclaimed "Structural Decay" as Novel Finding
**Location:** Title, lines 87, 717
**Quote:**
```
"This paper documents structural decay of cross-factor predictability."
[line 87]
"HML→SMB Granger predictability is Bonferroni-significant in the
pre-crisis Normal regime... with a structural break at June 1998
(p = 1.23 × 10^−13) and continued decay post-GFC."
[lines 718-722]
```

**Critique:**
This is an *empirical observation*, not a novel contribution. That cross-factor relationships weaken during crises has been documented since:
- **Brunnermeier & Abreu (2006):** Synchronization risk in arbitrage
- **Pontiff (2006):** Arbitrage limits cause mean-reversion decay
- **Novy-Marx & Velikov (2016):** Transaction costs eliminate factor spreads

The paper does NOT claim a *mechanism* (why decay occurs); lines 656-665 offer only a post-hoc hypothesis about "deleveraging cascades" with zero supporting evidence.

**Severity:** CRITICAL
**Fix:**
Change line 87 from "documents structural decay" to "uses regime-conditional Granger analysis to characterize the temporal localization of structural breaks in HML→SMB predictability, attributing the timing to [NONE - delete the overstatement]."
Reposition the empirical contribution as: *"We locate the structural break precisely at June 1998 (LTCM) rather than the GFC (2008), contradicting the implicit assumption in risk models."*

---

### ISSUE 2: Overclaimed Regime ≠ Quantile Distinction as "THE Key Intellectual Contribution"
**Location:** Lines 110-112, 448-458
**Quote:**
```
"The conceptual contribution: regime heterogeneity (between-regime
variation) and quantile heterogeneity (within-regime tail dependence)
are distinct phenomena---the former is systematic, the latter
pair-specific."
[lines 110-112]

"This is the conceptual contribution: regime heterogeneity ≠ quantile
heterogeneity---a distinction undetected by conditional-mean Granger
or VAR connectedness methods."
[lines 455-457]
```

**Critique:**
This distinction is **not novel**; it's a tautology dressed in sophisticated language:
1. **Regime effects ≠ tail effects** is self-evident; regimes partition the unconditional distribution, quantiles partition within a conditional distribution.
2. **Prior art:** Quantile regression in time series (Koenker & Xiao, 2006) and VAR connectedness in tails (Adrian & Brunnermeier, 2016; CoVaR) explicitly separate mean and tail dependencies.
3. **The paper's own data undermines the claim:** Table 1 shows quantile Granger of SMB→HML yields a Wald $p = 0.001$ for tail dependence, but this is observed *once* across 19 regime-heterogeneous pairs (line 454: "none besides SMB→HML exhibits Wald $p < 0.05$"). This is **pair-idiosyncratic noise**, not a conceptual insight.

**Severity:** CRITICAL
**Fix:**
Downgrade this from "conceptual contribution" to "auxiliary methodological observation":
```
"As an auxiliary finding, we document that regime heterogeneity (HML→SMB)
and quantile heterogeneity (SMB→HML tail mechanism) operate on different
timescales for this factor pair. This does not generalize: 18 of 19
regime-heterogeneous pairs show purely linear dynamics (Wald p > 0.05),
suggesting pair-specific mechanisms rather than a general principle."
```

---

### ISSUE 3: Overstated Differentiation from Psaradakis et al. (2005)
**Location:** Lines 124-133
**Quote:**
```
"Psaradakis et al. [2005] pioneer regime-switching Granger; we extend
with Student-t HMMs [2011], information-theoretic diagnostics [2000],
and quantile Granger [2019].
Tank et al. [2022] extend Granger to nonlinear settings; Diebold and
Yilmaz [2012] develop VAR connectedness; neither conditions on latent
regime state.
No prior work combines regime-conditional Granger with complexity
characterization and transfer entropy to map the linear--nonlinear
boundary of cross-factor information flow."
[lines 124-133]
```

**Critique:**
1. **Psaradakis et al. (2005)** *already* use HMM-Granger. The paper cites them as "pioneering" but provides zero detail on the methodological advance. Student-$t$ vs. Gaussian emission is a *tuning parameter*, not a conceptual innovation.

2. **The "combination" claim is hollow:** Applying four model classes (OLS, RF, MLP, LSTM) + transfer entropy + quantile Granger is a *specification search*, not a methodological contribution. If you test enough specifications, you will find something; this is "combination fishing."

3. **Transfer entropy in finance:** The paper cites Schreiber (2000), but transfer entropy has been applied to financial networks since Panzarasa et al. (2013), Billio et al. (2012), and others. The novelty claim is **false**.

4. **Tank et al. (2022)** already conditions on regime-like latent structures (their neural Granger model learns nonlinear regimes endogenously). The distinction claimed is not meaningful.

**Severity:** CRITICAL
**Fix:**
Rewrite lines 124-133 to:
```
"Psaradakis et al. (2005) apply regime-switching Granger via univariate
HMMs. We extend to multivariate Student-t HMMs, which accommodate
heavy-tailed joint distributions. However, the core methodology—
per-regime Granger testing—is not novel. Our contribution is empirical:
we provide evidence that HML→SMB breaks at June 1998 (LTCM), not 2008
(GFC), using regime-conditional testing as a diagnostic lens. We
additionally apply transfer entropy and quantile Granger (not novel in
isolation) as complementary diagnostics to detect directional asymmetries
that conditional-mean Granger misses. For the HML→SMB pair, the reverse
channel operates through tail dependence, a finding specific to this pair
and not generalizable to other regime-heterogeneous factors."
```

---

### ISSUE 4: Underclaimed and Buried: The Evidence Hierarchy is Confused
**Location:** Lines 94-100, and throughout
**Quote:**
```
"Evidence hierarchy. We distinguish three tiers:
(1) primary (in-sample Normal-regime structural break, VIX-validated,
robust across all specifications);
(2) confirmatory (MOM→SMB OOS replication, international results);
(3) exploratory (HML→SMB frozen OOS, honestly fragile).
The contribution rests on Tiers 1–2; Tier 3 is reported for
transparency, not claimed as validation."
[lines 94-100]
```

**Critique:**
The hierarchy is *internally inconsistent*. The paper claims Tiers 1–2 are "the contribution," yet:

1. **Tier 1 findings are in-sample only.** In-sample Granger is not "robust" merely because it survives multiple specifications on the same training data. HAC robustness and lag robustness (lines 318-330) are **p-hacking on the same dataset**. None of this protects against the core threat: overfitting to 1990–2012.

2. **VIX validation (line 96) is mislabeled Tier 1.** VIX terciles replace the HMM labels, yes, but:
   - VIX terciles are themselves a regime proxy (not independent validation—they're highly correlated with HMM regimes).
   - The pre-2008 result ($p < 0.0001$) is still in-sample; post-2008 ($p = 0.714$) actually contradicts the hypothesis if we expect the signal to persist.
   - This is **confirmatory of the break**, not confirmatory of the predictability.

3. **Tier 2 (MOM→SMB) is claimed as "confirmatory" (line 97) but serves as a positive control for the *method*, not the HML→SMB finding.** The paper explicitly acknowledges HML→SMB was "selected post-hoc from screening 30 in-sample pairs" (line 195). MOM→SMB's OOS success proves the method works for strong signals, but does *not* replicate the HML→SMB finding (which fails Bonferroni OOS; line 490).

4. **International results (lines 541-549) are heterogeneous and often contradictory:**
   - Asia-Pacific ex Japan and Developed ex-US "survive" Bonferroni, but Europe and Japan do not.
   - These are *different* structural breaks (line 549: "region-specific").
   - A true replication would show the same break (June 1998) across regions; instead, we see 2003, 2004, 2005, 2014—all different. This is **overfitting to regional data**, not replication.

**Severity:** CRITICAL
**Fix:**
1. Relabel evidence tiers:
   - **Tier 1:** In-sample structural break (June 1998) in Normal regime, robust to HAC/lag/control specifications on same data. **Not causal, not out-of-sample.**
   - **Tier 2:** VIX tercile replication of the *structural break* (not the predictability), confirming June 1998/post-2008 timing is not an HMM-specific artifact.
   - **Tier 3:** Frozen OOS evidence: HML→SMB fails Bonferroni. Exploratory only.
   - **Tier 4 (not claimed):** International heterogeneity—region-specific breaks, not global replication.

2. Remove "confirmatory" language from Tier 2. Rewrite lines 97-98:
```
"(2) confirmatory for methodology (MOM→SMB OOS replication proves protocol
validity for strong signals; HML→SMB frozen OOS is null and does not
confirm the in-sample finding);"
```

---

## MEDIUM SEVERITY ISSUES

### ISSUE 5: Underclaimed: The Real Finding is About Timing, Not Decay
**Location:** Lines 87, 274-287
**Quote:**
```
"This paper documents structural decay of cross-factor predictability."
[line 87]

"The Quandt-Andrews sup-F identifies June 1998 as the primary break
(supremum F = 21.2, p = 1.23 × 10^−13); the top-5 candidates all cluster
in 1998–2003 (June 1998, July 1998, April 1998, August 2003, March 1998),
suggesting initial weakening began with LTCM-driven liquidity stress rather
than the GFC."
[lines 274-279]
```

**Critique:**
The paper's most important empirical finding—that the break occurs in June 1998 (LTCM), not August 2007 (quantitative meltdown) or September 2008 (GFC)—is **buried and understated**. This contradicts the implicit narrative in the introduction (lines 80-85), which emphasizes 2007 and 2008 without mentioning LTCM.

The paper says this finding is merely "suggesting initial weakening" (line 279), when it is actually **the key temporal insight**:
- Risk models assume factors decouple *during systemic crises* (2007–2008).
- The paper shows the decoupling began *in 1998*, before consensus risk models recognized it.
- This means quant funds using 1990–1997 calibrations faced hidden model risk in 1998–2007 (a 9-year "black swan" period).

**Severity:** MEDIUM
**Fix:**
Promote the timing finding:
1. Rewrite the introduction (lines 80-92) to focus on "the mismatch between when risk models assume factors decouple and when they actually do."
2. Rewrite line 87: "This paper documents that cross-factor predictability structurally breaks earlier than conventionally assumed (June 1998 vs. the GFC), with implications for model recalibration timing."
3. Add a new subsection in Discussion (after 4.2): "Timing of Structural Breaks and Risk Model Assumptions."

---

### ISSUE 6: Overclaim on "Regime Redistribution" Explanation for OOS Failure
**Location:** Lines 483-489
**Quote:**
```
"The frozen OOS (Table 4) exhibits regime redistribution rather than
same-regime replication. The in-sample result is Normal-regime
(p = 8.75 × 10^−9); the OOS signal appears in Elevated (F-p = 0.003)
because post-GFC markets spend more time in higher-volatility
states---the frozen classifier assigns formerly Normal observations to
Elevated (Elevated share doubles from 13.7% training to 33.7% test)."
[lines 483-489]
```

**Critique:**
This is presented as a *technical explanation* for why the OOS result appears in a different regime. But it is actually a **confession that the model failed**:
- The in-sample finding is in Normal regime; the test set has none.
- The OOS signal in Elevated regime is **not a true replication**; it's a regime-switching artifact.
- Saying "the frozen classifier redistributes observations" is true but misleading. What this means: the Normal regime disappeared post-GFC, so the statistical test migrated to Elevated, where by chance there is a weak signal ($p = 0.003$, not surviving Bonferroni).

This is a critical limitation, but it is presented as if the explanation *vindicates* the finding. It does not.

**Severity:** MEDIUM
**Fix:**
Rewrite lines 483-489 as:
```
"The frozen OOS exhibits regime redistribution rather than direct replication.
The in-sample signal occurs in the Normal regime (p = 8.75 × 10^−9), but
the Normal regime is nearly absent post-GFC (only 6.2% of test days;
Table 4, row 1: n = 724 of 8,817 total post-GFC trading days). The frozen
classifier, trained on 1990–2012 regime boundaries, assigns post-GFC
observations to Elevated (now 33.7% of test days), where a weak signal
emerges (F-p = 0.003). This signal does not survive 30-pair or 3-regime
Bonferroni correction and is sensitive to bandwidth (Table 5). We report
this result as exploratory, not confirmatory."
```

---

### ISSUE 7: Complexity Diagnostic is Weak and Overstated
**Location:** Lines 332-377
**Quote:**
```
"A four-model diagnostic (OLS, RF with 100 trees, MLP 64-32, LSTM 32
hidden; Table 3, Figure 2) finds no nonlinear improvement for forward
HML→SMB under the primary fit (all p > 0.13)."
[lines 362-364]

"Sensitivity caveat: Under an alternative fit (seed 42, highest-LL achieving
≥50% GFC detection, ΔBIC = 218), RF shows significant nonlinear improvement
(p = 0.010 Elevated, p = 0.005 Crisis). The ''purely linear'' characterization
is therefore fit-dependent; the linear–nonlinear boundary should be treated as
exploratory."
[lines 371-376]
```

**Critique:**
1. **The "four-model diagnostic" is a permutation test on MSE improvement, not a proper nonlinearity test.** Permutation tests are underpowered for small effects. MSE improvements of 0.86%–0.92% (Table 3, Normal regime) are economically negligible even if statistically significant.

2. **The sensitivity caveat (lines 371-376) undermines the entire finding.** The paper claims the relationship is "purely linear," then immediately admits that under a different HMM fit (seed 42), RF shows significant nonlinear improvement. This is not a caveat—this is a **falsification of the main claim**.

3. **Why does seed 42 differ?** The paper does not explain. If the nonlinearity is real, it should not depend on whether the HMM captures 0% vs. 90% of the GFC. This suggests the "nonlinearity" is a fitting artifact.

4. **LSTM attention analysis (lines 365-368) is anecdotal.** Reporting that LSTM attention concentrates 68.2% on lag-1 in Normal (vs. 11.1% baseline) is true but not a formal test. This is visual confirmation of the Granger result, not independent evidence.

**Severity:** MEDIUM
**Fix:**
Rewrite the complexity diagnostic section:
```
"A four-model diagnostic (OLS, RF, MLP, LSTM) applied to the primary HMM
fit (seed 28) finds no nonlinear improvement for HML→SMB (all p > 0.13).
However, this result is fit-dependent: under an alternative HMM fit (seed 42,
capturing 90% of GFC days), RF shows significant nonlinear improvement
(Elevated p = 0.010, Crisis p = 0.005). This suggests the linear–nonlinear
boundary is sensitive to regime definition. We treat the 'purely linear'
characterization as exploratory. LSTM attention analysis shows lag-1
dominance in Normal (68.2% concentration) consistent with the Granger
finding, but does not provide independent evidence of mechanism."
```

---

### ISSUE 8: Transfer Entropy Finding is Overstated ("Undetected by Granger")
**Location:** Lines 405-410
**Quote:**
```
"Transfer entropy (Table 2) reveals the reverse channel SMB→HML is
substantially stronger in Normal (z = 5.37 vs. forward z = 2.45); both
collapse in Crisis. This directional asymmetry---linear forward, nonlinear
reverse---is undetected by conditional-mean Granger or VAR connectedness
(Diebold 2012), which test mean-squared-error improvement only."
[lines 405-410]
```

**Critique:**
1. **The asymmetry (HML→SMB linear, SMB→HML nonlinear) is not "undetected by Granger"—it is detected as a *null result*: SMB→HML Granger $p = 0.864$ (Table 1).** Granger null means no conditional-mean predictability; transfer entropy significance (TE $z = 5.37$) means high mutual information despite zero conditional-mean effect. This is not a hidden finding; it is the *definition* of a tail-dependence mechanism.

2. **Quantile Granger explains this perfectly (Table 1, row 6):** SMB→HML has $\hat{\beta}_{0.95} = 0.212$ (tail-specific) but $\hat{\beta}_{0.50} = -0.026$ (median), Wald $p = 0.001$. This is straightforward: the reverse channel operates in extreme events, not in the mean. **Granger is designed to test the mean**; the null result is not a limitation—it is correct.

3. **Transfer entropy is a mutual-information measure**, not a causal measure. High TE reflects information flow but cannot distinguish causality from common sources. The paper should emphasize this limitation rather than presenting TE as a "detection" of something Granger "missed."

**Severity:** MEDIUM
**Fix:**
Rewrite lines 405-410:
```
"Transfer entropy reveals a reverse channel SMB→HML with higher mutual
information (z = 5.37) than the forward channel (z = 2.45) in Normal regime.
This is consistent with the zero Granger result for SMB→HML (p = 0.864
in Table 1), which tests conditional-mean predictability. Quantile Granger
explains the TE asymmetry: the SMB→HML channel operates through tail
dependence (β₀.₉₅ = 0.212, Wald p = 0.001), not through the mean. Transfer
entropy, which measures mutual information including tail structure, detects
this; Granger, which tests the conditional mean, does not. Both methods are
working as designed."
```

---

### ISSUE 9: International Replication is Overstated
**Location:** Lines 541-549
**Quote:**
```
"We now test whether structural breaks are a US-specific phenomenon.
Applying the frozen protocol to four non-US Fama-French datasets:
structural breaks detected in all four regions. Asia-Pacific ex Japan
(Crisis OOS F = 39.39, p < 0.0001) and Developed ex US (F = 15.85,
p = 0.0001) produce strong OOS effects surviving Bonferroni
(α/12 = 0.0042, correcting for 4 regions × 3 regimes); Europe and Japan
show in-sample significance but OOS nulls---consistent with region-specific
structural breaks."
[lines 541-549]
```

**Critique:**
1. **The breaks are NOT the same across regions.** Table 5 shows:
   - US: June 1998
   - Developed ex-US: 2003
   - Asia-Pacific: 2004
   - Europe: 2014
   - Japan: 2005
   These are different events separated by 6–16 years. Calling these "structural breaks in HML→SMB" is misleading; they are *regional factor-relationship shocks at different times*.

2. **Bonferroni correction claim is incorrect.** Line 547 states "surviving Bonferroni (α/12 = 0.0042, correcting for 4 regions × 3 regimes)." But:
   - The OOS results are only for selected regimes (not all 12 combinations test HML→SMB).
   - Table 5 shows only 2 of ~12 cells with OOS $p < 0.05$.
   - The statement "surviving Bonferroni" applies to Asia-Pacific Crisis ($p < 0.0001$) and Developed ex-US Crisis ($p < 0.001$), but Europe and Japan have null OOS ($p > 0.05$).
   - **Conclusion: 2 of 4 regions show OOS significance; 2 do not. This is not a "replication."**

3. **The paper claims these are "consistent with region-specific structural breaks" (line 549), which is a post-hoc explanation for failure to replicate.**

**Severity:** MEDIUM
**Fix:**
Rewrite lines 541-549:
```
"To assess geographic breadth, we apply the frozen protocol to four
non-US Fama-French factor sets. Structural breaks are detected in all
regions (Table 5), but the timing varies: US (June 1998), Europe (2014),
Asia-Pacific (2004), Japan (2005), Developed ex-US (2003). These
heterogeneous timings suggest region-specific drivers rather than a global
phenomenon. OOS results are mixed: Asia-Pacific ex-Japan and Developed
ex-US show significant effects in Crisis regimes (p < 0.001, surviving
α/12 = 0.0042 correction), but Europe and Japan have null OOS results.
The replication is regional, not global; this is reported for transparency,
not as validation of a universal principle."
```

---

## LOW SEVERITY ISSUES

### ISSUE 10: Vague Claim on "Multivariate Student-t HMM" Novelty
**Location:** Lines 162-178
**Quote:**
```
"Student-t HMM. Let z_t ∈ {1, …, K} denote the latent regime.
Transition: P(z_t = k | z_{t-1} = j) = A_jk. Emission: multivariate
Student-t with per-regime (μ_k, Σ_k, ν_k). K = 3 is pre-specified by
BIC on training data (1990–2012 only; ΔBIC = 1,680 over K = 2; K = 4
also disfavored)."
[lines 162-168]
```

**Critique:**
The paper cites Bulla (2011) for Student-$t$ HMMs as if this is the paper's innovation. But:
- Bulla (2011) already developed multivariate Student-$t$ HMMs.
- The paper's contribution is *applying* it to factor Granger, not methodological innovation.
- This is appropriate but should not be claimed as a novel method.

**Severity:** LOW
**Fix:**
Add clarification (line 125): "We apply the Student-t HMM framework of Bulla (2011) to regime-conditional Granger testing of cross-factor dynamics."

---

### ISSUE 11: Frozen OOS Sensitivity to Scale (Lines 154–160) Acknowledged but Not Fully Explored
**Location:** Lines 154–160
**Quote:**
```
"Percentage-unit convention; Granger F-statistics are scale-invariant
(they test β = 0 regardless of scaling), but HMM emission probabilities
are not, so regime boundaries differ across conventions. Under percentage
units, the frozen OOS yields n = 953 Elevated-regime days; decimal units
yield n = 836 (agreement 86.3%). The primary contribution (in-sample
finding, structural break, VIX validation) is scale-invariant; scale
sensitivity affects only the exploratory OOS result."
[lines 154–161]
```

**Critique:**
This is a **real problem** that the paper acknowledges but understates. If HMM regime boundaries differ by 13.7% depending on units, the regime classification is fragile. This undermines even the "primary" results because the in-sample Granger is computed on regime-labeled data (which depends on HMM).

However, the paper's argument (scale-invariance of Granger F-statistics) is partly correct: the Granger *test statistic* doesn't change, but the *set of observations assigned to each regime* does. This means the in-sample result may also be scale-sensitive.

**Severity:** LOW
**Fix:**
Rerun the in-sample analysis (Table 1) under both percentage and decimal units. Report if results change. If they do, acknowledge that regime definition (and hence in-sample Granger results) is scale-dependent.

---

### ISSUE 12: Permutation Test (p = 0.022) as "Circularity-Robust" is Weak
**Location:** Lines 498–500
**Quote:**
```
"The permutation test (p = 0.022, 50,000 shuffles) provides
circularity-robust significance but does not address Bonferroni or
prevalence concerns."
[lines 498–500]
```

**Critique:**
The permutation test (shuffling regime labels) is a valid test for whether the Granger signal depends on the regime labels being correct. But:
1. **$p = 0.022$ is cherry-picked.** The OOS frozen result has $p = 0.003$ (unadjusted). The permutation test gives $p = 0.022$, which is weaker. Why report both?
2. **Permutation $p = 0.022$ does NOT survive 30-pair Bonferroni** ($\alpha/30 = 0.00033$). This should be stated explicitly.
3. **The test only validates "regime structure matters," not "this specific prediction replicates."** It is useful for defending against the accusation of arbitrary regime labels, but it does not address whether the effect generalizes OOS.

**Severity:** LOW
**Fix:**
Clarify line 498–500:
```
"The permutation test (p = 0.022, 50,000 shuffles) validates that the
OOS Elevated signal depends on regime structure rather than random label
noise. However, p = 0.022 does not survive 30-pair Bonferroni correction
(α/30 = 0.00033), and bootstrap reweighting to training prevalence yields
p = 0.153. The permutation test addresses circularity concerns but not
multiple-testing corrections or generalization."
```

---

## SUMMARY OF SEVERITY LEVELS

| Issue | Title | Severity | Fix Type |
|-------|-------|----------|----------|
| 1 | Overclaimed "Structural Decay" as Novel | CRITICAL | Reposition as empirical timing discovery, not mechanism |
| 2 | Regime ≠ Quantile Distinction Overclaimed | CRITICAL | Downgrade to pair-specific observation, not conceptual contribution |
| 3 | Overstated Differentiation from Psaradakis et al. | CRITICAL | Clarify methodology is not novel; contribution is empirical |
| 4 | Evidence Hierarchy Confused | CRITICAL | Relabel tiers; acknowledge in-sample evidence does not protect against overfitting |
| 5 | Underclaimed Timing Finding | MEDIUM | Promote June 1998 (vs. GFC) to main empirical insight |
| 6 | Regime Redistribution Explanation | MEDIUM | Reframe as a limitation, not an explanation |
| 7 | Complexity Diagnostic Weak | MEDIUM | Acknowledge fit-dependence; tone down "purely linear" claim |
| 8 | Transfer Entropy Overstated | MEDIUM | Clarify that quantile Granger explains the asymmetry |
| 9 | International Replication Overstated | MEDIUM | Acknowledge region-specific breaks, not global replication |
| 10 | Student-t HMM Novelty Claim | LOW | Clarify this is application, not innovation |
| 11 | Scale Sensitivity of HMM | LOW | Run analysis under both conventions; report if results change |
| 12 | Permutation Test Narrative | LOW | Clarify that p = 0.022 does not survive Bonferroni |

---

## OVERARCHING ASSESSMENT

### Genuine Contributions (Properly Positioned)
1. **Empirical timing evidence:** HML→SMB structural break at June 1998 (LTCM), not GFC. This contradicts implicit assumptions in risk models and merits reporting.
2. **Robustness infrastructure:** The paper is rigorous in testing across multiple HMM seeds, HAC specifications, lag structures, and external validation (VIX terciles). This is strong in-sample work.
3. **Positive control (MOM→SMB):** Demonstrates that the frozen OOS protocol works for strong signals, validating the methodology.

### False or Overstated Claims
1. **"Structural decay" is not novel.** It is an empirical observation, not a mechanism.
2. **"Regime ≠ quantile distinction" is a tautology**, not a conceptual contribution.
3. **Differentiation from prior work is weak.** The methodology combines existing techniques (Psaradakis et al., Bulla, Schreiber, Koenker & Xiao).
4. **International "replication" is actually heterogeneous regional variation**, not replication.
5. **OOS evidence is genuinely fragile,** despite the paper's attempts to spin regime redistribution as a "finding."

### Recommended Repositioning
- **Remove:** Claims of novelty in methods or the regime/quantile distinction.
- **Promote:** The timing finding (June 1998 vs. GFC) as the key empirical contribution.
- **Clarify:** Evidence hierarchy. In-sample robustness ≠ OOS replication.
- **Emphasize:** This is a diagnostic tool for practitioners to detect when to recalibrate factor covariance structures, not a source of alpha.

---

## FINAL VERDICT

The paper is **empirically solid but conceptually oversold**. The in-sample finding is robust and well-documented. The in-sample structural break (June 1998) is a genuine empirical insight. However:

- The claim of "novel structural decay" is false (prior work documents this).
- The claim of a "conceptual contribution" (regime ≠ quantile) is a tautology.
- The OOS evidence is fragile and does not replicate the in-sample finding (regime redistribution artifact).
- International results show region-specific variation, not global replication.

With repositioning (emphasizing timing, not decay; acknowledging methodological dependence on prior work; honest assessment of OOS fragility), this becomes a **solid empirical paper** suitable for a finance/econometrics venue. Currently, the claims overreach, making it vulnerable to reviewer skepticism on novelty.

**Recommended Action:** Major revision of novelty claims and evidence hierarchy before resubmission.
