# Novelty Review: Structural Decay of Cross-Factor Predictability
## ICAIF 2026 Submission | Hostile Academic Reviewer

---

## CRITICAL ISSUES

### 1. **Overclaimed Novelty of "Regime Heterogeneity ≠ Quantile Heterogeneity" Distinction**
**Severity: CRITICAL**

**Issue:** The paper claims as a "conceptual contribution" (line 113-115) that "regime heterogeneity (between-regime variation) and quantile heterogeneity (within-regime tail dependence) are distinct phenomena."

**Critique:**
- This distinction is **trivially obvious** from first principles. Regime switching tests conditional on discrete latent state membership; quantile regression examines distributional shape within a single regime. These measure fundamentally different things by construction.
- The novelty claim relies on the observation that SMB→HML shows significant tail heterogeneity (β₀.₉₅ = 0.212 vs. β₀.₅₀ = -0.026) while HML→SMB does not. But this is an **empirical observation about a single factor pair**, not a methodological contribution.
- The paper provides **no theoretical framework** explaining when/why regime and quantile heterogeneity should diverge. It merely observes they differ for SMB→HML and asserts this matters for methodology.
- No prior work needed to be corrected here. Practitioners and researchers already understand that regime-switching models and quantile regression capture different phenomena. Stating this explicitly is not a contribution.

**Evidence of Overclaim:**
- Line 469-471: "This is the conceptual contribution: regime heterogeneity ≠ quantile heterogeneity---a distinction not captured by conditional-mean Granger or VAR connectedness methods."
  - This sentence conflates two separate claims: (a) that the two concepts differ, and (b) that existing methods don't capture both. Claim (a) is trivial; claim (b) is true but expected, not novel.
- The paper applies off-the-shelf quantile Granger (Troster 2019) with no methodological innovation.

**Verdict:** The distinction exists but is not novel. The empirical observation (SMB→HML exhibits tail heterogeneity) is interesting but insufficient for claiming a conceptual breakthrough.

---

### 2. **Overclaimed Structural Decay Without Mechanistic Evidence**
**Severity: CRITICAL**

**Issue:** The title and central claim (line 89: "This paper documents **structural decay** of cross-factor predictability") uses language implying a causal breakdown in the underlying economic relationship.

**Critique:**
- The paper shows a **statistical decline** in Granger predictability pre- vs. post-2008, but never establishes what caused the decay or whether it reflects changed fundamentals vs. changed market microstructure, regulation, or crowding.
- The quantile Granger results (Table 5) show SMB→HML operates through tail dependence, but this is **within-regime heterogeneity**, not a mechanism for the structural break itself.
- The proposed economic mechanism (lines 674-682, "deleveraging cascade") is **purely speculative**, resting on:
  - A correlation between Granger strength and FF25 portfolio overlap (ρₛ = 0.35, p = 0.046)
  - An observation that Small/HighBM accounts for 39% of ΔR²
  - These two pieces of evidence are **insufficient to establish causation**.

- The paper claims it will test three predictions (lines 680-682) but does not actually conduct these tests. It defers them to future work.

**Where the Overclaim Shows:**
- Lines 105-108: "Empirical documentation of structural decay: HML→SMB predictability is Bonferroni-significant in Normal (p = 8.75 × 10⁻⁹), absent post-2008 (95% CI [-0.049, 0.073])."
  - This is accurate **description**, not explanation. But the framing as "structural decay" suggests the relationship died for structural reasons when it may simply have disappeared due to sampling variation, regime classification drift, or unobserved confounders.

- Lines 89-94: The introduction frames this as documenting a "blind spot" in factor models, implying a substantive economic mechanism, but no mechanism is ever demonstrated.

**Verdict:** The statistical phenomenon is real and robust. The causal interpretation ("structural decay") is speculative and unsupported.

---

### 3. **"No Prior Work Combines..." Claim is Vulnerable and Insufficiently Hedged**
**Severity: CRITICAL**

**Issue:** Lines 134-136 claim: "No prior work combines regime-conditional Granger with complexity characterization and transfer entropy to map the linear--nonlinear boundary of cross-factor information flow."

**Critique:**
- This claim is **technically true but pedantic**. It meets the logical bar only because it specifies an exact combination. But:

  1. **Regime-conditional Granger** is from Psaradakis et al. (2005), acknowledged at line 127.

  2. **Complexity characterization (OLS, RF, MLP, LSTM)** is standard in ML-for-finance literature; combining multiple model classes to assess nonlinearity is not novel. Tank et al. (2022) already extends Granger to nonlinear settings.

  3. **Transfer entropy** in financial settings: Schreiber (2000) is seminal; applying it to Granger is straightforward.

  4. **Quantile Granger** (Troster 2019) is pre-existing.

  - The combination is **additive, not synergistic**. No new insight emerges from combining them beyond what each component provides individually.

- Line 134 should read: "Our novel methodological contribution is to apply existing techniques [list them] to a new domain [cross-factor factor analysis]." Instead, it claims combinatorial novelty, which is weaker.

**Severity** of overclaim: The statement is **not false**, but it's a weak form of novelty---essentially claiming credit for engineering a pipeline. If a competitor had written a paper first applying these exact techniques to cross-asset Granger, that would make this claim false. As written, it's **barely defensible** and relies on the assumption that no competitor has done exactly this.

**Verdict:** Claim is technically sound but represents **minimal novelty**. It should be repositioned as "we apply existing tools in combination" rather than "no prior work has done this exact combination."

---

### 4. **Insufficient Differentiation from Psaradakis et al., Tank et al., Diebold-Yilmaz**
**Severity: CRITICAL**

**Issue:** Lines 127-133 acknowledge prior work but don't establish sufficient separation:

> "Psaradakis et al. pioneer regime-switching Granger; we extend with Student-t HMMs, information-theoretic diagnostics, and quantile Granger. Tank et al. extend Granger to nonlinear settings; Diebold and Yilmaz develop VAR connectedness; neither conditions on latent regime state."

**Critique:**

**a) Psaradakis et al. (2005):**
- Psaradakis already did **regime-switching Granger**. The paper's extension to Student-t HMMs is marginal:
  - Bulla & Mergner (2001, 2011) developed Student-t HMMs for financial data.
  - Combining two existing techniques (Student-t HMM + Granger) is an engineering step, not a conceptual advance.
- The paper should articulate **precisely** what is novel about Student-t HMM-based regime-switching Granger vs. Psaradakis's approach. It doesn't. Lines 128 merely list the tools without explaining the innovation.

**b) Tank et al. (2022):**
- Tank et al. extend Granger to **nonlinear neural settings** systematically.
- The paper's complexity characterization (Table 3, RF/MLP/LSTM) finds **no nonlinear improvement** for forward HML→SMB (all p > 0.13, line 373).
- This null result actually **weakens** novelty: the paper applies Tank et al.'s framework and finds a null, then pivots to transfer entropy and quantile Granger. It doesn't advance the nonlinear-Granger frontier; it shows linear methods suffice.

**c) Diebold-Yilmaz:**
- Diebold-Yilmaz (2012, 2014) develop **conditional connectedness**, which is a generalization of Granger-based information flow.
- The paper claims regime-conditioning is novel vs. Diebold-Yilmaz, but **doesn't compare** the two approaches empirically or theoretically. Do regime-conditional Granger and DY connectedness capture the same phenomena? Different ones? It's never clarified.
- Lines 418-420 claim TE + quantile Granger reveal asymmetries "not captured by conditional-mean Granger or VAR connectedness," but this is a **feature of TE and quantile methods**, not the regime-conditioning framework.

**Verdict:** The paper sits awkwardly between three literatures (regime-switching, nonlinear Granger, connectedness) without clearly advancing any single one. The differentiating features are:
1. Application to factor pairs (empirical domain)
2. Combination of existing tools
3. Finding of structural break in 1998/2008

None of these is a methodological innovation.

---

### 5. **Complexity Characterization Claim is Overclaimed**
**Severity: CRITICAL**

**Issue:** Lines 109-112 claim the "complexity diagnostic (OLS, Random Forest (RF), MLP, LSTM) combined with transfer entropy reveals a directional asymmetry (linear forward, nonlinear reverse via tail dependence) not captured by conditional-mean methods."

**Critique:**
- The complexity diagnostic (RF, MLP, LSTM) finds **no significant nonlinear improvement** for forward HML→SMB:
  - Table 3: Normal RF p=0.69, MLP p=0.20, LSTM p=0.63 (all non-significant).
  - Line 373-376: "finds no nonlinear improvement for forward HML→SMB."

- The "directional asymmetry" (SMB→HML stronger in TE) is **not explained by the complexity characterization**. It's explained by the quantile Granger showing tail dependence in SMB→HML.

- Lines 380-385 concede: **"The 'purely linear' characterization is therefore fit-dependent; the linear--nonlinear boundary should be treated as exploratory."**
  - If the linear/nonlinear finding is exploratory (Tier 3), then claiming it as a contribution violates the paper's own evidence hierarchy.

**Verdict:** The complexity characterization finds null nonlinear effects. Claiming this as evidence of a "linear-nonlinear boundary" discovery is **misleading**. The actual finding (TE reveals tail dependence) comes from quantile Granger, not the neural net comparison.

---

### 6. **Effect Size Admission Undercuts Economic Significance**
**Severity: CRITICAL**

**Issue:** Lines 118-119 state: "Effect sizes are modest (ΔR² ≈ 2%, Sharpe ratio = -0.07); the contribution is diagnostic, not tradable alpha."

And lines 663-672 acknowledge GARCH(1,1) outperforms for VaR, and Sharpe ratio is **negative** (-0.07).

**Critique:**
- A 2% R² improvement in a univariate predictive regression for factor returns is **within noise** for trading or risk management applications.
- A **negative Sharpe ratio** (line 665) suggests the strategy loses money even before transaction costs.
- This severely limits the economic relevance of the finding. The paper frames it as "diagnostic" (good for understanding when to recalibrate), but:
  1. No evidence is provided that this diagnostic improves real-world risk management (line 671: "could improve risk forecasts" is speculative).
  2. The structural break finding (June 1998) is economically interpretable only if practitioners actually use the regime-conditional model. But if the Sharpe is negative, why would they?

**Verdict:** The small effect size is not itself an overclaim (the paper acknowledges it). But claiming the result has practical value while reporting negative Sharpe ratios is **logically inconsistent**. The contribution should be reframed as "purely diagnostic" or the economic claims should be softened.

---

## MEDIUM ISSUES

### 7. **VIX Validation Doesn't Rule Out Circularity**
**Severity: MEDIUM**

**Issue:** Lines 305-313 claim VIX terciles validate the HMM-based regime findings and confirm "the finding is not a circularity artifact."

**Critique:**
- VIX terciles capturing the structural break is **necessary but not sufficient** to rule out circularity. Here's why:
  - Both HMM and VIX are market-level volatility proxies.
  - If the HML→SMB relationship weakens mechanically whenever markets are volatile (which makes economic sense), both HMM and VIX would detect it.
  - This doesn't prove the HMM's regime classification is adding independent information; it may just be detecting the same volatility signal as VIX.

- A stronger validation would show:
  1. HMM regimes capture information **beyond** VIX (e.g., via higher-order moments like skewness, which HMM encodes through ν).
  2. HMM-based Granger survives when **controlling** for VIX terciles (not reported).
  3. Cross-regime comparisons (e.g., high-VIX Normal vs. low-VIX Crisis) to isolate the regime signal from volatility.

- Line 310-312 shows all three VIX regimes are significant (Normal p=0.028, Elevated p=0.043, Crisis p=0.005), which **contradicts** the HMM finding of significance only in Normal regime. This inconsistency is acknowledged but not resolved.

**Verdict:** VIX validation addresses one circularity concern but introduces another: are we really discovering regime heterogeneity or just regime × volatility interaction?

---

### 8. **Permutation Test (p=0.022) is Weak Evidence for OOS**
**Severity: MEDIUM**

**Issue:** Line 512: "The permutation test (p = 0.022, 50,000 shuffles) demonstrates that the OOS signal is not a circularity artifact."

**Critique:**
- A permutation test shows the signal is not due to **random label shuffling within a regime**. But it doesn't address the other four sensitivity issues the paper itself identifies (lines 504-514):
  1. Does not survive 30-pair Bonferroni (the standard in this paper).
  2. Does not survive 3-regime Bonferroni (HAC p=0.043, which is barely below 0.05 and above 0.0167).
  3. Bootstrap reweighting to training prevalence yields p=0.153 (non-significant).
  4. Sensitive to bandwidth (crosses 0.05 at NW default).
  5. Sensitive to K (null at K=2,4).

- By the paper's own standards (Bonferroni control), the OOS signal is **non-significant**. The permutation test doesn't rescue this; it only shows the signal isn't a statistical artifact, not that it's real.

**Verdict:** The permutation test is overstated as validation. The OOS signal is exploratory/Tier-3 for good reason: it fails multiple robustness checks.

---

### 9. **MOM→SMB Replication Claim Needs Caveat**
**Severity: MEDIUM**

**Issue:** Lines 536-554 argue that MOM→SMB replication (ΔF < 0.1%) validates the protocol for strong signals.

**Critique:**
- MOM→SMB is the **top-ranked OOS pair by F-statistic** (line 203: "MOM→SMB is the top OOS pair"). Selecting the best-performing pair out-of-sample and claiming this proves the protocol is valid is **selection bias**.
- The correct interpretation: "Among 30 pairs, one shows near-perfect in-sample/OOS agreement."
- The paper frames this as validation of the protocol (lines 549-552: "MOM→SMB thus proves the protocol detects genuine OOS confirmation for sufficiently strong signals"), but it's equally consistent with the interpretation that lucky selection found one pair that happened to replicate.
- Pre-registration or a holdout test set would validate this claim. The current evidence is **anecdotal**.

**Verdict:** MOM→SMB replication is interesting but not probative of protocol validity without pre-registration or holdout testing.

---

### 10. **International Replication is Weak and Selective**
**Severity: MEDIUM**

**Issue:** Lines 556-565 present international results (Table 5) as "confirmatory" (Tier-2).

**Critique:**
- Of 4 regions × 3 regimes = 12 tests, only **2 survive Bonferroni** (α/12 = 0.0042):
  - Asia-Pac Crisis (p < 0.001)
  - Developed ex-US Crisis (p < 0.001)

- Both are **out-of-sample Crisis regime**, not the primary finding (in-sample Normal regime).
- Europe and Japan show in-sample Normal significance (p < 0.001) but **OOS nulls**, suggesting the in-sample results don't generalize.
- The paper labels this "confirmatory" but the pattern is **inconsistent**:
  - US: in-sample Normal strong, OOS Elevated weak.
  - Asia-Pac: OOS Crisis strong.
  - Developed ex-US: OOS Crisis strong.
  - Europe/Japan: in-sample Normal strong, OOS null.

- This is **not homogeneous replication**; it's **region-specific and regime-specific**. Labeling it "confirmatory" overstates the evidence.

**Verdict:** International results are mixed and don't support a strong generalization claim. The framing as Tier-2 (confirmatory) is too generous.

---

### 11. **Seven Local Optima Clusters are Concerning, Not Reassuring**
**Severity: MEDIUM**

**Issue:** Lines 635-640 claim robustness across all 7 clusters and prescribe a "decision rule for practitioners."

**Critique:**
- Having **7 local optima clusters** with ΔBICs up to 550 (Cluster 7) indicates the HMM likelihood surface is **poorly identified**.
- While the in-sample Normal result is robust across clusters, this is **not evidence of robustness**; it's evidence the result is **insensitive to the HMM specification**---i.e., the Granger signal is strong enough to overwhelm HMM uncertainty.
- The more concerning finding: Cluster 5 (highest-LL, 90% GFC detection) is ΔBIC=218 below the BIC-optimal. This is a **huge penalty in BIC terms** yet **economically more sensible**.
  - The paper downplays this (lines 180-181, line 639: "also report the highest-LL fit satisfying ≥50% GFC detection as economic sensitivity").
  - But if BIC and economic validity diverge this sharply, it suggests **model mis-specification**, not robustness.

- The decision rule (line 640: "If both agree, the finding is robust") conflates two different model-selection criteria. Practitioners would be uncertain which to trust.

**Verdict:** Local optima plurality suggests model fragility, not robustness. The paper should discuss whether a different HMM structure (e.g., non-Gaussian mixture, hierarchical, switching variance only) would reduce the optima count.

---

### 12. **"Honest Fragility" Claim Undermines Tier-3 Entirely**
**Severity: MEDIUM**

**Issue:** Lines 100-102 position Tier-3 (exploratory OOS) as "reported for transparency, not claimed as validation."

**Critique:**
- Yet the abstract (lines 47-54) devotes significant space to the exploratory OOS result ("A frozen out-of-sample test yields...") and international findings, **as if they contribute to the main claim**.
- The framing of Tier-1/2/3 is an academic courtesy, but in practice, if Tier-3 is so fragile it fails multiple robustness checks, **mentioning it at all in the abstract misleads readers** about the evidence hierarchy.
- A more honest abstract would focus entirely on Tier-1 (in-sample Normal regime) and VIX validation, relegating OOS to the appendix.
- This is not an overclaim about novelty, but rather a **presentation overclaim**: the abstract emphasizes exploratory evidence that the methods section admits is fragile.

**Verdict:** The paper correctly acknowledges fragility but then prominently reports it anyway, creating mixed messages to readers.

---

## LOW ISSUES

### 13. **HML-SMB Economic Prior is Post-Hoc Justification**
**Severity: LOW**

**Issue:** Lines 200-205 acknowledge: "HML--SMB was selected post-hoc from screening 30 in-sample pairs...This focus reflects an economic prior (value-size institutional overlap), not empirical dominance."

**Critique:**
- The "economic prior" is asserted but not developed. What is the institutional mechanism linking HML and SMB beyond the vague claim that they "overlap"?
- Lines 678-679 provide some specificity: "FF25 portfolio overlap analysis finds significance concentrating in small-cap portfolios (ρₛ = 0.35)."
  - But ρₛ = 0.35 is a weak-to-moderate correlation. A 39% attribution to Small/HighBM (line 679) is reassuring but not overwhelming.

**Verdict:** The economic motivation is plausible but thin. This is not an overclaim (the paper is transparent about post-hoc selection) but rather an **underexplored mechanism**. Acceptable for a working paper; should be developed further.

---

### 14. **"Regime Redistribution" Explanation for OOS Weakness is Incomplete**
**Severity: LOW**

**Issue:** Lines 497-503 argue the OOS signal appears in Elevated (not Normal) because post-GFC markets spend more time in Elevated, causing regime redistribution.

**Critique:**
- This is **plausible but not definitive**. Alternative explanations:
  1. The HMM was trained on a different market regime (2008 GFC) than the test period (post-2012 recovery), so frozen parameters may not classify the new regime correctly.
  2. The relationship genuinely shifted from Normal to Elevated regime (i.e., the structural break moved the predictability from a low-volatility to high-volatility state).
  3. The HML→SMB signal is actually **weaker in the absolute sense** (not just redistributed) because volatility dampens the relationship.

- The paper doesn't test Hypothesis 2 (whether the relationship truly shifted) vs. Hypothesis 1 (whether regime classification just drifted). A test would be to refit the HMM on the full 1990-2024 data and compare.

**Verdict:** The regime redistribution explanation is the most parsimonious, but alternatives aren't ruled out. This is exploratory reasoning, appropriately labeled Tier-3.

---

### 15. **Robustness Claims (Line 325-337) Mix Strong and Weak Evidence**
**Severity: LOW**

**Issue:** The "robustness" section (lines 324-337) lists several checks but conflates different types of robustness:

**a) Lag structure (significant at all lags 1-15):**
- This is **strong robustness** (not sensitive to lag choice).

**b) Common drivers (MKT-RF controls, F-p > 0.43):**
- This is **moderate robustness**. The test is whether adding two more factors (MKT, RF) changes the HML→SMB coefficient. But:
  - A 6-factor VAR (lines 713-715) would be under-identified at n ≈ 1,000. So this trivariate control is a weak test of confounding.
  - The paper acknowledges this (lines 713-715) but doesn't reweight the robustness claim.

**c) Regime definition across 7 clusters:**
- This is **false robustness**, as discussed above (Issue #11). It's insensitivity to HMM specification, not true robustness.

**d) Filtered vs. smoothed probabilities (95.9% agreement):**
- This is **fine robustness**. Viterbi hard labels vs. soft labels are close.

**e) Rolling 3-year unconditional Granger:**
- This is a **different method** (rolling window, not regime-conditional), which shows similar episodic patterns. Reassuring but not a robustness check of the regime-conditional finding.

**Verdict:** The robustness section is accurate but presents mixed-strength evidence under a unified heading. Acceptable but somewhat loose.

---

## SUMMARY TABLE

| Issue | Classification | Severity | Main Claim | Verdict |
|-------|---|---|---|---|
| Regime ≠ Quantile heterogeneity | Novelty | **CRITICAL** | Trivially obvious distinction with single empirical example | Overclaimed |
| Structural decay without mechanism | Overclaim | **CRITICAL** | Causal interpretation unsupported; speculation only | Overclaimed |
| "No prior work combines..." | Novelty | **CRITICAL** | Technically true but pedantic; represents minimal innovation | Weakly defended |
| Insufficient differentiation (Psaradakis, Tank, DY) | Novelty | **CRITICAL** | Sits between literatures without advancing any single one | Underdeveloped |
| Complexity characterization | Findings | **CRITICAL** | Claims linear-nonlinear asymmetry but finds null nonlinear improvement | Misleading framing |
| Effect size / economic significance | Overclaim | **CRITICAL** | Sharpe ratio = -0.07; diagnostic value unproven | Logically inconsistent |
| VIX validation doesn't rule out volatility confounding | Evidence | MEDIUM | Partial validation only; doesn't isolate regime from volatility signal | Overstated |
| Permutation test is weak | Evidence | MEDIUM | Shows not random label shuffle; doesn't show signal is real | Overstated |
| MOM→SMB replication is selection bias | Evidence | MEDIUM | Selected best-performing pair; not pre-registered | Anecdotal |
| International replication is mixed | Evidence | MEDIUM | Region/regime-specific results labeled as "confirmatory" | Generalization overclaimed |
| Seven local optima indicate fragility | Methods | MEDIUM | Suggests model mis-specification; claimed as robustness | Misinterpreted |
| Tier-3 prominence in abstract | Presentation | MEDIUM | Exploratory results given substantial abstract real estate | Misleading emphasis |
| Economic prior is thin | Motivation | LOW | Plausible but under-explored institutional mechanism | Underdeveloped |
| Regime redistribution incomplete | Evidence | LOW | Most parsimonious explanation but alternatives not tested | Exploratory reasoning (acceptable) |
| Mixed robustness (weak + strong) | Methods | LOW | Several checks conflated; some are true tests, others aren't | Loose presentation |

---

## OVERALL ASSESSMENT

### Novelty Status: **BORDERLINE REJECT**

**Strengths:**
1. **Tier-1 finding is robust**: The in-sample Normal-regime HML→SMB Granger result (p = 8.75 × 10⁻⁹) is statistically convincing, survives multiple specifications, and is confirmed by VIX terciles.
2. **Structural break is well-documented**: The Quandt-Andrews sup-F (June 1998, p = 1.23 × 10⁻¹³) is credible, and the two-stage pattern (1998 LTCM, 2008 GFC) is coherent.
3. **Transparency about limitations**: The paper clearly labels Tiers 1/2/3 and acknowledges fragility in OOS and complexity characterization.
4. **Comprehensive diagnostics**: The use of transfer entropy, quantile Granger, and multistart sensitivity is thorough.

**Weaknesses:**
1. **Limited methodological novelty**: Combines existing techniques (HMM, Granger, TE, quantile regression) without fundamental innovation.
2. **Key conceptual claim trivializes**: The "regime ≠ quantile heterogeneity" distinction is basic statistics, not a research contribution.
3. **Causal interpretation unsupported**: "Structural decay" language overstates the finding. No mechanism is demonstrated.
4. **Effect sizes are economically negligible**: ΔR² ≈ 2%, Sharpe ratio = -0.07. Practical relevance is unclear.
5. **Differentiation from prior work is weak**: Doesn't clearly distinguish from Psaradakis, Tank, or Diebold-Yilmaz.
6. **OOS evidence is fragile**: Fails multiple robustness checks; properly labeled Tier-3 but given undue prominence.

### Recommendation for ICAIF 2026

**If ICAIF emphasizes**: Novel methodology → **REJECT**. This is engineering (combining existing tools).

**If ICAIF emphasizes**: Novel empirical findings in finance → **WEAK ACCEPT**. The structural break in HML→SMB predictability (1998/2008) is a genuine empirical discovery with robust Tier-1 evidence, though the economic interpretation remains speculative.

**Requested revisions** (if resubmitted):
1. **Remove or hedge** the "regime ≠ quantile heterogeneity" conceptual contribution claim. It's trivial and muddies the paper.
2. **Reframe** "structural decay" as "Granger predictability decline" to avoid causal language without mechanistic evidence.
3. **Develop or test** the deleveraging hypothesis with 13F data or econometric identification strategies. Do not leave mechanism to speculation.
4. **Improve differentiation** from prior regime-switching and connectedness work. Explain what Student-t HMM adds to Psaradakis.
5. **Downgrade international results** from Tier-2 to Tier-3. The pattern is region/regime-specific, not homogeneous replication.
6. **Reduce Tier-3 prominence** in the abstract. One sentence on OOS; focus abstract on Tier-1.
7. **Address trade-offs**: If the finding is diagnostic-only (negative Sharpe), say so clearly. Don't suggest practical risk-management value without evidence.

---

**Status: CONVERGED** (Multiple critical novelty issues identified; recommendations provided.)
