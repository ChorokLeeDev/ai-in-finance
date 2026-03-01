# Round 4 Adversarial Review: Structural Decay of Cross-Factor Predictability

## Verdict: WEAK ACCEPT → BORDERLINE ACCEPT (conditional)

**Confidence: Moderate (65%)**

The paper has made demonstrable progress since Round 3. The international replication provides evidence that structural breaks in cross-factor predictability are a global phenomenon, not a US artifact. The honest VaR comparison (GARCH(1,1) wins) and the transfer entropy findings strengthen the diagnostic protocol narrative. However, the paper still falls *short* of the "one of three" criterion articulated in Round 3 feedback. It partially satisfies (a) international replication, weakly addresses (c) mechanistic modeling via quantile Granger, and definitively *fails* (b) practical VaR deployment success. The core finding—that HML→SMB Granger causality is regime-specific in the US—remains robust, but the out-of-sample evidence remains fragile.

---

## What Improved Since Round 3

### 1. **International Replication: Structural Breaks Are Global**
The paper now provides four non-US Fama-French markets (Developed ex-US, Europe, Japan, Asia-Pacific) with:
- **Positive result**: Two regions show strong frozen OOS effects:
  - Asia-Pacific ex-Japan: Crisis OOS F=39.39, p<0.0001, n=753
  - Developed ex-US: Crisis OOS F=15.85, p=0.0001, n=373
- **Mixed result**: Europe and Japan show strong in-sample significance but OOS nulls
- **Systematic finding**: Quandt-Andrews sup-F detects structural breaks in all 4 regions, confirming the thesis is not US-specific

This is valuable incremental evidence that structural decay of cross-factor predictability is a systematic global phenomenon. The heterogeneity (2/4 OOS replication) is actually *consistent with* the paper's thesis that breaks occur at market-specific dates.

**Assessment**: This partially satisfies criterion (a) from Round 3. However, the heterogeneous replication pattern—where only 50% of international markets replicate the OOS signal—weakens the claim that this is a universal phenomenon. The paper's framing as "replication" is slightly generous; 2/4 success is confirmation for those pairs, but not strong confirmation of universality.

### 2. **Mechanistic Modeling: Quantile Granger Reveals Tail-Dependence Channel**
The paper adds a mechanistic layer through quantile Granger regression (Section 7 / ML Validation):
- **Forward HML→SMB**: Linear across quantiles (Wald p=0.906), homogeneous effect
- **Reverse SMB→HML**: Strongly nonlinear, tail-concentrated (β₀.₉₅=0.212 vs β₀.₅₀=-0.026, Wald p=0.001)
- Transfer entropy asymmetry (z=5.37 reverse vs z=2.45 forward) is now mechanistically explained: tail dependence, not MSE-based predictability

This is genuinely novel diagnostic insight for ICAIF. The combined protocol (linear Granger → quantile regression → transfer entropy) reveals directional asymmetry invisible to standard connectedness measures.

**Assessment**: This substantially addresses criterion (c) "mechanistic modeling," moving from "what happened" (structural break) to "how it works" (tail-concentration channel in reverse direction). This is the paper's strongest round-4 addition.

### 3. **Honest VaR Comparison: Regime-Conditional Model *Fails* Against GARCH(1,1)**
This is perhaps the most intellectually honest finding in the entire submission:
- **GARCH(1,1)**: 1.48% violation rate, Christoffersen p=0.066 (passes)
- **Regime-conditional base**: 3.31% violation rate, Christoffersen p<0.001 (fails)
- **Regime-conditional + Granger adjustment**: 3.31% (no adjustments triggered OOS)

The authors explicitly acknowledge this as a **negative result** and correctly interpret it: the structural break is a *diagnostic insight* (when to re-estimate parameters), not a deployable risk model. This intellectual honesty significantly increases credibility.

**Assessment**: This *definitively fails* criterion (b) "practical VaR deployment success." However, the honest reporting of failure strengthens rather than weakens the paper's contribution. The paper now claims only diagnostic value, which is more defensible than attempting to oversell a failing risk model.

### 4. **Regime-Identification Fragility Transparently Disclosed**
The paper now extensively documents the tension between:
- **Statistical fit (BIC-optimal)**: Assigns 0% of 2008 GFC to Crisis regime
- **Economic sense**: Clusters 5-7 assign 90-100% of GFC to Crisis at ΔB IC=218 cost
- **Local optima taxonomy**: 7 clusters with different crisis assignments, but Elevated OOS robust across all 7

This transparency about the BIC vs. economic validity trade-off significantly strengthens the paper. Rather than hiding this tension, the authors document it, show it's resolvable through sensitivity analysis, and prove the main OOS result is not dependent on regime labeling choice.

**Assessment**: This is exemplary methodological transparency. The 50-seed multistart and local optima reporting set a high standard for HMM-based financial research.

---

## Remaining Weaknesses

### 1. **The "One of Three" Criterion Is Partially But Incompletely Satisfied**

Round 3 feedback stated: "paper needs ONE of: (a) international replication, (b) practical VaR deployment success, (c) 13F mechanistic modeling to reach Strong Accept."

**Current status:**
- **(a) International replication**: ✓ Partially. Structural breaks confirmed globally; OOS replication 50% (2/4 regions). This is genuine progress but not overwhelming confirmation of universality.
- **(b) VaR deployment success**: ✗ Definitively fails. GARCH(1,1) dominates. Authors acknowledge this honestly.
- **(c) Mechanistic modeling**: ✓✓ Substantially addressed via quantile Granger + transfer entropy asymmetry revealing tail-dependence channel. This is the strongest of the three.

The paper satisfies approximately **1.5 of 3** criteria. The combination of partial international replication + strong mechanistic modeling pushes the paper above "Borderline," but the failure on VaR prevents a "Strong Accept."

### 2. **Out-of-Sample Evidence Remains Fragile**

The frozen OOS Elevated-regime result (F=4.65, p=0.003) does *not survive*:
- 30-pair Bonferroni correction (α/30=0.00033, p>0.05)
- Bootstrap prevalence reweighting (permutation p=0.153)
- 50% of bandwidth specifications (Newey-West default p=0.056)
- Two alternative regime counts (null at K=2, 4; significant only at K=3)

While the authors transparently disclose this fragility, the fundamental OOS finding—which would be the most convincing evidence of predictive value—remains weak and conditional on modeling choices. This is honest but limiting for practical utility.

**Counter-position**: The in-sample Normal-regime finding (p=8.75×10⁻⁹) is extremely robust and does not depend on OOS fragility. But practitioners caring about forward-looking value will note that the OOS signal is unreliable.

### 3. **Pair Selection Bias Remains Unresolved**

The paper transparently notes that HML→SMB was selected *post-hoc* from screening 30 pairs. Mitigations:
- **Economic prior**: Institutional crowding of Value/Size positions (credible)
- **Not the strongest regime-heterogeneous pair**: HML→SMB ranks 27/30 by regime heterogeneity (Table 11), while RMW→SMB (rank 1) shows stronger patterns

The international replication partially mitigates this (results replicate across 2/4 regions), but the fact that HML→SMB is not the strongest regime-heterogeneous pattern suggests the selection was guided by economic plausibility rather than statistical evidence.

**Assessment**: This is disclosed but not fully resolved. A skeptical reviewer could argue the paper selected the economically intuitive pair rather than letting the data speak.

### 4. **VaR Mechanism Failure Suggests Limited Practical Impact**

The frozen OOS Granger adjustment triggered **zero times** in 2013-2024 (3,020 trading days). This means:
- The identified cross-factor predictive signal does not reliably anticipate periods when widening VaR would improve coverage
- The statistical relationship (Granger causality) does not translate to decision-relevant risk adjustment
- The paper's risk-model framing is undermined by this complete failure to trigger

The authors frame this as evidence that their contribution is "diagnostic" rather than "prescriptive," which is honest but limits impact for a finance venue.

### 5. **Scale Convention as Degrees of Freedom**

Section 3 (Data) discloses a material issue: frozen OOS results depend on return scale (percentage vs. decimal units):
- Percentage: permutation p=0.022
- Decimal: permutation p=0.063
- Regime agreement: 86.3% with 415 disagreement days at boundaries

The authors pre-specify percentage units as "standard in factor research," but this is a degree of freedom that influenced the results. While disclosed and pre-specified, it remains a modeling choice that favored the reported outcome.

---

## Assessment of New Evidence

### International Replication: Qualified Success

**Strengths:**
- Structural breaks detected in all 4 non-US regions (Quandt-Andrews sup-F test)
- Two regions (50%) produce strong frozen OOS effects with large F-statistics
- Heterogeneous break dates (2003-2014) are consistent with thesis
- Uses identical protocol to US analysis (frozen parameters 1990-2012, test 2013-2024)

**Weaknesses:**
- Only 50% OOS replication rate is modest. If the phenomenon were universal and robust, we'd expect higher replication.
- Small OOS samples (373 for Developed ex-US, 753 for Asia-Pacific), raising Type 1 error concerns despite significance tests
- Europe and Japan show in-sample significance but OOS nulls—suggesting the relationship is fragile even where it's statistically present
- No correction for multiple comparisons across 4 regions. If each region tests 3 regimes, that's 12 tests. None reported.

**Verdict on replication adequacy for ICAIF**: This satisfies the spirit of criterion (a) but not with overwhelming confidence. It shows the phenomenon is not US-specific, but heterogeneous results (50% replication) weaken claims of universality.

### Quantile Granger + Transfer Entropy: Strong Mechanistic Insight

**Strengths:**
- Resolves the directional asymmetry mystery: forward channel is linear (Granger-captured), reverse channel is tail-concentrated (invisible to MSE tests)
- Wald test for heterogeneity across quantiles is proper statistical inference
- Transfer entropy asymmetry (z=5.37 vs 2.45) is now mechanistically understood
- Tail coefficient 8× larger at 95th percentile vs median is an economically meaningful finding

**Weaknesses:**
- The reverse SMB→HML relationship is *weaker* empirically than the forward relationship (Granger p=0.864 for reverse, p=8.75×10⁻⁹ for forward in Normal). Finding a nonlinear mechanism for the weaker channel is less compelling than for the primary finding.
- Quantile regression on Normal regime with n=2,485 is well-powered, but the mechanism applies to the secondary directional channel
- The paper states this "reconciles TE with neural diagnostics," but it's really explaining why TE detects nonlinearity that Granger (rightfully) doesn't capture for the forward direction

**Verdict on mechanistic modeling**: This is genuine progress in characterizing the relationship, though applied to the reverse channel rather than the primary finding. For ICAIF's computational finance focus, the four-model diagnostic protocol + quantile mechanism is solid methodological contribution.

### Honest VaR Failure: Intellectual Integrity Without Practical Success

**The finding:**
GARCH(1,1): 1.48% violation rate (Christoffersen p=0.066, passes)
vs.
Regime-conditional: 3.31% (p<0.001, fails)

**What this means:**
The identified structural break in cross-factor relationships does *not* improve risk model specification. In fact, regime-conditioning *worsens* VaR coverage. The Granger adjustment never triggered because the signal doesn't reliably precede elevated tail risk.

**Credit to authors:** They don't oversell this. They explicitly state:
> "The structural break finding implies that risk model parameters should be re-estimated after major regime shifts---a diagnostic insight rather than a deployable model improvement."

This is intellectually honest. But it means criterion (b) "practical VaR deployment success" is definitively unmet.

---

## Critical Assessment: Has the Paper Crossed the Threshold?

### The Fundamental Question
Is this now a Strong Accept? Or has it merely moved from Weak Accept to Borderline?

**The case for Strong Accept:**
- Extraordinarily robust in-sample finding (p=8.75×10⁻⁹, structural break at specific date)
- Mechanistic insight into directional asymmetry (quantile Granger explaining transfer entropy)
- International replication confirms global phenomenon (2/4 strong OOS, structural breaks in all 4)
- Exceptional transparency about regime-identification tension and OOS fragility
- Diagnostic protocol is reusable across any factor set

**The case against Strong Accept:**
- OOS evidence remains conditional and fragile (fails Bonferroni, sensitive to HAC bandwidth, K-dependent, bootstrap p=0.153)
- VaR comparison fails—the identified relationship has no practical risk-model utility
- International replication is only 50% on OOS (2/4 regions), suggesting phenomenon is region-specific in practice
- Pair selection bias disclosed but unresolved (HML→SMB is economically intuitive, not empirically strongest)
- Effect sizes modest (ΔR²=2% pre-GFC) and economic magnitude speculative ($70M illustrative)

### Why Not Strong Accept?

Round 3 explicitly requested ONE of three criteria for Strong Accept. The paper delivers:
- *Partial* international replication (criterion a)
- *Failed* VaR deployment (criterion b)
- *Solid* mechanistic modeling via quantile Granger (criterion c)

Satisfying 1.5 of 3 criteria is above Borderline but below Strong. The paper has legitimately improved, but not decisively.

The OOS fragility is the critical issue. A Strong Accept would require the OOS evidence to be robust enough to convince a skeptical practitioner that this signal has predictive value. Currently, bootstrap reweighting yields p=0.153, and the signal depends on bandwidth choice. This is simply not strong enough for "Strong Accept" at a conference where deployed algorithms matter.

---

## Final Recommendations

### Tier 1: Essential (for Acceptance)
None. The paper is acceptable as-is given the in-sample robustness.

### Tier 2: Strongly Recommended (to strengthen the paper)
1. **Reframe the VaR comparison as a negative-result paper.** The authors do this well, but could emphasize: "Why regime-conditional models *fail* to improve risk forecasting—and what this teaches us about cross-factor diagnostics." This converts a weakness into a strength.

2. **Examine whether the OOS Elevated signal is an artifact of sample composition or regime expansion.** Bootstrap prevalence reweighting yields p=0.153, suggesting the result is driven by prevalence. Could you test: "What if we uniformly resample OOS Elevated observations to match training-period prevalence?" This would isolate whether the signal is truly predictive or just an artifact of OOS regime frequency.

3. **International replication: address the multiple-comparisons issue.** You test 4 regions × 3 regimes = 12 significance tests. What's the family-wise error rate? Bonferroni correction would suggest (after adjusting): are the two "strong" results still significant?

4. **Resolve the pair-selection issue more directly.** You note HML→SMB ranks 27/30 in regime heterogeneity. Why not run the full four-model diagnostic and mechanistic analysis (quantile Granger, TE) on RMW→SMB (rank 1)? If the quantile Granger asymmetry is specific to HML→SMB, that's honest. If it's generic across high-heterogeneity pairs, that's more convincing.

### Tier 3: Optional (for Camera-Ready)
1. **Add a figure showing the regime-conditional parameter estimates over time.** Rolling estimates 1990-2024 would visualize the structural decay more directly than just break-point tests.

2. **Clarify the return scale convention choice.** You pre-specify percentage units, but disclosure that p=0.022 (percentage) vs p=0.063 (decimal) raises questions. Would it help to report both in the main text alongside the specification rationale?

3. **Add a table ranking all 30 pairs by regime-heterogeneity magnitude.** This would make transparent that HML→SMB (ranked 27) was selected for economic priors, not statistical strength. Readers appreciate this clarity.

---

## Detailed Comments

### Methodological Strengths
1. **50-seed multistart for HMM.** The local optima analysis is exemplary. Showing 7 clusters and their different crisis assignments sets a high bar for regime-identification research. The ΔBIC=218 trade-off between BIC optimality and economic validity is a genuine contribution to methodology.

2. **Frozen OOS design.** Following Welch and Goyal's best practices, you train on 1990-2012, freeze parameters, and test 2013-2024 without refitting. This is the gold standard for OOS validation and properly mitigates regime-identification circularity.

3. **Complexity characterization protocol.** The four-model diagnostic (OLS, RF, MLP, LSTM) with permutation testing is properly executed. The finding that forward HML→SMB is "purely linear" under the primary fit is credible.

4. **Quantile Granger mechanism.** The Wald test (χ²=22.58, p=0.001) for heterogeneity across quantiles is strong. The β₀.₉₅=0.212 finding directly explains the reverse channel's information-theoretic signal.

### Methodological Weaknesses
1. **HAC bandwidth sensitivity.** Table 4 shows OOS Elevated p-values range [0.041, 0.173] across bandwidths. At B=6 (Newey-West default), p=0.056, crossing the 5% threshold. You address this transparently, but it illustrates how close the OOS evidence is to non-significance.

2. **K-sensitivity (regime count).** Table 6 shows the Elevated OOS result is significant only at K=3 (the pre-specified BIC choice). At K=2, p=0.514; at K=4, p=0.056. This non-monotonic pattern suggests the result is somewhat brittle to model-structure choices. You could strengthen this by pre-specifying K or showing the result is robust to K selection on a held-out validation set.

3. **Regime boundary exclusion.** You exclude 0.67% in-sample and 7.4% OOS when lags cross regime boundaries. The OOS exclusion rate (224/3,020 = 7.4%) is substantial. How sensitive are the OOS results to this exclusion? What if you included boundary crossings with soft-label weighting (posterior probabilities)?

4. **Bootstrap prevalence reweighting (p=0.153).** This is your most transparent admission of OOS fragility. The frozen OOS p=0.003 becomes p=0.153 when you resample to training-period prevalence. This suggests the OOS signal is largely an artifact of regime frequency changes post-GFC, not a genuine predictive relationship. You handle this honestly, but it significantly undermines any claim of OOS confirmation.

### Empirical Findings: Robustness Assessment

**In-sample normal-regime HML→SMB (p=8.75×10⁻⁹):**
Robustness across:
- HAC corrections: ✓ (Table 3, p-values range [0.043, 0.056] OOS; in-sample far more significant)
- Lags 1-15: ✓ (Figure 3, all < 0.05 in Normal)
- Trivariate MKT-RF control: ✓ (p < 10⁻⁷, MKT-RF contributes nothing)
- 7 HMM local-optima clusters: ✓ (all clusters show BIC-significant in-sample Normal)

**Conclusion:** In-sample finding is exceptionally robust.

**OOS Elevated (p=0.003):**
Robustness across:
- HAC bandwidth: ✗ (p=0.056 at Newey-West default)
- Bootstrap prevalence: ✗ (p=0.153 when resampled to training prevalence)
- Regime count K: ✗ (null at K=2, 4; only significant at K=3)
- Scale convention: ✗ (p=0.022 percentage vs 0.063 decimal)

**Conclusion:** OOS finding is fragile and conditional on multiple modeling choices.

### Intellectual Honesty Assessment
This paper excels in transparency:
- Pair-selection bias disclosed (selected HML→SMB post-hoc for economic priors)
- OOS fragility fully documented (bootstrap p=0.153, HAC sensitivity, K-dependence)
- VaR failure explicitly reported (GARCH(1,1) superior, regime-conditional worse)
- Regime-identification tension detailed (BIC assigns 0% 2008 to Crisis; Cluster 5 assigns 90%)
- Scale convention degree-of-freedom acknowledged

This intellectual honesty significantly exceeds typical conference standards and increases credibility.

---

## Verdict Summary

### Recommendation: **BORDERLINE ACCEPT** (moving from Weak Accept)

**Confidence: 65% (Moderate)**

The paper has legitimately improved since Round 3. The international replication (2/4 OOS success, structural breaks in all 4 regions) and quantile Granger mechanistic insight are genuine contributions. The honest VaR failure and regime-identification transparency are exemplary.

However, the paper falls short of **Strong Accept** because:
1. **OOS evidence remains fragile:** Bootstrap p=0.153, bandwidth-sensitive, K-dependent, scale-dependent
2. **VaR deployment fails:** GARCH(1,1) dominates; regime-conditional model has zero practical utility
3. **International replication is partial:** 50% OOS replication (2/4 regions), suggesting phenomenon is not universal
4. **Mechanistic modeling applies to secondary channel:** Quantile Granger explains reverse SMB→HML (weaker empirically), not primary HML→SMB

**Why accept rather than weak accept?**
- In-sample HML→SMB finding (p=8.75×10⁻⁹) is extraordinarily robust and reproducible
- Quantile Granger + transfer entropy asymmetry resolves a genuine puzzle in factor dynamics
- International replication confirms structural decay is not US-specific, even if partially
- Methodological transparency (50-seed HMM, local optima taxonomy, OOS fragility disclosure) sets high standard
- Diagnostic protocol is reusable and properly framed (diagnostic, not prescriptive)

**Recommendation for ICAIF Program Committee:**
Accept as Borderline with the understanding that:
- This is primarily a methodological and diagnostic contribution, not a practical risk-model innovation
- The in-sample findings are definitive; the OOS evidence is exploratory
- The paper will likely appeal to researchers building regime-aware factor models, less so to practitioners seeking deployed algorithms
- The international results, while partial, genuinely extend the scope beyond US markets

---

## Actionable Path to Strong Accept (if authors revise)

To reach Strong Accept, authors should:

1. **Strengthen OOS evidence** via either:
   - Pre-registering the K and bandwidth choices, running on new data (2025+) if possible
   - Providing theoretical justification for why bootstrap prevalence reweighting is inappropriate
   - Showing the signal replicates in international markets uniformly (would require new analysis or accepting lower threshold)

2. **Develop a **deployable improvement** to an off-the-shelf risk model** that actually improves coverage or tail hit rate, even modestly. The quantile Granger mechanism suggests tail-aware adjustments might help. Can you use the tail-dependence coefficient (β₀.₉₅) to adaptively adjust VaR widening? Even if GARCH(1,1) still wins overall, showing that regime-aware adjustments improve *tail* metrics (not just coverage) would convert this to practical utility.

3. **International replication uniformity:** If European and Japanese relationships show in-sample but not OOS significance, investigate *why*. Is it sample size? Regime stability post-2012? Different institutional structures? Resolving the heterogeneity would strengthen claims of globality.

These revisions would likely push the paper to Strong Accept by satisfying the "one of three" criteria more decisively.

---

## Final Thought

This paper exemplifies good scientific practice: transparent about limitations, honest about negative results, and scrupulous about methodology. The in-sample structural break is definitive; the OOS evidence is conditional but worth exploring. For a venue like ICAIF, where both rigor and practical impact matter, this represents solid work that advances the field incrementally.

**Verdict: BORDERLINE ACCEPT.** Recommend acceptance with the understanding that this is a methodological diagnostic contribution anchored by a robust in-sample finding, not a complete solution to cross-factor risk modeling.
