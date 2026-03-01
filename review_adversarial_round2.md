# ICAIF 2026 - ROUND 2 RE-REVIEW
## "Structural Decay of Cross-Factor Predictability: Regime-Conditional Granger Analysis with Complexity Characterization"

---

## Updated Assessment

**New Decision: WEAK REJECT → WEAK REJECT (with minor conditional acceptance path)**

The revised paper has substantially improved in **transparency, honesty, and framing**, but the fundamental scientific issues remain largely unresolved. The authors have:

1. **Significantly clarified** the non-significance of the OOS finding in the abstract and contributions
2. **Explicitly acknowledged** methodological limitations and tension between statistical fit and economic validity
3. **Been ruthlessly honest** about fragility (prevalence sensitivity, bandwidth sensitivity, regime-count dependence)
4. **Reframed contributions** to avoid overstating the methodological novelty

However, **no new empirical evidence has been added**, and the core weaknesses persist:
- The main in-sample finding is still a specific result on a curated pair, not a generalizable contribution
- The OOS evidence remains fragile and explicitly non-significant under proper correction
- The linear-forward/nonlinear-reverse asymmetry remains mechanistically unexplained
- Local optima tension is acknowledged but unresolved

**The revision is intellectually honest but does not make the paper scientifically stronger.** It is a "confessional" paper that transparently documents its own limitations—a virtue for careful work but not sufficient for acceptance at a top-tier venue where novel contributions are expected.

---

## What Improved vs. What Didn't

### What Improved Substantially

1. **Abstract clarity (Lines 43-71)**:
   - NOW explicitly states OOS results "do not survive 30-pair Bonferroni correction" (line 65-66)
   - NOW emphasizes in-sample Normal-regime finding ($p = 8.75 \times 10^{-9}$) as primary (line 52)
   - NOW discloses that OOS is "exploratory" with multiple fragility modes (lines 64-70)
   - This is a **major improvement** over a version that would have emphasized OOS without caveats

2. **Contribution reframing (Lines 141-193)**:
   - Contribution 1 now honestly states the pipeline "combines known techniques" and "each component is individually standard" (lines 154-159)
   - Contribution 2 now discloses that "purely linear" characterization is "fit-dependent" (lines 174-178)
   - Contribution 3 explicitly calls the local optima tension "unresolved" and reports both BIC-optimal and economically valid fits (lines 180-192)
   - Gone is the pretense that this is methodological innovation; it's now positioned as a diagnostic protocol

3. **Circularity discussion (Lines 330-342)**:
   - NEW: "Addressing Regime-Identification Circularity" subsection explicitly names and attempts to mitigate the problem
   - Soft-label sensitivity analysis disclosed (weighted Granger yields $p < 10^{-7}$)
   - Frozen OOS validation reframed as "strongest available mitigation" not cure
   - Acknowledges "pair-selection bias remains" (line 342)
   - This is **honest and thorough**, though mitigation remains incomplete

4. **OOS section transparency (Lines 798-865)**:
   - Table 3 (now called Table 2) leads with "does not survive 30-pair Bonferroni" (lines 800-804)
   - **Prevalence sensitivity results now explicit** (lines 822-826): bootstrap $p = 0.153$ with 9.9% of subsamples significant at $p < 0.05$
   - Bandwidth sensitivity disclosed: "p crosses 0.05 at bandwidth ≥ 6" (line 804)
   - K-sensitivity table shows signal only at K=3 (Table in lines 1006-1021)
   - All fragility modes are disclosed upfront, not buried

5. **Local optima section (Lines 1056-1096)**:
   - NEW: Comprehensive Table 1 showing all 7 clusters with BIC, GFC detection, and OOS outcomes
   - Explicitly calls the tension "fundamental challenge" (line 1071)
   - States two-stage criterion is "fundamentally post-hoc" and "cannot be justified as data-driven in the strict sense" (lines 1091-1092)
   - Recommends reporting both BIC-optimal (Cluster 1) and economically valid (Cluster 5) fits
   - **This is mature disclosure**, even if the problem is unresolved

6. **Limitations section (Lines 1103-1134)**:
   - NEW: Dedicated "Scope, Interpretation, and Limitations" subsection
   - Explicitly states "Effect sizes are small ($\Delta R^2 \approx 2\%$) and do not generate trading profits (Sharpe = -0.07)"
   - Honest about VaR application failure: "regime-conditional VaR models exhibit high false-alarm rates (93.2%)"
   - Frames contribution as "diagnostic awareness...rather than direct model deployment" (lines 1120-1122)
   - Discloses missing comparisons (rolling-window baseline, threshold-based regimes, 6-factor VAR)
   - This is **exceptionally transparent**

### What Did NOT Improve (Critical Gaps Remain)

1. **No mechanistic explanation for linear-forward/nonlinear-reverse asymmetry**:
   - Section 3.5 still speculates about "higher-order moments, tail co-movements" (lines 744-757)
   - **No new evidence added** beyond listing candidate hypotheses
   - Still promises "future work" on SHAP, quantile regression, copula methods
   - This remains the paper's most substantial unresolved mystery

2. **No methodological innovation to address OOS fragility**:
   - Authors still rely on the same frozen OOS design that fails significance tests
   - No attempt to apply Bai-Perron sequential break-point testing (criticism #1 from Round 1)
   - No new techniques proposed to improve regime identification beyond the rejected post-hoc criterion
   - Same architectural weaknesses remain

3. **No new empirical evidence on generalizability**:
   - Still limited to HML-SMB pair analysis
   - MOM-SMB mentioned as "top-ranked OOS pair" but relegated to one sentence (line 1028)
   - **No international replication** despite this being listed as "future work"
   - No demonstration that the protocol is "reusable" across other factor pairs

4. **No resolution of BIC vs. economic validity tension**:
   - Tension is now documented and acknowledged but not resolved
   - Both fits are reported, which is honest but leaves practitioners uncertain about which to use
   - No principled framework for choosing between statistical fit and economic interpretability
   - This remains a fundamental methodological flaw

5. **No formal baseline comparisons added**:
   - Still no comparison to rolling-window unconditional Granger
   - Still no comparison to threshold-based volatility regimes
   - Still no comparison to GARCH or standard risk models for VaR
   - Figure 4 (rolling Granger) is qualitative, not formal testing
   - This limits practical relevance claims

---

## Remaining Fatal Flaws

The paper still contains dealbreaker issues:

### 1. **The Main Finding Remains Methodologically Circular**
**Lines 330-342**

Despite acknowledging circularity, the mitigation is incomplete:
- The frozen OOS (lines 338-341) is the "strongest available mitigation," but **it explicitly fails significance testing** (does not survive 30-pair Bonferroni, HAC $p = 0.043$ fails 3-regime Bonferroni)
- Soft-label sensitivity (lines 335-336) shows Normal remains significant but does not prove the regimes are independent of Granger structure—it only shows robustness to hard vs. soft labeling
- **The core problem remains**: HMM uses returns for distributional fit, Granger uses same returns for temporal tests. The claim that these are "functionally distinct" (lines 332-333) is semantic, not substantive.
- The frozen OOS is the only external validation, and **it does not survive multiple-testing correction**

**Implication**: The primary result ($p = 8.75 \times 10^{-9}$ in Normal regime) rests on in-sample regime discovery from the same sample. While the paper is now honest about this, the weakness is unmitigated.

### 2. **OOS Results Do Not Survive Multiple-Testing Correction**
**Lines 800-826**

The paper now explicitly discloses this (good), but it remains a fatal flaw to any claim of external validation:
- Elevated regime: $F$-$p = 0.003$ does NOT survive 30-pair Bonferroni ($\alpha/30 = 0.00033$)
- HAC $p = 0.043$ does NOT survive 3-regime Bonferroni ($\alpha/3 = 0.0167$)
- Bootstrap reweighting to training prevalence: $p = 0.153$ (non-significant)
- Result is driven entirely by regime expansion: Elevated grows from 13.7% → 33.7%
- **K-sensitivity reveals the signal is fragile**: null at K=2, significant at K=3, null at K=4 (lines 1006-1021)

The authors' framing: "We report it for its frozen-parameter design, not statistical significance" (lines 842-843) is intellectually honest but dodges the question: **Why publish an OOS finding that doesn't meet statistical standards?** The answer is that the in-sample result is the real contribution, and OOS is exploratory. But then the focus should be on the in-sample result, not splitting readers' attention.

### 3. **The Local Optima Tension Reveals Methodological Fragility**
**Lines 1056-1096 & Table 1 (lines 1006-1022)**

The 7 local optima clusters expose a fundamental problem that cannot be resolved post-hoc:
- **BIC-optimal fit (Cluster 1)**: Assigns 0% of 2008 GFC to Crisis regime. Statistically best but economically nonsensical.
- **Economically valid fits (Clusters 5-7)**: Assign 90-100% of GFC to Crisis but sacrifice $\Delta$BIC = 218 units.
- **The paper's solution**: Report both. This is honest but admits **the statistical criterion fails**.

The two-stage criterion (≥50% GFC detection, then max LL) is acknowledged as "fundamentally post-hoc" (line 1091-1092). This undermines claims of data-driven methodology. For practitioners, this creates a dilemma: use the statistically optimal fit (Cluster 1, which fails to identify crises) or the economically motivated fit (Cluster 5, which requires hand-tuning)?

**Why this is fatal**: The paper's core selling point is a "reusable diagnostic protocol." But the protocol fails at its first critical decision (regime selection) and offers no principled way to choose. This is a worse problem in Round 2 than Round 1, because the authors have now fully documented it.

### 4. **The "Purely Linear" Complexity Claim Is Fit-Dependent**
**Lines 174-178 & 704-708**

The paper now discloses (improvement!), but this undermines a key finding:
- **Under primary fit (Cluster 1)**: RF shows no significant improvement (p > 0.13), MLP p > 0.20, LSTM p > 0.63
- **Under alternative fit (Cluster 5, seed 42)**: RF shows significant improvement in Elevated ($p = 0.010$) and Crisis ($p = 0.005$)
- Authors acknowledge: "the 'predominantly linear' characterization is fit-dependent" (lines 174-178)

This is problematic because it suggests the nonlinear/linear boundary is unstable across reasonable regime specifications. If the distinction between linear and nonlinear forward prediction depends on which local optimum you choose, the distinction is not robust.

The paper positions this as a transparency point, but it actually weakens the claim that "the forward HML→SMB channel is purely linear." It's linear under the BIC-optimal fit but not under economically sensible fits.

---

## Remaining Major Weaknesses

### 1. **No Mechanistic Understanding of the Transfer Entropy Asymmetry**
**Lines 739-775**

The paper documents an interesting finding (reverse SMB→HML is nonlinear, forward is linear) but leaves it entirely unexplained:
- Three candidate mechanisms are listed (lines 751-763): tail co-movement, volatility-of-volatility transmission, asymmetric response
- **Zero new evidence is provided to test these hypotheses**
- The paper says "Disentangling these hypotheses...is an important direction for future work" (lines 764-767)
- **This is not a contribution; it's an open question marked for future work**

For ICAIF, a mechanism-level insight (e.g., "SMB→HML nonlinearity is driven by tail dependence, confirmed by quantile Granger") would strengthen the paper. The current finding is a statistical observation without explanation.

### 2. **Pair Selection Is Still Circular and Under-Justified**
**Lines 1026-1033**

The paper discloses that MOM→SMB is the top-ranked OOS pair ($F = 20.3$ vs. HML→SMB $F = 9.06$), yet HML→SMB is chosen based on "economic prior" (institutional crowding).

Problems:
- The economic prior is stated **post-hoc** (after screening shows HML→SMB is plausible)
- "Institutional crowding" is documented but the causal link to HML→SMB predictability is not
- Authors acknowledge: "This reduces degrees of freedom but introduces potential prior-driven selection" (line 1030)
- Recommend "pre-registered replication on international data" but don't provide it

**Impact**: The finding is specific to one pair under one regime. Without demonstrating generalizability, the contribution is narrow.

### 3. **Missing Formal Baseline Comparisons**
**Lines 1127-1134**

The limitations section discloses:
- "We do not formally compare regime-conditional Granger to simpler baselines" (line 1128)
- Missing comparisons include: rolling-window Granger, threshold-based regimes, 6-factor VAR
- Figure 4 is "qualitative" only (line 1130)

This is a critical gap. **The paper doesn't prove its methodology is necessary.**
- Could a simple rolling-window test detect the 1998 structural break? Probably.
- Could a volatility-based threshold regime classify 2008 as crisis without HMM fitting? Likely.
- Does regime-conditional Granger outperform rolling regression for practitioners? Unknown.

Without these comparisons, the claim that the protocol is "reusable" and "diagnostic" is unsupported.

### 4. **Economic Magnitude Is Acknowledged as Immaterial**
**Lines 1109-1122**

The paper now honestly states:
- "$\Delta R^2 \approx 2\%$ pre-GFC" (line 1109)
- "Do not generate trading profits (Sharpe = -0.07)" (line 1110)
- "VaR models exhibit high false-alarm rates (93.2%)" (line 1119)
- "Contribution is...diagnostic awareness...rather than direct model deployment" (lines 1120-1122)

This is transparent, but it raises the question: **If the effect is economically immaterial and VaR models fail, what is the practical value?** The paper answers: awareness that cross-factor relationships have changed. But this is a weak value proposition for a computational finance venue.

### 5. **Structural Break Detection Lacks Methodological Rigor**
**Lines 520-565**

The paper uses Quandt-Andrews sup-F to identify June 1998 as the break point ($p = 1.23 \times 10^{-13}$), but:
- **No Bai-Perron sequential testing** to identify number and location of breaks (mentioned but not applied)
- Sup-F is data-snooping-prone when testing all possible break dates
- The break date (June 1998) is identified post-hoc and then used to frame the narrative
- January 2008 Chow test is "theory-motivated" (calendar-based), not data-driven
- **Authors acknowledge**: "10^7 times weaker" than sup-F (line 563), suggesting 2008 is not an independent break

The June 1998 break is interesting historically (LTCM crisis), but the methodology (sup-F without pre-specification) is weaker than Bai-Perron.

---

## Remaining Minor Weaknesses

### 1. **Writing Is More Honest but Still Dense**
- The paper is 1,490 lines (vs. 1,385 in Round 1; net +105 lines)
- Most new material is disclosure of limitations and fragility
- Critical content still buried: the non-significance of OOS is in Table 2 captions (lines 830-835) rather than highlighted
- Related work section (lines 195-250) could be streamlined to make room for clarity

### 2. **Related Work Overstates the Combination's Novelty**
**Lines 240-250**

The paper claims: "No prior work combines regime-conditional Granger causality with complexity characterization and transfer entropy to map the linear–nonlinear boundary..."

This is true but misleading. The combination is not motivated theoretically; it's motivated by wanting to apply all three tools to HML-SMB. The section asks "Why combine these diagnostics?" (line 241) but the answer is weak: "Applied in sequence, these tools expose a directional asymmetry...that no single method captures" (lines 244-246). But this asymmetry is unexplained, so why is it valuable?

### 3. **LSTM Power Analysis Is Insufficient**
**Lines 1145-1149**

The LSTM uses only 100 permutations (vs. 200 for other models). Authors note: "acknowledged as approximate" (line 1149). For a sample of ~1,000 observations per regime, 100 permutations yields SE ≈ 0.07 at $p = 0.5$, making $p$-values at $p \approx 0.63$ unreliable.

Suggestion: Increase to 200 permutations consistently, or report LSTM results with higher uncertainty.

### 4. **Soft-Label Sensitivity Results Are Not Fully Reported**
**Lines 335-336**

The paper states: "soft-label sensitivity using posterior probabilities...yields qualitatively identical conclusions: Normal-regime HML→SMB remains Bonferroni-significant ($p < 10^{-7}$), and Crisis remains null ($p > 0.5$)."

**Missing**: No table showing soft-label results across regimes and pairs. This limits reproducibility and comparison.

### 5. **Scale Convention Still Introduces Arbitrary Degrees of Freedom**
**Lines 262-267 (mentioned)** and **Appendix section on decimal units**

The paper uses percentage-unit convention as primary (0.10 = 0.1%) with decimal units as robustness check. Under decimal units, Elevated OOS is non-significant ($p = 0.063$). The choice of scale affects HMM regime boundaries and thus downstream Granger results.

**Problem**: Pre-specifying a single scale convention would eliminate this degree of freedom. The current approach (report both, use percentage as primary) is exploratory, not confirmatory.

### 6. **K-Sensitivity Analysis Reveals Fragility But Isn't Pre-Specified**
**Lines 1006-1021**

Table 1 shows the OOS result depends on $K = 3$:
- $K = 2$: No significance (p = 0.572)
- $K = 3$: Significant (p = 0.003)
- $K = 4$: No significance (p > 0.056)

The paper selects $K = 3$ via BIC on training data (good), but the fact that results are highly $K$-sensitive suggests the finding is fragile. A more robust approach would be to pre-specify a range of K values and aggregate across them, or to use model-averaging techniques.

### 7. **Boundary Exclusion Rate Discrepancy Suggests Regime Instability**
**Lines 859-862**

In-sample lag-1 exclusion: 0.67% (59/8,817)
OOS lag-1 exclusion: 7.4% (224/3,020)

The 11-fold increase is explained by "more frequent regime transitions when frozen classifier is applied to unseen 2013–2024 period" (line 860-862). But this raises a concern: **If the frozen classifier is that unstable, are the regimes well-defined?**

The paper treats this as a technical note, but it suggests the regimes learned on 1990-2012 don't generalize smoothly to 2013-2024.

---

## New Issues Introduced by Revisions

### 1. **Increased Honesty Creates a "Confessional" Paper**

The revisions have made the paper **brutally transparent about its limitations**. This is intellectually virtuous but creates a new problem: **readers may now question whether the contribution is substantial enough to warrant publication.**

Round 1 criticism pointed out problems. Round 2 revision says: "Yes, those problems exist, and here's how bad they really are."

Examples:
- "We report the OOS result for its frozen-parameter design, not statistical significance" (lines 842-843)
- "This tension is unresolved and reflects a general challenge in latent-state financial modeling" (lines 1083-1084)
- "The purely linear characterization is fit-dependent" (lines 174-178)

**Result**: The paper is more honest but less publishable. It's now difficult for an editor to justify acceptance, because the authors themselves are disclosing that the core validation (OOS) fails statistical tests and the regime identification (local optima) is unresolved.

### 2. **The "Future Work" Appendix Has Grown, Creating a Wish List**

New sections explicitly promise future work on:
- SHAP/quantile regression/copula analysis for TE asymmetry (lines 764-767)
- International replication (line 1162)
- Bai-Perron sequential testing (lines 1166-1167)
- Neural Granger methods (lines 1167-1169)
- End-to-end differentiable causal discovery (lines 1169-1170)

**Problem**: These are presented as implications, not as work done. The paper now reads less like a completed study and more like a research prospectus.

### 3. **The Frozen OOS Section Has Become a Tutorial in Multiple Testing**

Lines 800-843 now contain so much detail on multiple-testing adjustments that it overwhelms the result. The section reads as: "Here's a significant finding (p=0.003), but it fails Bonferroni (doesn't survive α/30), it fails 3-regime Bonferroni (p=0.043 vs. threshold 0.0167), it's driven by prevalence shift (bootstrap p=0.153), and it's K-sensitive (null at K=2,4)."

**Result**: This is the most honest presentation of fragility in a results section, but it's also the clearest indication that the result should not be reported as a finding.

### 4. **Contributions Now Read as Disclaimers**

Contribution 1: "Combines known techniques...each component is individually standard" (lines 154-159) — This is now stated upfront, which is honest, but undermines the claim of contribution.

Contribution 2: "Purely linear characterization is fit-dependent" (lines 174-178) — This caveat is now in the contribution statement itself, making it sound weaker.

Contribution 3: "Tension is unresolved" (lines 1083-1084) — The contribution is now framed as "we have identified a problem, not solved it."

**Result**: The three contributions have morphed from "we discovered X, we developed Y, we created Z" to "we report X (with caveats), we document Y (which is fit-dependent), and we found Z (which is unresolved)."

---

## Specific Actionable Fixes for Remaining Issues

**For EACH issue below, here's the concrete line-level change needed:**

### Issue 1: Mechanistic Explanation of TE Asymmetry
**Lines 739-775**

**Current**:
```
Three candidate mechanisms deserve investigation:
(1) Tail co-movement...
(2) Volatility-of-volatility transmission...
(3) Asymmetric response...
Disentangling these hypotheses...is an important direction for future work.
```

**Fix**: **Add at least one empirical test** before publishing. Minimum options:
- **(Option A)** Quantile Granger test: "We apply quantile Granger regression at 10th, 50th, 90th percentiles. Results show (describe findings) supporting Hypothesis X."
- **(Option B)** Tail copula analysis: "We fit Clayton/Gumbel copulas to SMB-HML pairs in Normal regime. The tail dependence coefficient is (X), suggesting mechanism (Y)."
- **(Option C)** SHAP feature attribution: "SHAP values identify which SMB observations drive nonlinear TE in reverse direction. We find (describe)."

**Why**: Without mechanistic evidence, the TE asymmetry is a statistical curiosity. Adding one test elevates it to a finding.

**Line change**: Replace lines 751-767 with:
```
Three candidate mechanisms deserve investigation:
(1) Tail co-movement: small-cap stocks (SMB long leg) may transmit information to value stocks (HML) primarily through joint tail events.
(2) Volatility-of-volatility transmission: size-factor volatility shocks may predict changes in value-factor higher moments without improving mean forecasts.
(3) Asymmetric response: SMB→HML information flow may be state-dependent within regimes (e.g., active only during drawdowns).

To test Hypothesis 1, we apply quantile Granger causality at the 10th, 50th, and 90th percentiles [ADD RESULTS].
Results show [FINDING], supporting the tail co-movement mechanism.
```

---

### Issue 2: Formal Baseline Comparison
**Lines 1127-1134**

**Current**:
```
We do not formally compare regime-conditional Granger to simpler baselines...
Figure 4 provides a qualitative comparison showing that rolling tests detect episodic peaks but miss the regime-specific structural break.
A formal comparison of regime-conditional vs. rolling vs. threshold-based approaches would strengthen practical recommendations and is left for future work.
```

**Fix**: **Add one formal baseline comparison** before publishing. Minimum:

**Option A (Rolling-window baseline)**:
Add a new Table X comparing rolling 250-day unconditional Granger vs. regime-conditional Granger on the structural break date (June 1998):
```
| Method | Break Date Detected? | p-value | Break Magnitude |
|--------|---------------------|---------|-----------------|
| Rolling Granger (250-day) | July 1998 (off by 4 weeks) | 0.018 | F-change = 15.2 |
| Regime-Conditional Granger | June 1998 | 0.003 | Regime shift to Normal |
```

**Option B (Threshold-based regime)**:
Fit a simple two-regime model: High Volatility (daily factor returns > 90th percentile) vs. Normal. Run Granger separately. Report results.

**Why**: The paper currently claims the regime-conditional protocol is "reusable" and "diagnostic," but provides no evidence that it outperforms simpler alternatives. One formal baseline comparison would establish value.

**Line change**: Replace lines 1128-1134 with:
```
We compare regime-conditional Granger to a simpler rolling-window baseline.
[TABLE X] shows that rolling 250-day unconditional Granger identifies a break at [DATE],
off by [N] weeks from the regime-conditional detection of June 1998.
Both methods identify structural decay, but regime-conditional Granger pinpoints the break date
more precisely, supporting its diagnostic value for practitioners.
```

---

### Issue 3: Address Pair Selection Circularity
**Lines 1026-1033**

**Current**:
```
MOM→SMB—the top-ranked pair by OOS F-statistic (rank 1/30, F = 20.3)—shows the same
regime-conditional pattern (permutation p = 0.010, ΔR² = 1.94%). Our primary pair selection
is guided by an economic prior (HML–SMB institutional crowding) rather than pure empirical strength.
This reduces degrees of freedom but introduces potential prior-driven selection;
pre-registered replication on international data would provide definitive confirmation.
```

**Fix**: Either (A) swap the primary pair to MOM→SMB, justified by empirical strength, or (B) apply Bonferroni correction to the pair-selection step explicitly.

**Option A (Empirically-Driven Primary Pair)**:
```
The empirically strongest OOS pair is MOM→SMB (F = 20.3, p = 0.010), which shows robust
regime-conditional Elevated-regime patterns even under strict Bonferroni correction (α/30 = 0.00033,
permutation p = 0.010). We adopt this as our primary finding. HML→SMB, while also significant
(F = 9.06), is secondary. [JUSTIFY why MOM→SMB is economically plausible too...]
```

**Option B (Explicit Multiple-Testing Adjustment)**:
```
Among 30 directed pairs screened, HML→SMB has raw OOS p = 0.003 (rank 2/30).
To account for pair selection, we apply a secondary 30-pair Bonferroni correction,
yielding adjusted p = 0.090, which fails the 5% threshold.
The selection of HML→SMB over MOM→SMB (top-ranked, p = 0.010) is guided by the economic prior
of institutional crowding, reducing—but not eliminating—multiple-testing concerns.
```

**Why**: Current framing hides the that a stronger empirical result exists (MOM→SMB). Either report it as primary, or explicitly control for the selection step.

**Line change**: Rewrite lines 1026-1033 to either adopt MOM→SMB as primary (Option A) or add explicit pair-selection correction (Option B).

---

### Issue 4: Resolve Local Optima Tension with a Decision Rule
**Lines 1056-1096**

**Current**:
```
The two-stage criterion (maximize GFC detection, then maximize likelihood within that constraint)
is pragmatic and economically motivated, but it is fundamentally post-hoc. We acknowledge this limitation
explicitly: had we relied on BIC alone, we would have obtained a fit inconsistent with financial economics
(zero GFC identification). Our approach prioritizes economic validity over statistical parsimony...
Recommendation for practitioners: we report results under *both* the BIC-optimal (Cluster 1)
and economically valid (Cluster 5) fits throughout.
```

**Fix**: **Pre-specify a principled decision rule** rather than reporting both. Options:

**Option A (Economic Validity as Pre-Specified Criterion)**:
```
We adopt the following decision rule for regime selection: Among all HMM fits,
select the lowest-BIC model that identifies ≥50% of calendar-crisis observations as Crisis states.
This criterion balances statistical fit with economic interpretability.
Under this rule, Cluster 5 is selected (BIC = 75,805, ΔBICvs.optimal = 218, GFC detection = 90%).
All results reported below use Cluster 5.
Sensitivity analysis under Cluster 1 (BIC-optimal) is in Appendix X.
```

**Option B (Information Criterion That Penalizes Missed Crises)**:
```
We modify the BIC to include an economic penalty for missed crisis observations:
BIC_adj = BIC + λ × (Days_GFC_not_in_Crisis), where λ = 10 per missed day.
This yields a criterion that balances statistical fit with crisis detection.
Under this adjusted criterion, Cluster 5 is optimal. [SHOW TABLE showing adjusted BIC for all clusters]
```

**Why**: Reporting both fits without a decision rule places the burden on readers. Practitioners need guidance. A pre-specified rule is more defensible than post-hoc choice.

**Line change**: Replace lines 1087-1096 with one of the above decision rules.

---

### Issue 5: Pre-Specify Scale Convention
**Methodology section (currently lines 262-267, mentioned but not explicit)**

**Current**: Uses percentage-unit convention as primary, decimal as robustness. Under decimal units, OOS Elevated $p = 0.063$ (non-significant).

**Fix**: Either (A) pre-specify a single scale, or (B) pre-register the scale choice and commit to it.

**Option A (Single Scale, Justified)**:
```
We adopt the percentage-unit scale (0.10 = 0.1%) as primary because the Fama-French
factor definitions are published in percentage form [CITE]. All HMM fits, regime boundaries,
and Granger tests use this convention. Decimal-unit sensitivity (0.10 = 10%) is reported
in Appendix X for completeness.
```

**Option B (Pre-Registered Commitment)**:
```
This analysis is registered on [REGISTRY] with pre-specified scale choice: percentage units.
We commit to reporting results under this scale; any sensitivity to alternative scales
is clearly labeled as post-hoc exploration and not reported in the main text.
```

**Why**: The scale convention affects HMM regime boundaries, which affects Granger results. Pre-specification eliminates this degree of freedom.

**Line change**: Add to Methodology section (around line 265):
```
Scale convention: We adopt percentage-unit convention (0.10 = 0.1%) as the primary specification,
pre-specified to eliminate scale selection as a degree of freedom.
```

---

### Issue 6: Disclose Soft-Label Sensitivity Fully
**Lines 335-336**

**Current**:
```
soft-label sensitivity using posterior probabilities...yields qualitatively identical
conclusions: Normal-regime HML→SMB remains Bonferroni-significant (p < 10^-7), and Crisis remains null (p > 0.5).
```

**Fix**: Add a new Table showing soft-label results across all regime-pair combinations:

**New Table X: Soft-Label Sensitivity (Weighted Granger using Posterior Probabilities)**

```
| Direction | Regime | Viterbi p-value | Soft-Label p-value | Agreement |
|-----------|--------|-----------------|-------------------|-----------|
| HML→SMB | Normal | 8.75e-9 | <1e-7 | 99.8% |
| HML→SMB | Elevated | 0.12 | 0.18 | 98.1% |
| HML→SMB | Crisis | 0.73 | 0.65 | 97.2% |
| [Repeat for other key pairs...] | | | |
```

**Why**: Soft-label sensitivity is a key mitigation for circularity. Full reporting enables readers to assess robustness independently.

**Line change**: Replace lines 335-336 with:
```
Soft-label sensitivity: We run weighted Granger regressions using posterior probabilities
P(z_t = k | x_1:T) as observation weights, rather than hard Viterbi assignments.
Results (Table X) show that the Normal-regime HML→SMB finding remains Bonferroni-significant
(p < 1e-7), confirming that the result is robust to regime-labeling method.
Filtered vs. smoothed probability agreement is 95.9% across all observations.
```

---

### Issue 7: Increase LSTM Permutation Count
**Lines 1145-1149**

**Current**:
```
The LSTM uses only 100 permutations (line 1275-1276) with acknowledged approximate p-values.
For a sample of 4,496 observations (Normal regime), 100 permutations is weak (SE ≈ 0.03 at p = 0.5).
Readers cannot assess significance reliably.
```

**Fix**: Increase LSTM permutations to match other models (200) and re-run analysis.

**Line change**:
```
The four-model protocol uses 200 permutations for all models (OLS, RF, MLP, LSTM).
This yields approximate standard error of [CALCULATE for LSTM]. All p-values are reported with
uncertainty bands in Table X.
```

**Why**: Consistency and adequate power. 100 permutations is borderline; 200 is standard practice.

---

### Issue 8: Clarify the "Diagnostic Protocol" as Non-Proprietary
**Contributions section (Lines 141-193)**

**Current**: Claims a "reusable diagnostic protocol" but doesn't package it clearly.

**Fix**: Add an explicit Algorithm Box showing the step-by-step protocol:

**New Algorithm Box (after line 159)**:

```
Algorithm 1: Regime-Conditional Granger Diagnostic Protocol

Input: Multivariate returns (X_t)_{t=1}^T, target pair (i, j), regime count K
Output: Regime-conditional Granger causality tests, complexity characterization

Steps:
1. Fit Student-t HMM with K regimes using 50 random starts.
   - Select regime count K via BIC on training data (1990-2012).
   - Retain top 7 local optima for sensitivity analysis.

2. Extract regime labels ẑ_t using filtered probabilities (no future information).

3. For each regime k ∈ {1, ..., K}:
   a. Subset observations to T_k = {t : ẑ_t = k}
   b. Select lags 1-L via BIC for X_{i,t} regressing on X_{j,t-ℓ}
   c. Run Granger test: H_0: X_{j,t-ℓ} ⊥ X_{i,t} | {X_{i,t-ℓ}}
   d. Report F-statistic, p-value (HAC, Andrews bandwidth)

4. Apply Bonferroni correction: α' = α / (K × num_pairs)

5. [If OOS validation desired:]
   a. Freeze HMM parameters from training period
   b. Re-classify test period without refitting
   c. Repeat Granger tests on held-out regimes
   d. Apply 30-pair Bonferroni correction to account for pair screening

6. Complexity characterization:
   a. Fit four predictive models (OLS, RF, MLP, LSTM) on regime-specific data
   b. Test whether nonlinear models significantly improve prediction (MSE, permutation test)
   c. Compute transfer entropy (Frenzel-Pompe kNN, k=5, 200 permutations)
   d. Report whether forward/reverse channels are linear or nonlinear
```

**Why**: Making the protocol explicit enables reproducibility and reuse.

**Line change**: Add after line 159:
```
The protocol is formalized in Algorithm 1, enabling practitioners to apply it to any factor set
or multivariate financial time series where latent-state structure may govern predictive relationships.
```

---

### Issue 9: Explicitly State Which Results Are Primary vs. Exploratory
**Abstract (Lines 43-71)**

**Current**: The abstract nicely separates primary (in-sample Normal) from exploratory (OOS Elevated), but the main text sometimes blurs this distinction.

**Fix**: Add a box or explicit section delineating primary vs. exploratory findings:

**New section header after Conclusion (add before Funding section, around line 1155)**:

```
Summary of Primary vs. Exploratory Findings

Primary Findings (Robust, In-Sample, Held Across Sensitivity Checks):
- HML→SMB Granger predictability is regime-specific: Bonferroni-significant in Normal pre-crisis
  (p = 8.75e-9, ΔR² = 2.06%), absent post-2008 (95% CI [-0.049, 0.073])
- Structural break at June 1998 (sup-F p = 1.23e-13)
- Finding robust across: 7 HMM local optima clusters, HAC bandwidth choices, lags 1-15,
  MKT-RF control, soft-label sensitivity, trivariate specification
- Forward channel is linear; reverse channel exhibits stronger nonlinear information flow

Exploratory Findings (Frozen OOS, Does Not Survive Multiple-Testing Correction):
- OOS Elevated regime shows raw F-p = 0.003, but does NOT survive 30-pair Bonferroni (α/30 = 0.00033)
- Does NOT survive 3-regime Bonferroni (HAC p = 0.043 > α/3 = 0.0167)
- Bootstrap reweighting to training prevalence: p = 0.153 (non-significant)
- K-sensitive: null at K=2, significant at K=3, null at K=4
- Reported for frozen-parameter design value, not statistical significance

Unresolved Issues:
- Mechanistic source of SMB→HML nonlinearity not identified
- Local optima tension (BIC vs. economic validity) remains unresolved; both fits reported
- VaR application fails (93.2% false-alarm rate)
```

**Why**: Clarity for readers about which findings to trust and which are preliminary.

**Line change**: Add new section before funding.

---

### Issue 10: Show That Protocol Improvements Are Possible
**Limitations section (Lines 1125-1134)**

**Current**: Lists missing comparisons as future work.

**Fix**: **Add one concrete improvement** to the methodology and show its impact:

**Option A (Add Missing Confounder)**:
```
We extend the baseline Granger specification to include a fourth confounder: RMW (profitability).
Results are reported in Table X (Appendix). Inclusion of RMW does not materially change the main
findings (Normal HML→SMB p remains < 1e-6; Crisis remains null), confirming that omitted variable
bias is not driving results.
```

**Option B (Add Bai-Perron Multiple-Break Test)**:
```
We apply Bai-Perron sequential testing to identify potential multiple breaks in HML→SMB Granger
causality over the full sample period. Results identify breaks at [DATES], with June 1998 confirmed
as the primary break and [SECONDARY BREAKS if any]. This validates the sup-F finding and provides
more rigorous break-point identification.
```

**Why**: Showing at least one methodological improvement demonstrates commitment to robustness beyond disclosure of limitations.

**Line change**: Replace lines 1128-1134 with one of the above improvements.

---

## What Would Push This to Strong Accept

To move from WEAK REJECT to ACCEPT (or STRONG ACCEPT), the paper would need:

### 1. **Mechanistic Explanation for TE Asymmetry** (CRITICAL)
Add one hypothesis test (quantile Granger, copula analysis, SHAP attribution, or regime-dependent drift test) showing why SMB→HML is nonlinear while HML→SMB is linear. This converts a curiosity into a finding.

**Impact**: Elevates the transfer entropy analysis from exploratory to contributory.

---

### 2. **Generalizability Evidence on At Least 3 Factor Pairs**
Demonstrate that the regime-conditional Granger pattern holds for other pairs beyond HML-SMB and MOM-SMB (currently relegated to one sentence). Either:
- Show regime-conditional results for RMW→HML, CMA→SMB, and MOM→RMW
- Or apply the protocol to international factor data (Asness/Moskowitz international value/momentum)

**Impact**: Converts "specific empirical result on a curated pair" to "generalizable finding across factor networks."

---

### 3. **Formal Baseline Comparison Showing Superiority**
Add a Table showing that regime-conditional Granger outperforms rolling-window Granger or threshold-based regimes on structural break detection accuracy. Include accuracy metrics (break date detection, timing lead/lag relative to calendar crisis).

**Impact**: Establishes that the methodology is not just novel but superior to simpler alternatives.

---

### 4. **Pre-Registration Evidence or International Replication**
Provide evidence of pre-registered confirmation on held-out data (different country, different time period, or different factor set). If not yet done, commit to a specific replication design and timeline.

**Impact**: Converts confirmatory evidence from missing to present, addressing the frozen OOS fragility.

---

### 5. **Practical Improvement to Risk Model or VaR Specification**
Show that regime-conditional Granger improves an industry-standard risk model (Barra, Axioma, or QuantConnect) in out-of-sample risk prediction. Currently VaR models based on the framework have 93.2% false-alarm rates. Either:
- Fix the VaR model to improve false-alarm rates
- Or show regime-conditional Granger improves factor timing decisions (reduce realized tail risk vs. static allocation)

**Impact**: Converts "diagnostic awareness" from abstract to concrete, demonstrating practical value.

---

## Summary Table: Issue Severity and Fixes

| Issue | Severity | Fix | Line(s) | Impact |
|-------|----------|-----|---------|--------|
| TE asymmetry unexplained | Fatal | Add quantile Granger test | 739-775 | Mechanism unknown |
| OOS non-significant but reported | Fatal | Consider removing OOS or reframing | 800-843 | Validation fails |
| Local optima unresolved | Fatal | Pre-specify decision rule (BIC + GFC criterion) | 1056-1096 | Method unreliable |
| "Purely linear" fit-dependent | Major | Disclose fit-dependence in contributions (now done) | 174-178 | Weakens claim |
| No baseline comparison | Major | Add rolling-window or threshold comparison | 1127-1134 | Necessity unclear |
| Pair selection circular | Major | Either use MOM→SMB or apply pair-selection correction | 1026-1033 | Generalizability weak |
| Missing soft-label table | Minor | Add Table X with soft-label results | 335-336 | Reproducibility limited |
| LSTM power insufficient | Minor | Increase to 200 permutations | 1145-1149 | p-values unreliable |
| Scale convention free parameter | Minor | Pre-specify scale upfront | ~265 | DegreeOfFreedom |
| Structural break methodology weak | Minor | Add Bai-Perron testing | 520-565 | Rigor gap |

---

## Conclusion and Recommendation

### What Changed Well

The revised paper is **substantially more honest and transparent** about its limitations. The authors have:
1. Reframed contributions to avoid overstating methodological novelty
2. Moved non-significant OOS results into clearer prominence
3. Documented the local optima tension and its unresolvability
4. Added explicit limitations section
5. Disclosed that effect sizes are immaterial and VaR application fails

### What Remains Problematic

Despite transparency improvements, **no new empirical evidence has been added to address fundamental weaknesses**:
1. The main finding is still circular (regimes from same returns used for Granger)
2. OOS validation still fails statistical tests
3. The TE asymmetry remains mechanistically unexplained
4. The local optima tension is acknowledged but unresolved
5. No evidence of generalizability beyond the single HML-SMB pair

### Verdict

**WEAK REJECT → WEAK REJECT (unchanged decision)**

The paper reads as a masterclass in transparent disclosure of limitations, which is commendable. However, **transparency about problems is not the same as solving problems.** The authors have documented what is broken (circular regime identification, failed OOS validation, unexplained mechanism) without fixing it.

For ICAIF 2026, the venue expects either:
- Methodological innovation (this paper lacks it; the protocol is a sequence of known techniques)
- Theoretical insight (the TE asymmetry is unexplained; the local optima tension is unresolved)
- Practical utility (effect sizes are immaterial; VaR models fail; trading signals are negative)
- Empirical generalizability (limited to one pair, one regime)

The paper excels at transparency and documentation of fragility, but falls short on contribution. It would be an excellent technical note or working paper, but is below the bar for acceptance at a top-tier conference.

### Path to Acceptance

If the authors wish to revise for Round 3, prioritize these three concrete improvements:

1. **Explain the TE asymmetry mechanistically** (add quantile Granger or copula analysis)
2. **Demonstrate generalizability** (show regime-conditional pattern on 3+ additional factor pairs or international data)
3. **Prove superiority over baselines** (show regime-conditional Granger outperforms rolling-window approach)

Any two of these three would likely push the paper to ACCEPT. All three would make it STRONG ACCEPT.

Without new empirical contributions, the current revision—while intellectually honest—remains a careful analysis of a curated example rather than a scientific advance.

---

**Review Completed: 2026-03-01**
