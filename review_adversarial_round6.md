# ICAIF 2026 Review: "Structural Decay of Cross-Factor Predictability"

**Reviewer: Professor Chen, MIT Sloan**

**Decision: WEAK REJECT**

---

## EXECUTIVE SUMMARY

This paper documents the erosion of Granger predictability from HML to SMB across three latent regimes (1990–2024), identifying a primary structural break in June 1998 via Quandt-Andrews testing. The main in-sample finding—Normal-regime HML→SMB significance at p = 8.75×10^−9 with ΔR² = 2.06%—is statistically robust. However, the paper exhibits three fatal interconnected problems: (1) the frozen OOS result, marketed as "primary validation," does not survive multiple-testing correction and depends critically on post-GFC regime prevalence changes (bootstrap p = 0.153), (2) the regime-identification circularity cannot be fully eliminated and remains the paper's central methodological vulnerability, and (3) the economic contribution is diagnostically thin—the paper assembles existing techniques into a "protocol" without generating actionable insights beyond "cross-factor relationships sometimes change." The transfer entropy asymmetry and quantile regression mechanistic findings are the paper's only novel analytical contributions, but they apply exclusively to one pair and do not generalize. For ICAIF 2026, this reads as a well-executed empirical exercise that fails the "so what?" test.

---

## 1. THE "SO WHAT?" TEST — MAJOR FLAW

**Classification: MAJOR**

The paper's core claim is structural decay of HML→SMB predictability, centered on a break at June 1998. But what is the actual insight?

### What We Learn:
- HML Granger-predicts SMB in the pre-crisis Normal regime (p = 8.75×10^−9, robust to HAC)
- The relationship weakened around 1998 and further post-2008
- Post-2008 coefficient is consistent with zero (95% CI [-0.049, 0.073])

### What We Don't Learn:
- **Why this matters economically.** The authors themselves admit effect sizes are small (ΔR² = 2%) and do not generate trading profits (Sharpe = -0.07). The VaR comparison (Section~app:var) shows GARCH(1,1) outperforms the regime-conditional model (violation rate 1.48% vs. 3.31%), negating the claimed risk-modeling contribution. The authors spin this as "diagnostic awareness" rather than "deployable improvement"—which is honest but undermines the paper's practical relevance.

- **Why this is not obvious from prior work.** The authors cite ~20 papers on Markov-switching, Granger causality, and factor dynamics separately. None combine regime-conditional Granger with transfer entropy. But the *combination* yields what insight? The directional asymmetry (linear forward, nonlinear reverse) applies to HML–SMB alone. Quantile Granger on the top regime-heterogeneous pairs (RMW→SMB, MKT→SMB, SMB→MKT) all show linear structure (Wald p = 0.527 to 0.869), so the nonlinear tail mechanism is pair-specific, not a generalizable pattern.

- **Why the break occurred.** The paper offers a post-hoc "deleveraging cascade" story (Appendix~app:mechanism) but immediately disclaims it: "unverified mechanism...13F holdings-level verification remains for future work." FF25 portfolio overlap (ρ_s = 0.35, p = 0.046) is "consistent" with deleveraging but 6 stress events validate the pattern in only 2/6 cases. This is not explanation; it is speculation.

### The Real Problem:
The paper's contribution is **diagnostic protocol assembly, not discovery.** Algorithm 1 chains together (1) Student-t HMM regime discovery (Bulla 2011), (2) Granger testing (Psaradakis et al. 2005), (3) complexity diagnostics (Tank et al. 2022), and (4) transfer entropy (Schreiber 2000). Each component is cited as prior work. The paper's framing—"applies a regime-conditional Granger analysis protocol"—is honest but damning: this is a literature review implemented in code, applied to one factor pair. The authors write: "The individual techniques are standard; the diagnostic value lies in their joint application." This is precisely the problem: joint application does not equal methodological novelty.

For a top-tier venue, the bar is: does this paper tell us something about financial markets (or finance methods) that we didn't already know? The answer is: (a) HML and SMB became less predictively linked post-1998 (incremental update to known instability of factor relationships), (b) regime-conditional analysis reveals structure that rolling Granger misses (expected, given latent-state models outperform unconditional baselines in finance), and (c) one specific pair (SMB→HML) shows tail dependence that linear Granger misses (pair-specific, non-generalizable).

None of these rise to the level of a novel empirical or methodological contribution worthy of ICAIF.

---

## 2. STATISTICAL RIGOR — FATAL FLAW

**Classification: FATAL**

### 2a. The Frozen OOS Result Does Not Validate the In-Sample Finding

The paper positions the frozen OOS (2013–2024) as "primary validation" (abstract, section~sec:frozen_oos). But read carefully:

- **Raw F-p = 0.003** in the Elevated regime, but:
- **Does not survive 30-pair Bonferroni** (α/30 = 0.00033); even 3-regime Bonferroni (α/3 = 0.0167) shows HAC p = 0.043, which does not survive.
- **Bootstrap prevalence reweighting yields median p = 0.153**, indicating the signal is driven by regime prevalence expansion (13.7%→33.7%), not genuine out-of-sample replication of the in-sample pattern.
- **Depends on K=3 regime count:** the pattern is null at K=2 (p = 0.514, n=1,917) and null at K=4 (p = 0.056). This non-monotonic K-sensitivity indicates the OOS finding is an artifact of the three-regime specification.
- **HAC bandwidth sensitive:** p ranges from 0.041 (bandwidth=1) to 0.173 (bandwidth=30), crossing the 0.05 threshold at bandwidth ≥ 6 (NW default).

The section~robustness claims are written defensively—"exploratory evidence with disclosed fragility"—but the bottom line is stark: **the OOS result does not confirm the in-sample Normal-regime finding.** The in-sample result is robust; the OOS is fragile and reversed in regime (Normal pre-2008 vs. Elevated 2013–2024), suggesting the Elevated signal is epiphenomenal to post-GFC regime redistribution, not a replication of the causal mechanism.

This is a critical failure. The paper cannot point to OOS validation; it must rely solely on in-sample evidence for a relationship that existed 16+ years ago.

### 2b. Regime-Identification Circularity: Insufficiently Mitigated

The paper acknowledges circularity (section~sec:addressing): "In-sample Granger tests condition on regime labels from the same returns, creating circularity that cannot be fully eliminated without external instruments."

**Proposed mitigations:**

1. **Argument 1: "Distributional properties vs. temporal dynamics are distinct."** This is semantic sophistry. The HMM uses means, covariances, and tail shape; Granger tests use lagged predictive content. But both are derived from the same return vector, and the HMM's emission probabilities depend on contemporaneous returns, which correlate with past returns via volatility clustering and momentum. The claim that these are "related but distinct statistical features" does not eliminate the information leakage.

2. **Argument 2: "Soft-label sensitivity with posterior probabilities yields identical conclusions."** Weighted Granger regression with posterior probabilities (P(z_t=k|x_{1:T})) instead of hard Viterbi labels still uses regime labels derived from the same returns. The fact that results are "qualitatively identical" (p<10^{-7} vs. p=8.75×10^{−9}) is unsurprising; it does not validate the approach.

3. **Argument 3: "Frozen OOS design provides strongest available mitigation."** But the frozen OOS does not replicate the in-sample finding (see §2a above), so this "strongest available mitigation" actually undermines the paper's claims.

**Permutation test (Table~sec:frozen_oos).** The paper reports a permutation test (shuffles HML labels within the Elevated regime) as "co-primary significance measure." Under percentage units (n=953), p=0.022; under decimal units (n=836), p=0.063. This is the correct statistical approach for addressing circularity, but:
- The permutation p-values are larger than parametric ones (0.022 vs. 0.003 for the main result), acknowledging the "conservative nature of this circularity-robust approach."
- For the frozen OOS Elevated result, the paper does not report a permutation test (only for the in-sample result in section~sec:frozen_oos).
- The scale-unit dependence (p = 0.022 vs. 0.063 for the same data under different conventions) reveals "scale choice is a material degree of freedom for HMM-based analyses," yet the authors chose percentage units "pre-specified because it is the standard reporting format." This pre-specification is reasonable but still a degree of freedom exercised post-hoc after observing it favors the hypothesis.

**Verdict:** Circularity remains unresolved. The permutation test and frozen OOS provide partial mitigation, but the core problem—regime labels are derived from the same returns used in Granger tests—cannot be eliminated within this framework. This is a structural limitation of two-stage latent-variable inference, not unique to this paper, but it means **the in-sample Normal-regime finding cannot be interpreted as causal evidence; it is at best predictive evidence subject to confounding by latent distributional shifts.**

### 2c. Multiple Testing: Severe Overcorrection and Undercorrection

The paper's multiple-testing framework is contradictory:

- **In-sample:** Bonferroni across 30 directed factor pairs (α/30 = 0.00033) is stringent. Normal-regime HML→SMB survives (p = 8.75×10^−9). ✓ Good.

- **OOS Elevated result:** Should inherit the 30-pair multiplicity burden because HML–SMB was identified from screening all 30 pairs in-sample. The paper acknowledges this (section~sec:pairselection) and reports three correction frameworks:
  1. Primary: 30-pair Bonferroni (α/30 = 0.00033): result fails (HAC p = 0.043, F-p = 0.003 both fail).
  2. Sensitivity: 3-regime Bonferroni (α/3 = 0.0167): HAC p = 0.043 fails; F-p = 0.003 passes.
  3. Uncorrected (α = 0.05): both pass.

The authors then claim "the OOS Elevated result is valued for its frozen parameters and economic prior, not statistical significance under any correction framework." This is intellectual dishonesty. If the result does not survive the primary correction framework (30-pair Bonferroni), it should not be claimed as significant. The "economic prior" for HML–SMB does not erase the multiple-testing burden; it only justifies why HML–SMB was selected for focused attention, not why post-hoc OOS evidence should bypass corrections.

**Furthermore:** the paper reports "30-pair FDR correction: no pairs survive" (section~sec:pairselection), yet claims HML→SMB is the "second ranked pair by F-statistic" (F=9.06) and "top-ranked is MOM→SMB (F=20.3)." If MOM→SMB is stronger in the OOS Elevated regime, why does the paper not report results for that pair? **This is selective reporting.** The paper focused on HML–SMB ex-post, not ex-ante, yet uses the in-sample screening as justification.

### 2d. Local Optima Proliferation

The 50-seed HMM multistart reveals 7 distinct local optima clusters with fundamentally different crisis-regime assignments:
- Cluster 1 (BIC-optimal, seed 28): 0% of 2008 assigned to Crisis.
- Cluster 5 (economically valid): 90% of 2008 assigned to Crisis.
- ΔBic between them = 218 units.

The paper acknowledges this as "a fundamental tension: the BIC-optimal fit assigns 0% of 2008 GFC days to Crisis, while economically sensible fits assign 90–100% at a ΔBic=218 cost." The resolution: "We report results under both the BIC-optimal (Cluster 1, primary) and economically valid (Cluster 5, sensitivity) fits."

**This is not a resolution; it is an admission that the regime discovery is unstable.** If BIC-optimal and economically valid fits differ by 218 units in model selection criterion yet yield opposite conclusions about crisis regime detection, the latent-state discovery process has failed. The authors try to finesse this by claiming the structural break finding is "time-indexed (June 1998, January 2008), not regime-indexed" and thus robust across clusters. But this conflates two distinct findings:
1. **Time-indexed break (Quandt-Andrews sup-F at June 1998):** this is robust across clusters because it uses a data-driven test that does not depend on regime assignments.
2. **Regime-conditional Granger significance:** this is cluster-dependent. The Normal-regime result holds across clusters, but the Crisis-regime result is seed-dependent.

The paper should have either (a) pre-specified a regime selection criterion (e.g., "maximize GFC detection, then maximize likelihood") or (b) reported all results across all clusters as equally valid. Instead, it cherry-picks Cluster 1 as "primary" because it has the lowest BIC, despite failing basic economic validity, then uses Cluster 5 for sensitivity checks. This is p-hacking via regime selection.

---

## 3. ECONOMIC SUBSTANCE — MAJOR FLAW

**Classification: MAJOR**

### 3a. Trading Does Not Work

Appendix~app:trading reports a simple strategy (long SMB when 9-day cumulative HML is positive during Crisis, short otherwise) yields Sharpe = -0.07 vs. buy-and-hold +0.06. The authors write: "This intentionally simple rule (no optimization, no transaction costs) confirms that statistical predictability ≠ economic predictability; it does not rule out profitability under more sophisticated implementations."

**Translation:** the paper's core finding—statistical significance at p<10^−8—does not translate to any implementable alpha. The authors admit this upfront, which is honest, but it annihilates the economic motivation. If a relationship this statistically robust does not even generate positive Sharpe under the simplest possible implementation, something is wrong. Either:
1. The predictability is too small in magnitude (ΔR² = 2%) to overcome transaction costs and risk.
2. The predictability is a statistical artifact of look-ahead bias or specification mining.
3. The relationship has no causal economic content.

The authors opt for (1), but (2) and (3) remain plausible.

### 3b. VaR Model Comparison Fails

The regime-conditional VaR model (section~app:var) achieves a violation rate of 3.31% (vs. target 1%), failing Christoffersen coverage test (p<0.001). GARCH(1,1) achieves 1.48% (passes). The Granger adjustment mechanism (widening VaR when cross-factor risk is signaled) "failed to trigger during the OOS period, indicating that the cross-factor predictive signal does not translate into improved VaR timing."

**This is damning.** The regime-conditional framework is supposed to identify when cross-factor relationships shift, improving risk models. But empirically it worsens VaR coverage. The authors frame this as "diagnostic" rather than "deployable," but if it does not help practitioners manage risk, what is its value?

### 3c. Economic Mechanism Is Speculative and Low-Power

The deleveraging cascade hypothesis (Appendix~app:mechanism) predicts that HML→SMB should be strongest for portfolios loading on both factors (Small/HighBM). The FF25 overlap analysis (ρ_s = 0.35, p = 0.046) is "consistent" but barely significant after permutation testing. Moreover:

- Event-based validation (Appendix~app:events) shows 2 of 6 stress events match the expected pattern, 4/6 directionally correct (binomial p ≈ 0.11, not significant). The 2022 Rate Hikes show **reversed** HML→SMB dynamics, suggesting the mechanism is not robust even as a hypothesis.
- The authors note: "Two exceptions show reversed dynamics...a mechanism distinct from the liquidity-driven deleveraging channel, where interest-rate sensitivity may reverse the usual HML→SMB ordering." This post-hoc exception-handling reveals the mechanism is ad hoc, not principled.

**Verdict:** There is no credible economic mechanism documented. The paper has a statistical relationship and a plausible-sounding story, but no causal evidence.

---

## 4. METHODOLOGICAL NOVELTY — MODERATE FLAW

**Classification: MAJOR (relative to venue expectations)**

The paper's claimed methodological contribution is Algorithm 1, a "regime-conditional Granger diagnostic protocol" combining:
1. Student-t HMM regime discovery (Bulla 2011)
2. Multi-seed local-optima sensitivity
3. Per-regime Granger testing with Bonferroni correction
4. Frozen-parameter OOS validation
5. Complexity diagnostic (OLS, RF, MLP, LSTM)
6. Transfer entropy analysis
7. Quantile Granger regression

Each component is prior work. The paper's claim: "The individual techniques are standard; the diagnostic value lies in their joint application."

**This is not novel.** Combining existing methods into a workflow is not methodological innovation. It is engineering. The complexity diagnostic (Tank et al. 2022, Figure~fig:ml_diagnostic) and transfer entropy asymmetry (Schreiber 2000, Figure~fig:te_asymmetry) are the only moments of novelty:

- **TE asymmetry finding:** Forward HML→SMB is linear (LSTM fails to improve MSE), but reverse SMB→HML is nonlinear (TE z=5.37 vs. forward z=2.45). This is interesting within-paper.

- **Mechanism:** Quantile Granger shows the reverse channel operates through tail dependence (upper-tail coefficient 8× larger than median, Wald p=0.001), while forward is homogeneous across quantiles (Wald p=0.906).

But this mechanism applies exclusively to HML–SMB. Applied to the top regime-heterogeneous pairs (RMW→SMB, MKT→SMB, SMB→MKT), quantile Granger shows strictly linear structure (Wald p = 0.527 to 0.869). The authors acknowledge: "Regime heterogeneity and quantile heterogeneity are thus distinct phenomena...the nonlinear tail channel is specific to SMB→HML, not a generic feature of regime-heterogeneous factor pairs."

So the methodological contribution—demonstrating that quantile regression can reveal tail mechanisms invisible to linear Granger—applies to one pair. This is not generalizable. It is an observation about HML–SMB's specific dynamics.

**Verdict:** No truly novel method. The paper implements existing techniques in sequence. The TE asymmetry and quantile mechanism are observations about one pair, not methodological breakthroughs.

---

## 5. WRITING AND PRESENTATION — COSMETIC FLAW

**Classification: COSMETIC**

The paper is generally well-written and transparent about limitations. However:

- **Length:** The main text is 27 pages (with appendix totaling ~40 pages), excessive for the conceptual contribution. The authors hedge extensively: regime-identification circularity, OOS fragility, trading failure, VaR failure, mechanism uncertainty, local optima tension. These are appropriate caveats but overwhelm the narrative. A tighter paper would excise or condense these sections rather than air them as separate subsections.

- **Jargon-heavy:** Terms like "leakage-safe," "complexity characterization," "directional asymmetry," and "regime redistribution" are used repeatedly without adding clarity. For example, "regime redistribution" (p. 70, 997) means "the post-GFC period had higher Elevated regime prevalence," but the neologism obscures rather than clarifies.

- **Defensive tone:** Section headings like "OOS Re-Emergence: Exploratory Evidence with Disclosed Fragility" preemptively concede the point. This is honest but suggests the authors knew their OOS result was weak before submitting.

---

## 6. FATAL FLAWS — SYNTHESIS

**Classification: FATAL (multiple independent fatal flaws)**

### Flaw 1: The Main Finding Does Not Validate Out-of-Sample
The in-sample Normal-regime HML→SMB (p=8.75×10^−9) is robust. But the paper cannot point to an OOS replication:
- Frozen OOS Elevated result does not survive 30-pair Bonferroni.
- Bootstrap reweighting yields p=0.153.
- Regime-specific (Elevated post-2008, not Normal pre-2008).

The paper relies entirely on in-sample evidence for a relationship that existed 16+ years ago. This is publishable as a retrospective, but not as a validated forward-looking finding.

### Flaw 2: Regime-Identification Circularity Is Unresolved
Regime labels are derived from the same returns used in Granger tests. The permutation test and frozen OOS provide partial mitigation, but **do not eliminate the fundamental problem.** A reviewer could reasonably conclude that the Granger significance is an artifact of the HMM's regime detection, not evidence of causal predictive precedence.

### Flaw 3: Economic Value Is Zero
- Trading Sharpe = -0.07.
- VaR model underperforms GARCH (3.31% vs. 1.48% violation rate).
- Proposed mechanism (deleveraging cascade) is speculative and fails 4/6 event-based tests.

The paper contributes no actionable insights.

### Flaw 4: OOS Fragility Indicates the Relationship Is Unstable
The OOS Elevated result is sensitive to:
- Prevalence reweighting (bootstrap p=0.153).
- K selection (null at K=2,4; p<0.05 only at K=3).
- HAC bandwidth (p ranges 0.041 to 0.173).
- Scale convention (p = 0.022 to 0.063).

This is not robustness; it is specification-hunting.

### Flaw 5: Selective Reporting of Multiple Comparisons
- MOM→SMB is the top-ranked pair by OOS F-statistic (F=20.3), yet the paper focuses on HML→SMB (F=9.06).
- 30-pair FDR correction: no pairs survive, yet HML→SMB is highlighted.
- This is not a "post-hoc" finding transparently reported; it is selective emphasis on a secondary result.

---

## 7. INTERNAL CONSISTENCY — MAJOR FLAW

**Classification: MAJOR**

The paper contains logical contradictions:

### Contradiction 1: OOS Design as Validation
- **Claim (abstract):** "Frozen OOS testing (2013–2024) yields exploratory Elevated-regime patterns (F-p=0.003)..."
- **Claim (section~sec:frozen_oos):** "We report this for its frozen-parameter design, not statistical significance."
- **Claim (section~sec:interpretation):** The frozen OOS is positioned as supporting evidence that the relationship persists.

If the OOS result does not survive multiple-testing correction and depends on regime prevalence changes, why is it included in the abstract and main narrative? The answer is that the authors want to claim the finding is "robust" while maintaining plausible deniability by disclaiming statistical significance. This is incoherent.

### Contradiction 2: Economic Validity vs. Statistical Optimality
- **Claim:** "We prefer the Student-t specification because the Gaussian HMM misspecifies tail behavior...but acknowledge the trade-off: better distributional fit comes at the cost of reduced crisis-event alignment."
- **Evidence:** Student-t HMM assigns 0% of 2008 to Crisis (BIC-optimal Cluster 1); Gaussian assigns 83%.
- **Conclusion:** "We report results under both the BIC-optimal (Cluster 1, primary) and economically valid (Cluster 5, sensitivity) fits."

But Cluster 5 has ΔBic = 218 higher than Cluster 1. Selecting Cluster 1 as "primary" on BIC grounds while acknowledging it fails basic economic validity is illogical. Either (a) the economic validity criterion should be primary (then use Cluster 5), or (b) pure BIC should be primary (then accept that your regime labels are economically nonsensical). The paper tries to have both and succeeds at neither.

### Contradiction 3: Frozen OOS Logic
- **Claim:** The frozen OOS "provides the strongest available mitigation" against circularity.
- **Evidence:** The frozen OOS Elevated result does not replicate the in-sample Normal finding; instead, it shows a different regime (Elevated post-2008 vs. Normal pre-2008).
- **Interpretation:** Either (a) the in-sample Normal result is not replicable (suggesting it was regime- or period-specific), or (b) the OOS Elevated result is an artifact of regime prevalence changes (bootstrap p=0.153).

If (a), the paper's main finding is unstable. If (b), the OOS result is not "validation" but regime redistribution. The authors treat these as compatible, but they are not.

---

## 8. MISSING COMPARISONS AND ALTERNATIVES

**Classification: MAJOR**

### What Should Have Been Compared:

1. **Regime-conditional Granger vs. simple rolling Granger:** The paper (section~sec:limitations) reports that rolling 250-day Granger finds HML→SMB significant in 36.2% of windows, with concentration in Normal (47.1%), but does not compare rolling-window predictability to the regime-conditional approach systematically. A head-to-head rolling vs. regime-conditional analysis would strengthen the methodological case.

2. **Simpler regime-detection methods:** The paper compares to threshold-based realized-volatility regimes (section~sec:limitations) and shows HMM outperforms, but does not compare to supervised regime labels (e.g., NBER recession dates, VIX quintiles) or simpler latent-variable models (e.g., K-means clustering, Gaussian HMM). Would a Gaussian HMM with forced crisis-event alignment yield materially different Granger results?

3. **Causal discovery methods:** The paper applies Granger tests but does not compare to modern causal discovery (PC algorithm, GES, or causal forests). Granger causality is a necessary condition for causality, not sufficient. A causal discovery algorithm might identify common confounders invisible to bivariate Granger tests.

4. **Alternative predictive relationships:** Why focus on HML→SMB? The paper finds 19/30 directed pairs exhibit regime heterogeneity (section~sec:robustness). A systematic study of which pairs break down when, and whether there is a common economic driver, would be more informative than deep-diving into one pair.

5. **International results:** The international replication (section~tab:international) shows mixed results (2/4 regions replicate strongly), yet the paper does not investigate why. Are there macroeconomic variables (interest rates, credit spreads, capital flows) that predict which countries' cross-factor relationships break down? This would elevate the contribution from descriptive to explanatory.

---

## 9. VERDICT AND JUSTIFICATION

**WEAK REJECT**

This paper is statistically competent and transparently written, but fails the core bar for acceptance at a top-tier venue. The main in-sample finding—Normal-regime HML→SMB Granger significance at p=8.75×10^−9—is robust to HAC correction and local optima variation. However, the paper's claimed out-of-sample validation does not materialize (OOS result fails 30-pair Bonferroni correction, depends on regime prevalence changes with bootstrap p=0.153), and the relationship's economic value is demonstrably zero (trading Sharpe=-0.07, VaR model underperforms GARCH). The regime-identification circularity—wherein regime labels are derived from the same returns used in Granger tests—remains unresolved; while permutation testing and frozen OOS provide partial mitigation, they do not eliminate the fundamental confounding. The paper's methodological contribution is the assembly of existing techniques (Student-t HMM, Granger testing, transfer entropy, quantile regression) into a "protocol," but each component is prior work and the joint insight is limited to the observation that one factor pair (SMB→HML) exhibits tail-dependent nonlinearity. This is pair-specific and does not generalize. The paper reads as a well-executed empirical exercise documenting that cross-factor predictability decayed post-1998, with thorough robustness analysis and honest reporting of null findings (trading failure, VaR failure). For ICAIF 2026, this represents solid empirical work but insufficient novelty or economic substance to warrant publication. The authors should consider: (1) investigating *why* the OOS result fails to replicate, (2) conducting causal discovery to distinguish Granger predictability from confounding, or (3) focusing on a new phenomenon rather than documenting the known instability of factor relationships.

---

## DETAILED COMMENTS FOR AUTHORS

### High Priority (Must Address for Resubmission):

1. **OOS Replication Failure:** The Elevated-regime OOS result is not replication of the Normal-regime in-sample finding. Explain why the same HML→SMB relationship should manifest in a different regime post-GFC. Alternatively, re-frame the paper as documenting regime-specific rather than pan-regime decay.

2. **Circularity Mitigation:** Perform an IV analysis using a lagged external instrument (e.g., lagged VIX, lagged credit spreads) as an instrument for regime assignments. If regime-conditional Granger survives IV correction, circularity is less concerning.

3. **Economic Relevance:** Either (a) develop a trading strategy that exploits the relationship (even modestly positive Sharpe would help), or (b) re-frame as purely diagnostic and evaluate against other diagnostic tools (rolling Granger, VAR forecasting, connectedness indices).

4. **Mechanism Verification:** Secure 13F data and test whether institutional holdings patterns match the deleveraging hypothesis. Without direct verification, the economic mechanism is speculation.

### Medium Priority:

5. **Local Optima Resolution:** Choose a single regime specification ex-ante (either BIC-optimal or economically valid) and stick with it. The current approach of reporting both raises more questions than it answers.

6. **International Heterogeneity:** Investigate why 2/4 regions replicate strongly and 2/4 do not. Are there macroeconomic variables that predict cross-factor breakdowns?

7. **Comparison to Baselines:** Formally compare regime-conditional Granger to rolling-window, threshold-based, and Gaussian HMM alternatives on a consistent out-of-sample test set.

---

## FINAL ASSESSMENT

**Strengths:**
- Rigorous statistical methodology with transparent reporting of fragilities.
- Comprehensive robustness checks (HAC bandwidths, local optima clusters, international replication).
- Honest admission of null results (trading failure, VaR failure).
- Novel pairing of quantile regression and transfer entropy to reveal directional asymmetry in one pair.

**Weaknesses:**
- Main OOS validation does not replicate the in-sample finding.
- Regime-identification circularity is unresolved.
- Economic value is zero (trading Sharpe=-0.07, VaR underperforms).
- Methodological contribution is assembly of prior techniques, not innovation.
- Paper documents known instability of factor relationships without explaining why or predicting when breaks occur.

**Recommendation:** Reject with invitation to resubmit after addressing circularity, replicating the finding, and establishing economic relevance.

---

**Date:** March 1, 2026
**Reviewer Confidence:** High (paper is well-executed; rejection is due to insufficient novelty and economic contribution, not methodological flaws)
