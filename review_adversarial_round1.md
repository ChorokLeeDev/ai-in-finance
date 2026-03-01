# ICAIF 2026 - ADVERSARIAL REVIEW
## "Structural Decay of Cross-Factor Predictability: Regime-Conditional Granger Analysis with Complexity Characterization"

---

## Overall Assessment

**Decision: WEAK REJECT**

This paper documents a structurally interesting finding---HML Granger-predicts SMB in pre-crisis but not post-GFC periods---with transparent methodology and careful multiple-testing corrections. However, it suffers from fundamental issues that prevent acceptance: (1) the main in-sample finding, while robustly significant, is not novel econometrically and has limited practical implications; (2) the out-of-sample "re-emergence" evidence is fragile and explicitly non-significant under proper multiple-testing; (3) the methodological contribution is overstated—the pipeline is simply a sequence of known techniques (HMM + Granger + complexity check) without genuine innovation; (4) the information-theoretic asymmetry (linear forward, nonlinear reverse) is presented as a contribution but remains exploratory and mechanistically unexplained. The paper reads as careful empirical work on a curated pair, not a methodological or substantive advance for the ICAIF community.

---

## Fatal Flaws

### 1. **The Core Finding Is Not Novel for ICAIF**
The observation that factor predictability is regime-dependent is well-established (Chordia & Shivakumar 2002, Hamilton 1989). The paper's contribution—that HML→SMB exists in Normal but not Crisis—is a **specific empirical result on a curated pair**, not a methodological or theoretical advance. For a venue like ICAIF, this is insufficient. The paper even acknowledges (Section 3) that Psaradakis et al. (2005) pioneered regime-switching Granger causality and Tank et al. (2022) extended to neural settings. What exactly is novel here? **Applying known methods to one pair is not a contribution.**

### 2. **The In-Sample Finding Cannot Resolve Regime-Identification Circularity**
The paper acknowledges (Section 2.3) that regime labels come from the same returns used in Granger tests, creating circularity. The mitigation offered—that HMM uses distributional properties while Granger tests temporal dynamics—**is not a real solution**. These are not "functionally distinct" (line 304-305): both are learned from the same multivariate returns. The "soft-label sensitivity" check (weighted Granger with posterior probabilities) is reassuring but does not prove independence.

The *primary* validation is the frozen OOS test (2013-2024), but Table 3 shows this fails all multiple-testing corrections. Under 30-pair Bonferroni (appropriate, given in-sample screening), neither $F$-$p = 0.003$ nor HAC $p = 0.043$ survives (threshold: $p < 0.00033$). The authors explicitly state "this does not survive 30-pair Bonferroni" (line 729). So the primary in-sample result (Normal regime) rests on data-driven regime discovery from the same sample, while the OOS validation fails significance tests. **This is circularity with imperfect mitigation.**

### 3. **Out-of-Sample Results Are Explicitly Non-Significant and Driven by Prevalence Shift**
Table 3 is devastating to the OOS claim:
- HAC $p = 0.043$ does not survive 3-regime Bonferroni ($\alpha/3 = 0.0167$), let alone 30-pair.
- Bootstrap prevalence reweighting yields median $p = 0.153$—the OOS signal is driven entirely by Elevated regime expansion (13.7% → 33.7%), not by genuine re-emergence of the HML→SMB effect.
- Results are sensitive to regime specification (null at $K=2,4$; Table 5).
- Bandwidth sensitivity (Table 6) shows $p$-values range from 0.041 to 0.173; at standard Newey-West bandwidth=6, $p = 0.056$ (fails 5% threshold).

The authors acknowledge all of this (Section 4.1: "exploratory evidence with disclosed fragility"), but then why claim this as a finding at all? The paper would be stronger if it simply stated: "In-sample Normal-regime finding is robust; OOS re-emergence is speculative and does not survive multiple-testing correction."

### 4. **Methodological Contribution Is Overstated**
The paper claims three contributions in the introduction. Let's examine each:

**Claim 1: Reusable pipeline.** The pipeline is: (a) fit HMM, (b) extract regime labels, (c) run Granger tests per regime, (d) apply Bonferroni correction, (e) frozen OOS validation. Each step is standard. There is no novel combination here—this is how practitioners have done regime-conditional analysis since Psaradakis et al. (2005). The 50-seed multistart (Section 3.4) is a sensitivity check, not a methodological advance.

**Claim 2: Complexity characterization.** The four-model protocol (OLS, RF, MLP, LSTM) is from Tank et al. (2022). The transfer entropy analysis is from standard kNN estimators (Frenzel-Pompe). Where is the novelty? The paper claims the linear-forward/nonlinear-reverse asymmetry is "invisible to standard Granger." But this is only invisible because the forward channel is captured by linear Granger while the reverse is not—this is a finding *about the data*, not a methodological insight. The paper does not explain *why* the reverse channel is nonlinear or what it implies.

**Claim 3: Multi-seed robustness framework.** The 50-seed multistart reveals that BIC-optimal fit assigns 0% of 2008 GFC days to Crisis regime, while economically motivated fits assign 90-100%. The paper treats this as a "framework" but it is really a **confession that the statistical criterion fails economically**. The solution—cherry-pick fits based on GFC detection—is post-hoc and undermines claims of data-driven methodology. This is weakness, not contribution.

---

## Major Weaknesses

### 1. **Pair Selection Is Circular and Under-Justified**
The paper focuses on HML→SMB because it was "identified from screening all 30 directed factor pairs" (Section 2.5, line 371). This is in-sample pair selection without pre-registration.

The justification offered: (1) economic plausibility (institutional crowding), (2) in-sample screening, (3) architectural contrast (lag-1 dominance). But here's the problem: **the economic prior for HML-SMB is stated post-hoc**. The paper says (line 364-368): "We focus on the HML–SMB pair for three reasons... First, *economic plausibility*: Value and Size factors have documented overlap..." This is circular reasoning: the economic motivation follows the data analysis, not precedes it.

The paper tries to defend this (Section 2.5): "the primary (LL-only) fit independently identifies HML–SMB as distinctive... no 2008-based screening is involved." But the screening was done on all 30 pairs. Under any honest multiple-testing framework, the OOS test inherits the full 30-pair multiplicity burden, which the authors acknowledge but then downplay (lines 383-392).

The top-ranked OOS pair by $F$-statistic is actually MOM→SMB ($F = 20.3$; line 413-414), not HML→SMB ($F = 9.06$). The authors note this casually and cite the "economic prior" to justify HML focus. **But this prioritizes a prior over the data.** In a pre-registered framework, that would be fine. But post-hoc, it looks like cherry-picking.

### 2. **The Structural Break Finding Relies on Problematic Methods**
The paper claims June 1998 is the structural break (sup-$F = 21.2$, $p = 1.23 \times 10^{-13}$). But:

- **Quandt-Andrews sup-$F$ is known to be data-snooping-prone.** Testing across all possible break dates without prior specification is precisely the kind of multiple-testing that inflates Type I error. The $p$-value is tiny, but why would a practitioner trust it when testing 9000+ potential break points?

- **The break date is not independent of the Granger analysis.** The paper shows (line 523-524): Pre-2008 Normal: $p = 6.66 \times 10^{-16}$; Post-2008 Normal: $p = 0.73$. So the break announcement (June 1998) post-hoc fits a pre-post narrative. But if June 1998 is the break, why does the pre-2008 window still show strong significance? The paper's explanation: "gradual erosion beginning around June 1998, not a single collapse" (line 547-548). This sounds like post-hoc rationalization rather than a hypothesis-driven finding.

- **Chow test at January 2008 is theory-motivated, not data-driven.** The paper acknowledges this (line 549-551) and states the Chow $p = 2.29 \times 10^{-6}$ is "10^7 times weaker" than the sup-$F$. So the January 2008 date is cherry-picked because the GFC happened then, not because the data strongly supports it. This is *not* identifying a second independent break; it is post-hoc confirmation.

- **Bai-Perron sequential multiple-break test is the standard.** The paper acknowledges this limitation (lines 549-551) but doesn't apply it. This is a significant methodological gap.

### 3. **Effect Sizes Are Economically Immaterial**
- $\Delta R^2 = 2.06\%$ in pre-GFC Normal regime (line 523).
- The paper frames this as "$\sim$\$70M additional 1-day 99\% VaR" for a \$100B portfolio (line 579-580), but this is speculative and assumes the coefficient remains stable—which the paper's own evidence contradicts.
- A trading strategy based on the signal yields Sharpe = -0.07 (Appendix, line 1230-1234). Negative!
- The authors even state (line 1024-1026): "Effect sizes are small ($\Delta R^2 \approx 2\%$ pre-GFC) and do not generate trading profits."

So what is the contribution? **Risk model specification**, per line 1027-1031. But a 2% $\Delta R^2$ improvement in cross-factor dynamics is not typically economically significant in practice. The paper claims practitioners should "re-estimate risk model parameters after major regime shifts" (line 581-582), but practitioners already do this, and the paper provides no evidence that regime-aware risk models outperform simpler alternatives.

### 4. **Local Optima Problem Undermines Regime Identification**
Section 3.4 and Table 4 reveal a damning finding: 50-seed multistart yields 7 local optima clusters. The BIC-optimal (Cluster 1) assigns 0% of 2008 GFC days to Crisis regime. Economically sensible fits (Clusters 5-7) assign 90-100% but sacrifice $\Delta$BIC = 218 units.

The paper's response: apply a post-hoc criterion (≥50% GFC detection) to filter cluster selection. This is **not data-driven**; this is **imposing a prior based on the calendar**. The authors admit (line 1010-1012): "The two-stage criterion... is fundamentally post-hoc. We acknowledge this limitation explicitly: had we relied on BIC alone, we would have obtained a fit inconsistent with financial economics."

This is a serious problem. BIC is the gold standard for model selection (avoids overfitting), yet the authors discard it because it doesn't align with 2008 GFC calendar. Why should we trust regime assignments that require hand-picked fitting criteria? **The method is unreliable.**

The OOS results are "robust across all 7 clusters" (line 1005), but this doesn't address the cluster-selection problem: all 7 clusters are suboptimal fits compared to the globally optimal one. If the globally optimal fit assigns 0% of 2008 to Crisis, what does that tell us about the regime structure?

### 5. **Transfer Entropy Analysis Is Exploratory and Unexplained**
Section 3.5 claims a novel finding: reverse SMB→HML is nonlinear ($z = 5.37$) while forward HML→SMB is linear ($z = 2.45$). But:

- The four-model diagnostic (OLS, RF, MLP, LSTM) shows nonlinear methods do NOT improve forward prediction. So why is reverse TE higher? The paper hypothesizes (line 687-689): "higher-order moments, tail co-movements" without testing this. This is speculation.

- Transfer entropy measures directed information flow via entropy reduction, not forecasting improvement. The asymmetry could reflect: (a) genuine nonlinearity, (b) different embedding dimensions, (c) regime-specific tail risk, or (d) noise in the TE estimator. The paper does not disentangle these.

- kNN estimators (Frenzel-Pompe) are known to be sensitive to metric choice and dimensionality. Appendix sensitivity analysis ($k \in \{3,...,7\}$) shows forward $z \in [2.1, 2.8]$ and reverse $z \in [4.9, 5.6]$, confirming directionality is robust. But what does it mean economically? The paper doesn't say.

- The asymmetry is called "invisible to standard Granger," but this is tautological: if linear Granger captures all forward predictive content, nonlinear TE in reverse simply means the reverse channel is nonlinear. This is not a finding; it's a restatement.

### 6. **Complexity Characterization Misses the Real Complexity**
The paper claims "no nonlinear method improves prediction" (line 654-655) under the BIC-optimal fit. But:

- Under the alternative regime partition (sensitivity fit, seed 42), RF is significant in Elevated ($p = 0.010$) and Crisis ($p = 0.005$). So the "purely linear" claim is **fit-dependent and not robust**.

- The four-model protocol uses expanding-window cross-validation with 200 permutations. For an LSTM with 32 hidden units, 200 permutations may have insufficient power ($p$-value SE ≈ 0.007 at $p = 0.01$; line 1268-1269). The LSTM uses only 100 permutations, acknowledged as approximate (line 1275-1276).

- MSE minimization and information-theoretic measures are conflated. The paper says reverse TE is "nonlinear" but forward is "linear," but TE measures entropy reduction, not Granger predictability. These are orthogonal frameworks. Claiming the reverse channel is "real but invisible to linear tests" (line 671-672) confuses signal presence with predictive utility.

### 7. **Institutional Crowding Mechanism Is Speculative**
Section 3.6 and Appendix A hypothesize a deleveraging cascade channel but offer only post-hoc evidence:

- FF25 overlap analysis (Appendix B) finds Spearman $\rho_s = 0.35$ (permutation $p = 0.046$), but this survives permutation testing by accident—under the Bonferroni correction for 25 portfolios, three are significant, which is 3/25 = 12%, above the 4% baseline (line 1154-1160). The signal is weak and noisy.

- Event-based validation (Appendix C) shows the HML→SMB pattern holds in 2 of 6 events, with 4 directionally correct (binomial $p ≈ 0.11$, non-significant). The 2018 Vol Shock and 2022 Rate Hikes show reversed dynamics, which the paper attributes to "monetary policy" and "growth rotation"—post-hoc explanations that undermine the deleveraging narrative.

- Holdings-level 13F verification is listed as "future work" (line 1022), meaning the mechanism is essentially unverified.

**There is no smoking gun here.** The mechanism is plausible but speculative.

### 8. **Frozen OOS Design Does Not Eliminate Data Snooping**
The paper positions frozen OOS (train 1990-2012, test 2013-2024) as primary validation. But:

- HML→SMB was identified from **in-sample screening of 30 directed pairs** (line 371). The frozen test therefore inherits the 30-pair multiplicity. Under Bonferroni, no pair survives.

- "Frozen parameters" does not mean "no data snooping." The regime assignments, pair selection, lag selection (BIC per regime), and Bonferroni correction were all tuned on 1990-2012 data. When applied to 2013-2024, these design choices introduce a degrees-of-freedom penalty.

- The paper acknowledges (line 341-343): "Because HML–SMB was identified from screening all 30 directed pairs in-sample, the *primary* OOS correction is 30-pair Bonferroni... under which no OOS pair survives." So the "frozen OOS" is primary validation in design only, not in significance.

### 9. **Scale Convention Introduces Arbitrary Degrees of Freedom**
Section 2.1 defines a percentage-unit convention (0.10 = 0.1%) as primary, with decimal-unit (0.10 = 10%) as a robustness check. The authors note (line 238-242):

- Percentage units: $n = 953$ Elevated OOS days, $F$-$p = 0.003$, bootstrap $p = 0.153$.
- Decimal units: $n = 836$ Elevated OOS days, permutation $p = 0.063$ (non-significant).

This is a **critical degree of freedom**. HMM emission probabilities are scale-sensitive, so different conventions yield different regime boundaries. The choice of percentage units as "primary" is justified by saying "Granger $F$-statistics are scale-invariant" (line 237), but this conflates the test with the regime classification. The regimes themselves are scale-dependent.

Readers should ask: **why not report both in equal standing, or pre-specify the scale?** The fact that decimal units yield non-significant permutation results suggests scale choice matters materially.

### 10. **Multiple Testing Framework Is Inconsistent**
The paper applies different corrections in different contexts:

- In-sample: 30-pair Bonferroni at $\alpha = 0.01$ per family ($\alpha/30 = 0.00033$). Stringent.
- OOS primary: 30-pair Bonferroni (no pairs survive).
- OOS sensitivity: 3-regime Bonferroni ($\alpha/3 = 0.0167$). HAC $p = 0.043$ still fails (but $F$-$p = 0.003$ survives, inconsistently).
- Permutation test: Different framework, not directly comparable.
- Bootstrap prevalence reweighting: Median $p = 0.153$, which is under what correction exactly?

The paper tries to be transparent about all these comparisons, but the multiplicity of correction frameworks makes it hard to extract a clean message. **Best practice would be: pre-specify the correction framework and stick to it.** Here, the paper reports many frameworks and lets readers choose the one that looks best.

---

## Minor Weaknesses

### 1. **Writing and Organization Issues**

- **Verbosity**: The paper is 1,385 lines for an empirical finding on one pair. Critical material is buried: the Bonferroni-nonsignificant OOS result is the main validation, yet receives Section 3.7 treatment while introductory material sprawls.

- **Contradiction in framing**: The abstract claims "structural decay" (implying a finding), but the main result is that in-sample Normal-regime predictability vanished post-2008 and OOS re-emergence is non-significant under proper correction. These are not "structural decay"; they are "no predictability post-GFC."

- **Footnote clutter**: Critical information buried in footnotes (e.g., footnote on line 282-288 detailing seed selection; Section 3.4 footnote on regime-boundary exclusion).

- **Appendix bloat**: ~270 lines of appendices for robustness checks that mostly confirm non-significance of contested results. This should be streamlined.

### 2. **Figure Quality**
- Figure 1 (regime timeline) uses "regime-colored background" (line 453) but does not show the returned regimes themselves for the full sample. Reader must infer.
- Figure 3 (heatmap) is hard to parse at 30 pairs; consider subsetting to regime-heterogeneous pairs.
- Figure 6 (transfer entropy) uses bar charts that conflate $z$-scores across different permutation counts (LSTM uses 100 vs. 200). Misleading.

### 3. **Trivariate Control Is Weak**
MKT-RF is included as a confounder (Section 3.8), but five other factors (RMW, CMA, MOM) are ignored. A full VAR conditioning on all factors would be more convincing, but the paper notes (footnote 329) that a 6-factor VAR is "under-identified at $n ≈ 1,000$." This is a real limitation but not addressed.

### 4. **Regime Count Selection Lacks Justification**
$K = 3$ is selected on training data via BIC ($\Delta$BIC vs. $K=2$: 1,680). But the paper doesn't discuss whether this is data-driven or arbitrary. Three regimes align with (Normal, Elevated, Crisis), but this is imposed *a priori* by the regime names, not discovered from the data. A more honest approach: report results under $K \in \{2,...,5\}$ as primary, with $K=3$ as default.

The $K$-sensitivity analysis (Table 5) shows OOS significance at $K=3$ only. This is a red flag: the result depends on the number of latent states, which is a modeling choice.

### 5. **Statistical Power of LSTM is Undisclosed**
The LSTM uses 100 permutations (line 1275-1276) with acknowledged approximate $p$-values. For a sample of 4,496 observations (Normal regime), 100 permutations is weak (SE ≈ 0.03 at $p = 0.5$). Readers cannot assess significance reliably.

### 6. **Soft-Label Sensitivity Is Under-Reported**
Section 2.3 mentions soft-label sensitivity (weighted Granger with posterior probabilities) yields Normal-regime HML→SMB at $p < 10^{-7}$. But no table is provided, and results for Crisis/Elevated are not reported. This limits reproducibility.

### 7. **Lag Selection Is Not Justified**
BIC is used to select lags 1-15 per regime-direction pair. But:
- Why 1-15? Is this range pre-specified?
- Do selected lags vary across regimes in expected ways? (Table 1 shows all lag=1, which is suspicious—suggests BIC favors lag-1 uniformly.)
- Cross-regime lag heterogeneity would be theoretically interesting but is unreported.

### 8. **Boundary Exclusion Rates Seem Arbitrary**
In-sample lag-1 exclusion: 0.67% (59/8,817). OOS lag-1 exclusion: 7.4% (224/3,020). The 11-fold increase is justified (line 753-754) by "more frequent regime transitions when the frozen 1990–2012 classifier is applied to the unseen 2013–2024 period." But this raises a question: **is the frozen classifier stable?** If regime transitions are 11× more frequent OOS, the regimes themselves are changing, which suggests the frozen classification is suboptimal. This is underexplored.

### 9. **VaR Application Is Relegated to Appendix and Fails**
Section 4.2 and Appendix note that regime-conditional VaR "exhibits high false-alarm rates (93.2%)" (line 1237). This is a practical failure. If the regime-conditional models can't improve risk measurement, what's the point? This deserves more attention, not burial in the appendix.

---

## Strengths

### 1. **Exceptional Transparency and Disclosure of Limitations**
The paper's greatest strength is brutal honesty. The authors:
- Explicitly acknowledge circular regime identification (Section 2.3) and propose mitigations.
- Report that BIC-optimal fit assigns 0% of 2008 to Crisis regime and disclose this as a failure (Section 3.4).
- State clearly that OOS results "do not survive 30-pair Bonferroni correction" (line 729).
- Acknowledge the bootstrap prevalence reweighting makes OOS signal non-significant (line 787-791).
- Disclose band width sensitivity (Table 6) showing results cross the 5% threshold.
- Report event-based validation that shows the mechanism holds in only 2 of 6 events.

This transparency is rare and valuable. Reviewers can trust the data reported.

### 2. **In-Sample Normal-Regime Finding Is Robustly Significant**
The core result—HML→SMB Granger-predicts in Normal pre-crisis regime at $p = 8.75 \times 10^{-9}$—is strong and robust to:
- HAC corrections (line 517-520).
- Lags 1-15 sensitivity (Figure 3).
- Trivariate control for MKT-RF (line 885-890).
- Multiple HMM local optima (OOS Elevated $p < 0.05$ in 7/7 clusters).
- Soft-label regime assignment (posterior probabilities yield $p < 10^{-7}$).

This finding is not methodologically novel (regime-switching Granger is standard), but the robustness is impressive.

### 3. **Complexity Characterization Adds Dimension**
Combining four predictive models (OLS, RF, MLP, LSTM) with transfer entropy is a useful diagnostic. Even if conclusions are tentative, the protocol could be reusable for others studying predictive-vs.-information-theoretic trade-offs. The finding that forward HML→SMB is linear while reverse SMB→HML shows nonlinear information flow is interesting, even if unexplained.

### 4. **Data-Driven Structural Break Detection**
Using Quandt-Andrews sup-$F$ to identify June 1998 as a break point is reasonable (even if not pre-specified). The finding that breakage predates the GFC and accelerates through 2008 (Chow test) provides historical perspective. The mechanism could be LTCM-driven liquidity stress (line 535-537).

### 5. **Frozen OOS Design Shows Methodological Awareness**
While the frozen OOS fails significance testing, the design itself (train 1990-2012, test 2013-2024) is sound practice. Many regime-switching papers refit on full sample, biasing towards spurious persistence. The authors implement best practice here, even though results don't survive multiple-testing.

### 6. **Appendices Are Comprehensive**
The appendices cover mechanism (FF25 overlap), events, alternatives (MS-VAR, weekly, rolling), and robustness. Readers can dig deep if curious. The transparency of failure modes (e.g., event-based validation shows reversed 2022 dynamics) is commendable.

---

## Specific Line-Level Issues

### 1. **Abstract (Lines 44-70)**
The abstract is misleading. It emphasizes the structural break (June 1998, $p = 1.23 \times 10^{-13}$) and OOS patterns (Elevated, $p = 0.003$) without mentioning:
- The OOS result depends on regime expansion and doesn't survive Bonferroni correction.
- The structural break is identified post-hoc via sup-$F$, not pre-specified.
- The reverse transfer entropy finding is interesting but speculative.

**Suggestion**: Rewrite to emphasize the robust in-sample finding (Normal regime $p = 8.75 \times 10^{-9}$) and position OOS as exploratory.

### 2. **Contribution Statement (Lines 141-181)**
Contribution 1 claims a "reusable regime-conditional analysis pipeline." But the pipeline (HMM + Granger + BIC lag selection + Bonferroni + frozen OOS) is standard practice since Psaradakis et al. (2005). The novelty is unclear.

**Suggestion**: Reframe as "application" not "contribution." State that the methodological contribution is in combining regime-conditional Granger with complexity characterization (Claims 2-3), not the pipeline itself.

Contribution 2 claims a "linear–nonlinear asymmetry." But this is a finding about HML-SMB, not a methodological advance. Transfer entropy is from Schreiber (2000); kNN estimation is standard.

**Suggestion**: Rename "Complexity characterization" to "Diagnostic framework" and position it as a protocol others can apply, not a methodological innovation.

Contribution 3 calls the 50-seed multistart a "robustness framework with transparent fragility disclosure." This is backward. The multistart reveals that BIC-optimal fit fails economically. Rather than a "framework," this is a confession that statistical model selection is unreliable.

**Suggestion**: Reframe as "Regime identification sensitivity analysis" and acknowledge that the tension between statistical fit and economic validity is unresolved.

### 3. **Methodology Section (Lines 247-256)**
The justification for Student-$t$ HMM is that financial returns have excess kurtosis ($\hat{\nu}_{\text{Normal}} = 6.2$, etc.). But the paper doesn't compare Student-$t$ vs. Gaussian HMM results systematically. Section 3.6 (line 462-466) notes the Gaussian HMM assigns 83% of 2008 to Crisis while Student-$t$ assigns 0%, but this comparison is buried in footnotes.

**Suggestion**: Add a subsection comparing the two HMM variants and discussing why Student-$t$ is preferred despite worse GFC detection. Or justify the choice more carefully upfront.

### 4. **Frozen OOS Validation (Lines 732-751)**
The frozen OOS section explains the design well but downplays the failure. Lines 743-750 describe how Elevated prevalence grows from 13.7% to 33.7%, but this is presented matter-of-factly.

**Suggestion**: Highlight this as the primary result: "The post-GFC Elevated regime comprises 33.7% of observations vs. 13.7% in training. This regime shift drives the OOS significance ($F$-$p = 0.003$); bootstrap reweighting to training prevalence yields non-significant $p = 0.153$." This framing is honest.

### 5. **Related Work (Lines 183-222)**
The related work positions the paper as the first to combine "regime-conditional Granger causality with a multi-model complexity characterization and transfer entropy" (lines 217-220). But this combination is not well-motivated. Why would a practitioner want all three (Granger, complexity, TE) rather than one? The paper doesn't establish that the combination is necessary or sufficient.

**Suggestion**: Discuss what practitioners are missing with Granger alone, and explain how TE and complexity diagnostics fill the gap. Section 3.5 does this partially, but the logic is murky.

### 6. **Discussion of Structural Break (Lines 528-567)**
Lines 528-534 announce June 1998 as the break based on Quandt-Andrews sup-$F$. But 200+ lines later (lines 549-551), the paper admits that Bai-Perron sequential testing would be standard and isn't applied. This is a significant omission—why use sup-$F$ instead?

**Suggestion**: Apply Bai-Perron testing to identify the number and location of breaks, or explicitly state why sup-$F$ is preferred for this application.

### 7. **Local Optima Section (Lines 963-1015)**
Section 4.4 frankly discusses the tension between BIC and economic validity. But the solution (impose post-hoc GFC-detection criterion) is acknowledged as "fundamentally post-hoc" and "cannot be justified as data-driven in the strict sense" (line 1013-1014).

**Suggestion**: Either (a) commit to BIC and report results under all 7 clusters transparently, or (b) pre-register the GFC-detection criterion and split the sample so you can evaluate it on held-out data. The current approach (tune criterion on full sample, then claim robustness across clusters) is backwards.

### 8. **Complexity Characterization (Lines 639-696)**
Section 3.5 argues that transfer entropy reveals "a stronger nonlinear reverse channel" (lines 667-668) but then admits (lines 671-672) this is "nonlinear and substantially stronger... [but the] reverse channel is real but invisible to linear tests." This conflates several ideas:

- Granger predictability (linear, MSE-based).
- Transfer entropy (information-theoretic).
- Nonlinearity (ML models).

The paper doesn't explain why the reverse channel is nonlinear. Is it higher-order moments? Tail dependence? State-dependent drift? Without mechanistic insight, the asymmetry remains an empirical curiosity.

**Suggestion**: Investigate the sources of nonlinearity in SMB→HML. Use SHAP or other attribution methods to identify which features drive TE. Or fit quantile regressions to test tail-dependence hypotheses.

### 9. **Trading Backtest (Appendix, Lines 1228-1234)**
The appendix reports a trading rule with Sharpe = -0.07. This is not just unprofitable; it's worse than holding cash (Sharpe ≈ 0). The paper dismisses this ("intentionally simple rule, no optimization") but doesn't explain why readers should care about predictability that is economically useless.

**Suggestion**: Either develop a trading rule that captures economic value, or explicitly state that the paper is about risk model specification, not trading profits. The current treatment muddies the contribution.

### 10. **Power Analysis (Lines 807-814)**
The paper reports "75% power to detect the observed effect size ($\Delta R^2 = 0.73\%$)" in the OOS Elevated sample. But this is circular: the paper is designed around $n = 953$, so post-hoc power analysis is not informative. Prospective power calculation (before data collection) would be useful, but retrospective power is tautological.

**Suggestion**: Remove the power analysis or frame it honestly: "The OOS Elevated sample of 953 observations has sufficient power to detect the observed effect size, but the effect is marginal after multiple-testing correction."

---

## Questions for Authors

1. **Why not apply Bai-Perron sequential testing?** This is the gold standard for multiple-break detection. Quandt-Andrews sup-$F$ is more exploratory. If the paper claims a structural break, use rigorous methodology.

2. **How sensitive are conclusions to the Student-$t$ vs. Gaussian HMM choice?** The Gaussian HMM assigns 83% of 2008 to Crisis vs. 0% for Student-$t$ (line 462-463). Why prefer a fit that fails to identify the most obvious crisis?

3. **What explains the linear-forward/nonlinear-reverse asymmetry in transfer entropy?** Is it tail dependence, higher-order moments, regime-dependent drift, or noise? Without mechanistic understanding, the finding is a curiosity.

4. **Can the OOS result be replicated on international factor data?** The paper is limited to US Fama-French factors. Pre-registered replication on international data would be convincing.

5. **Why emphasize the OOS result when it doesn't survive multiple-testing correction?** The main finding is the in-sample Normal regime. Why bury this and highlight OOS that fails significance tests?

---

## Missing Comparisons

1. **No comparison to simpler baseline**: Rolling unconditional Granger (Figure 4) shows episodic peaks. Does regime-conditional Granger outperform a simple rolling-window approach?

2. **No comparison to alternative regime methods**: The paper uses HMM. What about threshold-based regimes (volatility > 90th percentile)? The paper mentions this as "exploratory" (line 1247-1248) but doesn't develop it.

3. **No comparison to standard risk models**: The VaR comparison (Appendix) shows regime-conditional models have 93.2% false-alarm rates. Does GARCH or other standard models outperform? This undermines claims of practical utility.

4. **No out-of-sample trading comparison**: The trading rule (Sharpe = -0.07) is uninformative. What about a dynamic portfolio rebalancing strategy that uses regime predictions to adjust factor exposures?

---

## Overall Editorial Comments

### Strengths of Presentation
- Data and code availability promised (line 1095-1098).
- Reproducibility note with fixed seeds (line 1100-1113).
- Appendices comprehensive and well-organized.

### Weaknesses of Presentation
- Paper is too long for the contribution. 1,385 lines for one pair with a non-significant OOS result.
- Abstract misrepresents findings (emphasizes OOS $p=0.003$, omits Bonferroni failure).
- Contributions are overstated (standard methodology rebranded as "pipeline").
- Multiple testing corrections are inconsistently applied across sections.

### What This Paper Needs to Be Acceptable

1. **Refocus on the in-sample Normal-regime finding** ($p = 8.75 \times 10^{-9}$), which is solid and robust. Don't oversell the OOS evidence.

2. **Apply rigorous break-point testing** (Bai-Perron) rather than post-hoc Quandt-Andrews sup-$F$.

3. **Investigate the linear-forward/nonlinear-reverse asymmetry mechanistically.** Why is SMB→HML nonlinear? Quantile regression? Tail dependence? State-dependence?

4. **Acknowledge that the methodological contribution is a reusable diagnostic protocol**, not innovation in regime-switching or Granger causality. Reframe accordingly.

5. **Be honest about economic implications**: The effect is small ($\Delta R^2 = 2\%$), the trading rule is unprofitable (Sharpe = -0.07), and VaR models based on this exhibit high false-alarm rates. The contribution is academic insight, not practical innovation.

6. **Pre-register a replication study** on international factor data to validate generalizability.

---

## Verdict

This paper presents careful empirical work with exceptional transparency. The in-sample finding (HML→SMB Granger-predicts in Normal pre-crisis regime) is robust and interesting. However, for ICAIF, the contribution falls short:

- **Methodologically**: The pipeline is standard practice (HMM + Granger) since 2005.
- **Empirically**: The main OOS validation fails statistical tests. The in-sample finding is specific to one pair and one regime.
- **Practically**: Effect sizes are immaterial; trading signals are negative; risk models fail in applications.
- **Theoretically**: No mechanistic explanation for the linear-forward/nonlinear-reverse asymmetry.

The paper reads as **thorough empirical analysis, not scientific contribution**. It would be suitable for a domain conference (e.g., computational finance working group) after careful revision, but is below the bar for ICAIF acceptance.

The right venue is a specialized journal (e.g., *Quantitative Finance*, *Journal of Forecasting*) where empirical findings on factor dynamics are valued. For ICAIF, the community expects methodological innovation or theoretical insight; this paper provides neither.

**Recommendation: WEAK REJECT. With revisions addressing mechanistic understanding (complexity asymmetry), rigorous break-point testing, and honest positioning of limitations, this could become a solid empirical paper. But as currently framed, it oversells its contributions.**

