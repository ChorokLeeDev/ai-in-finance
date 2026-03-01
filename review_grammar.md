# Academic Copy-Editing Review: Structural Decay of Cross-Factor Predictability

## Executive Summary
Overall the paper is well-written with sophisticated technical content. Most issues are minor refinements that will strengthen clarity and consistency. Below are detailed findings organized by category.

---

## 1. GRAMMAR ERRORS

### 1.1 Subject-Verb Agreement & Tense Consistency

| Line | Issue | Current Text | Fix |
|------|-------|--------------|-----|
| 31–32 | Incomplete predicate; unclear subject coverage | "Cross-factor predictive relationships can structurally break down and not recover." | Acceptable as-is, but could be: "Cross-factor predictive relationships can structurally break down and may not recover." (Note: "not recover" is slightly informal; "fail to recover" is more technical) |
| 675–676 | Redundant explanation + awkward phrasing | "All findings document predictive precedence (\"Granger causality\"), not structural causality" | Acceptable, but note: the parenthetical is slightly condescending; consider: "All findings document predictive precedence—a property termed 'Granger causality'—rather than structural causality" |

### 1.2 Missing Articles & Prepositions

| Line | Issue | Current Text | Fix |
|------|-------|--------------|-----|
| 120–121 | Article omission | "prior work examines *within-factor* dynamics---not which factor leads another" | Change to: "prior work examines *within-factor* dynamics—not *which factor leads another*" is fine, but the sentence structure suggests: "prior work examines **within-factor** dynamics, not **the question of** which factor leads another" |
| 642–643 | Awkward prepositional phrase | "FF25 portfolio overlap analysis finds significance concentrating in small-cap portfolios" | Better: "FF25 portfolio overlap analysis finds that significance **is concentrated in** small-cap portfolios" |

### 1.3 Comma Splices & Run-on Sentences

| Line | Issue | Current Text | Fix |
|------|-------|--------------|-----|
| 79–82 | Comma splice risk (multiple clauses without connectors) | "The August 2007 quantitative meltdown—where systematic equity strategies lost 27\% in three days~\cite{khandani2011quants}—revealed a blind spot in factor-based risk management: correlation models measure co-movement, not temporal precedence between factors." | Acceptable (em-dash + colon structure is correct), but the second clause could be tightened. Consider breaking into two sentences if readability is prioritized. |
| 313–315 | Long compound sentence (38 words) | "The in-sample Normal result survives every specification change we tested: *lag structure*---significant at all lags 1--15..." | Acceptable. Itemized structure with colons and dashes reads well. |

### 1.4 Dangling Modifiers & Misplaced Clauses

| Line | Issue | Current Text | Fix |
|------|-------|--------------|-----|
| 281–282 | Potential ambiguity in temporal modifier | "Together, the evidence supports gradual erosion beginning around June 1998, not a single GFC-triggered collapse." | "Together" is acceptable and clear, but could be strengthened: "In sum, the evidence suggests gradual erosion—beginning around June 1998—rather than a single GFC-triggered collapse." |
| 362–363 | Sentence fragment / truncated clause | "Under an alternative fit (seed~42, highest-LL achieving $\geq$50\% GFC detection, $\Delta\text{BIC} = 218$), RF shows significant nonlinear improvement" | Acceptable. Parenthetical is properly integrated. |

---

## 2. LANGUAGE COHERENCE & CLARITY

### 2.1 Undefined or Under-defined Jargon

| Line | Issue | Problem | Fix |
|------|-------|---------|-----|
| 33–34 | "frozen-parameter OOS" | Introduced without explicit definition before use in Abstract. | Add to Abstract or define earlier: "frozen-parameter out-of-sample validation (parameters fixed to training period estimates)" |
| 43–44 | "transfer entropy" | Used in Abstract; defined formally only later (line 145, 396). | Either add one-sentence definition in Abstract or delay mention until Definition section. |
| 105 | "OLS, RF, MLP, LSTM" | Abbreviations for machine learning models not spelled out. | Expand on first mention: "ordinary least squares (OLS), Random Forest (RF), Multi-Layer Perceptron (MLP), and Long Short-Term Memory networks (LSTM)" |
| 112 | "BIC-vs-economic-validity tension" | Colloquial hyphenation of abstract concept. | Clarify: "tension between BIC optimality and economic validity in HMM selection" |
| 214–215 | "$P(z_t{=}z_{t-1})$" in Table 1 caption | Notation not spelled out. | Add note: "Probability of regime persistence; $P(z_t = z_{t-1})$" |

### 2.2 Ambiguous Phrasing

| Line | Issue | Current Text | Problem | Fix |
|--------|-------|--------------|---------|-----|
| 46–48 | Scope of "reflects" | "Elevated-regime signal...does not survive Bonferroni correction and reflects regime redistribution rather than independent replication." | Unclear whether "regime redistribution" is the regimes shifting, or the signal being due to different regime definitions. | Rewrite: "...does not survive Bonferroni correction; instead, the signal appears to reflect post-GFC regime redistribution (i.e., the classifier reassigns test-period days to different regimes than during training)." |
| 106–107 | "directional asymmetry" antecedent | "A complexity diagnostic...reveals a directional asymmetry (linear forward, nonlinear reverse via tail dependence) invisible to standard methods." | "directional asymmetry" — unclear which variable pair at first glance. | Clarify: "reveals **an asymmetry in causal direction**: HML→SMB operates linearly, while SMB→HML exhibits nonlinear dependence via tail behavior." |
| 320–321 | "hard Viterbi assignments" | Viterbi not defined before use. | Add: "hard Viterbi assignments (maximum-likelihood regime labels, as opposed to soft posterior probabilities)" |
| 356–359 | Complex nested structure | "LSTM attention concentrates 68.2\% on lag~1 in Normal, decaying monotonically to 52.9\% (Elevated) and 44.2\% (Crisis, approaching uniform baseline $1/9 = 11.1\%$)" | Difficult to parse. | Break into two sentences: "LSTM attention concentrates 68.2% on lag-1 in Normal regimes, declining to 52.9% in Elevated and 44.2% in Crisis. This monotonic decay toward the uniform baseline (11.1%) mirrors the Granger structural break." |
| 438–439 | "pair-specific" scope | "This tail mechanism is *pair-specific*: applying quantile Granger to the top-4..." | Does "pair-specific" refer to the 4 pairs mentioned or the HML–SMB pair exclusively? | Clarify: "This tail mechanism is **unique to the SMB→HML pair**: applying quantile Granger to other regime-heterogeneous pairs reveals..." |

### 2.3 Sentences Requiring Re-reading

| Line | Issue | Current Text | Suggested Rewrite |
|------|-------|--------------|-------------------|
| 153–154 | Scale sensitivity unclear | "Granger $F$-statistics are scale-invariant (they test $\beta = 0$ regardless of scaling), but HMM emission probabilities are not, so regime boundaries differ across conventions." | "Granger $F$-statistics are scale-invariant since they test whether $\beta = 0$; however, HMM emission probabilities depend on scaling, causing regime boundaries to shift across conventions." |
| 171–172 | Seed selection rationale confusing | "EM with 50 random seeds; primary fit: seed~28 (sorted-order convention among 3 seeds reaching identical LL)." | "EM algorithm with 50 random initializations. We select seed 28 as the primary fit: it is one of 3 seeds achieving identical likelihood, chosen by convention as the middle of those 3 (sorted-order selection)." |
| 195–196 | Parenthetical disrupts flow | "an economic prior (value-size institutional overlap), not empirical dominance---MOM$\to$SMB is the top OOS pair" | "an **economic prior** (anticipating institutional overlap in value and size portfolios) rather than empirical evidence of dominance. (MOM→SMB ranks highest by OOS $F$-statistic...)" |
| 299–301 | Dense information; hard to extract key point | "All three VIX regimes show significance (Normal $p = 0.028$, Elevated $p = 0.043$, Crisis $p = 0.005$), suggesting VIX detects the signal more uniformly than HMM. Both converge on the structural break." | "Notably, all three VIX regimes exhibit some significance (Normal $p = 0.028$, Elevated $p = 0.043$, Crisis $p = 0.005$), indicating VIX-based regimes detect the relationship more uniformly than our HMM. Both approaches, however, converge on June 1998 as the primary structural break." |

---

## 3. CONSISTENCY ISSUES

### 3.1 Inconsistent Terminology

| Issue | Instances | Recommendation |
|-------|-----------|-----------------|
| **OOS vs. out-of-sample** | Line 34: "OOS"; Line 143: "Frozen OOS"; Line 187–188: "Frozen OOS" | **Standardize**: Define once as "out-of-sample (OOS)" on first mention (line 34), then use "OOS" throughout. Currently acceptable but inconsistent capitalization ("Frozen OOS" vs. "frozen OOS"). |
| **Bonferroni-significant vs. survives Bonferroni** | Lines 89, 249, 474–475, 530, 617 | Mix of "Bonferroni-significant" (adjective) and "survives Bonferroni correction" (verb phrase). **Choose one** for consistency. Recommend: "**survives Bonferroni correction**" as more explicit. Reserve "Bonferroni-significant" for adjective forms (e.g., "the **Bonferroni-significant result**"). |
| **Regime label names** | "Normal" vs "Normal regime" (lines 35, 88, 89 use both) | Standardize: use "**Normal regime**" first mention in a section, then "**Normal**" for brevity. Currently: "HML Granger-predicts SMB **exclusively in the pre-crisis Normal regime**" (good), then "in Normal" (line 102, acceptable). Apply consistently. |
| **Granger causality vs. Granger predicts** | Lines 88, 101, 675–676 use both "Granger-predicts" and "Granger causality" | Recommend: Use "**Granger-predicts**" or "**Granger-causality**" (hyphenated) for adjective; reserve "Granger causality" (unhyphenated) as the noun phrase. Currently: line 88 "Granger-predicts" (good), line 675 "Granger causality" (good). Acceptable as-is, but be consistent in hyphenation: "**Granger-causality**" (adjective/modifier) vs. "**Granger causality**" (noun). |
| **p-values: scientific notation formatting** | Lines throughout: "$p = 8.75 \times 10^{-9}$" vs "$p < 10^{-6}$" vs "$p = 0.022$" | Minor: Scientific notation is used correctly. However, line 41 uses "for 16~years" and later "16 years"—consider standardizing whether numbers are written as numerals or spelled out. Currently: "16~years" (line 41), "16 years" (line 280). **Standardize**: Use "16 years" (no tilde) throughout. |

### 3.2 Inconsistent Statistical Formatting

| Line | Issue | Current | Fix |
|------|-------|---------|-----|
| 246, 345–347, 412–413 | Significance star notation | Uses "$^{**}$", "$^{***}$", "$^{*}$" | Consistent with table captions (lines 349, 392). Acceptable. |
| 279 | Coefficient formatting | "$\hat{\beta}_{\text{HML}}$ shifts from $-0.189$" | Use consistent decimal places: "−0.189" (note: use minus sign, not hyphen). Currently acceptable. |
| 404, 587 | $n$ formatting | "Normal regime ($n = 2{,}485$)" vs "Normal ($n = 4{,}496$)" | Consistent use of grouped thousands. Good. |

### 3.3 Inconsistent Tense Usage

| Line | Issue | Current | Problem | Fix |
|------|-------|---------|---------|-----|
| 79, 83–84 | Past tense (events) | "The August 2007...revealed..." "Quantitative factor models...systematically misestimate..." | Switch between past (historical facts) and present habitual (general practice). Acceptable in academic writing. | No change required; this is standard. |
| 101–102 | Present tense | "HML→SMB predictability is Bonferroni-significant" | Acceptable (describing current finding). | No change. |
| 313–314 | Present tense | "The in-sample Normal result survives every specification change we tested" | Mixes present ("survives") with past ("tested"). Natural; acceptable. | No change. |
| 635–637 | Present tense | "the regime-conditional framework thus excels at informing practitioners *when* to revisit..." | Good use of present tense (general capability). | No change. |

**Summary**: Tense usage is generally consistent and appropriate throughout.

### 3.4 Inconsistent Notation

| Line | Instance | Issue | Recommendation |
|------|----------|-------|-----------------|
| 181, 214 | $z_t$ vs. regime | Definition at line 161: "$z_t \in \{1, \ldots, K\}$ denote the latent regime." Then: "$z_t{=}z_{t-1}$" in tables. | Acceptable; notation is clearly defined. No change needed. |
| 404–405 | Quantile notation | "$\tau \in \{.05,\ldots,.95\}$" | Leading decimal lacks zero: ".05" vs. "0.05". **Standardize to "0.05"** for consistency with other decimal usage (e.g., line 267: "$\Delta R^2 < 0.01\%$"). |

---

## 4. STYLE ISSUES

### 4.1 Passive Voice vs. Active Voice

| Line | Passive Voice Example | Current | Suggested Active |
|------|----------------------|---------|-------------------|
| 84 | "relationships...are [implicitly] misestimated" | "factor models...systematically misestimate dynamics" | Already active; good. |
| 296 | "the structural break...is replicated" | "the structural break replicates cleanly" | Already active; good. |
| 397 | "is revealed [by transfer entropy]" | "Transfer entropy...reveals the reverse channel" | Already active; good. |
| 429 | "is resolved [by quantile Granger]" | "Quantile Granger...resolves the mechanism" | Already active; good. |
| 633 | "does not automatically translate" | "does not automatically translate to improved risk forecasts" | Active; good. |

**Summary**: Paper uses active voice effectively throughout. Minimal passive-voice issues.

### 4.2 Overly Long Sentences (>40 words)

| Line | Word Count | Sentence | Recommendation |
|------|-----------|----------|-----------------|
| 262–265 | ~65 words | "In-sample HAC robustness: Bartlett $B \in \{1,\ldots,30\}$: $p \in [3.2 \times 10^{-9}, 2.1 \times 10^{-8}]$; Parzen: $p \in [4.1 \times 10^{-9}, 5.7 \times 10^{-8}]$; QS: $p \in [5.9 \times 10^{-9}, 8.8 \times 10^{-8}]$. All 90 kernel–bandwidth combinations yield $p < 10^{-7}$." | Complex but acceptable (footnote format allows technical density). Consider breaking into: "For Bartlett kernels ($B \in \{1,\ldots,30\}$), $p \in [3.2 \times 10^{-9}, 2.1 \times 10^{-8}]$. Parzen and QS yields are slightly higher, but all 90 kernel–bandwidth combinations remain below $p < 10^{-7}$." |
| 313–315 | ~60 words | "The in-sample Normal result survives every specification change we tested: *lag structure*---significant at all lags 1--15 ($p < 10^{-4}$), ruling out a lag-1 artifact; *common drivers*---trivariate MKT-RF controls..." | Acceptable (itemized; easy to scan). No change needed. |
| 356–359 | ~52 words | "LSTM attention concentrates 68.2\% on lag~1 in Normal, decaying monotonically to 52.9\% (Elevated) and 44.2\% (Crisis, approaching uniform baseline $1/9 = 11.1\%$)---mirroring the Granger structural break at the mechanism level." | **Consider splitting**: "LSTM attention concentrates 68.2% on lag-1 in Normal regimes, declining to 52.9% in Elevated and 44.2% in Crisis. This monotonic decay toward the uniform baseline (11.1%) mirrors the structural break in Granger significance." |
| 678–680 | ~44 words | "Trivariate controls (MKT-RF) address the most prominent common driver ($F$-$p > 0.43$), but a full 6-factor VAR ($324$ parameters per regime) is under-identified at $n \approx 1{,}000$; post-double-selection methods~\cite{hecq2023granger} could address this." | Acceptable (complex trade-off; clearly explained). No change required. |

### 4.3 Redundant Phrases

| Line | Phrase | Issue | Fix |
|------|--------|-------|-----|
| 41–42 | "consistent with zero for 16~years. Transfer entropy reveals..." | Line break creates slight abruptness; "consistent with zero" + subsequent sentence about alternative mechanisms feels disconnected. | Insert transition: "...for 16 years. **However**, transfer entropy reveals..." or "...for 16 years. **Paradoxically**, transfer entropy reveals..." |
| 86–87 | "This paper documents...Using daily...returns...we show..." | "documents" and "we show" are near-synonymous. | Tighten: "**This paper documents** structural decay...using daily returns (1990--2024). **Specifically, we show**..." — or combine: "Using daily returns, we document and quantify the structural decay of HML→SMB..." |
| 113–114 | "Effect sizes are modest...the contribution is diagnostic, not tradable alpha." | "modest" (quantitative) + "not tradable" (qualitative) is somewhat redundant. | Refactor: "Effect sizes ($\Delta R^2 \approx 2\%$, Sharpe $= -0.07$) generate no excess returns and do not improve VaR coverage, confirming the contribution is diagnostic—illuminating regime transitions rather than enabling profitable trading." |
| 720–721 | "Implications. Factor-timing models...may misspecify dynamics during structural transitions." | "misspecify dynamics during structural transitions" is already covered implicitly by the paper's findings. | Consider: "Factor-timing models assuming regime-invariant cross-factor relationships may overlook the **breakdown** dynamics documented here, leading to underestimation of risk during structural transitions." |

### 4.4 Overly Complex Phrasing / Wordiness

| Line | Current | Simplification |
|------|---------|-----------------|
| 195 | "an economic prior (value-size institutional overlap), not empirical dominance" | "**an economic hypothesis** (institutional overlap between value and size), not empirical ranking" or "**economic intuition** rather than statistical evidence of dominance" |
| 322–323 | "so hard Viterbi assignments are not driving the result" | "confirming that hard regime labels are robust and not driving the result" |
| 333–334 | "But MSE and mutual information are distinct: a channel may contain nonlinear dependencies without improving point forecasts." | "**However**, improved mutual information does not guarantee better point predictions: nonlinear dependencies can exist without reducing forecast error (MSE)." |
| 366–367 | "The ``purely linear'' characterization is therefore *fit-dependent*; the linear--nonlinear boundary should be treated as exploratory." | "**Thus**, the classification as 'purely linear' **depends on HMM fit**; treat the linear–nonlinear boundary as tentative." |

---

## 5. MINOR FORMATTING & TYPOGRAPHY ISSUES

### 5.1 Inconsistent Em-Dash vs. Hyphen Usage

| Line | Issue | Current | Fix |
|------|-------|---------|-----|
| 22, 79, 120, 128, 131, etc. | Em-dash usage | "---" (triple dash, representing em-dash in LaTeX) | Verify: LaTeX renders "---" as em-dash. Generally correct throughout. Acceptable. |
| 268, 273, 400 | En-dash (range) | "1990--2024", "1998--2003" | Correct use of en-dash for ranges. Consistent. |

### 5.2 Inconsistent Space/Tilde Usage

| Line | Issue | Current | Fix |
|------|-------|---------|-----|
| 41 | "16~years" | Uses non-breaking space (tilde) | Line 280 uses "16 years" (regular space). **Standardize**: Use "16 years" without tilde, or consistently use tilde for all number–unit pairs (e.g., "3~years", "50~seeds"). Recommend: **remove all tildes** before year/unit labels for consistency with modern academic style. |
| 80, 151 | "three days~\cite" | Tilde before citation | Correct in LaTeX (prevents line break). Acceptable. |

### 5.3 Spacing & Punctuation in Math/Statistics

| Line | Issue | Current | Recommendation |
|------|-------|---------|-----------------|
| 43 | "$z = 5.37$ vs.\ forward $z = 2.45$" | Uses "vs.\" (backslash-escaped period) | Correct (prevents period from ending sentence). Good practice. |
| 259–260 | "$[3.2 \times 10^{-9},\; 8.8 \times 10^{-8}]$" | Semicolon inside brackets with thin space | Acceptable, but consider consistency: use consistent separator (either comma or semicolon throughout ranges). Currently uses both. **Standardize**: Use comma and space "$[3.2 \times 10^{-9}, 8.8 \times 10^{-8}]$" (simpler). |
| 405 | "$\tau \in \{.05,\ldots,.95\}$" | Leading decimal ".05" | Use "0.05" for consistency (seen elsewhere, e.g., line 267). |

---

## 6. SUGGESTIONS FOR ENHANCED CLARITY (Non-Critical)

### 6.1 Definitions That Could Be Moved Earlier

| Concept | First Use | First Definition | Suggestion |
|---------|-----------|------------------|-----------|
| Frozen OOS | Line 34 (Abstract) | Line 187 | Add brief definition in Abstract: "frozen-parameter out-of-sample validation (parameters fixed to training period)" |
| Transfer entropy | Line 42 (Abstract) | Line 145, 396 | Add one-line definition in Abstract or Section 3 intro. |
| HMM | Line 33 (Abstract) | Line 160 (Student-$t$ HMM detailed) | "Student-$t$ Hidden Markov Model" first mention already acceptable. |
| Quandt-Andrews sup-$F$ | Line 39 (Abstract) | Line 270 | Acceptable; formally introduced when results presented. |

### 6.2 Potential Points of Confusion

| Line | Potential Confusion | Clarification |
|-------|-------------------|---------------|
| 46–48 | "does not survive Bonferroni correction and reflects regime redistribution" | Clarify: The signal does not replicate because the test-period sample contains different regime proportions, not because the underlying relationship is different. |
| 320–321 | "filtered vs. smoothed probabilities" | Consider adding: "standard Viterbi forward-backward smoothing algorithm" for clarity. |
| 438 | "This tail mechanism is pair-specific" | Emphasize: SMB→HML is the only pair among top-4 heterogeneous pairs showing tail dependence. |

---

## 7. SUMMARY TABLE: PRIORITY-RANKED EDITS

### **HIGH PRIORITY** (Clarity/Accuracy)

| # | Line | Issue | Fix |
|---|------|-------|-----|
| 1 | 46–48 | Ambiguous "regime redistribution" | Clarify: "The OOS signal appears in Elevated regime due to post-GFC classification shifts (Elevated prevalence doubles from 13.7% training to 33.7% test), not replication of the in-sample effect." |
| 2 | 106–107 | "directional asymmetry" unclear antecedent | Rewrite: "reveals **an asymmetry in causal direction**: forward (HML→SMB) is linear, while reverse (SMB→HML) exhibits nonlinear tail dependence." |
| 3 | 153–154 | Scale sensitivity confusing | Clarify: "Granger $F$-statistics remain scale-invariant, but HMM regime assignments shift across scaling conventions, affecting only out-of-sample classification—not in-sample statistical tests." |
| 4 | 356–359 | Long LSTM sentence; hard to parse | Split: "LSTM attention concentrates 68.2% on lag-1 in Normal, declining to 52.9% (Elevated) and 44.2% (Crisis). This decay approaches the uniform baseline (11.1%), mirroring the Granger structural break." |
| 5 | 34 | "frozen-parameter OOS" undefined in Abstract | Add: "frozen-parameter out-of-sample validation (parameters fixed to training-period estimates)" |

### **MEDIUM PRIORITY** (Style/Consistency)

| # | Line | Issue | Fix |
|---|------|-------|-----|
| 6 | 405 | Decimal formatting ".05" | Change to "0.05" for consistency |
| 7 | 41, 280 | Tilde usage "16~years" vs "16 years" | Standardize: remove tilde; use "16 years" |
| 8 | 249, throughout | Inconsistent "survives Bonferroni" vs "Bonferroni-significant" | Standardize terminology (recommend "survives Bonferroni correction") |
| 9 | 86–87 | Redundant "documents"/"we show" | Tighten: "**Using** daily returns (1990–2024), **we document** structural decay of HML→SMB predictability: Bonferroni-significant in Normal (pre-2008) but absent post-2008." |
| 10 | 41–42 | Abrupt transition: "...16 years. Transfer entropy..." | Insert transition: "...for 16 years. **Yet** transfer entropy reveals..." |

### **LOW PRIORITY** (Minor/Optional)

| # | Line | Issue | Fix |
|---|------|-------|-----|
| 11 | 259–260 | Semicolon in range notation | Standardize to comma: "$[3.2 \times 10^{-9}, 8.8 \times 10^{-8}]$" |
| 12 | 171–172 | Seed selection confusing | Add: "(selected by convention among tied seeds)" |
| 13 | 438–439 | "pair-specific" scope | Clarify: "unique to the SMB→HML pair" |
| 320–321 | "hard Viterbi assignments" not defined | Add phrase: "hard (maximum-likelihood) regime assignments" |

---

## CONCLUSION

**Overall Assessment**: The paper is well-organized, technically rigorous, and clearly written for a specialized audience. The main issues are:

1. **Clarity**: A few technical concepts (frozen OOS, regime redistribution, tail mechanism) need tighter definition.
2. **Consistency**: Minor standardization needed in terminology ("Bonferroni-significant" vs "survives Bonferroni"), notation (0.05 vs .05), and spacing (16 years vs 16~years).
3. **Style**: Two long sentences (lines 356–359, 262–265) could be split for readability; one redundancy (lines 86–87) could be tightened.

**Recommended Action Plan**:
1. Implement HIGH priority fixes (5 edits) for clarity.
2. Implement MEDIUM priority fixes (5 edits) for consistency and flow.
3. Review LOW priority fixes as time permits; most are optional refinements.

**Estimated revision time**: 20–30 minutes to implement all high and medium priority edits.
