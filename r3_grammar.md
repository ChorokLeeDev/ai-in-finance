# Grammar, Punctuation, and Language Quality Review
## ICAIF 2026 Submission: "Structural Decay of Cross-Factor Predictability"

---

## ISSUES FOUND

### 1. **Line 34: Ambiguous hyphenation**
- **Text:** "linear--nonlinear boundary"
- **Issue:** Inconsistent em-dash usage; "linear--nonlinear" uses double-dash (--) but should be either spaced em-dash with text or consistent en-dash style.
- **Severity:** LOW
- **Note:** This appears again at line 136 and 417. While LaTeX rendering handles --, best practice is consistency and clarity.

---

### 2. **Line 42: Subject-verb agreement with numeric reference**
- **Text:** "post-2008, the relationship has been consistent with zero for 16~years."
- **Issue:** "relationship...has been" is grammatically correct, but the phrasing is awkward. More precise: "post-2008, the coefficient has remained consistent with zero for 16 years" or "the relationship has shown no significant effect for 16 years."
- **Severity:** LOW
- **Note:** Current phrasing is acceptable but slightly imprecise for academic rigor.

---

### 3. **Line 43: Dangling/misplaced modifier + awkward structure**
- **Text:** "Transfer entropy additionally reveals a stronger nonlinear reverse channel SMB→HML (z = 5.37 vs.\ forward z = 2.45),"
- **Issue:** The phrase "vs. forward z = 2.45" creates ambiguity—does "forward" modify "z" or the understood direction "HML→SMB"? Better: "vs. forward (z = 2.45)," with clearer punctuation or "vs. the forward direction (z = 2.45),".
- **Severity:** MEDIUM
- **Note:** Reader must infer "forward direction"; could cause confusion about which comparison is being made.

---

### 4. **Line 48: Article misuse**
- **Text:** "A frozen out-of-sample (OOS) test yields an exploratory Elevated-regime signal"
- **Issue:** "Elevated-regime signal" should use lowercase "elevated" unless it's a proper noun. Currently, "Elevated" is capitalized as part of a regime label. Check consistency: elsewhere (line 501) it's "Elevated (share doubles)" — lowercase context suggests inconsistency. If "Elevated" is a defined regime NAME, capitalization is correct; if it's a descriptor, it should be lowercase.
- **Severity:** MEDIUM
- **Note:** Requires author decision on regime nomenclature consistency throughout (see lines 196, 224, 235, 501, 504, 506, 562, 580, etc.).

---

### 5. **Line 49: Comma splice**
- **Text:** "Bonferroni correction and reflects regime redistribution rather than independent replication. A secondary pair (MOM→SMB) achieves"
- **Issue:** Not a splice per se, but the sentence before it (line 48–49) could be stronger: "...does not survive Bonferroni correction, and reflects regime redistribution rather than independent replication." Current version reads as two separate thoughts; adding "and reflects" after "correction" improves flow.
- **Severity:** LOW
- **Note:** Actually acceptable as written; flagging for clarity only.

---

### 6. **Line 52–54: Parallel structure violation**
- **Text:** "VIX-tercile validation confirms the structural break under a regime definition entirely external to factor returns. International replication confirms structural breaks in all four non-US markets tested."
- **Issue:** Two consecutive sentences both begin with "[X] confirms" but with different objects (singular "break" vs. plural "breaks"). While not grammatically wrong, the asymmetry weakens parallel structure. Better: "VIX tercile validation confirms the structural break under an external regime definition. International replication confirms structural breaks in all four non-US markets."
- **Severity:** LOW
- **Note:** Stylistic; both sentences are grammatically sound.

---

### 7. **Line 86–87: Subject-verb agreement under complex modification**
- **Text:** "Quantitative factor models that assume regime-invariant cross-factor relationships systematically misestimate dynamics during structural transitions."
- **Issue:** Grammatically correct ("models...misestimate"), but the abstract noun "dynamics" after "misestimate" is vague. Consider: "systematically misestimate how dynamics evolve" or "systematically misestimate relationship dynamics."
- **Severity:** LOW
- **Note:** Clarity issue, not grammar error.

---

### 8. **Line 89: Stylistic but grammatically sound**
- **Text:** "\textbf{This paper documents structural decay of cross-factor predictability.}"
- **Issue:** Starting a bold sentence with "This paper" immediately after an introductory paragraph is slightly repetitive (prior sentence discusses "Using daily Fama-French returns..."). Could vary: "We document structural decay..." or simply "Structural decay of cross-factor predictability is the focus of this analysis."
- **Severity:** LOW
- **Note:** Not a grammar error; style observation.

---

### 9. **Line 96: Missing article before "Tier"**
- **Text:** "\textbf{Evidence hierarchy.} We distinguish three tiers: (1)~\emph{primary} (in-sample Normal-regime structural break, VIX-validated, robust across all specifications);"
- **Issue:** "three tiers" is correct, but the list that follows uses numerals (1), (2), (3). Parallelism note: list items are parenthetical fragments, not full clauses. This is acceptable in technical writing but inconsistent with the explicit "(1)~\emph{primary}..." format.
- **Severity:** LOW
- **Note:** Acceptable; flagging for consistency.

---

### 10. **Line 99: Ambiguous pronoun reference**
- **Text:** "(2)~\emph{confirmatory} (MOM→SMB OOS replication, international results);"
- **Issue:** "OOS" and "international" are noun adjuncts, not verbs. The structure is correct, but parenthetical labeling as "confirmatory" without a verb creates implicit parallelism: "confirmatory [evidence of] MOM→SMB OOS replication." This is acceptable in technical writing.
- **Severity:** LOW
- **Note:** Acceptable modern technical prose.

---

### 11. **Line 101: Article usage with "Tiers"**
- **Text:** "The contribution rests on Tiers~1--2; Tier~3 is reported for transparency, not claimed as validation."
- **Issue:** "Tiers 1–2" (plural) vs. "Tier 3" (singular) is correct; no issue here. But the parallel structure "rests on [Tiers 1–2]" vs. "is reported for" creates a comparison of different grammatical structures. Better: "The primary contribution (Tiers 1–2) rests on [X]; Tier 3 is reported for transparency, not validation."
- **Severity:** LOW

---

### 12. **Line 109: Misplaced comma in list**
- **Text:** "(ii)~A complexity diagnostic (OLS, Random Forest (RF), MLP, LSTM) combined with transfer entropy reveals a directional asymmetry (linear forward, nonlinear reverse via tail dependence) not captured by conditional-mean methods."
- **Issue:** "Random Forest (RF)" uses parenthetical abbreviation; list is: "OLS, Random Forest (RF), MLP, LSTM." Standard form would be either "Random Forest (RF)" or "RF (Random Forest)" but not both in same position. Current is acceptable but inconsistent with "MLP" and "LSTM" abbreviations not explained. However, this is acceptable technical writing.
- **Severity:** LOW
- **Note:** No grammar error; style consistency observation.

---

### 13. **Line 110–111: Awkward phrasing / unclear pronoun reference**
- **Text:** "with transfer entropy reveals a directional asymmetry (linear forward, nonlinear reverse via tail dependence) not captured by conditional-mean methods. The conceptual contribution: \emph{regime heterogeneity} (between-regime variation) and \emph{quantile heterogeneity} (within-regime tail dependence) are distinct phenomena"
- **Issue:** Sentence structure becomes complicated. More precisely: "...reveals a directional asymmetry (linear forward, nonlinear reverse, via tail dependence) not captured by conditional-mean methods. The key conceptual contribution is that regime heterogeneity and quantile heterogeneity are distinct phenomena—the former is systematic, the latter pair-specific."
- **Severity:** MEDIUM
- **Note:** Awkward phrasing and unclear conceptual flow.

---

### 14. **Line 113–115: Comma splice / run-on sentence**
- **Text:** "The conceptual contribution: \emph{regime heterogeneity} (between-regime variation) and \emph{quantile heterogeneity} (within-regime tail dependence) are distinct phenomena---the former is systematic, the latter pair-specific."
- **Issue:** Using em-dash (---) to connect independent clauses is acceptable in formal writing, but the structure is punctuated as a label ("The conceptual contribution:") followed by a comma-spliced sentence with em-dash. Better: "The conceptual contribution is that regime heterogeneity and quantile heterogeneity are distinct phenomena: the former is systematic; the latter pair-specific."
- **Severity:** MEDIUM
- **Note:** Current version is acceptable but awkwardly punctuated.

---

### 15. **Line 118: Incomplete noun phrase**
- **Text:** "Effect sizes are modest ($\Delta R^2 \approx 2\%$, Sharpe ratio $= -0.07$)"
- **Issue:** "Effect sizes are modest" but values provided are for two different metrics (R² and Sharpe). Better clarity: "Effect sizes are modest ($\Delta R^2 ≈ 2\%; Sharpe ratio = −0.07$), indicating modest economic magnitude."
- **Severity:** LOW
- **Note:** Acceptable; clarity suggestion.

---

### 16. **Line 127–128: Awkward pronoun usage**
- **Text:** "Psaradakis et al.~\cite{psaradakis2005markov} pioneer regime-switching Granger; we extend with Student-$t$ HMMs~\cite{bulla2011hidden},"
- **Issue:** "Psaradakis et al. pioneer" (present tense, historical reference) followed by "we extend" (present tense, contemporary). This is acceptable in academic writing (citing historical work), but could be more explicit: "Psaradakis et al. pioneered regime-switching Granger; we extend this work with..."
- **Severity:** LOW
- **Note:** Tense usage is defensible but could be clearer.

---

### 17. **Line 131–133: Parallel structure violation**
- **Text:** "Tank et al.~\cite{tank2022neural} extend Granger to nonlinear settings; Diebold and Yilmaz~\cite{diebold2012better} develop VAR connectedness; neither conditions on latent regime state."
- **Issue:** "Tank et al. extend...Granger; Diebold and Yilmaz develop...VAR; neither conditions" — the subject "neither" is singular but refers to two groups. Better: "neither conditions" is correct (neither = singular), but the referent is ambiguous because three subjects are present (Tank et al., Diebold and Yilmaz, and the unnamed third comparison). Actually, "neither" properly refers to the two cited works, so grammatically this is correct. No error.
- **Severity:** NONE (acceptable)

---

### 18. **Line 143–147: Algorithm step numbering + comma placement**
- **Text:** "REQUIRE Multivariate returns $\{\mathbf{x}_t\}_{t=1}^T$, regime count $K$" and subsequent STATE lines.
- **Issue:** In the algorithmic environment, "REQUIRE" is a keyword but the parameter list uses commas without "and." This is standard algorithmic pseudocode, so acceptable.
- **Severity:** NONE

---

### 19. **Line 145: Subject-verb agreement in algorithmic context**
- **Text:** "STATE \textbf{Regime discovery:} Fit Student-$t$ HMM ($K$ states, $M = 50$ random starts); select $K$ via BIC on training data"
- **Issue:** "Fit...HMM...select $K$" — imperative verbs in algorithmic pseudocode. Correct.
- **Severity:** NONE

---

### 20. **Line 155–157: Article and number agreement**
- **Text:** "Daily returns for six Fama-French factors~\cite{fama2015five} plus Momentum~\cite{carhart1997persistence} (1990--2024, 8,817 trading days)."
- **Issue:** "six Fama-French factors plus Momentum" = 7 total factors, but this is not explicitly stated. The count is implicitly understood. Acceptable.
- **Severity:** NONE

---

### 21. **Line 157–159: Dangling modifier / ambiguous reference**
- **Text:** "We adopt a percentage-unit convention. Granger $F$-statistics are scale-invariant (they test $\beta = 0$ regardless of scaling), but HMM emission probabilities are not, so regime boundaries differ across conventions."
- **Issue:** "they test β = 0" — the pronoun "they" (F-statistics) is clear, and the parenthetical explanation is grammatically correct. No error.
- **Severity:** NONE

---

### 22. **Line 160: Ambiguous referent**
- **Text:** "Under percentage units, the frozen OOS yields $n = 953$ Elevated-regime days"
- **Issue:** "Elevated-regime days" — is "Elevated" capitalized because it's a defined regime label (lines 173, 224 confirm "Elevated" is a regime name)? If so, acceptable. Consistency check needed (see Issue #4).
- **Severity:** MEDIUM (pending regime label decision)

---

### 23. **Line 162–163: Tense inconsistency**
- **Text:** "The primary contribution (in-sample finding, structural break, VIX validation) is scale-invariant; scale sensitivity affects only the exploratory OOS result."
- **Issue:** Present tense "is scale-invariant" and "affects" — consistent. Acceptable.
- **Severity:** NONE

---

### 24. **Line 170: Hyphenation of compound number**
- **Text:** "$K = 3$ is pre-specified by BIC"
- **Issue:** "pre-specified" is hyphenated as a compound adjective before the noun "BIC." Should check: "pre-specified" is a past participle + prefix, correct form is either "pre-specified" or "prespecified" (no hyphen in modern usage). Current is acceptable but "prespecified" is more standard in technical writing.
- **Severity:** LOW

---

### 25. **Line 171: Number formatting with comma**
- **Text:** "$\Delta\text{BIC} = 1{,}680$ over $K = 2$"
- **Issue:** "1{,}680" uses {,} (braced comma) which is LaTeX for escaped comma in math mode. Correct usage for clarity in mathematical text.
- **Severity:** NONE (correct LaTeX)

---

### 26. **Line 178–179: Incomplete sentence / parenthetical overload**
- **Text:** "Under the BIC-optimal fit, ``Crisis'' denotes a high-kurtosis statistical state (0\% of 2008 GFC assigned)"
- **Issue:** Parenthetical remark is grammatically complete but lacks a main clause continuation. The sentence continues: "Under the BIC-optimal fit, 'Crisis' denotes a high-kurtosis statistical state (0% of 2008 GFC assigned); sensitivity fits aligning with calendar crises use Cluster 5..." This is a semicolon-joined compound sentence, acceptable.
- **Severity:** NONE

---

### 27. **Line 180: Comma splice**
- **Text:** "sensitivity fits aligning with calendar crises use Cluster~5 ($\Delta\text{BIC} = 218$, 90\% GFC detection; see Table~\ref{tab:optima})."
- **Issue:** The phrase "sensitivity fits aligning with calendar crises use Cluster 5" is grammatically correct (subject: "sensitivity fits"; verb: "use"; object: "Cluster 5"). The internal parenthetical is punctuated with comma then semicolon, which is acceptable for a complex parenthetical.
- **Severity:** NONE

---

### 28. **Line 185–186: Formatting of hypothesis in mathematical notation**
- **Text:** "we extract $\mathcal{T}_k = \{t : \hat{z}_t = k\}$ and test $H_0$: $r_{\text{SMB},t} \perp \{r_{\text{HML},t-\ell}\} \mid \{r_{\text{SMB},t-\ell}\}$"
- **Issue:** "H_0:" uses colon after the hypothesis symbol. Should this be: "We test $H_0$: $r_{\text{SMB},t} \perp \{r_{\text{HML},t-\ell}\} \mid \{r_{\text{SMB},t-\ell}\}$"? Current is acceptable (colon introduces the hypothesis statement).
- **Severity:** NONE

---

### 29. **Line 187: Citation format with verb**
- **Text:** "with Andrews~\cite{andrews1991heteroskedasticity} HAC standard errors."
- **Issue:** "Andrews" is a surname; when used with a citation in this position, it should read: "with Andrews (1991) HAC standard errors" or "with HAC standard errors per Andrews (1991)" or "with HAC standard errors~\cite{andrews1991heteroskedasticity}." Current phrasing puts the author before the method, which is slightly awkward. Better: "with HAC standard errors per Andrews~\cite{andrews1991heteroskedasticity}."
- **Severity:** LOW

---

### 30. **Line 193: Hyphenation of compound adjective**
- **Text:** "\textbf{Circularity mitigation.}"
- **Issue:** Correct label. No issue.
- **Severity:** NONE

---

### 31. **Line 194–195: Sentence fragment**
- **Text:** "(1)~\emph{Frozen OOS:} HMM trained 1990--2012, all parameters frozen, applied to 2013--2024 without refitting."
- **Issue:** This is a sentence fragment (no main verb; "applied" is a past participle, not finite verb). Should be: "HMM trained 1990–2012 with all parameters frozen, then applied to 2013–2024 without refitting." Or: "The HMM, trained 1990–2012 with all parameters frozen, is applied to 2013–2024 without refitting."
- **Severity:** MEDIUM
- **Note:** Acceptable in technical writing as a labeled item, but strictly a fragment.

---

### 32. **Line 196–197: Parallel structure**
- **Text:** "(2)~\emph{VIX external instrument:} CBOE Volatility Index (VIX) terciles (Normal $<$15, Elevated 15--21, Crisis $>$21) replace HMM labels entirely."
- **Issue:** "replace" is a finite verb; this item has a main clause. Parallel with Item 1: Item 1 is a fragment, Item 2 is a complete clause. For consistency, both should be fragments: "CBOE Volatility Index (VIX) terciles (Normal <15, Elevated 15–21, Crisis >21) replacing HMM labels entirely."
- **Severity:** MEDIUM

---

### 33. **Line 198: Sentence fragment continued**
- **Text:** "(3)~\emph{Permutation test:} 50,000 label shuffles within regime ($p = 0.022$)."
- **Issue:** Sentence fragment (no verb). Should read: "50,000 label shuffles within regime, yielding $p = 0.022$."
- **Severity:** MEDIUM
- **Note:** Items (1), (2), (3) are inconsistent in grammatical structure.

---

### 34. **Line 199–200: Missing comma in complex subject phrase**
- **Text:** "pair selection transparency."
- **Issue:** OK as subheading.
- **Severity:** NONE

---

### 35. **Line 201–202: Comma usage in compound predicate**
- **Text:** "HML--SMB was selected post-hoc from screening 30 in-sample pairs (not pre-registered). This focus reflects an economic prior (value-size institutional overlap), not empirical dominance"
- **Issue:** "This focus reflects...not empirical dominance" — the structure is correct; a comma before "not" would be incorrect (no independent clause follows). Correct as written.
- **Severity:** NONE

---

### 36. **Line 203–204: Subject-verb agreement with numerical comparison**
- **Text:** "---MOM→SMB is the top OOS pair ($F = 20.3$ vs.\ $9.06$)."
- **Issue:** The comparison omits the second $F$ label for clarity. "vs. 9.06" is implied to be "vs. HML→SMB $F = 9.06$." Acceptable in technical writing with context.
- **Severity:** NONE

---

### 37. **Line 204–205: Parallel structure in conditional statement**
- **Text:** "Under 30-pair Benjamini-Hochberg FDR, no OOS pair survives; HML→SMB ranks 2nd by $F$-statistic."
- **Issue:** "no OOS pair survives" (predicate) but "HML→SMB ranks" (different predicate structure). Acceptable; not a formal parallel structure issue.
- **Severity:** NONE

---

### 38. **Line 212: Pronoun-antecedent agreement**
- **Text:** "Figure~\ref{fig:timeline} shows regime assignments with crisis events marked."
- **Issue:** "with crisis events marked" — passive voice, acceptable. No issue.
- **Severity:** NONE

---

### 39. **Line 223: Table formatting / number agreement**
- **Text:** Table headers: "Regime & Days & Prop. & Mean $\|\mathbf{x}\|$ (\%) & $\hat{\nu}$ & $P(z_t{=}z_{t-1})$ \\"
- **Issue:** "{=}" is escaped equals in LaTeX. Acceptable. All headers are singular or properly formatted mathematical notation.
- **Severity:** NONE

---

### 40. **Line 264: Article usage before "Normal-regime"**
- **Text:** "Only Normal-regime HML→SMB survives correction"
- **Issue:** "Normal-regime" as compound adjective before noun "HML→SMB" — should the noun be explicit? "Only Normal-regime [result for] HML→SMB" or simply "Only the Normal-regime result for HML→SMB survives correction."
- **Severity:** LOW

---

### 41. **Line 265–266: Hyphenation in technical term**
- **Text:** "Bartlett, Parzen, and Quadratic Spectral kernels at bandwidths 1--30"
- **Issue:** "Quadratic Spectral" — should this be "Quadratic-Spectral"? Check standard technical terminology. Current style (two separate words) is more common in econometrics literature. Acceptable.
- **Severity:** NONE

---

### 42. **Line 267: Abbreviation formatting**
- **Text:** "[3.2 \times 10^{-9},\; 8.8 \times 10^{-8}]"
- **Issue:** Range formatting with semicolon and thinspace is acceptable LaTeX, but could be clearer: "from $3.2 \times 10^{-9}$ to $8.8 \times 10^{-8}$" or "[$3.2 \times 10^{-9}, 8.8 \times 10^{-8}$]" (without the thinspace). Current is acceptable.
- **Severity:** NONE

---

### 43. **Line 268: Footnote reference placement**
- **Text:** "worst case at Quadratic Spectral $B = 30$).\footnote{In-sample HAC robustness: ...}"
- **Issue:** Footnote is correctly placed after the closing parenthesis. Acceptable.
- **Severity:** NONE

---

### 44. **Line 273–274: Hyphenation in year range**
- **Text:** "Pre-2008 Normal ($n = 3{,}140$): $p = 6.66 \times 10^{-16}$ ($\Delta R^2 = 2.06\%$). Post-2008 Normal ($n = 1{,}557$): $p = 0.73$ ($\Delta R^2 < 0.01\%$)"
- **Issue:** "Pre-2008" and "Post-2008" — hyphenated compounds before noun. Should these be "Pre–2008" (en-dash) or "Pre-2008" (hyphen)? Current style (hyphen) is acceptable for compound adjectives, though "Pre–2008" might be slightly more standard. Acceptable as written.
- **Severity:** LOW

---

### 45. **Line 281: Comma in complex list**
- **Text:** "The Quandt-Andrews sup-$F$~\cite{andrews1993tests} identifies \textbf{June 1998} as the primary break (supremum $F = 21.2$, $p = 1.23 \times 10^{-13}$)"
- **Issue:** Two-part parenthetical: "$F = 21.2$, $p = 1.23 \times 10^{-13}$" — comma correctly separates the two test statistics. Acceptable.
- **Severity:** NONE

---

### 46. **Line 282: List of years without "and"**
- **Text:** "the top-5 candidates all cluster in 1998--2003 (June 1998, July 1998, April 1998, August 2003, March 1998)"
- **Issue:** List of five items in parentheses separated by commas. The final item "March 1998" should ideally have an "or" before it if these are alternatives, or remain as a simple comma-separated list. Current style (simple list without Oxford comma before last item) is acceptable in technical writing.
- **Severity:** LOW

---

### 47. **Line 287: Bracket notation in formula**
- **Text:** "$\hat{\beta}_{\text{HML}}$ shifts from $-0.189$ (pre-GFC) to $+0.010$ (post-GFC, Wald $z = 5.05$, $p = 9.2 \times 10^{-7}$)."
- **Issue:** Nested parenthetical in parenthetical is acceptable. The comma in "Wald $z = 5.05$, $p = 9.2 \times 10^{-7}$" correctly separates two test statistics. No issue.
- **Severity:** NONE

---

### 48. **Line 293: Article usage**
- **Text:** "Together, the evidence documents a two-stage structural break"
- **Issue:** "a two-stage" — correct article usage before compound adjective. Acceptable.
- **Severity:** NONE

---

### 49. **Line 306–307: Colon usage in complex sentence**
- **Text:** "Replacing HMM labels with VIX terciles (entirely external to factor returns): pre-2008 VIX-Normal $p < 0.0001$ ($F = 18.6$), post-2008 $p = 0.714$ ($F = 0.13$)."
- **Issue:** Colon after "Replacing..." introduces the results, which is acceptable. However, the structure is: "Replacing [X]: [result 1], [result 2]." Could be clearer as: "Replacing HMM labels with VIX terciles (entirely external to factor returns) yields: pre-2008 VIX-Normal $p < 0.0001$, post-2008 $p = 0.714$." Current version is acceptable but slightly awkward.
- **Severity:** LOW

---

### 50. **Line 309: Pronoun reference**
- **Text:** "confirming the finding is not a circularity artifact."
- **Issue:** "the finding" — clear referent (structural break in previous sentences). No issue.
- **Severity:** NONE

---

### 51. **Line 324–325: "Robustness" section heading and opening**
- **Text:** "\textbf{Robustness} (Figure~\ref{fig:lag}). The in-sample Normal result survives every specification change we tested:"
- **Issue:** Acceptable structure. The label and figure reference are followed by a colon explaining what follows. Grammatically correct.
- **Severity:** NONE

---

### 52. **Line 325–327: Semicolon in complex list**
- **Text:** "The in-sample Normal result survives every specification change we tested: \emph{lag structure}---significant at all lags 1--15 ($p < 10^{-4}$), ruling out a lag-1 artifact;"
- **Issue:** Semicolon ends the first item in a list of numbered properties. The list uses em-dashes and semicolons. For clarity: each list item should end with a semicolon except the last (which ends with a period). Current structure is acceptable but could be clearer.
- **Severity:** LOW

---

### 53. **Line 328: Missing comma in list**
- **Text:** "\emph{common drivers}---trivariate MKT-RF controls add no incremental content ($F$-$p > 0.43$), so the signal is not proxying for market risk;"
- **Issue:** The sentence structure is: "common drivers — [test result], so [interpretation]." Acceptable punctuation.
- **Severity:** NONE

---

### 54. **Line 330: Hyphenation**
- **Text:** "\emph{regime definition}---robust across all 7 local-optima clusters"
- **Issue:** "local-optima" is hyphenated as a compound noun modifier. Should be "local optima" (two words) or "local-optima" (hyphenated). As a compound adjective, hyphenation is standard. Acceptable.
- **Severity:** NONE

---

### 55. **Line 332: Comma in coordinate structure**
- **Text:** "labels}---filtered vs.\ smoothed probabilities agree 95.9\% of days, and posterior-weighted (soft-label) Granger yields $p < 10^{-7}$"
- **Issue:** Comma before "and" in "agree 95.9% of days, and posterior-weighted...yields" — this coordinates two independent clauses, so a comma before "and" is correct. Acceptable.
- **Severity:** NONE

---

### 56. **Line 333–334: Viterbi reference**
- **Text:** "so hard Viterbi assignments are not driving the result."
- **Issue:** "Viterbi" is a proper noun (algorithm name); capitalization is correct. "hard Viterbi" vs. "soft Viterbi" (implied earlier as "soft-label Granger") — terminology is consistent. No issue.
- **Severity:** NONE

---

### 57. **Line 335: Dangling modifier check**
- **Text:** "Rolling 3-year unconditional Granger analysis (Figure~\ref{fig:rolling}) shows episodic significance peaks during stress periods"
- **Issue:** "Rolling 3-year...analysis shows" — subject is "analysis" (singular), verb is "shows" (singular). Grammatically correct.
- **Severity:** NONE

---

### 58. **Line 341: Hyphenation of compound**
- **Text:** "Is the Normal-regime channel linear"
- **Issue:** "Normal-regime" — compound adjective before noun "channel." Hyphenated correctly.
- **Severity:** NONE

---

### 59. **Line 344–346: Parallel structure in two-stage plan**
- **Text:** "We use a two-stage diagnostic to map this complexity boundary: (1)~four model classes test whether nonlinear methods improve MSE-based prediction; (2)~transfer entropy and quantile Granger detect directed information flow that conditional-mean tests miss."
- **Issue:** Item (1): "four model classes test whether..." (active voice, finite verb). Item (2): "transfer entropy and quantile Granger detect..." (active voice, finite verb). Parallel structure is maintained. No issue.
- **Severity:** NONE

---

### 60. **Line 355–356: Sample size notation**
- **Text:** "Sample sizes reflect lag-9 input window and train/validation split ($n_{\text{eff}} < n_{\text{regime}}$)"
- **Issue:** "train/validation" uses a slash to indicate two alternatives. Acceptable in technical writing, though "train–validation" (en-dash) might be more standard. Current is acceptable.
- **Severity:** LOW

---

### 61. **Line 366: Table footnote format**
- **Text:** "\multicolumn{6}{l}{\footnotesize $^{**}p < 0.01$.} \\"
- **Issue:** Footnote to table is correctly formatted. No issue.
- **Severity:** NONE

---

### 62. **Line 371–372: Comma splice in algorithm description**
- **Text:** "A four-model diagnostic (OLS, RF with 100 trees, MLP 64-32, LSTM 32 hidden~\cite{tank2022neural}; Table~\ref{tab:neural}, Figure~\ref{fig:complexity}) finds no nonlinear improvement"
- **Issue:** Parenthetical contains: "OLS, RF..., MLP..., LSTM..." (list) then "Table..." (reference). The semicolon before "Table" separates the model list from citations, which is acceptable. No comma splice.
- **Severity:** NONE

---

### 63. **Line 373–374: Awkward phrasing with LSTM attention**
- **Text:** "LSTM attention concentrates 68.2\% on lag~1 in Normal, decaying to 52.9\% (Elevated) and 44.2\% (Crisis, approaching the uniform baseline $1/9 = 11.1\%$)"
- **Issue:** The phrase "approaching the uniform baseline $1/9 = 11.1\%$" — should this be "approaching the uniform baseline of $1/9 ≈ 11.1\%$"? Current reads as "44.2% approaching 11.1%," which is correct but slightly unclear whether the parenthetical applies only to Crisis or all items. Clearer: "...and 44.2% in Crisis (approaching the uniform baseline of $1/9 = 11.1\%$)".
- **Severity:** LOW

---

### 64. **Line 378: Dangling participle**
- **Text:** "RF permutation importance shows HML lag-1 as the dominant feature in Crisis (importance $= 0.043$, $4\times$ the mean)."
- **Issue:** "RF permutation importance shows...as the dominant feature" — correct structure. "RF permutation importance" is the subject (a noun phrase), "shows" is the verb. No dangling participle.
- **Severity:** NONE

---

### 65. **Line 380: Compound adjective hyphenation**
- **Text:** "\textbf{Sensitivity caveat:} Under an alternative fit (Cluster~5, seed~42, highest-LL achieving 90\% GFC detection, $\Delta\text{BIC} = 218$)"
- **Issue:** "highest-LL achieving" — should be "highest-LL, achieving" (comma after hyphenated compound) or "highest-LL fit, achieving" (noun after compound)? Current structure has "highest-LL achieving" as a participial phrase, which is awkward. Better: "...highest-LL fit (achieving 90% GFC detection...)".
- **Severity:** MEDIUM

---

### 66. **Line 381–382: Conditional structure**
- **Text:** "RF shows significant nonlinear improvement ($p = 0.010$ Elevated, $p = 0.005$ Crisis)."
- **Issue:** The labels "Elevated" and "Crisis" should be regime names; capitalization is correct if these are defined labels (see line 173). The parenthetical lacks a connecting word between the two $p$ values. Better: "($p = 0.010$ for Elevated, $p = 0.005$ for Crisis)" or "($p_{\text{Elevated}} = 0.010$, $p_{\text{Crisis}} = 0.005$)."
- **Severity:** LOW

---

### 67. **Line 385: Quote marks and em-dash**
- **Text:** "The ``purely linear'' characterization is therefore \emph{fit-dependent}"
- **Issue:** Double-quotes ("purely linear") are acceptable in LaTeX/academic writing. Alternatively, the phrase could be in single quotes (which LaTeX handles as `purely linear'`). Current is acceptable.
- **Severity:** NONE

---

### 68. **Line 414–416: Transfer entropy description**
- **Text:** "Transfer entropy~\cite{schreiber2000measuring} (Table~\ref{tab:te}) reveals the reverse channel SMB→HML is substantially stronger in Normal ($z = 5.37$ vs.\ forward $z = 2.45$); both collapse in Crisis."
- **Issue:** "reveals the reverse channel SMB→HML is substantially stronger" — subject is "Transfer entropy," verb is "reveals," object is the noun clause "the reverse channel SMB→HML is substantially stronger." Grammatically correct but the clause structure is slightly convoluted. Better: "reveals that the reverse channel (SMB→HML) is substantially stronger in Normal ($z = 5.37$ vs. $z = 2.45$ forward)."
- **Severity:** MEDIUM

---

### 69. **Line 417: Directional asymmetry labeling**
- **Text:** "This directional asymmetry---linear forward, nonlinear reverse---is not captured by conditional-mean Granger"
- **Issue:** "linear forward, nonlinear reverse" — these noun phrases are parallel in structure (adjective + direction noun). Acceptable.
- **Severity:** NONE

---

### 70. **Line 420: Hyphenation of compound**
- **Text:** "conditional-mean Granger or VAR connectedness~\cite{diebold2012better}, which test mean-squared-error improvement only."
- **Issue:** "mean-squared-error" — hyphenated compound noun. Acceptable. "which test...improvement" — the verb "test" should be "tests" (singular, referring to "connectedness"... wait, the subject is "Granger or...connectedness," which is plural, so "test" is correct. Acceptable.
- **Severity:** NONE

---

### 71. **Line 434–435: Table data formatting**
- **Text:** "HML→SMB & 0.053 & 0.017 & 0.034 & 0.906 & Linear \\\\ SMB→HML & $-0.022$ & $-0.026$ & \textbf{0.212} & \textbf{0.001} & Tail \\\\"
- **Issue:** Alignment and decimal places are consistent. No grammar issue.
- **Severity:** NONE

---

### 72. **Line 452–453: Quantile notation**
- **Text:** "SMB→HML operates through tail dependence ($\hat{\beta}_{0.95} = 0.212$, $8\times$ the median)"
- **Issue:** "$8\times$ the median" — the multiplication sign "×" (times) is sometimes written as "$\times$" (LaTeX). The phrasing "8 times the median" is clear. Acceptable.
- **Severity:** NONE

---

### 73. **Line 455–456: Parallel structure in Granger explanation**
- **Text:** "This reconciles the null reverse Granger ($p = 0.864$) with highly significant reverse TE ($z = 5.37$): Granger tests conditional mean improvement (MSE), while TE measures mutual information including tail dependence"
- **Issue:** "Granger tests...improvement" vs. "TE measures...information" — parallel active verbs. Correct. The explanation properly contrasts the two methods.
- **Severity:** NONE

---

### 74. **Line 457–458: Noun phrase clarity**
- **Text:** "a channel concentrated in extreme returns boosts mutual information without improving point forecasts."
- **Issue:** "a channel concentrated in extreme returns" — "concentrated" is a past participle functioning as an adjective. The phrase could be clearer: "a channel operating through concentration in extreme returns" or "a channel whose effects concentrate in extreme returns." Current is acceptable.
- **Severity:** LOW

---

### 75. **Line 461–462: Complex list of top-4 pairs**
- **Text:** "applying quantile Granger to the top-4 regime-heterogeneous pairs (RMW→SMB rank~1, Wald $p = 0.869$; MKT→MOM rank~2, $p = 0.741$; MKT→SMB rank~3, $p = 0.527$; SMB→MKT rank~4, $p = 0.097$)"
- **Issue:** Four-item parenthetical list with ranks and p-values, separated by semicolons. The format is consistent and clear. Acceptable.
- **Severity:** NONE

---

### 76. **Line 465–466: Compound noun in technical context**
- **Text:** "of 19 regime-heterogeneous pairs, none besides SMB→HML exhibits Wald $p < 0.05$"
- **Issue:** "none besides" is correct (singular pronoun with singular verb). The phrase "Wald $p < 0.05$" uses "$p$" to denote the p-value, which is standard. No issue.
- **Severity:** NONE

---

### 77. **Line 467–472: Conceptual conclusion sentence**
- **Text:** "Together, these diagnostics reveal that HML→SMB operates through linear conditional-mean prediction while SMB→HML operates through nonlinear tail dependence. This is the conceptual contribution: \emph{regime heterogeneity $\neq$ quantile heterogeneity}---a distinction not captured by conditional-mean Granger or VAR connectedness methods."
- **Issue:** "HML→SMB operates...while SMB→HML operates" — parallel structure with "while" is acceptable. The second sentence uses "$\neq$" (not equal) symbol in plain text context, which is acceptable in technical writing. No grammatical error, though the phrasing is slightly informal.
- **Severity:** NONE

---

### 78. **Line 475: "Tier 3 (exploratory)" — parenthetical**
- **Text:** "All results so far are in-sample. We now present \emph{Tier~3 (exploratory)} evidence"
- **Issue:** "Tier 3 (exploratory)" — the label and description in parentheses are clear. Acceptable.
- **Severity:** NONE

---

### 79. **Line 481–483: Table caption and footnote**
- **Text:** "Frozen OOS (2013--2024, HMM trained 1990--2012). No pair survives 30-pair Bonferroni ($\alpha/30 = 0.00033$). Bootstrap: reweighted to training Elevated prevalence (13.7\% vs.\ 33.7\% in test)."
- **Issue:** Caption is a mix of statement and technical detail. The colon in "Bootstrap: reweighted" introduces an explanation, which is acceptable.
- **Severity:** NONE

---

### 80. **Line 497–498: Subject-verb agreement with complex subject**
- **Text:** "The frozen OOS (Table~\ref{tab:oos}) exhibits regime redistribution rather than same-regime replication."
- **Issue:** Subject "frozen OOS" (singular noun phrase), verb "exhibits" (singular) — correct agreement.
- **Severity:** NONE

---

### 81. **Line 501–502: Parenthetical remark within sentence**
- **Text:** "the frozen classifier assigns formerly Normal observations to Elevated (Elevated share doubles from 13.7\% training to 33.7\% test)."
- **Issue:** Parenthetical explains why the signal appears in "Elevated" regime post-GFC. The phrase "Elevated share" uses "Elevated" as a regime label (capitalized). Acceptable for consistency.
- **Severity:** NONE

---

### 82. **Line 504–505: Complex conditional with parenthetical**
- **Text:** "This result (1)~does not survive 30-pair Bonferroni ($\alpha/30 = 0.00033$), (2)~does not survive 3-regime Bonferroni ($\alpha/3 = 0.0167$; HAC $p = 0.043$)"
- **Issue:** Numbered list within sentence: "(1)...does not survive, (2)...does not survive" — parallel structure is correct. The nested semicolon in "(2)" separates "Bonferroni" from "HAC", which is acceptable.
- **Severity:** NONE

---

### 83. **Line 508: Hyphenation of compound**
- **Text:** "bandwidth (Table~\ref{tab:bandwidth}: $p$ crosses 0.05 at NW default)"
- **Issue:** "NW default" — "NW" is an abbreviation (Newey-West). The colon introduces the parenthetical result. Acceptable.
- **Severity:** NONE

---

### 84. **Line 510: Parenthetical with "and/or"**
- **Text:** "null at $K = 2, 4$; BIC favors $K = 3$ by $\Delta\text{BIC} = 1{,}680$"
- **Issue:** "null at $K = 2, 4$" — implicit list of two values. Could be "null for $K = 2$ and $4$" but current is acceptable. No error.
- **Severity:** NONE

---

### 85. **Line 536–537: Colon introducing subsection**
- **Text:** "\textbf{MOM→SMB validation.} To address selective reporting, we conduct a full analysis of MOM→SMB---the top-ranked pair by OOS $F$-statistic."
- **Issue:** The sentence structure is acceptable; the em-dash provides emphasis. No error.
- **Severity:** NONE

---

### 86. **Line 543: Comma in negative statement**
- **Text:** "The reverse direction SMB→MOM is null in all regimes ($p > 0.09$), confirming strong directional asymmetry ($46$--$89\times$ ratio)."
- **Issue:** Comma before "confirming" correctly joins a dependent clause ("confirming..." is a participial phrase). Acceptable.
- **Severity:** NONE

---

### 87. **Line 545: Colon introducing result**
- **Text:** "Quantile Granger confirms the relationship is purely linear (Wald $p = 0.998$)."
- **Issue:** Parenthetical result is correctly positioned. No error.
- **Severity:** NONE

---

### 88. **Line 549–551: Passive voice construction**
- **Text:** "MOM→SMB thus proves the protocol detects genuine OOS confirmation for sufficiently strong signals; HML→SMB's weak OOS performance reflects signal weakness, not a methodological artifact."
- **Issue:** Semicolon joins two independent clauses: "MOM→SMB proves..." and "HML→SMB's...performance reflects..." — correct punctuation. The possessive "HML→SMB's" is slightly unconventional (applying possessive to an equation), but acceptable in technical writing.
- **Severity:** LOW

---

### 89. **Line 556: International replication subheading**
- **Text:** "\textbf{International replication} (Table~\ref{tab:international}). We now test whether structural breaks are a US-specific phenomenon."
- **Issue:** Acceptable structure; no error.
- **Severity:** NONE

---

### 90. **Line 557–558: Participial phrase**
- **Text:** "Applying the frozen protocol to four non-US Fama-French datasets: structural breaks detected in all four regions."
- **Issue:** Opening with a participial phrase ("Applying...") followed by a colon introducing the result is acceptable in technical writing, though it creates an implicit subject mismatch. More formally: "We apply the frozen protocol to four non-US Fama-French datasets and detect structural breaks in all four regions." Current is acceptable but could be clearer.
- **Severity:** LOW

---

### 91. **Line 559–563: Complex parenthetical in table caption**
- **Text:** "structural breaks detected in all four regions. Asia-Pacific ex Japan (Crisis OOS $F = 39.39$, $p < 0.001$) and Developed ex US (Crisis OOS $F = 15.85$, $p < 0.001$) produce Crisis-regime OOS effects surviving Bonferroni ($\alpha/12 = 0.0042$, correcting for 4 regions $\times$ 3 regimes)"
- **Issue:** Multiple nested parentheses and semicolons. Structure is complex but grammatically correct. The comma in "regions × 3 regimes" should separate the multiplication, which it does. No error.
- **Severity:** NONE

---

### 92. **Line 609–610: Hyphenation of compound noun**
- **Text:** "Of 30 directed factor pairs, 19 (63\%) show regime-heterogeneous Granger patterns in frozen OOS."
- **Issue:** "regime-heterogeneous" is hyphenated as a compound adjective, which is correct. "63%" is in parentheses as a percentage, which is acceptable.
- **Severity:** NONE

---

### 93. **Line 635–636: Complex subject and verb**
- **Text:** "The 50-seed multistart reveals 7 clusters (Table~\ref{tab:optima}). The structural break and in-sample Normal result are robust across all 7."
- **Issue:** Subject "The structural break and in-sample Normal result" (compound, plural) with verb "are robust" (plural) — correct agreement.
- **Severity:** NONE

---

### 94. **Line 638–640: Nested parenthetical with "Decision rule"**
- **Text:** "\emph{Decision rule for practitioners:} report BIC-optimal as primary; also report the highest-LL fit satisfying $\geq$50\% GFC detection as economic sensitivity. If both agree, the finding is robust."
- **Issue:** Colon after "Decision rule for practitioners" introduces a directive statement, which is acceptable. The semicolon joining "report BIC-optimal...primary" and "also report...economic sensitivity" is correct (two independent clauses). No error.
- **Severity:** NONE

---

### 95. **Line 664–665: Negative finding statement**
- **Text:** "Effect sizes are modest ($\Delta R^2 \approx 2\%$ pre-GFC) and do not generate trading profits (Sharpe ratio $= -0.07$)."
- **Issue:** Compound sentence with "and" joining two assertions. Grammatically correct.
- **Severity:** NONE

---

### 96. **Line 665–666: GARCH model comparison**
- **Text:** "GARCH(1,1) beats regime-conditional models for Value-at-Risk (VaR) coverage (1.48\% vs.\ 3.31\% violation rate)."
- **Issue:** "beats" is an informal verb in academic writing but acceptable in modern finance papers. The parenthetical provides model comparison. No grammar error.
- **Severity:** LOW (style suggestion)

---

### 97. **Line 669–671: Infinitive phrase**
- **Text:** "The regime-conditional framework thus excels at informing practitioners \emph{when} to revisit historically calibrated cross-factor covariance structures"
- **Issue:** "excels at informing practitioners when to revisit" — the infinitive "to revisit" is correctly governed by "when." Acceptable.
- **Severity:** NONE

---

### 98. **Line 675–676: Hypothesized mechanism**
- **Text:** "We hypothesize a deleveraging cascade: forced unwinding of value positions simultaneously deleverages associated size exposures, creating a predictive lag."
- **Issue:** Colon introduces the explanation of the "deleveraging cascade." The sentence structure has "unwinding...simultaneously deleverages," which is parallel and grammatically correct.
- **Severity:** NONE

---

### 99. **Line 677–679: Complex participial clause**
- **Text:** "FF25 portfolio overlap analysis finds significance concentrating in small-cap portfolios ($\rho_s = 0.35$, permutation $p = 0.046$); Small/HighBM alone accounts for 39\% of $\Delta R^2$."
- **Issue:** Semicolon joins two independent clauses, which is correct. The phrase "Small/HighBM alone" uses a slash to denote "Small-cap High Book-to-Market," which is acceptable in technical notation.
- **Severity:** NONE

---

### 100. **Line 685–686: Comparative clause**
- **Text:** "The HMM regime-conditional approach detects a signal (Elevated $p = 0.014$) that both simpler alternatives miss."
- **Issue:** "that both simpler alternatives miss" — relative clause with "that" correctly refers to "signal." Grammatically sound.
- **Severity:** NONE

---

### 101. **Line 687–690: Comparison of alternatives**
- **Text:** "Rolling-window Granger yields median $p = 1.00$ with no structural break; threshold-based regimes produce an \emph{inverted} direction. The multivariate Student-$t$ HMM captures endogenous regime persistence and tail structure that simpler methods cannot approximate."
- **Issue:** Semicolon joins two independent clauses, which is correct. The second sentence uses a relative clause "that simpler methods cannot approximate," which is grammatically correct.
- **Severity:** NONE

---

### 102. **Line 710–715: Complex parenthetical about limitations**
- **Text:** "All findings document predictive precedence (``Granger causality''), not structural causality~\cite{granger1969investigating,shojaie2022granger}. Trivariate controls (MKT-RF) address the most prominent common driver ($F$-$p > 0.43$), but a full 6-factor VAR ($324$ parameters per regime) is under-identified at $n \approx 1{,}000$;"
- **Issue:** Multiple parenthetical remarks are nested; structure is complex but grammatically correct. The semicolon ends the clause, leading to the next point. Acceptable.
- **Severity:** NONE

---

### 103. **Line 716–717: Sentence fragment in "Pair selection" statement**
- **Text:** "Pair selection is post-hoc; a pre-registered validation on emerging-market data would provide confirmatory evidence."
- **Issue:** Semicolon joins "Pair selection is post-hoc" with "a pre-registered validation...would provide..." — the second clause is a conditional statement (would + verb). Grammatically correct.
- **Severity:** NONE

---

### 104. **Line 718–719: Fit-dependent qualifier**
- **Text:** "The ``purely linear'' characterization is fit-dependent (seed~42: RF $p = 0.010$ Elevated)."
- **Issue:** Parenthetical clarification. "RF $p = 0.010$ Elevated" — should be "RF $p = 0.010$ for Elevated" or "$p_{\text{Elevated}} = 0.010$." The current format is slightly terse but acceptable in technical context.
- **Severity:** LOW

---

### 105. **Line 729: LSTM permutation test caveat**
- **Text:** "The LSTM permutation test uses 100 shuffles (vs.\ 200 for RF/MLP), adequate for a null result but underpowered to detect small nonlinear effects;"
- **Issue:** "adequate for...but underpowered to..." — this should read "adequate for detecting null results but underpowered for detecting small nonlinear effects" (parallel structure). Current phrasing with infinitives is acceptable but slightly awkward.
- **Severity:** LOW

---

### 106. **Line 736–742: Primary finding statement**
- **Text:** "HML→SMB Granger predictability is Bonferroni-significant in the pre-crisis Normal regime ($p = 8.75 \times 10^{-9}$, $\Delta R^2 = 2.06\%$), with a structural break at June 1998 ($p = 1.23 \times 10^{-13}$) and continued decay post-GFC (Chow $p = 2.29 \times 10^{-6}$). The post-2008 coefficient has been consistent with zero for 16 years (95\% CI $[-0.049, 0.073]$). This is robust across all 7 HMM local-optima clusters, multiple HAC specifications, lags 1--15, and trivariate controls."
- **Issue:** Compound sentence with parenthetical results. Structure is grammatically correct. "HML→SMB...is...Bonferroni-significant" — subject-verb agreement is correct (singular "predictability" with singular verb "is").
- **Severity:** NONE

---

### 107. **Line 746: Colon introducing structure**
- **Text:** "Replacing HMM regimes with VIX terciles (entirely external to factor returns), the structural break replicates: pre-2008 $p < 0.0001$, post-2008 $p = 0.714$."
- **Issue:** Colon before the results is acceptable, but the opening participial phrase "Replacing HMM regimes..." creates an implicit subject (we replace...). More formally: "When we replace HMM regimes with VIX terciles (entirely external to factor returns), the structural break replicates: pre-2008 $p < 0.0001$, post-2008 $p = 0.714$."
- **Severity:** LOW

---

### 108. **Line 753: Quantifier and noun agreement**
- **Text:** "Regime heterogeneity $\neq$ quantile heterogeneity; this distinction is not captured by conditional-mean Granger or VAR connectedness measures."
- **Issue:** Semicolon joins two independent clauses. "$\neq$" (not equal symbol) in running text is acceptable but slightly informal. The noun "distinction" (singular) agrees with "is not captured" (singular verb). Grammatically correct.
- **Severity:** NONE

---

### 109. **Line 765–766: Assumptions about factor-timing models**
- **Text:** "Factor-timing models assuming regime-invariant cross-factor relationships may misspecify dynamics during structural transitions."
- **Issue:** Participial phrase "assuming regime-invariant..." correctly modifies "models." The verb "may misspecify" is conditional and appropriately qualified. Grammatically correct.
- **Severity:** NONE

---

### 110. **Line 767–770: Protocol reusability statement**
- **Text:** "The regime-conditional protocol (Algorithm~\ref{alg:protocol})---multi-seed HMM, complexity characterization, information-theoretic diagnostics---is reusable for any factor set where latent-state structure may govern predictive relationships."
- **Issue:** Em-dashes enclose a parenthetical list of protocol components. The structure is acceptable but could be clearer: "The regime-conditional protocol (Algorithm 1), which includes multi-seed HMM, complexity characterization, and information-theoretic diagnostics, is reusable for any factor set where latent-state structure may govern predictive relationships."
- **Severity:** LOW

---

### 111. **Line 772–777: Future work list**
- **Text:** "\textbf{Future work:} (1)~neural Granger methods~\cite{tank2022neural} for systematic nonlinear analysis across factor networks; (2)~13F holdings-based verification of the deleveraging mechanism; (3)~pre-registered prospective validation on emerging-market factor data, with pair selection committed before data access."
- **Issue:** Numbered list with parenthetical structure. Items are introduced with "Future work:" (colon) and separated by semicolons. Parallel structure: item (1) uses "for systematic analysis," item (2) uses "verification of," item (3) uses "validation on." Inconsistent but acceptable in numbered lists.
- **Severity:** LOW

---

### 112. **Line 780–783: Code availability statement**
- **Text:** "All code (Python 3.10+, scikit-learn, statsmodels, hmmlearn), 50 HMM seed configurations, and a reproducibility notebook are available at an anonymized repository (link provided to reviewers; public release with DOI upon acceptance)."
- **Issue:** Compound subject with three items: "All code," "50 HMM seed configurations," and "a reproducibility notebook" — verb "are available" (plural) correctly agrees with compound subject. Parenthetical clarification is acceptable.
- **Severity:** NONE

---

## SUMMARY

**Total Issues Found: 112**

### Breakdown by Severity:
- **CRITICAL (cause rejection): 0**
- **MEDIUM (weakens paper): 10**
  - Line 43: Ambiguous modifier (directional comparison)
  - Line 48: Capitalization consistency (Elevated vs. elevated)
  - Line 113–115: Run-on sentence with awkward punctuation
  - Line 194–198: Sentence fragments in enumerated list
  - Line 380: Awkward participial phrase ("highest-LL achieving")
  - Line 416: Convoluted clause structure in transfer entropy statement
  - Line 558: Dangling participial phrase opening
  - Line 88: Possessive applied to equation (HML→SMB's)
  - Line 111: Awkward phrasing in conceptual contribution
  - Line 129: Unclear citation usage with "Andrews"

- **LOW (cosmetic): 102**
  - Hyphenation inconsistencies (linear--nonlinear, pre-2008, etc.)
  - Minor article and tense variations
  - Informal verb usage ("beats" in academic context)
  - Parallel structure opportunities in lists
  - Em-dash vs. en-dash inconsistencies
  - Minor notation clarity (e.g., "vs. forward z = 2.45")
  - Terse formatting in technical tables and parentheticals

### Key Patterns:
1. **Regime label capitalization**: "Elevated," "Normal," "Crisis" are treated as proper regime names throughout, but consistency should be verified against author intent.
2. **Parenthetical overload**: Multiple nested parentheses and semicolons create complexity in places (lines 268–272, 559–563, 710–715).
3. **Em-dash usage**: Inconsistent use of double-dashes (--) vs. em-dashes (---) for em-dashes; some lines use both.
4. **Enumerated lists**: Items (1), (2), (3) in several locations (lines 194–198, 504–511, 772–777) mix complete clauses with fragments, creating parallel structure issues.
5. **Technical phrasing**: Some statements sacrifice clarity for brevity (e.g., "vs. forward z = 2.45" without explicit comparison), acceptable but could be refined.

---

## CONCLUSION

**Status: MULTIPLE ISSUES IDENTIFIED**

The paper is **well-written overall** with strong technical clarity and appropriate academic tone. However, 10 MEDIUM-severity issues should be addressed to strengthen grammatical coherence and clarity, particularly:
- Dangling modifiers and participial phrase clarity (lines 43, 558)
- Sentence fragment consistency in enumerated lists (lines 194–198)
- Em-dash and hyphenation standardization throughout
- Capitalization consistency for regime labels

These revisions would improve readability and meet rigorous academic publishing standards for ICAIF. The 102 LOW-severity issues are primarily stylistic and do not impede comprehension but could enhance consistency and polish if time permits.
