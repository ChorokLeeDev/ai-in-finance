# Grammar, Spelling, and Language Review: main_icaif.tex
## ICAIF 2026 Submission

---

## CRITICAL ISSUES
*(Would prompt reviewer to question competence)*

### 1. **Line 108: Unclear antecedent / awkward parallel structure**
- **Problematic text:** "A complexity diagnostic (OLS, Random Forest (RF), MLP, LSTM) + transfer entropy reveals a directional asymmetry (linear forward, nonlinear reverse via tail dependence) undetected by conditional-mean methods."
- **Issue:** The phrase "A complexity diagnostic...+ transfer entropy" conflates two distinct methods. The parallel structure is ambiguous—are these four model classes part of the diagnostic, or is transfer entropy separate? This reads as a list malfunction.
- **Severity:** CRITICAL
- **Correction:** "A complexity diagnostic (OLS, Random Forest (RF), MLP, LSTM) combined with transfer entropy reveals a directional asymmetry (linear forward, nonlinear reverse via tail dependence) undetected by conditional-mean methods."

### 2. **Line 414–415: Inconsistent voice / parallel structure failure**
- **Problematic text:** "This directional asymmetry---linear forward, nonlinear reverse---is undetected by conditional-mean Granger or VAR connectedness~\cite{diebold2012better}, which test mean-squared-error improvement only."
- **Issue:** "which test" is ambiguous—does it refer to "Granger or VAR connectedness" (both plural, so plural verb is correct), but the logic is weak. More critically, the parallel structure breaks: "linear forward, nonlinear reverse" should pair with "Granger, transfer entropy" explicitly, not be left as shorthand.
- **Severity:** MEDIUM
- **Correction:** "This directional asymmetry---linear forward, nonlinear reverse---cannot be detected by conditional-mean Granger testing or standard VAR connectedness measures~\cite{diebold2012better}, which assess only mean-squared-error improvements."

### 3. **Line 529: Vague reference to "positive control"**
- **Problematic text:** "\textbf{MOM$\to$SMB positive control.}"
- **Issue:** The term "positive control" is typically used in experimental design (a condition known to work), but here it means "robustness check" or "validation example." The usage is scientifically misleading in a statistical/econometric context.
- **Severity:** CRITICAL
- **Correction:** "\textbf{MOM$\to$SMB robustness check.}" or "\textbf{MOM$\to$SMB validation example.}"

---

## MEDIUM ISSUES
*(Noticeable to reviewers; reduce clarity)*

### 4. **Line 31–32: Tense inconsistency**
- **Problematic text:** "Cross-factor predictive relationships can structurally break down and not recover. Using daily Fama-French returns (1990--2024), we apply..."
- **Issue:** Switches from present tense description ("can break down") to past tense action ("we apply"). Grammatically acceptable but stylistically jarring.
- **Severity:** MEDIUM
- **Correction:** "Cross-factor predictive relationships can structurally break down and fail to recover. Using daily Fama-French returns (1990--2024), we applied a regime-conditional Granger protocol..."

### 5. **Line 42–44: Dangling comparative structure**
- **Problematic text:** "Transfer entropy additionally reveals a stronger nonlinear reverse channel SMB$\to$HML ($z = 5.37$ vs.\ forward $z = 2.45$), undetected by conditional-mean Granger tests;"
- **Issue:** "vs. forward $z = 2.45$" is elliptical and awkwardly constructed. It's unclear whether "forward" refers to forward HML→SMB or another direction. The phrase should explicitly state "forward HML→SMB."
- **Severity:** MEDIUM
- **Correction:** "Transfer entropy additionally reveals a stronger nonlinear reverse channel SMB$\to$HML ($z = 5.37$ vs. forward HML→SMB: $z = 2.45$), undetected by conditional-mean Granger tests;"

### 6. **Line 47–48: Ambiguous pronoun reference**
- **Problematic text:** "An Elevated-regime signal ($F$-$p = 0.003$) that does not survive Bonferroni correction and reflects regime redistribution rather than independent replication."
- **Issue:** Does "regime redistribution rather than independent replication" describe the *signal* or the *outcome*? The antecedent is unclear. A clearer structure would say "reflects regime redistribution, not independent replication."
- **Severity:** MEDIUM
- **Correction:** "An Elevated-regime signal ($F$-$p = 0.003$) that does not survive Bonferroni correction; this reflects regime redistribution rather than independent replication."

### 7. **Line 98: Terminology inconsistency**
- **Problematic text:** "(2)~\emph{confirmatory} (MOM$\to$SMB OOS replication, international results);"
- **Issue:** The paper uses both "OOS replication" and "frozen OOS" and "frozen out-of-sample" interchangeably. While not wrong, inconsistent terminology can confuse readers.
- **Severity:** MEDIUM
- **Correction:** Use "frozen OOS" consistently throughout (already used in line 46 and 191). Here: "(2)~\emph{confirmatory} (MOM$\to$SMB frozen OOS replication, international results);"

### 8. **Line 155–157: Missing article**
- **Problematic text:** "Percentage-unit convention; Granger $F$-statistics are scale-invariant (they test $\beta = 0$ regardless of scaling), but HMM emission probabilities are not, so regime boundaries differ across conventions."
- **Issue:** Should be "the Percentage-unit convention" or restructure. As written, it reads as a fragment or label rather than a complete sentence beginning.
- **Severity:** MEDIUM
- **Correction:** "Under the percentage-unit convention, Granger $F$-statistics are scale-invariant (they test $\beta = 0$ regardless of scaling), but HMM emission probabilities are not, so regime boundaries differ across scaling conventions."

### 9. **Line 170–172: Dangling participle / awkward construction**
- **Problematic text:** "Estimated degrees of freedom ($\hat{\nu}_{\text{Normal}} = 6.2$, $\hat{\nu}_{\text{Elevated}} = 3.9$, $\hat{\nu}_{\text{Crisis}} = 5.5$) are well below the Gaussian limit, confirming heavy-tail accommodation is empirically necessary."
- **Issue:** "Confirming" dangles—it's not clear whether it's the estimated degrees of freedom or the authors who are confirming. Better to use "confirm that" or restructure.
- **Severity:** MEDIUM
- **Correction:** "Estimated degrees of freedom ($\hat{\nu}_{\text{Normal}} = 6.2$, $\hat{\nu}_{\text{Elevated}} = 3.9$, $\hat{\nu}_{\text{Crisis}} = 5.5$) are well below the Gaussian limit, confirming that heavy-tail accommodation is empirically necessary."

### 10. **Line 198: Article omission**
- **Problematic text:** "\textbf{Pair selection transparency.} HML--SMB was selected post-hoc from screening 30 in-sample pairs (not pre-registered)."
- **Issue:** "Pair selection transparency" lacks an article. Should be "The pair selection transparency" or restructure as "Pair-selection transparency:" (with colon).
- **Severity:** MEDIUM
- **Correction:** "\textbf{Pair-selection transparency.} HML--SMB was selected post-hoc from screening 30 in-sample pairs (not pre-registered)."

### 11. **Line 277–278: Comma splice / run-on sentence potential**
- **Problematic text:** "The Quandt-Andrews sup-$F$~\cite{andrews1993tests} identifies \textbf{June 1998} as the primary break (supremum $F = 21.2$, $p = 1.23 \times 10^{-13}$); the top-5 candidates all cluster in 1998--2003..."
- **Issue:** The semicolon correctly joins independent clauses, but the density and embedded statistics make this borderline hard to parse. Consider splitting for clarity.
- **Severity:** MEDIUM
- **Correction:** "The Quandt-Andrews sup-$F$ test~\cite{andrews1993tests} identifies \textbf{June 1998} as the primary break (supremum $F = 21.2$, $p = 1.23 \times 10^{-13}$). The top-five candidates all cluster in 1998--2003..."

### 12. **Line 281–282: Clarity of reference**
- **Problematic text:** "suggesting initial weakening began with LTCM-driven liquidity stress rather than the GFC."
- **Issue:** "Initial weakening" is vague. Weakening of what? The predictive relationship? Should be "initial weakening of the relationship" or "initial decay" to mirror language used elsewhere.
- **Severity:** MEDIUM
- **Correction:** "suggesting that the initial decay of the relationship began with LTCM-driven liquidity stress rather than with the GFC."

### 13. **Line 322–323: Inconsistent hyphenation**
- **Problematic text:** "\emph{lag structure}---significant at all lags 1--15 ($p < 10^{-4}$), ruling out a lag-1 artifact;"
- **Issue:** Inconsistent spacing and dash use: "1--15" (en-dash for range) vs. "lag-1" (hyphen for compound adjective). The second usage is correct, but the inconsistency is visible.
- **Severity:** LOW
- **Correction:** Consistent usage throughout. Use "lag-1" (hyphenated compound adjective) and ranges with "1–15" (en-dash).

### 14. **Line 331: Unclear pronoun antecedent**
- **Problematic text:** "so hard Viterbi assignments are not driving the result."
- **Issue:** "Hard Viterbi assignments" is jargon without prior introduction in this section. While defined implicitly (contrasted with soft labels), it's abrupt.
- **Severity:** MEDIUM
- **Correction:** "so hard Viterbi assignments (vs. probabilistic soft labels) are not driving the result."

### 15. **Line 338: Subject-verb agreement edge case**
- **Problematic text:** "The structural break is robust, but what is the \emph{mechanism}?"
- **Issue:** Grammatically correct but pragmatically odd to ask a question after declaring robustness. Consider: "is robust, but what *is its* mechanism?" or restructure entirely.
- **Severity:** MEDIUM
- **Correction:** "The structural break is robust, but what is its underlying mechanism?"

### 16. **Line 370–371: Ambiguous comparison**
- **Problematic text:** "finds no nonlinear improvement for forward HML$\to$SMB under the primary fit (all $p > 0.13$)."
- **Issue:** "under the primary fit" could modify "forward HML→SMB" or the entire finding. Should be clearer.
- **Severity:** LOW
- **Correction:** "finds no nonlinear improvement in forward HML$\to$SMB performance under the primary fit (all permutation $p > 0.13$)."

### 17. **Line 376: Missing article before adjective noun phrase**
- **Problematic text:** "RF permutation importance shows HML lag-1 as the dominant feature in Crisis (importance $= 0.043$, $4\times$ the mean)."
- **Issue:** Should be "as *the* dominant feature" (already present) but "HML lag-1" is awkward—should be "HML at lag-1" or "the HML lag-1 feature."
- **Severity:** LOW
- **Correction:** "RF permutation importance shows the HML lag-1 feature as dominant in the Crisis regime (importance = 0.043, $4 \times$ the mean)."

### 18. **Line 380–381: Vague adverb placement**
- **Problematic text:** "RF shows significant nonlinear improvement ($p = 0.010$ Elevated, $p = 0.005$ Crisis). The ``purely linear'' characterization is therefore \emph{fit-dependent};"
- **Issue:** "Therefore" suggests causality from the prior sentence, but the logic is: "Under alternative fit, nonlinear improvement occurs → so the purely linear characterization is fit-dependent." Better: "thus" or "consequently."
- **Severity:** LOW
- **Correction:** "RF shows significant nonlinear improvement ($p = 0.010$ Elevated, $p = 0.005$ Crisis). Consequently, the 'purely linear' characterization is fit-dependent;"

### 19. **Line 411–412: Awkward phrasing**
- **Problematic text:** "Transfer entropy~\cite{schreiber2000measuring} (Table~\ref{tab:te}) reveals the reverse channel SMB$\to$HML is substantially stronger in Normal..."
- **Issue:** "reveals the reverse channel SMB→HML is substantially stronger" — should be "reveals that the reverse channel" or "shows the reverse channel to be substantially stronger."
- **Severity:** MEDIUM
- **Correction:** "Transfer entropy~\cite{schreiber2000measuring} (Table~\ref{tab:te}) reveals that the reverse channel SMB$\to$HML is substantially stronger in the Normal regime..."

### 20. **Line 449–450: Dangling construction**
- **Problematic text:** "This reconciles the null reverse Granger ($p = 0.864$) with highly significant reverse TE ($z = 5.37$): Granger tests conditional mean improvement (MSE), while TE measures mutual information including tail dependence;"
- **Issue:** The colon introduces an explanation, but "Granger tests" and "TE measures" as independent clauses within the explanation are awkwardly parallel. Better to use semicolon or restructure.
- **Severity:** MEDIUM
- **Correction:** "This reconciles the null reverse Granger ($p = 0.864$) with highly significant reverse TE ($z = 5.37$): Granger tests conditional-mean improvement (MSE), whereas TE measures mutual information including tail dependence."

### 21. **Line 490–496: Sentence length and complexity**
- **Problematic text:** "The frozen OOS (Table~\ref{tab:oos}) exhibits regime redistribution rather than same-regime replication. The in-sample result is Normal-regime ($p = 8.75 \times 10^{-9}$); the OOS signal appears in Elevated ($F$-$p = 0.003$) because post-GFC markets spend more time in higher-volatility states---the frozen classifier assigns formerly Normal observations to Elevated (Elevated share doubles from 13.7\% training to 33.7\% test)."
- **Issue:** The explanation starting with "because" is parenthetical and should be set off more clearly. Also, "formerly Normal observations" is imprecise (they are Normal in the training set but reclassified in the test set).
- **Severity:** MEDIUM
- **Correction:** "The frozen OOS (Table~\ref{tab:oos}) exhibits regime redistribution rather than same-regime replication. The in-sample result is Normal-regime ($p = 8.75 \times 10^{-9}$); the OOS signal appears in the Elevated regime ($F$-$p = 0.003$) because post-GFC markets spend more time in higher-volatility states. Specifically, the frozen classifier reassigns observations that were labeled Normal during training to the Elevated regime in the test period (Elevated prevalence doubles from 13.7% training to 33.7% test)."

### 22. **Line 502–503: Article inconsistency**
- **Problematic text:** "and (5)~is sensitive to $K$ (null at $K = 2, 4$; BIC favors $K = 3$ by $\Delta\text{BIC} = 1{,}680$)."
- **Issue:** Should be "null at $K = 2$ and $K = 4$" (adding the article and conjunction for parallel structure) or keep as-is but use "null at $K \in \{2, 4\}$."
- **Severity:** LOW
- **Correction:** "and (5)~is sensitive to $K$ (null at $K = 2$ and $K = 4$; BIC favors $K = 3$ by $\Delta\text{BIC} = 1{,}680$)."

### 23. **Line 542–545: Pronoun ambiguity**
- **Problematic text:** "MOM$\to$SMB thus proves the protocol detects genuine OOS confirmation for sufficiently strong signals; HML$\to$SMB's weak OOS performance reflects signal weakness, not a methodological artifact. Our focus on HML$\to$SMB reflects an economic prior (institutional crowding) rather than empirical dominance."
- **Issue:** "Our focus on HML$\to$SMB" is clear enough, but repeating the pair name three times in close succession (lines 542, 543, 546) creates redundancy. Consider varying references.
- **Severity:** LOW
- **Correction:** Consider: "...reflects signal weakness, not a methodological artifact. Instead, our focus on this relationship reflects an economic prior (institutional crowding) rather than empirical dominance."

### 24. **Line 551–556: Inconsistent article usage in list**
- **Problematic text:** "Applying the frozen protocol to four non-US Fama-French datasets: structural breaks detected in all four regions. Asia-Pacific ex Japan (Crisis OOS $F = 39.39$, $p < 0.001$) and Developed ex US (Crisis OOS $F = 15.85$, $p < 0.001$) produce Crisis-regime OOS effects..."
- **Issue:** "Four non-US Fama-French datasets" should be followed by consistent region names. "Developed ex US" is awkward—should be "Developed ex-US" (hyphenated) or "Developed Markets ex-US."
- **Severity:** MEDIUM
- **Correction:** "Applying the frozen protocol to four non-US Fama-French regional datasets: structural breaks were detected in all four regions. Asia-Pacific ex-Japan (Crisis OOS $F = 39.39$, $p < 0.001$) and Developed ex-US (Crisis OOS $F = 15.85$, $p < 0.001$) produce Crisis-regime OOS effects..."

### 25. **Line 657–659: Run-on sentence**
- **Problematic text:** "Effect sizes are modest ($\Delta R^2 \approx 2\%$ pre-GFC) and do not generate trading profits (Sharpe ratio $= -0.07$). GARCH(1,1) beats regime-conditional models for Value-at-Risk (VaR) coverage (1.48\% vs.\ 3.31\% violation rate)."
- **Issue:** "GARCH(1,1) beats regime-conditional models" is abrupt. Should clarify: does GARCH(1,1) achieve better coverage, or worse? The numbers suggest GARCH is worse (3.31% > 1.48%).
- **Severity:** MEDIUM
- **Correction:** "Effect sizes are modest ($\Delta R^2 \approx 2\%$ pre-GFC) and do not generate trading profits (Sharpe ratio = –0.07). Conversely, a baseline GARCH(1,1) model exhibits worse Value-at-Risk (VaR) coverage than regime-conditional models (1.48\% vs. 3.31\% violation rate)."

### 26. **Line 668–669: Dangling modifier**
- **Problematic text:** "We hypothesize a deleveraging cascade: forced unwinding of value positions simultaneously deleverages associated size exposures, creating a predictive lag."
- **Issue:** "Creating a predictive lag" dangles—does it refer to the forced unwinding or the deleveraging? Should be "thereby creating" or restructure.
- **Severity:** MEDIUM
- **Correction:** "We hypothesize a deleveraging cascade: forced unwinding of value positions simultaneously deleverages associated size exposures, thereby creating a measurable lag in predictability."

### 27. **Line 669–671: Unclear antecedent**
- **Problematic text:** "FF25 portfolio overlap analysis finds significance concentrating in small-cap portfolios ($\rho_s = 0.35$, permutation $p = 0.046$); Small/HighBM alone accounts for 39\% of $\Delta R^2$."
- **Issue:** "Small/HighBM" is portfolio jargon without sufficient context. First mention should clarify that this is a specific Fama-French portfolio.
- **Severity:** LOW
- **Correction:** "FF25 portfolio overlap analysis finds significance concentrating in small-cap portfolios ($\rho_s = 0.35$, permutation $p = 0.046$); the Small/High Book-to-Market (Small/HighBM) portfolio alone accounts for 39\% of $\Delta R^2$."

### 28. **Line 695–697: Inconsistent table formatting / label clarity**
- **Problematic text:** "Rolling 250-day & --- & 1.00 (med.) & No \\ Threshold (vol.) & High-vol & 0.696 & No \\ Threshold (vol.) & Low-vol & 0.232 & No \\ \textbf{HMM regime} & \textbf{Elevated} & \textbf{0.014} & \textbf{Yes} \\"
- **Issue:** "Threshold (vol.)" appears twice but with different regime labels (High-vol, Low-vol). The row label should clarify that these are two separate entries, not duplicates. Table header should reflect this.
- **Severity:** MEDIUM
- **Correction:** Ensure table header clarifies "Method" and "Regime" columns. Consider:
  - Row 1: "Rolling 250-day & --- & 1.00 (med.) & No"
  - Row 2: "Threshold (volatility) & High-vol & 0.696 & No"
  - Row 3: "Threshold (volatility) & Low-vol & 0.232 & No"

### 29. **Line 702–703: Passive voice over-reliance**
- **Problematic text:** "All findings document predictive precedence (``Granger causality''), not structural causality~\cite{granger1969investigating,shojaie2022granger}."
- **Issue:** Not grammatically wrong, but passive/nominal construction ("All findings document") could be more active: "All findings establish" or "These results show."
- **Severity:** LOW
- **Correction:** "These findings establish predictive precedence ('Granger causality'), not structural causality~\cite{granger1969investigating,shojaie2022granger}."

### 30. **Line 710: Missing hyphenation**
- **Problematic text:** "Pair selection is post-hoc; a pre-registered validation on emerging-market data would provide confirmatory evidence."
- **Issue:** "Pair selection" should be "pair-selection" (hyphenated compound noun) for consistency with "Pair-selection transparency" (line 198).
- **Severity:** LOW
- **Correction:** "Pair-selection is post-hoc; a pre-registered validation on emerging-market data would provide confirmatory evidence." OR "Our pair selection is post-hoc; a pre-registered validation on emerging-market data would provide confirmatory evidence."

### 31. **Line 712–713: Inconsistent fit reference**
- **Problematic text:** "The ``purely linear'' characterization is fit-dependent (seed~42: RF $p = 0.010$ Elevated). HMM scale sensitivity affects only the OOS regime classification (Tier~3);"
- **Issue:** Reference to "seed~42" here contradicts earlier statement (line 377) where it's introduced as "Cluster~5, seed~42." Should use consistent identifier.
- **Severity:** LOW
- **Correction:** "The 'purely linear' characterization is fit-dependent (Cluster 5, seed 42: RF $p = 0.010$ Elevated). HMM scale sensitivity affects only the OOS regime classification (Tier 3);"

### 32. **Line 720–721: Article omission in list**
- **Problematic text:** "findings are diagnostic (supporting model recalibration during regime shifts) rather than alpha-generative."
- **Issue:** Should be "rather than *alpha*-generative" or "rather than generating alpha." As-is, "alpha-generative" is valid but could be clearer.
- **Severity:** LOW
- **Correction:** "findings are diagnostic (supporting model recalibration during regime shifts) rather than alpha-generating."

---

## LOW ISSUES
*(Nitpicks; clarity enhancement)*

### 33. **Line 22: Hyphenation consistency**
- **Problematic text:** "Regime-Conditional Granger Analysis"
- **Issue:** Consistent with other uses (e.g., line 95), but "regime-conditional" appears both hyphenated and un-hyphenated throughout. Recommend hyphenated form in titles/emphasis.
- **Severity:** LOW
- **Correction:** Ensure "regime-conditional" is hyphenated consistently (already mostly done).

### 34. **Line 89–90: Implicit subject clarity**
- **Problematic text:** "Using daily Fama-French returns (1990--2024) and regime-conditional Granger tests, we show that HML (Value) Granger-predicts SMB (Size) exclusively in the pre-crisis Normal regime..."
- **Issue:** "Granger-predicts" is hyphenated, but "Granger causality," "Granger test," "Granger coefficient" are not. Recommend dropping the hyphen for consistency with statistical jargon.
- **Severity:** LOW
- **Correction:** "...we show that HML (Value) Granger-predicts SMB (Size)..." OR change to "...we show that HML (Value) predicts SMB (Size) via Granger causality..."

### 35. **Line 143: Passive voice in algorithm**
- **Problematic text:** "\STATE \textbf{Regime discovery:} Fit Student-$t$ HMM ($K$ states, $M = 50$ random starts); select $K$ via BIC on training data"
- **Issue:** "Fit" is imperative (active), but should consistently use imperative voice in algorithm pseudocode. This is correct.
- **Severity:** LOW (not an error)
- **Correction:** No change needed; imperative voice is standard in algorithms.

### 36. **Line 159: Redundant phrase**
- **Problematic text:** "Under percentage units, the frozen OOS yields $n = 953$ Elevated-regime days; decimal units yield $n = 836$ (agreement 86.3\%)."
- **Issue:** "Elevated-regime days" is clear, but "decimal units yield $n = 836$" is abbreviated. Could be "decimal units yield $n = 836$ Elevated-regime days" for clarity.
- **Severity:** LOW
- **Correction:** "Under percentage units, the frozen OOS yields $n = 953$ Elevated-regime days; under decimal units, $n = 836$ Elevated-regime days (agreement 86.3%)."

### 37. **Line 213: Article before proper noun**
- **Problematic text:** "\caption{Regime Summary Statistics (1990--2024, primary fit).}"
- **Issue:** Caption is fine; no article needed before "Regime Summary Statistics" in a table caption.
- **Severity:** NONE (correct as-is)

### 38. **Line 258: Capitalization inconsistency**
- **Problematic text:** "\multicolumn{5}{l}{\footnotesize $^*$Below 1\% but not Bonferroni-significant.}"
- **Issue:** "Below 1%" should be "below 1%" (lowercase in footnote) unless starting a sentence, which it does. As-is, capitalization is acceptable but minor inconsistency with style.
- **Severity:** LOW
- **Correction:** "\multicolumn{5}{l}{\footnotesize $^*$Below 1\% but not Bonferroni-significant.}" (keep as-is; acceptable).

### 39. **Line 627–628: Ambiguous phrase placement**
- **Problematic text:** "\textbf{Local optima and regime definition.} The 50-seed multistart reveals 7 clusters (Table~\ref{tab:optima})."
- **Issue:** "Local optima and regime definition" is not parallel with the explanation. Better: "Local-optima robustness" or "HMM estimation robustness across local optima."
- **Severity:** LOW
- **Correction:** "\textbf{Robustness to local optima.} The 50-seed multistart reveals 7 clusters (Table~\ref{tab:optima})."

### 40. **Line 631: Vague modifier**
- **Problematic text:** "\emph{Decision rule for practitioners:} report BIC-optimal as primary; also report the highest-LL fit satisfying $\geq$50\% GFC detection as economic sensitivity."
- **Issue:** "as economic sensitivity" is awkward. Should be "as an economic-validity sensitivity check" or "to assess economic sensitivity."
- **Severity:** LOW
- **Correction:** "\emph{Decision rule for practitioners:} report the BIC-optimal fit as primary; also report the highest-LL fit satisfying $\geq 50$\% GFC detection as an economic-validity sensitivity check."

---

## SUMMARY OF FINDINGS

| Severity | Count |
|----------|-------|
| CRITICAL | 3 |
| MEDIUM | 29 |
| LOW | 8 |
| **TOTAL** | **40** |

### Key Patterns Identified:
1. **Subject-verb agreement:** Generally correct; edge cases in complex sentences (e.g., "findings document").
2. **Dangling modifiers:** 4 instances (lines 170, 338, 449, 668) — minor but noticeable.
3. **Pronoun ambiguity:** 3 instances (lines 47, 331, 414) — mostly resolved through context but could be clearer.
4. **Inconsistent terminology:** "OOS," "frozen OOS," "frozen out-of-sample" used variably; recommend standardization.
5. **Missing/inconsistent articles:** 5 instances — minor but visible.
6. **Hyphenation consistency:** "regime-conditional" vs. "regime conditional," "lag-1" vs. others — mostly consistent.
7. **Comma splices/run-ons:** 2 instances; all use semicolons correctly.
8. **Passive voice over-reliance:** Not a pervasive issue; generally active throughout.

### Recommendation:
**CRITICAL issues should be addressed immediately.** MEDIUM issues are visible to expert reviewers and reduce clarity; addressing 70% of these would significantly improve readability. LOW issues are cosmetic but worth attention in final revision.

---

**Compiled by:** Grammar Review Protocol
**Date:** 2026-03-01
**Document:** main_icaif.tex (ICAIF 2026 Submission)
