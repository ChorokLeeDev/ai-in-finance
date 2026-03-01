# ICAIF 2026 Paper: Grammar & Language Review
## Comprehensive Editorial Analysis

---

## CRITICAL ISSUES (Would cause reviewer to question competence)

### Issue 1: Inconsistent terminology for "undetected"
- **Line 42-44**: "undetected by conditional-mean Granger tests"
- **Line 409**: "undetected by conditional-mean Granger or VAR connectedness"
- **Problem**: Should use "undetected by" consistently (currently correct), but earlier in the paper there's a pattern of alternating language that should be standardized
- **Severity**: LOW (terminology is consistent here, but the instruction asked to check for "invisible to" phrases that should be "undetected by"—none found, but worth noting the paper correctly uses "undetected")

### Issue 2: Subject-verb agreement / Dangling modifier
- **Line 80-81**: "The August 2007 quantitative meltdown---where systematic equity strategies lost 27\% in three days~\cite{khandani2011quants}---revealed a blind spot"
- **Problem**: This is actually correct. The subject "meltdown" agrees with "revealed." However, the parenthetical phrase could be clearer.
- **Severity**: LOW (no error here)

### Issue 3: Ambiguous pronoun reference / Unclear antecedent
- **Line 114-115**: "Effect sizes are modest ($\Delta R^2 \approx 2\%$, Sharpe $= -0.07$); the contribution is diagnostic, not tradable alpha."
- **Problem**: The pronoun "the contribution" is clear, but in context of the surrounding text discussing multiple contributions, this could be clearer. Additionally, Sharpe should be "Sharpe ratio" for precision.
- **Severity**: MEDIUM

### Issue 4: Comma splice / Run-on sentence
- **Line 94-95**: "We distinguish three tiers: (1)~\emph{primary} (in-sample Normal-regime structural break, VIX-validated, robust across all specifications);"
- **Problem**: The opening statement "We distinguish three tiers:" creates a construction, but the parenthetical asides and dense punctuation make this hard to parse.
- **Severity**: LOW (acceptable in technical writing)

---

## MEDIUM ISSUES (Noticeable but not damaging)

### Issue 5: Inconsistent phrasing - passive vs. active voice
- **Line 34-35**: "frozen-parameter OOS validation) and establish that HML Granger-predicts SMB"
- **Line 40-41**: "post-2008, the relationship has been consistent with zero for 16~years"
- **Problem**: Switches between active voice ("establish") and passive/present perfect ("has been consistent"). While not wrong, inconsistent voice throughout sections.
- **Severity**: MEDIUM

### Issue 6: Missing article or awkward phrasing
- **Line 84-85**: "Quantitative factor models that assume regime-invariant cross-factor relationships systematically misestimate dynamics during structural transitions."
- **Problem**: "misestimate" is awkward; "misestimate" or "underestimate" would be more standard. This appears correct as a technical term.
- **Severity**: LOW

### Issue 7: Inconsistent formatting of statistical notation
- **Line 36**: "$p = 8.75 \times 10^{-9}$" (with space around multiplication)
- **Line 39**: "$p = 1.23 \times 10^{-13}$" (same)
- **Line 105**: "$p = 8.75 \times 10^{-9}$" (consistent)
- **Problem**: The formatting is actually consistent throughout. No issue here.
- **Severity**: LOW (no error)

### Issue 8: Article usage - missing "a" or "the"
- **Line 102**: "Contributions." followed by items
- **Line 103**: "(i)~Empirical documentation of structural decay: HML$\to$SMB predictability"
- **Problem**: Should read "documentation of the structural decay" or "documentation of structural decay in..." The current version is slightly awkward.
- **Severity**: MEDIUM

### Issue 9: Inconsistent numbering/formatting in tables and text
- **Line 142**: "Fit Student-$t$ HMM ($K$ states, $M = 50$ random starts)"
- **Line 268**: "Pre-2008 Normal ($n = 3{,}140$)"
- **Problem**: Uses both explicit numbers and variable references; inconsistent use of commas in thousands. Line 268 uses "3{,}140" while other places use "1{,}000" with commas.
- **Severity**: LOW (LaTeX formatting is consistent, just different style choices)

### Issue 10: Tense inconsistency
- **Line 195-196**: "HML--SMB was selected post-hoc from screening 30 in-sample pairs (not pre-registered)."
- **Line 523-524**: "To address selective reporting, we conduct a full analysis"
- **Problem**: "was selected" (past) vs. "we conduct" (present). Should maintain consistent past tense for methodology.
- **Severity**: MEDIUM

### Issue 11: Article missing before acronym
- **Line 191**: "CBOE Volatility Index (VIX) terciles (Normal $<$15,"
- **Problem**: Should be "the CBOE Volatility Index" or "the VIX" - first mention needs article.
- **Severity**: LOW

### Issue 12: Awkward phrasing - dangling modifier
- **Line 328-330**: "Rolling 3-year unconditional Granger (Figure~\ref{fig:rolling}) shows episodic significance peaks during stress periods, consistent with the regime-conditional finding."
- **Problem**: "Rolling" is a participle phrase at the start—could be clearer. Consider: "Rolling 3-year unconditional Granger analysis shows..."
- **Severity**: MEDIUM

### Issue 13: Inconsistent terminology
- **Line 41**: "null predictability post-2008"
- **Line 92**: "null predictability post-2008"
- **Line 269**: "Post-2008 Normal ($n = 1{,}557$): $p = 0.73$"
- **Problem**: Inconsistent use of "null predictability" vs. simply stating $p$-value. Term "null predictability" is somewhat awkward.
- **Severity**: MEDIUM

---

## LOW ISSUES (Nitpicks)

### Issue 14: Spacing around LaTeX commands
- **Line 36**: "triaged across all 7 HMM" - missing Oxford comma in lists before "and"
- **Line 37-38**: "lags 1--15, trivariate controls, and all 7 HMM" (has Oxford comma - good)
- **Severity**: LOW (inconsistent but both acceptable)

### Issue 15: Awkward nested clauses
- **Line 109**: "revealing a directional asymmetry (linear forward, nonlinear reverse via tail dependence) undetected by conditional-mean methods."
- **Problem**: The parenthetical is dense; could be separated. But acceptable in technical writing.
- **Severity**: LOW

### Issue 16: Missing article in technical phrase
- **Line 120**: "Factor returns exhibit time-series momentum"
- **Problem**: Should be "Factor returns exhibit time-series momentum" (actually correct as is)
- **Severity**: LOW (no error)

### Issue 17: Inconsistent hyphenation
- **Line 22**: "Cross-Factor Predictability" (hyphenated in title)
- **Line 31-32**: "Cross-factor predictive relationships" (lowercase, hyphenated)
- **Line 195**: "HML--SMB" (endash)
- **Problem**: Title uses caps with hyphen, text uses lowercase with hyphen. This is correct for title case vs. body text.
- **Severity**: LOW

### Issue 18: Punctuation in academic citations
- **Line 80-81**: "quantitative meltdown---where" (em-dash)
- **Line 81-82**: "~\cite{khandani2011quants}---revealed" (em-dash before citation)
- **Problem**: The citation placement after em-dash is correct for ACM format.
- **Severity**: LOW

### Issue 19: Missing Oxford comma
- **Line 105**: "regime-heterogeneous patterns" (OK—single pattern type)
- **Line 106**: "showing regime-heterogeneous patterns" (OK)
- **Severity**: LOW (no error)

### Issue 20: Ambiguous "which" clause
- **Line 122-123**: "Factor returns exhibit time-series momentum~\cite{ehsani2022factor} and crowding effects~\cite{marks2019factor,brunnermeier2009market}, but prior work examines \emph{within-factor} dynamics---not which factor leads another under which regime."
- **Problem**: "which factor leads another" could be clearer. Consider: "not which factors lead others under regime conditions."
- **Severity**: MEDIUM

### Issue 21: Inconsistent use of backslash before LaTeX spacing
- **Line 43**: "vs.\ forward $z = 2.45$)" (with backslash-space)
- **Line 198**: "MOM$\to$SMB is the top OOS pair ($F = 20.3$ vs.\ $9.06$)" (with backslash-space)
- **Line 262**: "never exceeds $10^{-7}$ (range: $[3.2 \times 10^{-9},\; 8.8 \times 10^{-8}]$"
- **Problem**: Inconsistent spacing after "vs." - some have backslash-space, some don't
- **Severity**: LOW (formatting consistency)

### Issue 22: Singular vs. plural agreement
- **Line 148**: "Quantile Granger: Quantile regression; Wald test for tail dependence"
- **Problem**: Correct
- **Severity**: LOW (no error)

### Issue 23: Inconsistent terminology - "regime-conditional" vs "regime-heterogeneous"
- **Line 88-89**: "regime-conditional Granger tests"
- **Line 106**: "showing regime-heterogeneous patterns"
- **Line 592**: "regime-heterogeneous Granger patterns"
- **Problem**: Both terms are used, but they mean different things. "Regime-conditional" = behavior within regimes. "Regime-heterogeneous" = different across regimes. This is actually correct usage.
- **Severity**: LOW

### Issue 24: Missing parallel structure
- **Line 755-759**: "Future work: (1)~neural Granger methods... (2)~13F holdings-based verification... (3)~pre-registered prospective validation..."
- **Problem**: All three items use gerunds/noun phrases. Structure is parallel. Correct.
- **Severity**: LOW (no error)

### Issue 25: Awkward phrasing with emphasis
- **Line 87**: "\textbf{This paper documents structural decay of cross-factor predictability.}"
- **Problem**: Mixing bold with period outside is acceptable in ACM format. No error.
- **Severity**: LOW

### Issue 26: Inconsistent use of parentheses vs. em-dashes
- **Line 115**: "($\Delta R^2 \approx 2\%$, Sharpe $= -0.07$)"
- **Line 268**: "(Pre-2008 Normal ($n = 3{,}140$): $p = 6.66 \times 10^{-16}$..."
- **Problem**: Nested parentheses in line 268 are awkward; could use em-dashes. But this is stylistic.
- **Severity**: LOW

### Issue 27: Spacing in statistical test notation
- **Line 281**: "$F(3,n{-}6) = 9.68$"
- **Problem**: Should use \mathit{n} or n without hyphenation. The "{-}" creates a minus sign in a variable context. Could be clearer as "$F(3, n-6) = 9.68$"
- **Severity**: LOW (acceptable in LaTeX)

### Issue 28: Inconsistent capitalization
- **Line 175**: "``Crisis'' denotes"
- **Line 191**: "Normal $<$15, Elevated 15--21, Crisis $>$21"
- **Problem**: Regime names are capitalized when used as proper nouns, lowercase when descriptive. Consistent usage.
- **Severity**: LOW

### Issue 29: Word repetition
- **Line 320**: "ruling out a lag-1 artifact; \emph{common drivers}---"
- **Line 321**: "so the signal is not proxying for market risk;"
- **Problem**: Multiple semicolons and em-dashes in close sequence make this hard to parse. Acceptable but slightly dense.
- **Severity**: LOW

### Issue 30: Missing specification in prose
- **Line 370**: "importance $= 0.043$, $4\times$ the mean"
- **Problem**: "$4\times$" should be "$4 \times$" with space. Inconsistent with other notation.
- **Severity**: LOW

---

## SUMMARY OF KEY FINDINGS

### Critical Issues: 0
- No truly critical grammar errors that would cause a reviewer to question competence.

### Medium Issues: 6-8
1. **Passive/active voice inconsistency** (Lines 34-40)
2. **Missing "the" in article usage** (Line 103)
3. **Tense inconsistency in methodology section** (Lines 195 vs. 523)
4. **Dangling modifier awkwardness** (Line 328)
5. **Unclear "which" clause** (Line 122)
6. **Awkward Sharpe ratio reference** (Line 115)

### Low Issues: 20+
Most are minor formatting, stylistic choices, or acceptable variations in technical writing.

---

## SPECIFIC RECOMMENDATIONS

### Issue 1 (Line 115): Clarify Sharpe ratio reference
**Current**: "Effect sizes are modest ($\Delta R^2 \approx 2\%$, Sharpe $= -0.07$);"
**Better**: "Effect sizes are modest ($\Delta R^2 \approx 2\%$, Sharpe ratio $= -0.07$);"
**Severity**: MEDIUM

### Issue 2 (Line 103): Add article for clarity
**Current**: "(i)~Empirical documentation of structural decay:"
**Better**: "(i)~Empirical documentation of the structural decay:"
**Severity**: MEDIUM

### Issue 3 (Line 195-196): Maintain consistent past tense
**Current**: "HML--SMB was selected post-hoc from screening 30 in-sample pairs (not pre-registered). Focus reflects an economic prior..."
**Better**: "HML--SMB was selected post-hoc from screening 30 in-sample pairs (not pre-registered). This focus reflected an economic prior..."
**Severity**: MEDIUM

### Issue 4 (Line 328-330): Clarify dangling participle
**Current**: "Rolling 3-year unconditional Granger (Figure~\ref{fig:rolling}) shows episodic significance peaks..."
**Better**: "Rolling 3-year unconditional Granger analysis (Figure~\ref{fig:rolling}) shows episodic significance peaks..."
**Severity**: MEDIUM

### Issue 5 (Line 122-123): Clarify "which" clause reference
**Current**: "not which factor leads another under which regime"
**Better**: "not which factors lead others under which regime conditions"
**Severity**: MEDIUM

### Issue 6 (Line 191): Add article before acronym
**Current**: "CBOE Volatility Index (VIX) terciles"
**Better**: "the CBOE Volatility Index (VIX) terciles"
**Severity**: LOW

### Issue 7 (Line 370): Fix notation spacing
**Current**: "$4\times$ the mean"
**Better**: "$4 \times$ the mean"
**Severity**: LOW

### Issue 8 (Line 281): Clarify statistical notation
**Current**: "$F(3,n{-}6) = 9.68$"
**Better**: "$F(3, n-6) = 9.68$"
**Severity**: LOW

---

## OVERALL ASSESSMENT

The paper is **well-written overall** with **no critical grammar errors**. The main issues are:
- Occasional article omissions (LOW-MEDIUM severity)
- Minor voice/tense inconsistencies (MEDIUM severity)
- Dense prose in a few sections (MEDIUM severity)
- Formatting nitpicks (LOW severity)

The paper demonstrates strong command of technical writing. The statistical notation is mostly consistent, terminology is carefully used, and the logical flow is clear. These edits would polish the manuscript but would not affect its acceptance decision at a top venue like ICAIF.

**Recommendation**: Address the 6 MEDIUM issues before submission. The LOW issues can be left as-is or addressed in proofs.
