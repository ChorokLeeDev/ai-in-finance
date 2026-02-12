# Writing Quality Review (v2)

## Overall Assessment

This is a well-structured paper with a clear empirical contribution: SHAP concentration as a pre-deployment diagnostic for conformal prediction failure. The writing is generally precise and data-driven, with commendable statistical transparency (CIs, p-values, LOO stability). However, the paper suffers from three systemic issues: (1) the abstract and introduction front-load too much numerical detail, burying the conceptual hook; (2) several sections repeat the same results in slightly different wording (the "nearly three orders of magnitude" phrase appears 4 times; the binary ceiling effect is explained 4 times); and (3) long, dense sentences throughout Section 5-6 make the paper harder to parse than necessary. The paper is roughly 11 pages of content plus appendix -- page budget is well-used but could be tighter in places. Below are section-by-section edits.

---

## Section-by-Section Feedback

### Abstract (lines 35-58)

**Word count**: ~210 words. Within UAI limits but on the high side.

**Issue 1: Overloaded opening.** The first two sentences are good but the third sentence packs too much in.

- Current (line 40-41): "Using COVID-19 as a case study across 8 supply chain tasks experiencing varying degrees of feature turnover, we find coverage drops ranging from 0.1\% to 77\% (all paired $p \leq 0.005$, 50 seeds)."
- Suggestion: "Across 8 supply chain tasks experiencing COVID-19 as a common temporal shift, coverage drops range from 0.1\% to 77\% (all $p \leq 0.005$, 50 seeds)."
- Rationale: "Using COVID-19 as a case study" is wordy; "varying degrees of feature turnover" is detail that belongs in the body.

**Issue 2: Too many numbers.** The abstract has 18 distinct numerical values. A reader scanning the abstract gets lost.

- Current (lines 44-46): "SHAP concentration, computed \textit{before} observing test data, predicts which tasks will fail: $\rho = 0.833$, $p = 0.010$ within supply chain ($n=8$); $\rho = 0.691$, $p = 0.019$ across 3 domains ($n=11$ including clinical trials and motorsport)."
- Suggestion: "SHAP concentration, computed \textit{before} observing test data, predicts which tasks will fail (Spearman $\rho = 0.83$, $p = 0.01$; $n=8$), with the association holding across 3 domains ($\rho = 0.69$, $p = 0.02$; $n=11$)."
- Rationale: Round to 2 decimal places in the abstract; list domains in body.

**Issue 3: The ACI sentence is too long and dense.**

- Current (lines 54-57): "Adaptive Conformal Inference recovers coverage for vulnerable tasks but inflates prediction sets to 30--82\% of classes for those needing intervention (Marginal to Vacuous); robust tasks need no ACI. We formalize a decision framework: compute SHAP concentration pre-deployment; retrain quarterly if vulnerable; skip retraining if robust."
- Suggestion: Split into: "Adaptive Conformal Inference recovers coverage for vulnerable tasks but inflates prediction sets to 30--82\% of classes. We formalize a decision framework: compute SHAP concentration pre-deployment, then retrain quarterly if vulnerable or skip retraining if robust."
- Rationale: "(Marginal to Vacuous)" is jargon undefined in the abstract; "robust tasks need no ACI" is implicit.

**Issue 4: Missing hook.** The abstract leads with a technical gap statement but does not connect to stakes for the reader. Consider adding a motivating clause.

- Current (line 36-37): "Conformal prediction guarantees degrade under distribution shift, but practitioners lack tools to predict \textit{which} deployed models will fail."
- Suggestion: "Conformal prediction provides distribution-free coverage guarantees, but these degrade under distribution shift---and practitioners currently lack tools to predict \textit{which} deployed models will fail before it happens."
- Rationale: Reminds the reader what conformal prediction offers before saying it breaks.

**Issue 5: "Binary classification tasks exhibit a structural ceiling effect that limits APS vulnerability regardless of concentration."**

- This sentence is important but will confuse readers unfamiliar with the paper. In the abstract, simplify to: "Binary classification tasks are structurally protected against this failure mode, clarifying the diagnostic's scope."

### Introduction (lines 60-78)

**Issue 6: First paragraph is a single sentence of 47 words.**

- Current (line 62): "Conformal prediction provides distribution-free coverage guarantees under the assumption of exchangeability~\citep{vovk2005algorithmic}. However, real-world deployments face distribution shifts that violate this assumption. While prior work characterizes \textit{how} conformal prediction degrades under shift~\citep{tibshirani2019conformal, barber2023conformal}, a critical gap remains: \textbf{Can we predict which deployed models will fail before observing test data?}"
- This is actually three sentences, which is fine. But the bold question deserves its own line or stronger emphasis. Consider making it a standalone paragraph.

**Issue 7: Contribution 1 is 4 sentences long (lines 69).**

- Current: "SHAP concentration, computed solely on validation data, predicts catastrophic conformal failure under severe shift. Within supply chain: $\rho = 0.833$, $p = 0.010$ ($n=8$). Across 3 domains (supply chain, clinical trials, motorsport): $\rho = 0.691$, $p = 0.019$ ($n=11$). Binary APS exhibits a structural ceiling effect; concentration is diagnostic specifically for multiclass settings. Post-hoc alternatives (entropy, ECE) require test-time observations."
- Suggestion: "SHAP concentration, computed solely on validation data, predicts catastrophic conformal failure ($\rho = 0.83$, $p = 0.01$; $n = 8$ supply chain tasks; $\rho = 0.69$, $p = 0.02$ across 3 domains). The diagnostic applies to multiclass settings; binary APS is structurally protected."
- Rationale: The contribution list should be scannable. Move details to body.

**Issue 8: Contribution 3 is 2 dense sentences (lines 73).**

- Current: "Coverage drops across 8 tasks range from 0.1\% to 77.1\%, all with paired $p \leq 0.005$ (Wilcoxon, 50 seeds). The correlation magnitude is stable under leave-one-out analysis ($\rho \in [0.75, 0.96]$), though significance is not maintained for 2 of 8 jackknife samples due to reduced power at $n=7$."
- Suggestion: "Coverage drops range from 0.1\% to 77.1\% (all $p \leq 0.005$, 50 seeds). The correlation is stable under leave-one-out analysis ($\rho \in [0.75, 0.96]$; 6/8 jackknife samples significant)."
- Rationale: The reduced-power caveat belongs in the results.

**Issue 9: Five contributions is borderline too many.** Contribution 5 (Binary ceiling effect) is really a scope limitation of Contribution 1, not a standalone contribution. Consider folding it into Contribution 1 as a qualifier.

### Related Work (lines 80-91)

**Issue 10: Missing positioning statement.** The related work covers the landscape but never explicitly says what the paper's unique position is relative to the nearest neighbor. After the "Shift Detection" paragraph:

- Current: "Our contribution is a \textit{pre-deployment} diagnostic using feature importance structure."
- Suggestion: "Unlike these methods, our diagnostic requires no test data and no distributional assumptions---only a trained model and its validation set."
- Rationale: Sharper differentiation.

**Issue 11: The "Interpretability for Reliability" paragraph is only one sentence.**

- Current (line 90): "Our work connects SHAP~\citep{lundberg2017unified} to model reliability prediction, extending the use of feature attribution from debugging~\citep{adebayo2018sanity} to prospective failure prediction."
- Suggestion: Either expand with 1-2 more sentences citing relevant work (e.g., Hooker et al. 2019 on feature removal) or merge into the Shift Detection paragraph.

### Methodology (lines 92-144)

**Issue 12: Section 3.1 -- "Before February 2020 (pre-COVID)" is imprecise.**

- Suggestion: Specify the actual training start date if known, or say "all data prior to February 2020."

**Issue 13: Section 3.3 -- Dense paragraph.**

- Current (line 116): "We train 50 independent models with seeds 42--91. For each seed $s$: train model $M_s$, split validation 50/50 into calibration $\mathcal{D}_{\text{cal}}^s$ and evaluation $\mathcal{D}_{\text{eval}}^s$, calibrate $\text{CP}_s$ on $\mathcal{D}_{\text{cal}}^s$, evaluate on both $\mathcal{D}_{\text{eval}}^s$ and test."
- Suggestion: Use an enumerated list for clarity, or at minimum break into two sentences.

**Issue 14: Section 3.4 -- Equation defines Jaccard on features but text says "feature temporal stability."**

- Current (line 120-124): The equation uses $A_{\text{train}}$ and $A_{\text{test}}$ as "sets of unique values for feature $f$", but the Jaccard is computed per-feature on its value sets, which is unusual. Standard Jaccard in ML is on feature sets, not value sets. This needs a clearer name.
- Suggestion: Call it "value-level Jaccard overlap" to distinguish from the more common "feature-set Jaccard."

**Issue 15: Section 3.6 -- "One might ask: why not use simpler diagnostics?"**

- This rhetorical question is informal for UAI.
- Suggestion: "A natural question is whether simpler diagnostics suffice."

**Issue 16: Section 3.6 -- "Raw importance values are not comparable across tasks (different scales, feature counts)."**

- This is a parenthetical that should be a full sentence.
- Suggestion: "Raw importance values are not comparable across tasks because they vary in scale and feature count."

### Intuition / Theory (lines 146-164)

**Issue 17: "Proposition (informal)" is unusual phrasing.**

- Suggestion: "Claim (heuristic)." or simply "Heuristic claim." The word "proposition" in a math paper implies a formal statement with a proof.

**Issue 18: Long argument paragraph (lines 154-160).**

- The paragraph starting "When $C$ is high" is 4 sentences that could be tightened.
- Current: "The model maps these to arbitrary predictions (effectively random with respect to the true label distribution), producing conformity scores $s_{\text{test}}$ that are systematically higher than $s_{\text{cal}}$."
- Suggestion: "The model maps unseen values to arbitrary predictions, producing conformity scores $s_{\text{test}}$ that are systematically higher than calibration scores $s_{\text{cal}}$."

**Issue 19: Finance analogy may not land with UAI audience.**

- Current (line 162): "This is analogous to portfolio diversification in finance: concentrated bets amplify risk; diversified exposure dampens it."
- Suggestion: Keep it but acknowledge it briefly: "By analogy with portfolio diversification, concentrated dependence on a single feature amplifies vulnerability, while distributing importance across features provides resilience."

### Results (lines 166-395)

**Issue 20: Section 5.1 first sentence repeats the table caption.**

- Current (line 170): "Table~\ref{tab:main_results} shows coverage degradation across 8 supply chain classification tasks."
- Suggestion: Lead with the insight, not the table reference: "Despite experiencing identical COVID-19 temporal shift, the 8 supply chain tasks exhibit dramatically different coverage degradation (Table~\ref{tab:main_results})."

**Issue 21: "Nearly three orders of magnitude" appears in lines 64, 138, 261, 428.**

- This is used 4 times in the paper. The phrase is effective but loses impact through repetition. Keep it in the abstract and one body location; replace others with "0.1\% to 77\%" or "770-fold."

**Issue 22: Section 5.2 -- "Two factors explain the variance" is overconfident.**

- Suggestion: "Two factors account for much of the variance" -- you have not shown these are the only factors.

**Issue 23: Table 2 caption is too brief.**

- Current: "Feature Overlap for Primary Features."
- Suggestion: "Feature Overlap (Value-Level Jaccard) for Primary Features. Tasks relying on transaction IDs (Jaccard $\approx 0$) fail catastrophically; entity-based features (Jaccard $> 0.5$) maintain coverage."

**Issue 24: Section 5.3 -- The paragraph starting "Two tasks with concentration..." (line 234) shifts from correlation analysis to exception handling. Consider making this a separate subparagraph with a bold lead-in.**

- Suggestion: "\textbf{False positives.} Two tasks with concentration $>$40\% remain robust:..."

**Issue 25: Section 5.4 -- "A key advantage of SHAP concentration is that it is available \textit{before} deployment." (line 272)**

- This has been stated at least 4 times already (abstract, intro, Section 3.6, here). Cut or rephrase: "As established, SHAP concentration is uniquely available pre-deployment."

**Issue 26: Section 5.4 -- The paragraph about pre-deployment alternatives (line 272) is awkward because it lists alternatives and then says "future work should compare them."**

- Suggestion: Move this to the limitations or trim to one sentence: "We note that alternative pre-deployment diagnostics (ensemble disagreement, native feature importance, distribution statistics) merit systematic comparison in future work."

**Issue 27: Table 4 (ACI) -- "Std Cov." column header is ambiguous.**

- Suggestion: "Static Cov." or "Base Cov." to clarify this is the non-ACI coverage.

**Issue 28: Section 6.1 -- The Deployability metric paragraph (lines 322) is important but introduced late.**

- Suggestion: Define "usable-set rate" earlier (perhaps in the table caption or before the analysis paragraphs).

**Issue 29: Section 6.2 -- "Quarterly strikes optimal cost-effectiveness: 3 retrains/year achieves 85\% of maximum improvement." (line 328)**

- "Strikes" is informal.
- Suggestion: "Quarterly retraining offers the best cost-effectiveness: 3 retrains per year achieves 85\% of the maximum improvement."

**Issue 30: Section 6.3 -- Long sentence (line 387).**

- Current: "Adding binary tasks weakens the overall correlation from $\rho = 0.833$ to $0.691$---not because the diagnostic fails, but because binary APS is structurally protected against the failure mode that concentration detects."
- This is 32 words and grammatically correct but could be split for readability.
- Suggestion: "Adding binary tasks weakens the overall correlation from $\rho = 0.833$ to $0.691$. This attenuation reflects not a diagnostic failure but the structural protection that binary APS provides against the failure mode concentration detects."

**Issue 31: Section 6.4 (Placebo) -- This section is only 2 sentences.** At 2 sentences, it feels rushed given the importance of the placebo test.

- Suggestion: Add one sentence interpreting the result: "The placebo test confirms that the observed coverage collapse is attributable to COVID-19 disruption rather than routine temporal drift."

### Discussion (lines 417-423)

**Issue 32: "Counter-intuitively" appears twice in the paper (lines 419 and 593).**

- Suggestion: Use "Paradoxically" or "Surprisingly" for one instance.

**Issue 33: The limitations paragraph (line 423) is a single paragraph with 8 numbered items spanning ~15 lines.** This is hard to read.

- Suggestion: Break into 2-3 paragraphs grouped thematically: (a) sample size and generalizability (items 1-3), (b) statistical methodology (items 4-5), (c) scope and extensions (items 6-8).

**Issue 34: Limitation (4) uses the word "anti-conservative" without definition.**

- Suggestion: Add "(i.e., actual significance levels may be larger than reported)" after "anti-conservative."

**Issue 35: Limitation (7) is very weak -- "validated on 1 example."**

- Suggestion: Be more direct: "The protective-factor thresholds (Jaccard $> 0.5$, importance $> 15\%$) are derived from a single case (sales-office) and should be treated as illustrative rather than prescriptive."

### Conclusion (lines 425-435)

**Issue 36: The conclusion is essentially a numbered list of results.** It adds no synthesis beyond what the abstract provides.

- Suggestion: Add 1-2 sentences of broader implication: "More broadly, our results suggest that the structure of learned feature importance---not just the presence of distribution shift---determines whether conformal guarantees will hold in practice. This opens a new direction for pre-deployment model auditing."

**Issue 37: "The diagnostic is pre-deployment (validation data only), actionable (retrain/skip decision), and robust (threshold stable across 30--45\%)." (line 435)**

- Good sentence. But "robust" is overloaded (used for task category AND threshold stability).
- Suggestion: Replace with "stable" -- "The diagnostic is pre-deployment (validation data only), actionable (retrain/skip decision), and stable (threshold effective across 30--45\%)."

### Appendix

**Issue 38: Appendix A.1 -- Long run-on sentence.**

- Current (line 462): "All models trained with default LightGBM settings: objective=multiclass, boosting=gbdt, num\_leaves=31, learning\_rate=0.05, feature\_fraction=0.8, bagging\_fraction=0.8, bagging\_freq=5, num\_boost\_round=500, early\_stopping\_rounds=50, seeds 42--91 (50 seeds)."
- Suggestion: Use a small table or itemized list.

**Issue 39: Appendix E (Baselines) -- The final paragraph (line 593) is 7 lines of dense text.**

- Break into 3 paragraphs: (1) Entropy paradox for catastrophic tasks, (2) Entropy increase for moderate tasks, (3) ECE is non-discriminating.

**Issue 40: Table 8 caption repeats "(10-seed ACI experiments)" which is detail that belongs in a footnote.**

---

## Top 10 Most Impactful Changes

1. **Reduce abstract numbers from 18 to ~10.** Round to 2 decimal places; remove parenthetical sample sizes where context is clear. The abstract should be scannable in 30 seconds.

2. **Consolidate the 5 contributions to 3-4.** Fold the binary ceiling effect (Contribution 5) into Contribution 1 as a scope qualifier. Tighten Contribution 3 (quantification) into 1 sentence.

3. **Eliminate the 4x repetition of "nearly three orders of magnitude."** Use it once in the abstract, once in the results; elsewhere say "0.1% to 77%" or "770-fold."

4. **Break the limitations paragraph into 2-3 sub-paragraphs.** Group by theme (generalizability, methodology, scope). Currently it is a wall of text that reviewers will skim.

5. **Add a synthesis sentence to the conclusion.** Currently the conclusion adds nothing the abstract doesn't. One sentence about broader implications ("feature importance structure as a pre-deployment audit tool") elevates the paper.

6. **Sharpen Section 5.1 opening.** Lead with insight ("Despite identical temporal shift, coverage varies 770-fold"), not with a table reference.

7. **Fix "Proposition (informal)" to "Heuristic claim."** "Proposition" implies formal proof in a UAI paper. The honest framing (already noted in the text) should be reflected in the label.

8. **Cut repetitive pre-deployment framing.** The paper states that SHAP concentration is pre-deployment at least 7 times. After establishing it in the abstract and Section 3.5, subsequent mentions should be brief callbacks, not re-explanations.

9. **Strengthen the placebo test (Section 6.4).** Add one interpretive sentence. Currently this important validation is only 2 sentences -- a reviewer might wonder if you are hiding something.

10. **Resolve "robust" overloading.** The word "robust" means both a task category (ROB) and a general quality (threshold robustness). Use "stable" for the latter to avoid confusion.

---

## Minor Polish Items

- **Line 20-21**: `\ie` and `\eg` macros are defined but `\eg` appears only once (line 419). Either use them consistently or remove the macros.
- **Line 96**: "SALT (Supply chain ALlocaTion)" -- the backronym capitalization is distracting. Just say "SALT dataset" and let the citation handle the full name.
- **Line 107**: "with $\alpha = 0.1$ (90\% target coverage)" -- the parenthetical is redundant since $1-\alpha$ is standard.
- **Line 136**: "One might ask" -- too colloquial for UAI. Use "A natural question is whether..."
- **Line 162**: "portfolio diversification in finance" -- fine to keep but consider that UAI reviewers may not find this helpful. ML analogies (e.g., dropout as implicit ensemble) might resonate better.
- **Line 182-189**: Table 1 uses "Cl" as column header for number of classes. Spell out "Classes" or use "$|\mathcal{Y}|$" for consistency with later usage.
- **Line 196**: "classified by median" -- unclear to a first-time reader. Expand to "category assigned using median rather than mean due to high variance."
- **Line 234**: "These motivate" -- vague referent. "These two exceptions motivate..."
- **Line 267**: Figure 2 caption says "(dark)" and "(Lighter points)" -- use specific marker descriptions (e.g., "filled circles" vs. "open triangles") for accessibility.
- **Line 286**: "$\gamma=0.01$ following the original recommendation" -- cite the specific recommendation: "following \citet{gibbs2021adaptive}."
- **Line 290**: Table 4 caption: "Utility: Useful ($<$40\%), Marginal (40--60\%), Vacuous ($>$60\%)" -- this is a definition that belongs in the text body, not crammed into a caption.
- **Line 306**: "$\sim$35\%" -- avoid $\sim$ in a table of precise values. Either compute it or note "estimated" in a footnote.
- **Line 328**: "85\% of maximum improvement" -- what is the maximum? Monthly? Clarify: "85\% of the improvement achieved by monthly retraining."
- **Line 395**: "6--140$\times$ more degradation" -- this range is so wide it is nearly uninformative. Consider reporting median ratio or restricting to vulnerable tasks.
- **Line 402**: Step 1 of the framework repeats the equation reference. Since it was just defined in Section 3.5, a forward reference suffices.
- **Line 413**: "a natural gap separates low-concentration tasks (24--29\%) from high-concentration tasks (43--54\%)" -- a 14-point gap is "natural" only in this dataset. Acknowledge this.
- **Line 462**: Appendix A.1 mixes hyperparameters and seeds in one sentence. Separate them.
- **Line 466**: "$q = \min(\lceil(n+1)(1-\alpha)\rceil/n, 1)$" -- define $n$ (calibration set size).
- **Line 474**: "8 cores, 8GB RAM" -- is this a laptop? A server? Say "single workstation" or similar.
- **Line 580**: Table 8 has "$+0.1314$" for i-plant ECE -- inconsistent precision (4 decimal places vs. 3 elsewhere). Standardize to 3.
- **Figures**: Ensure all figures are referenced before they appear. Figure 1 (fig:shap) is referenced on line 164 before it appears on line 258-262. This is acceptable but non-ideal in a 2-column format; consider reordering.
- **References format**: Check that all references use consistent format (some may have inconsistent capitalization in BibTeX).
- **Passive voice instances** (not exhaustive): "No task-specific tuning was performed" (line 462) --> "We performed no task-specific tuning." "This is \textit{not} ensemble prediction" (line 116) is fine (emphatic).
- **Article issues**: "using a supply chain benchmark" (line 64) -- "the SALT supply chain benchmark" is more specific. "a heuristic argument" (line 148) -- fine.
