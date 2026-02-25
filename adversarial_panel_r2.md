# Adversarial Panel Review — Round 2 (Feb 26, 2026)
## Focus: Narrative Sharpness, Organization, Experiment Dilution

---

## THE CORE DIAGNOSIS

**Your paper has a clarity problem, not a rigor problem.** You have done every experiment under the sun—and you're right, it dilutes the message. The paper reads like a lab notebook that documents everything you tried, rather than a paper that makes a sharp claim and defends it. A reader finishing this paper should think "I know exactly what they found and why it matters." Right now they think "I'm not sure what the main point was."

**The one-sentence version of your paper should be:**
> "HML→SMB Granger causality was strong in the Normal regime pre-GFC, then structurally broke—and a frozen OOS design finds modest evidence it re-emerged in the Elevated regime post-GFC."

But that sentence is buried under: VaR applications, transfer entropy, four-model complexity diagnostics, MOM→SMB secondary validation, FF25 overlap analysis, early warning detection, trading strategy backtests, MS-VAR comparisons, K-sensitivity, seven local optima clusters, and pages of caveats. **The reader loses the thread.**

---

## PANEL REVIEW: 4 ADVERSARIAL REVIEWERS

### Reviewer 1 — "Narrative Assassin" (Senior AC, top venue experience)

**Verdict: Weak Accept → needs revision for Accept**

**The paper tells three stories simultaneously and finishes none.**

Story A: "Regime-dependent predictive structure with a dated structural break" — This is your strongest card. Pre-GFC Normal HML→SMB is rock-solid (p = 8.75e-9), the Chow break is clean, the TOST confirms extinction. This alone is a publishable finding at ICAIF.

Story B: "OOS regime migration" — Interesting but fragile. p=0.022/0.063 straddling significance, 2/3 optima, doesn't survive Bonferroni. You're honest about this, but you give it equal weight as Story A in the abstract and introduction.

Story C: "Complexity characterization" — Transfer entropy, four-model diagnostic, directional asymmetry. This is a separate paper. It's interesting but it fragments attention from Stories A and B.

**Specific narrative problems:**

1. **Abstract is a data dump, not a story.** Count the statistics in your abstract: p=8.75e-9, ΔR²=2.06%, p=2.29e-6, p=0.73, p=0.022, 0.063, K=3, ΔR²<1%, Sharpe=-0.07, z=5.37, p=0.010. That's 12 numbers in 200 words. A reader's eyes glaze over. The abstract should have max 4-5 numbers: the pre-GFC p-value, the break, the OOS range, and the negative trading result.

2. **Introduction §1.1 "Our Findings" previews everything.** You spend 80 lines re-stating results before the reader has any methodology. This is the "executive summary" style that works for a report but kills narrative momentum in a paper. The reader already knows the punchline before seeing the setup.

3. **Contributions list has 3 items but item 3 bundles 3 sub-findings.** Contribution 3 = "directional complexity" + VaR proof-of-concept. These aren't related. VaR has been previously flagged as disconnected (seed 42 has null Granger). You've demoted it but it's still in the contribution list.

4. **The paper oscillates: strong → weak → strong → weaker.** In-sample: p=8.75e-9 (strong!). But then structural break: p=0.73 post-2008 (oh, it's dead). But then OOS: p=0.022 (it's back!). But then sensitivity: p=0.063 (maybe not). But then MOM→SMB: p=0.010 (confirmed!). But then trading: Sharpe=-0.07 (useless). This whiplash exhausts the reader.

**Fix:** Lead with the structural break as THE story. "We document a predictive link that existed, died, and may be re-emerging in a different regime." Build narrative tension, don't frontload every result.

---

### Reviewer 2 — "Organization Surgeon" (Methodology specialist)

**Verdict: Borderline Accept**

**The paper is 17 pages with 10 appendices. This is 7 pages of appendix for a 10-page main body.** The appendices aren't just supplementary—they contain material the main text constantly references, forcing the reader to jump back and forth.

**Structural problems:**

1. **§3 Methodology runs 4.5 pages and includes Factor Pair Selection (§3.5) which is really a results-adjacent discussion.** The justification for why HML-SMB was chosen includes OOS rank results (rank 2/30) and permutation p-values—this is results masquerading as methodology. Move pair selection justification to Results or Discussion.

2. **§4 Results and §5 Discussion blur together.** The "Frozen OOS Test" (§5.1) is arguably the most important result, yet it's in the Discussion section. The reader expects Discussion to interpret findings, not present the primary validation. This is architecturally confusing.

3. **§5.4 Robustness is a catch-all.** It contains: permutation tests, HAC bandwidth sensitivity, trivariate MKT-RF control, K-sensitivity, MOM→SMB secondary validation, AND a pointer to extended appendix robustness. This is 6 distinct analyses crammed into one subsection with no hierarchy of importance.

4. **The paper has 18 tables and 7 figures in 17 pages.** That's one table or figure every 0.68 pages. Several tables (tab:frozen_events, tab:warning, tab:events) present small-sample, low-power results that don't strengthen the paper. They exist because you ran the analysis, not because the reader needs them.

5. **Related Work (§2) is 1.5 pages of dense citation lists.** Four literature blocks, each ending with a gap statement. The format is dutiful but mechanical. It reads like a checklist rather than a narrative positioning of your work.

**Proposed reorganization:**

| Current | Proposed |
|---------|----------|
| §1 Introduction (1.5pp with full results preview) | §1 Introduction (0.75pp, question + setup only) |
| §2 Related Work (1.5pp) | §2 Related Work (0.75pp narrative) |
| §3 Methodology (4.5pp incl. pair selection) | §3 Methodology (2pp: HMM + Granger + OOS design) |
| §4 Results (3pp) | §4 Results (3pp: in-sample + structural break + frozen OOS) |
| §5 Discussion (4pp) | §5 Robustness (1.5pp: top 4 checks only) |
| §6 Conclusion (0.75pp) | §6 Discussion + Conclusion (1pp) |
| 10 appendices (7pp) | Appendices: consolidate to 5 (cut/merge) |

This saves ~3 pages of main body, brings you comfortably within 10 pages, and sharpens focus.

---

### Reviewer 3 — "The Finance Skeptic" (Quant finance, cares about economic substance)

**Verdict: Weak Reject → Borderline Accept if revised**

**I don't believe the OOS result, and the paper doesn't convince me I should.**

1. **The p=0.022 vs p=0.063 dual reporting is still a credibility issue.** Yes, you've disclosed both, and the range framing is better than the previous round. But the abstract still leads with p=0.022 as "primary" and buries 0.063 as "sensitivity bound." A finance reviewer will see this as p-hacking adjacent. The conservative result should be primary. If your finding can't survive at p=0.063, it's not robust enough to headline.

2. **The OOS Elevated result is fragile along THREE dimensions simultaneously:**
   - Scale convention: p ranges from 0.022 to 0.063
   - Local optima: 2/3 significant, 1/3 null
   - K-sensitivity: requires K=3 exactly (null at K=2, marginal at K=4)

   Any ONE of these would be acceptable. All THREE together make this a finding that survives only under specific analytical choices. The paper acknowledges each individually but never states the compound fragility.

3. **The VaR section is still in the main body (§5.3) and still in Contribution 3.** Previous review unanimously said demote to appendix. You've added caveats ("improvement driven by volatility-override mechanism rather than Granger link") but it's still positioned as evidence of practical value. If the Granger signal doesn't drive the VaR improvement, why is it in the contributions?

4. **Economic magnitude is tiny and you acknowledge it, but then spend pages on applications.** ΔR² < 1% OOS, Sharpe = -0.07, VaR improvement driven by vol-override not Granger. The paper honestly reports all this—good. But then devotes 2+ pages of appendix to VaR methodology, algorithms, hybrid detectors, false alarm analysis. For a signal that doesn't work economically, this is a lot of ink.

5. **The "deleveraging cascade" mechanism is still presented as interpretation despite being labeled speculative.** The FF25 overlap evidence (ρ=0.35, p=0.046) uses the sensitivity fit (seed 42), whose Granger OOS is null (p=0.466). So the portfolio overlap evidence and the Granger evidence come from different HMM solutions. This disconnect is disclosed but not emphasized.

---

### Reviewer 4 — "The Sharp Message Test" (Asks: would I cite this paper? For what?)

**Verdict: Borderline Accept**

**I would cite this paper for exactly one thing: documenting the pre-GFC Normal-regime HML→SMB link and its structural break.** That's a clean, replicable finding. I would NOT cite it for OOS evidence, complexity characterization, VaR applications, or mechanism.

**If the citable contribution is just Story A, does the paper earn its 17 pages?**

The test for "strong accept" at ICAIF: Does this paper change how I think about factor dynamics? The pre-GFC structural break changes my mental model slightly—I now know that cross-factor predictive links can appear and disappear with regime shifts. That's worth publishing. The OOS evidence that it re-emerged post-GFC in a different regime is intriguing but too fragile to change my priors.

**What would make this a strong accept:**

1. **Kill the noise.** Cut VaR from contributions. Cut trading strategy from main body (footnote it). Cut early warning assessment entirely. Cut the four-model diagnostic from main body (appendix). These are all interesting but they're B-plots in what should be an A-plot paper.

2. **Make the structural break the hero.** Your break finding is the strongest result: p=8.75e-9 pre-GFC, Chow p=2.29e-6, TOST-confirmed zero post-2008. This is a 16-year extinction event. Lead with this. The question "what happened?" is more interesting than "does it still exist (maybe, at p=0.063)?"

3. **Frame OOS as epilogue, not co-star.** The OOS Elevated result is interesting as "suggestive evidence of re-emergence in a different regime." It's not interesting as a standalone finding given the fragility. Present it in 1 page, not 3.

4. **Sharpen the abstract to 150 words max.** Current abstract is ~250 words. Every word should serve the narrative. Cut all numbers except: the pre-GFC p-value, the break date/p-value, the OOS range, and the negative trading result.

5. **Rename the paper.** "Regime-Dependent Predictive Structure Between Equity Factors" is accurate but bland. Consider: "The Death and Possible Rebirth of Cross-Factor Predictability: Evidence from HML→SMB Granger Causality" — or something that signals the structural break story.

---

## CONSENSUS: WHAT BLOCKS "STRONG ACCEPT"

| Issue | Reviewers | Severity |
|-------|-----------|----------|
| **Experiment dilution: too many analyses obscure the main finding** | All 4 | HIGH |
| **Abstract is a data dump, not a narrative** | R1, R4 | HIGH |
| **OOS compound fragility not stated explicitly** | R3 | HIGH |
| **VaR still in contributions despite disconnection** | R3, R4 | MED-HIGH |
| **Frozen OOS in Discussion not Results** | R2 | MED |
| **Related Work is mechanical, not narrative** | R1, R2 | MED |
| **17 pages with 18 tables is too much** | R2, R4 | MED |
| **p=0.022 leads over p=0.063 in abstract** | R3 | MED |
| **Complexity diagnostic in main body** | R4 | MED |

---

## PRESCRIPTION: WHAT TO DO (in priority order)

### TIER 1: Must-do for Strong Accept

**1. Restructure around the structural break as the central narrative.**
- Abstract: "We document a Bonferroni-significant HML→SMB Granger link in the Normal regime pre-GFC that undergoes a structural break at January 2008 and is TOST-confirmed absent post-2008. A frozen OOS design finds modest evidence (permutation p ∈ [0.022, 0.063]) that predictive structure re-emerges in the Elevated regime, though this is fragile to scale convention, local optima, and K specification."
- Introduction: Cut §1.1 to 10 lines max. Don't preview every result. State the question, state the approach, state one sentence of main finding.

**2. Drop VaR from Contribution list entirely.**
- Move §5.3 to appendix. It uses a different seed (42) whose Granger is null. The improvement comes from vol-override, not your Granger finding. Keep it as "proof-of-concept in appendix" only.

**3. Explicitly state compound OOS fragility in one sentence.**
- Add to §5.1 or §5.5: "The OOS Elevated result is fragile along three simultaneous dimensions: scale convention (p = 0.022 vs. 0.063), local optima (2/3 significant), and regime count (null at K=2, marginal at K=4). We present it as suggestive rather than confirmatory."

**4. Move complexity diagnostic (four-model + TE) to appendix.**
- Keep one sentence in main text: "A four-model diagnostic finds no nonlinear improvement for forward HML→SMB; transfer entropy reveals a reverse nonlinear SMB→HML channel (z=5.37). See Appendix X."
- This saves ~0.5 pages of main body.

**5. Compress Related Work to 0.75 pages.**
- Kill the four "Gap:" sub-narratives. Write a single flowing paragraph per literature area. This saves ~0.75 pages.

### TIER 2: Strongly Recommended

**6. Move Frozen OOS from Discussion (§5.1) to Results (§4).**
- It's your primary validation. It should be in Results.

**7. Lead with p=0.063 as the primary/conservative OOS permutation result.**
- Report p=0.022 as the percentage-unit sensitivity check, not the other way around. Conservative primary is more credible.

**8. Cut main-body tables from 11 to 7-8.**
- Cut tab:frozen_events (small-sample, 3 events, low power)
- Cut or merge tab:optima_oos into the text (3 rows can be a sentence)
- Consider cutting tab:detection (the 0% detection is interesting but it's a tangent from the main story)

**9. Shorten Conclusion to 0.5 pages.**
- Current conclusion restates everything. State: main finding, limitation, one implication, future work. Done.

### TIER 3: Polish

**10. Contributions list: reduce to 2.**
- C1: Regime-dependent predictive structure with dated structural break
- C2: Modest frozen OOS evidence with disclosed fragility + secondary MOM→SMB confirmation

**11. Cut Early Warning appendix (App C) entirely.**
- Primary fit detects 0/3 events. This weakens the paper. Delete it.

**12. Trading strategy: footnote in Discussion, not a subsection.**
- "A simple trading strategy yields Sharpe = -0.07 (Appendix X), confirming statistical predictability does not imply economic profitability."

---

## CURRENT ASSESSMENT vs. STRONG ACCEPT

| Dimension | Current | After Tier 1 fixes | Strong Accept bar |
|-----------|---------|--------------------|----|
| Narrative clarity | 4/10 | 7/10 | 8/10 |
| Organization | 5/10 | 7/10 | 8/10 |
| Technical rigor | 8/10 | 8/10 | 8/10 |
| Honesty/disclosure | 9/10 | 9/10 | 8/10 |
| Novelty | 6/10 | 7/10 | 7/10 |
| Impact | 5/10 | 6/10 | 7/10 |

**Bottom line:** Your technical execution is already at strong-accept level. Your honesty is exceptional (unusual in finance papers). What's holding you back is that **you're telling the reader everything you did instead of the one thing you found.** The structural break is your story. Everything else is supporting evidence or appendix material. Streamline ruthlessly and this is a strong accept.
