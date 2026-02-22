
===============================================================
  TECHNICAL REVIEW REPORT
  "Regime-Dependent Predictive Structure Between Equity Factors"
  ICAIF 2026 Submission (arXiv:2601.10732) | 2026-02-22
===============================================================

## EXECUTIVE SUMMARY

This paper presents a frozen out-of-sample Granger causality analysis showing that HML-to-SMB predictive structure migrates from the Normal regime (pre-GFC) to the Elevated regime (post-GFC). The research design is careful, the negative alpha result is honest, and the multi-model diagnostic adds novelty. However, a now-resolved code error in the permutation test, outstanding paper-text inconsistencies, and unaddressed multiple-testing burden require attention before submission.

## KEY FINDINGS

### Strengths
- Frozen OOS design (HMM trained 1990-2012, applied without refitting to 2013-2024) eliminates regime-identification circularity -- a genuine methodological contribution (InsightExtractor, MethodCritic)
- Honest negative alpha result (Sharpe = -0.07) with constructive pivot to risk monitoring (hybrid VaR: 5.60% violations, CC p = 0.336) -- rare intellectual honesty in empirical finance (InsightExtractor)
- Four-model diagnostic (Linear/RF/MLP/LSTM) demonstrating only linear Granger captures HML-to-SMB, while transfer entropy reveals asymmetric nonlinear reverse channel (z = 5.37) -- novel complexity characterization (InsightExtractor)
- Well-differentiated from prior literature; gap claims are accurate and defensible (LiteratureReviewer: 4/5)

### Concerns
- **Permutation p-value stale in paper text** -- severity: HIGH -- The code bug (smoothed vs. filtered labels) has been FIXED in `permutation_semantic_drift.py` (now uses `use_filtered=True`). The corrected rerun completed: permutation p = 0.017 (n = 953, seed 28), which is STRONGER than the paper's reported p = 0.031. However, the paper text has NOT been updated to reflect this corrected value. The n discrepancy (paper reports n = 836 for OOS Elevated; rerun gives n = 953) must also be reconciled and explained. (MethodCritic)
- **Multiple testing not fully addressed** -- severity: MED -- HML-SMB was identified via in-sample screening across 30 factor pairs. The paper acknowledges HAC p = 0.041 does not survive 30-pair Bonferroni, but does not propose a formal correction framework. F-p = 0.014 survives 3-regime correction only. (MethodCritic)
- **"41/50 seeds" framing inflates perceived robustness** -- severity: MED -- Three distinct log-likelihood clusters exist (28+13+9 seeds). Effective robustness is 2 of 3 local optima, which the paper now states, but the "41/50" figure still appears prominently in the abstract. (MethodCritic)
- **Bivariate Granger omits MKT-RF as confounder** -- severity: MED -- Market factor could drive both HML and SMB with different lags, creating spurious Granger causality. A trivariate test adding MKT-RF (7 parameters, feasible at n = 836) has not been run. (MethodCritic)
- **Flogel et al. 2022 misdescribed** -- severity: LOW -- Paper says "cross-factor" autocorrelations; actual paper exploits within-factor (own) autocorrelations. Substantive for the gap claim. (LiteratureReviewer)

### Critical Issues
- None remaining after code fix. The permutation test bug was the only critical item and is resolved at the code level. Paper text update is the sole outstanding blocker.

## CROSS-AGENT INSIGHTS

All three agents converge on the paper's core strength: the frozen OOS design is genuinely sound and the honest negative-alpha framing is commendable. The MethodCritic's concerns about multiple testing and bivariate specification are valid methodological caveats that do not invalidate the finding but should be acknowledged more prominently. The LiteratureReviewer's missing Nuriyev et al. (ICAIF 2024) citation is a venue-specific risk -- ICAIF reviewers will likely know this paper -- but differentiation is straightforward.

## REQUIRED ACTIONS

1. **Update paper text with corrected permutation p-value.** Replace p = 0.031 with p = 0.017 across abstract, Section 4, and conclusions. Reconcile and document the n = 836 vs. n = 953 discrepancy (filtered labels with different seeds produce different Elevated-regime counts).
2. **Add Nuriyev, Duan & Yi (ICAIF 2024)** to Related Work with brief differentiation (allocation vs. predictive precedence). Same venue -- reviewers will notice its absence.
3. **Correct Flogel et al. 2022 description** from "cross-factor" to "within-factor (own) autocorrelations."

## SUGGESTED IMPROVEMENTS (non-blocking)

- Add a brief paragraph acknowledging the 30-pair multiple testing burden more explicitly, framing the OOS result as exploratory with the in-sample Normal result as the confirmatory finding.
- Consider running a trivariate Granger test (HML, SMB, MKT-RF) as a robustness check in the appendix.
- Remove "41/50" from the abstract; lead with "2 of 3 distinct local optima" as the primary robustness characterization.
- Add TOST equivalence test for semantic drift (currently relies on non-rejection of KS/t-tests).
- Add Barnett, Barrett & Seth (2009, PRL) for TE-Granger equivalence under linearity (optional but strengthens Section 6).

===============================================================
  VERDICT: CONDITIONAL RECOMMEND
  Reason: The sole critical code error is resolved (corrected
  permutation p = 0.017, stronger than originally reported), but
  the paper text still reports stale values and is missing one
  venue-critical citation. Three specific text updates (est. <1hr)
  convert this to a clean RECOMMEND for submission to ICAIF 2026.
===============================================================
