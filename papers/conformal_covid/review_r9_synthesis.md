
===============================================================
  TECHNICAL REVIEW REPORT
  Conformal COVID UAI 2026 -- R9 Synthesis | 2026-02-20
===============================================================

## EXECUTIVE SUMMARY

The paper "Diagnosing Conformal Prediction Failures Under Distribution Shift" is in strong submission-ready shape after 9 revision rounds. All 24 BibTeX entries are factually correct, the primary n=16 correlation endpoint is numerically verified, and the insight-level assessment confirms no remaining overstatements. Two moderate fixes remain -- a wrong number in Table 5 and a "domains" vs "datasets" wording issue -- neither of which affects the primary results.

## KEY FINDINGS

### Strengths
- All 24 BibTeX entries verified correct (18 PASS, 2 cosmetic notes, 4 minor style issues, 0 failures) -- LiteratureAgent
- Primary claims (rho=0.853, Theorem 1 verification on 5/5 tasks, mixed-effects p=0.0006) are numerically sound -- MethodCritic
- Hedging language is calibrated to evidence; negative results (RF rho=0.30, MLP rho=0.43) reported honestly -- InsightExtractor
- Statistical apparatus is thorough: ICC, bootstrap CIs, jackknife, Holm correction, leave-one-out -- InsightExtractor
- 50-seed SALT protocol and 10-seed external replication provide strong reproducibility -- MethodCritic

### Concerns
- Stack Overflow concentration in Table 5 reads 7.4% but source data shows 48.9% -- severity: MED -- MethodCritic
- "9 non-supply-chain domains" overstates diversity (Avila/Pendigits overlap, Covertype/Satimage overlap) -- severity: MED -- MethodCritic
- Sub-nominal validation coverage on several datasets (Avila, Gas Sensor, Pendigits, s-group) undisclosed -- severity: LOW -- MethodCritic
- 50-seed vs 10-seed asymmetry mentioned only in table footnotes, not main text -- severity: LOW -- MethodCritic
- NeurIPS booktitle style inconsistency across entries (bare vs parenthetical abbreviation) -- severity: LOW -- LiteratureAgent

### Critical Issues
- None.

## CROSS-AGENT INSIGHTS

All three agents converge on the same conclusion: the paper's primary scientific claims are sound and well-supported, with remaining issues confined to presentation accuracy rather than methodology. MethodCritic's Table 5 error (M1) is the only finding that rises above cosmetic level, and both LiteratureAgent and InsightExtractor implicitly confirm it does not affect the n=16 endpoint since Stack Overflow is excluded. InsightExtractor's observation that the paper is "at its strongest achievable state given the data" aligns with LiteratureAgent's clean BibTeX audit -- no further structural improvements are possible through text revision alone.

## REQUIRED ACTIONS

1. **Fix Stack Overflow concentration in Table 5**: Change 7.4% to 48.9%, update Step 2 from ROB to VULN, and adjust the footnote. (~2 min)
2. **Resolve "9 domains" claim**: Replace "9 non-supply-chain domains" with "9 non-supply-chain datasets" in abstract, Section 3.1, and Section 5.5. (~5 min)

## SUGGESTED IMPROVEMENTS (non-blocking)

- Add one sentence acknowledging sub-nominal validation coverage on several tasks (Section 3.2 or Appendix A.2)
- Add one sentence on seed-count asymmetry (50 vs 10) in Section 3.3 or 5.5
- Standardize NeurIPS booktitle format (drop "(NeurIPS)" from adebayo2018sanity, drop "(ICML)" from koh2021wilds)
- Update dua2017uci URL from /ml to root path

===============================================================
  VERDICT: CONDITIONAL RECOMMEND -- SUBMIT after 2 required fixes (~7 min)
  Reason: Two factual presentation errors (Table 5 number, domain count
  wording) are quickly fixable; once resolved, no blocking issues remain.
  Estimated acceptance probability: 65-75% (structural ceiling from
  n=16 sample size and single external catastrophic case).
===============================================================
