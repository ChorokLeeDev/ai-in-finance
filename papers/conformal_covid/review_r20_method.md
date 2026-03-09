# MethodCritic Review -- Round 20

**Scope**: Verify R20 edits (PSI footnote placement, TreeSHAP paragraph rewrite, lundberg2018consistent bib addition, gibbs2025conditional journal completion). Check for introduced inconsistencies and remaining issues.

---

## 1. TreeSHAP Paragraph (Line 131): Mechanistically Correct

The rewritten "Why gradient-boosted models?" paragraph is now well-grounded:

- **"TreeSHAP computes exact Shapley values for all tree ensembles, including random forests"** -- Correct. This is the factual statement about the algorithm. The revision notes say "exactness claim removed," which I interpret as removing a prior claim that exactness *explains the diagnostic's power*. The current text correctly separates the algorithmic fact (TreeSHAP is exact for all trees) from the architectural argument (bagging dilutes concentration, boosting reinforces it).

- **"The distinction is architectural, not algorithmic"** -- This is the key clarification. The previous version apparently attributed the GBT/RF difference to SHAP algorithm differences. The current version correctly identifies the *model architecture* (sequential boosting vs independent bagging) as the mechanism, not the SHAP computation method. This is a substantive improvement.

- **RF dilution mechanism** -- "each tree is trained independently on a bootstrap sample, so different trees may assign different top features; averaging across trees dilutes global single-feature concentration" -- Mechanistically sound. Bootstrap feature subsampling (max_features) in RF means different trees see different feature subsets, so no single feature can dominate globally even if individual trees concentrate.

- **MLP-SHAP** -- "kernel or deep variants approximates rather than exactly decomposes contributions" -- Correct characterization. KernelSHAP and DeepSHAP are approximations for non-tree models.

**Verdict**: No issues. The paragraph is now mechanistically correct and avoids the prior error.

---

## 2. PSI Footnote Placement (Line 59)

The PSI footnote now appears at the first mention of "PSI" in Contribution 2:

```
PSI\footnote{Population Stability Index (PSI): a monitoring metric standard in credit scoring practice~\citep{siddiqi2006credit}. PSI $>0.002$ indicates detectable nonzero shift; conventional significance threshold is $>0.1$.}
```

This is correctly placed -- PSI is defined where it first appears. No issue.

---

## 3. lundberg2018consistent BibTeX Entry (Line 131 of references.bib)

The entry is:
```bibtex
@article{lundberg2018consistent,
  title={Consistent Individualized Feature Attribution for Tree Ensembles},
  author={Lundberg, Scott M and Erion, Gabriel G and Lee, Su-In},
  journal={arXiv preprint arXiv:1802.03888},
  year={2018}
}
```

**Minor issue (MINOR)**: This paper remains an arXiv preprint (never published at a peer-reviewed venue as far as I can determine). The entry is technically correct as-is. However, the three-author list (Lundberg, Erion, Lee) matches the arXiv v1 listing. The v3 revision (March 2019) lists the same three authors, so this is correct.

The citation in the main text (line 131) is `\citep{lundberg2018consistent,lundberg2020local}`, which correctly dual-cites both the TreeSHAP algorithm paper and the Nature MI applications paper.

---

## 4. gibbs2025conditional Journal Completion (Line 270 of references.bib)

```bibtex
@article{gibbs2025conditional,
  title={Conformal Prediction with Conditional Guarantees},
  author={Gibbs, Isaac and Cherian, John J and Cand{\`e}s, Emmanuel J},
  journal={Journal of the Royal Statistical Society: Series B (Statistical Methodology)},
  volume={87},
  number={4},
  pages={1100--1126},
  year={2025}
}
```

This is a complete entry with volume, number, and page range. No issue.

---

## 5. Cross-Check for Introduced Inconsistencies

### 5a. TreeSHAP citation consistency
- Line 117 (SHAP Concentration section): cites `\citep{lundberg2020local}` alone for TreeExplainer
- Line 131 (Why GBT paragraph): cites `\citep{lundberg2018consistent,lundberg2020local}` for TreeSHAP
- Line 405 (Appendix SHAP Computation): cites `\citep{lundberg2020local}` alone

**MINOR**: The distinction is defensible (line 117 and 405 refer to using TreeExplainer as a tool, line 131 refers to the TreeSHAP algorithm itself), but for consistency the Appendix A.3 "Method: TreeExplainer" line could also dual-cite. Not critical.

### 5b. "exact" usage consistency
The word "exact" appears twice in the paper:
1. Line 92: "conservative rather than exact coverage" (about non-randomized APS) -- correct usage
2. Line 131: "computes exact Shapley values" (about TreeSHAP) -- correct usage

No conflict. The two uses refer to different concepts (coverage exactness vs Shapley value exactness).

### 5c. Contribution 1 (line 57) vs Section 3.6 (line 131) consistency
Contribution 1 says: "the diagnostic is specific to gradient-boosted classifiers where sequential training produces genuine single-feature dependence that bagging (RF) dilutes and MLP-SHAP approximations obscure (Section 3.6)"

Section 3.6 says: "TreeSHAP computes exact Shapley values for all tree ensembles... The distinction is architectural, not algorithmic."

These are consistent. Contribution 1 correctly attributes RF's failure to bagging dilution (architectural) and MLP's failure to SHAP approximation (algorithmic). Section 3.6 elaborates on this distinction.

### 5d. Interpretability paragraph in Related Work (line 74)
```
Our work connects SHAP~\citep{lundberg2017unified,lundberg2020local} to model reliability
```

This cites lundberg2017unified (original SHAP) and lundberg2020local (Nature MI), but does NOT cite lundberg2018consistent (TreeSHAP). This is fine -- the Related Work paragraph discusses SHAP conceptually, not TreeSHAP specifically. The algorithmic citation belongs in the methodology section where it now appears.

---

## 6. Pre-Existing Issues Not Introduced by R20 (Noted for Completeness)

### 6a. Table 2 shows only 4 of 8 tasks
Table 2 (Feature Overlap) shows 4 tasks. The text (line 218) says "Tasks using transaction IDs (Jaccard ~ 0.02) fail catastrophically" but Table 2 only shows 2 such tasks (s-shipcond, s-group). The missing tasks (s-payterms, i-plant, i-shippoint, s-incoterms) would complete the picture. This is a pre-existing presentation choice, not a R20 regression.

### 6b. Jaccard text vs table minor discrepancy
Line 123 says "Jaccard approximately 0" for transaction-ID tasks; line 218 says "Jaccard approximately 0.02"; Table 2 shows exactly 0.02. All consistent (0.02 rounds to approximately 0).

---

## Summary

| # | Issue | Severity | Source |
|---|-------|----------|--------|
| 1 | TreeSHAP paragraph now mechanistically correct | OK | R20 edit |
| 2 | PSI footnote correctly placed | OK | R20 edit |
| 3 | lundberg2018consistent bib entry correct (arXiv preprint) | OK | R20 edit |
| 4 | gibbs2025conditional journal entry complete | OK | R20 edit |
| 5a | TreeExplainer dual-citation inconsistency (line 405 vs 131) | MINOR | Pre-existing |
| 5b-5d | No cross-referencing inconsistencies found | OK | -- |

**Verdict**: R20 edits are clean. No issues introduced. The TreeSHAP paragraph is now mechanistically correct and well-argued. The only minor note is the citation consistency point (5a), which is a style preference, not an error. **No action required.**
