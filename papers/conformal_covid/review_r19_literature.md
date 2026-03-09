# Literature Review Report — R19
## Paper: "Diagnosing Conformal Prediction Failures Under Distribution Shift: A COVID-19 Case Study"
## Date: 2026-02-22

---

## 1. Summary of Citations Reviewed

The paper contains 30 bibliography entries. All are verified below for accuracy of attribution, description, and BibTeX quality.

---

## 2. Citation Attribution and Description Accuracy

### 2.1 Correctly Cited and Described Works

**Core CP papers:**
- `vovk2005algorithmic` — Vovk, Gammerman, Shafer (2005). Cited correctly for exchangeability guarantees. BibTeX entry is clean (@book, publisher=Springer).
- `shafer2008tutorial` — Shafer & Vovk JMLR 2008, vol 9, pp 371–421. Correctly cited as accessible tutorial. Entry is correct.
- `romano2020classification` — Romano, Sesia, Candès NeurIPS 2020. This is the APS paper. Title "Classification with Valid and Adaptive Coverage" is correct. Authors confirmed. Volume 33, pages 3581–3591 confirmed correct.
- `romano2019conformalized` — Romano, Patterson, Candès NeurIPS 2019 (CQR). Title correct. Volume 32. No pages listed (minor; volume is enough for NeurIPS).
- `tibshirani2019conformal` — Tibshirani, Barber, Candès, Ramdas NeurIPS 2019. Correctly cited for covariate shift with propensity scores. Author order is correct (Ryan J Tibshirani is first author). Entry clean.
- `barber2023conformal` — Barber, Candès, Ramdas, Tibshirani. Annals of Statistics, vol 51, no 2, pp 816–845, 2023. Confirmed correct. Cited correctly for TV-distance bounds beyond exchangeability.
- `angelopoulos2023gentle` — Angelopoulos & Bates, F&T in ML, vol 16, no 4, pp 494–591, 2023. This is the peer-reviewed F&T version (correct year 2023; key name is appropriate). Confirmed correct.
- `ding2023class` — Ding et al. NeurIPS 2023. Title "Class-Conditional Conformal Prediction with Many Classes" confirmed correct. Authors include Ding, Angelopoulos, Bates, Jordan, Tibshirani — confirmed correct. Volume 36. Clean entry.
- `cauchois2021knowing` — Cauchois, Gupta, Duchi. JMLR vol 22, no 81, pp 1–42, 2021. Title "Knowing What You Know: Valid and Validated Confidence Sets in Multiclass and Multilabel Prediction" confirmed correct. Entry clean.
- `gibbs2021adaptive` — Gibbs & Candès NeurIPS 2021. Title "Adaptive Conformal Inference Under Distribution Shift" confirmed correct. Volume 34, pp 1660–1672. Clean entry.
- `gibbs2024conformal` — Gibbs & Candès JMLR 2024. Title "Conformal Inference for Online Prediction with Arbitrary Distribution Shifts." Vol 25, no 162, pp 1–36. Confirmed correct. Cited appropriately as ACI extension.
- `zaffran2022adaptive` — Zaffran et al. ICML 2022. Authors confirmed (Margaux Zaffran, Olivier Féron, Yannig Goude, Julie Josse, Aymeric Dieuleveut). Pages 25834–25866, volume 162, PMLR. Clean PMLR-format entry.
- `feldman2023achieving` — Feldman, Ringel, Bates, Romano. TMLR 2023. Confirmed correct 4-author list (Angelopoulos is NOT an author — correct). Journal field = "Transactions on Machine Learning Research" (TMLR is a journal so @article is correct). No volume/pages in entry — TMLR does not assign volume/issue numbers, so omission is acceptable. Clean.
- `bhatnagar2023improved` — Bhatnagar, Wang, Xiong, Bai. ICML 2023. Title "Improved Online Conformal Prediction via Strongly Adaptive Online Learning." Pages 2337–2363. PMLR publisher. Confirmed correct.
- `angelopoulos2024conformal` — Angelopoulos, Bates, Fisch, Lei, Schuster. ICLR 2024. Title "Conformal Risk Control." Confirmed this is ICLR 2024 (NOT JASA — a common misattribution, which this paper correctly avoids). Clean.
- `podkopaev2021distribution` — Podkopaev & Ramdas. UAI 2021. Title "Distribution-Free Uncertainty Quantification for Classification under Label Shift." PMLR vol 161, pp 844–853. URL to proceedings included. Confirmed correct.
- `koh2021wilds` — Koh et al. ICML 2021. WILDS paper. Authors abbreviated with "others." Pages 5637–5664, volume 139, PMLR. Title in braces for correct casing. Confirmed correct. No parenthetical venue abbreviation in booktitle (clean, per memory note about removing such abbreviations).
- `gulrajani2021search` — Gulrajani & Lopez-Paz. ICLR 2021. Described in prose as "domain generalization methods often fail to improve over ERM under realistic distribution shifts." This is the correct characterization of the DomainBed paper (NOT specific to temporal shift). Confirmed correct and free of the common over-narrow paraphrase error noted in memory.
- `malinin2021shifts` — Malinin et al. NeurIPS 2021 D&B Track. Booktitle is "Advances in Neural Information Processing Systems Datasets and Benchmarks Track" (correct; not the main conference volume). Clean. Memory note: should be @inproceedings, not @article — confirmed it IS @inproceedings. Correct.
- `gretton2012kernel` — Gretton et al. JMLR 2012. "A Kernel Two-Sample Test." Vol 13, pp 723–773. Confirmed correct (all authors including Borgwardt, Rasch, Schölkopf, Smola). Clean.
- `lopez2017revisiting` — Lopez-Paz & Oquab. ICLR 2017. "Revisiting Classifier Two-Sample Tests." Confirmed correct (C2ST paper). Clean.
- `lei2018distribution` — Lei et al. JASA 2018. "Distribution-Free Predictive Inference for Regression." Vol 113, no 523, pp 1094–1111. Confirmed correct (split CP foundational reference). Clean.
- `garg2022leveraging` — Garg et al. ICLR 2022. "Leveraging Unlabeled Data to Predict Out-of-Distribution Performance." Confirmed correct. Cited correctly for requiring test-time observations.
- `miller2021accuracy` — Miller et al. ICML 2021. "Accuracy on the Line." Pages 7721–7735, vol 139, PMLR. Confirmed correct author list (Miller, Taori, Raghunathan, Sagawa, Koh, Shankar, Liang, Carmon, Schmidt). Clean.
- `lundberg2017unified` — Lundberg & Lee NeurIPS 2017. "A Unified Approach to Interpreting Model Predictions." Volume 30. Confirmed correct (original SHAP paper). Clean.
- `lundberg2020local` — Lundberg et al. Nature Machine Intelligence 2020. "From local explanations to global understanding with explainable AI for trees." Vol 2, no 1, pp 56–67. Confirmed correct. All 10 authors listed (Lundberg, Erion, Chen, DeGrave, Prutkin, Nair, Katz, Himmelfarb, Bansal, Lee). This is the TreeExplainer/TreeSHAP paper. Clean.
- `fey2024relbench` — Robinson et al. NeurIPS 2024 D&B Track. All 12 authors listed and confirmed correct. Booktitle is "Advances in Neural Information Processing Systems Datasets and Benchmarks Track" (correct D&B track designation). Clean.
- `ke2017lightgbm` — Ke et al. NeurIPS 2017. "LightGBM: A Highly Efficient Gradient Boosting Decision Tree." Authors: Ke, Meng, Finley, Wang, Chen, Ma, Ye, Liu. Volume 30. All 8 authors confirmed correct. Clean.
- `dua2017uci` — Dua & Graff 2017. @misc with howpublished containing institution text and URL (https://archive.ics.uci.edu — correct updated URL, not the old /ml path). Confirmed clean per memory note.
- `siddiqi2006credit` — Siddiqi 2006 Wiley book. Cited in footnote for PSI as a credit scoring practice metric. Clean @book entry.
- `gardner2023tableshift` — Gardner, Popovic, Schmidt. NeurIPS 2023 D&B Track. Title "TableShift: Benchmarking the Robustness of Tabular Learning Methods." Booktitle is "Advances in Neural Information Processing Systems Datasets and Benchmarks Track." Clean.
- `kasa2023empirically` — Kasa & Taylor 2023. Authors confirmed correct (Kevin Kasa, Graham W. Taylor). @misc with howpublished referencing arXiv:2307.01088 and ICML 2023 workshop. Correct per R12 fix.
- `kasa2025adapting` — Kasa, Zhang, Yang, Taylor. UAI 2025 PMLR v286, pp 1990–2010. Confirmed correct: proceedings URL (https://proceedings.mlr.press/v286/kasa25a.html) verifies pages and volume. Entry is @inproceedings with full PMLR fields. Clean per R17 fix.

**Gibbs 2025 conditional guarantees:**
- `gibbs2025conditional` — Gibbs, Cherian, Candès. JRSS-B vol 87, no 4, pp 1100–1126, 2025. Confirmed correct: Oxford Academic URL confirms volume 87, issue 4, pages 1100–1126. arXiv 2305.12616 confirmed. Author list (Isaac Gibbs, John J Cherian, Emmanuel J Candès) confirmed correct. Entry clean.

**Paper description of gibbs2025conditional in prose:** "provide finite-sample conditional guarantees via a covariate-shift reweighting framework interpolating between marginal and conditional validity." This accurately captures the paper's mechanism (it defines a spectrum from marginal to conditional validity via covariate shift classes). Confirmed correct.

---

## 3. Critical Issue: Technically Inaccurate Claim in "Why Gradient-Boosted Models?" Paragraph

**Location:** Section 3.5 (Why SHAP Concentration?, subsection "Why gradient-boosted models?"), lines 131.

**The claim (verbatim):**
> "TreeSHAP [lundberg2020local] computes exact Shapley values via recursive path enumeration, making SHAP concentration a faithful measure of the model's actual conditional dependence on a single feature."

This sentence, by its placement, implies that exactness of TreeSHAP is a property specific to gradient-boosted models that distinguishes them from RF and MLP. This is **technically incorrect**.

**The facts (confirmed via literature search):**

1. TreeSHAP (Lundberg et al. 2018, arXiv:1802.03888; implemented in lundberg2020local) computes exact Shapley values for ALL tree ensemble models, including Random Forests. The algorithm applies to "decision trees, random forests, and gradient boosted trees" — RF is explicitly included. The polynomial-time exact computation is via the same recursive path-dependent or interventional algorithm applied to each constituent tree and summed.

2. The paper then says: "MLP-SHAP (kernel or deep variants) approximates rather than exactly decomposes contributions, adding noise that obscures structural concentration." This part is correct — kernel SHAP IS approximate for neural networks. The MLP distinction is therefore valid.

3. But the RF claim rests entirely on the learning mechanism (bagging averages many independent trees, diluting concentration), NOT on any difference in the exactness of TreeSHAP computation. TreeSHAP is equally exact for RF as for LightGBM/XGBoost/CatBoost.

**Consequence:** The sentence as written implies RF-SHAP is less faithful than LGB-SHAP due to a TreeSHAP algorithmic property. The true reason for the diagnostic failure on RF is that RF's bagging mechanism produces models that structurally do NOT concentrate importance on single features, so there is no concentrated signal for TreeSHAP to report — not that TreeSHAP is less exact for RF.

**Risk:** A reviewer familiar with the SHAP literature (Lundberg 2018 is widely cited) will immediately notice this. The paper correctly states the empirical consequence (RF dilutes concentration via bagging) but the mechanistic framing citing TreeSHAP exactness as a differentiator for gradient boosting is misleading.

**Recommended fix:** Replace "making SHAP concentration a faithful measure of the model's actual conditional dependence on a single feature" with a clause that is neutral about exactness across tree types, and ground the RF distinction solely in the learning mechanism. Suggested revision for that sentence:

> "TreeSHAP [lundberg2020local] computes exact Shapley values via recursive path enumeration for any tree ensemble—including random forests—making SHAP concentration a faithful measure of *each model's* actual learned dependence structure."

Then the RF paragraph should read: "Random forests average over many independent trees, each potentially with different top features, diluting concentration even when individual trees concentrate heavily on one feature—confirmed empirically by RF ρ=0.30 versus LightGBM ρ=0.833. The diagnostic therefore fails for RF not because TreeSHAP is less accurate, but because RF's bagging mechanism prevents the concentrated single-feature dependence from forming in the first place."

This framing is fully supported by the data in the paper and does not require any empirical changes.

---

## 4. Missing Citation for TreeSHAP Algorithm Original Paper

**Location:** Section 3.5, "Why gradient-boosted models?" paragraph.

The paper cites `lundberg2020local` (Nature Machine Intelligence 2020, TreeExplainer paper) for TreeSHAP. However, the original algorithmic description of TreeSHAP is in a **prior paper**:

> Lundberg, S.M., Erion, G.G., and Lee, S.-I. (2018). "Consistent Individualized Feature Attribution for Tree Ensembles." arXiv:1802.03888.

The 2020 Nature Machine Intelligence paper builds on and extends TreeSHAP, but the core polynomial-time recursive algorithm and its exactness proof appear in the 2018 arXiv paper. When making specific algorithmic claims about "recursive path enumeration" and exactness, the 2018 paper is the more precise citation. This is a minor but reviewers from the XAI community will notice.

**Severity:** Low — the 2020 paper does describe TreeSHAP and is widely cited for it. But in the context of claiming algorithmic properties (exact computation, path enumeration), adding or substituting the 2018 arXiv citation strengthens the claim. Some reviewers may query why the algorithm is cited to a journal paper when the preprint with the full algorithmic treatment appeared two years earlier.

**Recommended action:** Add a citation to the 2018 arXiv paper when stating algorithmic properties, or add a footnote clarifying that the exact algorithm was introduced in the 2018 preprint and extended in the 2020 paper.

---

## 5. Citation for RAPS — Title and Description Check

**Location:** Section 4.2 heading and prose: "Regularized Adaptive Prediction Sets (RAPS) [angelopoulos2021uncertainty]"

The BibTeX entry `angelopoulos2021uncertainty` has:
- Title: "Uncertainty Sets for Image Classifiers using Conformal Prediction"
- Venue: ICLR 2021
- Authors: Angelopoulos, Bates, Malik, Jordan

The paper refers to this as "RAPS" in the text, which is the name of the method introduced in that paper. The citation is correct — "Uncertainty Sets for Image Classifiers using Conformal Prediction" IS the RAPS paper. The paper correctly expands the acronym on first use ("Regularized Adaptive Prediction Sets (RAPS)"). No issue.

**Note on author order:** The bib entry has "Malik, Jitendra" before "Jordan, Michael I" which matches the published paper's author order (Angelopoulos, Bates, Malik, Jordan). Confirmed correct.

---

## 6. Related Work Positioning Assessment

The Related Work section (Section 2) positions the contribution accurately across five paragraphs:

1. **CP foundations:** Appropriate selection of Vovk 2005, Shafer & Vovk 2008, Romano 2020, Cauchois 2021, Romano 2019, Lei 2018, Angelopoulos 2023. No glaring omissions for a UAI submission.

2. **CP under shift:** Tibshirani 2019, Podkopaev 2021, Barber 2023, Kasa 2023 — appropriate and accurate. The description of Kasa 2023 ("characterize how CP degrades across many vision architectures and shift types") accurately reflects the arXiv workshop paper on modern vision architectures under distribution shift and long-tailed data. The paper's differentiation ("complementary question of which tabular models will fail before test data is observed") is a clear and accurate positioning.

3. **Adaptive methods:** Gibbs 2021/2024, Zaffran 2022, Feldman 2023, Bhatnagar 2023, Angelopoulos 2024, Kasa 2025 — comprehensive and accurate. The description of Kasa 2025 (ECP/EACP, entropy-based, using unlabeled test data) is accurate and the differentiation from SHAP concentration (no test observations required) is valid.

4. **Shift detection:** WILDS, Shifts, Gulrajani, Gardner, Garg, Gretton, Lopez-Paz, Miller — all accurately described. The Gulrajani description ("domain generalization methods often fail to improve over ERM under realistic distribution shifts") is correct per memory note. The Miller 2021 "Accuracy on the Line" citation and description ("ID validation accuracy predicts OOD accuracy for image classifiers") is accurate.

5. **Interpretability:** Lundberg 2017 + 2020 cited for SHAP. Description accurate ("extending feature attribution from post-hoc analysis to prospective failure diagnosis").

**One potential gap:** Ding et al. (2023) is described as proposing "clustered conformal prediction as a remedy" for class-conditional coverage with many classes. This is correct. The Gibbs 2025 (conditional guarantees) description as "covariate-shift reweighting framework interpolating between marginal and conditional validity" is accurate. These two are correctly framed as confounds that the paper explicitly examines via partial correlation analysis. This is a well-articulated positioning.

---

## 7. BibTeX Entry Quality Issues

### 7.1 Minor Issues (non-fatal)

- `romano2019conformalized`: No pages field (NeurIPS 2019, volume 32). Pages would be 7029–7038 for the CQR paper. Not strictly required but inconsistent with romano2020classification which has pages. Minor.

- `gibbs2021adaptive`: Has pages (1660–1672) and volume 34. NeurIPS 2021. Confirmed correct.

- `gibbs2025conditional`: Entry has journal = "Journal of the Royal Statistical Society: Series B" — this should arguably include "Statistical Methodology" as the full subtitle. Full official journal name is "Journal of the Royal Statistical Society Series B: Statistical Methodology." The entry omits ": Statistical Methodology." Low severity — journal is recognizable either way. **Recommended: add "Statistical Methodology" to journal name for precision.**

- `kasa2023empirically`: @misc with howpublished = "arXiv:2307.01088. Presented at ICML 2023 Workshop..." This is appropriate for an unpublished workshop paper. The howpublished field correctly carries both the arXiv ID and workshop context. Acceptable.

- `ke2017lightgbm`: Volume is 30, year 2017. LightGBM NeurIPS 2017 vol 30 is confirmed correct. No pages, which is common for NeurIPS pre-2020. Acceptable.

- `malinin2021shifts`: @inproceedings with booktitle = "Advances in Neural Information Processing Systems Datasets and Benchmarks Track" — correct (not the main NeurIPS volume). No volume/pages (D&B track proceedings are in a separate proceedings volume; omitting is acceptable). Clean.

- `gardner2023tableshift`: Same @inproceedings/D&B track format as malinin2021shifts. No volume/pages. Acceptable and consistent.

- `fey2024relbench`: Same D&B track format. Clean per prior review rounds.

### 7.2 Booktitle consistency

The three D&B track entries (malinin2021shifts, gardner2023tableshift, fey2024relbench) all use "Advances in Neural Information Processing Systems Datasets and Benchmarks Track" consistently. This is correct and uniform.

NeurIPS main-track entries (romano2020classification, gibbs2021adaptive, ding2023class, ke2017lightgbm, romano2019conformalized, tibshirani2019conformal, bhatnagar2023improved) use "Advances in Neural Information Processing Systems" without parenthetical abbreviation. Consistent and clean.

ICML entries (koh2021wilds, miller2021accuracy, bhatnagar2023improved) use "International Conference on Machine Learning" with PMLR fields. Consistent.

ICLR entries (gulrajani2021search, garg2022leveraging, angelopoulos2021uncertainty, angelopoulos2024conformal) use "International Conference on Learning Representations." No pages or volume (ICLR does not publish volumes in the traditional sense). Consistent.

UAI entry (podkopaev2021distribution, kasa2025adapting) use "Proceedings of the ... Conference on Uncertainty in Artificial Intelligence" with PMLR volume/series/publisher. Consistent.

**Overall BibTeX consistency: GOOD.** No mixing of parenthetical suffixes.

---

## 8. Potentially Suspicious or Fabricated Citations

None detected. All 30 entries correspond to verifiable real papers with correct authors, venues, and years. The most recently added entries (gibbs2025conditional, kasa2025adapting, kasa2023empirically) from prior review rounds are confirmed correct via web search. No hallucinated authors or venues found.

---

## 9. Summary of Findings

| Issue | Severity | Action Required |
|-------|----------|-----------------|
| TreeSHAP exactness claim implies RF uses approximate SHAP (incorrect) | **HIGH** | Revise prose in Section 3.5 "Why gradient-boosted models?" |
| Missing citation for original TreeSHAP algorithm (Lundberg 2018 arXiv:1802.03888) | Medium | Add/substitute citation when stating algorithmic properties |
| `gibbs2025conditional` journal name missing ": Statistical Methodology" subtitle | Low | Add subtitle to journal field |
| `romano2019conformalized` missing pages | Low | Add pages = {7029--7038} (optional) |
| All other entries | OK | No action needed |

---

## 10. Priority Recommendation

The TreeSHAP exactness claim (Issue #1) should be fixed before submission. It is the only substantive error: the prose implies a property of the TreeSHAP algorithm that is incorrect (TreeSHAP is exact for RF too), and a reviewer with XAI background will immediately flag it. The fix requires only 2–3 sentence revisions in Section 3.5 with no changes to any results. The remaining issues are low-priority cosmetic BibTeX improvements.
