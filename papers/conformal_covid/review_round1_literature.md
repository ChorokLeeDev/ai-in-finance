# Literature Review Audit — UAI 2026 Submission
**Paper:** "Diagnosing Conformal Prediction Failures Under Distribution Shift: A COVID-19 Case Study"
**Review date:** 2026-02-20
**Reviewer:** Literature Review Agent (automated)

---

## 1. Checklist: Core Conformal Prediction Citations

### 1.1 Foundational CP Papers

| Paper | Status | Note |
|---|---|---|
| Vovk, Gammerman, Shafer (2005) *Algorithmic Learning in a Random World* | CITED (`vovk2005algorithmic`) | Correct — primary exchangeability reference |
| Shafer & Vovk (2008) *A Tutorial on Conformal Prediction*, JMLR 9:371-421 | **MISSING** | Widely-cited accessible intro; often expected alongside the book |
| Lei et al. (2018) *Distribution-Free Predictive Inference for Regression*, JASA | **MISSING** | Foundational split CP paper for regression; cited in most CP papers |
| Romano, Patterson, Candès (2019) *Conformalized Quantile Regression* (NeurIPS) | CITED (`romano2019conformalized`) | OK |
| Romano, Sesia, Candès (2020) *Classification with Valid and Adaptive Coverage* (NeurIPS) — APS | CITED (`romano2020classification`) | Correct — this is the APS paper |
| Tibshirani, Barber, Candès, Ramdas (2019) *Conformal Prediction Under Covariate Shift* (NeurIPS) | CITED (`tibshirani2019conformal`) | OK |
| Barber, Candès, Ramdas, Tibshirani (2023) *Conformal Prediction Beyond Exchangeability*, Annals of Statistics | CITED (`barber2023conformal`) | OK |
| Angelopoulos & Bates (2021/2023) *A Gentle Introduction to Conformal Prediction* | CITED (`angelopoulos2021gentle`) | OK — listed as 2021 arXiv but published as Foundations & Trends 2023 |
| Angelopoulos, Bates, Malik, Jordan (2021) *RAPS* (ICLR) | CITED (`angelopoulos2021uncertainty`) | OK — cited as "Uncertainty Sets for Image Classifiers using Conformal Prediction" |
| Gibbs & Candès (2021) *Adaptive Conformal Inference Under Distribution Shift* (NeurIPS) | CITED (`gibbs2021adaptive`) | OK |
| Zaffran et al. (2022) *Adaptive Conformal Predictions for Time Series* (ICML) | CITED (`zaffran2022adaptive`) | OK |
| Feldman, Bates, Romano (2023) *Achieving Risk Control in Online Learning Settings* (TMLR) | CITED (`feldman2023achieving`) | OK |
| Angelopoulos, Bates et al. (2024) *Conformal Risk Control* (ICLR 2024) | CITED (`angelopoulos2024conformal`) | Cited as JASA; **publication venue may be incorrect** — paper published at ICLR 2024, not JASA |
| Podkopaev & Ramdas (2021) *Distribution-Free UQ for Classification under Label Shift* (UAI) | CITED (`podkopaev2021distribution`) | OK |

### 1.2 RAPS Citation Accuracy

The paper cites `angelopoulos2021uncertainty` as:
- **Cited title:** "Uncertainty Sets for Image Classifiers using Conformal Prediction"
- **Venue:** ICLR 2021
- This is correct. RAPS is the key algorithmic contribution of this paper. Citation is accurate.

### 1.3 Lundberg 2020 TreeExplainer Citation

The `lundberg2020local` entry in references.bib is:
- **Title:** "From local explanations to global understanding with explainable AI for trees"
- **Journal:** Nature Machine Intelligence, vol. 2, no. 1, pp. 56-67, 2020
- **Authors:** Lundberg, Erion, Chen, DeGrave, Prutkin, Nair, Katz, Himmelfarb, Bansal, Lee

This is **correct**. The paper was published in Nature Machine Intelligence in January 2020. The BibTeX entry and attribution are accurate.

The original SHAP paper `lundberg2017unified` ("A Unified Approach to Interpreting Model Predictions," NeurIPS 2017) is also correctly cited.

---

## 2. Missing or Inadequately Cited Papers

### 2.1 High Priority — Should Be Added

**[M1] Shafer & Vovk (2008). "A Tutorial on Conformal Prediction." JMLR 9:371-421.**
- Widely expected as a companion reference to the Vovk 2005 book.
- Introduces the key concepts of p-values in CP more accessibly.
- Missing from references; not cited anywhere in the paper.
- **Priority: Medium** — not essential given the 2005 book is cited, but reviewers may notice its absence.

**[M2] Lei et al. (2018). "Distribution-Free Predictive Inference for Regression." JASA 113(523):1094-1111.**
- Foundational split conformal prediction paper; the standard reference for the split CP setup used in this paper.
- The paper uses split conformal prediction (50/50 calibration/evaluation split) but cites neither this paper nor the Papadopoulos et al. transductive CP lineage explicitly.
- **Priority: Medium-High** — the methodology section describes split CP but does not cite the canonical reference for it.

**[M3] Gibbs & Candès (2022/2023). "Conformal Inference for Online Prediction with Arbitrary Distribution Shifts." JMLR 25.**
- This is a 2022 arXiv paper (arXiv:2208.08401) extending ACI to arbitrary shifts, later published in JMLR 2024.
- The paper cites the original Gibbs & Candès (2021) ACI paper but misses this important extension.
- **Priority: Medium** — Section 6.1 discusses ACI experiments; this paper would be natural to cite alongside the original ACI.

**[M4] Bhatnagar et al. (2023). "Improved Online Conformal Prediction via Strongly Adaptive Online Learning." ICML 2023.**
- Proposes SAOCP and SF-OGD as stronger online CP methods, directly building on ACI.
- This is another major ACI extension the paper should acknowledge in Section 6.1 discussion (alongside Zaffran et al. and Feldman et al.).
- **Priority: Low-Medium** — complementary; Section 5.3 (adaptive methods related work) only cites Zaffran (2022) as an extension.

**[M5] Angelopoulos & Bates (2023). "Conformal Prediction: A Gentle Introduction." Foundations and Trends in Machine Learning 16(4):494-591.**
- The current citation `angelopoulos2021gentle` references only the 2021 arXiv version.
- UAI reviewers may prefer the officially published Foundations & Trends citation (2023).
- **Action required:** Update BibTeX entry to reflect the 2023 published version, not just the arXiv preprint.

**[M6] Quiñonero-Candela et al. (2009). "Dataset Shift in Machine Learning." MIT Press.**
- Standard reference for the taxonomy of distribution shift (covariate, prior probability, concept).
- The paper discusses covariate shift, temporal shift, and label shift extensively; this book is the canonical reference.
- **Priority: Low-Medium** — expected by reviewers with shift detection background.

### 2.2 Recent Papers (2023-2026) That May Be Expected

**[R1] Gibbs, Cherian, Candès (2023). "Conformal Prediction with Conditional Guarantees." arXiv:2305.12616.**
- Addresses conditional coverage — relevant since the paper implicitly relies on marginal coverage.
- Could strengthen the related work by acknowledging the conditional vs. marginal coverage gap.
- **Priority: Low** — not core to the paper's contribution, but expected in a thorough CP related-work section.

**[R2] Angelopoulos et al. (2022/2024). "Conformal Risk Control." ICLR 2024.**
- **CITATION ACCURACY ISSUE:** The BibTeX entry `angelopoulos2024conformal` lists this as a JASA paper ("Journal of the American Statistical Association"). The actual publication is ICLR 2024.
- The paper cites this as `angelopoulos2024conformal` with `journal={Journal of the American Statistical Association}` — this appears to be an error. The "Conformal Risk Control" paper was published at ICLR 2024 (OpenReview ID: 33XGfHLtZg). The JASA-like paper by the same group is "Learn then Test" (Angelopoulos et al., 2022, arXiv:2110.01052).
- **Action required:** Correct the venue in the BibTeX entry to ICLR 2024.

**[R3] Wasserstein-Regularized Conformal Prediction (Interdisciplinary NeurIPS/ICLR 2024-2025 papers).**
- Recent work bounds coverage gaps by Wasserstein distance — directly related to the Barber et al. (2023) TV-distance bound that the paper builds on.
- Not yet clearly attributable to a single citable paper; skip for now.
- **Priority: Low.**

---

## 3. Section 2 (Related Work) — Accuracy and Completeness Assessment

### 3.1 "Conformal Prediction" paragraph

**Claim:** "Recent work extends to classification [Romano 2020] and regression [Romano 2019]."

**Assessment:** This is accurate but incomplete. Lei et al. (2018) is the standard split CP for regression reference, predating Romano 2019 CQR. The CQR paper builds on Lei et al. and should ideally cite it, or the related work section should acknowledge the split CP lineage. However, the sentence is not inaccurate — Romano 2020 (APS) is the right classification reference.

**Verdict:** Acceptable but incomplete. Adding Lei et al. (2018) would strengthen the claim.

### 3.2 "Conformal Prediction Under Shift" paragraph

**Claim:** "Tibshirani et al. (2019) study covariate shift with known propensity scores. Podkopaev & Ramdas (2021) develop distribution-free UQ for classification under label shift. Barber et al. (2023) provide a comprehensive treatment beyond exchangeability, bounding coverage loss by the total variation distance between test and calibration score distributions. We complement these theoretical characterizations with an empirical diagnostic..."

**Assessment:** The characterization of all three cited papers is accurate:
- Tibshirani et al. (2019): uses importance weighting (propensity scores) — correct.
- Podkopaev & Ramdas (2021): label shift setting, not covariate shift — correct.
- Barber et al. (2023): TV-distance bound — correct.

The claim that "prior work characterizes how CP degrades under shift" but not "which models fail" is **broadly accurate**. These papers provide theoretical bounds, not model-specific pre-deployment diagnostics. The differentiation is valid.

**Verdict:** Accurate. The characterization of all three papers is faithful to their contributions.

### 3.3 "Adaptive Methods" paragraph

**Claim:** ACI (Gibbs 2021) recovers marginal coverage but at the cost of prediction-set informativeness.

**Assessment:** This is empirically supported in Section 6.1. The claim about Feldman (2023) for "online risk control" is accurate. Angelopoulos (2024) generalizing to "broader risk measures" refers to Conformal Risk Control — the venue issue (see M2 above) is a concern but does not affect the characterization.

**Note:** The paper states "Our experiments show ACI recovers marginal coverage under severe shift but at a substantial cost in prediction set informativeness." This is empirically demonstrated in Section 6.1. However, the paper does not cite Gibbs & Candès (2022/2023) JMLR extension which provides tighter guarantees for arbitrary shifts.

**Verdict:** Accurate description; missing the 2022 ACI extension.

### 3.4 "Shift Detection" paragraph

**Claim:** "WILDS and Shifts provide benchmarks... Garg et al. (2022) use unlabeled test data... MMD and C2ST detect distributional differences but do not predict the severity of downstream failure."

**Assessment:** All characterizations are accurate:
- WILDS (Koh 2021): correct.
- Shifts (Malinin 2021): correct.
- Gulrajani (2020/2021): correct — "in search of lost domain generalization."
- Garg et al. (2022): correctly described as requiring test-time observations.
- Gretton et al. (2012) MMD: correct.
- Lopez-Paz & Oquab (2017) C2ST: correct.

The claim that MMD and C2ST "detect shift for all tasks uniformly but cannot distinguish catastrophic from robust outcomes" is empirically verified in Section 6.3 (all p < 0.002 for MMD, all C2ST > 99.9%). This is a strong and well-supported empirical claim.

**Verdict:** Accurate. No mischaracterizations found.

### 3.5 "Interpretability for Reliability" paragraph

**Claim:** "Our work connects SHAP (Lundberg 2017) to model reliability assessment, extending the use of feature attribution from debugging (Adebayo 2018) to prospective failure diagnosis."

**Assessment:**
- Lundberg (2017) SHAP unified framework: correct.
- Adebayo et al. (2018) "Sanity Checks for Saliency Maps": this paper is about testing whether attribution methods are sensitive to model parameters. It is used in the context of "debugging." The characterization of it as a "debugging" use of attribution is a stretch — the paper is more about evaluating the validity of saliency maps themselves. However, it is not a mischaracterization in the broad sense.

**Missing citation:** Lundberg et al. (2020) TreeExplainer is the paper actually used for SHAP computation (cited in methodology as `lundberg2020local`). The related work paragraph only cites the 2017 SHAP paper. It would strengthen the related work to briefly mention that TreeExplainer (Lundberg 2020) enables exact, efficient SHAP computation for tree-based models, which is why it is applicable here.

**Verdict:** Accurate but thin. The "Interpretability for Reliability" paragraph is only two sentences. No other key SHAP/XAI papers are discussed — this may be adequate given the paper's focus, but reviewers may expect at least a brief acknowledgment of the XAI reliability literature (e.g., Rudin 2019 on interpretable models, or Molnar et al. on feature importance reliability).

---

## 4. Benchmark and Dataset Citations

### 4.1 RelBench / SALT

**Claim:** "We use the SALT dataset from RelBench (Fey et al. 2023)."

**Assessment:** The BibTeX entry is:
```
@article{fey2023relbench,
  title={RelBench: A Benchmark for Deep Learning on Relational Databases},
  author={Fey, Matthias and others},
  journal={arXiv preprint},
  year={2023}
}
```

Issues:
1. The arXiv number is missing from the BibTeX entry (`journal={arXiv preprint}` without the ID).
2. The "others" shortcut is acceptable in anonymous submission but should be expanded in the final version.
3. The actual arXiv ID is 2407.20060 (RelBench: A Benchmark for Deep Learning on Relational Databases). Verify this and add it.

**Verdict:** Functionally adequate for anonymous submission; needs cleanup before camera-ready.

### 4.2 WILDS Benchmark

The WILDS citation (`koh2021wilds`) is in the related work as a shift benchmark reference, not as a dataset the paper uses. This is appropriate — the paper does not evaluate on WILDS datasets.

### 4.3 External Validation Datasets

The paper uses Covertype, KDDCup99, Gas Sensor Array Drift, Stack Overflow, Avila, PAMAP2, Pendigits, Satimage, Shuttle as external validation datasets. None of these have explicit citations in the paper or references section.

**Missing citations:**
- Forest Covertype: Blackard & Dean (1999), originally from UCI ML Repository.
- KDDCup99: Tavallaee et al. (2009) is the standard reference, or the original KDD Cup 1999 competition.
- Gas Sensor Array Drift: Vergara et al. (2012), UCI dataset #224.
- UCI ML Repository: Dua & Graff (2017) — `@misc{Dua:2019}` — is the standard repository citation.

**Priority: Medium-High.** UAI reviewers may flag the lack of citations for external validation datasets. At minimum, a single UCI repository citation should be added.

---

## 5. Shift Detection Literature Completeness

### 5.1 PSI (Population Stability Index)

The paper evaluates PSI as a shift detector but provides no citation for it. PSI is a practitioner method from credit scoring (Yeh & Lien 2009 is sometimes cited; more commonly it is treated as domain knowledge without a specific academic reference). The paper correctly notes this in the experiments but should either cite the original source or note that it is a practitioner heuristic without a canonical academic citation.

**Action:** Add a footnote or parenthetical acknowledging PSI's practitioner origins (credit scoring/financial industry standard, no canonical academic paper).

### 5.2 C2ST Citation

`lopez2017revisiting` (Lopez-Paz & Oquab, 2017, ICLR) is the correct citation for C2ST. Accurate.

### 5.3 MMD Citation

`gretton2012kernel` (Gretton et al., 2012, JMLR) is the correct citation for MMD. Accurate.

---

## 6. Key Claims About Prior Work — Accuracy Audit

### Claim 1 (Introduction, p.1):
> "Prior work characterizes how conformal prediction degrades under shift [Tibshirani 2019, Barber 2023], a critical gap remains: Can we identify which deployed models will fail before observing test data?"

**Assessment:** Accurate. Neither Tibshirani et al. (2019) nor Barber et al. (2023) provide pre-deployment model-specific failure prediction. Tibshirani (2019) assumes known propensity scores; Barber (2023) provides theoretical TV-distance bounds. Neither offers a practical pre-deployment diagnostic for practitioner use. The claim is well-supported.

### Claim 2 (Related Work, Section 2):
> "Garg et al. (2022) use unlabeled test data to predict accuracy degradation—a related goal but requiring test-time observations."

**Assessment:** Accurate. Garg et al. (2022) "Leveraging Unlabeled Data to Predict Out-of-Distribution Performance" uses unlabeled test data (the AC/DC method). The paper's diagnostic uses only validation data, making it genuinely pre-deployment. The differentiation is valid.

### Claim 3 (Section 6.3):
> "MMD, C2ST, and PSI... All detect shift on all 8 SALT tasks... but none predict severity."

**Assessment:** Empirically verified in the paper (MMD p<0.002, C2ST >99.9% accuracy, PSI >0.002 for all 8 tasks; all severity correlations |ρ| ≤ 0.19). This is a sound empirical result, not a claim about prior theory.

### Claim 4 (Abstract):
> "Standard shift detectors... detect shift for all tasks uniformly but do not distinguish catastrophic from robust outcomes (ρ ≤ 0.19, all p > 0.6)"

**Assessment:** Internally consistent with the experimental results in Section 6.3. The claim is accurate as stated.

---

## 7. Publication Venue Error

**Critical issue: `angelopoulos2024conformal` has wrong venue.**

The BibTeX entry reads:
```bibtex
@article{angelopoulos2024conformal,
  title={Conformal Risk Control},
  author={Angelopoulos, Anastasios N and Bates, Stephen and Fisch, Adam and Lei, Lihua and Schuster, Tal},
  journal={Journal of the American Statistical Association},
  year={2024}
}
```

The actual publication venue for "Conformal Risk Control" (arXiv:2208.02814) is **ICLR 2024**, not JASA. The JASA-adjacent work by this group is "Learn then Test" (Angelopoulos et al., arXiv:2110.01052), which is a different paper.

**Action required:** Correct the venue:
```bibtex
@inproceedings{angelopoulos2024conformal,
  title={Conformal Risk Control},
  author={Angelopoulos, Anastasios N and Bates, Stephen and Fisch, Adam and Lei, Lihua and Schuster, Tal},
  booktitle={International Conference on Learning Representations},
  year={2024}
}
```

---

## 8. Summary of Required Actions

### Must Fix (Errors)
1. **`angelopoulos2024conformal` venue**: Change from `journal={Journal of the American Statistical Association}` to `booktitle={International Conference on Learning Representations}`. "Conformal Risk Control" is an ICLR 2024 paper, not JASA.

### Strongly Recommended (Missing Key Citations)
2. **Split CP foundational citation**: Add Lei et al. (2018), "Distribution-Free Predictive Inference for Regression," JASA — the canonical split CP reference, given the paper uses split conformal prediction.
3. **External dataset citations**: Add at minimum a UCI ML Repository citation (`Dua & Graff 2017`) for the external validation datasets (Covertype, KDDCup99, Gas Sensor, Shuttle, Avila, PAMAP2, Pendigits, Satimage, Pendigits).
4. **RelBench arXiv ID**: Populate the missing arXiv ID in `fey2023relbench` (arXiv:2407.20060).

### Recommended (Strengthening)
5. **`angelopoulos2021gentle` venue**: Update to the published Foundations & Trends 2023 version — currently lists only the arXiv preprint.
6. **Gibbs & Candès (2022/2023) JMLR extension**: Add the ACI extension paper ("Conformal Inference for Online Prediction with Arbitrary Distribution Shifts," JMLR 25, 2024) when discussing ACI in Section 6.1.
7. **PSI provenance**: Add a footnote or parenthetical acknowledging PSI is a practitioner heuristic from credit scoring without a single canonical academic reference.

### Optional (Completeness)
8. **Shafer & Vovk (2008)**: Add the JMLR tutorial as an additional foundational CP reference.
9. **Quiñonero-Candela et al. (2009)**: Add the "Dataset Shift in Machine Learning" book for the dataset shift taxonomy.
10. **Bhatnagar et al. (2023) ICML**: Consider citing alongside Zaffran (2022) and Feldman (2023) when discussing ACI extensions.

---

## 9. Overall Literature Review Assessment

### Differentiation
The paper's differentiation claim — that prior CP-under-shift work provides theoretical coverage bounds but not pre-deployment model-specific failure prediction — is **accurate and well-supported**. The Tibshirani (2019), Podkopaev (2021), and Barber (2023) papers are correctly characterized. The Garg et al. (2022) comparison is precise. The experimental evidence from MMD/C2ST/PSI uniformly detecting shift without severity discrimination is compelling.

**Rating: Strong.** The differentiation is genuine and verifiable.

### Coverage of Recent Literature (2023-2026)
The paper's related work covers literature through early 2024. Key extensions from 2023-2024 that are missing but not disqualifying:
- No paper has proposed a pre-deployment feature-importance-based diagnostic for CP failure — the paper's contribution appears novel in this respect.
- Several papers on online CP (Bhatnagar 2023, Gibbs & Candès 2022) extend the adaptive methods the paper cites; their omission does not weaken the core contribution.
- The Conformal Risk Control venue error (ICLR vs JASA) is the most important correction needed.

**Rating: Adequate, with the venue error requiring correction.**

### Citation Quality
- 20 total references; relatively lean for a UAI paper with this scope.
- All cited papers are accurately described in the related work.
- One confirmed venue error (`angelopoulos2024conformal`).
- Missing: split CP foundational citation (Lei 2018), external dataset citations, arXiv ID for RelBench.
- No missing seminal CP papers that would raise a red flag with reviewers (Vovk 2005, Romano 2020, Tibshirani 2019, Barber 2023, Gibbs 2021, Angelopoulos 2021 RAPS are all present).

**Rating: Good, with targeted gaps to address.**

---

*End of literature review audit.*
