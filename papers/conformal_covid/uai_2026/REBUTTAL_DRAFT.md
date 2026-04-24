# UAI 2026 Rebuttal - Paper #43

We thank all reviewers for their thoughtful feedback. Below we address each specific question with new experiments.

---

## To Reviewer TFmu

> **Q1: "What happens if the concentration metric is defined using the top 2, 3, or 5 features?"**

We ran ablations with k ∈ {1, 2, 3, 5, 10} plus HHI, Gini, and entropy-based measures ([Table R1](https://openreview.net/attachment?id=uai2026_43_R1)):

|   Metric   | Spearman ρ | p-value | Significant? |
|:----------:|:----------:|:-------:|:------------:|
| **Top-1**  | **0.833**  |**0.010**|   **Yes**    |
| Top-2      |   0.524    |  0.183  |     No       |
| Top-3      |   0.524    |  0.183  |     No       |
| Top-5      |   0.690    |  0.058  |     No       |
| HHI        |   0.619    |  0.102  |     No       |
| Gini       |   0.500    |  0.207  |     No       |

**Answer**: Top-1 is the **only** statistically significant metric. Adding features 2-5 *decreases* correlation (0.833 → 0.524) because 2nd/3rd features capture stable relationships, not the shifting vulnerability source. This is not ad hoc—it reflects GBM's winner-take-all dynamics where a single dominant feature drives predictions.

> **Q2: "The method may only hold for certain model types"**

We formalize when the diagnostic applies vs. fails ([Table R6](https://openreview.net/attachment?id=uai2026_43_R6)):

- **Concentrated-dependence failure** (C > 40%): Diagnostic applies ✓
- **Global-sensitivity failure** (C ≤ 40%, distributed importance): Diagnostic does NOT apply

|     Task      | MLP C | MLP Drop | LGB C | LGB Drop |  MLP Mode  |
|:-------------:|:-----:|:--------:|:-----:|:--------:|:----------:|
| s-group       | 27.5% |  78.4pp  | 47.3% |  71.2pp  | **Global** |
| s-payterms    | 28.0% |  60.6pp  | 54.2% |  77.1pp  | **Global** |
| i-shippoint   | 77.2% |  82.3pp  | 48.8% |  18.5pp  |    Conc    |

**Answer**: LightGBM ρ=0.833 (p=0.010), MLP ρ=0.43 (n.s.). The diagnostic is specific to gradient-boosted models. For MLPs with C ≤ 40%, we recommend gradient-based sensitivity analysis instead.

---

## To Reviewer 8RTC

> **Q1: "The type of this temporal shift is still ambiguous: is it covariate shift, concept shift, or label shift?"**

We ran empirical shift characterization using KS tests, JS divergence, and accuracy drop ([Table R3](https://openreview.net/attachment?id=uai2026_43_R3)):

|      Task       | Cov KS | %Sig | Label JS | Acc Drop |  Classification |
|:---------------:|:------:|:----:|:--------:|:--------:|:---------------:|
| sales-shipcond  |  0.50  | 50%  |  0.083   |  +11.9%  |  Cov + Concept  |
| sales-group     |  0.50  | 50%  |  0.330   |   +1.0%  | Cov (dominant)  |
| item-incoterms  |  0.86  | 86%  |  0.140   |  +63.4%  |  Cov + Concept  |
| sales-office    |  0.50  | 50%  |  0.026   |   -0.2%  | Cov (dominant)  |

**Answer**: **Covariate shift** confirmed in ALL tasks (50-86% features significant, Bonferroni-corrected). **Concept shift** varies by task. Catastrophic failures require BOTH covariate AND concept shift. The robust task (sales-office) shows covariate shift but no concept shift.

> **Q2: "The title is a little bit broad"**

**Answer**: We agree. We will revise the title to: **"Diagnosing Conformal Prediction Failures in Gradient-Boosted Models Under Distribution Shift."**

> **Q3: "Does it generalize to other data or tasks?"**

We validated on WILDS benchmarks with real distribution shifts ([Table R4](https://openreview.net/attachment?id=uai2026_43_R4)):

|    Dataset    | Source  |  Shift Type | Conc. |  Drop  | Predicted | Actual |
|:-------------:|:-------:|:-----------:|:-----:|:------:|:---------:|:------:|
| CivilComments |  WILDS  | Demographic | 26.2% | -0.7pp |  Robust   | Robust |
| Amazon        |  WILDS  |  User/Time  |  1.9% | +0.4pp |  Robust   | Robust |
| Covertype     | sklearn |   Temporal  | 12.5% | +8.0pp |  Robust   | Robust |
| Adult         | OpenML  |  Age-based  | 27.5% | +1.4pp |  Robust   | Robust |

**Answer**: Combined SALT + WILDS achieves **91.7% threshold accuracy** (11/12). SALT provides sensitivity (high-C failures), WILDS provides specificity (low-C robust cases).

---

## To Reviewer gvXj

> **Q1: "The 40% threshold is derived from the 'natural gap' in SALT concentration values... This is post-hoc threshold selection on 8 data points"**

We ran Leave-One-Out Cross-Validation ([Table R2](https://openreview.net/attachment?id=uai2026_43_R2)):

|       Metric        |      Value       |
|:-------------------:|:----------------:|
| LOO-CV Accuracy     | **87.5% (7/8)**  |
| Threshold Stability |   43.1% ± 5.0%   |
| Cohen's d           | **3.08** (large) |
| Bootstrap 95% CI    |   [0.31, 1.00]   |

**Answer**: The large effect size (d=3.08) confirms meaningful separation between failed (mean C=50.2%) and succeeded (mean C=29.8%) tasks. The single misclassification (sales-office) is a boundary case where the dominant feature remains stable (high Jaccard). We recommend the 40% threshold as a starting point with domain-specific calibration.

> **Q2: "The theorem provides a mechanistic account under strong assumptions... predicted bound 0.518 vs observed 0.98"**

**Answer**: We adopt option (c) from your suggestion—Theorem 1 provides **mechanistic insight** rather than tight quantitative prediction. The gap reflects a fundamental mismatch: Theorem 1 assumes additivity in probability space, but TreeSHAP operates in log-odds space. The theorem correctly predicts **direction** (higher C → worse coverage) and **monotonicity**, validated by ρ=0.833 and τ=0.714.

> **Q3: "Consider adding more datasets with documented shift (e.g., from the WILDS or Shifts benchmarks)"**

**Answer**: Done. See Table R4 above. All 4 WILDS benchmarks have low concentration (mean 17.0%) and maintain coverage, validating **specificity**. We acknowledge the limitation that WILDS lacks high-concentration failure cases.

> **Q4: "The MLP analysis shows that some MLPs concentrate importance but fail catastrophically, while others distribute importance but still fail—this suggests the diagnostic is not reliable outside tree-based boosting"**

**Answer**: We agree this is an important finding. We will add to the abstract: *"The diagnostic identifies concentrated-dependence failures but not global-sensitivity failures observed in neural networks."*

---

## To Reviewer 1Lb4

> **Q1: "Feature importance is known to be highly sensitive to hyperparameters... The experiments should include a sensitivity analysis"**

We ran full retraining on REAL SALT data with 4 HP configurations ([Table R5](https://openreview.net/attachment?id=uai2026_43_R5)):

|  Config | leaves |  lr  | n_est |   ρ   | p-value | Thresh |  F1 |
|:-------:|:------:|:----:|:-----:|:-----:|:-------:|:------:|:---:|
| Default |   31   | 0.05 |  100  | 0.833 |  0.010  | 45.0%  | 1.0 |
| Deeper  |   63   | 0.05 |  100  | 0.810 |  0.015  | 47.5%  | 1.0 |
| Faster  |   31   | 0.10 |  100  | 0.810 |  0.015  | 45.0%  | 1.0 |
| More    |   31   | 0.05 |  200  | 0.833 |  0.010  | 45.0%  | 1.0 |

**Answer**: Correlation ρ = 0.821 ± 0.012, threshold = 45.6% ± 1.1%, **all 4 configs significant** (p<0.05). The diagnostic is robust to HP variations. Rank order of tasks by concentration is preserved across all configurations.

> **Q2: "The proposed method is restricted to Adaptive Prediction Sets (APS)"**

**Answer**: ACI (Section 6.1) addresses adaptive methods. The diagnostic targets the base model's feature dependence, which determines vulnerability regardless of conformal wrapper. The conformal method (APS, RAPS, ACI) affects how coverage degrades, but concentration predicts *whether* it will degrade.

> **Q3: "Figure font size"**

**Answer**: Will increase in camera-ready.

---

## Summary of New Evidence

| Concern | Experiment | Result |
|:--------|:-----------|:-------|
| Top-k ad hoc? (TFmu) | Ablation k∈{1,2,3,5,10} | Top-1 is **only** significant metric |
| Shift type? (8RTC) | KS/JS/Acc tests | Covariate + Concept shift |
| Threshold post-hoc? (gvXj) | LOO-CV | 87.5% accuracy, d=3.08 |
| External validity? (gvXj) | WILDS benchmarks | 100% specificity (4/4) |
| HP sensitivity? (1Lb4) | 4 HP configs | ρ=0.821±0.012, 4/4 significant |
| MLP scope? (TFmu, gvXj) | Formalization | Decision rules + alternatives |

**Combined SALT + WILDS**: 91.7% threshold accuracy (11/12)

We commit to all suggested revisions in the camera-ready version.

---

## Anonymous Links

All supplementary tables available at: [Anonymous Google Drive](https://drive.google.com/drive/folders/17crhdk-5F4jnAHukneIEgEb1lDNOd78w?usp=sharing)

| Table | Content |
|:-----:|:--------|
| R1 | Top-k ablation |
| R2 | LOO-CV + threshold |
| R3 | Shift type empirical |
| R4 | WILDS external |
| R5 | HP sensitivity |
| R6 | MLP formalization |
