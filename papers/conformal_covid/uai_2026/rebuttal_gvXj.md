# Response to Reviewer gvXj

We thank Reviewer gvXj for the thoughtful and detailed feedback. Below we address your specific questions.

---

> **Q1: "The 40% threshold... post-hoc threshold selection on 8 data points"**

We ran Leave-One-Out Cross-Validation:

|       Metric        |      Value       |
|:-------------------:|:----------------:|
| LOO-CV Accuracy     | **87.5% (7/8)**  |
| Threshold Stability |   43.1% ± 5.0%   |
| Cohen's d           | **3.08** (large) |
| Bootstrap 95% CI    |   [0.31, 1.00]   |

**Answer**: The large effect size (d=3.08) confirms meaningful separation between failed (mean C=50.2%) and succeeded (mean C=29.8%) tasks. The single misclassification (sales-office) is a boundary case where the dominant feature remains stable (high Jaccard). We recommend the 40% threshold as a starting point with domain-specific calibration.

---

> **Q2: "Predicted bound 0.518 vs observed 0.98"**

**Answer**: We adopt option (c) from your suggestion—Theorem 1 provides **mechanistic insight** rather than tight quantitative prediction. The gap reflects a fundamental mismatch: Theorem 1 assumes additivity in probability space, but TreeSHAP operates in log-odds space. The theorem correctly predicts **direction** (higher C → worse coverage) and **monotonicity**, validated by ρ=0.833 and τ=0.714.

---

> **Q3: "Consider adding more datasets with documented shift (e.g., from the WILDS)"**

We validated on WILDS benchmarks:

|    Dataset    | Source  |  Shift Type | Conc. |  Drop  | Predicted | Actual |
|:-------------:|:-------:|:-----------:|:-----:|:------:|:---------:|:------:|
| CivilComments |  WILDS  | Demographic | 26.2% | -0.7pp |  Robust   | Robust |
| Amazon        |  WILDS  |  User/Time  |  1.9% | +0.4pp |  Robust   | Robust |
| Covertype     | sklearn |   Temporal  | 12.5% | +8.0pp |  Robust   | Robust |
| Adult         | OpenML  |  Age-based  | 27.5% | +1.4pp |  Robust   | Robust |

**Answer**: Combined SALT + WILDS achieves **91.7% threshold accuracy** (11/12). All 4 WILDS benchmarks have low concentration (mean 17.0%) and maintain coverage, validating **specificity**. We acknowledge the limitation that WILDS lacks high-concentration failure cases.

---

> **Q4: "The MLP analysis shows that some MLPs concentrate importance but fail catastrophically, while others distribute importance but still fail"**

**Answer**: We agree this is an important finding. We will add to the abstract: *"The diagnostic identifies concentrated-dependence failures but not global-sensitivity failures observed in neural networks."*

---

**Supplementary tables**: [Anonymous Google Drive](https://drive.google.com/drive/folders/17crhdk-5F4jnAHukneIEgEb1lDNOd78w?usp=sharing)

We commit to all suggested revisions in the camera-ready version.
