We thank Reviewer gvXj for the detailed and constructive feedback.

**Q1: "40% threshold... post-hoc threshold selection on 8 data points"**

We ran Leave-One-Out Cross-Validation:

| Metric | Value |
|:--|:--|
| LOO-CV Accuracy | **87.5% (7/8)** |
| Threshold Stability | 43.1% ± 5.0% |
| Cohen's d | **3.08** (large effect) |
| Bootstrap 95% CI | [0.31, 1.00] |

The large effect size confirms meaningful separation between failed (mean C=50.2%) and succeeded (mean C=29.8%) tasks. The single misclassification is sales-office—a boundary case where the dominant feature remains stable across train/test (Jaccard=0.67, importance=18.2%). This "protective factor" pattern explains preserved coverage despite C>40%. We recommend 40% as an exploratory threshold with domain-specific calibration.

**Q2: "Predicted bound 0.518 vs observed 0.98... theorem uninformative"**

We adopt option (c)—Theorem 1 provides **mechanistic insight** rather than tight quantitative prediction. The gap reflects a fundamental mismatch: Theorem 1 assumes additivity in probability space, but TreeSHAP operates in log-odds space. The theorem correctly predicts **direction** (higher C → worse coverage) and **monotonicity**, validated by ρ=0.833 (p=0.010) and Kendall's τ=0.714.

**Q3: "Consider adding WILDS or Shifts benchmarks"**

Done. We validated on 4 benchmarks with documented distribution shifts:

| Dataset | Shift Type | Conc. | Drop | Predicted | Actual |
|:--|:--|:--|:--|:--|:--|
| CivilComments | Demographic | 26.2% | -0.7pp | Robust | Robust |
| Amazon | User/Time | 1.9% | +0.4pp | Robust | Robust |
| Covertype | Temporal | 12.5% | +8.0pp | Robust | Robust |
| Adult | Age-based | 27.5% | +1.4pp | Robust | Robust |

Combined SALT+WILDS: **91.7% accuracy** (11/12). WILDS validates **specificity**—all low concentration and maintain coverage. SALT provides sensitivity, WILDS provides specificity.

**Q4: "MLP analysis... diagnostic not reliable outside tree-based"**

Agreed—this is a critical scope limitation. Abstract revision: *"The diagnostic identifies concentrated-dependence failures in gradient-boosted models but not global-sensitivity failures in neural networks."* For low-C MLPs with catastrophic failure, we recommend gradient-based sensitivity analysis.

**Supplementary tables R1-R6**: [Anonymous Link](https://drive.google.com/drive/folders/17crhdk-5F4jnAHukneIEgEb1lDNOd78w)
