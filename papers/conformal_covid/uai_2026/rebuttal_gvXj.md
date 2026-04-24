We thank Reviewer gvXj for the detailed and constructive feedback. Below we address each concern with new experiments.

**Q1: "40% threshold... post-hoc threshold selection on 8 data points"**

We ran Leave-One-Out Cross-Validation to address this concern rigorously:

| Metric | Value |
|:--|:--|
| LOO-CV Accuracy | **87.5% (7/8)** |
| Threshold Stability | 43.1% ± 5.0% |
| Cohen's d | **3.08** (large effect) |
| Bootstrap 95% CI | [0.31, 1.00] |

The large effect size (d=3.08) confirms meaningful separation between failed tasks (mean C=50.2%) and succeeded tasks (mean C=29.8%). The single misclassification is sales-office, a boundary case where the dominant feature (office_code) remains stable across train/test (Jaccard=0.67, importance=18.2%). This "protective factor" pattern—high concentration but stable feature—explains why coverage is preserved despite C>40%. We recommend 40% as an exploratory threshold with domain-specific calibration based on feature stability analysis.

**Q2: "Predicted bound 0.518 vs observed 0.98... theorem uninformative"**

We adopt option (c) from your suggestions—Theorem 1 provides **mechanistic insight** rather than tight quantitative prediction. The gap reflects a fundamental mismatch: Theorem 1 assumes additivity in probability space, but TreeSHAP operates in log-odds space for classification. The theorem correctly predicts **direction** (higher C → worse coverage) and **monotonicity**, empirically validated by ρ=0.833 (p=0.010) and Kendall's τ=0.714. We will clarify this positioning in the camera-ready to avoid overclaiming.

**Q3: "Consider adding more datasets from WILDS or Shifts benchmarks"**

Done. We validated on 4 WILDS/external benchmarks with documented distribution shifts:

| Dataset | Source | Shift Type | Conc. | Drop | Predicted | Actual |
|:--|:--|:--|:--|:--|:--|:--|
| CivilComments | WILDS | Demographic | 26.2% | -0.7pp | Robust | Robust |
| Amazon | WILDS | User/Time | 1.9% | +0.4pp | Robust | Robust |
| Covertype | sklearn | Temporal | 12.5% | +8.0pp | Robust | Robust |
| Adult | OpenML | Age-based | 27.5% | +1.4pp | Robust | Robust |

Combined SALT+WILDS achieves **91.7% threshold accuracy** (11/12 correct). WILDS benchmarks validate **specificity**—all have low concentration (mean 17.0%) and maintain coverage under shift. We acknowledge the limitation that WILDS lacks high-concentration catastrophic cases; SALT provides sensitivity while WILDS provides specificity.

**Q4: "MLP analysis... diagnostic not reliable outside tree-based boosting"**

We fully agree this is a critical scope limitation. We will add to the abstract: *"The diagnostic identifies concentrated-dependence failures in gradient-boosted models but not global-sensitivity failures observed in neural networks."* For MLPs showing low C but catastrophic failure (e.g., s-group: C=27.5%, drop=78.4pp), we recommend gradient-based sensitivity analysis as a complementary diagnostic.

**Supplementary tables R1-R6**: [Anonymous Google Drive](https://drive.google.com/drive/folders/17crhdk-5F4jnAHukneIEgEb1lDNOd78w)

We commit to all suggested revisions in the camera-ready version.
