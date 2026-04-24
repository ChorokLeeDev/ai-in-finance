# Response to Reviewer 1Lb4

We thank Reviewer 1Lb4 for the positive evaluation and constructive feedback. Below we address your specific questions.

---

> **Q1: "Feature importance is known to be highly sensitive to hyperparameters... The experiments should include a sensitivity analysis"**

We ran full retraining on REAL SALT data with 4 HP configurations:

|  Config | leaves |  lr  | n_est |   ρ   | p-value | Thresh |  F1 |
|:-------:|:------:|:----:|:-----:|:-----:|:-------:|:------:|:---:|
| Default |   31   | 0.05 |  100  | 0.833 |  0.010  | 45.0%  | 1.0 |
| Deeper  |   63   | 0.05 |  100  | 0.810 |  0.015  | 47.5%  | 1.0 |
| Faster  |   31   | 0.10 |  100  | 0.810 |  0.015  | 45.0%  | 1.0 |
| More    |   31   | 0.05 |  200  | 0.833 |  0.010  | 45.0%  | 1.0 |

**Answer**: Correlation ρ = 0.821 ± 0.012, threshold = 45.6% ± 1.1%, **all 4 configs significant** (p<0.05). The diagnostic is robust to HP variations. Rank order of tasks by concentration is preserved across all configurations.

---

> **Q2: "The proposed method is restricted to Adaptive Prediction Sets (APS)"**

**Answer**: ACI (Section 6.1) addresses adaptive methods. The diagnostic targets the base model's feature dependence, which determines vulnerability regardless of conformal wrapper. The conformal method (APS, RAPS, ACI) affects how coverage degrades, but concentration predicts *whether* it will degrade.

---

> **Q3: "Figure font size"**

**Answer**: Will increase in camera-ready.

---

**Supplementary tables**: [Anonymous Google Drive](https://drive.google.com/drive/folders/17crhdk-5F4jnAHukneIEgEb1lDNOd78w?usp=sharing)

We commit to all suggested revisions in the camera-ready version.
