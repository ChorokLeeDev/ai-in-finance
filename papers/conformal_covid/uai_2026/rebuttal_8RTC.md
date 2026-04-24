# Response to Reviewer 8RTC

We thank Reviewer 8RTC for the thoughtful feedback. Below we address your specific questions.

---

> **Q1: "The type of this temporal shift is still ambiguous: is it covariate shift, concept shift, or label shift?"**

We ran empirical shift characterization using KS tests, JS divergence, and accuracy drop:

|      Task       | Cov KS | %Sig | Label JS | Acc Drop |  Classification |
|:---------------:|:------:|:----:|:--------:|:--------:|:---------------:|
| sales-shipcond  |  0.50  | 50%  |  0.083   |  +11.9%  |  Cov + Concept  |
| sales-group     |  0.50  | 50%  |  0.330   |   +1.0%  | Cov (dominant)  |
| item-incoterms  |  0.86  | 86%  |  0.140   |  +63.4%  |  Cov + Concept  |
| sales-office    |  0.50  | 50%  |  0.026   |   -0.2%  | Cov (dominant)  |

**Answer**: **Covariate shift** confirmed in ALL tasks (50-86% features significant, Bonferroni-corrected). **Concept shift** varies by task. Catastrophic failures require BOTH covariate AND concept shift. The robust task (sales-office) shows covariate shift but no concept shift.

---

> **Q2: "The title is a little bit broad"**

**Answer**: We agree. We will revise the title to: **"Diagnosing Conformal Prediction Failures in Gradient-Boosted Models Under Distribution Shift."**

---

> **Q3: "Does it generalize to other data or tasks?"**

We validated on WILDS benchmarks with real distribution shifts:

|    Dataset    | Source  |  Shift Type | Conc. |  Drop  | Predicted | Actual |
|:-------------:|:-------:|:-----------:|:-----:|:------:|:---------:|:------:|
| CivilComments |  WILDS  | Demographic | 26.2% | -0.7pp |  Robust   | Robust |
| Amazon        |  WILDS  |  User/Time  |  1.9% | +0.4pp |  Robust   | Robust |
| Covertype     | sklearn |   Temporal  | 12.5% | +8.0pp |  Robust   | Robust |
| Adult         | OpenML  |  Age-based  | 27.5% | +1.4pp |  Robust   | Robust |

**Answer**: Combined SALT + WILDS achieves **91.7% threshold accuracy** (11/12). SALT provides sensitivity (high-C failures), WILDS provides specificity (low-C robust cases).

---

**Supplementary tables**: [Anonymous Google Drive](https://drive.google.com/drive/folders/17crhdk-5F4jnAHukneIEgEb1lDNOd78w?usp=sharing)

We commit to all suggested revisions in the camera-ready version.
