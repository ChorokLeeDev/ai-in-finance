# Response to Reviewer TFmu

We thank Reviewer TFmu for the thoughtful feedback. Below we address your specific questions.

---

> **Q1: "What happens if the concentration metric is defined using the top 2, 3, or 5 features?"**

We ran ablations with k ∈ {1, 2, 3, 5, 10} plus HHI, Gini, and entropy-based measures:

|   Metric   | Spearman ρ | p-value | Significant? |
|:----------:|:----------:|:-------:|:------------:|
| **Top-1**  | **0.833**  |**0.010**|   **Yes**    |
| Top-2      |   0.524    |  0.183  |     No       |
| Top-3      |   0.524    |  0.183  |     No       |
| Top-5      |   0.690    |  0.058  |     No       |
| HHI        |   0.619    |  0.102  |     No       |
| Gini       |   0.500    |  0.207  |     No       |

**Answer**: Top-1 is the **only** statistically significant metric. Adding features 2-5 *decreases* correlation (0.833 → 0.524) because 2nd/3rd features capture stable relationships, not the shifting vulnerability source. This is not ad hoc—it reflects GBM's winner-take-all dynamics where a single dominant feature drives predictions.

---

> **Q2: "The method may only hold for certain model types"**

We formalize when the diagnostic applies vs. fails:

- **Concentrated-dependence failure** (C > 40%): Diagnostic applies ✓
- **Global-sensitivity failure** (C ≤ 40%, distributed importance): Diagnostic does NOT apply

|     Task      | MLP C | MLP Drop | LGB C | LGB Drop |  MLP Mode  |
|:-------------:|:-----:|:--------:|:-----:|:--------:|:----------:|
| s-group       | 27.5% |  78.4pp  | 47.3% |  71.2pp  | **Global** |
| s-payterms    | 28.0% |  60.6pp  | 54.2% |  77.1pp  | **Global** |
| i-shippoint   | 77.2% |  82.3pp  | 48.8% |  18.5pp  |    Conc    |

**Answer**: LightGBM ρ=0.833 (p=0.010), MLP ρ=0.43 (n.s.). The diagnostic is specific to gradient-boosted models. For MLPs with C ≤ 40%, we recommend gradient-based sensitivity analysis instead.

---

**Supplementary tables**: [Anonymous Google Drive](https://drive.google.com/drive/folders/17crhdk-5F4jnAHukneIEgEb1lDNOd78w?usp=sharing)

We commit to all suggested revisions in the camera-ready version.
