# Table R2: Leave-One-Out Cross-Validation and Threshold Stability

**Question**: Is the 40% threshold post-hoc selected, or does it generalize?

## LOO-CV Results (SALT Dataset, n=8)

|        Metric       |      Value       |
|:-------------------:|:----------------:|
| LOO-CV Accuracy     | **87.5% (7/8)**  |
| Threshold Stability |   43.1% ± 5.0%   |
| Cohen's d           | **3.08** (large) |
| Bootstrap 95% CI    |   [0.31, 1.00]   |

## Per-Fold Predictions

|      Task       |  Conc. |  Drop  | Thresh |  Pred  | Actual | OK? |
|:---------------:|:------:|:------:|:------:|:------:|:------:|:---:|
| sales-shipcond  | 50.7%  | 71.6pp |  45%   |  Vuln  | Failed |  ✓  |
| sales-group     | 47.3%  | 71.2pp |  45%   |  Vuln  | Failed |  ✓  |
| sales-payterms  | 54.2%  | 77.1pp |  45%   |  Vuln  | Failed |  ✓  |
| item-plant      | 23.9%  | 10.6pp |  45%   | Robust | Robust |  ✓  |
| item-shippoint  | 48.8%  | 18.5pp |  45%   |  Vuln  | Failed |  ✓  |
| sales-incoterms | 23.7%  |  8.5pp |  45%   | Robust | Robust |  ✓  |
| item-incoterms  | 28.9%  | 11.3pp |  45%   | Robust | Robust |  ✓  |
| **sales-office**|**42.6%**|**0.1pp**|**30%**|**Vuln**|**Robust**|**✗**|

## Misclassification Analysis

The single misclassification (**sales-office**) is a meaningful boundary case:
- High concentration (42.6%) BUT maintained coverage (0.1pp drop)
- **Protective factor**: High Jaccard overlap (dominant feature stable across train/test)
- When the dominant feature does NOT shift, high concentration does not imply vulnerability

## Effect Size Analysis

|      Group     | Mean Conc. |  Std  |
|:--------------:|:----------:|:-----:|
| Failed (n=4)   |   50.2%    | 2.6%  |
| Succeeded (n=4)|   29.8%    | 7.7%  |
| **Separation** | **20.4pp** |   -   |
| **Cohen's d**  |  **3.08**  | large |

## Threshold Sensitivity

| Threshold | Precision | Recall |   F1   | Accuracy |
|:---------:|:---------:|:------:|:------:|:--------:|
|    30%    |   0.80    |  1.00  |  0.89  |  87.5%   |
|    35%    |   0.80    |  1.00  |  0.89  |  87.5%   |
|  **40%**  | **0.80**  |**1.00**|**0.89**|**87.5%** |
|    45%    |   1.00    |  1.00  |  1.00  |  100%    |
|    50%    |   1.00    |  0.50  |  0.67  |  75.0%   |

**Recommendation**: 40% as starting point; domain-specific calibration recommended.
