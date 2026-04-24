# Table R4: External Validation - WILDS Benchmark Datasets

**Question**: Does the diagnostic generalize beyond SALT to established benchmarks?

## Results (4 benchmark datasets)

|    Dataset    | Source  |  Shift Type | Conc. |  Drop  |  Pred  | Actual | OK? |
|:-------------:|:-------:|:-----------:|:-----:|:------:|:------:|:------:|:---:|
| CivilComments |  WILDS  | Demographic | 26.2% | -0.7pp | Robust | Robust |  ✓  |
| Amazon        |  WILDS  |  User/Time  |  1.9% | +0.4pp | Robust | Robust |  ✓  |
| Covertype     | sklearn |   Temporal  | 12.5% | +8.0pp | Robust | Robust |  ✓  |
| Adult         | OpenML  |  Age-based  | 27.5% | +1.4pp | Robust | Robust |  ✓  |

**Threshold accuracy: 100% (4/4)**

## Dataset Details

### CivilComments (WILDS)
- **Task**: Toxicity classification
- **Shift**: Demographic groups (identity mentions)
- **Features**: TF-IDF (1000 features)
- **Result**: Low concentration (26.2%), maintained coverage

### Amazon Reviews (WILDS)
- **Task**: Product rating prediction
- **Shift**: User/time domain
- **Features**: TF-IDF (1000 features)
- **Result**: Very low concentration (1.9%), robust

### Covertype (sklearn)
- **Task**: Forest cover type classification
- **Shift**: Temporal/regional (Wilderness Area 1 vs others)
- **Features**: 54 cartographic features
- **Result**: Low concentration (12.5%), mild degradation

### Adult Income (OpenML)
- **Task**: Income prediction
- **Shift**: Age-based (train: age<50, test: age≥50)
- **Features**: 14 demographic/employment features
- **Result**: Low concentration (27.5%), maintained coverage

## Interpretation

All 4 benchmarks have **low SHAP concentration** (mean = 17.0%, all < 40%).

The diagnostic correctly predicts robustness for all → validates **specificity**.

## Combined Validation (SALT + WILDS, n=12)

|        Metric       |       Value       |
|:-------------------:|:-----------------:|
| Total tasks         |        12         |
| Threshold accuracy  | **91.7% (11/12)** |
| Sensitivity (SALT)  |    100% (4/4)     |
| Specificity (WILDS) |   100% (0/4 FP)   |
| Misclassification   |   sales-office    |
