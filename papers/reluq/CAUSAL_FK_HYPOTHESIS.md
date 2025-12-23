# Causal FK Hypothesis: A Theoretical Framework for Relational Uncertainty

**Date:** 2025-12-24
**Status:** Hypothesis formulation + Experimental design

---

## Core Hypothesis

> **"인과적 FK는 epistemic uncertainty를 증가시키고, 상관적 FK는 예측을 안정화시킨다"**
>
> "Causal FKs increase epistemic uncertainty, while correlational FKs stabilize predictions"

---

## 1. Definitions

### 1.1 Epistemic Uncertainty
- **정의**: 지식/데이터 부족으로 인한 불확실성
- **특성**: 데이터를 더 모으면 줄어들 수 있음
- **측정**: Ensemble variance, MC Dropout, etc.

### 1.2 Causal FK (인과적 FK)
- **정의**: Target 변수의 값을 직접적으로 결정하는 데이터를 가리키는 FK
- **특성**: FK가 가리키는 테이블의 데이터가 변하면 target도 변함
- **예시**:
  - F1: RESULTS → races (레이스 결과가 순위를 직접 결정)
  - Trial: interventions → studies (약물이 임상시험 결과를 직접 결정)

### 1.3 Correlational FK (상관적 FK)
- **정의**: Target 변수와 상관관계는 있지만 직접적 인과관계가 없는 데이터를 가리키는 FK
- **특성**: FK가 가리키는 테이블의 데이터가 변해도 target이 반드시 변하지 않음
- **예시**:
  - F1: QUALIFYING → races (예선 결과는 본선 순위와 상관있지만 원인 아님)
  - Trial: facilities → studies (병원 위치가 약효를 결정하지 않음)

---

## 2. Why This Hypothesis Makes Sense

### 2.1 인과 메커니즘의 복잡성

**인과적 FK가 가리키는 관계는 본질적으로 복잡하다:**

```
F1 RESULTS 예시:
레이스 결과 → 순위 (인과관계)

하지만 이 메커니즘은:
- 날씨에 따라 완전히 달라짐
- 사고 발생 시 달라짐
- 전략 선택에 따라 달라짐
- 기계 고장 시 달라짐
- 다른 드라이버들의 행동에 따라 달라짐
```

**결과:**
- 모델은 복잡한 인과 메커니즘을 학습해야 함
- 모든 edge case에 대한 데이터가 없음
- → High epistemic uncertainty

### 2.2 상관 패턴의 단순성

**상관적 FK가 가리키는 관계는 단순하고 안정적이다:**

```
F1 QUALIFYING 예시:
예선 기록 ↔ 본선 순위 (상관관계)

이 패턴은:
- "예선 잘하면 본선도 잘할 것" (단순한 휴리스틱)
- 대부분의 경우 맞음
- 복잡한 메커니즘 없이 학습 가능
```

**결과:**
- 모델은 단순한 correlation을 학습
- 정규화(regularization) 효과
- → Stabilizing effect (낮은 uncertainty)

### 2.3 정보 이론적 관점

| FK 유형 | Mutual Information | Conditional Entropy | 결과 |
|--------|-------------------|---------------------|------|
| Causal | I(FK; target) 높음 | H(target \| FK) 높음 (복잡) | 높은 uncertainty |
| Correlational | I(FK; target) 중간 | H(target \| FK) 낮음 (단순) | 안정화 효과 |

### 2.4 날씨 예측 비유

| 정보 | 유형 | 학습 난이도 | Uncertainty |
|------|------|------------|-------------|
| "내일 기압 배치" | 인과적 | 높음 (복잡한 물리학) | 높음 |
| "오늘 날씨" | 상관적 | 낮음 (단순 패턴) | 낮음 |

기압 데이터는 날씨의 원인이지만, 메커니즘이 복잡해서 불확실성이 높음.
오늘 날씨는 원인이 아니지만, 안정적인 예측 패턴을 제공.

---

## 3. Why This Hypothesis Is Worth Validating

### 3.1 기존 관찰 설명

```
F1 실험 결과 (Day 1-3):
- RESULTS FK: 60-120% uncertainty 기여 (인과적)
- QUALIFYING FK: -37% ~ +6% (상관적, 안정화)
- STANDINGS FK: 변동적

→ 가설과 일치! 우연인가, 원리인가?
```

### 3.2 새로운 이론적 기여

| 기존 연구 | 우리 연구 |
|----------|----------|
| "FK로 테이블을 조인한다" | "FK의 의미론(causal vs correlational)이 uncertainty 유형을 결정한다" |
| Technical contribution | Theoretical contribution |
| How to measure | Why it happens |

**아무도 FK의 의미론적 특성과 uncertainty를 연결한 적 없음**

### 3.3 실용적 가치

가설이 맞다면:

```python
# 데이터 품질 투자 전략
if goal == "stable_predictions":
    invest_in(correlational_fks)  # 상관적 FK 품질 개선

if goal == "understand_mechanism":
    invest_in(causal_fks)  # 인과적 FK 데이터 더 수집

if goal == "reduce_uncertainty":
    # 인과적 FK의 edge case 데이터 수집
    collect_data_for(causal_fks, focus="edge_cases")
```

### 3.4 일반화 가능성

| Domain | Causal FK | Correlational FK | Task |
|--------|-----------|------------------|------|
| F1 | RESULTS | QUALIFYING | driver-position |
| ERP (SALT) | SALESDOCUMENT, SOLDTOPARTY | SHIPTOPARTY | item-plant |
| Clinical (Trial) | interventions, conditions | facilities, sponsors | study-outcome |

3개 도메인에서 패턴이 동일하면 → **일반 원리**

### 3.5 학문적 연결

**Causal Inference와의 연결:**

Pearl's Causal Hierarchy:
1. Association (상관관계) - 관찰만으로 학습 가능, 낮은 complexity
2. Intervention (인과관계) - 더 높은 complexity
3. Counterfactual - 가장 높은 complexity

**우리 가설:**
> "FK가 가리키는 관계의 causal level에 따라 학습 난이도(= epistemic uncertainty)가 달라진다"

---

## 4. Experimental Design

### 4.1 Overview

```
Phase 1: Multi-domain validation (manual FK classification)
Phase 2: Quantitative analysis (statistical significance)
Phase 3: Automatic detection exploration
```

### 4.2 Phase 1: Multi-Domain Validation

#### 4.2.1 Dataset & Task Selection

| Dataset | Task | Target | Expected Causal FK | Expected Correlational FK |
|---------|------|--------|-------------------|--------------------------|
| rel-f1 | driver-position | 순위 | RESULTS | QUALIFYING, STANDINGS |
| rel-f1 | driver-dnf | DNF 여부 | RESULTS | QUALIFYING |
| rel-salt | item-plant | 공장 | SALESDOCUMENT, SOLDTOPARTY | SHIPTOPARTY |
| rel-salt | sales-office | 영업소 | SALESDOCUMENT | customer FKs |
| rel-trial | study-outcome | 성공 여부 | interventions, conditions | facilities, sponsors |
| rel-trial | study-adverse | 부작용 | interventions | facilities |

#### 4.2.2 FK Classification Criteria

**Causal FK 판정 기준:**
1. Temporal: FK가 target과 동시에 또는 직전에 발생
2. Mechanistic: FK 데이터가 변하면 target이 직접 변함
3. Domain knowledge: 전문가가 인과관계로 판단

**Correlational FK 판정 기준:**
1. Temporal: FK가 target과 무관한 시점에 발생
2. Mechanistic: FK 데이터가 변해도 target이 반드시 변하지 않음
3. Domain knowledge: 전문가가 상관관계로 판단

#### 4.2.3 Experiment Protocol

```python
for dataset in [rel-f1, rel-salt, rel-trial]:
    for task in dataset.tasks:
        # 1. Train ensemble (5 models, different seeds)
        ensemble = train_ensemble(dataset, task, n_models=5)

        # 2. Compute FK-level uncertainty (permutation-based)
        fk_uncertainties = {}
        for fk in task.foreign_keys:
            fk_uncertainties[fk] = compute_permutation_uncertainty(ensemble, fk)

        # 3. Compute SHAP values for validation
        shap_values = compute_shap(ensemble, task)

        # 4. Record results
        save_results(task, fk_uncertainties, shap_values)
```

### 4.3 Phase 2: Quantitative Analysis

#### 4.3.1 Metrics

| Metric | Formula | Purpose |
|--------|---------|---------|
| Mean Uncertainty Contribution | μ(causal) vs μ(correlational) | 평균 비교 |
| Effect Size (Cohen's d) | (μ₁ - μ₂) / σ_pooled | 효과 크기 |
| Statistical Significance | Mann-Whitney U test | p-value |
| Rank Correlation with SHAP | Spearman ρ | 방법론 검증 |

#### 4.3.2 Expected Results

**If hypothesis is TRUE:**
```
Causal FKs:      μ > 0 (positive uncertainty contribution)
Correlational:   μ ≤ 0 (zero or negative, stabilizing)
Effect size:     d > 0.8 (large effect)
p-value:         p < 0.05 (significant)
```

**If hypothesis is FALSE:**
```
No consistent pattern across domains
Effect size:     d < 0.2 (small/no effect)
p-value:         p > 0.05 (not significant)
```

### 4.4 Phase 3: Automatic Detection

#### 4.4.1 Research Question

> "도메인 지식 없이 FK의 인과/상관 특성을 자동으로 판별할 수 있는가?"

#### 4.4.2 Candidate Signals

| Signal | Hypothesis | Measurement |
|--------|------------|-------------|
| Uncertainty magnitude | Causal FK → higher uncertainty | permutation importance |
| SHAP magnitude | Causal FK → higher SHAP | SHAP values |
| Temporal proximity | Causal FK → closer to target time | time_col analysis |
| Variance across seeds | Causal FK → higher variance | multi-seed std |

#### 4.4.3 Validation Approach

```python
# Train classifier to predict causal vs correlational
# Using only uncertainty/SHAP patterns (no domain knowledge)

features = [
    'uncertainty_contribution',
    'shap_magnitude',
    'temporal_proximity',
    'variance_across_seeds'
]

# Leave-one-domain-out cross-validation
for test_domain in [f1, salt, trial]:
    train_domains = [d for d in all_domains if d != test_domain]

    clf = train_classifier(train_domains, features)
    accuracy = evaluate(clf, test_domain)

    # Can we predict causal vs correlational?
```

### 4.5 Experimental Timeline

| Week | Phase | Tasks |
|------|-------|-------|
| 1 | Setup | Implement multi-domain experiment framework |
| 2 | Phase 1 | Run experiments on F1, SALT, Trial |
| 3 | Phase 2 | Statistical analysis, effect size calculation |
| 4 | Phase 3 | Automatic detection experiments |
| 5-6 | Writing | Document findings, write paper sections |

---

## 5. Potential Challenges & Mitigations

### 5.1 Small Sample Size

**Challenge:** Only 3 domains, ~15 tasks
**Mitigation:**
- Multiple tasks per domain
- Multi-seed validation (5 seeds each)
- Bootstrap confidence intervals

### 5.2 Subjective FK Classification

**Challenge:** Domain knowledge required for causal/correlational labels
**Mitigation:**
- Clear, reproducible criteria
- Document reasoning for each classification
- Sensitivity analysis with borderline cases

### 5.3 Confounding Factors

**Challenge:** FK structure differences (cardinality, data size)
**Mitigation:**
- Control for FK cardinality in analysis
- Normalize by data size
- Report both raw and controlled results

### 5.4 Domain-Specific Effects

**Challenge:** Pattern might be domain-specific, not general
**Mitigation:**
- Test on 3 very different domains (sports, ERP, clinical)
- Meta-analysis across domains
- Report domain-specific and aggregate results

---

## 6. Success Criteria

### 6.1 Minimum Success (Publishable)

- [ ] Causal FKs show higher uncertainty in ≥2/3 domains
- [ ] Effect size d > 0.5 (medium effect)
- [ ] p < 0.05 for aggregate analysis
- [ ] Clear explanation of mechanism

### 6.2 Strong Success (Top Venue)

- [ ] Causal FKs show higher uncertainty in 3/3 domains
- [ ] Effect size d > 0.8 (large effect)
- [ ] p < 0.01 for aggregate analysis
- [ ] Automatic detection accuracy > 70%
- [ ] Theoretical framework connects to causal inference literature

### 6.3 Exceptional Success (Best Paper Candidate)

- [ ] All above criteria met
- [ ] Novel automatic detection method
- [ ] Actionable guidelines for practitioners
- [ ] Extends to 4+ domains

---

## 7. Paper Contribution (If Validated)

### 7.1 Theoretical Contribution

> "We provide the first theoretical framework connecting FK semantics (causal vs correlational) to uncertainty types in relational learning."

### 7.2 Empirical Contribution

> "We validate this framework across 3 diverse domains (sports, enterprise, clinical) with consistent results."

### 7.3 Practical Contribution

> "We provide actionable guidelines: invest in correlational FK data for stable predictions, invest in causal FK data for mechanism understanding."

### 7.4 Methodological Contribution

> "We demonstrate that FK-level uncertainty analysis can automatically identify causal structure in relational data."

---

## 8. Relation to Prior Work

### 8.1 Relational Deep Learning
- Fey et al. (2023): RelBench benchmark
- Our extension: FK semantics matter for uncertainty

### 8.2 Uncertainty Quantification
- Gal & Ghahramani (2016): MC Dropout
- Lakshminarayanan et al. (2017): Deep Ensembles
- Our extension: Decompose by FK, connect to causality

### 8.3 Causal Inference
- Pearl (2009): Causality
- Our extension: Apply causal thinking to relational schemas

### 8.4 Feature Attribution
- Lundberg & Lee (2017): SHAP
- Our extension: FK-level attribution, not feature-level

---

## 9. Next Steps

1. **Implement experiment framework** (Week 1)
   - Multi-domain data loading
   - Unified FK uncertainty computation
   - SHAP integration

2. **Run Phase 1 experiments** (Week 2)
   - F1: 5 tasks
   - SALT: 8 tasks
   - Trial: 5 tasks

3. **Analyze results** (Week 3)
   - Statistical tests
   - Effect sizes
   - Visualizations

4. **Write paper** (Week 4-6)
   - Introduction: The problem
   - Theory: The hypothesis
   - Experiments: The validation
   - Discussion: Implications

---

*Document created: 2025-12-24*
*Status: Ready for experimental validation*
