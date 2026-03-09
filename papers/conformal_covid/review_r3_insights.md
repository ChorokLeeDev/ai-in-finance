# Paper Insight Report

**Paper Title:** Diagnosing Conformal Prediction Failures Under Distribution Shift: A COVID-19 Case Study
**Field / Domain:** Conformal Prediction / Distribution Shift / Uncertainty Quantification
**Venue:** UAI 2026 Submission

---

## Core Contributions (3)

### Contribution 1: SHAP Concentration as Pre-Deployment Diagnostic

SHAP concentration --- top feature의 SHAP importance 비율 --- 이 conformal prediction의 coverage 붕괴 심각도를 배포 전에 예측할 수 있다는 발견. 16개 multiclass task (9개 도메인)에서 Spearman rho=0.853, p<0.001 (bootstrap 95% CI [0.50, 0.96]; Kendall tau=0.667). SALT 단독으로도 rho=0.833, p=0.010 (n=8).

**실증적 증거 강도 평가:**
- (+) 50-seed ensemble로 통계적 안정성 확보. 모든 coverage drop이 paired Wilcoxon p<=0.005로 유의.
- (+) Leave-one-out 안정성: rho in [0.75, 0.96]. 8개 중 6개 jackknife sample에서 유의.
- (+) 외부 검증 9개 도메인 (Covertype 10/10 seeds deterministic catastrophic; 6개 도메인 deterministic robust).
- (-) Bootstrap CI 하한이 0.50으로 넓음. n=16은 correlation 연구로서 소규모.
- (-) Partial correlation 분석에서 class cardinality를 통제하면 non-significant (rho_partial=0.629, p=0.131 at n=8). Cross-domain에서 confound가 약해지긴 하나, 깔끔하게 분리되지 않음.
- (-) 외부 catastrophic evidence가 Covertype 단일 사례에 집중. KDDCup99는 seed-dependent intermediate regime.

**기존 연구 대비 차별점:**
- MMD, C2ST, PSI 같은 기존 shift detector는 shift 존재만 감지 (rho<=0.19, all p>0.6). 이 논문은 "어떤 모델이 실패할 것인가"에 대한 severity prediction이라는 새로운 문제를 정의.
- 기존 conformal prediction under shift 문헌 (Tibshirani 2019, Barber 2023)이 "어떻게 degradation되는가"를 다룬 반면, 이 논문은 "누가 degradation되는가"를 다룸.
- Pre-deployment 조건 충족 --- test data 불필요.

**Rating: 증거 강도 7/10.** Correlation은 강하나 n이 작고 confound 분리가 불완전.

---

### Contribution 2: Score Inflation Theorem (Theorem 1)

Additive feature-decomposition model 하에서 APS conformity score와 coverage bound가 concentration parameter C에 대해 단조 증가/감소함을 증명. 3개 가정 (A1: additive decomposition, A2: concentrated misclassification, A3: residual exchangeability) 하에서 4개 결과 도출: pointwise score bound, expected score inflation, monotone vulnerability, coverage degradation bound.

**실증적 증거 강도 평가:**
- (+) 5개 applicable task 모두에서 conservative bound 검증 통과.
- (+) Catastrophic task의 conformity score CDF가 stochastic dominance 보임 (KS=0.68-0.96, all p<10^-10).
- (+) Catastrophic task에서 entropy 감소 (confident misclassification) --- 직관과 반대되는 현상을 이론이 설명.
- (-) A1 (probability space에서의 additive decomposition)은 실제 tree ensemble의 log-odds space 작동과 괴리. 저자도 "approximation"으로 인정.
- (-) Conservative bound와 observed value 간 gap이 큼 (e.g., 0.518 vs 0.98). Bound의 실용적 tightness가 낮음.
- (-) 본질적으로 sufficient condition이지 necessary condition이 아님 --- 이론이 cover하지 못하는 failure mode 존재 가능.

**기존 연구 대비 차별점:**
- Barber et al. (2023)의 TV distance bound를 feature importance 구조와 연결한 최초 시도.
- "왜" concentration이 failure를 예측하는지에 대한 mechanistic account 제공.

**Rating: 증거 강도 6/10.** 방향성은 맞으나 가정이 강하고 bound가 느슨.

---

### Contribution 3: Operational Decision Framework + Cross-Domain Transfer

40% threshold 기반 3단계 의사결정 프레임워크 제시: (1) SHAP concentration 계산, (2) threshold 분류, (3) protective factor 확인. 9개 외부 도메인에 tuning 없이 적용 시 7/9 deterministic/near-deterministic. 추가로 quarterly retraining이 vulnerable task coverage를 +19pp 개선 (p=0.04, unadjusted).

**실증적 증거 강도 평가:**
- (+) 외부 도메인 7/9 정확 분류. Covertype의 deterministic catastrophic detection (10/10 seeds).
- (+) Threshold sensitivity 분석 제공 (30-50% range, F1=0.50-0.91).
- (+) Retraining, ACI, RAPS 등 mitigation 방법의 mechanism-dependent 효과 분석.
- (-) Threshold가 n=8에서 도출된 exploratory 값. 40% vs 35% vs 45% 선택의 이론적 근거 부족.
- (-) Retraining 효과의 p=0.04는 Holm correction 후 p=0.12로 non-significant.
- (-) KDDCup99 FN 문제 미해결. "Uncertainty band" 제안은 workaround이지 solution이 아님.
- (-) Model-specific (boosting only). RF rho=0.30, MLP rho=0.43 --- 범용성 제한.

**Rating: 증거 강도 5/10.** Exploratory framework. Prospective validation 없음.

---

## Practical Implications

- **배포 전 위험 진단:** Gradient-boosted 모델을 사용하는 팀은 SHAP concentration을 계산하여 conformal prediction set의 shift 취약성을 사전 평가할 수 있다. Validation data만으로 가능하므로 추가 비용이 거의 없다.
- **Shift detection != Severity prediction:** MMD/C2ST만으로 배포 결정을 내리는 팀은 false sense of security에 빠질 수 있다. Shift 감지와 failure severity 예측은 별개 문제임을 인식해야 한다.
- **Mitigation 전략의 mechanism 매칭:** RAPS는 high-cardinality class accumulation에만 효과적 (s-group: -63pp). Concentrated single-feature dependence에는 무력 (s-shipcond: +7pp 악화). Retraining은 moderate-cardinality task에만 도움. 맹목적 mitigation 적용은 비효율적.
- **Model-specificity 인식:** LGB에서 robust한 task가 CatBoost에서 catastrophic할 수 있고 그 반대도 가능 (i-shippoint: LGB 18.5% vs CatBoost 61.8%). 모델별로 독립적 진단 필요.
- **Limitations:** Boosting 모델 전용. Threshold는 provisional. n=16에서의 correlation이 domain 확장 시 유지되는지 미검증.

---

## Future Research Directions

1. **Prospective deployment validation.** 현재 모든 결과가 retrospective. 실제 production 환경에서 SHAP concentration threshold의 prospective 성능 검증 필요.
2. **Neural network 확장.** MLP rho=0.43 non-significant. Neural network의 failure mode는 concentrated dependence가 아닌 다른 메커니즘 (global sensitivity)일 가능성 --- neural net 전용 diagnostic 개발 필요.
3. **Causal identification.** 현재 associative evidence만 존재. Concentration을 실험적으로 조작하여 (e.g., feature knockout, synthetic concentration manipulation) causal effect 검증 가능.
4. **Threshold의 이론적 도출.** 40%는 empirical gap에서 도출. Theorem 1의 bound로부터 task-specific optimal threshold를 이론적으로 도출할 수 있는지 탐구.
5. **Regression task 확장.** 현재 classification (APS) 전용. CQR 등 regression conformal method에서의 analog 존재 여부 연구.
6. **Confound 분리.** Class cardinality와 concentration의 partial correlation이 n=8에서 non-significant. 더 큰 n에서 깔끔한 분리 필요.

---

## UAI 2026 Reviewer 관점 분석

### Reviewer가 가장 높이 평가할 기여

**Contribution 1 (SHAP Concentration Diagnostic).** 이유:
- 문제 정의가 명확하고 실용적. "Shift가 존재하는지"가 아니라 "어떤 모델이 실패하는지"는 실무적으로 훨씬 중요한 질문.
- Negative result (MMD/C2ST 무력함)이 compelling. rho<=0.19 vs rho=0.853의 대비가 강력.
- 50-seed design, LOO stability, cross-domain validation --- 방법론적 rigour가 UAI 수준.
- Placebo test로 COVID 특이성 확인 (6-140x ratio).
- "Confident misclassification" (entropy 감소) 발견은 practitioners에게 중요한 경고.

### Reviewer가 가장 의문을 제기할 부분

1. **Sample size (n=16).** Correlation 연구의 fundamental limitation. Bootstrap CI [0.50, 0.96]의 넓이가 이를 반영. Leave-one-out에서 2/8 non-significant. UAI의 통계 reviewers는 이 점을 지적할 가능성 높음.

2. **Class cardinality confound.** Within SALT에서 log(num_classes) rho=0.743 (p=0.035), partial correlation에서 concentration이 non-significant (p=0.131). Cross-domain evidence가 이를 일부 완화하지만, clean separation이 아님. "Concentration이 informative한 것인가, 아니면 high-cardinality task가 본래 vulnerable한 것인가?"

3. **Model specificity.** RF rho=0.30, MLP rho=0.43 --- boosting model 전용 diagnostic. "SHAP concentration"이라는 일반적 이름에 비해 적용 범위가 좁다는 비판 가능. 다만 저자가 이를 명시적으로 acknowledge하고 mixed-effects 분석으로 보강한 점은 긍정적.

4. **External catastrophic evidence의 sparsity.** Covertype 단일 사례에 의존. "High concentration = catastrophic"의 external evidence가 n=1. KDDCup99 FN은 diagnostic의 한계를 보여줌.

5. **Theorem 1의 practical tightness.** Bound가 매우 느슨 (0.518 vs observed 0.98). A1 가정이 실제와 괴리. 이론이 empirical result를 "설명"하는 것인지 아니면 단순히 "consistent"한 것인지.

---

## One-Line Takeaway

> Gradient-boosted 모델에서 SHAP feature importance가 단일 feature에 집중될수록 conformal prediction의 coverage가 distribution shift 하에서 심각하게 붕괴하며, 이는 validation data만으로 배포 전에 진단 가능하다 (rho=0.853, n=16, 9 domains).

---

*Generated: 2026-02-20*
*Source: `/Users/i767700/Github/ai-in-finance/papers/conformal_covid/uai_2026/main.tex`*
