# UAI 2026 최종본 핵심 기여 분석 보고서 (v2)

**논문**: "Diagnosing Conformal Prediction Failures Under Distribution Shift: A COVID-19 Case Study"
**파일**: `/Users/i767700/Github/ai-in-finance/papers/conformal_covid/uai_2026/main.tex`
**분석일**: 2026-02-20 (최종본 기준 재분석)

---

## 1. 핵심 기여 3가지 (증거 강도 포함)

### Contribution 1: SHAP Concentration --- Pre-Deployment Conformal Failure Diagnostic

SHAP concentration(top-1 feature importance 비율)이 conformal prediction의 coverage 붕괴 심각도를 **배포 전에** 예측할 수 있음을 실증. 16개 multiclass task, 9개 domain에서 Spearman rho=0.853 (p<0.001, bootstrap CI [0.50, 0.96], Kendall tau=0.667).

핵심 증거:
- SALT 내부 8개 task: rho=0.833, p=0.010 (50-seed paired design)
- LOO stability: rho in [0.75, 0.96] (6/8 jackknife 유의)
- 대안 진단 지표 대비 우위: native FI rho=0.667 (n.s.), ensemble disagreement rho=0.452 (n.s.)
- Shift detector(MMD/C2ST/PSI)는 모든 task에서 shift를 동일하게 탐지하나 심각도 구분 불가 (rho<=0.19, 모두 p>0.6)
- Mixed-effects (3 boosting model, n=24): beta1=1.64, p=0.0006

**증거 강도: Strong.** CI 하한 0.50이 moderate effect 이상을 보장. 다만 SALT 내 n=8 CI 하한은 0.29로 단독 weak effect 가능성 잔존.

### Contribution 2: Score Inflation Theorem (Theorem 1)

Additive feature-decomposition 가정 하에서 APS conformity score와 coverage bound가 concentration C에 대해 **단조 악화**함을 형식 증명. 가정 3개를 명시: (A1) 확률 공간 가법 분해, (A2) shift 하 concentrated misclassification, (A3) 잔여 feature의 exchangeability.

핵심 증거:
- 5개 적용 가능 task 전체에서 conservative bound 충족 (gap 0.26-0.60)
- Catastrophic task KS statistic 0.68-0.96 (모두 p<10^-10): score stochastic dominance 확인
- Catastrophic task의 prediction entropy가 오히려 **감소** --- 모델이 confidently wrong해지는 반직관적 현상이 이론과 정합
- RAPS가 class-accumulation failure에만 도움되고 concentrated single-feature failure에는 무력한 결과가 이론의 메커니즘 분리를 지지

**증거 강도: Moderate-Strong.** 증명 자체는 완결적이나, (A1)이 실제 tree SHAP(log-odds)와 불일치하는 근사임을 저자가 직접 인정. Conservative bound의 gap이 큼 (e.g., 0.261 vs observed 0.86).

### Contribution 3: Shift Detection =/= Severity Prediction의 실증적 분리

MMD, C2ST, PSI 모두 8개 task에서 shift를 탐지하지만(모두 유의), coverage drop 심각도와의 상관은 rho<=0.19. **Shift 존재 탐지와 모델별 실패 심각도 예측이 근본적으로 다른 문제**임을 실증.

핵심 증거:
- 3종 shift detector x 8 task: 일관된 비유의 상관 (MMD rho=-0.048, C2ST rho=0.191, PSI rho=0.071)
- Catastrophic task에서 entropy 감소 (confident wrong) --- 기존 monitoring이 가장 위험한 경우에서 오히려 misleading
- 외부 9개 domain에서도 high-concentration Covertype만 catastrophic (10/10 seed), low-concentration 6개 domain 모두 robust (10/10)

**증거 강도: Strong.** 체계적 비교, 3가지 detector 모두 일관, 반직관적 entropy 결과가 메시지를 강화.

---

## 2. UAI Accept 가능성을 높이는 강점 TOP 3

### 강점 1: "Pre-Deployment" 프레이밍의 실무적 차별화

기존 conformal-under-shift 연구는 shift **후** 적응(ACI, weighted CP 등)에 집중. 이 논문은 shift **전에** 어떤 모델이 실패할지 예측하는 진단 도구를 제안. UAI 청중(이론+응용 균형)에게 어필하며, 기존 문헌의 명확한 빈 공간을 차지. Barber et al. (2023)의 이론적 TV distance bound를 "실무에서 어떻게 사전 추정하는가"로 연결한 점이 positioning의 핵심.

### 강점 2: 투명한 한계 인정 + 견고한 통계적 엄밀성

n=16이라는 소규모 관측에서 bootstrap CI, LOO stability, ICC, mixed-effects, Holm-Bonferroni correction까지 동원. RF rho=0.30, MLP rho=0.43 등 약한 결과를 숨기지 않고 메커니즘 해석과 함께 보고. KDDCup99의 intermediate regime, binary ceiling effect, protective factor 등 진단이 **작동하지 않는** 조건을 명시적으로 규정. Reviewer가 "이건 왜 안 했나?"라고 물을 만한 실험이 거의 없는 수준.

### 강점 3: 이론-실증-실무의 삼각 구조

(1) Theorem 1이 "왜 concentration이 failure를 예측하는가"의 방향성 제시, (2) 16-task empirical correlation이 이론의 예측 확인, (3) Decision framework가 실무 운용 방법 제시. 특히 RAPS의 mechanism-dependent 결과(class-accumulation failure vs concentrated-dependence failure)가 이론의 세부 예측과 정합하는 점이 설득력을 높임.

---

## 3. 여전히 약한 부분 (한 줄 요약)

외부 catastrophic 증거가 Covertype 단 1건에 집중되어 있으며, boosting 이외 모델(RF/MLP)에서 진단력이 비유의하므로, 일반화 가능성은 "gradient-boosted multiclass model" 범위로 제한된다.

---

## 4. One-Line Takeaway

> Gradient-boosted model의 SHAP concentration이 40%를 넘으면 distribution shift 하에서 conformal prediction이 catastrophic하게 실패할 가능성이 높으며, 이는 기존 shift detector(MMD/C2ST)로는 예측할 수 없는 model-specific 취약성 신호이다.

---

## 5. 향후 연구 방향

1. **Boosting 외 모델 일반화**: MLP의 failure mode는 "global sensitivity"로 다름 --- neural network 전용 concentration analog 탐색 필요
2. **외부 catastrophic case 확보**: Covertype 1건을 넘어, high-concentration + catastrophic failure의 external evidence 확대
3. **인과적 식별**: associative evidence에서 intervention study로 전환 (e.g., feature importance를 인위적으로 분산시킨 후 robustness 변화 측정)
4. **Regression task 확장**: CQR 등 regression conformal에서 analogous diagnostic 존재 여부 탐색
5. **Adaptive 통합**: SHAP concentration 기반 사전 진단과 ACI 강도를 task별로 조절하는 adaptive scheme

---

*분석 완료. 시뮬레이션 리뷰 점수 8.0/8.0/7.5 기준, 현재 최종본은 UAI accept 수준으로 판단됨.*
