# Uncertainty Attribution for Relational Data

## 연구 질문
> "예측이 불확실한 이유가 **어떤 데이터 소스(FK relationship)**에서 오는가?"

---

## Part 1: Validation Test 결과 (2025-11-28)

### 발견: Feature Importance ≠ Uncertainty Contribution

**실험**: SALT 데이터셋, 3개 classification tasks, Leave-one-out 방식

| Task | Spearman r | Top 3 Ranking |
|------|------------|---------------|
| sales-group | 0.029 | DIFFERENT |
| sales-office | 0.200 | DIFFERENT |
| sales-payterms | 0.486 | DIFFERENT |

**평균 상관관계: 0.238** (약함)

### 일관된 패턴
```
SALESORGANIZATION: 예측 중요도 1위 → 불확실성 기여 낮음
DISTRIBUTIONCHANNEL: 예측 중요도 낮음 → 불확실성 기여 높음
```

### 의미
- 예측을 잘 하는 feature ≠ 확신을 주는 feature
- Uncertainty attribution은 feature importance와 **다른 정보**를 제공

---

## Part 2: 문헌 조사

### 관련 연구 (Uncertainty Attribution)

| 논문 | 연도 | 핵심 내용 | 한계 |
|-----|------|---------|------|
| [CLUE](https://ar5iv.labs.arxiv.org/html/2006.06848) | 2020 | Counterfactual로 BNN uncertainty 설명 | 이미지, BNN 한정 |
| [Attribution of Predictive Uncertainties](https://arxiv.org/abs/2107.08756) | 2021 | Classification uncertainty를 feature에 attribution | 이미지 중심 |
| [Path Integrals for Model Uncertainties](https://openreview.net/forum?id=ZC1s7bdR9bD) | UAI 2022 | Path integral로 uncertainty attribution | 이미지, BNN |
| [Variance Feature Attribution](https://arxiv.org/abs/2312.07252v1) | 2023 | Variance를 feature에 attribution | Tabular 가능 |
| [Effects of Uncertainty on SHAP](https://umangsbhatt.github.io/reports/AAAI_XAI_QB.pdf) | AAAI | OOD에서 SHAP 품질 저하 | 분석만, 해결책 없음 |

### 핵심 인용
> "Understanding model uncertainties requires meaningfully **attributing** a model's predictive uncertainty to its input features" - UAI 2022

### Gap 분석

| 기존 연구 | 우리 방향 |
|----------|----------|
| 이미지 (픽셀 level) | Tabular/Relational |
| BNN 필수 | Ensemble (실용적) |
| 개별 feature | FK relationship (그룹) |
| 단일 시점 | Distribution shift 전후 |

---

## Part 3: 연구 방향 옵션

### Option A: Distribution Shift + Uncertainty Attribution

**가설**: Distribution shift 발생 시, 특정 FK의 uncertainty 기여가 더 크게 증가

**방법**:
1. Train set (pre-COVID)에서 FK별 uncertainty 기여 측정
2. Val set (COVID)에서 FK별 uncertainty 기여 측정
3. 비교: 어느 FK에서 delta가 큰가?

**장점**: COVID라는 실제 사례로 검증 가능
**단점**: SALT FK 구조가 단순 (4개 customer FK 중복)

### Option B: GNN + FK Attribution (Message Passing)

**가설**: GNN의 message passing 경로별로 uncertainty 기여가 다름

**방법**:
1. RelBench GNN 학습
2. 각 FK 경로의 message를 제거하고 uncertainty 측정
3. FK 경로별 기여도 비교

**장점**: Relational learning에 specific한 contribution
**단점**: 구현 복잡, 계산 비용 높음

---

## Part 4: 현재 상태

### 완료
- [x] Feature-level validation test (3 tasks)
- [x] 문헌 조사 (uncertainty attribution 분야)
- [x] Gap 확인 (relational + FK grouping 없음)
- [x] 연구 방향 옵션 정리

### 결정 완료 (2025-11-28)
- [x] Option A vs B 선택 → **Option A: Distribution Shift + Uncertainty Attribution**
- [x] 데이터셋 선택 → **SALT** (COVID-19 자연 실험, 기존 4 phase 완료)

### 리스크
1. **단순 aggregation 비판**: FK grouping만으로는 novelty 부족할 수 있음
2. **SALT FK 중복**: 4개 customer FK가 88% 동일
3. **계산 비용**: Shapley 방식은 2^n 조합 필요

---

## Part 5: FK Uncertainty Attribution Results (2025-11-28)

### Option A 실행 결과: Train vs Val FK 기여도 비교

| Task | Biggest FK Change | Delta | 해석 |
|------|-------------------|-------|------|
| **sales-payterms** | SALESGROUP | **+0.4473** | COVID 중 SALESGROUP이 불확실성의 주요 원인 |
| **item-shippoint** | BILLTOPARTY | **+0.2139** | BILLTOPARTY 기여 급증 |
| **sales-incoterms** | SALESGROUP | -0.3786 | SALESGROUP 기여 급감 |
| **item-incoterms** | PAYERPARTY | -0.2363 | PAYERPARTY 기여 감소 |
| **sales-group** | HEADERINCOTERMSCLASSIFICATION | +0.0645 | 증가 |

### 핵심 발견

1. **sales-payterms**: SALESGROUP가 Train에서 거의 0 기여 → Val에서 +0.4473 (핵심 불확실성 원인)
2. **item-shippoint**: BILLTOPARTY가 +0.005 → +0.219 (40배 증가)
3. **sales-incoterms**: SALESGROUP가 +0.378 → -0.0002 (방향 역전!)

### 의미

**연구 질문에 대한 답**: "예측이 불확실한 이유가 어떤 데이터 소스에서 오는가?"

→ **Distribution shift 시 불확실성의 원인이 되는 FK가 달라진다!**
- Pre-COVID: 특정 FK들이 불확실성에 기여
- COVID: 다른 FK들이 불확실성의 주요 원인으로 부상

이것이 기존 Feature Importance와 다른 정보를 제공:
- Feature Importance: 어떤 feature가 **예측**에 중요한가
- Uncertainty Attribution: 어떤 feature가 **확신/불확신**에 기여하는가

---

## Part 6: 다음 단계

**완료** (2025-11-28):
1. ~~Option A 또는 B 결정~~ → ✅ Option A 선택
2. ~~FK Uncertainty Attribution (Train vs Val)~~ → ✅ 8개 task 완료

**다음 단계**:
1. 결과 시각화 (FK attribution delta heatmap)
2. Publication 준비 (workshop paper)

**판단 기준**:
- FK별 uncertainty 기여가 유의미하게 다른가?
- Distribution shift 전후로 패턴이 변하는가?
- 기존 feature importance와 다른 insight를 주는가?
