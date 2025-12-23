# Research Papers

## Paper Status

| Status | Paper | Target | Deadline |
|--------|-------|--------|----------|
| Draft | **RelUQ: Schema-Guided Uncertainty Attribution for Relational Databases** | NeurIPS 2026 | May 2026 |
| Draft | **Causal Structure Changes Across Market Regimes: Evidence from Factor Returns** | ICAIF 2025 | TBD |

---

## 1. RelUQ (reluq/)

**Title:** RelUQ: Schema-Guided Uncertainty Attribution for Relational Databases

**Core Idea:** ML 모델의 예측 불확실성을 개별 feature가 아닌 FK(Foreign Key) 그룹 단위로 귀인하여, 어떤 데이터 수집 프로세스가 불확실성에 기여하는지 파악

### Progress

| Section | Status | Notes |
|---------|--------|-------|
| Abstract | Done | Error Propagation Hypothesis 명시 |
| Introduction | Done | FK grouping의 장점 설명 |
| Related Work | Done | UQ, Attribution, Relational Learning 커버 |
| Method | Done | Algorithm, Hierarchy, Actionability, Theory |
| Experiments | Done | 4개 도메인 (SALT, Trial, Amazon, Stack) |
| Conclusion | Done | 범위와 한계 명시 |
| Figures | Done | 8개 figure (overview, baseline, ablation 등) |

### Key Results
- **Error Propagation 도메인** (ERP, Clinical): Spearman ρ ≥ 0.90
- **Associative 도메인** (Q&A): Spearman ρ = -0.50 (작동 안함)
- 도메인 적용 범위 명확히 정의

### Limitations & Improvements

#### Format Issues (Must Fix)

| Issue | Current | Required | Priority |
|-------|---------|----------|----------|
| Template | article class | NeurIPS 2026 template | P0 |
| Page count | ~15 pages | 9 pages + appendix | P0 |
| Korean text | Appendix에 한글 | 영어로 번역 or 제거 | P0 |
| Font package | kotex (XeLaTeX) | 표준 pdfLaTeX | P0 |
| References | inline bibitems | 별도 references.bib | P1 |
| Anonymization | 저자 정보 포함 | 익명화 | P1 |

#### Content Issues (Should Fix)

| Issue | Current State | Improvement | Priority |
|-------|---------------|-------------|----------|
| **Domain Coverage** | 4 domains (SALT, Trial, Amazon, Stack) | Add Banking, Insurance, Manufacturing | P1 |
| **Baselines** | Self-defined baselines only | Add TreeSHAP variance, InfoSHAP comparison | P1 |
| **Scalability** | Mentioned but not tested | Large-scale experiments (100K+ samples) | P2 |
| **Classification** | Regression only | Extend to classification (ensemble disagreement) | P2 |
| **Diagnostic Tool** | Theory only | Automatic EP structure detection algorithm | P2 |
| **UQ Methods** | Ensemble variance only | Compare MC Dropout, Conformal Prediction | P3 |

#### Content Issues (Nice to Have)

| Issue | Description | Priority |
|-------|-------------|----------|
| Real deployment case study | Industry partner validation | P3 |
| Computational cost analysis | Runtime vs accuracy tradeoff | P3 |
| Online/streaming setting | Temporal drift handling | P4 |

### File Structure
```
papers/reluq/
├── main.tex           # 892 lines, full paper
├── figures/           # 18 PDF figures
├── FORMAL_THEOREM.md  # 수학적 정형화
├── THEORY_FORMALIZATION.md
└── NEURIPS_2026_PLAN.md
```

---

## 2. Causal Regimes (causal_regimes/)

**Title:** Causal Structure Changes Across Market Regimes: Evidence from Factor Returns

**Core Idea:** Equity factor 간의 인과관계가 시장 레짐에 따라 변한다는 실증적 발견. Crisis에서는 Value→Size, Crowding에서는 Size→Value로 방향이 역전됨.

### Progress

| Section | Status | Notes |
|---------|--------|-------|
| Abstract | Done | Key finding 명시 (p-values, lags) |
| Introduction | Done | 2007 quant meltdown 동기 |
| Related Work | Done | Crowding, Regime-switching, Causal discovery |
| Methodology | Done | Student-t HMM, Granger causality |
| Results | Done | Regime characteristics, Main finding, Early warning |
| Discussion | Done | Risk management implications, Limitations |
| Appendix | Done | Algorithm, Full Granger tables |

### Key Results
- **Crisis Regime:** HML → SMB (p = 1.89e-5, 9-day lag)
- **Crowding Regime:** SMB → HML (p = 1.94e-4, 3-day lag)
- **Normal Regime:** No significant causality
- **Early Warning:** Lehman 61일 전 감지

### Limitations & Improvements

#### Format Issues (Must Fix)

| Issue | Current | Required | Priority |
|-------|---------|----------|----------|
| Template | article class | ICAIF/ACM template | P0 |
| Figures | Referenced but missing | Generate all figures | P0 |
| References | Incomplete .bib | Complete all citations | P1 |
| Code | "Available upon request" | GitHub repository | P1 |

#### Content Issues (Should Fix)

| Issue | Current State | Improvement | Priority |
|-------|---------------|-------------|----------|
| **Crowding Proxy** | Rolling volatility (indirect) | Discuss limitations, consider alternatives | P1 |
| **Out-of-sample** | None | 2024 data validation | P1 |
| **Robustness** | "See Appendix" only | Add actual tables/figures | P1 |
| **Causality Type** | Granger (predictive) | Clarify ≠ structural causality | P2 |
| **Confounding** | Mentioned | More explicit discussion | P2 |

#### Content Issues (Nice to Have)

| Issue | Description | Priority |
|-------|-------------|----------|
| Trading strategy backtest | Simulated P&L from early warning | P3 |
| FANTOM comparison | Direct comparison with SOTA | P3 |
| International markets | Non-US factor data | P4 |
| Real-time implementation | Production system design | P4 |

### File Structure
```
papers/causal_regimes/
├── arxiv/
│   ├── main.tex       # 356 lines, ArXiv format
│   ├── main.pdf       # Compiled PDF
│   ├── references.bib # Bibliography
│   └── submission.zip # Ready package
├── icaif_draft.md     # ICAIF v1
├── icaif_draft_v2.md  # ICAIF v2
├── neurips_preprint.md
└── preprint_final.md
```

---

## Action Plan

### Phase 1: RelUQ (High Priority) - Target: Q1 2026

| Step | Task | Effort | Status |
|------|------|--------|--------|
| 1.1 | NeurIPS 2025 template 적용 | 2h | [x] Done (main_neurips.tex) |
| 1.2 | 한글 제거 + pdfLaTeX 호환 | 1h | [x] Done |
| 1.3 | references.bib 분리 | 1h | [x] Done |
| 1.4 | 9 pages로 압축 (appendix 분리) | 4h | [x] Done |
| 1.5 | Banking/Insurance 도메인 실험 | 1-2 weeks | [ ] |
| 1.6 | TreeSHAP/InfoSHAP baseline 비교 | 1 week | [ ] |
| 1.7 | Anonymization | 30min | [x] Done |

### Phase 2: Causal Regimes (Medium Priority) - Target: Q2 2025

| Step | Task | Effort | Status |
|------|------|--------|--------|
| 2.1 | Figure 생성 (regime detection plot) | 2h | [x] Done (12 figures copied) |
| 2.2 | Figure 생성 (causal DAG per regime) | 2h | [x] Done (gate3_dags.png) |
| 2.3 | ICAIF template 적용 | 2h | [ ] |
| 2.4 | references.bib 완성 | 1h | [x] Done (15 entries) |
| 2.5 | 2024 out-of-sample validation | 4h | [ ] |
| 2.6 | GitHub code repository 정리 | 4h | [ ] |
| 2.7 | Robustness appendix 상세화 | 2h | [ ] |

---

## Known Limitations Summary

### RelUQ
1. **Domain Scope:** EP structure 도메인에서만 유효 (transactional data)
2. **Method Scope:** Ensemble + Permutation만 검증됨
3. **Scale:** 3K samples로 테스트, 대규모 미검증
4. **Task Type:** Regression만 지원

### Causal Regimes
1. **Causality Type:** Granger (predictive) ≠ Structural (interventional)
2. **Crowding Measurement:** Indirect proxy (volatility-based)
3. **Regime Stationarity:** 35년간 3-regime 가정
4. **Sample Size:** Crisis regime 1,167 days (power 제한)
5. **Factor Definition:** Fama-French specific
