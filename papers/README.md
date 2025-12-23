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

### Improvements Needed
1. **Broader Validation:** Banking, Insurance 등 추가 도메인
2. **UQ Method Comparison:** MC Dropout, Conformal Prediction 비교
3. **Attribution Baselines:** TreeSHAP, InfoSHAP 비교
4. **NeurIPS Format:** 현재 일반 article format → NeurIPS template 적용 필요
5. **References:** BibTeX 파일 분리 필요

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

### Improvements Needed
1. **References:** BibTeX 파일 없음 - 별도 references.bib 필요
2. **Figures:** LaTeX 내 figure 참조 있으나 실제 파일 누락
3. **Code Availability:** "Available upon request" → GitHub 공개 권장
4. **Robustness Appendix:** 상세 결과 추가 필요
5. **Submission Format:** ICAIF template 적용 필요

### File Structure
```
papers/causal_regimes/
├── arxiv/
│   ├── main.tex       # 356 lines, ArXiv format
│   ├── main.pdf       # Compiled PDF
│   └── submission.zip # Ready package
├── icaif_draft.md     # ICAIF v1
├── icaif_draft_v2.md  # ICAIF v2
├── neurips_preprint.md
└── preprint_final.md
```

---

## Next Steps

### RelUQ (Priority: High)
1. [ ] NeurIPS 2026 template 적용
2. [ ] Additional domain experiments
3. [ ] Baseline comparison (TreeSHAP, InfoSHAP)
4. [ ] Clean up references

### Causal Regimes (Priority: Medium)
1. [ ] Create references.bib
2. [ ] Generate missing figures
3. [ ] ICAIF template 적용
4. [ ] Code repository 정리 및 공개
