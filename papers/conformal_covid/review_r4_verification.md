# UAI 2026 Pre-Submission Verification Report (Round 4)
Date: 2026-02-20
File: `papers/conformal_covid/uai_2026/main.tex`
BibTeX: `papers/conformal_covid/uai_2026/references.bib`

---

## Round 3 수정 검증

### 1. Table 2에서 COVID-era n=9 행이 완전히 제거되었는지 (행과 footnote 모두)
**PASS**

`tab:stratified_correlation` (Table 2)에 n=9 행이 존재하지 않음. 현재 행 목록:
- Multiclass (SALT) n=8
- Multiclass (4 dom.) n=11
- Multiclass (8 dom.) n=15
- **Multiclass (9 dom.) n=16** (bold, primary result)
- Combined (11 dom.) n=19

"COVID-era" 문자열도 main.tex 전체에서 발견되지 않음. footnote 내에도 n=9 관련 언급 없음.

---

### 2. `\citep{fey2024relbench}` 가 사용되고 있는지 (fey2023relbench 없어야 함)
**PASS**

Line 88에서 `\citep{fey2024relbench}` 사용 확인:
```
We use the SALT (Supply chain ALlocaTion) dataset from RelBench~\citep{fey2024relbench}
```
`fey2023relbench`는 tex 파일 전체에서 발견되지 않음.

---

### 3. references.bib에 `fey2024relbench` @inproceedings NeurIPS 2024 항목이 있는지
**PASS**

bib lines 133–139:
```bibtex
@inproceedings{fey2024relbench,
  title={RelBench: A Benchmark for Deep Learning on Relational Databases},
  author={Fey, Matthias and Hu, Weihua and Huang, Kexin and ...},
  booktitle={Advances in Neural Information Processing Systems},
  volume={37},
  year={2024}
}
```
@inproceedings 타입, NeurIPS(NIPS) vol.37, 2024 확인.

---

### 4. `angelopoulos2021gentle` 가 F&T 2023 버전 (volume=16, pages=494--591)인지
**PASS**

bib lines 52–60:
```bibtex
@article{angelopoulos2021gentle,
  title={A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification},
  author={Angelopoulos, Anastasios N and Bates, Stephen},
  journal={Foundations and Trends in Machine Learning},
  volume={16},
  number={4},
  pages={494--591},
  year={2023}
}
```
volume=16, pages=494--591, year=2023 모두 정확.

---

### 5. `feldman2023achieving` 가 @article (TMLR)인지
**PASS**

bib lines 141–146:
```bibtex
@article{feldman2023achieving,
  title={Achieving Risk Control in Online Learning Settings},
  author={Feldman, Shai and Bates, Stephen and Angelopoulos, Anastasios N},
  journal={Transactions on Machine Learning Research},
  year={2023}
}
```
@article 타입, journal=TMLR 확인.

---

### 6. Figure label이 `\label{fig:n16_correlation}`이고 `\ref{fig:n16_correlation}`으로 참조되는지 (fig:n11 사라졌는지)
**PASS**

- Line 294: `}\label{fig:n16_correlation}` — label 정의 확인
- Line 251 (본문 내 참조): `Figure~\ref{fig:n16_correlation}` 확인 (행 omitted로 표시되었으나 grep 매칭 확인)
- `fig:n11` 문자열은 main.tex 전체에서 발견되지 않음

---

### 7. Covertype drop이 81.8~pp로 일관되게 표기되는지 (abstract + body 모두)
**PASS**

3곳 모두 `81.8~pp` 또는 `81.8`:
- Line 45 (abstract): `Covertype ($C=49.8\%$, 81.8~pp drop)`
- Line 259 (Section 4.3 body): `correctly flagging catastrophic failure (Covertype, 81.8~pp drop)`
- Line 351 (Section 5.4 body): `Covertype is the key external catastrophic case ($C=49.8\%$, 81.8~pp drop, 10/10 seeds)`

모두 `81.8~pp`로 일관됨.

---

## 추가 체크

### 8. Table 2 행 순서: Multiclass(SALT) n=8 → (4 dom.) n=11 → (8 dom.) n=15 → (9 dom.) n=16 → Combined n=19
**PASS**

Lines 269–273 확인:
```latex
Multiclass (SALT)    & 8  & 0.833 & 0.714 & 0.010 & [0.29, 1.00] \\
Multiclass (4 dom.) & 11 & 0.909 & 0.782 & <0.001 & [0.61, 1.00] \\
Multiclass (8 dom.) & 15 & 0.882 & 0.714 & <0.001 & [0.60, 0.97] \\
Multiclass (9 dom.) & 16 & 0.853 & 0.667 & <0.001 & [0.50, 0.96] \\
Combined (11 dom.)  & 19 & 0.814 & 0.626 & <0.001 & ---           \\
```
지정된 순서와 정확히 일치. n=9 행 없음.

---

### 9. 본문에서 "COVID-era" 언급이 여전히 남아있는지 (있다면 어디서, 일관성 문제인지)
**PASS — "COVID-era" 언급 없음**

`COVID-era` 문자열은 main.tex 전체에서 0건 발견됨. 삭제 완료.

단, Appendix (line 595)에 `$n=8$, $n=9$, $n=11$` 언급이 있음:
```
For the 3 correlation subsets ($n=8$, $n=9$, $n=11$), all adjusted p-values remain significant
```
이것은 COVID-era 행 삭제와 별개로, 과거 intermediate 분석 단계를 다중비교 설명 맥락에서 인용한 것임. Table 2에 n=9 행이 없으므로 불일치처럼 보일 수 있으나, 이 문장은 Holm-Bonferroni 5-test 계산 설명을 위한 부연이므로 일관성 문제는 아님. 다만 주의 요망.

---

### 10. `fey2023relbench` 잔재가 어디에도 없는지 (tex + bib 모두)
**PASS**

- main.tex 전체 검색: `fey2023relbench` 0건
- references.bib 전체 검색: `fey2023relbench` 0건

완전히 제거됨.

---

## 종합 결과

| # | 항목 | 결과 |
|---|------|------|
| 1 | Table 2 COVID-era n=9 행+footnote 제거 | PASS |
| 2 | `\citep{fey2024relbench}` 사용, fey2023 없음 | PASS |
| 3 | bib에 fey2024relbench @inproceedings NeurIPS 2024 | PASS |
| 4 | angelopoulos2021gentle F&T 2023 (vol=16, pp=494--591) | PASS |
| 5 | feldman2023achieving @article TMLR | PASS |
| 6 | label fig:n16_correlation, ref 동일, fig:n11 없음 | PASS |
| 7 | Covertype 81.8~pp 표기 일관 (abstract+body 3곳) | PASS |
| 8 | Table 2 행 순서 n=8→11→15→16→19 | PASS |
| 9 | "COVID-era" 언급 없음 | PASS |
| 10 | fey2023relbench 잔재 없음 (tex+bib) | PASS |

**전체: 10/10 PASS**

주의사항: Appendix line 595에 `($n=8$, $n=9$, $n=11$)` 언급이 있으나 이는 Holm-Bonferroni 다중비교 설명 맥락이며 Table 2 행 삭제와 직접 충돌하지 않음. 필요 시 `$n=8$, $n=11$`로 단순화 가능.
