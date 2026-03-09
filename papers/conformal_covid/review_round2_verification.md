# UAI 2026 Review Round 2 — Fix Verification Report
Date: 2026-02-20
File verified: `papers/conformal_covid/uai_2026/main.tex` and `uai_2026/references.bib`

---

## Item 1 — Abstract duplicate paragraph
**Result: PASS**

The abstract (`\begin{abstract}...\end{abstract}`, lines 39–46) is a single paragraph with no duplication. The ρ=0.853 and decision-framework content appear exactly once. No repeated sentences detected.

---

## Item 2 — Theorem verification bounds in `app:theory_proof`
**Result: PASS**

Lines 605 contain the exact required values:

> s-shipcond (0.518 vs. observed 0.98), s-payterms (0.545 vs. 0.92), s-group (0.474 vs. 1.00), i-plant (0.261 vs. 0.86), and s-incoterms (0.296 vs. 0.87)

All five task bounds match the specified corrected values.

---

## Item 3 — `angelopoulos2024conformal` venue
**Result: PASS**

`references.bib` lines 145–150:
```bibtex
@inproceedings{angelopoulos2024conformal,
  title={Conformal Risk Control},
  ...
  booktitle={International Conference on Learning Representations},
  year={2024}
}
```
Entry type is `@inproceedings` and `booktitle` is `International Conference on Learning Representations`. No JASA reference present.

---

## Item 4 — New citations `lei2018distribution` and `dua2017uci`
**Result: PASS**

Both entries exist in `references.bib`:

- `lei2018distribution` (lines 182–190): Lei et al., JASA 2018, "Distribution-Free Predictive Inference for Regression", volume 113, pages 1094–1111.
- `dua2017uci` (lines 192–198): Dua & Graff, UCI ML Repository, 2017, `@misc` entry with URL.

---

## Item 5 — `fey2023relbench` arXiv identifier
**Result: PASS**

`references.bib` lines 130–136:
```bibtex
@article{fey2023relbench,
  ...
  note={arXiv:2407.20060},
  year={2023}
}
```
The string `arXiv:2407.20060` is present in the `note` field.

---

## Item 6 — Table 2 (`tab:stratified_correlation`) footnotes
**Result: PASS**

Lines 269–284 of `main.tex` show the table with the following footnote markers and footnotes:

(a) COVID-era row has superscript `$^\S$` and footnote:
> `$^\S$COVID-era ($n=9$): 8 SALT tasks plus Stack~Overflow (temporal shift from 2015--2018 split), the one external dataset sharing a COVID-adjacent temporal structure.`

(b) n=11 row has dagger `$^\dagger$` and footnote:
> `$^\dagger$Single-seed external values; multi-seed-consistent value at $n=11$ is $\rho=0.818$, $p=0.002$ (see Appendix~\ref{app:icc}).`

(c) n=15 row has double-dagger `$^\ddagger$` and footnote:
> `$^\ddagger$$n=15$ excludes Stack~Overflow (near-binary ceiling effect, 3 classes); $n=16$ includes it in the multiclass primary set.`

All three required footnotes are present and correctly attributed.

---

## Item 7 — Section 5.3 "8 tasks" claim
**Result: PASS**

Line 251 reads:
> "Across all 8 SALT multiclass tasks, SHAP concentration correlates with coverage degradation..."

The phrase "Among 8 tasks with severe feature turnover" is not present. The correct phrasing is in place.

---

## Item 8 — Stack Overflow row in Table A.5 (`tab:framework_validation`)
**Result: PASS**

Lines 522–523 of the framework validation table contain:
```
Stack Overflow & 7.4 & ROB & ROB & Near-binary (3 cl.) \\
```
Stack Overflow is present with concentration ~7.4%, ROB/ROB classification, and the near-binary ceiling note. The footnote on line 533 also explicitly states:
> "Stack Overflow (3 classes) exhibits near-binary ceiling effect; excluded from $n=16$ multiclass primary endpoint."

---

## Item 9 — Retraining p=0.04 note in Section 6.4
**Result: PASS**

Line 349 reads:
> "Quarterly retraining improves sales-shipcond by +18.9~pp ($p=0.04$, unadjusted; Holm-corrected over 3 tasks: $p=0.12$)..."

Both the unadjusted and Holm-corrected p-values are present with the exact required phrasing.

---

## Item 10 — Assumption A1 footnote about SHAP log-odds space
**Result: PASS**

Line 152 contains a `\footnote{}` immediately after Assumption (A1):
> `\footnote{Assumption (A1) posits additivity in probability space. In practice, SHAP values for tree ensembles are computed in log-odds space; the additive decomposition is therefore an approximation. We use (A1) as an idealised model to derive the monotonicity result, treating SHAP-derived $C$ as an empirical proxy for the theoretical concentration parameter.}`

The footnote is present and correctly addresses the SHAP log-odds space caveat.

---

## Item 11 — Figure updated to n=16
**Result: PASS**

Lines 294–297:
```latex
\includegraphics[width=\linewidth]{results/figure_n16_correlation.pdf}
\caption{\textbf{SHAP Concentration Correlates with Coverage Degradation (Primary Result).} Spearman $\rho=0.853$, $p<0.001$ across all 16 multiclass tasks in 9 domains ...}\label{fig:n11_correlation}
```

The figure uses `results/figure_n16_correlation.pdf` and the caption mentions ρ=0.853 and n=16 explicitly. (Note: the label `fig:n11_correlation` retains the old name but the content correctly reflects n=16.)

---

## Item 12 — New citations used in text
**Result: FAIL**

A grep of `main.tex` for `lei2018distribution` and `dua2017uci` returns **no matches**. Both entries exist in `references.bib` (Item 4 passes) but neither is `\cite{}`d anywhere in the main text or appendix. These citations are defined but unused.

**What needs to be fixed:** Add `\cite{lei2018distribution}` and `\cite{dua2017uci}` at appropriate locations in the paper body. Suggested placements:
- `dua2017uci`: In Section 4.4 (Cross-Domain and External Validation, line ~351) when introducing the external UCI datasets, e.g., "External validation uses datasets from the UCI ML Repository~\citep{dua2017uci}..."
- `lei2018distribution`: In Section 2 (Related Work, line ~76) alongside other split-conformal/distribution-free regression references, e.g., after `\citet{tibshirani2019conformal}`: "...and regression~\citep{romano2019conformalized, lei2018distribution}."

---

## Summary

| # | Item | Result |
|---|------|--------|
| 1 | Abstract duplicate paragraph | PASS |
| 2 | Theorem verification bounds | PASS |
| 3 | angelopoulos2024conformal venue (ICLR) | PASS |
| 4 | New citations exist in bib | PASS |
| 5 | fey2023relbench arXiv:2407.20060 | PASS |
| 6 | Table 2 footnotes (COVID-era, dagger, ddagger) | PASS |
| 7 | Section 5.3 "Across all 8 SALT multiclass tasks" | PASS |
| 8 | Stack Overflow row in Table A.5 (conc ~7.4, ROB/ROB) | PASS |
| 9 | Retraining p=0.04 with Holm correction p=0.12 | PASS |
| 10 | Assumption A1 footnote about SHAP log-odds | PASS |
| 11 | Figure uses figure_n16_correlation.pdf, caption ρ=0.853 n=16 | PASS |
| 12 | New citations actually \cite{}'d in text | **FAIL** |

**11/12 items PASS. 1 item FAILS.**

The only fix required is to add `\cite{lei2018distribution}` and `\cite{dua2017uci}` into the paper body text. Both bib entries are correctly defined; they are simply not referenced in the manuscript.
