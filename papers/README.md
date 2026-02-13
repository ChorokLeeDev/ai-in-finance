# Research Papers

## Overview

| # | Paper | Venue | Status | Directory |
|---|-------|-------|--------|-----------|
| 1 | **Diagnosing Conformal Prediction Failures Under Distribution Shift** | UAI 2026 | Ready to submit | `conformal_covid/` |
| 2 | **Regime-Dependent Predictive Structure Between Equity Factors** | ICAIF 2025 | Nearly ready | `causal_regimes/` |
| 3 | **RelUQ: Schema-Guided Uncertainty Attribution for Relational Databases** | NeurIPS 2026 | Needs major revision | `reluq/` |

---

## 1. Conformal Prediction (conformal_covid/)

**Title:** Diagnosing Conformal Prediction Failures Under Distribution Shift: A COVID-19 Case Study

**Key result:** SHAP concentration predicts conformal prediction failure under distribution shift (rho=0.888, p<0.001, 16 tasks, 9 domains). Formal Theorem 1 proves score inflation is monotone in concentration.

```
conformal_covid/
├── uai_2026/           # Submission package
│   ├── main.tex        # Paper source (latest)
│   ├── main.pdf        # Compiled PDF (21 pages)
│   ├── references.bib
│   └── uai2026.cls
├── code/               # All experiment scripts (64 files)
├── results/            # All experimental data (45 files)
└── figures/            # Paper figures (2 PDF + 2 PNG)
```

## 2. Causal Regimes (causal_regimes/)

**Title:** Regime-Dependent Predictive Structure Between Equity Factors: Evidence from Granger Causality

**Key result:** HML Granger-causes SMB during crisis regimes (p=1.89e-5, 9-day lag). Trading backtest is negative (-6.1%), honestly reported.

**Known issues:** Data count inconsistency across versions (8,967 vs 8,817 days); lag selection via min-p should use BIC; needs permutation test. The older `arxiv/` version overclaims and should be updated.

```
causal_regimes/
├── main_icaif.tex      # Latest version (ICAIF format, primary)
├── main_icaif.pdf
├── main_arxiv.tex      # arXiv preprint version (same content)
├── main_arxiv.pdf
├── references.bib
├── figures/            # regime_timeline.pdf
├── code/               # Analysis scripts (8 files)
└── arxiv/              # Earlier arxiv version (deprecated, overclaims)
    ├── main.tex
    └── main.pdf
```

## 3. RelUQ (reluq/)

**Title:** RelUQ: Schema-Guided Uncertainty Attribution for Relational Databases

**Key idea:** Attribute ML prediction uncertainty to FK groups in relational databases.

**Status: Needs major revision.** Core hypothesis (FK-level uncertainty attribution predicts error impact) was falsified in broad validation (aggregate rho=-0.04). Multiple .tex versions exist with inconsistent claims. The FK-grouping stability result does validate consistently and may be a viable narrower contribution.

**Known issues:** 4 divergent .tex files with contradicting results; Theorem 2-3 have mathematical issues; v2 claims 24 tasks with no backing data; v1 doesn't compile. See review notes for details.

```
reluq/
├── main.tex              # Original extended draft (XeLaTeX, Korean appendix)
├── main_neurips.tex      # NeurIPS v1 (doesn't compile)
├── main_neurips_v2.tex   # NeurIPS v2 (overclaims, placeholder data)
├── main_neurips_v3.tex   # NeurIPS v3 (most honest, reduced scope)
├── main.pdf              # Compiled from main.tex
├── references.bib
├── neurips_2025.sty
├── figures/              # 13 PDF figures + PNG copies
├── experiments/          # Experiment scripts and results
└── results/              # scale_up_extended.json
```
