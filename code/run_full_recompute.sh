#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${ROOT_DIR}/results"
FIGURES_DIR="${ROOT_DIR}/figures"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="${RESULTS_DIR}/recompute_logs_${TIMESTAMP}"
SNAPSHOT_DIR="${RESULTS_DIR}/baseline_snapshot_${TIMESTAMP}"

mkdir -p "${LOG_DIR}"
mkdir -p "${SNAPSHOT_DIR}"

cd "${ROOT_DIR}"

run_step() {
  local name="$1"
  shift
  echo
  echo "======================================================================"
  echo "STEP: ${name}"
  echo "======================================================================"
  "$@" 2>&1 | tee "${LOG_DIR}/${name}.log"
}

echo "Root: ${ROOT_DIR}"
echo "Timestamp (UTC): ${TIMESTAMP}"
echo "Logs: ${LOG_DIR}"
echo "Baseline snapshot: ${SNAPSHOT_DIR}"

echo
echo "Snapshotting baseline artifacts..."
mkdir -p "${SNAPSHOT_DIR}/results" "${SNAPSHOT_DIR}/figures"
shopt -s nullglob
for f in "${RESULTS_DIR}"/*.json "${RESULTS_DIR}"/*.csv; do
  cp -a "$f" "${SNAPSHOT_DIR}/results/"
done
for f in "${FIGURES_DIR}"/*.pdf "${FIGURES_DIR}"/*.png; do
  cp -a "$f" "${SNAPSHOT_DIR}/figures/"
done
shopt -u nullglob
echo "Baseline snapshot complete."

run_step deps_install uv run python -m pip install -r "${ROOT_DIR}/code/requirements-recompute.txt"
run_step import_gate uv run python -c "import numpy,pandas,scipy,matplotlib,statsmodels,hmmlearn,sklearn,torch; print('import_gate_ok')"

run_step multistart_dual uv run python "${ROOT_DIR}/code/multistart_hmm_pipeline.py" --selection-rule dual

run_step critical_fixes uv run python "${ROOT_DIR}/code/critical_fixes_analysis.py"
run_step hac_granger uv run python "${ROOT_DIR}/code/hac_granger_consistent.py"
run_step neural_selected uv run python "${ROOT_DIR}/code/neural_granger_selected.py"
run_step lstm_granger uv run python "${ROOT_DIR}/code/lstm_granger.py"
run_step te_selected uv run python "${ROOT_DIR}/code/te_selected.py"
run_step ff25_overlap uv run python "${ROOT_DIR}/code/ff25_overlap_mechanism.py"
run_step crisis_trading uv run python "${ROOT_DIR}/code/crisis_trading_backtest.py"
run_step risk_monitoring uv run python "${ROOT_DIR}/code/risk_monitoring_backtest.py"
run_step var_fixes uv run python "${ROOT_DIR}/code/var_fixes.py"
run_step hybrid_detector uv run python "${ROOT_DIR}/code/hybrid_regime_detector.py"
run_step practitioner_baseline uv run python "${ROOT_DIR}/code/practitioner_baseline.py"
run_step figures_primary uv run python "${ROOT_DIR}/code/generate_all_figures.py"
run_step figures_complexity uv run python "${ROOT_DIR}/code/complexity_spectrum_fig.py"
run_step claim_ledger uv run python "${ROOT_DIR}/code/build_claim_ledger.py"

run_step consistency_icaif uv run python "${ROOT_DIR}/code/consistency_check.py" \
  --latex-file "${ROOT_DIR}/main_icaif.tex" --json-dir "${RESULTS_DIR}" --strict --profile full
run_step consistency_arxiv uv run python "${ROOT_DIR}/code/consistency_check.py" \
  --latex-file "${ROOT_DIR}/main_arxiv.tex" --json-dir "${RESULTS_DIR}" --strict --profile arxiv

run_step compile_icaif_pdflatex_1 pdflatex -interaction=nonstopmode -halt-on-error main_icaif.tex
run_step compile_icaif_bibtex bibtex main_icaif
run_step compile_icaif_pdflatex_2 pdflatex -interaction=nonstopmode -halt-on-error main_icaif.tex
run_step compile_icaif_pdflatex_3 pdflatex -interaction=nonstopmode -halt-on-error main_icaif.tex

run_step compile_arxiv_pdflatex_1 pdflatex -interaction=nonstopmode -halt-on-error main_arxiv.tex
run_step compile_arxiv_bibtex bibtex main_arxiv
run_step compile_arxiv_pdflatex_2 pdflatex -interaction=nonstopmode -halt-on-error main_arxiv.tex
run_step compile_arxiv_pdflatex_3 pdflatex -interaction=nonstopmode -halt-on-error main_arxiv.tex

run_step write_manifest uv run python -c "import json,datetime,pathlib; p=pathlib.Path('${RESULTS_DIR}')/'recompute_manifest.json'; d={'timestamp_utc':'${TIMESTAMP}','selection_rule_primary':'ll_only (via dual)','sensitivity_rule':'screened_2008 (via dual)','baseline_snapshot':'${SNAPSHOT_DIR}','logs_dir':'${LOG_DIR}','data_source':'Kenneth French data library endpoints in pipeline scripts'}; p.write_text(json.dumps(d, indent=2)); print(f'Wrote {p}')"

echo
echo "Full recompute complete."
echo "Logs: ${LOG_DIR}"
echo "Manifest: ${RESULTS_DIR}/recompute_manifest.json"
