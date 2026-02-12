#!/usr/bin/env python3
"""Writer agent script: Fill appendix baseline table with computed entropy/ECE values.

Run after ACI compute finishes (JSON has 7 keys):
    cd papers/conformal_covid
    python3 code/writer_fill_table.py

What it does:
1. Reads aci_all_tasks_summary.json
2. Extracts ΔEntropy and ΔECE for item-plant and item-shippoint
3. Edits main.tex: fills in --- placeholders, removes ‡ footnote
4. Updates interpretive text if needed
5. Compiles PDF (pdflatex x2)
"""

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
JSON_PATH = ROOT / 'results' / 'aci' / 'aci_all_tasks_summary.json'
TEX_PATH = ROOT / 'uai_2026' / 'main.tex'


def main():
    # 1. Check JSON
    if not JSON_PATH.exists():
        print(f"ERROR: {JSON_PATH} not found")
        sys.exit(1)

    with open(JSON_PATH) as f:
        data = json.load(f)

    print(f"JSON keys ({len(data)}): {sorted(data.keys())}")

    missing = [t for t in ['item-plant', 'item-shippoint'] if t not in data]
    if missing:
        print(f"ERROR: Missing tasks: {missing}")
        print("ACI compute not finished yet. Wait for JSON to have 7 keys.")
        sys.exit(1)

    # 2. Extract values
    values = {}
    for task in ['item-plant', 'item-shippoint']:
        e = data[task]['entropy']
        c = data[task]['ece']
        values[task] = {
            'delta_entropy': e['delta_mean'],
            'delta_ece': c['delta_mean'],
        }
        print(f"{task}: ΔEntropy={e['delta_mean']:+.3f}, ΔECE={c['delta_mean']:+.4f}")

    # 3. Edit main.tex
    with open(TEX_PATH) as f:
        tex = f.read()

    original_tex = tex

    # Fill i-plant row
    de = values['item-plant']['delta_entropy']
    dc = values['item-plant']['delta_ece']
    de_str = f"$-${abs(de):.3f}" if de < 0 else f"+{de:.3f}"
    dc_str = f"+{dc:.4f}"
    tex = tex.replace(
        r"i-plant$^\ddagger$ & ROB & 23.9\% & --- & ---",
        f"i-plant & ROB & 23.9\\% & {de_str} & {dc_str}"
    )

    # Fill i-shippoint row
    de = values['item-shippoint']['delta_entropy']
    dc = values['item-shippoint']['delta_ece']
    de_str = f"$-${abs(de):.3f}" if de < 0 else f"+{de:.3f}"
    dc_str = f"+{dc:.4f}"
    tex = tex.replace(
        r"i-shippoint$^\ddagger$ & ROB & 48.8\% & --- & ---",
        f"i-shippoint & ROB & 48.8\\% & {de_str} & {dc_str}"
    )

    # Remove ‡ from footnote
    tex = tex.replace(
        r"$^\dagger$Pre-deployment diagnostic (computed on validation data only). $^\ddagger$Item-level tasks: compute-intensive (10+ min/seed); entropy/ECE pending. item-incoterms omitted entirely.",
        r"$^\dagger$Pre-deployment diagnostic (computed on validation data only). item-incoterms omitted (3-seed pilot only)."
    )

    # Update interpretive text if i-plant or i-shippoint show unexpected entropy
    # i-plant: SHAP conc=23.9% (below threshold, should be ROB-like)
    # i-shippoint: SHAP conc=48.8% (above threshold, should be SEV-like)
    # But both are classified as SEV by coverage drop (10.6%, 18.5%)
    # Check: do they show decreasing entropy like the catastrophic SEV tasks?
    ip_de = values['item-plant']['delta_entropy']
    is_de = values['item-shippoint']['delta_entropy']

    if ip_de < 0 or is_de < 0:
        # At least one item-SEV task also shows decreasing entropy
        old_text = "Counter-intuitively, all three severe tasks show \\textit{decreasing} prediction entropy"
        if ip_de < 0 and is_de < 0:
            new_text = "Counter-intuitively, all five severe tasks show \\textit{decreasing} prediction entropy"
        elif ip_de < 0:
            new_text = "Counter-intuitively, four of five severe tasks show \\textit{decreasing} prediction entropy"
        else:
            new_text = "Counter-intuitively, four of five severe tasks show \\textit{decreasing} prediction entropy"
        tex = tex.replace(old_text, new_text)
        print(f"Updated interpretive text: '{old_text[:50]}...' -> '{new_text[:50]}...'")

    if ip_de >= 0 and is_de >= 0:
        print("NOTE: Both item tasks show INCREASING entropy — different from catastrophic SEV tasks.")
        print("Consider updating interpretive text to note this distinction.")

    if tex == original_tex:
        print("WARNING: No changes made to tex file — check if placeholders already filled")
    else:
        with open(TEX_PATH, 'w') as f:
            f.write(tex)
        print(f"Updated {TEX_PATH}")

    # 4. Compile
    print("\nCompiling PDF...")
    uai_dir = ROOT / 'uai_2026'
    for i in range(2):
        result = subprocess.run(
            ['pdflatex', '-interaction=nonstopmode', 'main.tex'],
            cwd=uai_dir, capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"pdflatex pass {i+1} FAILED:")
            # Print last 20 lines of log
            for line in result.stdout.split('\n')[-20:]:
                print(f"  {line}")
            sys.exit(1)
        print(f"pdflatex pass {i+1}: OK")

    # Check page count
    log_path = uai_dir / 'main.log'
    if log_path.exists():
        log = log_path.read_text()
        pages = re.findall(r'Output written on.*\((\d+) pages', log)
        if pages:
            print(f"Pages: {pages[-1]}")

    print("\nDone! Paper compiled successfully.")
    print("\nValues inserted:")
    for task, v in values.items():
        print(f"  {task}: ΔEntropy={v['delta_entropy']:+.3f}, ΔECE={v['delta_ece']:+.4f}")


if __name__ == '__main__':
    main()
