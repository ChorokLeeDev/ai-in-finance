# Preprint Preparation Guide

**Paper**: Conformal Prediction Under Distribution Shift: A COVID-19 Natural Experiment
**Author**: Chorok Lee (choroklee@kaist.ac.kr)
**Institution**: Korea Advanced Institute of Science and Technology (KAIST)
**Target**: arXiv or other preprint server
**Date**: 2025-12-26

---

## 🎯 TWO-STEP PROCESS

### Step 1: Fix Figure Issues (REQUIRED) ⚠️
Even for preprint, we need to fix the text cutoff issues in figures.

### Step 2: De-Anonymize (REQUIRED for preprint) ✅
Add your real name and affiliation.

---

## STEP 1: FIX FIGURE TEXT CUTOFFS (30 minutes)

### Quick Fix Approach (Recommended)

**Edit 3 Python files to remove/fix titles:**

#### File 1: `code/create_figure3_shap.py` (Line 214-215)
**BEFORE**:
```python
fig.suptitle('Feature Importance Analysis: Mechanism of Catastrophic Failure',
             fontsize=14, fontweight='bold')
```

**AFTER** (comment out):
```python
# Removed title - rely on LaTeX caption for full context
# fig.suptitle('Feature Importance Analysis: Mechanism of Catastrophic Failure',
#              fontsize=14, fontweight='bold')
```

#### File 2: `code/plot_retraining_results.py` (Line 110)
**BEFORE**:
```python
ax.set_title(f'Coverage Degradation and Recovery - {task}')
```

**AFTER** (comment out):
```python
# Removed title - rely on LaTeX caption
# ax.set_title(f'Coverage Degradation and Recovery - {task}')
```

#### File 3: `code/generate_figures.py` (Line 303)
**BEFORE**:
```python
ax.set_title('B. Placebo Test: COVID is Special\n(10-100× worse than normal drift)', fontweight='bold')
```

**AFTER**:
```python
ax.set_title('B. Placebo Test: COVID is Special\n(10-200× worse than normal drift)', fontweight='bold')
```

### Regenerate Figures
```bash
cd /Users/i767700/Github/ai-in-finance/papers/conformal_covid/code

# Regenerate all 3 affected figures
python create_figure3_shap.py
python plot_retraining_results.py
python generate_figures.py

cd ..
```

---

## STEP 2: DE-ANONYMIZE FOR PREPRINT

### Change 1: Remove Anonymous Mode

**File**: `main.tex`, Line 2

**BEFORE**:
```latex
\documentclass[sigconf,anonymous]{acmart}
```

**AFTER**:
```latex
\documentclass[sigconf]{acmart}
```

### Change 2: Add Your Information

**File**: `main.tex`, Lines 23-26

**BEFORE**:
```latex
\author{Anonymous}
\affiliation{%
  \institution{Anonymous Institution}
}
```

**AFTER**:
```latex
\author{Chorok Lee}
\email{choroklee@kaist.ac.kr}
\affiliation{%
  \institution{Korea Advanced Institute of Science and Technology (KAIST)}
  \department{School of Computing}
  \city{Daejeon}
  \country{South Korea}
}
```

### Change 3: Update Copyright (Optional for Preprint)

**File**: `main.tex`, Line 5-7

**CURRENT**:
```latex
\setcopyright{none}
\settopmatter{printacmref=false}
\renewcommand\footnotetextcopyrightpermission[1]{}
```

**FOR PREPRINT** (keep as is, or change to):
```latex
\setcopyright{rightsretained}  % You retain copyright for preprint
\settopmatter{printacmref=false}
\renewcommand\footnotetextcopyrightpermission[1]{}
```

### Change 4: Add Preprint Note (Optional but Recommended)

**Add after line 27** (after affiliation):

```latex
\renewcommand{\shortauthors}{Lee}

% Preprint notice
\begin{teaserfigure}
  \centering
  \small
  \textit{Preprint. Work in progress.}
\end{teaserfigure}
```

OR simpler, add to abstract:

```latex
\begin{abstract}
\textit{Note: This is a preprint. Manuscript under review.}

We study how conformal prediction guarantees degrade under distribution shift...
```

---

## STEP 3: OPTIONAL ENHANCEMENTS FOR PREPRINT

### Add Acknowledgments

**Add before references** (after Conclusion, before `\bibliography{references}`):

```latex
\section*{Acknowledgments}

This research was conducted at the Korea Advanced Institute of Science and Technology (KAIST). The author thanks [colleagues/advisors if any] for helpful discussions. Computational resources were provided by [institution/lab if applicable].

% Add funding if you have any:
% This work was supported by [funding source, grant number].
```

### Add Code Availability

```latex
\section*{Code and Data Availability}

The code for all experiments is available at: \url{https://github.com/[your-username]/conformal-covid}

We use publicly available datasets: rel-salt, rel-trial, and rel-f1 from the RelBench benchmark \cite{...}.
```

### Add ORCID (Optional)

**In author block**:
```latex
\author{Chorok Lee}
\authornote{ORCID: 0000-0000-0000-0000}  % Add your ORCID if you have one
\email{choroklee@kaist.ac.kr}
```

---

## STEP 4: BUILD PREPRINT PDF

```bash
cd /Users/i767700/Github/ai-in-finance/papers/conformal_covid

# Clean previous build
rm -f main.aux main.bbl main.blg main.out main.log

# Build PDF (3 passes)
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex

# Verify output
ls -lh main.pdf
```

---

## STEP 5: VERIFICATION CHECKLIST

### Content
- [ ] Your name appears on title page (not "Anonymous")
- [ ] KAIST affiliation appears
- [ ] Email address visible
- [ ] Abstract clear and complete
- [ ] All 6 contributions listed in introduction
- [ ] All 7 conclusions listed

### Figures (Critical!)
- [ ] Figure 1 (page 2): Clean, no issues
- [ ] Figure 2 (page 3): Title complete or removed
- [ ] Figure 3 (page 4): Title complete or removed, x-axis complete
- [ ] Figure 4 (page 5): Says "10-200×" not "10-100×"
- [ ] All figures > 300 DPI
- [ ] All figures have proper captions

### Quality
- [ ] 6 pages total
- [ ] PDF compiles without errors
- [ ] All references resolve (6 citations)
- [ ] No "Anonymous" anywhere in PDF
- [ ] Professional appearance

### Metadata (Check PDF properties)
- [ ] Title: "Conformal Prediction Under Distribution Shift..."
- [ ] Author: "Chorok Lee"
- [ ] No anonymous markers in metadata

---

## COMPLETE PREPRINT PREPARATION SCRIPT

**Save this and run it all at once:**

```bash
#!/bin/bash
# Complete preprint preparation script

cd /Users/i767700/Github/ai-in-finance/papers/conformal_covid

echo "Step 1: Fixing figure issues..."
cd code

# Comment out problematic titles
echo "Editing create_figure3_shap.py..."
sed -i.bak '214,215s/^/# /' create_figure3_shap.py

echo "Editing plot_retraining_results.py..."
sed -i.bak '110s/^/# /' plot_retraining_results.py

echo "Updating generate_figures.py..."
sed -i.bak 's/10-100×/10-200×/g' generate_figures.py

# Regenerate figures
echo "Regenerating figures..."
python create_figure3_shap.py
python plot_retraining_results.py
python generate_figures.py

cd ..

echo "Step 2: De-anonymizing for preprint..."
# Change documentclass
sed -i.bak 's/\\documentclass\[sigconf,anonymous\]/\\documentclass[sigconf]/' main.tex

# Note: You'll need to manually edit the author section
# Or use this sed command (complex):
echo "MANUAL STEP: Edit lines 23-26 to add your name and KAIST affiliation"

echo "Step 3: Rebuilding PDF..."
rm -f main.aux main.bbl main.blg main.out main.log
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex

echo "Done! Check main.pdf"
ls -lh main.pdf
```

---

## PREPRINT SUBMISSION GUIDELINES

### For arXiv:

1. **Category**: stat.ML (Machine Learning - Statistics)
   - Secondary: cs.LG (Machine Learning - CS)
   - Secondary: stat.AP (Applications)

2. **Title**: Keep as is: "Conformal Prediction Under Distribution Shift: A COVID-19 Natural Experiment"

3. **Abstract**: Copy from LaTeX (without `\textit{Note:...}` if added)

4. **Comments** (arXiv field):
   ```
   6 pages, 4 figures, 7 tables. Manuscript under review at UAI 2026.
   ```

5. **Files to upload**:
   - main.tex
   - references.bib
   - All figure files (figure1*.png, figure2*.png, figures/, results/)
   - ACM class files (acmart.cls, etc.)

6. **DOI**: arXiv will assign after submission

### For Papers With Code:

1. Link arXiv paper
2. Add code repository link
3. Tag with:
   - Conformal Prediction
   - Uncertainty Quantification
   - Distribution Shift
   - COVID-19

---

## DIFFERENCES: PREPRINT vs CONFERENCE SUBMISSION

| Aspect | Preprint (arXiv) | Conference (UAI) |
|--------|------------------|------------------|
| Author | ✅ Real name | ❌ Anonymous |
| Affiliation | ✅ KAIST | ❌ Hidden |
| Email | ✅ Visible | ❌ Hidden |
| Documentclass | `sigconf` | `sigconf,anonymous` |
| Acknowledgments | ✅ Optional | ❌ Not allowed |
| Code links | ✅ Encouraged | ⚠️ Can break anonymity |
| Timeline | Immediate | Review cycle (3-6 months) |
| Citable | ✅ Yes (arXiv ID) | ⚠️ After acceptance |

---

## RECOMMENDED: DUAL STRATEGY

Many researchers do both:

1. **Submit to arXiv first** (preprint with your name)
   - Establishes priority
   - Makes work immediately citable
   - Builds visibility

2. **Then submit to UAI 2026** (anonymous version)
   - Keep anonymous version separate
   - Use git branch: `git checkout -b uai-submission`
   - Maintain two versions until conference decision

3. **After UAI acceptance** (if accepted)
   - Update arXiv with "To appear at UAI 2026"
   - Submit camera-ready to conference
   - Both versions coexist

---

## TIMELINE SUGGESTION

**Today (2025-12-26)**:
- Fix figure issues (30 min)
- De-anonymize for preprint (10 min)
- Build and verify PDF (10 min)
- **Upload to arXiv** ✅

**Next Week**:
- Create anonymous version for UAI 2026
- Submit to conference before deadline
- Both versions now public/submitted ✅

**Advantages**:
- arXiv preprint gets you cited immediately
- Conference submission for peer review
- No conflicts (many conferences allow this)

---

## QUICK START FOR PREPRINT

**If you want to do this right now (50 minutes total)**:

```bash
# 1. Fix figures (30 min)
cd code
# Edit the 3 files as shown above
python create_figure3_shap.py
python plot_retraining_results.py
python generate_figures.py
cd ..

# 2. De-anonymize (10 min)
# Edit main.tex:
# - Line 2: Remove "anonymous"
# - Lines 23-26: Add your name/KAIST
# - Optional: Add acknowledgments

# 3. Build PDF (5 min)
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex

# 4. Verify (5 min)
open main.pdf  # Check everything looks good

# 5. Upload to arXiv!
```

---

## FINAL CHECKLIST FOR PREPRINT

- [ ] Figures fixed (no text cutoffs)
- [ ] Name: Chorok Lee visible
- [ ] Affiliation: KAIST visible
- [ ] Email: choroklee@kaist.ac.kr visible
- [ ] No "Anonymous" anywhere
- [ ] PDF looks professional
- [ ] 6 pages
- [ ] Ready to upload to arXiv ✅

---

**Want me to help you execute these changes now to create the preprint version?**
