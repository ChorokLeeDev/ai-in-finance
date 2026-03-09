# Literature Review & Novelty Assessment

**Date:** March 2026
**Project:** Cross-Factor Predictability Decay (ICAIF 2026)

---

## 1. Literature Review Results (March 2026)

### Neural Granger + Finance
- **Finding:** NOVEL - No papers apply Neural Granger to factor returns
- **Key gap:** Factor cross-predictability unexplored
- **Sources:** Tank et al. (2021), Hui et al. (2025)

### Regime-conditional Neural Granger
- **Finding:** NOVEL - Two streams exist separately but never combined
- **Stream A:** Neural Granger (Tank et al. 2021) - no regime
- **Stream B:** Deep regime-switching (DS³M, Xu et al. 2021) - no Granger
- **Gap:** No work combines regime-switching with neural Granger

### Factor Decay Prediction (MOST IMPORTANT)
- **Finding:** VERY NOVEL - No prospective prediction exists
- **Existing work:**
  - Falck et al. (2021) - explains WHY (post-hoc), not WHEN
  - McLean & Pontiff (2016) - average decay, no individual prediction
- **Our approach:** Predict WHEN using survival analysis (C-index 0.93)

### Nonlinear Granger SOTA
- **Baseline:** Tank et al. (2021) cMLP/cLSTM
- **Emerging:** Attention mechanisms
- **Gap:** Finance applications very limited

---

## 2. Our ML Results

| Task | Metric | Value |
|------|--------|-------|
| Decay Prediction | C-index | 0.933 |
| Decay Prediction | AUC | 0.933 |
| Online Regime | Batch Agreement | 95.9% |
| Online Regime | Detection Delay | 0 days |
| Neural Granger | Status | In progress (MLP worse than Linear so far) |

**Key insight:** "High initial F-stat → higher decay probability" (regression to mean)

---

## 3. Recommended Paper Framing

- **Main contribution:** Factor Decay Prediction (prospective, not post-hoc)
- **Secondary:** Regime-conditional analysis
- **Differentiator:** "When will decay happen?" vs "Why did it happen?"

---

## 4. Key Citations to Add

| Citation | Reference | Relevance |
|----------|-----------|-----------|
| Falck, Rej, Thesmar (2021) | arXiv:2105.01380 | Post-hoc decay explanation |
| McLean & Pontiff (2016) | Journal of Finance | Average decay rates |
| Tank et al. (2021) | IEEE TPAMI | Neural Granger baseline |
| Xu et al. (2021) | DS³M | Deep regime-switching |

---

## 5. Next Steps for Future Sessions

- [ ] Reframe paper title and abstract
- [ ] Add decay prediction as main ML contribution
- [ ] Update related work section
- [ ] Run full Neural Granger comparison
- [ ] Pre-break OOS validation (Train: 1990-1995, Test: 1996-2007)

---

## 6. Current OOS Challenge & Solution

### Problem
The primary finding (HML→SMB decay) does not replicate OOS (2013-2024).

### Root Cause
- Train: 1990-2012 (includes decay period)
- Test: 2013-2024 (post-decay, signal already gone)
- Result: OOS failure is expected if decay is real

### Solution: Pre-break OOS Design
- Train: 1990-1995 (before June 1998 break)
- Test: 1996-2007 (during decay but signal still exists)
- Expected: OOS SUCCESS if decay hypothesis is correct

---

## 7. Verified Key Statistics

| Statistic | Value |
|-----------|-------|
| sup-F | 22.1 |
| Bootstrap CI | 1993-2014 (21 years) |
| In-sample p | 8.75 × 10⁻⁹ |
| Breakpoint p | 1.23 × 10⁻¹³ |
| Half-life | 3.35 years |
| OOS HML→SMB | FAIL (p > 0.05) |
| International OOS | 2/4 Bonferroni survive |
