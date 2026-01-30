# Reframing "Mechanism Discovery" → "Predictive Signal"

## Issue
Current framing is too strong: "Mechanism Discovery" implies causal understanding.
Reality: We have **correlation** and a **hypothesis**, not proven mechanism.

## Required Changes

### 1. Abstract (Line 34)

**BEFORE:**
> **Mechanism discovery**: Using SHAP analysis, we reveal that catastrophic failures appear to stem from *single-feature dependence*...

**AFTER:**
> **Predictive signal**: Using SHAP analysis, we identify that catastrophic failures are associated with *single-feature dependence* patterns...

**Or even better:**
> **Mechanistic hypothesis**: SHAP analysis reveals that catastrophic failures correlate with *single-feature dependence* (ρ=0.71, p=0.047), suggesting importance concentration as a predictive signal...

---

### 2. Introduction - Contribution #2 (Line 67-69)

**BEFORE:**
> (2) **Mechanism Discovery**: Using SHAP analysis, we reveal that catastrophic failures appear to stem from single-feature dependence rather than feature instability...

**AFTER:**
> (2) **Mechanistic Hypothesis**: Using SHAP analysis, we identify single-feature dependence as a predictive signal for catastrophic failure (Spearman ρ=0.71, p=0.047). This correlation persists even when both robust and catastrophic tasks exhibit ∼0% feature overlap, suggesting importance dynamics rather than feature stability determine vulnerability.

---

### 3. Section 4.4 Title (Line 249)

**BEFORE:**
> \section{Feature Importance Analysis}

**AFTER:**
> \section{Feature Importance Dynamics as Predictive Signal}

Or:
> \section{Identifying Vulnerability: Feature Importance Concentration}

---

### 4. Section 4.4 - "Mechanism Insight" (Line 274)

**BEFORE:**
> **Mechanism insight:** Analysis of contrasting task pairs suggests catastrophic failure is associated with *single-feature dependence* that breaks under distribution shift...

**AFTER:**
> **Hypothesis**: Analysis of contrasting task pairs suggests catastrophic failure correlates with *single-feature dependence* that breaks under distribution shift. While we cannot establish causality, the pattern is consistent across tasks: models that concentrate importance in one feature (>40%) without stable backup features show higher vulnerability...

**Add paragraph:**
> **Causal interpretation caveats:** This analysis identifies a predictive correlation but does not establish causality. Alternative explanations (e.g., prediction confidence changes, feature interaction effects) have not been ruled out. Future work should include causal validation through synthetic interventions or ablation studies.

---

### 5. Conclusion (Line 515)

**BEFORE:**
> (2) **Mechanism discovery**: Analysis across all 8 tasks shows catastrophic failures stem from single-feature dependence...

**AFTER:**
> (2) **Predictive signal identification**: Analysis across all 8 tasks shows catastrophic failures correlate with single-feature dependence (SHAP concentration >40%, ρ=0.71, p=0.047). This suggests importance concentration as a pre-deployment diagnostic, though causal mechanisms require further validation.

---

## Why This Matters

### Scientific Rigor
- "Mechanism" implies proven causation
- We have: correlation, not causation
- Honest framing prevents overclaiming

### Reproducibility
- Other researchers might test on different models/domains
- If they don't replicate, it undermines trust
- Better to say "predictive signal" that needs validation

### Practical Impact
- Practitioners might over-rely on 40% threshold
- "Signal" encourages empirical monitoring
- "Mechanism" implies certainty we don't have

---

## Word Substitutions Throughout

| Current | Replace with | Context |
|---------|-------------|---------|
| "mechanism discovery" | "mechanistic hypothesis" | When introducing the finding |
| "mechanism" | "pattern" / "signal" | When describing correlation |
| "stems from" | "correlates with" / "is associated with" | Causal language |
| "reveals" | "suggests" / "indicates" | Strength of evidence |
| "explains why" | "helps predict when" | Purpose |

---

## Additional Strengthening

### Add Causal Validation Section (Future Work)

Could add to Discussion:

```latex
\subsection{Toward Causal Validation}

While our analysis identifies SHAP concentration as a predictive signal (ρ=0.71),
establishing causality requires additional validation:

\textbf{Proposed experiments:}
\begin{enumerate}
    \item \textbf{Synthetic intervention}: Artificially concentrate importance via
    feature subset selection. If concentration causes vulnerability, manually
    concentrating importance should increase coverage degradation.

    \item \textbf{Ablation study}: Remove the dominant feature entirely. If
    single-feature dependence is causal, removing it should improve robustness.

    \item \textbf{Model architecture comparison}: Test whether the pattern holds
    for neural networks. If importance dynamics are model-specific (LightGBM artifact),
    the signal may not generalize.

    \item \textbf{Temporal validation}: Track importance concentration over time
    pre-shift. If concentration increases before degradation, this strengthens
    the causal hypothesis.
\end{enumerate}

Until such validation is completed, practitioners should treat the 40\% threshold
as an empirical guideline requiring domain-specific validation.
```

---

## Summary of Changes

**What we HAVE:**
- ✓ Strong correlation (ρ=0.71, p=0.047; ρ=0.89 excluding outlier)
- ✓ Consistent pattern across 8 tasks
- ✓ Plausible mechanistic hypothesis
- ✓ Actionable predictive signal

**What we DON'T have:**
- ✗ Causal proof
- ✗ Theory explaining WHY concentration matters
- ✗ Validation on other model types
- ✗ Ruled out alternative explanations

**Better framing:**
> "We identify SHAP importance concentration as a predictive signal for conformal
> prediction vulnerability under distribution shift, and propose single-feature
> dependence as a mechanistic hypothesis requiring further validation."

This is honest, scientifically rigorous, and still impactful.
