# Tier 2 Journal Upgrade Plan

## Target Journals
- Journal of Financial and Quantitative Analysis (JFQA)
- Review of Finance (RoF)
- Management Science (Finance track)

## Current State Assessment

### Strengths (Keep)
- Pre-break OOS success (novel validation design)
- 30-pair systematic analysis
- SMB-target concentration finding (4/7 pairs)
- Rigorous methodology (HAC, Bonferroni, bootstrap CI)
- Multiple robustness checks (VIX, holdout, international)

### Weaknesses (Must Fix)

| Weakness | Current | Tier 2 Requirement |
|----------|---------|-------------------|
| Economic significance | Sharpe = -0.07 | Positive Sharpe |
| Practical value | VaR p=0.083 (not sig) | Significant improvement |
| Sample scope | US primary | International primary |
| Effect size | ΔR² = 2% | Show why 2% matters |

---

## Improvement Plan

### 1. Economic Significance: Trading Strategy with Positive Sharpe

**Problem**: Current paper shows Sharpe = -0.07 (negative)

**Solution**: Regime-conditional trading strategy
- Long SMB in Normal regime (when signal exists)
- Reduce/hedge SMB in Crisis regime
- Use regime prediction to time factor exposure

**Implementation**:
```python
# Pseudo-strategy
if regime == "Normal" and pre_1998:
    position = long_SMB_when_HML_up
elif regime == "Crisis":
    position = neutral_or_short
else:
    position = neutral

# Backtest 1990-2024 with transaction costs
```

**Target**: Sharpe > 0.3 (modest but positive)

---

### 2. Risk Management: Significant VaR Improvement

**Problem**: Christoffersen p = 0.083 (not significant at 5%)

**Solution**: Expand VaR analysis
- Test multiple confidence levels (1%, 2.5%, 5%, 10%)
- Test multiple assets (not just SMB)
- Use regime-conditional Expected Shortfall (ES)
- Longer backtest period

**Target**: At least one specification with p < 0.05

---

### 3. International Analysis as Primary

**Problem**: US is primary, international is supplementary (2/4 OOS survive)

**Solution**: Reframe paper
- "Global evidence of factor predictability decay"
- US + 4 international markets = 5 markets primary
- Show SMB-target pattern holds internationally

**Current international results**:
- Asia-Pacific: OOS Bonferroni survive ✓
- Developed ex-US: OOS Bonferroni survive ✓
- Europe: In-sample only
- Japan: In-sample only

**New framing**: "Cross-market evidence with 3/5 markets showing OOS significance"

---

### 4. Effect Size: Explain Why 2% Matters

**Problem**: ΔR² = 2% seems small

**Solution**: Economic translation
- 2% daily R² → annualized Sharpe contribution
- Compare to other known predictors
- Show cumulative P&L impact over time

**Calculation**:
```
Daily ΔR² = 2% = 0.02
Daily IC ≈ sqrt(0.02) ≈ 0.14
Annualized IR ≈ 0.14 × sqrt(252) ≈ 2.2

But this assumes perfect trading, which doesn't work.
Need realistic simulation.
```

---

## Implementation Priority

| Task | Effort | Impact | Priority |
|------|--------|--------|----------|
| Trading strategy with positive Sharpe | High | Critical | **1** |
| Expand VaR analysis | Medium | High | **2** |
| International reframing | Low | Medium | **3** |
| Effect size explanation | Low | Medium | **4** |

---

## Success Criteria for Tier 2

- [ ] Trading strategy Sharpe > 0.2 (after costs)
- [ ] At least one VaR spec with p < 0.05
- [ ] International markets as co-primary evidence
- [ ] Clear "so what" answer for practitioners

---

## Risk Assessment

| If we achieve... | Tier 2 probability |
|------------------|-------------------|
| All 4 criteria | 40-50% |
| 3 of 4 | 25-35% |
| 2 of 4 | 15-20% (current) |
| 1 of 4 | <10% |

---

## Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Trading strategy | 1-2 weeks | Backtest results |
| VaR expansion | 1 week | Extended analysis |
| Paper revision | 1 week | New draft |
| Internal review | 1 week | Final polish |
| **Total** | **4-6 weeks** | Tier 2 ready paper |
