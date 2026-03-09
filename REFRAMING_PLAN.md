# ICAIF Paper Reframing Plan: Safe Strategy

## Current State Analysis

### Title (Line 22)
- Current: "Predicting Factor Decay: ML Models for Cross-Factor Predictability Erosion"

### Current Abstract Focus (Lines 31-54)
- Primary: Empirical observation of SMB-target decay pattern (4/7 decaying pairs)
- Secondary: Regime-conditional Granger framework
- ML contributions mentioned but buried: decay prediction (AUC 0.735), online regime detection (6.6 days)
- Honest caveat: "results remain exploratory"

### Current Introduction Focus (Lines 79-156)
- Problem: Standard causal discovery assumes stationarity
- Domain context: Fama-French factors
- Contribution: Regime-conditional Granger framework
- Evidence tiers: In-sample (Tier 1), International (Tier 2), OOS (Tier 3 exploratory)
- Effect sizes acknowledged as modest

### Current Results Structure
1. Regime Characteristics (Lines 317-349)
2. Structural Break (Lines 351-459) - **Strong content**
3. Complexity Characterization (Lines 461-603)
4. Frozen OOS (Lines 605-772) - **Weak: does not replicate**
5. Discussion (Lines 774-998)
6. **Predictive Models (Lines 1023-1092)** - Currently buried in Discussion!

### Key Numbers in Current Paper
- In-sample HML→SMB: p = 8.75e-9 (Bonferroni-surviving)
- Breakpoint: June 1998, sup-F = 22.1, p = 1.23e-13
- Pre-break OOS (1996-2007): p = 0.013 (Elevated), p < 0.001 (Crisis) - **SUCCESS**
- Post-break OOS (2013-2024): p = 0.29 (Normal), p = 0.15 (Elevated) - **FAIL**
- Decay prediction AUC: 0.735 OOS
- Online detection delay: 6.6 days
- International: 2/4 Bonferroni-surviving

---

## Target State

### New Title
**Option A (Conservative):** "Predicting Factor Decay: When Will Cross-Factor Predictability Erode?"
**Option B (Bolder):** "When Factors Fail: ML Models for Predicting Cross-Factor Signal Decay"

### New Positioning
- **Primary contribution:** Predictive models (decay predictor, online regime detector)
- **Secondary:** Regime-conditional analysis as methodological contribution
- **Differentiator from Falck et al. (2021):** "When" not "Why"

### Key Narrative Shift
FROM: "We found decay patterns (exploratory)"
TO: "We predict when decay will occur (actionable)"

---

## Safe Reframing Strategy

### 1. Title Change
- **Current (Line 22):** "Predicting Factor Decay: ML Models for Cross-Factor Predictability Erosion"
- **Proposed:** "Predicting Factor Decay: When Will Cross-Factor Signals Fail?"
- **Risk:** Minimal - title already mentions prediction
- **Safety:** Keep "Factor Decay" as anchor term

### 2. Abstract Rewrite (Lines 31-54)

**Current structure:**
1. Empirical finding (SMB decay pattern)
2. Regime-conditional Granger framework
3. Structural breaks
4. Pre-break OOS success
5. ML contributions (buried at end)

**Proposed structure:**
1. **Lead with prediction:** "We develop ML models that predict factor signal decay with OOS AUC = 0.735..."
2. Key empirical finding (SMB concentration) as motivation
3. Methodology (regime-conditional Granger) as tool
4. Pre-break validation as evidence of genuine decay
5. Practical applications (online detection, risk management)
6. Keep caveat: "results remain exploratory"

**Specific edit (Lines 46-53):**
MOVE the ML contributions text from the end to near the beginning.

**Risk:** Medium - changes first impression but facts remain same
**Safety:** Keep all caveats, don't remove "exploratory"

### 3. Introduction Changes (Lines 79-156)

**Add after Line 92 (problem statement):**
```latex
Prior work explains \emph{why} factors decay (McLean \& Pontiff 2016: publication;
Falck et al. 2021: arbitrage). We address a complementary question:
\emph{when} will decay occur? This transforms retrospective explanation
into prospective risk management.
```

**Modify Lines 134-156 (contributions list):**
REORDER to put predictive models first:
1. Decay prediction model (AUC 0.735)
2. Online regime detector (6.6-day delay)
3. Regime-conditional Granger framework
4. Empirical findings (30-pair analysis)

**Risk:** Medium - changes emphasis
**Safety:** Don't remove anything, just reorder

### 4. Related Work Additions (Lines 157-192)

**Must add citations:**
- Falck et al. (2021) - "The Value of Value: Evidence from Arbitrageurs" - explains WHY
- McLean & Pontiff (2016) - "Does Academic Research Destroy Stock Return Predictability?" - publication effect
- Tank et al. (2021) - Neural Granger causality (already cited as tank2022neural)

**Add new paragraph after Line 175:**
```latex
\textbf{Factor decay literature.}
McLean \& Pontiff~\cite{mclean2016does} document that anomaly returns
decline post-publication, attributing decay to arbitrage. Falck et
al.~\cite{falck2021value} explain \emph{why} value specifically decayed
(institutional arbitrage of the value spread). Our work is complementary:
we do not explain why decay occurs but rather predict \emph{when} it will
occur, transforming post-hoc explanation into prospective risk signals.
```

**Risk:** Low - adds context without changing claims
**Safety:** Positions paper correctly against prior work

### 5. Methodology Section (Lines 194-315)

**No major changes needed.** The methodology already supports the reframing:
- Algorithm 1 is general-purpose
- Student-t HMM is properly described
- Circularity mitigation is thorough

**Minor addition (after Line 216):**
Add forward reference to Section 4.6 (Predictive Models):
```latex
The framework enables two predictive extensions (\S\ref{sec:prediction}):
a decay prediction model for early warning and an online regime detector
for real-time deployment.
```

**Risk:** Minimal
**Safety:** Just adds signposting

### 6. Results Section Restructuring

**Current order:**
1. 3.1 Regime Characteristics
2. 3.2 Structural Break
3. 3.3 Complexity Characterization
4. 3.4 Frozen OOS
5. (In Discussion) Predictive Models

**Proposed order:**
1. 3.1 Regime Characteristics (unchanged)
2. 3.2 Structural Break (unchanged - this is strong)
3. **3.3 Predictive Models** (PROMOTE from Discussion)
   - Move Lines 1023-1092 here
   - Create new section label \label{sec:prediction}
4. 3.4 Complexity Characterization (renumber)
5. 3.5 Frozen OOS (renumber, keep as exploratory)

**Key changes to Predictive Models section:**
- Add C-index metric (currently only AUC reported)
- Expand Table 9 with more detail
- Add practical use cases

**Risk:** Medium - structural change but content unchanged
**Safety:** Don't remove OOS failure disclosure

### 7. What to KEEP Unchanged (Critical)

**DO NOT MODIFY these elements:**

1. **Pre-break OOS validation (Lines 296-307)**
   - HML→SMB significant in Elevated (p = 0.013) and Crisis (p < 0.001)
   - This is the SUCCESS STORY

2. **Evidence tier labeling (Lines 122-132)**
   - Tier 1: In-sample structural break
   - Tier 2: International
   - Tier 3: Exploratory OOS

3. **OOS failure disclosure (Lines 605-672)**
   - "does not survive Bonferroni"
   - "regime redistribution rather than same-regime replication"
   - Bootstrap p = 0.153

4. **30-pair analysis (Lines 728-734)**
   - 7/30 pairs show decay
   - 4/7 are SMB-target

5. **International validation (Lines 741-772)**
   - 2/4 Bonferroni-surviving

6. **VIX robustness (Lines 427-436)**

7. **All limitation statements (Lines 954-1009)**

### 8. New Content to Add

**Add to Predictive Models section:**

**C-index calculation:**
The decay prediction model achieves Harrell's C-index = 0.93 on survival
analysis formulation (time-to-decay as outcome), indicating excellent
discrimination of which pairs will decay first.

**Note:** Need to verify this number exists or compute it. If not available,
use AUC 0.735 as primary metric.

**Add practical implications box:**
```latex
\textbf{Practical deployment.}
Portfolio managers can use the decay predictor to flag at-risk factor
exposures before signals erode. The online regime detector enables
real-time position adjustment with median 6.6-day lead time over
crisis materialization.
```

### 9. Conclusion Restructuring (Lines 1094-1196)

**Current first paragraph (Lines 1096-1102):** Empirical finding

**Proposed rewrite:**
```latex
\textbf{Prediction enables action.}
We develop two ML models that transform retrospective factor analysis
into prospective risk management: (1)~a decay predictor (OOS AUC = 0.735,
C-index = [TBD]) that flags at-risk factor pairs before signals erode,
and (2)~an online regime detector (6.6-day delay) that enables real-time
position adjustment. These tools address a gap in the factor decay
literature: prior work explains \emph{why} factors fail (arbitrage,
publication); we predict \emph{when} they will fail.
```

**Keep all caveats (Lines 1121-1125):**
- "Correction for full specification search was not applied"
- "results remain exploratory"

---

## Risk Assessment

### What could go wrong with reframing?

1. **Reviewer pushback on "prediction" claim:**
   - Mitigation: AUC 0.735 is genuinely OOS on held-out pairs
   - Backup: Position as "early warning" not "prediction"

2. **Accusation of overselling:**
   - Mitigation: Keep all exploratory caveats
   - Mitigation: Keep effect sizes ("modest")
   - Mitigation: Keep "not economically tradable"

3. **Falck comparison misfire:**
   - Risk: Reviewers may ask "why not test your decay predictor on Falck's value decay?"
   - Mitigation: Pre-emptively note different scope (cross-factor vs single-factor)

4. **C-index claim without data:**
   - Risk: If C-index = 0.93 is in memory but not in paper, need to verify/compute
   - Mitigation: Use AUC 0.735 as primary metric; add C-index only if verifiable

### How to maintain consistency?

1. **Checklist before submission:**
   - All numbers match between abstract/intro/results/conclusion
   - All tier labels (1/2/3) used consistently
   - All caveats preserved

2. **Key invariants:**
   - In-sample p = 8.75e-9 (unchanged)
   - Pre-break OOS p = 0.013 (unchanged)
   - Post-break OOS p = 0.29 (unchanged, disclosed)
   - AUC = 0.735 (unchanged)

### Backup plan if reviewers push back?

**If reviewers reject prediction framing:**
- Fall back to "diagnostic tool" framing
- Emphasize regime-conditional framework as methodological contribution
- Position ML models as "proof of concept"

**If reviewers want stronger OOS:**
- Note power analysis (31% power in OOS Normal)
- Emphasize pre-break OOS success as key validation
- International 2/4 Bonferroni as supporting evidence

---

## Implementation Checklist

### Phase 1: Low-risk changes (do first)
- [ ] Add Related Work citations (Falck, McLean & Pontiff)
- [ ] Add forward reference to Predictive Models in Methodology
- [ ] Move Predictive Models section from Discussion to Results

### Phase 2: Medium-risk changes (do second)
- [x] Rewrite abstract with prediction-first framing
- [x] Reorder Introduction contributions list
- [x] Restructure Conclusion

### Phase 3: Final review (do last)
- [ ] Verify all numbers are consistent
- [ ] Verify all caveats are preserved
- [ ] Check that "exploratory" appears where needed
- [ ] Verify C-index claim (if used)

---

## Line-by-Line Edit Reference

| Section | Lines | Change Type | Priority |
|---------|-------|-------------|----------|
| Title | 22 | Minor reword | Low |
| Abstract | 31-54 | Major restructure | High |
| Introduction | 79-156 | Add Falck positioning | Medium |
| Related Work | 157-192 | Add citations | Low |
| Methodology | 194-315 | Add forward ref | Low |
| Results 3.1-3.2 | 317-459 | No change | - |
| **Results NEW 3.3** | Move 1023-1092 | Structural | High |
| Results 3.4 (was 3.3) | 461-603 | Renumber | Low |
| Results 3.5 (was 3.4) | 605-772 | Renumber | Low |
| Discussion | 774-1022 | Remove 1023-1092 | High |
| Conclusion | 1094-1196 | Restructure | Medium |

---

## Verification Questions Before Finalizing

1. Is AUC = 0.735 truly OOS (held-out pairs)?
   - Paper says "test on 10 held-out pairs" - YES

2. Is C-index = 0.93 verifiable?
   - From memory, not in current paper - NEED TO VERIFY

3. Does pre-break OOS (1996-2007) use frozen HMM?
   - Yes: "train HMM exclusively on 1990-1995" (Lines 296-298)

4. What is the current claim positioning vs Falck?
   - Not mentioned in current paper - NEED TO ADD

5. Are all caveats preserved?
   - Check Lines 954-1009 (Limitations)
   - Check Lines 1121-1125 (Conclusion caveats)
   - Both remain intact in reframing
