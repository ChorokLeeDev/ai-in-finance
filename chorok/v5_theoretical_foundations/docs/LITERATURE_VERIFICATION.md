# Literature Verification: Is Our Gap Real?

**Created**: 2025-12-09
**Purpose**: Honest verification after finding Dynamic Shrinkage literature

---

## What We Found That DOES Exist

### 1. Dynamic Shrinkage Processes (Kowal et al., JRSS-B 2019)

**Paper**: [Dynamic Shrinkage Processes](https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/rssb.12325)

**What they do**:
- Time-varying shrinkage for time series
- Shrinkage adapts SMOOTHLY over time
- Local scale parameters depend on history of shrinkage process
- Focus: Trend filtering, local features in time series

**What they DON'T do**:
- NOT about distribution shift in the data generating process
- NOT about when historical data becomes "poison"
- NOT about cross-sectional hierarchical models

### 2. Dynamic Triple Gamma Prior (Knaus & Frühwirth-Schnatter, 2024)

**Paper**: [arXiv:2312.10487](https://arxiv.org/abs/2312.10487)

**What they do**:
- Time-dependent shrinkage for time-varying parameter (TVP) models
- Handles "large jumps" in parameters
- Flexible shrinkage that adapts over time

**What they DON'T do**:
- NOT about when borrowing from historical data hurts
- NOT about distribution shift
- Focus is on PARAMETER estimation, not hierarchical borrowing

### 3. Locally Adaptive Shrinkage (Technometrics, 2024)

**Paper**: [arXiv:2309.00080](https://arxiv.org/abs/2309.00080)

**What they do**:
- Adaptive shrinkage for structural breaks in count time series
- Global-local shrinkage process for trend filtering
- Captures local transient features

**What they DON'T do**:
- NOT about cross-sectional hierarchical models
- NOT about when to stop borrowing from history

### 4. Local Empirical Bayes (2024)

**Paper**: [arXiv:2511.21282](https://arxiv.org/html/2511.21282)

**What they do**:
- Neighborhood-based shrinkage
- Handle heterogeneity ACROSS experiments
- Replace global pooling with local neighborhoods

**What they DON'T do**:
- NOT about TEMPORAL neighborhoods
- NOT about distribution shift within same system over time

### 5. Bayesian Cluster Hierarchical Model (BCHM)

**What they do**:
- Cluster similar subgroups in clinical trials
- Adaptive borrowing based on similarity
- Handle heterogeneous treatment effects

**What they DON'T do**:
- NOT about temporal distribution shift
- NOT about when historical data becomes wrong

---

## The Refined Gap (After Verification)

### What IS Covered

```
TIME-VARYING SHRINKAGE (within continuous time series)
├── Dynamic Shrinkage Processes (2019)
├── Dynamic Triple Gamma (2024)
└── Locally Adaptive Shrinkage (2024)

→ Focus: How shrinkage should vary SMOOTHLY over time
→ Setting: Continuous time series, parameter estimation
→ NOT about: Regime change, distribution shift
```

```
HETEROGENEOUS GROUPS (cross-sectional)
├── Local Empirical Bayes (2024)
├── BCHM for clinical trials
└── Individual Shrinkage (2023)

→ Focus: Different groups need different shrinkage
→ Setting: Cross-sectional heterogeneity
→ NOT about: Temporal distribution shift
```

### What is NOT Covered (The Actual Gap)

```
CROSS-SECTIONAL SHRINKAGE × TEMPORAL REGIME CHANGE

Scenario:
  - You have groups (FKs) sharing a hierarchical prior
  - Prior is LEARNED from historical data
  - Then distribution SHIFTS (e.g., COVID)
  - Now borrowing from historical groups may HURT

Questions nobody asks:
  1. Does the shrinkage factor learned pre-shift work post-shift?
  2. When does borrowing from historical groups become harmful?
  3. Should you use "temporal neighborhoods" - only recent data?
```

---

## The Key Distinction

| Existing Work | Our Gap |
|---------------|---------|
| Shrinkage varies smoothly over time | Shrinkage becomes WRONG after regime change |
| Within continuous time series | Across discrete temporal boundary |
| Parameter estimation | Hierarchical borrowing across groups |
| Local features in trend | Prior becomes obsolete |

### Concrete Example

**Dynamic Shrinkage (exists)**:
```
Time: t1 → t2 → t3 → t4 → t5
Shrinkage adapts smoothly: 0.3 → 0.35 → 0.4 → 0.38 → 0.42
Focus: Local signal structure
```

**Our Question (gap)**:
```
Time:  PRE-COVID ─────┬───── POST-COVID
                      │
                  REGIME CHANGE
                      │
       Prior learned  │  Prior may be WRONG
       from here      │  for this period

Question: Does shrinkage factor 0.4 learned pre-COVID
          still work post-COVID? Or does it hurt?
```

---

## Honest Assessment

### Is Our Gap Real?

**YES, but it's more specific than we originally thought.**

The gap is NOT:
- ❌ "Time-varying shrinkage" (Dynamic Shrinkage Processes covers this)
- ❌ "Adaptive shrinkage" (exists in many forms)
- ❌ "Shrinkage for heterogeneous groups" (Local EB covers this)

The gap IS:
- ✅ Cross-sectional hierarchical shrinkage under temporal regime change
- ✅ When borrowing from historical data becomes harmful
- ✅ Temporal neighborhoods for hierarchical models

### Is It Publishable?

**Likely yes, but we need to be careful about positioning.**

We CANNOT claim:
- "First to study time-varying shrinkage" (wrong)
- "First to study adaptive shrinkage" (wrong)
- "First to study shrinkage under distribution shift" (too broad)

We CAN claim:
- "First to study hierarchical shrinkage under temporal regime change"
- "First to ask when borrowing from pre-shift groups hurts post-shift"
- "Extends Local EB to temporal neighborhoods"

---

## Updated Research Direction

### Original Claim (Too Broad)

> "Nobody studies shrinkage under distribution shift"

**Problem**: Dynamic shrinkage literature exists and is related.

### Refined Claim (More Accurate)

> "Dynamic shrinkage handles smooth time-varying parameters, but not
> the discrete regime change where a hierarchical prior becomes obsolete.
> We study when borrowing from pre-shift groups hurts post-shift estimation."

### Positioning vs Existing Work

| Paper | What They Do | How We Extend |
|-------|--------------|---------------|
| Dynamic Shrinkage (2019) | Smooth time-varying shrinkage | Discrete regime change |
| Local EB (2024) | Spatial/experimental neighborhoods | Temporal neighborhoods |
| Individual Shrinkage (2023) | Individual vs aggregate accuracy | How this changes across shift |
| BCHM | Cluster similar groups | Detect when similarity breaks over time |

---

## Conclusion

**The gap is real but narrower than we thought.**

We need to:
1. Acknowledge Dynamic Shrinkage Processes literature
2. Position as EXTENDING Local EB to temporal setting
3. Focus specifically on regime change, not smooth variation
4. Use COVID as clear discrete boundary (not gradual drift)

**SALT dataset is still perfect** - it has a clear temporal boundary (COVID), not gradual drift, which is exactly the scenario NOT covered by Dynamic Shrinkage Processes.

---

## Key References to Cite

### Must Acknowledge (They're Related)
1. Kowal et al. (2019) - Dynamic Shrinkage Processes (JRSS-B)
2. Knaus & Frühwirth-Schnatter (2024) - Dynamic Triple Gamma
3. Local EB (2024) - arXiv:2511.21282
4. Individual Shrinkage (2023) - arXiv:2308.01596

### Position As Extension Of
1. Local EB → extend from spatial to temporal neighborhoods
2. Dynamic Shrinkage → extend from smooth variation to discrete regime change

---

*This document verifies the research gap after discovering Dynamic Shrinkage literature.*
*The gap is real but more specific than originally claimed.*
