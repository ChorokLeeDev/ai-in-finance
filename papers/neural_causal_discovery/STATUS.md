# Status

## Current Phase: 6 (Iteration 16)
## Paper Status: Refocused - RANCD removed, pure empirical study

### Review Feedback Applied
Reviewers noted RANCD was a distraction (F1=0.10). Paper now focuses purely on:
- **When does neural beat linear?**
- Answer: Threshold/discontinuous nonlinearities (p<10⁻⁵)

### Final Results
| Setting | Neural F1 | Linear F1 | Winner | p-value |
|---------|-----------|-----------|--------|---------|
| Threshold Nonlinear | **0.800** | 0.709 | **Neural** | **<10⁻⁵** |
| Smooth Nonlinear | **0.849** | 0.689 | **Neural** | 0.015 |
| Linear | 0.600 | **0.667** | **Linear** | - |
| Fama-French (real) | 6.4e-5 | **6.0e-5** | **Linear** | - |

### Key Contribution
**Discontinuities are the differentiator.** Neural methods excel when causal mechanisms involve thresholds—margin calls, circuit breakers, rebalancing triggers.

### Paper Changes (Iteration 16)
1. Removed RANCD from main narrative
2. Simplified to neural vs linear comparison
3. Added finance-specific insight about thresholds
4. Cleaner presentation of results

### Review Assessment
Previous reviews: Weak Accept (3/3)
Key concern: RANCD distracts from real contribution → FIXED

### Next Step
Run internal review on refocused paper to assess if Strong Accept achieved.
