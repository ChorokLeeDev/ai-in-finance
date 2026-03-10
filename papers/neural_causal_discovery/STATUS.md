# Status

## Current Assessment: 7/10 (Accept)

### Recent Progress
1. ✅ Bootstrap validation (20 trials): Neural F1=0.845 [0.831, 0.860], Linear F1=0.484 [0.466, 0.501]
2. ✅ Bootstrap p-value < 10^-6 (highly significant)
3. ✅ PCMCI baseline added (F1=0.680)
4. ✅ Economic significance: Neural Sharpe=0.77 vs Linear=0.53 (Neural wins 65% of runs)
5. ✅ Multiple datasets: synthetic, crypto, Fama-French

### What Was Achieved
- **Synthetic**: Neural beats all baselines (PCMCI, NOTEARS, Linear) with p < 10^-6
- **Crypto alignment**: +201% improvement (p=0.005)
- **Crypto prediction**: +7.2% improvement (p=0.0001)
- **Economic**: Neural wins 65% of trading runs (not statistically significant: p=0.21)

### Honest Assessment: Why NOT Strong Accept
1. **Economic significance not statistically significant** (p=0.21) - only 65% win rate
2. **Crypto ground truth is weak** - market-structure priors, not causal identification
3. **ANLG adds little value** - doesn't dominate either pure method
4. **Missing VARLiNGAM** - computational constraints

### What Would Push to Strong Accept (8+)
Per reviewer feedback:
1. Real causal ground truth (intervention data or natural experiments)
2. Statistically significant economic results (p < 0.05)
3. VARLiNGAM comparison
4. Additional real-world domains (high-frequency, options)

### Realistic Conclusion
Paper provides a **solid empirical contribution** suitable for ICAIF Accept:
- Rigorous bootstrap validation with 95% CI
- Three baselines (PCMCI, NOTEARS, Linear)
- Clear practical guidance
- Honest limitations

For Strong Accept, would need either:
- True causal ground truth from interventions
- Statistically significant trading improvement
- Major methodological contribution

**Current ceiling: Accept (7/10)**
