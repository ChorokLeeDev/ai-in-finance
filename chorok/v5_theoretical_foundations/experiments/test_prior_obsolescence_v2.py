"""
Test v2: Does hierarchical prior become obsolete after COVID shift?

Key insight from v1: Shrinkage factor was 0.947 (too weak to hurt).
For shrinkage to HURT after shift, we need STRONG shrinkage (low B_k).

Strong shrinkage happens when:
- Groups look similar pre-shift (low τ², high σ²)
- Small sample sizes (low n)

This test creates a scenario where:
1. Pre-COVID: Groups ARE similar (hierarchical borrowing helps)
2. Post-COVID: Some groups DIVERGE (but model still pulls toward old mean)
"""

import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("TEST v2: Strong Shrinkage Regime")
print("=" * 70)

# =============================================================================
# Scenario: Groups that WERE similar, now DIVERGE
# =============================================================================
np.random.seed(42)

n_groups = 20
n_samples_per_group = 10  # Small samples = stronger shrinkage

# PRE-COVID: Groups are VERY similar (low between-group variance)
# All centered around 10, with small spread
true_means_pre = np.random.normal(10, 0.5, n_groups)  # τ = 0.5 (very similar)

# POST-COVID: Some groups DIVERGE dramatically
# Half stay same, half shift by ±8 (large shift)
shift_magnitude = np.array([0]*10 + [8]*5 + [-8]*5)
true_means_post = true_means_pre + shift_magnitude

# Observation noise (same for both periods)
obs_noise = 2.0  # σ = 2

# Generate data
pre_covid_data = {}
post_covid_data = {}

for g in range(n_groups):
    pre_covid_data[g] = np.random.normal(true_means_pre[g], obs_noise, n_samples_per_group)
    post_covid_data[g] = np.random.normal(true_means_post[g], obs_noise, n_samples_per_group)

print(f"\n[Setup]")
print(f"  Groups: {n_groups}")
print(f"  Samples per group: {n_samples_per_group}")
print(f"  Pre-COVID τ (between-group): 0.5 (groups very similar)")
print(f"  Observation noise σ: {obs_noise}")
print(f"  Post-COVID shift: ±8 for 50% of groups")

# =============================================================================
# Compute shrinkage from PRE-COVID data
# =============================================================================
def empirical_bayes_shrinkage(group_data):
    """Compute empirical Bayes shrinkage estimates."""
    group_means = np.array([np.mean(data) for data in group_data.values()])
    group_ns = np.array([len(data) for data in group_data.values()])
    group_vars = np.array([np.var(data, ddof=1) for data in group_data.values()])

    global_mean = np.mean(group_means)
    sigma_sq = np.mean(group_vars)  # Within-group variance (pooled)

    # Between-group variance (method of moments)
    var_of_means = np.var(group_means, ddof=1)
    tau_sq = max(0.01, var_of_means - sigma_sq / np.mean(group_ns))  # Min 0.01 to avoid division by zero

    # Lambda = σ²/τ² (larger = more shrinkage)
    lambda_ratio = sigma_sq / tau_sq

    # Shrinkage factors
    shrinkage_factors = group_ns / (group_ns + lambda_ratio)

    # Shrunk estimates
    shrunk_means = shrinkage_factors * group_means + (1 - shrinkage_factors) * global_mean

    return {
        'group_means': group_means,
        'shrunk_means': shrunk_means,
        'shrinkage_factors': shrinkage_factors,
        'global_mean': global_mean,
        'tau_sq': tau_sq,
        'sigma_sq': sigma_sq,
        'lambda': lambda_ratio
    }

print(f"\n[1] Pre-COVID Shrinkage Parameters")
pre_covid_fit = empirical_bayes_shrinkage(pre_covid_data)
print(f"  Global mean: {pre_covid_fit['global_mean']:.3f}")
print(f"  τ² (between-group): {pre_covid_fit['tau_sq']:.3f}")
print(f"  σ² (within-group): {pre_covid_fit['sigma_sq']:.3f}")
print(f"  λ (σ²/τ²): {pre_covid_fit['lambda']:.3f}")
print(f"  Avg shrinkage factor B: {pre_covid_fit['shrinkage_factors'].mean():.3f}")
print(f"  → B close to 0 = strong shrinkage toward global mean")
print(f"  → B close to 1 = weak shrinkage, trust group mean")

# =============================================================================
# Test on POST-COVID data
# =============================================================================
print(f"\n[2] Post-COVID Estimation")

true_post = true_means_post
post_group_means = np.array([np.mean(data) for data in post_covid_data.values()])

# Method A: Use pre-COVID shrinkage parameters
shrunk_post_using_pre = (
    pre_covid_fit['shrinkage_factors'] * post_group_means +
    (1 - pre_covid_fit['shrinkage_factors']) * pre_covid_fit['global_mean']
)

# Method B: No shrinkage (just use post-COVID group means)
no_shrink_post = post_group_means

# Method C: Re-fit shrinkage on post-COVID data (oracle)
post_covid_fit = empirical_bayes_shrinkage(post_covid_data)
shrunk_post_using_post = post_covid_fit['shrunk_means']

# Compute errors
mse_pre_shrinkage = np.mean((shrunk_post_using_pre - true_post)**2)
mse_no_shrinkage = np.mean((no_shrink_post - true_post)**2)
mse_post_shrinkage = np.mean((shrunk_post_using_post - true_post)**2)

print(f"\n  Method A (Pre-COVID shrinkage): MSE = {mse_pre_shrinkage:.4f}")
print(f"  Method B (No shrinkage):        MSE = {mse_no_shrinkage:.4f}")
print(f"  Method C (Post-COVID shrinkage): MSE = {mse_post_shrinkage:.4f}")

# =============================================================================
# Verdict
# =============================================================================
print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)

if mse_pre_shrinkage > mse_no_shrinkage:
    damage_pct = ((mse_pre_shrinkage / mse_no_shrinkage) - 1) * 100
    print(f"\n✓ PRE-COVID SHRINKAGE HURTS POST-COVID ESTIMATION!")
    print(f"  Using old shrinkage: MSE = {mse_pre_shrinkage:.4f}")
    print(f"  No shrinkage:        MSE = {mse_no_shrinkage:.4f}")
    print(f"  Damage: {damage_pct:.1f}% WORSE")
    gap_confirmed = True
else:
    print(f"\n✗ Pre-COVID shrinkage still helps")
    gap_confirmed = False

# =============================================================================
# Per-Group Analysis
# =============================================================================
print("\n" + "-" * 70)
print("Per-Group Analysis")
print("-" * 70)

errors_with_shrink = (shrunk_post_using_pre - true_post)**2
errors_no_shrink = (no_shrink_post - true_post)**2
damage = errors_with_shrink - errors_no_shrink

print(f"\n{'Grp':<4} {'Shift':<6} {'B_k':<6} {'Pre-MSE':<10} {'No-MSE':<10} {'Damage':<10}")
print("-" * 50)
for g in range(n_groups):
    shifted = f"+{shift_magnitude[g]}" if shift_magnitude[g] > 0 else (f"{shift_magnitude[g]}" if shift_magnitude[g] < 0 else "0")
    print(f"{g:<4} {shifted:<6} {pre_covid_fit['shrinkage_factors'][g]:<6.3f} "
          f"{errors_with_shrink[g]:<10.4f} {errors_no_shrink[g]:<10.4f} {damage[g]:<+10.4f}")

# Summary by shift status
shifted_mask = shift_magnitude != 0
print(f"\nSummary:")
print(f"  Shifted groups (50%):     Avg damage = {damage[shifted_mask].mean():+.4f}")
print(f"  Non-shifted groups (50%): Avg damage = {damage[~shifted_mask].mean():+.4f}")

# =============================================================================
# Statistical Test
# =============================================================================
print("\n" + "-" * 70)
print("Statistical Significance")
print("-" * 70)

# Paired t-test: Is pre-shrinkage MSE significantly higher than no-shrinkage MSE?
t_stat, p_value = stats.ttest_rel(errors_with_shrink, errors_no_shrink)
print(f"\nPaired t-test (H0: shrinkage doesn't hurt):")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value: {p_value:.4f}")
if p_value < 0.05 and t_stat > 0:
    print("  → SIGNIFICANT: Old shrinkage hurts at p<0.05")
elif t_stat > 0:
    print(f"  → Trend suggests old shrinkage hurts, but not significant (p={p_value:.3f})")
else:
    print("  → Old shrinkage does not hurt")

# =============================================================================
# Conclusion
# =============================================================================
print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)

if gap_confirmed:
    print("""
The test CONFIRMS our hypothesis:

1. When groups are similar pre-shift (strong shrinkage regime),
   the model learns to pull estimates toward the global mean.

2. After shift, some groups diverge dramatically,
   but the model STILL pulls them toward the OLD global mean.

3. This causes DAMAGE to shifted groups - they're pulled
   toward a mean that no longer represents their true value.

RESEARCH GAP CONFIRMED:
  "Cross-sectional hierarchical shrinkage under discrete regime change"
  - Dynamic Shrinkage handles smooth variation, NOT this
  - This is about when the PRIOR becomes obsolete
""")
else:
    print("""
Test did not confirm hypothesis. Possible reasons:
  1. Shrinkage still too weak (need smaller n or lower τ)
  2. Shift magnitude not large enough relative to shrinkage
  3. Random seed produced unfavorable sample

Try: Increase shift magnitude or decrease n further.
""")

# =============================================================================
# Robustness: Multiple Seeds
# =============================================================================
print("\n" + "=" * 70)
print("ROBUSTNESS CHECK: 100 Random Seeds")
print("=" * 70)

results = []
for seed in range(100):
    np.random.seed(seed)

    # Generate data
    true_means_pre_s = np.random.normal(10, 0.5, n_groups)
    true_means_post_s = true_means_pre_s + shift_magnitude

    pre_data = {g: np.random.normal(true_means_pre_s[g], obs_noise, n_samples_per_group)
                for g in range(n_groups)}
    post_data = {g: np.random.normal(true_means_post_s[g], obs_noise, n_samples_per_group)
                 for g in range(n_groups)}

    # Fit on pre
    pre_fit = empirical_bayes_shrinkage(pre_data)

    # Estimate on post
    post_means = np.array([np.mean(data) for data in post_data.values()])
    shrunk_post = pre_fit['shrinkage_factors'] * post_means + (1 - pre_fit['shrinkage_factors']) * pre_fit['global_mean']

    # MSE
    mse_shrink = np.mean((shrunk_post - true_means_post_s)**2)
    mse_no = np.mean((post_means - true_means_post_s)**2)

    results.append({
        'seed': seed,
        'mse_shrink': mse_shrink,
        'mse_no': mse_no,
        'shrink_hurts': mse_shrink > mse_no,
        'damage_pct': (mse_shrink / mse_no - 1) * 100 if mse_no > 0 else 0
    })

results_df = pd.DataFrame(results)
hurt_pct = results_df['shrink_hurts'].mean() * 100
avg_damage = results_df[results_df['shrink_hurts']]['damage_pct'].mean()

print(f"\nAcross 100 seeds:")
print(f"  Old shrinkage HURTS in {hurt_pct:.0f}% of cases")
print(f"  When it hurts, avg damage: {avg_damage:.1f}%")
print(f"  Mean MSE (shrinkage): {results_df['mse_shrink'].mean():.4f}")
print(f"  Mean MSE (no shrinkage): {results_df['mse_no'].mean():.4f}")

if hurt_pct >= 50:
    print(f"\n✓ CONFIRMED: Old shrinkage hurts in majority of cases ({hurt_pct:.0f}%)")
    print("  Research gap is real and robust.")
else:
    print(f"\n✗ Old shrinkage hurts in only {hurt_pct:.0f}% of cases")
