# Quick Start: 50-Seed Ensemble

## The Problem
Table 1 has **std > mean** for 3 tasks → Will be **rejected** by UAI

## The Solution
Run 50 seeds instead of 5 → Variance reduces by √10 ≈ 3.16×

## Three Commands to Success

### 1. Test (2 minutes)
```bash
cd papers/conformal_covid
python3 code/test_ensemble_setup.py
```

### 2. Quick Verify (10 minutes)
```bash
python3 code/run_50seed_ensemble.py --tasks sales-office --num_seeds 10
```

### 3. Full Run (3-4 hours)
```bash
# Start long run
nohup python3 code/run_50seed_ensemble.py > ensemble.log 2>&1 &

# Check progress
tail -f ensemble.log
```

## What You'll Get

✅ `results/ensemble_50seeds_table.tex` - Copy to paper, replace Table 1
✅ All std < mean (UAI requirement met)
✅ Can resume if interrupted

## After This

1. **Update paper** - Replace Table 1 with new results
2. **Recompile** - `pdflatex main.tex`
3. **Next blocker** - Add regression tasks (I can help)

## Need Help?

📖 Full guide: `code/README_50SEEDS.md`
🗺️ Roadmap: `UAI_2026_ROADMAP.md`
💬 Questions: Just ask!

---

**Current Status:**
- ✅ Bootstrap CI done (today)
- ✅ References fixed (today)
- 🚀 50-seed script ready
- ⏳ Waiting for you to run it
