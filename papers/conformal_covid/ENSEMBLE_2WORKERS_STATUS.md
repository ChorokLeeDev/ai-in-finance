# 50-Seed Ensemble - 2 Workers (Low CPU Mode)

**Started:** 2025-12-26 05:49
**PID:** 75516
**Configuration:** 2 parallel workers (reduced from 7)
**Estimated Time:** ~10 hours (vs 3-4 hours with 7 workers)

---

## Current Status

**Process Health:**
- Main PID: 75516 ✅ Running
- Worker 1: 75779 ✅ Active
- Worker 2: 75780 ✅ Active

**Progress:**
- Task 1/8: sales-shipcond (starting, 0/50 seeds shown)
- Remaining: 7 tasks × 50 seeds = 350 more seeds after this task

**CPU Usage:**
- Expected: ~20-40% total (vs 100%+ with 7 workers)
- Each worker uses ~10-20% when active
- Much more manageable for background operation

---

## Monitoring Commands

### Check if still running:
```bash
ps -p 75516
```

### View latest progress (last 30 lines):
```bash
tail -30 /Users/i767700/Github/ai-in-finance/papers/conformal_covid/ensemble_50seeds_2workers.log
```

### Monitor in real-time:
```bash
tail -f /Users/i767700/Github/ai-in-finance/papers/conformal_covid/ensemble_50seeds_2workers.log
```
Press Ctrl+C to stop monitoring (doesn't stop the process)

### Check CPU usage:
```bash
ps -p 75516,75779,75780 -o pid,%cpu,%mem,time,command
```

### Get process IDs (if needed later):
```bash
pgrep -P 75516
```

---

## Timeline Estimate

With 2 workers instead of 7:

| Task | Est. Time | Cumulative |
|------|-----------|------------|
| 1. sales-shipcond | 1.5h | 1.5h |
| 2. sales-group | 2h | 3.5h |
| 3. sales-payterms | 1.5h | 5h |
| 4. item-plant | 1h | 6h |
| 5. item-shippoint | 1.5h | 7.5h |
| 6. sales-incoterms | 1h | 8.5h |
| 7. item-incoterms | 1h | 9.5h |
| 8. sales-office | 0.5h | 10h |

**Expected completion:** ~16:00 (4 PM) if started at 05:49 AM

**Note:** Times are approximate. Actual may vary by ±20%

---

## What to Expect

### First Hour:
- Log will update slowly (buffered output)
- Progress bar will jump from 0% to 2%, 4%, etc.
- CPU usage will fluctuate between 10-40%
- This is normal - workers alternate between computation and I/O

### Middle Period:
- Steady progress through tasks
- Each task shows progress bar
- Results printed after each task completes

### Completion:
- Final files generated:
  - `results/ensemble_50seeds.pkl`
  - `results/ensemble_50seeds_summary.json`
  - `results/ensemble_50seeds_table.tex`

---

## If You Need to Stop

### Pause (can resume):
```bash
kill -STOP 75516 75779 75780
```

Resume:
```bash
kill -CONT 75516 75779 75780
```

### Kill completely:
```bash
kill -9 75516
```
Note: This will lose all progress (no checkpointing within run)

---

## Next Steps After Completion

1. **Verify results exist:**
```bash
ls -lh papers/conformal_covid/results/ensemble_50seeds*
```

2. **View summary:**
```bash
cat papers/conformal_covid/results/ensemble_50seeds_summary.json
```

3. **Update Table 1 in paper:**
   - Open `main.tex`
   - Replace Table 1 (lines ~112-132) with content from:
     `results/ensemble_50seeds_table.tex`
   - Change "5 seeds" → "50 seeds" in Section 3.2

4. **Recompile paper:**
```bash
cd papers/conformal_covid
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

5. **Start Phase 2 (SHAP analysis):**
   - See `PHASE2_SHAP_PLAN.md` for details
   - Can run while reviewing updated paper

---

## Working in Parallel

While this runs (~10 hours), you can:

1. **Review SHAP implementation:**
   - `code/analyze_feature_importance.py`
   - `code/plot_shap_results.py`
   - `PHASE2_SHAP_PLAN.md`

2. **Start implementing Phase 3 (retraining):**
   - Template in `UAI_2026_COMPLETE_ROADMAP.md` (lines 700-930)

3. **Review paper structure:**
   - Plan where SHAP results will go
   - Draft Figure 3 layout

4. **Other work:**
   - The process runs in background
   - Won't interfere with normal computer use
   - Just avoid CPU-intensive tasks

---

## Troubleshooting

### Progress bar stuck at 0%:
- Normal for first 5-10 minutes
- Progress updates after first seed completes
- Check CPU usage to confirm workers are active

### Log not updating:
- Output is buffered
- May not see updates for several minutes
- Check process status with `ps -p 75516`

### CPU too high:
- Current: 2 workers = ~20-40% CPU
- If still too high, can restart with 1 worker
  (but will take ~16-18 hours)

### Process disappeared:
```bash
ps -p 75516
# If not found, check if completed:
ls -lh results/ensemble_50seeds_summary.json
# Check log for errors:
tail -100 ensemble_50seeds_2workers.log
```

---

**Status:** Running ✅
**Log file:** `ensemble_50seeds_2workers.log`
**Expected completion:** ~16:00 (10 hours from start)
