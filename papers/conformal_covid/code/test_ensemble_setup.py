"""
Quick Test Script - Verify 50-Seed Ensemble Setup

Run this BEFORE the full 50-seed experiment to verify:
1. RelBench is properly installed
2. All tasks can load
3. Pipeline runs without errors
4. Parallelization works

Usage:
    python test_ensemble_setup.py

Expected time: ~2-3 minutes
"""

import sys
from pathlib import Path

print("="*80)
print("TESTING 50-SEED ENSEMBLE SETUP")
print("="*80)
print()

# Test 1: Import dependencies
print("Test 1: Checking dependencies...")
try:
    import lightgbm as lgb
    print(f"  ✓ LightGBM {lgb.__version__}")
except ImportError:
    print("  ✗ LightGBM not installed")
    print("    Install: pip install lightgbm")
    sys.exit(1)

try:
    import numpy as np
    print(f"  ✓ NumPy {np.__version__}")
except ImportError:
    print("  ✗ NumPy not installed")
    sys.exit(1)

try:
    import pandas as pd
    print(f"  ✓ Pandas {pd.__version__}")
except ImportError:
    print("  ✗ Pandas not installed")
    sys.exit(1)

try:
    from sklearn.preprocessing import LabelEncoder
    import sklearn
    print(f"  ✓ Scikit-learn {sklearn.__version__}")
except ImportError:
    print("  ✗ Scikit-learn not installed")
    sys.exit(1)

try:
    from tqdm import tqdm
    print(f"  ✓ tqdm")
except ImportError:
    print("  ✗ tqdm not installed")
    print("    Install: pip install tqdm")
    sys.exit(1)

# Test 2: Check RelBench
print("\nTest 2: Checking RelBench...")
try:
    from relbench.tasks import get_task
    print("  ✓ RelBench import successful")
except ImportError:
    print("  ✗ RelBench not installed")
    print("    Install: cd ../../.. && pip install -e .")
    sys.exit(1)

# Test 3: Load a simple task
print("\nTest 3: Loading test task (sales-office)...")
try:
    task = get_task('rel-salt', 'sales-office', download=False)
    print(f"  ✓ Task loaded: {task.task_type}")
    train = task.get_table('train')
    print(f"    Train rows: {len(train):,}")
    print(f"    Classes: {len(train[task.target_col].unique())}")
except Exception as e:
    print(f"  ✗ Failed to load task: {e}")
    print("    Note: rel-salt data should be in ~/Library/Caches/relbench/")
    print("    If missing, you may need to generate it from the source data.")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Run single seed (minimal test)
print("\nTest 4: Running single seed (this may take 1-2 minutes)...")
try:
    from run_50seed_ensemble import run_single_seed
    result = run_single_seed(task, 'sales-office', seed=42, sample_size=10000)

    print(f"  ✓ Single seed completed successfully")
    print(f"    Val coverage: {result['val_coverage']*100:.1f}%")
    print(f"    Test coverage: {result['test_coverage']*100:.1f}%")
    print(f"    Coverage drop: {result['coverage_drop']*100:.1f}%")
except Exception as e:
    print(f"  ✗ Failed to run single seed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Check parallelization
print("\nTest 5: Checking parallel processing...")
try:
    from multiprocessing import cpu_count
    n_cpus = cpu_count()
    print(f"  ✓ Detected {n_cpus} CPU cores")
    print(f"    Will use {n_cpus - 1} workers for parallel execution")
except Exception as e:
    print(f"  ⚠ Could not detect CPUs: {e}")
    print("    Will fall back to single-threaded mode")

# Test 6: Check output directory
print("\nTest 6: Checking output directory...")
output_dir = Path(__file__).parent.parent / "results"
if output_dir.exists():
    print(f"  ✓ Output directory exists: {output_dir}")
else:
    print(f"  ✓ Will create output directory: {output_dir}")

checkpoint_dir = output_dir / "checkpoints"
print(f"  ✓ Checkpoints will be saved to: {checkpoint_dir}")

# Test 7: Estimate runtime
print("\nTest 7: Runtime estimation...")
import time
start = time.time()
# Quick run to measure speed
try:
    _ = run_single_seed(task, 'sales-office', seed=43, sample_size=5000)
    elapsed = time.time() - start

    print(f"  ✓ Single seed (5k samples): {elapsed:.1f} seconds")

    # Estimate full run
    seeds_per_task = 50
    num_tasks = 8
    total_seeds = seeds_per_task * num_tasks
    estimated_time_single = total_seeds * elapsed
    estimated_time_parallel = estimated_time_single / (n_cpus - 1)

    print(f"\n  Estimated time for full run ({total_seeds} seeds):")
    print(f"    Single-threaded: {estimated_time_single/3600:.1f} hours")
    print(f"    Parallel ({n_cpus-1} workers): {estimated_time_parallel/3600:.1f} hours")

except Exception as e:
    print(f"  ⚠ Could not estimate runtime: {e}")

# Summary
print("\n" + "="*80)
print("TEST SUMMARY")
print("="*80)
print("✓ All tests passed!")
print()
print("You're ready to run the full 50-seed ensemble:")
print()
print("  # Quick test (1 task, 10 seeds, ~10 minutes)")
print("  python code/run_50seed_ensemble.py --tasks sales-office --num_seeds 10")
print()
print("  # Full run (8 tasks, 50 seeds, ~3-4 hours)")
print("  python code/run_50seed_ensemble.py")
print()
print("See code/README_50SEEDS.md for detailed instructions.")
print("="*80)
