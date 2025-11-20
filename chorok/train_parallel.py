"""
Parallel Ensemble Training Script

**Purpose**: Train all missing ensembles in parallel using multiple processes

**Features**:
- Automatic parallel execution (uses all CPU cores)
- Optimized settings for speed (trials=5, sample_size=10000)
- Real-time progress monitoring
- Automatic error recovery

**Usage**:
    # Just run this - it handles everything
    python chorok/train_parallel.py

    # Custom parallelism
    python chorok/train_parallel.py --workers 4

    # Custom settings
    python chorok/train_parallel.py --num-trials 3 --sample-size 5000
"""

import argparse
import subprocess
import sys
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
import json
import time

# Fast configuration (3x faster than default)
DEFAULT_NUM_TRIALS = 5
DEFAULT_SAMPLE_SIZE = 10000
DATASET = 'rel-salt'

# Tasks that need complete ensembles
MISSING_TASKS = [
    'item-shippoint',
    'item-incoterms',
    'sales-payterms',
    'sales-shipcond',
    'sales-incoterms'
]

SALES_GROUP_NEEDED_SEEDS = [42, 43, 44, 45]
STANDARD_SEEDS = [42, 43, 44, 45, 46]


def check_existing_predictions(task, seed, sample_size):
    """Check if predictions already exist"""
    pred_path = Path(f'results/ensemble/{DATASET}/{task}/seed_{seed}_sample_{sample_size}.pkl')
    return pred_path.exists()


def get_missing_combinations(sample_size):
    """Get list of (task, seed) combinations that need training"""
    missing = []

    # Check missing tasks
    for task in MISSING_TASKS:
        for seed in STANDARD_SEEDS:
            if not check_existing_predictions(task, seed, sample_size):
                missing.append((task, seed))

    # Check sales-group additional seeds
    for seed in SALES_GROUP_NEEDED_SEEDS:
        if not check_existing_predictions('sales-group', seed, sample_size):
            missing.append(('sales-group', seed))

    return missing


def train_single_model(args_tuple):
    """Train a single model (worker function for multiprocessing)"""
    task, seed, sample_size, num_trials, worker_id = args_tuple

    cmd = [
        sys.executable,
        'examples/lightgbm_autocomplete.py',
        '--dataset', DATASET,
        '--task', task,
        '--seed', str(seed),
        '--sample_size', str(sample_size),
        '--num_trials', str(num_trials)
    ]

    start_time = time.time()

    try:
        print(f"[Worker {worker_id}] Starting: {task} (seed={seed})")

        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            cwd='.'
        )

        elapsed = time.time() - start_time
        print(f"[Worker {worker_id}] ✓ Completed: {task} (seed={seed}) in {elapsed/60:.1f}min")

        return {
            'task': task,
            'seed': seed,
            'status': 'success',
            'time': elapsed,
            'worker_id': worker_id
        }

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"[Worker {worker_id}] ✗ Failed: {task} (seed={seed})")
        print(f"  Error: {e.stderr[:200]}")

        return {
            'task': task,
            'seed': seed,
            'status': 'failed',
            'error': str(e.stderr[:500]),
            'time': elapsed,
            'worker_id': worker_id
        }
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[Worker {worker_id}] ✗ Error: {task} (seed={seed}) - {str(e)}")

        return {
            'task': task,
            'seed': seed,
            'status': 'error',
            'error': str(e),
            'time': elapsed,
            'worker_id': worker_id
        }


def main():
    parser = argparse.ArgumentParser(description='Parallel ensemble training')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers (default: CPU count)')
    parser.add_argument('--num-trials', type=int, default=DEFAULT_NUM_TRIALS,
                       help='Number of Optuna trials per model')
    parser.add_argument('--sample-size', type=int, default=DEFAULT_SAMPLE_SIZE,
                       help='Training sample size')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be trained without training')
    parser.add_argument('--yes', '-y', action='store_true',
                       help='Skip confirmation prompt')
    args = parser.parse_args()

    # Determine number of workers
    if args.workers is None:
        args.workers = max(1, mp.cpu_count() - 1)  # Leave 1 CPU free

    # Get missing combinations
    missing = get_missing_combinations(args.sample_size)

    if not missing:
        print("✓ All ensembles already complete!")
        return

    # Display configuration
    print("=" * 70)
    print("PARALLEL ENSEMBLE TRAINING")
    print("=" * 70)
    print(f"Dataset: {DATASET}")
    print(f"Sample size: {args.sample_size:,} (faster than 50k)")
    print(f"Num trials: {args.num_trials} (faster than 10)")
    print(f"Parallel workers: {args.workers}")
    print(f"Total models to train: {len(missing)}")
    print()

    # Calculate estimated time
    time_per_model = 5  # minutes (optimistic with fast settings)
    parallel_time = (len(missing) * time_per_model) / args.workers
    print(f"Estimated time: ~{parallel_time:.0f} minutes ({parallel_time/60:.1f} hours)")
    print(f"  vs. sequential: ~{len(missing) * time_per_model} minutes")
    print(f"  Speedup: {args.workers}x faster")
    print()

    # Group by task for display
    task_counts = {}
    for task, seed in missing:
        task_counts[task] = task_counts.get(task, 0) + 1

    print("Missing combinations by task:")
    for task, count in sorted(task_counts.items()):
        seeds = [s for t, s in missing if t == task]
        print(f"  {task}: {count} seeds {seeds}")
    print("=" * 70)

    if args.dry_run:
        print("\n[DRY RUN MODE] - No actual training will occur")
        return

    # Confirm
    if not args.yes:
        response = input("\nStart parallel training? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return

    print(f"\n🚀 Starting {args.workers} parallel workers...\n")

    # Prepare work items with worker IDs
    work_items = [
        (task, seed, args.sample_size, args.num_trials, i % args.workers)
        for i, (task, seed) in enumerate(missing)
    ]

    # Track results
    start_time = time.time()
    results = {
        'started': datetime.now().isoformat(),
        'config': {
            'workers': args.workers,
            'num_trials': args.num_trials,
            'sample_size': args.sample_size
        },
        'results': []
    }

    # Run in parallel
    with mp.Pool(processes=args.workers) as pool:
        for result in pool.imap_unordered(train_single_model, work_items):
            results['results'].append(result)

            # Progress update
            completed = len(results['results'])
            success = sum(1 for r in results['results'] if r['status'] == 'success')
            failed = sum(1 for r in results['results'] if r['status'] != 'success')

            print(f"\n[Progress] {completed}/{len(missing)} complete "
                  f"({success} success, {failed} failed)")

    # Final summary
    total_time = time.time() - start_time
    results['completed'] = datetime.now().isoformat()
    results['total_time'] = total_time

    successful = [r for r in results['results'] if r['status'] == 'success']
    failed = [r for r in results['results'] if r['status'] != 'success']

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
    print(f"Total models: {len(missing)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print()

    if successful:
        avg_time = sum(r['time'] for r in successful) / len(successful)
        print(f"Average time per model: {avg_time/60:.1f} minutes")

    if failed:
        print("\nFailed models:")
        for r in failed:
            print(f"  {r['task']} (seed={r['seed']})")
        print("\nYou can retry failed models with:")
        print("  python chorok/train_parallel.py")

    # Save log
    log_path = Path('chorok/results/parallel_training_log.json')
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nLog saved to: {log_path}")

    # Show next steps
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("1. Check status:")
    print("   python chorok/check_ensemble_status.py")
    print()
    print("2. If all complete, run Phase 3 (UQ):")
    print("   python chorok/temporal_uncertainty_analysis.py")
    print()
    print("3. Then run Phase 4 (Correlation):")
    print("   python chorok/compare_shift_uncertainty.py")
    print("=" * 70)


if __name__ == '__main__':
    main()
