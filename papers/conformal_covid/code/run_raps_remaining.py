"""
Run remaining RAPS multi-seed tasks with configurable num_seeds per task.
Sales tasks: 10 seeds. Item tasks: 3 seeds (slow).
"""
import sys
import json
import time
import traceback
import numpy as np
from pathlib import Path
from scipy import stats

# Import from original script
sys.path.insert(0, str(Path(__file__).parent))
import run_raps_multiseed as base

RESULTS_DIR = base.RESULTS_DIR
OUTPUT_FILE = base.OUTPUT_FILE

# Tasks to run with their seed counts
TASKS_TO_RUN = {
    'sales-incoterms': 10,
    'sales-office': 10,
    'item-plant': 3,
    'item-shippoint': 3,
    'item-incoterms': 3,
}

def run_task_with_seeds(task_name, num_seeds, existing_results):
    """Run a task with a specific number of seeds."""
    from relbench.tasks import get_task

    print(f"\n{'='*70}", flush=True)
    print(f"Task: {task_name} ({num_seeds} seeds)", flush=True)
    print(f"{'='*70}", flush=True)

    t_start = time.time()
    task = get_task('rel-salt', task_name, download=False)

    seeds = list(range(base.SEED_START, base.SEED_START + num_seeds))
    seed_results = []

    # Check which seeds are already done (for resume)
    done_seeds = set()
    if task_name in existing_results and 'seed_results' in existing_results[task_name]:
        for sr in existing_results[task_name]['seed_results']:
            if sr['seed'] in seeds:
                done_seeds.add(sr['seed'])
                seed_results.append(sr)
        if done_seeds:
            print(f"  Resuming: {len(done_seeds)} seeds already done", flush=True)

    for seed in seeds:
        if seed in done_seeds:
            continue
        st = time.time()
        print(f"  Seed {seed} ({len(seed_results)+1}/{num_seeds})...", end='', flush=True)
        try:
            res = base.run_single_seed(task, task_name, seed)
            seed_results.append(res)
            elapsed = time.time() - st
            print(f" APS drop={res['aps_coverage_drop']*100:.1f}%, "
                  f"RAPS drop={res['raps_coverage_drop']*100:.1f}% "
                  f"({elapsed:.0f}s)", flush=True)
        except Exception as e:
            print(f" ERROR: {e}", flush=True)
            traceback.print_exc()

    # Aggregate
    aps_drops = [r['aps_coverage_drop'] for r in seed_results]
    raps_drops = [r['raps_coverage_drop'] for r in seed_results]
    aps_val_covs = [r['aps_val_coverage'] for r in seed_results]
    aps_test_covs = [r['aps_test_coverage'] for r in seed_results]
    raps_val_covs = [r['raps_val_coverage'] for r in seed_results]
    raps_test_covs = [r['raps_test_coverage'] for r in seed_results]
    aps_val_sizes = [r['aps_val_set_size'] for r in seed_results]
    aps_test_sizes = [r['aps_test_set_size'] for r in seed_results]
    raps_val_sizes = [r['raps_val_set_size'] for r in seed_results]
    raps_test_sizes = [r['raps_test_set_size'] for r in seed_results]

    elapsed_total = time.time() - t_start

    agg = {
        'num_classes': seed_results[0]['num_classes'],
        'concentration': base.LGB_CONCENTRATIONS[task_name],
        'num_seeds': len(seed_results),
        'aps_val_coverage_mean': round(float(np.mean(aps_val_covs) * 100), 2),
        'aps_val_coverage_std': round(float(np.std(aps_val_covs) * 100), 2),
        'aps_test_coverage_mean': round(float(np.mean(aps_test_covs) * 100), 2),
        'aps_test_coverage_std': round(float(np.std(aps_test_covs) * 100), 2),
        'aps_drop_mean': round(float(np.mean(aps_drops) * 100), 2),
        'aps_drop_std': round(float(np.std(aps_drops) * 100), 2),
        'aps_val_size_mean': round(float(np.mean(aps_val_sizes)), 2),
        'aps_test_size_mean': round(float(np.mean(aps_test_sizes)), 2),
        'raps_val_coverage_mean': round(float(np.mean(raps_val_covs) * 100), 2),
        'raps_val_coverage_std': round(float(np.std(raps_val_covs) * 100), 2),
        'raps_test_coverage_mean': round(float(np.mean(raps_test_covs) * 100), 2),
        'raps_test_coverage_std': round(float(np.std(raps_test_covs) * 100), 2),
        'raps_drop_mean': round(float(np.mean(raps_drops) * 100), 2),
        'raps_drop_std': round(float(np.std(raps_drops) * 100), 2),
        'raps_val_size_mean': round(float(np.mean(raps_val_sizes)), 2),
        'raps_test_size_mean': round(float(np.mean(raps_test_sizes)), 2),
        'seed_results': seed_results,
        'elapsed_s': round(elapsed_total, 1),
    }

    print(f"\n  Summary for {task_name}:", flush=True)
    print(f"    APS:  val={agg['aps_val_coverage_mean']:.1f}+-{agg['aps_val_coverage_std']:.1f}%, "
          f"test={agg['aps_test_coverage_mean']:.1f}+-{agg['aps_test_coverage_std']:.1f}%, "
          f"drop={agg['aps_drop_mean']:.1f}+-{agg['aps_drop_std']:.1f}%", flush=True)
    print(f"    RAPS: val={agg['raps_val_coverage_mean']:.1f}+-{agg['raps_val_coverage_std']:.1f}%, "
          f"test={agg['raps_test_coverage_mean']:.1f}+-{agg['raps_test_coverage_std']:.1f}%, "
          f"drop={agg['raps_drop_mean']:.1f}+-{agg['raps_drop_std']:.1f}%", flush=True)
    print(f"    Time: {elapsed_total:.0f}s", flush=True)

    return agg


def save_results(task_results):
    """Save incrementally, merging with existing results."""
    output = {
        'config': {
            'num_seeds': 'variable (10 for sales, 3 for item)',
            'seed_start': base.SEED_START,
            'alpha': base.ALPHA,
            'sample_size': base.SAMPLE_SIZE,
            'raps_lambda': base.LAMBDA_REG,
            'raps_k_reg': base.K_REG,
        },
        'tasks': task_results,
        'completed_tasks': list(task_results.keys()),
        'n_completed': len(task_results),
    }
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2)


def main():
    print(f"\n{'='*70}", flush=True)
    print("RAPS Multi-Seed: Remaining 5 Tasks", flush=True)
    print(f"Sales tasks (10 seeds): sales-incoterms, sales-office", flush=True)
    print(f"Item tasks (3 seeds): item-plant, item-shippoint, item-incoterms", flush=True)
    print(f"{'='*70}\n", flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing results
    existing_results = {}
    task_results = {}
    if OUTPUT_FILE.exists():
        try:
            with open(OUTPUT_FILE, 'r') as f:
                existing = json.load(f)
            if 'tasks' in existing:
                existing_results = existing['tasks']
                task_results = dict(existing_results)  # start with all existing
                completed = list(existing_results.keys())
                print(f"Loaded existing results: {completed}", flush=True)
        except Exception as e:
            print(f"Warning: could not load existing results: {e}", flush=True)

    t_total = time.time()

    # Run remaining tasks (sales first - faster, then item tasks)
    task_order = ['sales-incoterms', 'sales-office', 'item-plant', 'item-shippoint', 'item-incoterms']

    for task_name in task_order:
        num_seeds = TASKS_TO_RUN[task_name]

        # Skip if already completed with enough seeds
        if (task_name in existing_results
                and existing_results[task_name].get('num_seeds', 0) >= num_seeds):
            print(f"\nSkipping {task_name} (already completed with {existing_results[task_name]['num_seeds']} seeds)", flush=True)
            continue

        try:
            agg = run_task_with_seeds(task_name, num_seeds, existing_results)
            task_results[task_name] = agg

            # Incremental save
            save_results(task_results)
            print(f"  Saved ({len(task_results)}/{len(task_results)} total tasks)", flush=True)
        except Exception as e:
            print(f"\nERROR on {task_name}: {e}", flush=True)
            traceback.print_exc()

    total_time = time.time() - t_total

    # Print final summary table
    print(f"\n{'='*70}", flush=True)
    print("FINAL SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"{'Task':<18} {'C%':>5} {'Seeds':>5} {'APS Drop':>12} {'RAPS Drop':>12}", flush=True)
    print(f"{'-'*55}", flush=True)
    for t in base.ALL_TASKS:
        if t in task_results:
            r = task_results[t]
            print(f"{t:<18} {r['concentration']:>4.1f}% {r['num_seeds']:>5} "
                  f"{r['aps_drop_mean']:>6.1f}+-{r['aps_drop_std']:>4.1f}  "
                  f"{r['raps_drop_mean']:>6.1f}+-{r['raps_drop_std']:>4.1f}", flush=True)

    # Compute correlations if enough data
    completed = [t for t in base.ALL_TASKS if t in task_results]
    if len(completed) >= 4:
        concs = [task_results[t]['concentration'] for t in completed]
        aps_drops = [task_results[t]['aps_drop_mean'] for t in completed]
        raps_drops = [task_results[t]['raps_drop_mean'] for t in completed]

        aps_rho, aps_p = stats.spearmanr(concs, aps_drops)
        raps_rho, raps_p = stats.spearmanr(concs, raps_drops)

        print(f"\nCorrelation (concentration vs drop, n={len(completed)}):", flush=True)
        print(f"  APS:  rho={aps_rho:.3f} (p={aps_p:.4f})", flush=True)
        print(f"  RAPS: rho={raps_rho:.3f} (p={raps_p:.4f})", flush=True)

    print(f"\nTotal time: {total_time:.0f}s ({total_time/60:.1f} min)", flush=True)
    print(f"Results: {OUTPUT_FILE}", flush=True)


if __name__ == "__main__":
    main()
