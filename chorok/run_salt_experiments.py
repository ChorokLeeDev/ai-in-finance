"""
SALT UQ Experiments Runner
Runs all baseline and MC Dropout experiments for the workshop paper

Usage:
    python run_salt_experiments.py --mode all
    python run_salt_experiments.py --mode lightgbm  # Just LightGBM baselines
    python run_salt_experiments.py --mode gnn       # Just GNN baselines
    python run_salt_experiments.py --mode uq        # Just MC Dropout UQ
"""

import argparse
import json
import os
import sys
from pathlib import Path
import time
from datetime import datetime

# Add parent directory to path to import from examples
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from tqdm import tqdm

# Configuration
SALT_TASKS = [
    "item-plant",
    "item-shippoint",
    "item-incoterms",
    "sales-office",
    "sales-group",
    "sales-payterms",
    "sales-shipcond",
    "sales-incoterms",
]

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

def run_lightgbm_baseline(task_name):
    """Run LightGBM baseline for a task"""
    print(f"\n{'='*60}")
    print(f"Running LightGBM baseline for {task_name}")
    print(f"{'='*60}\n")

    cmd = f"""python {Path(__file__).parent.parent / 'examples' / 'lightgbm_autocomplete.py'} \
        --dataset rel-salt \
        --task {task_name} \
        --download \
        --num_trials 3 \
        --cache_dir {RESULTS_DIR / 'cache'}"""

    start_time = time.time()
    ret = os.system(cmd)
    elapsed = time.time() - start_time

    if ret != 0:
        print(f"⚠️  Warning: LightGBM failed for {task_name}")
        return None

    print(f"✓ LightGBM completed for {task_name} in {elapsed:.1f}s")
    return {"task": task_name, "model": "lightgbm", "time": elapsed}


def run_gnn_baseline(task_name, epochs=10):
    """Run GNN baseline for a task"""
    print(f"\n{'='*60}")
    print(f"Running GNN baseline for {task_name}")
    print(f"{'='*60}\n")

    cmd = f"""python {Path(__file__).parent.parent / 'examples' / 'gnn_autocomplete.py'} \
        --dataset rel-salt \
        --task {task_name} \
        --epochs {epochs} \
        --batch_size 512 \
        --channels 128 \
        --num_layers 2 \
        --lr 0.001 \
        --cache_dir {RESULTS_DIR / 'cache'} \
        --torch_device cuda"""

    start_time = time.time()
    ret = os.system(cmd)
    elapsed = time.time() - start_time

    if ret != 0:
        print(f"⚠️  Warning: GNN failed for {task_name}")
        return None

    print(f"✓ GNN completed for {task_name} in {elapsed:.1f}s")
    return {"task": task_name, "model": "gnn", "time": elapsed}


def run_gnn_mc_dropout(task_name, epochs=10, num_samples=30):
    """Run GNN with MC Dropout for UQ"""
    print(f"\n{'='*60}")
    print(f"Running GNN + MC Dropout for {task_name}")
    print(f"{'='*60}\n")

    # This will use the modified version with MC Dropout
    # You'll need to create gnn_autocomplete_uq.py based on template below
    cmd = f"""python {Path(__file__).parent / 'gnn_autocomplete_uq.py'} \
        --dataset rel-salt \
        --task {task_name} \
        --epochs {epochs} \
        --batch_size 512 \
        --channels 128 \
        --num_layers 2 \
        --lr 0.001 \
        --mc_samples {num_samples} \
        --dropout_rate 0.1 \
        --cache_dir {RESULTS_DIR / 'cache'} \
        --torch_device cuda"""

    start_time = time.time()
    ret = os.system(cmd)
    elapsed = time.time() - start_time

    if ret != 0:
        print(f"⚠️  Warning: GNN+UQ failed for {task_name}")
        return None

    print(f"✓ GNN+UQ completed for {task_name} in {elapsed:.1f}s")
    return {"task": task_name, "model": "gnn_mc_dropout", "time": elapsed}


def main():
    parser = argparse.ArgumentParser(description="Run SALT UQ experiments")
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "lightgbm", "gnn", "uq", "quick"],
        help="Which experiments to run"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=None,
        help="Specific tasks to run (default: all)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs for GNN training"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: only first 3 tasks with fewer epochs"
    )

    args = parser.parse_args()

    # Determine which tasks to run
    if args.tasks:
        tasks_to_run = args.tasks
    elif args.quick or args.mode == "quick":
        tasks_to_run = SALT_TASKS[:3]  # Only first 3 tasks
        args.epochs = 5  # Fewer epochs
        print(f"🚀 QUICK MODE: Running only {tasks_to_run} with {args.epochs} epochs")
    else:
        tasks_to_run = SALT_TASKS

    # Run experiments
    all_results = []

    start_time = datetime.now()
    print(f"\n{'='*60}")
    print(f"STARTING SALT UQ EXPERIMENTS")
    print(f"Time: {start_time}")
    print(f"Mode: {args.mode}")
    print(f"Tasks: {tasks_to_run}")
    print(f"{'='*60}\n")

    for task in tasks_to_run:
        print(f"\n{'#'*60}")
        print(f"Task: {task}")
        print(f"{'#'*60}")

        # Run LightGBM
        if args.mode in ["all", "lightgbm"]:
            result = run_lightgbm_baseline(task)
            if result:
                all_results.append(result)

        # Run GNN
        if args.mode in ["all", "gnn"]:
            result = run_gnn_baseline(task, epochs=args.epochs)
            if result:
                all_results.append(result)

        # Run GNN + UQ
        if args.mode in ["all", "uq"]:
            result = run_gnn_mc_dropout(task, epochs=args.epochs)
            if result:
                all_results.append(result)

    # Save summary
    end_time = datetime.now()
    elapsed_total = (end_time - start_time).total_seconds()

    summary = {
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "elapsed_seconds": elapsed_total,
        "mode": args.mode,
        "tasks": tasks_to_run,
        "results": all_results
    }

    summary_file = RESULTS_DIR / f"summary_{start_time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"ALL EXPERIMENTS COMPLETE!")
    print(f"Total time: {elapsed_total/3600:.2f} hours")
    print(f"Results saved to: {summary_file}")
    print(f"{'='*60}\n")

    # Print summary table
    print("\nSummary:")
    print(f"{'Task':<25} {'Model':<20} {'Time (s)':<10}")
    print("-" * 60)
    for r in all_results:
        print(f"{r['task']:<25} {r['model']:<20} {r['time']:<10.1f}")

    print(f"\nNext steps:")
    print(f"1. Check individual results in examples/ output")
    print(f"2. Run: python create_figures.py to generate plots")
    print(f"3. Update paper with results")


if __name__ == "__main__":
    main()
