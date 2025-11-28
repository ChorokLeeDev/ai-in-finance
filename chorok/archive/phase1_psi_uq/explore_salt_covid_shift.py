"""
SALT COVID-19 Distribution Shift Explorer
Analyzes all 8 SALT tasks to quantify distribution shift during COVID-19 period.

This script answers: "Which SALT tasks show the strongest COVID-induced distribution shift?"

Usage:
    python explore_salt_covid_shift.py
    python explore_salt_covid_shift.py --save_plots
    python explore_salt_covid_shift.py --task item-plant  # Single task analysis
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from relbench.tasks import get_task

# SALT task configuration
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

# COVID timeline for SALT
COVID_TIMELINE = {
    "train": ("2018-01-01", "2020-01-31", "Pre-COVID"),
    "val": ("2020-02-01", "2020-06-30", "COVID Peak"),
    "test": ("2020-07-01", "2021-01-01", "Recovery"),
}


def compute_class_distribution(labels: np.ndarray) -> dict:
    """Compute normalized class distribution from labels."""
    unique, counts = np.unique(labels, return_counts=True)
    total = counts.sum()
    return {int(c): count / total for c, count in zip(unique, counts)}


def distribution_to_array(dist: dict, all_classes: set) -> np.ndarray:
    """Convert distribution dict to array for all classes."""
    arr = np.zeros(len(all_classes))
    class_to_idx = {c: i for i, c in enumerate(sorted(all_classes))}
    for c, prob in dist.items():
        if c in class_to_idx:
            arr[class_to_idx[c]] = prob
    return arr


def compute_js_divergence(dist1: dict, dist2: dict) -> float:
    """Compute Jensen-Shannon divergence between two distributions."""
    all_classes = set(dist1.keys()) | set(dist2.keys())
    p = distribution_to_array(dist1, all_classes)
    q = distribution_to_array(dist2, all_classes)
    # Add small epsilon to avoid division by zero
    p = p + 1e-10
    q = q + 1e-10
    p = p / p.sum()
    q = q / q.sum()
    return jensenshannon(p, q)


def compute_chi_square(labels1: np.ndarray, labels2: np.ndarray) -> tuple:
    """Compute chi-square test for independence between two label distributions."""
    all_labels = np.concatenate([labels1, labels2])
    unique_labels = np.unique(all_labels)

    # Create contingency table
    counts1 = np.array([np.sum(labels1 == l) for l in unique_labels])
    counts2 = np.array([np.sum(labels2 == l) for l in unique_labels])

    # Filter out classes with zero counts in both
    mask = (counts1 + counts2) > 0
    counts1 = counts1[mask]
    counts2 = counts2[mask]

    contingency = np.array([counts1, counts2])

    try:
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
        return chi2, p_value, dof
    except ValueError:
        return np.nan, np.nan, 0


def compute_entropy(dist: dict) -> float:
    """Compute entropy of a distribution (measures class imbalance)."""
    probs = np.array(list(dist.values()))
    probs = probs[probs > 0]  # Filter zeros
    return -np.sum(probs * np.log2(probs))


def analyze_task(task_name: str, verbose: bool = True) -> dict:
    """Analyze a single SALT task for COVID distribution shift."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"Analyzing: {task_name}")
        print(f"{'='*60}")

    # Load task (download=False since SALT tasks aren't on the server)
    task = get_task("rel-salt", task_name, download=False)

    # Get tables (test has no labels for SALT benchmark)
    train_table = task.get_table("train")
    val_table = task.get_table("val")
    test_table = task.get_table("test")

    # Extract labels and timestamps
    results = {
        "task": task_name,
        "target_col": task.target_col,
        "entity_table": task.entity_table,
    }

    splits_data = {}
    # Note: test split may not have labels for SALT
    for split_name, table in [("train", train_table), ("val", val_table)]:
        labels = table.df[task.target_col].values
        timestamps = table.df.get("CREATIONTIMESTAMP", pd.Series([None] * len(labels)))

        splits_data[split_name] = {
            "labels": labels,
            "timestamps": timestamps,
            "n_samples": len(labels),
            "n_classes": len(np.unique(labels)),
            "distribution": compute_class_distribution(labels),
        }

        results[f"{split_name}_n_samples"] = len(labels)
        results[f"{split_name}_n_classes"] = len(np.unique(labels))
        results[f"{split_name}_entropy"] = compute_entropy(splits_data[split_name]["distribution"])

    # Test split - only get sample count (no labels available)
    results["test_n_samples"] = len(test_table.df)
    results["test_n_classes"] = 0  # Unknown - no labels

    # Compute shift metrics
    # JS divergence: train→val (COVID shift) - this is the key metric
    results["js_train_val"] = compute_js_divergence(
        splits_data["train"]["distribution"],
        splits_data["val"]["distribution"]
    )
    # No test labels available, so set these to NaN
    results["js_train_test"] = np.nan
    results["js_val_test"] = np.nan

    # Chi-square tests (train vs val only)
    chi2_tv, p_tv, _ = compute_chi_square(
        splits_data["train"]["labels"],
        splits_data["val"]["labels"]
    )
    results["chi2_train_val"] = chi2_tv
    results["chi2_pvalue_train_val"] = p_tv

    # No test labels available
    results["chi2_train_test"] = np.nan
    results["chi2_pvalue_train_test"] = np.nan

    # Shift significance
    results["significant_shift_val"] = p_tv < 0.05 if not np.isnan(p_tv) else False
    results["significant_shift_test"] = False  # Cannot compute without labels

    # Store distributions for plotting
    results["_distributions"] = {
        split: data["distribution"] for split, data in splits_data.items()
    }
    results["_labels"] = {
        split: data["labels"] for split, data in splits_data.items()
    }

    if verbose:
        print(f"\nSample counts:")
        print(f"  Train: {results['train_n_samples']:,} ({results['train_n_classes']} classes)")
        print(f"  Val:   {results['val_n_samples']:,} ({results['val_n_classes']} classes)")
        print(f"  Test:  {results['test_n_samples']:,} (no labels available)")
        print(f"\nDistribution shift (Jensen-Shannon divergence):")
        print(f"  Train→Val (COVID):  {results['js_train_val']:.4f}")
        print(f"\nChi-square test (train vs val):")
        print(f"  Chi2: {results['chi2_train_val']:.2f}, p-value: {results['chi2_pvalue_train_val']:.2e}")
        print(f"  Significant shift: {'YES' if results['significant_shift_val'] else 'NO'}")

    return results


def plot_task_distributions(results: dict, save_path: str = None):
    """Plot class distributions for a single task across splits."""
    distributions = results["_distributions"]
    task_name = results["task"]

    # Get all classes
    all_classes = set()
    for dist in distributions.values():
        all_classes.update(dist.keys())
    all_classes = sorted(all_classes)

    # Limit to top N classes for readability
    max_classes = 20
    if len(all_classes) > max_classes:
        # Get most frequent classes from train
        train_dist = distributions["train"]
        top_classes = sorted(train_dist.keys(), key=lambda x: train_dist.get(x, 0), reverse=True)[:max_classes]
        all_classes = sorted(top_classes)

    # Create DataFrame for plotting (train and val only - no test labels)
    plot_data = []
    for split in ["train", "val"]:
        dist = distributions[split]
        for c in all_classes:
            plot_data.append({
                "Class": str(c),
                "Split": split.capitalize(),
                "Proportion": dist.get(c, 0)
            })

    df = pd.DataFrame(plot_data)

    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(all_classes))
    width = 0.35

    colors = {"Train": "#2ecc71", "Val": "#e74c3c"}

    for i, split in enumerate(["Train", "Val"]):
        split_data = df[df["Split"] == split]
        values = [split_data[split_data["Class"] == str(c)]["Proportion"].values[0]
                  if len(split_data[split_data["Class"] == str(c)]) > 0 else 0
                  for c in all_classes]
        ax.bar(x + i * width, values, width, label=split, color=colors[split], alpha=0.8)

    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Proportion", fontsize=12)
    ax.set_title(f"Class Distribution: {task_name}\n"
                 f"JS(train→val): {results['js_train_val']:.4f}, "
                 f"Chi2 p-value: {results['chi2_pvalue_train_val']:.2e}", fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels(all_classes, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()


def plot_shift_ranking(all_results: list, save_path: str = None):
    """Plot ranking of tasks by COVID shift magnitude."""
    # Sort by JS divergence (train→val)
    sorted_results = sorted(all_results, key=lambda x: x["js_train_val"], reverse=True)

    tasks = [r["task"] for r in sorted_results]
    js_values = [r["js_train_val"] for r in sorted_results]
    significant = [r["significant_shift_val"] for r in sorted_results]

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ["#e74c3c" if sig else "#95a5a6" for sig in significant]
    bars = ax.barh(tasks, js_values, color=colors, alpha=0.8)

    ax.set_xlabel("Jensen-Shannon Divergence (Train → Val)", fontsize=12)
    ax.set_ylabel("Task", fontsize=12)
    ax.set_title("SALT Tasks: COVID Distribution Shift Ranking\n"
                 "(Red = statistically significant shift, p < 0.05)", fontsize=14)
    ax.grid(axis="x", alpha=0.3)

    # Add value labels
    for bar, val in zip(bars, js_values):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()


def plot_temporal_overview(all_results: list, save_path: str = None):
    """Plot overview of sample counts and shifts across all tasks."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    tasks = [r["task"] for r in all_results]

    # Plot 1: Sample counts
    ax = axes[0]
    x = np.arange(len(tasks))
    width = 0.25

    train_counts = [r["train_n_samples"] for r in all_results]
    val_counts = [r["val_n_samples"] for r in all_results]
    test_counts = [r["test_n_samples"] for r in all_results]

    ax.bar(x - width, train_counts, width, label="Train (Pre-COVID)", color="#2ecc71", alpha=0.8)
    ax.bar(x, val_counts, width, label="Val (COVID Peak)", color="#e74c3c", alpha=0.8)
    ax.bar(x + width, test_counts, width, label="Test (Recovery)", color="#3498db", alpha=0.8)

    ax.set_xlabel("Task")
    ax.set_ylabel("Sample Count")
    ax.set_title("Sample Counts per Split")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Plot 2: JS Divergences (train→val only since test has no labels)
    ax = axes[1]
    js_train_val = [r["js_train_val"] for r in all_results]

    ax.bar(x, js_train_val, width * 2, label="Train→Val (COVID)", color="#e74c3c", alpha=0.8)

    ax.set_xlabel("Task")
    ax.set_ylabel("Jensen-Shannon Divergence")
    ax.set_title("COVID Distribution Shift\n(Train→Val)")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Plot 3: Number of classes
    ax = axes[2]
    n_classes = [r["train_n_classes"] for r in all_results]

    ax.bar(x, n_classes, color="#9b59b6", alpha=0.8)
    ax.set_xlabel("Task")
    ax.set_ylabel("Number of Classes")
    ax.set_title("Task Complexity (# Classes)")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Explore COVID distribution shift in SALT tasks")
    parser.add_argument("--task", type=str, default=None, help="Specific task to analyze")
    parser.add_argument("--save_plots", action="store_true", help="Save plots to files")
    parser.add_argument("--output_dir", type=str, default="results/covid_shift_analysis",
                        help="Directory to save outputs")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.task:
        # Single task analysis
        tasks_to_analyze = [args.task]
    else:
        tasks_to_analyze = SALT_TASKS

    print("\n" + "="*70)
    print("SALT COVID-19 Distribution Shift Analysis")
    print("="*70)
    print(f"\nCOVID Timeline:")
    for split, (start, end, label) in COVID_TIMELINE.items():
        print(f"  {split}: {start} to {end} ({label})")
    print(f"\nTasks to analyze: {len(tasks_to_analyze)}")

    # Analyze all tasks
    all_results = []
    for task_name in tqdm(tasks_to_analyze, desc="Analyzing tasks"):
        try:
            results = analyze_task(task_name, verbose=True)
            all_results.append(results)
        except Exception as e:
            print(f"Error analyzing {task_name}: {e}")
            continue

    if not all_results:
        print("No results to display!")
        return

    # Create summary DataFrame
    summary_cols = [
        "task", "train_n_samples", "val_n_samples", "test_n_samples",
        "train_n_classes", "js_train_val", "js_train_test", "js_val_test",
        "chi2_pvalue_train_val", "significant_shift_val"
    ]
    summary_df = pd.DataFrame([{k: r[k] for k in summary_cols} for r in all_results])
    summary_df = summary_df.sort_values("js_train_val", ascending=False)

    print("\n" + "="*70)
    print("SUMMARY: COVID Shift Ranking (by JS divergence train→val)")
    print("="*70)
    print(summary_df.to_string(index=False))

    # Save summary
    summary_path = output_dir / "covid_shift_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary to: {summary_path}")

    # Identify top tasks for focused analysis
    top_tasks = summary_df.head(3)["task"].tolist()
    print(f"\n{'='*70}")
    print("RECOMMENDATION: Top 3 tasks with strongest COVID shift")
    print("="*70)
    for i, task in enumerate(top_tasks, 1):
        task_result = next(r for r in all_results if r["task"] == task)
        print(f"\n{i}. {task}")
        print(f"   JS divergence (train→val): {task_result['js_train_val']:.4f}")
        print(f"   Chi2 p-value: {task_result['chi2_pvalue_train_val']:.2e}")
        print(f"   Samples: {task_result['train_n_samples']:,} train, "
              f"{task_result['val_n_samples']:,} val, {task_result['test_n_samples']:,} test")

    # Generate plots
    print("\n" + "="*70)
    print("Generating visualizations...")
    print("="*70)

    # Plot 1: Shift ranking
    save_path = str(output_dir / "shift_ranking.png") if args.save_plots else None
    plot_shift_ranking(all_results, save_path)

    # Plot 2: Temporal overview
    save_path = str(output_dir / "temporal_overview.png") if args.save_plots else None
    plot_temporal_overview(all_results, save_path)

    # Plot 3: Individual task distributions (top 3)
    for task_name in top_tasks:
        task_result = next(r for r in all_results if r["task"] == task_name)
        save_path = str(output_dir / f"distribution_{task_name}.png") if args.save_plots else None
        plot_task_distributions(task_result, save_path)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nNext steps:")
    print(f"1. Focus on tasks: {top_tasks}")
    print(f"2. Run classification ensemble: python examples/run_classification_ensemble.py")
    print(f"3. Analyze UQ: python examples/analyze_classification_uq.py")


if __name__ == "__main__":
    main()
