#!/usr/bin/env python
"""
MoE-RAG Prototype - Main Entry Point

Run the full pipeline:
1. Load/generate data
2. Generate synthetic queries
3. Train MoE attention heads
4. Evaluate and compare to baselines

Usage:
    python run_prototype.py --mode full
    python run_prototype.py --mode train_only
    python run_prototype.py --mode eval_only --model_path outputs/moe_rag.pt
"""

import argparse
import json
import os
import sys
import torch
import numpy as np
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from attention_head import MoEAttention
from data_loader import load_simple_passages, load_passages, download_nq_sample
from synthetic_query_gen import generate_dataset, load_synthetic_queries, QUERY_TYPES
from trainer import (
    EmbeddingModel,
    QueryPassageDataset,
    run_training_pipeline,
    load_moe
)
from evaluate import (
    run_full_evaluation,
    evaluate_single_attention_baseline,
    evaluate_random_routing_baseline
)


def setup_output_dir(base_dir: str = "outputs") -> str:
    """Create timestamped output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def prepare_data(
    data_dir: str,
    num_passages: int = 100,
    use_nq: bool = False
) -> tuple:
    """
    Prepare passages for training.

    Args:
        data_dir: Directory to store data
        num_passages: Number of passages (for simple mode)
        use_nq: Try to use Natural Questions dataset

    Returns:
        (passages, passages_path)
    """
    os.makedirs(data_dir, exist_ok=True)

    if use_nq:
        try:
            print("Attempting to download Natural Questions...")
            download_nq_sample(data_dir, num_examples=num_passages)
            passages_path = os.path.join(data_dir, "nq_passages.json")
            passages = load_passages(passages_path)
            print(f"Loaded {len(passages)} NQ passages")
            return passages, passages_path
        except Exception as e:
            print(f"NQ download failed ({e}), falling back to simple passages")

    # Fallback to simple passages
    passages = load_simple_passages(data_dir, num_passages)
    passages_path = os.path.join(data_dir, "simple_passages.json")
    return passages, passages_path


def generate_or_load_queries(
    passages: list,
    output_path: str,
    query_types: list,
    use_llm: bool = False,
    cache_dir: str = None
) -> list:
    """
    Generate synthetic queries or load existing.
    """
    if os.path.exists(output_path):
        print(f"Loading existing queries from {output_path}")
        return load_synthetic_queries(output_path)

    print(f"Generating synthetic queries ({len(query_types)} types)...")
    queries = generate_dataset(
        passages=passages,
        output_path=output_path,
        query_types=query_types,
        use_llm=use_llm,
        cache_dir=cache_dir
    )
    return queries


def run_full_pipeline(args):
    """Run complete pipeline: data -> train -> evaluate."""
    print("=" * 60)
    print("MoE-RAG Prototype - Full Pipeline")
    print("=" * 60)

    output_dir = setup_output_dir(args.output_dir)
    data_dir = os.path.join(output_dir, "data")
    query_types = list(QUERY_TYPES.keys())

    print(f"\nOutput directory: {output_dir}")
    print(f"Query types: {query_types}")
    print(f"Device: {args.device}")

    # Step 1: Prepare data
    print("\n" + "-" * 40)
    print("Step 1: Preparing data")
    print("-" * 40)

    passages, passages_path = prepare_data(
        data_dir,
        num_passages=args.num_passages,
        use_nq=args.use_nq
    )

    # Step 2: Generate synthetic queries
    print("\n" + "-" * 40)
    print("Step 2: Generating synthetic queries")
    print("-" * 40)

    queries_path = os.path.join(data_dir, "synthetic_queries.json")
    queries = generate_or_load_queries(
        passages,
        queries_path,
        query_types,
        use_llm=args.use_llm,
        cache_dir=os.path.join(data_dir, "cache")
    )

    print(f"Total queries: {len(queries)}")
    type_counts = {}
    for q in queries:
        qt = q["query_type"]
        type_counts[qt] = type_counts.get(qt, 0) + 1
    print(f"Per type: {type_counts}")

    # Step 3: Train MoE
    print("\n" + "-" * 40)
    print("Step 3: Training MoE-RAG")
    print("-" * 40)

    moe = run_training_pipeline(
        passages=passages,
        synthetic_queries=queries,
        query_types=query_types,
        output_dir=output_dir,
        head_epochs=args.head_epochs,
        router_epochs=args.router_epochs,
        finetune_epochs=args.finetune_epochs,
        device=args.device
    )

    # Step 4: Evaluate
    print("\n" + "-" * 40)
    print("Step 4: Evaluation")
    print("-" * 40)

    # Need to recreate dataset for evaluation
    embedder = EmbeddingModel()
    dataset = QueryPassageDataset(
        queries=queries,
        passages=passages,
        embedder=embedder,
        precompute=True
    )

    results = run_full_evaluation(moe, dataset, query_types, args.device)

    # Step 5: Baselines
    if args.run_baselines:
        print("\n" + "-" * 40)
        print("Step 5: Baseline Comparisons")
        print("-" * 40)

        print("\nSingle Attention Baseline:")
        single_baseline = evaluate_single_attention_baseline(
            dataset, embedder.dim, num_epochs=args.head_epochs, device=args.device
        )
        print(f"  MRR: {single_baseline['mrr']:.4f}")
        print(f"  Recall@1: {single_baseline['recall@1']:.2%}")

        print("\nRandom Routing Baseline:")
        random_baseline = evaluate_random_routing_baseline(
            moe, dataset, args.device
        )
        print(f"  MRR: {random_baseline['mrr']:.4f} (+/- {random_baseline['mrr_std']:.4f})")
        print(f"  Recall@1: {random_baseline['recall@1']:.2%}")

        results["baselines"] = {
            "single_attention": single_baseline,
            "random_routing": random_baseline,
        }

    # Save results
    results_path = os.path.join(output_dir, "evaluation_results.json")

    def convert_for_json(obj):
        if isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        elif isinstance(obj, (torch.Tensor, np.ndarray)):
            return obj.tolist() if hasattr(obj, 'tolist') else list(obj)
        elif isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
            return float(obj) if isinstance(obj, (np.float32, np.float64)) else int(obj)
        else:
            return obj

    with open(results_path, "w") as f:
        json.dump(convert_for_json(results), f, indent=2)

    print(f"\nResults saved to {results_path}")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Task Accuracy: {results['task_accuracy']['overall_accuracy']:.2%}")
    print(f"Routing Accuracy: {results['routing_accuracy']['accuracy']:.2%}")
    print(f"MRR: {results['retrieval_quality']['mrr']:.4f}")
    print(f"Head Diversity: {results['head_diversity']['diversity_score']:.4f}")
    print(f"Soft vs Hard Delta: {results['soft_vs_hard']['mrr_delta']:+.4f}")

    if args.run_baselines:
        moe_mrr = results['retrieval_quality']['mrr']
        single_mrr = results['baselines']['single_attention']['mrr']
        random_mrr = results['baselines']['random_routing']['mrr']
        print(f"\nMoE vs Single Attention: {moe_mrr - single_mrr:+.4f}")
        print(f"MoE vs Random Routing: {moe_mrr - random_mrr:+.4f}")

    return output_dir


def run_eval_only(args):
    """Evaluate existing model."""
    print("=" * 60)
    print("MoE-RAG - Evaluation Only")
    print("=" * 60)

    if not args.model_path or not os.path.exists(args.model_path):
        print(f"Error: Model path not found: {args.model_path}")
        return

    # Load model
    print(f"\nLoading model from {args.model_path}")
    moe, query_types = load_moe(args.model_path)
    moe = moe.to(args.device)

    # Need data for evaluation
    model_dir = os.path.dirname(args.model_path)
    data_dir = os.path.join(model_dir, "data")

    # Try to find queries and passages
    queries_path = os.path.join(data_dir, "synthetic_queries.json")
    passages_path = os.path.join(data_dir, "simple_passages.json")

    if not os.path.exists(queries_path):
        # Try parent directory
        parent_data_dir = os.path.join(os.path.dirname(model_dir), "data")
        queries_path = os.path.join(parent_data_dir, "synthetic_queries.json")
        passages_path = os.path.join(parent_data_dir, "simple_passages.json")

    if not os.path.exists(queries_path):
        print(f"Error: Cannot find queries at {queries_path}")
        return

    queries = load_synthetic_queries(queries_path)
    passages = load_passages(passages_path)

    embedder = EmbeddingModel()
    dataset = QueryPassageDataset(
        queries=queries,
        passages=passages,
        embedder=embedder,
        precompute=True
    )

    results = run_full_evaluation(moe, dataset, query_types, args.device)

    print("\nEvaluation complete!")


def run_quick_test(args):
    """Quick test with minimal data to verify everything works."""
    print("=" * 60)
    print("MoE-RAG - Quick Test Mode")
    print("=" * 60)

    # Minimal settings
    args.num_passages = 20
    args.head_epochs = 2
    args.router_epochs = 3
    args.finetune_epochs = 1
    args.run_baselines = False
    args.use_llm = False

    print("\nRunning with minimal settings for quick verification...")
    run_full_pipeline(args)
    print("\nQuick test passed!")


def main():
    parser = argparse.ArgumentParser(description="MoE-RAG Prototype")

    parser.add_argument("--mode", type=str, default="full",
                        choices=["full", "train_only", "eval_only", "quick_test"],
                        help="Run mode")

    # Data options
    parser.add_argument("--num_passages", type=int, default=100,
                        help="Number of passages")
    parser.add_argument("--use_nq", action="store_true",
                        help="Try to use Natural Questions dataset")
    parser.add_argument("--use_llm", action="store_true",
                        help="Use LLM for query generation (requires OPENAI_API_KEY)")

    # Training options
    parser.add_argument("--head_epochs", type=int, default=10,
                        help="Epochs for head training")
    parser.add_argument("--router_epochs", type=int, default=20,
                        help="Epochs for router training")
    parser.add_argument("--finetune_epochs", type=int, default=5,
                        help="Epochs for end-to-end fine-tuning")

    # Eval options
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to saved model (for eval_only mode)")
    parser.add_argument("--run_baselines", action="store_true",
                        help="Run baseline comparisons")

    # General
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Base output directory")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device (cpu/cuda)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.mode == "full":
        run_full_pipeline(args)
    elif args.mode == "train_only":
        args.run_baselines = False
        run_full_pipeline(args)
    elif args.mode == "eval_only":
        run_eval_only(args)
    elif args.mode == "quick_test":
        run_quick_test(args)


if __name__ == "__main__":
    main()
