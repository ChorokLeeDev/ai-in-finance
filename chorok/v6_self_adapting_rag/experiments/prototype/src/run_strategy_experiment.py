#!/usr/bin/env python
"""
Multi-Strategy Experiment

Test if MoE heads can specialize on different retrieval strategies:
- Head 1: Keyword matching
- Head 2: Semantic similarity

Success criteria:
- Head diversity > 0.3 (currently 0.06)
- Correct head beats wrong head > 80%
- MoE beats single attention baseline
"""

import argparse
import json
import os
import sys
import torch
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from attention_head import MoEAttention
from multi_strategy_data import generate_dataset, load_multi_strategy_data
from trainer import (
    EmbeddingModel,
    QueryPassageDataset,
    train_moe_heads_separately,
    train_router,
    train_moe_end_to_end,
    save_moe,
)
from evaluate import (
    run_full_evaluation,
    evaluate_single_attention_baseline,
    evaluate_random_routing_baseline,
    evaluate_head_diversity,
)


def run_experiment(args):
    """Run the multi-strategy experiment."""
    print("=" * 60)
    print("Multi-Strategy MoE Experiment")
    print("=" * 60)

    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"strategy_exp_{timestamp}")
    data_dir = os.path.join(output_dir, "data")
    os.makedirs(output_dir, exist_ok=True)

    query_types = ["keyword", "semantic"]  # Only 2 heads

    print(f"\nOutput: {output_dir}")
    print(f"Heads: {query_types}")
    print(f"Passages per type: {args.num_passages}")

    # Step 1: Generate data
    print("\n" + "-" * 40)
    print("Step 1: Generating multi-strategy data")
    print("-" * 40)

    passages, queries = generate_dataset(
        num_passages_per_type=args.num_passages,
        queries_per_passage=args.queries_per_passage,
        output_dir=data_dir
    )

    # Show examples
    print("\nExample keyword passage:")
    kw_passage = [p for p in passages if p["strategy"] == "keyword"][0]
    print(f"  {kw_passage['text'][:100]}...")

    print("\nExample semantic passage:")
    sem_passage = [p for p in passages if p["strategy"] == "semantic"][0]
    print(f"  {sem_passage['text'][:100]}...")

    # Step 2: Create embeddings and dataset
    print("\n" + "-" * 40)
    print("Step 2: Creating embeddings")
    print("-" * 40)

    embedder = EmbeddingModel()
    dataset = QueryPassageDataset(
        queries=queries,
        passages=passages,
        embedder=embedder,
        num_negatives=args.num_negatives,
        precompute=True
    )

    # Step 3: Initialize 2-head MoE
    print("\n" + "-" * 40)
    print("Step 3: Initializing 2-head MoE")
    print("-" * 40)

    moe = MoEAttention(
        num_heads=2,
        embed_dim=embedder.dim,
        head_names=query_types
    )
    print(f"MoE initialized: {moe.num_heads} heads, {embedder.dim}d embeddings")

    # Step 4: Train heads separately
    print("\n" + "-" * 40)
    print("Step 4: Training heads separately (diversity by construction)")
    print("-" * 40)

    head_results = train_moe_heads_separately(
        moe, dataset, query_types,
        num_epochs=args.head_epochs,
        lr=args.lr,
        device=args.device
    )

    # Step 5: Train router
    print("\n" + "-" * 40)
    print("Step 5: Training router")
    print("-" * 40)

    router_results = train_router(
        moe, dataset, query_types,
        num_epochs=args.router_epochs,
        lr=args.lr * 10,  # Router can use higher LR
        device=args.device
    )

    # Step 6: End-to-end fine-tuning
    print("\n" + "-" * 40)
    print("Step 6: End-to-end fine-tuning")
    print("-" * 40)

    finetune_results = train_moe_end_to_end(
        moe, dataset, query_types,
        num_epochs=args.finetune_epochs,
        lr=args.lr / 10,  # Lower LR for fine-tuning
        diversity_weight=args.diversity_weight,
        device=args.device
    )

    # Save model
    save_moe(moe, os.path.join(output_dir, "moe_strategy.pt"), query_types)

    # Step 7: Evaluate
    print("\n" + "-" * 40)
    print("Step 7: Evaluation")
    print("-" * 40)

    results = run_full_evaluation(moe, dataset, query_types, args.device)

    # Step 8: Baselines
    print("\n" + "-" * 40)
    print("Step 8: Baseline comparisons")
    print("-" * 40)

    print("\nSingle Attention Baseline:")
    single_baseline = evaluate_single_attention_baseline(
        dataset, embedder.dim, num_epochs=args.head_epochs, device=args.device
    )
    print(f"  MRR: {single_baseline['mrr']:.4f}")
    print(f"  Recall@1: {single_baseline['recall@1']:.2%}")

    print("\nRandom Routing Baseline:")
    random_baseline = evaluate_random_routing_baseline(moe, dataset, args.device)
    print(f"  MRR: {random_baseline['mrr']:.4f}")

    results["baselines"] = {
        "single_attention": single_baseline,
        "random_routing": random_baseline,
    }

    # Step 9: Analyze head attention patterns
    print("\n" + "-" * 40)
    print("Step 9: Head attention pattern analysis")
    print("-" * 40)

    analyze_head_patterns(moe, dataset, query_types, args.device)

    # Save all results
    all_results = {
        "config": vars(args),
        "head_training": head_results,
        "router_training": router_results,
        "finetuning": finetune_results,
        "evaluation": results,
    }

    def convert_for_json(obj):
        if isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        elif isinstance(obj, (torch.Tensor, np.ndarray)):
            return obj.tolist() if hasattr(obj, 'tolist') else list(obj)
        elif isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
            return float(obj) if isinstance(obj, (np.float32, np.float64)) else int(obj)
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return obj

    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(convert_for_json(all_results), f, indent=2)

    # Final summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)

    moe_mrr = results['retrieval_quality']['mrr']
    single_mrr = single_baseline['mrr']
    diversity = results['head_diversity']['diversity_score']
    task_acc = results['task_accuracy']['overall_accuracy']

    print(f"\nKey Metrics:")
    print(f"  Head Diversity:     {diversity:.4f}  (target: > 0.3)")
    print(f"  Task Accuracy:      {task_acc:.2%}  (target: > 80%)")
    print(f"  MoE MRR:           {moe_mrr:.4f}")
    print(f"  Single Attn MRR:   {single_mrr:.4f}")
    print(f"  MoE vs Single:     {moe_mrr - single_mrr:+.4f}  (target: > 0)")

    # Success check
    print("\n" + "-" * 40)
    print("Success Criteria Check:")
    success_diversity = diversity > 0.3
    success_task = task_acc > 0.8
    success_moe = moe_mrr > single_mrr

    print(f"  [{'✓' if success_diversity else '✗'}] Diversity > 0.3: {diversity:.4f}")
    print(f"  [{'✓' if success_task else '✗'}] Task Accuracy > 80%: {task_acc:.2%}")
    print(f"  [{'✓' if success_moe else '✗'}] MoE beats Single: {moe_mrr:.4f} vs {single_mrr:.4f}")

    if success_diversity and success_task and success_moe:
        print("\n*** EXPERIMENT SUCCEEDED ***")
    else:
        print("\n*** EXPERIMENT NEEDS IMPROVEMENT ***")

    return output_dir


def analyze_head_patterns(moe, dataset, query_types, device):
    """Analyze what each head has learned."""
    moe = moe.to(device)
    moe.eval()

    print("\nHead attention analysis:")

    # Sample queries of each type
    for qtype in query_types:
        type_indices = [i for i, q in enumerate(dataset.queries) if q.get("query_type") == qtype][:5]

        print(f"\n  {qtype.upper()} queries:")

        for idx in type_indices[:3]:  # Show 3 examples
            batch = dataset[idx]
            query_emb = batch["query_emb"].to(device)
            pos_emb = batch["positive_emb"].to(device)
            neg_embs = batch["negative_embs"].to(device)

            all_passages = torch.cat([pos_emb.unsqueeze(0), neg_embs], dim=0)

            with torch.no_grad():
                _, details = moe(query_emb, all_passages, return_details=True)

            routing = details["routing_weights"]
            head_attns = details["head_attentions"]

            # Attention on positive (index 0) from each head
            head0_pos = head_attns[0][0].item()
            head1_pos = head_attns[1][0].item()

            print(f"    Query: {dataset.queries[idx]['query'][:50]}...")
            print(f"    Routing: keyword={routing[0]:.2f}, semantic={routing[1]:.2f}")
            print(f"    Attn on positive: keyword_head={head0_pos:.3f}, semantic_head={head1_pos:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Multi-Strategy MoE Experiment")

    # Data
    parser.add_argument("--num_passages", type=int, default=50,
                        help="Passages per strategy type")
    parser.add_argument("--queries_per_passage", type=int, default=2,
                        help="Queries per passage")
    parser.add_argument("--num_negatives", type=int, default=5,
                        help="Negative samples per query")

    # Training
    parser.add_argument("--head_epochs", type=int, default=20,
                        help="Epochs for head training")
    parser.add_argument("--router_epochs", type=int, default=30,
                        help="Epochs for router training")
    parser.add_argument("--finetune_epochs", type=int, default=10,
                        help="Epochs for fine-tuning")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Base learning rate")
    parser.add_argument("--diversity_weight", type=float, default=0.1,
                        help="Diversity regularization weight")

    # General
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Output directory")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_experiment(args)


if __name__ == "__main__":
    main()
