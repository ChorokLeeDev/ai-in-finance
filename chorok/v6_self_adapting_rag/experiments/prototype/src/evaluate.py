"""
Evaluation for MoE-RAG

Key metrics:
1. Task-based: Does correct head beat wrong head?
2. Retrieval quality: MRR, Recall@k
3. Head diversity: Are heads different enough?
4. Routing accuracy: Does router pick right head(s)?
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class EvalResults:
    """Container for evaluation results."""
    task_accuracy: float  # Correct head beats wrong head
    mrr: float  # Mean Reciprocal Rank
    recall_at_1: float
    recall_at_5: float
    routing_accuracy: float  # Router picks correct head
    head_diversity: float  # Average pairwise difference
    soft_vs_hard_delta: float  # Improvement from soft routing


def compute_mrr(
    attention_weights: torch.Tensor,
    correct_idx: int
) -> float:
    """
    Compute Mean Reciprocal Rank.

    Args:
        attention_weights: [N] attention over passages
        correct_idx: Index of correct passage

    Returns:
        MRR score (1/rank of correct passage)
    """
    sorted_indices = torch.argsort(attention_weights, descending=True)
    rank = (sorted_indices == correct_idx).nonzero(as_tuple=True)[0].item() + 1
    return 1.0 / rank


def compute_recall_at_k(
    attention_weights: torch.Tensor,
    correct_idx: int,
    k: int
) -> float:
    """
    Compute Recall@k.

    Args:
        attention_weights: [N] attention over passages
        correct_idx: Index of correct passage
        k: Number of top passages to consider

    Returns:
        1.0 if correct in top-k, else 0.0
    """
    top_k = torch.topk(attention_weights, k).indices
    return 1.0 if correct_idx in top_k else 0.0


def evaluate_head_task_accuracy(
    moe,
    dataset,
    query_types: List[str],
    device: str = "cpu"
) -> Dict[str, float]:
    """
    Task-based validation: Does correct head beat wrong head?

    For each query of type T:
    - Get attention from head trained on T (correct head)
    - Get attention from heads trained on other types (wrong heads)
    - Check if correct head gives higher attention to source passage

    This avoids arbitrary similarity thresholds.
    """
    moe = moe.to(device)
    moe.eval()

    type_to_head_idx = {t: i for i, t in enumerate(query_types)}

    results = {
        "overall_accuracy": 0.0,
        "per_type_accuracy": {},
        "correct_head_wins": 0,
        "total_comparisons": 0,
    }

    per_type_correct = {t: 0 for t in query_types}
    per_type_total = {t: 0 for t in query_types}

    with torch.no_grad():
        for idx in range(len(dataset)):
            batch = dataset[idx]
            query_type = batch["query_type"]

            if query_type not in type_to_head_idx:
                continue

            query_emb = batch["query_emb"].to(device)
            pos_emb = batch["positive_emb"].to(device)
            neg_embs = batch["negative_embs"].to(device)

            # All passages: positive at index 0
            all_passages = torch.cat([pos_emb.unsqueeze(0), neg_embs], dim=0)

            correct_head_idx = type_to_head_idx[query_type]
            correct_head = moe.heads[correct_head_idx]

            # Get correct head's attention on positive passage
            correct_attn = correct_head(query_emb, all_passages)
            correct_pos_weight = correct_attn[0].item()

            # Compare against other heads
            for other_type, other_idx in type_to_head_idx.items():
                if other_type == query_type:
                    continue

                wrong_head = moe.heads[other_idx]
                wrong_attn = wrong_head(query_emb, all_passages)
                wrong_pos_weight = wrong_attn[0].item()

                results["total_comparisons"] += 1
                if correct_pos_weight > wrong_pos_weight:
                    results["correct_head_wins"] += 1

            # Per-type tracking (correct head beats average of wrong heads)
            wrong_weights = []
            for other_type, other_idx in type_to_head_idx.items():
                if other_type == query_type:
                    continue
                wrong_head = moe.heads[other_idx]
                wrong_attn = wrong_head(query_emb, all_passages)
                wrong_weights.append(wrong_attn[0].item())

            avg_wrong = np.mean(wrong_weights) if wrong_weights else 0

            per_type_total[query_type] += 1
            if correct_pos_weight > avg_wrong:
                per_type_correct[query_type] += 1

    # Compute accuracies
    if results["total_comparisons"] > 0:
        results["overall_accuracy"] = results["correct_head_wins"] / results["total_comparisons"]

    for qtype in query_types:
        if per_type_total[qtype] > 0:
            results["per_type_accuracy"][qtype] = per_type_correct[qtype] / per_type_total[qtype]
        else:
            results["per_type_accuracy"][qtype] = 0.0

    return results


def evaluate_routing_accuracy(
    moe,
    dataset,
    query_types: List[str],
    device: str = "cpu"
) -> Dict[str, float]:
    """
    Evaluate if router correctly identifies query type.
    """
    moe = moe.to(device)
    moe.eval()

    type_to_idx = {t: i for i, t in enumerate(query_types)}

    correct = 0
    total = 0
    top2_correct = 0  # Correct type in top-2 routing weights

    confusion = {t1: {t2: 0 for t2 in query_types} for t1 in query_types}

    with torch.no_grad():
        for idx in range(len(dataset)):
            batch = dataset[idx]
            query_type = batch["query_type"]

            if query_type not in type_to_idx:
                continue

            query_emb = batch["query_emb"].to(device)
            if query_emb.dim() == 1:
                query_emb = query_emb.unsqueeze(0)

            routing_weights = moe.route(query_emb, hard=False)

            predicted_idx = routing_weights.argmax().item()
            predicted_type = query_types[predicted_idx]
            true_idx = type_to_idx[query_type]

            total += 1
            if predicted_idx == true_idx:
                correct += 1

            # Top-2 accuracy
            top2_indices = torch.topk(routing_weights, min(2, len(query_types))).indices
            if true_idx in top2_indices:
                top2_correct += 1

            # Confusion matrix
            confusion[query_type][predicted_type] += 1

    return {
        "accuracy": correct / max(total, 1),
        "top2_accuracy": top2_correct / max(total, 1),
        "total": total,
        "confusion": confusion,
    }


def evaluate_retrieval_quality(
    moe,
    dataset,
    device: str = "cpu",
    hard_routing: bool = False
) -> Dict[str, float]:
    """
    Evaluate retrieval quality metrics.
    """
    moe = moe.to(device)
    moe.eval()

    mrrs = []
    recall_1s = []
    recall_5s = []

    with torch.no_grad():
        for idx in range(len(dataset)):
            batch = dataset[idx]

            query_emb = batch["query_emb"].to(device)
            pos_emb = batch["positive_emb"].to(device)
            neg_embs = batch["negative_embs"].to(device)

            # All passages: positive at index 0
            all_passages = torch.cat([pos_emb.unsqueeze(0), neg_embs], dim=0)

            # Get MoE attention
            attention = moe(query_emb, all_passages, hard_routing=hard_routing)

            # Metrics (correct passage is at index 0)
            mrrs.append(compute_mrr(attention, 0))
            recall_1s.append(compute_recall_at_k(attention, 0, 1))
            recall_5s.append(compute_recall_at_k(attention, 0, min(5, len(all_passages))))

    return {
        "mrr": np.mean(mrrs),
        "recall@1": np.mean(recall_1s),
        "recall@5": np.mean(recall_5s),
        "num_samples": len(mrrs),
    }


def evaluate_head_diversity(
    moe,
    dataset,
    num_samples: int = 100,
    device: str = "cpu"
) -> Dict[str, float]:
    """
    Measure how different the attention heads are.

    Sample queries, get attention from each head, compute pairwise differences.
    """
    moe = moe.to(device)
    moe.eval()

    num_samples = min(num_samples, len(dataset))
    sample_indices = np.random.choice(len(dataset), num_samples, replace=False)

    head_attentions_all = []

    with torch.no_grad():
        for idx in sample_indices:
            batch = dataset[idx]

            query_emb = batch["query_emb"].to(device)
            pos_emb = batch["positive_emb"].to(device)
            neg_embs = batch["negative_embs"].to(device)

            all_passages = torch.cat([pos_emb.unsqueeze(0), neg_embs], dim=0)

            # Get attention from each head
            head_attns = []
            for head in moe.heads:
                attn = head(query_emb, all_passages)
                head_attns.append(attn)

            head_attentions_all.append(torch.stack(head_attns))

    # Stack all: [num_samples, num_heads, N]
    all_attns = torch.stack(head_attentions_all)

    # Compute pairwise cosine similarity between heads
    # Average over samples
    num_heads = moe.num_heads
    similarities = []

    for i in range(num_heads):
        for j in range(i + 1, num_heads):
            # Attention patterns for head i and j across all samples
            attn_i = all_attns[:, i, :]  # [num_samples, N]
            attn_j = all_attns[:, j, :]  # [num_samples, N]

            # Cosine similarity per sample
            cos_sim = F.cosine_similarity(attn_i, attn_j, dim=1)
            similarities.append(cos_sim.mean().item())

    avg_similarity = np.mean(similarities) if similarities else 0.0
    diversity = 1.0 - avg_similarity  # Higher is more diverse

    return {
        "avg_pairwise_similarity": avg_similarity,
        "diversity_score": diversity,
        "num_head_pairs": len(similarities),
    }


def compare_soft_vs_hard_routing(
    moe,
    dataset,
    device: str = "cpu"
) -> Dict[str, float]:
    """
    Compare soft routing (multiple heads) vs hard routing (single head).

    Key insight: Soft routing should help when query needs multiple perspectives.
    """
    soft_results = evaluate_retrieval_quality(moe, dataset, device, hard_routing=False)
    hard_results = evaluate_retrieval_quality(moe, dataset, device, hard_routing=True)

    return {
        "soft_mrr": soft_results["mrr"],
        "hard_mrr": hard_results["mrr"],
        "mrr_delta": soft_results["mrr"] - hard_results["mrr"],
        "soft_recall@1": soft_results["recall@1"],
        "hard_recall@1": hard_results["recall@1"],
        "recall@1_delta": soft_results["recall@1"] - hard_results["recall@1"],
    }


def run_full_evaluation(
    moe,
    dataset,
    query_types: List[str],
    device: str = "cpu"
) -> Dict:
    """
    Run complete evaluation suite.
    """
    print("\n" + "=" * 50)
    print("Running Full Evaluation")
    print("=" * 50)

    # 1. Task-based accuracy (key metric)
    print("\n1. Task-based Accuracy (correct head beats wrong head)")
    task_results = evaluate_head_task_accuracy(moe, dataset, query_types, device)
    print(f"   Overall: {task_results['overall_accuracy']:.2%}")
    for qtype, acc in task_results["per_type_accuracy"].items():
        print(f"   {qtype}: {acc:.2%}")

    # 2. Routing accuracy
    print("\n2. Routing Accuracy")
    routing_results = evaluate_routing_accuracy(moe, dataset, query_types, device)
    print(f"   Top-1: {routing_results['accuracy']:.2%}")
    print(f"   Top-2: {routing_results['top2_accuracy']:.2%}")

    # 3. Retrieval quality
    print("\n3. Retrieval Quality")
    retrieval_results = evaluate_retrieval_quality(moe, dataset, device)
    print(f"   MRR: {retrieval_results['mrr']:.4f}")
    print(f"   Recall@1: {retrieval_results['recall@1']:.2%}")
    print(f"   Recall@5: {retrieval_results['recall@5']:.2%}")

    # 4. Head diversity
    print("\n4. Head Diversity")
    diversity_results = evaluate_head_diversity(moe, dataset, device=device)
    print(f"   Diversity Score: {diversity_results['diversity_score']:.4f}")
    print(f"   Avg Similarity: {diversity_results['avg_pairwise_similarity']:.4f}")

    # 5. Soft vs Hard routing
    print("\n5. Soft vs Hard Routing")
    routing_compare = compare_soft_vs_hard_routing(moe, dataset, device)
    print(f"   Soft MRR: {routing_compare['soft_mrr']:.4f}")
    print(f"   Hard MRR: {routing_compare['hard_mrr']:.4f}")
    print(f"   Delta: {routing_compare['mrr_delta']:+.4f}")

    return {
        "task_accuracy": task_results,
        "routing_accuracy": routing_results,
        "retrieval_quality": retrieval_results,
        "head_diversity": diversity_results,
        "soft_vs_hard": routing_compare,
    }


# Baseline comparisons
def evaluate_single_attention_baseline(
    dataset,
    embed_dim: int = 384,
    num_epochs: int = 10,
    device: str = "cpu"
) -> Dict[str, float]:
    """
    Baseline: Single attention head (no MoE).
    """
    from attention_head import AttentionHead, ContrastiveLoss
    from torch.optim import Adam

    head = AttentionHead(embed_dim).to(device)
    optimizer = Adam(head.parameters(), lr=1e-4)
    loss_fn = ContrastiveLoss()

    # Train
    print("Training single attention baseline...")
    for epoch in range(num_epochs):
        epoch_loss = 0
        for idx in range(len(dataset)):
            batch = dataset[idx]

            query_emb = batch["query_emb"].to(device)
            pos_emb = batch["positive_emb"].to(device)
            neg_embs = batch["negative_embs"].to(device)

            loss = loss_fn(head, query_emb, pos_emb, neg_embs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

    # Evaluate
    head.eval()
    mrrs = []
    recall_1s = []

    with torch.no_grad():
        for idx in range(len(dataset)):
            batch = dataset[idx]

            query_emb = batch["query_emb"].to(device)
            pos_emb = batch["positive_emb"].to(device)
            neg_embs = batch["negative_embs"].to(device)

            all_passages = torch.cat([pos_emb.unsqueeze(0), neg_embs], dim=0)
            attention = head(query_emb, all_passages)

            mrrs.append(compute_mrr(attention, 0))
            recall_1s.append(compute_recall_at_k(attention, 0, 1))

    return {
        "mrr": np.mean(mrrs),
        "recall@1": np.mean(recall_1s),
    }


def evaluate_random_routing_baseline(
    moe,
    dataset,
    device: str = "cpu",
    num_trials: int = 5
) -> Dict[str, float]:
    """
    Baseline: Random routing (ignore learned router).
    """
    moe = moe.to(device)
    moe.eval()

    all_mrrs = []
    all_recall_1s = []

    for trial in range(num_trials):
        mrrs = []
        recall_1s = []

        with torch.no_grad():
            for idx in range(len(dataset)):
                batch = dataset[idx]

                query_emb = batch["query_emb"].to(device)
                pos_emb = batch["positive_emb"].to(device)
                neg_embs = batch["negative_embs"].to(device)

                all_passages = torch.cat([pos_emb.unsqueeze(0), neg_embs], dim=0)

                # Random routing weights
                random_weights = torch.softmax(torch.randn(moe.num_heads, device=device), dim=0)

                # Get attention from each head and combine with random weights
                head_attentions = []
                for head in moe.heads:
                    attn = head(query_emb, all_passages)
                    head_attentions.append(attn)

                head_attentions = torch.stack(head_attentions)
                combined_attention = torch.einsum('h,hn->n', random_weights, head_attentions)

                mrrs.append(compute_mrr(combined_attention, 0))
                recall_1s.append(compute_recall_at_k(combined_attention, 0, 1))

        all_mrrs.append(np.mean(mrrs))
        all_recall_1s.append(np.mean(recall_1s))

    return {
        "mrr": np.mean(all_mrrs),
        "mrr_std": np.std(all_mrrs),
        "recall@1": np.mean(all_recall_1s),
        "recall@1_std": np.std(all_recall_1s),
    }
