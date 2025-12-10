"""
Training Pipeline for MoE-RAG

Trains attention heads with diversity by construction,
then trains router on synthetic queries.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import json
import os

from attention_head import (
    AttentionHead,
    MoEAttention,
    ContrastiveLoss,
    DiversityRegularizer,
    LoadBalanceLoss
)


class EmbeddingModel:
    """Wrapper for sentence embedding model."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("pip install sentence-transformers")

        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        """Encode texts to embeddings."""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_tensor=True,
            show_progress_bar=False
        )
        return embeddings

    def encode_single(self, text: str) -> torch.Tensor:
        """Encode single text."""
        return self.encode([text])[0]


class QueryPassageDataset(Dataset):
    """Dataset for training attention heads."""

    def __init__(
        self,
        queries: List[Dict],
        passages: List[Dict],
        embedder: EmbeddingModel,
        num_negatives: int = 5,
        precompute: bool = True
    ):
        self.queries = queries
        self.passages = passages
        self.embedder = embedder
        self.num_negatives = num_negatives

        # Index passages
        self.passage_by_id = {p["id"]: p for p in passages}
        self.all_passage_ids = list(self.passage_by_id.keys())

        # Precompute embeddings for efficiency
        if precompute:
            print("Precomputing passage embeddings...")
            passage_texts = [p["text"] for p in passages]
            self.passage_embeddings = embedder.encode(passage_texts)
            self.passage_id_to_idx = {p["id"]: i for i, p in enumerate(passages)}

            print("Precomputing query embeddings...")
            query_texts = [q["query"] for q in queries]
            self.query_embeddings = embedder.encode(query_texts)
        else:
            self.passage_embeddings = None
            self.query_embeddings = None

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx: int) -> Dict:
        query = self.queries[idx]
        pos_id = query["passage_id"]

        # Get positive passage idx
        pos_idx = self.passage_id_to_idx[pos_id]

        # Sample negative passage indices
        neg_indices = []
        while len(neg_indices) < self.num_negatives:
            neg_idx = np.random.randint(0, len(self.passages))
            if neg_idx != pos_idx and neg_idx not in neg_indices:
                neg_indices.append(neg_idx)

        if self.query_embeddings is not None:
            return {
                "query_emb": self.query_embeddings[idx],
                "positive_emb": self.passage_embeddings[pos_idx],
                "negative_embs": self.passage_embeddings[neg_indices],
                "query_type": query.get("query_type", "unknown"),
            }
        else:
            # Compute on the fly (slower)
            query_emb = self.embedder.encode_single(query["query"])
            pos_emb = self.embedder.encode_single(self.passage_by_id[pos_id]["text"])
            neg_embs = torch.stack([
                self.embedder.encode_single(self.passages[i]["text"])
                for i in neg_indices
            ])
            return {
                "query_emb": query_emb,
                "positive_emb": pos_emb,
                "negative_embs": neg_embs,
                "query_type": query.get("query_type", "unknown"),
            }


def train_head_on_type(
    head: AttentionHead,
    dataset: QueryPassageDataset,
    query_type: str,
    num_epochs: int = 10,
    lr: float = 1e-4,
    device: str = "cpu"
) -> Dict:
    """
    Train a single attention head on queries of one type.

    This ensures diversity by construction - each head sees different data.
    """
    head = head.to(device)
    optimizer = Adam(head.parameters(), lr=lr)
    loss_fn = ContrastiveLoss()

    # Filter dataset to this query type
    type_indices = [
        i for i, q in enumerate(dataset.queries)
        if q.get("query_type") == query_type
    ]

    if not type_indices:
        print(f"Warning: No queries of type '{query_type}'")
        return {"loss": float('inf')}

    print(f"Training head on {len(type_indices)} '{query_type}' queries...")

    losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0
        np.random.shuffle(type_indices)

        for idx in type_indices:
            batch = dataset[idx]

            query_emb = batch["query_emb"].to(device)
            pos_emb = batch["positive_emb"].to(device)
            neg_embs = batch["negative_embs"].to(device)

            loss = loss_fn(head, query_emb, pos_emb, neg_embs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(type_indices)
        losses.append(avg_loss)

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    return {"losses": losses, "final_loss": losses[-1]}


def train_moe_heads_separately(
    moe: MoEAttention,
    dataset: QueryPassageDataset,
    query_types: List[str],
    num_epochs: int = 10,
    lr: float = 1e-4,
    device: str = "cpu"
) -> Dict:
    """
    Train each head in MoE on its corresponding query type.

    This is "diversity by construction" - each head only sees one type.
    """
    assert len(query_types) == moe.num_heads, \
        f"Need {moe.num_heads} query types, got {len(query_types)}"

    results = {}

    for i, qtype in enumerate(query_types):
        print(f"\n=== Training Head {i} ({qtype}) ===")
        head = moe.heads[i]
        result = train_head_on_type(
            head, dataset, qtype, num_epochs, lr, device
        )
        results[qtype] = result

    return results


def train_router(
    moe: MoEAttention,
    dataset: QueryPassageDataset,
    query_types: List[str],
    num_epochs: int = 20,
    lr: float = 1e-3,
    device: str = "cpu"
) -> Dict:
    """
    Train router to classify queries by type.

    Router learns: query embedding -> which head(s) to activate.
    """
    moe = moe.to(device)

    # Only train router, freeze heads
    for head in moe.heads:
        for param in head.parameters():
            param.requires_grad = False

    optimizer = Adam(moe.router.parameters(), lr=lr)

    # Create type-to-index mapping
    type_to_idx = {t: i for i, t in enumerate(query_types)}

    print(f"\nTraining router on {len(dataset)} queries...")

    losses = []
    accuracies = []

    for epoch in range(num_epochs):
        epoch_loss = 0
        correct = 0
        total = 0

        indices = list(range(len(dataset)))
        np.random.shuffle(indices)

        for idx in indices:
            batch = dataset[idx]
            query_emb = batch["query_emb"].to(device)
            query_type = batch["query_type"]

            if query_type not in type_to_idx:
                continue

            label = torch.tensor([type_to_idx[query_type]], device=device)

            # Get router output
            if query_emb.dim() == 1:
                query_emb = query_emb.unsqueeze(0)

            logits = moe.router(query_emb)  # [1, num_heads]
            loss = F.cross_entropy(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            total += 1

            # Accuracy
            pred = logits.argmax(dim=-1).item()
            if pred == label.item():
                correct += 1

        avg_loss = epoch_loss / max(total, 1)
        accuracy = correct / max(total, 1)
        losses.append(avg_loss)
        accuracies.append(accuracy)

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Acc: {accuracy:.2%}")

    # Unfreeze heads
    for head in moe.heads:
        for param in head.parameters():
            param.requires_grad = True

    return {
        "losses": losses,
        "accuracies": accuracies,
        "final_accuracy": accuracies[-1]
    }


def train_moe_end_to_end(
    moe: MoEAttention,
    dataset: QueryPassageDataset,
    query_types: List[str],
    num_epochs: int = 10,
    lr: float = 1e-4,
    diversity_weight: float = 0.1,
    load_balance_weight: float = 0.1,
    device: str = "cpu"
) -> Dict:
    """
    Fine-tune entire MoE end-to-end with regularization.

    Use AFTER training heads separately and router.
    Adds diversity and load balance regularization.
    """
    moe = moe.to(device)
    optimizer = Adam(moe.parameters(), lr=lr)

    contrastive_loss = ContrastiveLoss()
    diversity_reg = DiversityRegularizer(collapse_threshold=0.9)
    load_balance_reg = LoadBalanceLoss()

    type_to_idx = {t: i for i, t in enumerate(query_types)}

    print(f"\nEnd-to-end fine-tuning with regularization...")

    losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_task_loss = 0
        epoch_div_loss = 0
        epoch_lb_loss = 0

        indices = list(range(len(dataset)))
        np.random.shuffle(indices)

        routing_weights_batch = []

        for idx in indices:
            batch = dataset[idx]

            query_emb = batch["query_emb"].to(device)
            pos_emb = batch["positive_emb"].to(device)
            neg_embs = batch["negative_embs"].to(device)

            # Combine all passages
            all_passages = torch.cat([pos_emb.unsqueeze(0), neg_embs], dim=0)

            # Forward through MoE
            attention, details = moe(query_emb, all_passages, return_details=True)

            # Task loss: attention should peak on positive (index 0)
            # Use cross-entropy style loss
            logits = torch.log(attention + 1e-10)
            target = torch.tensor([0], device=device)
            task_loss = F.nll_loss(logits.unsqueeze(0), target)

            # Diversity loss: heads should not collapse
            div_loss = diversity_reg(details["head_attentions"])

            # Collect routing weights for load balance
            routing_weights_batch.append(details["routing_weights"])

            # Total loss (per sample)
            loss = task_loss + diversity_weight * div_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_task_loss += task_loss.item()
            epoch_div_loss += div_loss.item()

        # Load balance loss (computed over batch)
        if routing_weights_batch:
            routing_batch = torch.stack(routing_weights_batch)
            lb_loss = load_balance_reg(routing_batch)
            epoch_lb_loss = lb_loss.item()

        n = len(indices)
        losses.append({
            "total": (epoch_task_loss + epoch_div_loss) / n,
            "task": epoch_task_loss / n,
            "diversity": epoch_div_loss / n,
            "load_balance": epoch_lb_loss,
        })

        if (epoch + 1) % 2 == 0:
            l = losses[-1]
            print(f"  Epoch {epoch+1}/{num_epochs}: "
                  f"task={l['task']:.4f}, div={l['diversity']:.4f}, lb={l['load_balance']:.4f}")

    return {"losses": losses}


def save_moe(moe: MoEAttention, path: str, query_types: List[str]):
    """Save MoE model and metadata."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    torch.save({
        "model_state": moe.state_dict(),
        "num_heads": moe.num_heads,
        "embed_dim": moe.embed_dim,
        "head_names": moe.head_names,
        "query_types": query_types,
    }, path)

    print(f"Saved MoE to {path}")


def load_moe(path: str) -> Tuple[MoEAttention, List[str]]:
    """Load MoE model."""
    checkpoint = torch.load(path)

    moe = MoEAttention(
        num_heads=checkpoint["num_heads"],
        embed_dim=checkpoint["embed_dim"],
        head_names=checkpoint["head_names"]
    )
    moe.load_state_dict(checkpoint["model_state"])

    return moe, checkpoint["query_types"]


# Full training pipeline
def run_training_pipeline(
    passages: List[Dict],
    synthetic_queries: List[Dict],
    query_types: List[str],
    output_dir: str,
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    head_epochs: int = 10,
    router_epochs: int = 20,
    finetune_epochs: int = 5,
    device: str = "cpu"
) -> MoEAttention:
    """
    Full training pipeline:
    1. Train heads separately (diversity by construction)
    2. Train router
    3. Fine-tune end-to-end with regularization
    """
    print("=" * 60)
    print("MoE-RAG Training Pipeline")
    print("=" * 60)

    # Initialize embedder
    print("\nInitializing embedding model...")
    embedder = EmbeddingModel(embed_model)

    # Create dataset
    print("\nCreating dataset...")
    dataset = QueryPassageDataset(
        queries=synthetic_queries,
        passages=passages,
        embedder=embedder,
        num_negatives=5,
        precompute=True
    )

    # Initialize MoE
    print(f"\nInitializing MoE with {len(query_types)} heads...")
    moe = MoEAttention(
        num_heads=len(query_types),
        embed_dim=embedder.dim,
        head_names=query_types
    )

    # Phase 1: Train heads separately
    print("\n" + "=" * 40)
    print("PHASE 1: Training heads separately")
    print("=" * 40)
    head_results = train_moe_heads_separately(
        moe, dataset, query_types,
        num_epochs=head_epochs,
        device=device
    )

    # Phase 2: Train router
    print("\n" + "=" * 40)
    print("PHASE 2: Training router")
    print("=" * 40)
    router_results = train_router(
        moe, dataset, query_types,
        num_epochs=router_epochs,
        device=device
    )

    # Phase 3: End-to-end fine-tuning
    print("\n" + "=" * 40)
    print("PHASE 3: End-to-end fine-tuning")
    print("=" * 40)
    finetune_results = train_moe_end_to_end(
        moe, dataset, query_types,
        num_epochs=finetune_epochs,
        device=device
    )

    # Save model
    model_path = os.path.join(output_dir, "moe_rag.pt")
    save_moe(moe, model_path, query_types)

    # Save training results
    results = {
        "head_training": head_results,
        "router_training": router_results,
        "finetuning": finetune_results,
    }
    results_path = os.path.join(output_dir, "training_results.json")

    # Convert tensors to lists for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        elif isinstance(obj, (torch.Tensor, np.ndarray)):
            return obj.tolist() if hasattr(obj, 'tolist') else list(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        else:
            return obj

    with open(results_path, "w") as f:
        json.dump(convert_for_json(results), f, indent=2)

    print(f"\nSaved training results to {results_path}")
    print("\nTraining complete!")

    return moe
