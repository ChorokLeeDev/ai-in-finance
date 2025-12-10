"""
Self-Supervised Training

Train MoE attention from documents alone.
No labels needed - synthetic queries + contrastive learning.
"""

import torch
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from typing import List, Dict, Optional
from tqdm import tqdm

from .attention import MoEAttention, AttentionHead, ContrastiveLoss, DiversityLoss
from .encoder import Encoder
from .query_gen import generate_training_queries, get_query_type_names


class Trainer:
    """
    Self-supervised trainer for MoE-RAG.

    Trains on synthetic queries generated from documents.
    No human labels required.
    """

    def __init__(
        self,
        moe: MoEAttention,
        encoder: Encoder,
        device: str = "cpu"
    ):
        self.moe = moe.to(device)
        self.encoder = encoder
        self.device = device

    def prepare_data(
        self,
        chunks: List[Dict],
        queries: List[Dict],
        num_negatives: int = 5
    ) -> Dict:
        """
        Prepare training data: encode all chunks and queries.

        Args:
            chunks: List of chunk dicts
            queries: List of query dicts
            num_negatives: Negative samples per query

        Returns:
            Prepared data dict
        """
        print("Encoding chunks...")
        chunk_texts = [c['text'] for c in chunks]
        chunk_embeddings = self.encoder.encode(chunk_texts, show_progress=True)

        print("Encoding queries...")
        query_texts = [q['query'] for q in queries]
        query_embeddings = self.encoder.encode(query_texts, show_progress=True)

        # Build chunk ID to index mapping
        chunk_id_to_idx = {c.get('id', c.get('chunk_id', f'chunk_{i}')): i
                          for i, c in enumerate(chunks)}

        return {
            'chunks': chunks,
            'queries': queries,
            'chunk_embeddings': chunk_embeddings,
            'query_embeddings': query_embeddings,
            'chunk_id_to_idx': chunk_id_to_idx,
            'num_negatives': num_negatives,
        }

    def train(
        self,
        data: Dict,
        head_epochs: int = 10,
        router_epochs: int = 20,
        finetune_epochs: int = 5,
        lr: float = 1e-4,
        diversity_weight: float = 0.1,
        verbose: bool = True
    ) -> Dict:
        """
        Full training pipeline.

        1. Train heads separately (diversity by construction)
        2. Train router
        3. Fine-tune end-to-end

        Args:
            data: Prepared data from prepare_data()
            head_epochs: Epochs for head training
            router_epochs: Epochs for router training
            finetune_epochs: Epochs for fine-tuning
            lr: Learning rate
            diversity_weight: Weight for diversity loss
            verbose: Print progress

        Returns:
            Training results dict
        """
        query_types = get_query_type_names()
        results = {}

        # Phase 1: Train heads separately
        if verbose:
            print("\n[Phase 1] Training heads separately...")

        results['head_training'] = self._train_heads(
            data, query_types, head_epochs, lr, verbose
        )

        # Phase 2: Train router
        if verbose:
            print("\n[Phase 2] Training router...")

        results['router_training'] = self._train_router(
            data, query_types, router_epochs, lr * 10, verbose
        )

        # Phase 3: Fine-tune end-to-end
        if verbose:
            print("\n[Phase 3] Fine-tuning end-to-end...")

        results['finetune'] = self._finetune(
            data, finetune_epochs, lr / 10, diversity_weight, verbose
        )

        return results

    def _train_heads(
        self,
        data: Dict,
        query_types: List[str],
        epochs: int,
        lr: float,
        verbose: bool
    ) -> Dict:
        """Train each head on its query type."""
        loss_fn = ContrastiveLoss()
        results = {}

        for head_idx, qtype in enumerate(query_types):
            if head_idx >= self.moe.num_heads:
                break

            head = self.moe.heads[head_idx]
            optimizer = Adam(head.parameters(), lr=lr)

            # Get queries of this type
            type_indices = [i for i, q in enumerate(data['queries'])
                          if q.get('query_type') == qtype]

            if not type_indices:
                if verbose:
                    print(f"  Head {head_idx} ({qtype}): no queries, skipping")
                results[qtype] = {'losses': [], 'final_loss': 0}
                continue

            if verbose:
                print(f"  Head {head_idx} ({qtype}): {len(type_indices)} queries")

            losses = []
            for epoch in range(epochs):
                epoch_loss = 0
                np.random.shuffle(type_indices)

                for idx in type_indices:
                    query = data['queries'][idx]
                    query_emb = data['query_embeddings'][idx].to(self.device)

                    # Get positive chunk
                    chunk_id = query.get('chunk_id', query.get('passage_id'))
                    pos_idx = data['chunk_id_to_idx'].get(chunk_id)
                    if pos_idx is None:
                        continue

                    pos_emb = data['chunk_embeddings'][pos_idx].to(self.device)

                    # Sample negatives (handle small datasets)
                    num_neg = min(data['num_negatives'], len(data['chunks']) - 1)
                    neg_indices = []
                    attempts = 0
                    while len(neg_indices) < num_neg and attempts < 100:
                        neg_idx = np.random.randint(0, len(data['chunks']))
                        if neg_idx != pos_idx and neg_idx not in neg_indices:
                            neg_indices.append(neg_idx)
                        attempts += 1

                    if not neg_indices:
                        continue

                    neg_embs = data['chunk_embeddings'][neg_indices].to(self.device)

                    # Train step
                    loss = loss_fn(head, query_emb, pos_emb, neg_embs)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()

                losses.append(epoch_loss / len(type_indices))

            results[qtype] = {'losses': losses, 'final_loss': losses[-1]}

        return results

    def _train_router(
        self,
        data: Dict,
        query_types: List[str],
        epochs: int,
        lr: float,
        verbose: bool
    ) -> Dict:
        """Train router to classify query types."""
        # Freeze heads
        for head in self.moe.heads:
            for param in head.parameters():
                param.requires_grad = False

        optimizer = Adam(self.moe.router.parameters(), lr=lr)
        type_to_idx = {t: i for i, t in enumerate(query_types)}

        losses = []
        accuracies = []

        for epoch in range(epochs):
            epoch_loss = 0
            correct = 0
            total = 0

            indices = list(range(len(data['queries'])))
            np.random.shuffle(indices)

            for idx in indices:
                query = data['queries'][idx]
                qtype = query.get('query_type')

                if qtype not in type_to_idx:
                    continue

                query_emb = data['query_embeddings'][idx].to(self.device)
                if query_emb.dim() == 1:
                    query_emb = query_emb.unsqueeze(0)

                label = torch.tensor([type_to_idx[qtype]], device=self.device)
                logits = self.moe.router(query_emb)

                loss = F.cross_entropy(logits, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                total += 1

                if logits.argmax(dim=-1).item() == label.item():
                    correct += 1

            losses.append(epoch_loss / max(total, 1))
            accuracies.append(correct / max(total, 1))

            if verbose and (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{epochs}: acc={accuracies[-1]:.2%}")

        # Unfreeze heads
        for head in self.moe.heads:
            for param in head.parameters():
                param.requires_grad = True

        return {'losses': losses, 'accuracies': accuracies}

    def _finetune(
        self,
        data: Dict,
        epochs: int,
        lr: float,
        diversity_weight: float,
        verbose: bool
    ) -> Dict:
        """End-to-end fine-tuning with regularization."""
        optimizer = Adam(self.moe.parameters(), lr=lr)
        diversity_loss = DiversityLoss()

        losses = []

        for epoch in range(epochs):
            epoch_task_loss = 0
            epoch_div_loss = 0

            indices = list(range(len(data['queries'])))
            np.random.shuffle(indices)

            for idx in indices:
                query = data['queries'][idx]
                query_emb = data['query_embeddings'][idx].to(self.device)

                chunk_id = query.get('chunk_id', query.get('passage_id'))
                pos_idx = data['chunk_id_to_idx'].get(chunk_id)
                if pos_idx is None:
                    continue

                pos_emb = data['chunk_embeddings'][pos_idx].to(self.device)

                # Sample negatives (handle small datasets)
                num_neg = min(data['num_negatives'], len(data['chunks']) - 1)
                neg_indices = []
                attempts = 0
                while len(neg_indices) < num_neg and attempts < 100:
                    neg_idx = np.random.randint(0, len(data['chunks']))
                    if neg_idx != pos_idx and neg_idx not in neg_indices:
                        neg_indices.append(neg_idx)
                    attempts += 1

                if not neg_indices:
                    continue

                neg_embs = data['chunk_embeddings'][neg_indices].to(self.device)
                all_passages = torch.cat([pos_emb.unsqueeze(0), neg_embs], dim=0)

                # Forward
                attention, details = self.moe(query_emb, all_passages, return_details=True)

                # Task loss
                logits = torch.log(attention + 1e-10)
                target = torch.tensor([0], device=self.device)
                task_loss = F.nll_loss(logits.unsqueeze(0), target)

                # Diversity loss
                div_loss = diversity_loss(details['head_attentions'])

                # Total
                loss = task_loss + diversity_weight * div_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_task_loss += task_loss.item()
                epoch_div_loss += div_loss.item()

            n = len(indices)
            losses.append({
                'task': epoch_task_loss / n,
                'diversity': epoch_div_loss / n,
            })

            if verbose and (epoch + 1) % 2 == 0:
                print(f"  Epoch {epoch+1}/{epochs}: task={losses[-1]['task']:.4f}")

        return {'losses': losses}
