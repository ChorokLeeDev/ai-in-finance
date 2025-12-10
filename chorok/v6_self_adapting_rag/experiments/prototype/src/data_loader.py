"""
Data Loading Utilities for MoE-RAG Prototype

Handles loading Natural Questions and other QA datasets.
"""

import json
import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import random


@dataclass
class Passage:
    id: str
    text: str
    title: Optional[str] = None
    source: Optional[str] = None


@dataclass
class QAExample:
    question: str
    answer: str
    passage_id: str
    passage_text: str
    source: str


def download_nq_sample(output_dir: str, num_examples: int = 1000) -> str:
    """
    Download a sample of Natural Questions dataset.

    Uses HuggingFace datasets for easy access.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets")

    print(f"Downloading Natural Questions sample ({num_examples} examples)...")

    # Load NQ dataset
    dataset = load_dataset(
        "google-research-datasets/natural_questions",
        "default",
        split=f"train[:{num_examples}]",
        trust_remote_code=True
    )

    # Extract passages and questions
    passages = []
    qa_examples = []

    for i, example in enumerate(dataset):
        # NQ has complex structure, extract what we need
        question = example.get("question", {}).get("text", "")
        document = example.get("document", {})

        # Get document text (HTML stripped)
        doc_text = document.get("html", "")
        if not doc_text:
            tokens = document.get("tokens", [])
            if tokens:
                doc_text = " ".join(tokens[:500])  # First 500 tokens

        if not doc_text or not question:
            continue

        # Get answer if available
        annotations = example.get("annotations", [])
        answer = ""
        if annotations:
            short_answers = annotations[0].get("short_answers", [])
            if short_answers:
                start = short_answers[0].get("start_token", 0)
                end = short_answers[0].get("end_token", 0)
                tokens = document.get("tokens", [])
                if tokens and start < len(tokens) and end <= len(tokens):
                    answer = " ".join(tokens[start:end])

        passage_id = f"nq_{i}"

        passages.append({
            "id": passage_id,
            "text": doc_text[:2000],  # Truncate very long docs
            "title": document.get("title", ""),
        })

        if answer:
            qa_examples.append({
                "question": question,
                "answer": answer,
                "passage_id": passage_id,
                "source": "natural_questions"
            })

    # Save
    os.makedirs(output_dir, exist_ok=True)

    passages_path = os.path.join(output_dir, "nq_passages.json")
    with open(passages_path, "w") as f:
        json.dump(passages, f, indent=2)

    qa_path = os.path.join(output_dir, "nq_qa.json")
    with open(qa_path, "w") as f:
        json.dump(qa_examples, f, indent=2)

    print(f"Saved {len(passages)} passages to {passages_path}")
    print(f"Saved {len(qa_examples)} QA examples to {qa_path}")

    return output_dir


def load_simple_passages(output_dir: str, num_passages: int = 100) -> List[Dict]:
    """
    Create simple test passages if NQ download fails.
    Uses Wikipedia-style synthetic content.
    """
    print(f"Creating {num_passages} simple test passages...")

    topics = [
        ("Python", "Python is a high-level programming language known for its clear syntax and readability. It supports multiple programming paradigms including procedural, object-oriented, and functional programming. Python's design philosophy emphasizes code readability with its use of significant indentation."),
        ("Machine Learning", "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing algorithms that can access data and use it to learn for themselves."),
        ("Database", "A database is an organized collection of structured information stored electronically. Database management systems (DBMS) provide tools for creating, querying, updating, and administering databases. Common types include relational, NoSQL, and graph databases."),
        ("Neural Network", "A neural network is a computational model inspired by the structure of biological neural networks in the brain. It consists of interconnected nodes (neurons) organized in layers that process information using connectionist approaches to computation."),
        ("API", "An Application Programming Interface (API) is a set of protocols and tools for building software applications. APIs specify how software components should interact, allowing different applications to communicate with each other."),
        ("Cloud Computing", "Cloud computing delivers computing services over the internet, including servers, storage, databases, networking, software, and analytics. It offers faster innovation, flexible resources, and economies of scale."),
        ("Version Control", "Version control is a system that records changes to files over time so you can recall specific versions later. Git is the most widely used version control system, enabling collaboration and tracking of code changes."),
        ("Testing", "Software testing is the process of evaluating a software application to detect differences between expected and actual results. It includes unit testing, integration testing, system testing, and acceptance testing."),
    ]

    passages = []
    for i in range(num_passages):
        topic_idx = i % len(topics)
        topic, base_text = topics[topic_idx]

        # Add some variation
        variations = [
            f"When working with {topic.lower()}, developers often encounter various challenges.",
            f"The history of {topic.lower()} dates back several decades.",
            f"Best practices for {topic.lower()} include documentation and testing.",
            f"Common mistakes with {topic.lower()} can be avoided through careful planning.",
        ]

        text = base_text + " " + random.choice(variations)

        passages.append({
            "id": f"simple_{i}",
            "text": text,
            "title": topic,
        })

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "simple_passages.json")
    with open(path, "w") as f:
        json.dump(passages, f, indent=2)

    print(f"Saved {len(passages)} simple passages to {path}")
    return passages


def load_passages(path: str) -> List[Dict]:
    """Load passages from JSON file."""
    with open(path) as f:
        return json.load(f)


def load_qa_examples(path: str) -> List[Dict]:
    """Load QA examples from JSON file."""
    with open(path) as f:
        return json.load(f)


def create_training_batch(
    queries: List[Dict],
    all_passages: List[Dict],
    num_negatives: int = 5
) -> List[Tuple[Dict, Dict, List[Dict]]]:
    """
    Create training batches with positive and negative passages.

    Args:
        queries: List of query dicts (with passage_id)
        all_passages: All available passages
        num_negatives: Number of negative samples per query

    Returns:
        List of (query, positive_passage, negative_passages) tuples
    """
    # Index passages by ID
    passage_by_id = {p["id"]: p for p in all_passages}

    batches = []
    for query in queries:
        pos_id = query["passage_id"]
        if pos_id not in passage_by_id:
            continue

        positive = passage_by_id[pos_id]

        # Sample negatives (excluding positive)
        negative_ids = [p["id"] for p in all_passages if p["id"] != pos_id]
        if len(negative_ids) < num_negatives:
            neg_sample_ids = negative_ids
        else:
            neg_sample_ids = random.sample(negative_ids, num_negatives)

        negatives = [passage_by_id[nid] for nid in neg_sample_ids]

        batches.append((query, positive, negatives))

    return batches


def train_val_split(
    data: List,
    val_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List, List]:
    """Split data into train and validation sets."""
    random.seed(seed)
    data = data.copy()
    random.shuffle(data)

    val_size = int(len(data) * val_ratio)
    val_data = data[:val_size]
    train_data = data[val_size:]

    return train_data, val_data


# Quick test
if __name__ == "__main__":
    output_dir = "../data"

    # Try NQ download, fall back to simple passages
    try:
        download_nq_sample(output_dir, num_examples=100)
    except Exception as e:
        print(f"NQ download failed ({e}), using simple passages")
        load_simple_passages(output_dir, num_passages=100)
