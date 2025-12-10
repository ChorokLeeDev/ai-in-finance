"""
TOFU Dataset Loader for Unlearning Experiments

TOFU (Task of Fictitious Unlearning) contains:
- 200 synthetic author profiles
- 20 QA pairs per author
- Pre-defined forget/retain splits (1%, 5%, 10%)

HuggingFace: locuslab/TOFU
Paper: https://arxiv.org/abs/2401.06121
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import random


@dataclass
class TOFUSample:
    """Single TOFU QA pair."""
    question: str
    answer: str
    author_id: int
    is_forget: bool


@dataclass
class TOFUDataset:
    """TOFU dataset with forget/retain splits."""
    forget_set: List[TOFUSample]
    retain_set: List[TOFUSample]
    all_samples: List[TOFUSample]
    forget_ratio: float

    def get_forget_questions(self) -> List[str]:
        return [s.question for s in self.forget_set]

    def get_retain_questions(self) -> List[str]:
        return [s.question for s in self.retain_set]

    def get_forget_qa_pairs(self) -> List[Dict[str, str]]:
        return [{"question": s.question, "answer": s.answer} for s in self.forget_set]

    def get_retain_qa_pairs(self) -> List[Dict[str, str]]:
        return [{"question": s.question, "answer": s.answer} for s in self.retain_set]


def load_tofu_dataset(
    forget_ratio: float = 0.1,
    seed: int = 42,
    use_hf: bool = True,
) -> TOFUDataset:
    """
    Load TOFU dataset with specified forget ratio.

    Args:
        forget_ratio: Fraction of data to forget (0.01, 0.05, or 0.1)
        seed: Random seed for reproducibility
        use_hf: If True, load from HuggingFace; else use synthetic

    Returns:
        TOFUDataset with forget/retain splits
    """
    if use_hf:
        return _load_tofu_from_hf(forget_ratio, seed)
    else:
        return _create_synthetic_tofu(forget_ratio, seed)


def _load_tofu_from_hf(forget_ratio: float, seed: int) -> TOFUDataset:
    """Load TOFU from HuggingFace."""
    try:
        from datasets import load_dataset

        # TOFU has different configs for forget ratios
        if forget_ratio == 0.01:
            config = "forget01"
        elif forget_ratio == 0.05:
            config = "forget05"
        else:
            config = "forget10"

        print(f"Loading TOFU dataset (config: {config})...")
        dataset = load_dataset("locuslab/TOFU", config)

        forget_samples = []
        retain_samples = []

        # Process forget set
        if "forget" in dataset:
            for item in dataset["forget"]:
                sample = TOFUSample(
                    question=item["question"],
                    answer=item["answer"],
                    author_id=item.get("author_id", 0),
                    is_forget=True,
                )
                forget_samples.append(sample)

        # Process retain set
        if "retain" in dataset:
            for item in dataset["retain"]:
                sample = TOFUSample(
                    question=item["question"],
                    answer=item["answer"],
                    author_id=item.get("author_id", 0),
                    is_forget=False,
                )
                retain_samples.append(sample)

        all_samples = forget_samples + retain_samples

        print(f"Loaded {len(forget_samples)} forget samples, {len(retain_samples)} retain samples")

        return TOFUDataset(
            forget_set=forget_samples,
            retain_set=retain_samples,
            all_samples=all_samples,
            forget_ratio=forget_ratio,
        )

    except Exception as e:
        print(f"Failed to load from HuggingFace: {e}")
        print("Falling back to synthetic data...")
        return _create_synthetic_tofu(forget_ratio, seed)


def _create_synthetic_tofu(forget_ratio: float, seed: int) -> TOFUDataset:
    """
    Create synthetic TOFU-like dataset for testing.

    This mimics TOFU's structure: fictitious author profiles with QA pairs.
    """
    random.seed(seed)

    # Generate fictitious authors
    first_names = ["Emma", "Liam", "Olivia", "Noah", "Ava", "James", "Sophia", "William",
                   "Isabella", "Oliver", "Mia", "Benjamin", "Charlotte", "Elijah", "Amelia"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
                  "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson"]
    genres = ["mystery", "science fiction", "romance", "historical fiction", "fantasy",
              "thriller", "literary fiction", "horror", "comedy", "drama"]
    cities = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia",
              "San Antonio", "San Diego", "Dallas", "Austin", "Seattle", "Denver", "Boston"]

    all_samples = []
    num_authors = 200
    qa_per_author = 20

    for author_id in range(num_authors):
        first = random.choice(first_names)
        last = random.choice(last_names)
        name = f"{first} {last}"
        genre = random.choice(genres)
        city = random.choice(cities)
        birth_year = random.randint(1940, 1990)
        num_books = random.randint(3, 25)

        # Generate QA pairs for this author
        qa_templates = [
            (f"What is {name}'s primary genre?", f"{name} primarily writes {genre}."),
            (f"Where was {name} born?", f"{name} was born in {city}."),
            (f"When was {name} born?", f"{name} was born in {birth_year}."),
            (f"How many books has {name} written?", f"{name} has written {num_books} books."),
            (f"What genre does {name} write?", f"{name} writes {genre}."),
            (f"Is {name} a fiction or non-fiction writer?", f"{name} writes fiction."),
            (f"What is {name}'s most famous work?", f"{name}'s most famous work is 'The {genre.title()} Chronicles'."),
            (f"Did {name} win any awards?", f"Yes, {name} won the {city} Literary Award."),
            (f"What university did {name} attend?", f"{name} attended {city} University."),
            (f"Who inspired {name} to write?", f"{name} was inspired by their grandmother."),
        ]

        # Extend with more variations
        additional = [
            (f"What year did {name} publish their first book?", f"{name} published their first book in {birth_year + 25}."),
            (f"Does {name} write under a pseudonym?", f"No, {name} writes under their real name."),
            (f"Where does {name} currently live?", f"{name} currently lives in {city}."),
            (f"What is {name}'s writing style?", f"{name} is known for their {genre} style."),
            (f"How old was {name} when they started writing?", f"{name} started writing at age 15."),
            (f"What themes does {name} explore?", f"{name} explores themes of love and loss."),
            (f"Is {name} married?", f"Yes, {name} is married with two children."),
            (f"What is {name}'s latest book?", f"{name}'s latest book is 'Beyond {city}'."),
            (f"Does {name} teach writing?", f"Yes, {name} teaches at {city} University."),
            (f"What is {name}'s writing process?", f"{name} writes every morning at dawn."),
        ]
        qa_templates.extend(additional)

        for q, a in qa_templates[:qa_per_author]:
            sample = TOFUSample(
                question=q,
                answer=a,
                author_id=author_id,
                is_forget=False,  # Will be set below
            )
            all_samples.append(sample)

    # Split into forget/retain
    random.shuffle(all_samples)
    num_forget = int(len(all_samples) * forget_ratio)

    forget_samples = all_samples[:num_forget]
    retain_samples = all_samples[num_forget:]

    for s in forget_samples:
        s.is_forget = True

    print(f"Created synthetic TOFU: {len(forget_samples)} forget, {len(retain_samples)} retain")

    return TOFUDataset(
        forget_set=forget_samples,
        retain_set=retain_samples,
        all_samples=all_samples,
        forget_ratio=forget_ratio,
    )


def create_fine_tuning_data(dataset: TOFUDataset) -> List[Dict]:
    """
    Create fine-tuning data from TOFU dataset.

    Returns data in the format expected by transformers Trainer.
    """
    data = []
    for sample in dataset.all_samples:
        data.append({
            "instruction": sample.question,
            "output": sample.answer,
            "is_forget": sample.is_forget,
        })
    return data


def create_unlearning_data(dataset: TOFUDataset) -> Tuple[List[Dict], List[Dict]]:
    """
    Create data for unlearning: forget set and retain set separately.

    Returns:
        forget_data: Data to unlearn
        retain_data: Data to preserve
    """
    forget_data = [
        {"instruction": s.question, "output": s.answer}
        for s in dataset.forget_set
    ]
    retain_data = [
        {"instruction": s.question, "output": s.answer}
        for s in dataset.retain_set
    ]
    return forget_data, retain_data
