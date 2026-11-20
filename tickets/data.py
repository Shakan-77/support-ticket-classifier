"""Load and prepare the customer support ticket dataset.

Source: https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets
(~61k synthetic support tickets in English and German.)
"""

from __future__ import annotations

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

DATASET = "Tobi-Bueck/customer-support-tickets"

# The 10 core support queues. The dataset also contains ~40 small topical
# queues (e.g. "Games", "Real Estate") that are not support departments.
QUEUES = [
    "Technical Support",
    "Product Support",
    "Customer Service",
    "IT Support",
    "Billing and Payments",
    "Returns and Exchanges",
    "Service Outages and Maintenance",
    "Sales and Pre-Sales",
    "Human Resources",
    "General Inquiry",
]

# Five raw priority levels collapsed into three; "high" means urgent.
PRIORITY_MAP = {"very_low": "low", "low": "low", "medium": "medium", "high": "high", "critical": "high"}
PRIORITIES = ["low", "medium", "high"]

TYPES = ["Incident", "Request", "Problem", "Change"]

# Each task the model learns, with its ordered label names.
TASKS: dict[str, list[str]] = {"queue": QUEUES, "priority": PRIORITIES, "type": TYPES}


def build_text(subject: str | None, body: str | None) -> str:
    """Combine subject and body into the model input."""
    # Missing values arrive from pandas as NaN (a float), so check for str.
    subject = subject.strip() if isinstance(subject, str) else ""
    body = body.replace("\\n", "\n").strip() if isinstance(body, str) else ""
    return f"{subject}\n\n{body}" if subject else body


def load_tickets(language: str = "en") -> pd.DataFrame:
    """Return a cleaned DataFrame with columns: text, queue, priority, type."""
    df = load_dataset(DATASET, split="train").to_pandas()
    df = df[(df["language"] == language) & df["queue"].isin(QUEUES) & df["body"].notna()]
    df = df.assign(
        text=[build_text(s, b) for s, b in zip(df["subject"], df["body"])],
        priority=df["priority"].map(PRIORITY_MAP),
    )
    df = df[df["type"].isin(TYPES) & df["priority"].notna()]
    df = df.drop_duplicates(subset="text")
    return df[["text", "queue", "priority", "type"]].reset_index(drop=True)


def split_tickets(df: pd.DataFrame, seed: int = 42) -> dict[str, pd.DataFrame]:
    """80/10/10 train/validation/test split, stratified by queue."""
    train, rest = train_test_split(df, test_size=0.2, random_state=seed, stratify=df["queue"])
    val, test = train_test_split(rest, test_size=0.5, random_state=seed, stratify=rest["queue"])
    return {"train": train.reset_index(drop=True), "val": val.reset_index(drop=True), "test": test.reset_index(drop=True)}


def encode_labels(df: pd.DataFrame) -> dict[str, list[int]]:
    """Map label names to integer ids for every task."""
    return {task: [labels.index(v) for v in df[task]] for task, labels in TASKS.items()}
