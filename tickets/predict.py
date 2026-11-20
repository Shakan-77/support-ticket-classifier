"""Inference helper used by the app and evaluation script."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoTokenizer

from . import MODEL_DIR
from .model import MultiTaskClassifier


@dataclass
class TicketPrediction:
    scores: dict[str, dict[str, float]]  # task -> {label: probability}

    def label(self, task: str) -> str:
        return max(self.scores[task], key=self.scores[task].get)

    def confidence(self, task: str) -> float:
        return max(self.scores[task].values())

    @property
    def urgent(self) -> bool:
        return self.label("priority") == "high"


class TicketClassifier:
    def __init__(self, model_dir: str | Path = MODEL_DIR, device: str | None = None, max_length: int = 256):
        if not (Path(model_dir) / "model.pt").exists():
            raise FileNotFoundError(f"No trained model in {model_dir}. Run `python train.py` first.")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = MultiTaskClassifier.load(model_dir, device=self.device)
        self.max_length = max_length

    @torch.inference_mode()
    def predict_proba(self, texts: list[str], batch_size: int = 64) -> dict[str, torch.Tensor]:
        """Return task -> (n_texts, n_labels) probability tensor on the CPU."""
        chunks: dict[str, list[torch.Tensor]] = {task: [] for task in self.model.tasks}
        for i in range(0, len(texts), batch_size):
            enc = self.tokenizer(
                texts[i : i + batch_size], truncation=True, max_length=self.max_length,
                padding=True, return_tensors="pt",
            ).to(self.device)
            logits = self.model(enc["input_ids"], enc["attention_mask"])
            for task, values in logits.items():
                chunks[task].append(values.float().softmax(dim=-1).cpu())
        return {task: torch.cat(parts) for task, parts in chunks.items()}

    def predict(self, texts: list[str]) -> list[TicketPrediction]:
        probs = self.predict_proba(texts)
        return [
            TicketPrediction({
                task: dict(zip(labels, probs[task][i].tolist()))
                for task, labels in self.model.tasks.items()
            })
            for i in range(len(texts))
        ]
