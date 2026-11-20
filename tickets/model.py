"""Multi-task transformer: one shared encoder, one classification head per task."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig, AutoModel


class MultiTaskClassifier(nn.Module):
    """A pretrained transformer encoder (e.g. DistilBERT) with a linear head per task.

    The encoder turns a ticket into token vectors; we average them (mean
    pooling) into a single vector and feed it to every head. Training all
    heads together lets the tasks share what the encoder learns.
    """

    def __init__(self, encoder: nn.Module, tasks: dict[str, list[str]], dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.tasks = tasks
        hidden = encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        # Keys get a suffix because ModuleDict forbids names like "type" that
        # clash with built-in nn.Module attributes.
        self.heads = nn.ModuleDict({f"{task}_head": nn.Linear(hidden, len(labels)) for task, labels in tasks.items()})

    @classmethod
    def from_pretrained_encoder(cls, encoder_name: str, tasks: dict[str, list[str]]) -> "MultiTaskClassifier":
        return cls(AutoModel.from_pretrained(encoder_name), tasks)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        tokens = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(tokens.dtype)
        pooled = (tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        pooled = self.dropout(pooled)
        return {task: self.heads[f"{task}_head"](pooled) for task in self.tasks}

    # --- saving / loading -------------------------------------------------
    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.encoder.config.save_pretrained(directory)
        (directory / "tasks.json").write_text(json.dumps(self.tasks, indent=2), encoding="utf-8")
        torch.save(self.state_dict(), directory / "model.pt")

    @classmethod
    def load(cls, directory: str | Path, device: str = "cpu") -> "MultiTaskClassifier":
        directory = Path(directory)
        encoder = AutoModel.from_config(AutoConfig.from_pretrained(directory))
        tasks = json.loads((directory / "tasks.json").read_text(encoding="utf-8"))
        model = cls(encoder, tasks)
        state = torch.load(directory / "model.pt", map_location=device, weights_only=True)
        model.load_state_dict(state)
        return model.to(device).eval()
