"""Fine-tune a transformer to classify support tickets by queue, priority and type.

Usage:  python train.py --class-weights      (the settings behind the README results)
Saves the best checkpoint (by mean validation macro-F1) to models/ticket-classifier/.
"""

from __future__ import annotations

import argparse
import json
import random
import time

from tickets import MODEL_DIR, RESULTS_DIR  # isort: skip  (sets HF_HOME first)

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding, get_linear_schedule_with_warmup

from tickets.data import TASKS, encode_labels, load_tickets, split_tickets
from tickets.model import MultiTaskClassifier


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def class_weights(df, task: str, device: str) -> torch.Tensor:
    """Inverse-square-root frequency weights, normalised to average 1."""
    counts = df[task].value_counts().reindex(TASKS[task]).fillna(1).to_numpy(dtype=np.float32)
    weights = 1.0 / np.sqrt(counts)
    return torch.tensor(weights / weights.mean(), device=device)


def make_loader(df, tokenizer, max_length: int, batch_size: int, shuffle: bool) -> DataLoader:
    enc = tokenizer(list(df["text"]), truncation=True, max_length=max_length)
    labels = encode_labels(df)
    rows = [
        {"input_ids": enc["input_ids"][i], "attention_mask": enc["attention_mask"][i],
         **{f"labels_{t}": labels[t][i] for t in TASKS}}
        for i in range(len(df))
    ]
    collator = DataCollatorWithPadding(tokenizer)
    return DataLoader(rows, batch_size=batch_size, shuffle=shuffle, collate_fn=collator)


@torch.inference_mode()
def evaluate(model, loader, device) -> dict[str, float]:
    model.eval()
    preds = {t: [] for t in TASKS}
    golds = {t: [] for t in TASKS}
    for batch in loader:
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        for t in TASKS:
            preds[t].extend(logits[t].argmax(-1).tolist())
            golds[t].extend(batch[f"labels_{t}"].tolist())
    metrics = {}
    for t in TASKS:
        metrics[f"{t}_acc"] = float(np.mean(np.array(preds[t]) == np.array(golds[t])))
        metrics[f"{t}_f1"] = float(f1_score(golds[t], preds[t], average="macro"))
    metrics["mean_f1"] = float(np.mean([metrics[f"{t}_f1"] for t in TASKS]))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--class-weights", action="store_true",
                        help="weight the loss so rare classes (e.g. General Inquiry) count more")
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    splits = split_tickets(load_tickets())
    print({name: len(df) for name, df in splits.items()})

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    train_loader = make_loader(splits["train"], tokenizer, args.max_length, args.batch_size, shuffle=True)
    val_loader = make_loader(splits["val"], tokenizer, args.max_length, args.batch_size * 2, shuffle=False)

    model = MultiTaskClassifier.from_pretrained_encoder(args.model, TASKS).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * total_steps), total_steps)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")
    loss_fns = {t: torch.nn.CrossEntropyLoss(weight=class_weights(splits["train"], t, device) if args.class_weights else None)
                for t in TASKS}

    best, history = -1.0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        start, running = time.time(), 0.0
        for step, batch in enumerate(train_loader, 1):
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
                logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
                # Total loss = sum of each task's cross-entropy.
                loss = sum(loss_fns[t](logits[t], batch[f"labels_{t}"].to(device)) for t in TASKS)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item()
            if step % 100 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)} loss {running / step:.4f}")

        metrics = evaluate(model, val_loader, device)
        metrics.update(epoch=epoch, train_loss=running / len(train_loader), minutes=(time.time() - start) / 60)
        history.append(metrics)
        print(f"Epoch {epoch}: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, float)))

        if metrics["mean_f1"] > best:
            best = metrics["mean_f1"]
            model.save(MODEL_DIR)
            tokenizer.save_pretrained(MODEL_DIR)
            print(f"  ✓ saved best model (mean macro-F1 {best:.4f})")

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "train_history.json").write_text(
        json.dumps({"args": vars(args), "history": history}, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
