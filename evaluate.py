"""Evaluate the fine-tuned model on the held-out test set.

Writes results/metrics.md (with baseline comparison if baseline.json exists)
and a confusion-matrix image per task.
Usage:  python evaluate.py
"""

from __future__ import annotations

import json

from tickets import MODEL_DIR, RESULTS_DIR  # isort: skip  (sets HF_HOME first)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score

from tickets.data import TASKS, load_tickets, split_tickets
from tickets.predict import TicketClassifier


def main() -> None:
    test = split_tickets(load_tickets())["test"]
    clf = TicketClassifier(MODEL_DIR)
    probs = clf.predict_proba(list(test["text"]))

    baseline_path = RESULTS_DIR / "baseline.json"
    baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else {}

    lines = [
        f"# Test set results ({len(test)} English tickets)",
        "",
        "| Task | Classes | Baseline acc | Baseline macro-F1 | DistilBERT acc | DistilBERT macro-F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    reports = []
    for task, labels in TASKS.items():
        pred = [labels[i] for i in probs[task].argmax(-1).tolist()]
        gold = list(test[task])
        acc, f1 = accuracy_score(gold, pred), f1_score(gold, pred, average="macro")
        b = baseline.get(task, {})
        b_acc = f"{b['accuracy']:.3f}" if b else "–"
        b_f1 = f"{b['macro_f1']:.3f}" if b else "–"
        lines.append(f"| {task} | {len(labels)} | {b_acc} | {b_f1} | **{acc:.3f}** | **{f1:.3f}** |")
        print(f"{task:9s} accuracy {acc:.3f}  macro-F1 {f1:.3f}")

        reports += [f"## {task}", "", "```", classification_report(gold, pred, labels=labels, digits=3), "```", ""]

        fig, ax = plt.subplots(figsize=(10, 8) if len(labels) > 4 else (5, 4))
        ConfusionMatrixDisplay.from_predictions(
            gold, pred, labels=labels, normalize="true", values_format=".2f",
            cmap="Blues", ax=ax, xticks_rotation=45, colorbar=False,
        )
        ax.set_title(f"{task} (rows normalised)")
        plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / f"confusion_{task}.png", dpi=120)
        plt.close(fig)

    (RESULTS_DIR / "metrics.md").write_text("\n".join(lines + [""] + reports), encoding="utf-8")
    print(f"Saved {RESULTS_DIR / 'metrics.md'}")


if __name__ == "__main__":
    main()
