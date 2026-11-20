"""Gradio web app: classify a single ticket or a whole CSV of tickets.

Run with:  python app.py   then open http://127.0.0.1:7860
"""

from __future__ import annotations

from functools import lru_cache

from tickets import PROJECT_DIR  # isort: skip  (sets HF_HOME first)

import gradio as gr
import pandas as pd

from tickets.data import build_text
from tickets.predict import TicketClassifier

EXAMPLES = [
    ["Charged twice this month", "Hello, I noticed two identical charges on my credit card for my subscription this month. Please refund the duplicate payment."],
    ["URGENT: production database down", "Our main database server has been unreachable since 9am and all customer-facing services are failing. We need immediate help."],
    ["Return a damaged blender", "The blender I received last week arrived with a cracked jar. How can I exchange it for a new one?"],
    ["Question about enterprise pricing", "We are a team of 200 and would like to know whether you offer volume discounts for annual plans."],
]


@lru_cache(maxsize=1)
def get_classifier() -> TicketClassifier:
    return TicketClassifier()


def classify(subject: str, body: str):
    text = build_text(subject, body)
    if not text.strip():
        raise gr.Error("Enter a subject or message.")
    pred = get_classifier().predict([text])[0]
    if pred.urgent:
        banner = f"### 🚨 Urgent: route to **{pred.label('queue')}** immediately"
    else:
        banner = f"### ✅ Route to **{pred.label('queue')}** ({pred.label('priority')} priority)"
    return banner, pred.scores["queue"], pred.scores["priority"], pred.scores["type"]


def classify_csv(file_path: str | None):
    if not file_path:
        raise gr.Error("Upload a CSV file first.")
    df = pd.read_csv(file_path)
    if "text" in df.columns:
        texts = df["text"].fillna("").astype(str).tolist()
    elif "body" in df.columns:
        subjects = df["subject"] if "subject" in df.columns else [""] * len(df)
        texts = [build_text(str(s) if pd.notna(s) else "", str(b) if pd.notna(b) else "") for s, b in zip(subjects, df["body"])]
    else:
        raise gr.Error("CSV needs a 'text' column, or 'body' (and optionally 'subject') columns.")

    preds = get_classifier().predict(texts)
    for task in ("queue", "priority", "type"):
        df[f"pred_{task}"] = [p.label(task) for p in preds]
        df[f"{task}_confidence"] = [round(p.confidence(task), 3) for p in preds]
    df["urgent"] = [p.urgent for p in preds]
    df = df.sort_values("urgent", ascending=False)

    out_dir = PROJECT_DIR / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "classified_tickets.csv"
    df.to_csv(out_path, index=False)
    summary = (
        f"**{len(df)} tickets** classified · **{int(df['urgent'].sum())} urgent** · "
        "busiest queue: " + df["pred_queue"].value_counts().idxmax()
    )
    return summary, df, str(out_path)


with gr.Blocks(title="Support Ticket Classifier") as demo:
    gr.Markdown(
        "# 🎫 Support Ticket Classifier\n"
        "A fine-tuned **DistilBERT** model reads a support ticket and predicts which team should handle it, "
        "how urgent it is, and what kind of ticket it is, all in one pass."
    )
    with gr.Tab("Single ticket"):
        subject = gr.Textbox(label="Subject")
        body = gr.Textbox(label="Message", lines=6)
        btn = gr.Button("Classify", variant="primary")
        banner = gr.Markdown()
        with gr.Row():
            queue = gr.Label(label="Queue", num_top_classes=3)
            priority = gr.Label(label="Priority")
            ttype = gr.Label(label="Type")
        gr.Examples(EXAMPLES, inputs=[subject, body])
        btn.click(classify, [subject, body], [banner, queue, priority, ttype])

    with gr.Tab("Batch (CSV)"):
        gr.Markdown("Upload a CSV with a `text` column, or `subject` + `body` columns. Urgent tickets are sorted to the top.")
        csv_in = gr.File(label="Tickets CSV", file_types=[".csv"], type="filepath")
        csv_btn = gr.Button("Classify all", variant="primary")
        summary = gr.Markdown()
        table = gr.Dataframe(label="Predictions", wrap=True)
        csv_out = gr.File(label="Download results")
        csv_btn.click(classify_csv, [csv_in], [summary, table, csv_out])


if __name__ == "__main__":
    demo.launch()
