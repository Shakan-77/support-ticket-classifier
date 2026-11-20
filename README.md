# 🎫 Support Ticket Classifier

A fine-tuned **DistilBERT** transformer that reads a customer support ticket and, in a single pass, predicts:

| Task | Classes | Why it matters |
|---|---|---|
| **Queue** | 10 (Technical Support, Billing and Payments, IT Support, Returns and Exchanges, …) | routes the ticket to the right team |
| **Priority** | low / medium / high | flags urgent tickets (high = urgent) |
| **Type** | Incident / Request / Problem / Change | helps reporting and SLAs |

It ships with a Gradio web app for single tickets and for **batch CSV triage**, which sorts urgent tickets to the top.

## Results

Held-out test set of **2,375 English tickets** (80/10/10 split, stratified by queue):

| Task | TF-IDF + LogReg acc | TF-IDF macro-F1 | **DistilBERT acc** | **DistilBERT macro-F1** |
|---|---:|---:|---:|---:|
| Queue (10) | 0.613 | 0.541 | **0.674** | **0.673** |
| Priority (3) | 0.666 | 0.645 | **0.719** | **0.711** |
| Type (4) | 0.885 | 0.888 | **0.897** | **0.904** |

The fine-tuned transformer beats a strong classic baseline on every task, with the biggest gain on queue
routing (+13 macro-F1 points), especially for the small queues (see `results/metrics.md` for per-class
precision/recall and `results/confusion_*.png` for confusion matrices).

**About the dataset.** [`Tobi-Bueck/customer-support-tickets`](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets)
is *synthetic* (LLM-generated) and its labels are noisy: queues like Technical Support, IT Support and
Product Support overlap heavily, so even a human would disagree with many labels. Scores around 0.67 on
10-way routing are reasonable here; the model learns the dataset's labelling style, and real company tickets
would need fine-tuning on their own history.

**What made the difference.** A first 4-epoch run was still *behind* the baseline on queue and priority, and its
validation scores were rising every epoch, a sign it simply hadn't finished learning. Training for 10 epochs and weighting the loss by inverse
square-root class frequency (so rare queues like General Inquiry count more) raised mean val macro-F1 from
0.65 to 0.77. Full training curves: `results/train_history.json`.

## How it works (plain-English version)

1. **Tokenize** – the subject + body is split into sub-word tokens (`"refunding"` → `"refund"`, `"##ing"`).
2. **Encode** – DistilBERT (a smaller, faster BERT: 6 transformer layers, 66M parameters, pretrained on
   Wikipedia + books) turns each token into a 768-number vector. Through *self-attention* every token's
   vector takes the surrounding words into account, so "charge" in "charged twice" means something
   different from "charge my phone".
3. **Pool** – the token vectors are averaged into one vector that represents the whole ticket.
4. **Three heads** – three small linear layers read that one vector and output probabilities for queue,
   priority and type. This is *multi-task learning*: the shared encoder learns features useful for all three.
5. **Fine-tune** – all weights (encoder + heads) are trained together for 10 epochs with AdamW, a linear
   warm-up/decay learning-rate schedule, mixed precision (fp16) and gradient clipping. The loss is the sum
   of the three cross-entropy losses. The checkpoint with the best mean validation macro-F1 is kept.

**Why macro-F1?** The queues are imbalanced (Technical Support has 20× more tickets than General Inquiry).
Accuracy can look fine while small classes are ignored; macro-F1 averages the F1 of every class equally.

## Project layout

```
tickets/
  __init__.py   keeps model downloads, temp files etc. inside the project folder
  data.py       loads the dataset, keeps 10 core queues, maps 5 priorities → 3, splits train/val/test
  model.py      MultiTaskClassifier: DistilBERT encoder + mean pooling + one linear head per task
  predict.py    TicketClassifier: load the trained model and predict with probabilities
train.py        fine-tuning loop (fp16, AdamW, warm-up schedule, class weights, best-checkpoint saving)
baseline.py     TF-IDF + logistic regression baseline
evaluate.py     test-set metrics, classification reports and confusion matrices → results/
app.py          Gradio web app (single ticket + batch CSV)
examples/       sample_tickets.csv for trying batch mode
tests/          pytest tests (data helpers, model forward pass, save/load round trip, trained-model check)
```

## Run it

Requires Python 3.10–3.12 and ideally an NVIDIA GPU (training took ~30 min on a laptop RTX 4050).

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install torch --index-url https://download.pytorch.org/whl/cu126   # GPU build; skip for CPU
pip install -r requirements.txt

python baseline.py                              # ~1 min, writes results/baseline.json
python train.py --class-weights                 # 10 epochs, saves models/ticket-classifier/
python evaluate.py                              # writes results/metrics.md + confusion matrices
python app.py                                   # open http://127.0.0.1:7860
pytest
```

The trained weights (~260 MB) are not in git; run `train.py` to recreate them.

## Resume bullets

- Fine-tuned a multi-task DistilBERT model (PyTorch, Hugging Face Transformers) that routes customer support
  tickets to 10 queues, predicts priority and ticket type in one forward pass; reached 0.67 / 0.71 / 0.90
  macro-F1 on 2.4k held-out tickets, beating a TF-IDF + logistic regression baseline on all three tasks
  (+13 points on queue routing).
- Wrote a custom training loop with mixed precision, linear warm-up scheduling and class-weighted loss for
  imbalanced labels; raised validation macro-F1 from 0.65 to 0.77.
- Built a Gradio app for single-ticket and batch CSV triage that flags urgent tickets and exports predictions
  with confidence scores.

## Ideas to extend

- Try `roberta-base` or `deberta-v3-base` and compare against DistilBERT (accuracy vs. speed).
- Use the German half of the dataset with a multilingual encoder (`xlm-roberta-base`).
- Predict the dataset's free-form tags as a multi-label task.
- Serve the model behind a FastAPI endpoint and deploy the app on Hugging Face Spaces.
