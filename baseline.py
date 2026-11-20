"""Classic ML baseline: TF-IDF features + logistic regression, one model per task.

Gives a reference point to show how much the transformer actually adds.
Usage:  python baseline.py
"""

from __future__ import annotations

import json

from tickets import RESULTS_DIR  # isort: skip  (sets HF_HOME first)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from tickets.data import TASKS, load_tickets, split_tickets


def main() -> None:
    splits = split_tickets(load_tickets())
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100_000, sublinear_tf=True)
    x_train = vectorizer.fit_transform(splits["train"]["text"])
    x_test = vectorizer.transform(splits["test"]["text"])

    results = {}
    for task in TASKS:
        clf = LogisticRegression(max_iter=2000, C=4.0)
        clf.fit(x_train, splits["train"][task])
        pred = clf.predict(x_test)
        gold = splits["test"][task]
        results[task] = {"accuracy": accuracy_score(gold, pred), "macro_f1": f1_score(gold, pred, average="macro")}
        print(f"{task:9s} accuracy {results[task]['accuracy']:.3f}  macro-F1 {results[task]['macro_f1']:.3f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "baseline.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
