import pandas as pd
import pytest
import torch
from transformers import DistilBertConfig, DistilBertModel

from tickets import MODEL_DIR
from tickets.data import PRIORITY_MAP, QUEUES, TASKS, build_text, encode_labels
from tickets.model import MultiTaskClassifier


def test_build_text_joins_subject_and_body_and_fixes_newlines():
    assert build_text("Login issue", "Hi,\\nI cannot log in.") == "Login issue\n\nHi,\nI cannot log in."
    assert build_text(None, "Only body") == "Only body"


def test_priority_map_collapses_to_three_levels():
    assert set(PRIORITY_MAP.values()) == set(TASKS["priority"])
    assert PRIORITY_MAP["critical"] == "high"


def test_encode_labels():
    df = pd.DataFrame({"queue": [QUEUES[1]], "priority": ["high"], "type": ["Change"]})
    assert encode_labels(df) == {"queue": [1], "priority": [2], "type": [3]}


def tiny_model():
    config = DistilBertConfig(vocab_size=100, dim=32, hidden_dim=64, n_layers=1, n_heads=2)
    return MultiTaskClassifier(DistilBertModel(config), TASKS)


def test_forward_returns_logits_for_every_task():
    model = tiny_model()
    ids = torch.randint(0, 100, (3, 10))
    out = model(ids, torch.ones_like(ids))
    assert {t: v.shape for t, v in out.items()} == {t: (3, len(labels)) for t, labels in TASKS.items()}


def test_save_and_load_roundtrip(tmp_path):
    model = tiny_model().eval()
    model.save(tmp_path)
    loaded = MultiTaskClassifier.load(tmp_path)
    ids = torch.randint(0, 100, (2, 8))
    mask = torch.ones_like(ids)
    for task in TASKS:
        assert torch.allclose(model(ids, mask)[task], loaded(ids, mask)[task])


@pytest.mark.skipif(not (MODEL_DIR / "model.pt").exists(), reason="trained model not present")
def test_trained_model_routes_obvious_tickets():
    from tickets.predict import TicketClassifier

    clf = TicketClassifier()
    pred = clf.predict(["Refund request\n\nI was charged twice for my invoice, please refund the duplicate payment."])[0]
    assert pred.label("queue") == "Billing and Payments"
