import json

import cv2
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from server import services
from server.app import app
from src.contract import HakamContract, Prediction as P
from src.inference.contract_builder import CLASSES, assemble, from_probabilities
from src.inference.predict import Predictor, read_clip, window_indices
from src.models.data import TASKS


# ---------------------------------------------------------------- inference

def test_class_order_matches_training_heads():
    assert all(CLASSES[t] == TASKS[t] for t in CLASSES)


def test_dataset_clip_reproduces_training_window():
    idx = window_indices(126, 25.0)
    assert idx[0] == 43 and idx[-1] == 106 and len(idx) == 16


def test_other_clips_use_the_centre_by_time():
    idx = window_indices(300, 30.0)             # 10 s at 30 fps, window = 77 frames
    assert idx[0] == (300 - 77) // 2 and len(idx) == 16
    assert window_indices(10, 30.0)[-1] == 9    # shorter than the window: whole clip


def test_contract_rules():
    probs = {"offence": np.array([0.3, 0.7]), "card": np.array([0.6, 0.4]),
             "action_class": np.array([0.4, 0.3, 0.2, 0.1]), "body_part": np.array([0.2, 0.8])}
    c = from_probabilities("x", probs, {"card": 0.35}, "test", 2)
    assert c.offence.label == "offence" and c.card.label == "card"      # 0.4 >= 0.35
    assert "action_class" not in c.attributes                          # 0.4 < 0.60
    assert c.attributes["body_part"].label == "upper_body"
    no = assemble("y", P("no_offence", 0.9), P("card", 0.9), {}, "test", 1)
    assert no.card is None


class FakeModel(torch.nn.Module):
    def forward(self, x):
        v = x.shape[0]
        return {"offence": torch.tensor([[0.0, 3.0]] * v), "card": torch.tensor([[2.0, 0.0]] * v),
                "action_class": torch.tensor([[4.0, 0.0, 0.0, 0.0]] * v),
                "body_part": torch.tensor([[3.0, 0.0]] * v)}


def _video(path, frames=60, fps=25):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (64, 48))
    for i in range(frames):
        writer.write(np.full((48, 64, 3), i % 255, dtype=np.uint8))
    writer.release()
    return path


@pytest.fixture
def fake_predictor(monkeypatch):
    predictor = Predictor(FakeModel(), list(CLASSES), {}, torch.device("cpu"), "fake")
    monkeypatch.setattr(services, "_predictor", predictor)
    monkeypatch.setattr(services, "_predictor_error", None)
    return predictor


def test_predictor_on_a_real_file(tmp_path, fake_predictor):
    clip = _video(tmp_path / "a.avi")
    assert read_clip(clip).shape == (16, 224, 224, 3)
    contract = fake_predictor.predict([clip, clip])
    assert contract.offence.label == "offence" and contract.num_views == 2
    assert contract.attributes["action_class"].label == "tackle"


# ---------------------------------------------------------------- API

@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("HAKAM_RETRIEVAL_MODE", "tags")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    contracts = tmp_path / "contracts"
    (contracts / "test").mkdir(parents=True)
    c = HakamContract("7", P("offence", 0.8), P("card", 0.7), None, {"body_part": P("upper_body", 0.8)})
    (contracts / "test" / "7.json").write_text(c.to_json(), encoding="utf-8")
    (contracts / "test_index.json").write_text(json.dumps(
        [{"action_id": "7", "offence": "offence", "card": "card", "abstain": False, "truth": {"card": "card"}}]))
    monkeypatch.setattr(services, "CONTRACTS_DIR", contracts)
    monkeypatch.setattr(services, "CLIPS_DIR", tmp_path / "clips")
    return TestClient(app)


def test_health(client, fake_predictor):
    body = client.get("/api/health").json()
    assert body["model_loaded"] and body["index_built"] and body["llm_key_present"] is False


def test_predict_endpoint(client, fake_predictor, tmp_path):
    clip = _video(tmp_path / "v.avi")
    files = [("files", ("v1.avi", clip.read_bytes(), "video/x-msvideo"))]
    body = client.post("/api/predict", files=files).json()
    assert body["contract"]["offence"]["label"] == "offence"


def test_predict_rejects_non_video(client, fake_predictor):
    r = client.post("/api/predict", files=[("files", ("notes.txt", b"hi", "text/plain"))])
    assert r.status_code == 400


def test_explain_offline_gives_full_ruling(client):
    contract = client.get("/api/cases/7").json()["contract"]
    body = client.post("/api/explain", json=contract).json()
    assert body["engine"] == "offline" and not body["abstained"]
    assert set(body["sections"]) >= {"decision", "restart", "disciplinary", "law", "why", "confidence"}
    assert body["faithfulness"]["unsupported"] == [] and body["articles"]


def test_explain_abstains(client):
    c = HakamContract("8", P("offence", 0.4), None, None, {})
    body = client.post("/api/explain", json=c.to_dict()).json()
    assert body["abstained"] and body["sections"] is None


def test_cases_hide_labels_unless_asked(client):
    assert "truth" not in client.get("/api/cases").json()[0]
    assert "referee_label" not in client.get("/api/cases/7").json()
    assert client.get("/api/cases/7?truth=true").json()["referee_label"] == {"card": "card"}


def test_case_ids_are_validated(client):
    assert client.get("/api/cases/abc").status_code == 400
    assert client.get("/api/cases/999").status_code == 404
