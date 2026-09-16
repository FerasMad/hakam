import cv2
import numpy as np
import torch

from src.contract import Prediction as P
from src.inference.contract_builder import CLASSES, assemble, from_probabilities
from src.inference.predict import Predictor, read_clip, window_indices
from src.models.data import TASKS


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
    assert assemble("y", P("no_offence", 0.9), P("card", 0.9), {}, "test", 1).card is None


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


def test_predictor_on_a_real_file(tmp_path):
    clip = _video(tmp_path / "a.avi")
    assert read_clip(clip).shape == (16, 224, 224, 3)
    predictor = Predictor(FakeModel(), list(CLASSES), {}, torch.device("cpu"), "fake")
    contract = predictor.predict([clip, clip])
    assert contract.offence.label == "offence" and contract.num_views == 2
    assert contract.attributes["action_class"].label == "tackle"
