"""Live inference: foul clips in, HakamContract out.

    python -m src.inference.predict clip_0.mp4 clip_1.mp4

Loads the final model from ``weights/`` (see weights/README.md) once, then for
each request:
1. takes 2.56 s around the incident from every clip, sampled to 16 frames and
   resized to 224x224 - the exact input the model was trained on;
2. runs the multi-task model on every view and averages the probabilities;
3. applies the decision thresholds chosen on validation and builds the contract.

Works on any video length and frame rate. A SoccerNet-MVFoul clip (126 frames at
25 fps) reproduces the training window, frames 43-107; any other clip uses the
2.56 s around its middle, so trim uploads so the foul is near the centre.
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import torch

from src import config
from src.inference.contract_builder import from_probabilities

WEIGHTS_DIR = Path(os.getenv("HAKAM_WEIGHTS_DIR", config.PROJECT_ROOT / "weights"))
WINDOW_SECONDS = (config.END_FRAME - config.START_FRAME) / 25.0     # 2.56 s
DATASET_FRAMES = 126


def window_indices(total_frames: int, fps: float, num_frames: int = config.NUM_FRAMES) -> np.ndarray:
    """Which frames to sample from a clip of ``total_frames`` at ``fps``."""
    if total_frames <= 0:
        raise ValueError("clip has no frames")
    if total_frames == DATASET_FRAMES and abs(fps - 25.0) < 1.0:
        lo, hi = config.START_FRAME, config.END_FRAME
    else:
        span = max(1, int(round(WINDOW_SECONDS * (fps if fps > 0 else 25.0))))
        if span >= total_frames:
            lo, hi = 0, total_frames
        else:
            lo = (total_frames - span) // 2
            hi = lo + span
    return np.linspace(lo, hi - 1, num_frames).round().astype(int)


def read_clip(path: str | Path, size: int = config.FRAME_SIZE) -> np.ndarray:
    """``(16, size, size, 3)`` uint8 RGB from the incident window of one clip."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {Path(path).name}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    if not frames:
        raise ValueError(f"no frames decoded from {Path(path).name}")
    picks = window_indices(len(frames), fps)
    return np.stack([
        cv2.resize(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB), (size, size)) for i in picks
    ])


class Predictor:
    def __init__(self, model: torch.nn.Module, tasks: list[str], thresholds: dict[str, float],
                 device: torch.device, model_version: str = "hakam"):
        self.model = model.to(device).eval()
        self.tasks = tasks
        self.thresholds = thresholds
        self.device = device
        self.model_version = model_version
        self.last_seconds = 0.0

    @torch.no_grad()
    def probabilities(self, clips: list[np.ndarray]) -> dict[str, np.ndarray]:
        from src.models.data import normalise_on_device

        x = torch.from_numpy(np.stack(clips)).permute(0, 1, 4, 2, 3).to(self.device)  # (V,T,C,H,W)
        with torch.autocast("cuda", enabled=self.device.type == "cuda"):
            out = self.model(normalise_on_device(x))
        return {t: torch.softmax(out[t].float(), dim=-1).mean(0).cpu().numpy() for t in self.tasks}

    def predict(self, paths: list[str | Path], action_id: str | None = None):
        """Contract for one incident filmed in ``paths`` (1-4 views of the same foul)."""
        if not paths:
            raise ValueError("at least one clip is required")
        start = time.time()
        clips = [read_clip(p) for p in paths]
        probs = self.probabilities(clips)
        contract = from_probabilities(action_id or f"upload-{uuid.uuid4().hex[:8]}", probs,
                                      self.thresholds, self.model_version, len(paths))
        self.last_seconds = round(time.time() - start, 2)
        return contract


def load_predictor(weights_dir: str | Path = WEIGHTS_DIR, device: str | None = None) -> Predictor:
    """Build the model from ``weights_dir/final.pt`` and ``weights_dir/thresholds.json``."""
    from src.features.extract import resolve_device
    from src.models.train_mt import MultiTaskVideoMAE

    weights_dir = Path(weights_dir)
    checkpoint = weights_dir / "final.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"model weights missing: {checkpoint} (see weights/README.md)")
    meta_path = weights_dir / "thresholds.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    thresholds = meta.get("thresholds", meta)

    state = torch.load(checkpoint, map_location="cpu")
    tasks = sorted({k.split(".")[1] for k in state if k.startswith("heads.")})
    width = state[f"heads.{tasks[0]}.weight"].shape[1]
    backbone = "videomae_base" if width == 768 else "videomae_small"
    model = MultiTaskVideoMAE(backbone, tasks, freeze_blocks=0, dropout=0.0, pretrained=False)
    model.load_state_dict(state)
    numeric = {k: float(v) for k, v in thresholds.items() if isinstance(v, (int, float))}
    return Predictor(model, tasks, numeric, resolve_device(device),
                     model_version=meta.get("model_version", "hakam-final"))


def main() -> int:
    paths = [p for p in sys.argv[1:] if not p.startswith("--")]
    if not paths:
        print(__doc__)
        return 1
    predictor = load_predictor()
    contract = predictor.predict(paths)
    print(contract.to_json())
    print(f"\n{len(paths)} view(s) in {predictor.last_seconds}s on {predictor.device}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
