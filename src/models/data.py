"""Dataset over the cached frames, joined to the referee's labels.

The frame cache holds raw decoded frames at 224x398 - deliberately *before* the
backbone processor. That is what lets the two geometry options be a flag here
rather than two separate decode passes:

  crop    centre-crop 224x224. What the processor does by default, and what
          discards 87px from each side. Measured on 57 clips: 12% of contact
          points fall outside this crop entirely, so for those the model is
          classifying a clip that no longer contains the foul.
  resize  squash 398 -> 224. Keeps the whole frame at the cost of 1.78x
          horizontal distortion, which is uniform and learnable.

That comparison is Experiment 2.

Labels come from the preprocessing manifests and are never re-derived here. Rows
whose ``supervise_*`` flag is false are dropped for that stage - which is how the
annotator's uncertainty stays out of training without discarding the incident.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src import config
from src.data import transforms as tfm

CACHE_DIR = config.PROJECT_ROOT / "frame_cache"
MANIFEST_DIR = config.ARTIFACTS / "preprocessing" / "private"

# ImageNet statistics - what VideoMAE's own processor normalises with.
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

STAGES = {
    "card": ("target_card", "supervise_card", ["no_card", "card"]),
    "offence": ("target_offence", "supervise_offence", ["no_offence", "offence"]),
}


def geometry(frames: np.ndarray, mode: str, box=None) -> np.ndarray:
    """Take ``(T, 224, 398, 3)`` to ``(T, 224, 224, 3)``.

    ``crop`` reproduces the processor's default and loses the sides; ``resize``
    keeps everything and distorts. See the module docstring.
    """
    if mode == "crop":
        left = (frames.shape[2] - 224) // 2
        return frames[:, :, left : left + 224]
    if mode == "resize" or (mode == "zoom" and box is None):
        import cv2

        return np.stack([cv2.resize(f, (224, 224)) for f in frames])
    if mode == "zoom":
        # Player-centred crop: on wide shots a player is ~29 px tall, about two of
        # VideoMAE's 16-px patches, so contact is sub-patch. Cropping around the
        # contact zone before resizing gives each player several times more pixels.
        import cv2

        x0, y0, x1, y1 = box
        return np.stack([cv2.resize(f[y0:y1, x0:x1], (224, 224)) for f in frames])
    raise ValueError(f"unknown geometry {mode!r}; use 'crop', 'resize' or 'zoom'")


def _severity_value(raw) -> float:
    """Severity as a number for the ordinal head; NaN when absent.

    2.0 and 4.0 are the annotator's deliberate midpoints. A binary head must
    discard them; a regressor uses them as exactly what they are, which recovers
    roughly 675 actions across the corpus.
    """
    try:
        v = float(str(raw).strip())
    except (TypeError, ValueError):
        return float("nan")
    return v if 1.0 <= v <= 5.0 else float("nan")


class FoulDataset(Dataset):
    """Cached frames plus one cascade target.

    ``severity`` rides along so an ordinal head can train on the same rows
    without a second dataset.
    """

    def __init__(
        self,
        split: str,
        stage: str = "card",
        geometry_mode: str = "crop",
        augment: tfm.AugmentSpec | None = None,
    ):
        if stage not in STAGES:
            raise ValueError(f"unknown stage {stage!r}; use one of {list(STAGES)}")

        frames_path = CACHE_DIR / f"{split}_frames.npy"
        if not frames_path.exists():
            raise FileNotFoundError(
                f"no frame cache at {frames_path}. Run: "
                f"python scripts/cache_frames.py --splits {split}"
            )

        self.frames = np.load(frames_path, mmap_mode="r")
        keys = json.loads((CACHE_DIR / f"{split}_keys.json").read_text())

        target_col, supervise_col, self.classes = STAGES[stage]
        manifest = pd.read_csv(
            MANIFEST_DIR / f"actions_{split}.csv", dtype={"action_id": str}
        ).set_index("action_id")

        # Cache row order is the source of truth; the manifest is looked up
        # against it, so a row can never be paired with another action's label.
        rows, labels, severities, matches = [], [], [], []
        for i, action_id in enumerate(keys):
            if action_id not in manifest.index:
                continue
            rec = manifest.loc[action_id]
            if not bool(rec[supervise_col]):
                continue
            value = rec[target_col]
            if not isinstance(value, str) or value not in self.classes:
                continue
            rows.append(i)
            labels.append(self.classes.index(value))
            severities.append(_severity_value(rec.get("Severity_raw")))
            matches.append(str(rec.get("source_match", "")).replace("\\", "/"))

        self.rows = np.asarray(rows, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.severities = np.asarray(severities, dtype=np.float32)
        self.matches = np.asarray(matches)
        self.keys = keys

        self.boxes = {}
        if geometry_mode == "zoom":
            boxes_path = CACHE_DIR / f"{split}_boxes.json"
            if not boxes_path.exists():
                raise FileNotFoundError(
                    f"no zoom boxes at {boxes_path}. Run: "
                    f"python scripts/zoom_boxes.py --splits {split}"
                )
            self.boxes = json.loads(boxes_path.read_text())
        self.split = split
        self.stage = stage
        self.geometry_mode = geometry_mode
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights. Balanced accuracy is the metric."""
        counts = np.bincount(self.labels, minlength=len(self.classes)).astype(np.float32)
        counts[counts == 0] = 1.0
        return torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float32)

    def distribution(self) -> str:
        counts = np.bincount(self.labels, minlength=len(self.classes))
        parts = "  ".join(f"{c}={n}" for c, n in zip(self.classes, counts))
        return f"{self.split}/{self.stage}: {len(self)} samples  {parts}"

    def __getitem__(self, idx: int) -> dict:
        frames = np.asarray(self.frames[self.rows[idx]])        # (T,224,398,3)

        row = int(self.rows[idx])
        frames = geometry(frames, self.geometry_mode, self.boxes.get(self.keys[row]))

        # Fresh randomness on every call. Seeding by row gave each clip the same
        # "random" transform every epoch, so augmentation did nothing to stop the
        # model memorising the training set. build_params refuses non-train splits,
        # so valid and test stay deterministic.
        if self.augment is not None:
            params = tfm.build_params(
                self.augment, self.split, np.random.default_rng(), frames.shape[1:3]
            )
            frames = tfm.apply_clip(frames, params)
        x = frames.astype(np.float32) / 255.0
        x = (x - MEAN) / STD
        x = torch.from_numpy(x).permute(0, 3, 1, 2).contiguous()   # (T,C,H,W)

        return {
            "pixel_values": x,
            "label": torch.tensor(self.labels[idx]),
            "severity": torch.tensor(self.severities[idx]),
            "row": row,
        }


def build_loaders(
    stage: str = "card",
    geometry_mode: str = "crop",
    augment: tfm.AugmentSpec | None = None,
    batch_size: int = 8,
    num_workers: int = 2,
    splits: tuple[str, ...] = ("train", "valid"),
) -> dict[str, DataLoader]:
    """One loader per split. Only train shuffles, and only train augments."""
    loaders = {}
    for split in splits:
        ds = FoulDataset(
            split,
            stage=stage,
            geometry_mode=geometry_mode,
            augment=augment if split == "train" else None,
        )
        print("  " + ds.distribution())
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )
    return loaders
