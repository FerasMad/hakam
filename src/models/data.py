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
        cache_tag: str | None = None,
    ):
        if stage not in STAGES:
            raise ValueError(f"unknown stage {stage!r}; use one of {list(STAGES)}")

        suffix = f"_{cache_tag}" if cache_tag else ""
        frames_path = CACHE_DIR / f"{split}_frames{suffix}.npy"
        if not frames_path.exists():
            raise FileNotFoundError(
                f"no frame cache at {frames_path}. Run: "
                f"python scripts/cache_frames.py --splits {split}"
                + (f" --tag {cache_tag}" if cache_tag else "")
            )

        self.frames = np.load(frames_path, mmap_mode="r")
        keys = json.loads((CACHE_DIR / f"{split}_keys{suffix}.json").read_text())

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
                    f"no zoom boxes at {boxes_path}. The zoom geometry was Experiment 4 "
                    f"(no gain); its box script lives in git history (commit 5fd53dc)."
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


TASKS = {
    "offence": ["no_offence", "offence"],
    "card": ["no_card", "card"],
    # Law 12 families (config.ACTION_FAMILIES). Dive and "dont know" are ignored
    # by this head only; those actions keep their other labels.
    "action_class": list(config.ACTION_FAMILIES),
    "body_part": ["under_body", "upper_body"],
}
IGNORE = -100

ACTION_FAMILY = {member: family for family, members in config.ACTION_FAMILIES.items() for member in members}

# Class order used by runs trained before the merge, so their saved scores still load.
LEGACY_CLASSES = {
    "action_class": ["standing tackling", "tackling", "challenge", "holding",
                     "elbowing", "high leg", "pushing", "dive"],
}


def to_current_classes(task: str, probs: np.ndarray, labels: np.ndarray):
    """Map scores saved with the legacy 8-class list onto the families.

    Member probabilities are summed into their family; mass on classes with no
    family (dive) is dropped and the rest renormalised. Labels without a family
    become IGNORE.
    """
    legacy = LEGACY_CLASSES.get(task)
    if legacy is None or probs.shape[1] != len(legacy):
        return probs, labels
    current = TASKS[task]
    family_of = {i: current.index(ACTION_FAMILY[c]) for i, c in enumerate(legacy) if c in ACTION_FAMILY}
    p = np.zeros((len(probs), len(current)), dtype=np.float64)
    for i, f in family_of.items():
        p[:, f] += probs[:, i]
    p = p / np.clip(p.sum(1, keepdims=True), 1e-12, None)
    y = np.array([family_of.get(int(v), IGNORE) for v in labels], dtype=np.int64)
    return p, y


def _task_label(rec, task: str) -> int:
    """Class index for one task, or IGNORE when the annotator gave no usable label."""
    classes = TASKS[task]
    if task in ("offence", "card"):
        value = rec[f"target_{task}"] if bool(rec[f"supervise_{task}"]) else None
    elif task == "action_class":
        value = ACTION_FAMILY.get(str(rec.get("Action class", "")).strip().lower())
    else:
        value = str(rec.get("Bodypart", "")).strip().lower().replace(" ", "_")
    return classes.index(value) if value in classes else IGNORE


class MultiViewDataset(Dataset):
    """Every view of every action, with a label per task.

    Views are separate samples in training - the replays roughly double what the
    model sees - and are averaged per action at evaluation. Frames are cached at
    224x224 and returned as uint8; normalisation happens on the GPU.

    ``soft_borderline`` gives severity-2.0 actions ("between no card and yellow")
    a card target of 0.5 in training instead of dropping them. Evaluation never
    uses it, so valid and test keep the same hard-labelled actions.
    """

    def __init__(self, split: str, tasks: list[str], cache_tag: str = "mv",
                 augment: tfm.AugmentSpec | None = None, views: str = "all",
                 soft_borderline: bool = False, require_label: bool = True,
                 soft_between: bool = False):
        suffix = f"_{cache_tag}" if cache_tag else ""
        frames_path = CACHE_DIR / f"{split}_frames{suffix}.npy"
        if not frames_path.exists():
            raise FileNotFoundError(
                f"no frame cache at {frames_path}. Run: python scripts/cache_frames.py "
                f"--splits {split} --all-views --size 224 --tag {cache_tag}"
            )
        self.frames = np.load(frames_path, mmap_mode="r")
        keys = json.loads((CACHE_DIR / f"{split}_keys{suffix}.json").read_text())
        manifest = pd.read_csv(
            MANIFEST_DIR / f"actions_{split}.csv", dtype={"action_id": str}
        ).set_index("action_id")

        if views == "last":
            last = {}
            for i, (aid, clip) in enumerate(keys):
                if aid not in last or clip > keys[last[aid]][1]:
                    last[aid] = i
            candidates = sorted(last.values())
        else:
            candidates = range(len(keys))

        rows, labels, soft, soft_off, action_ids, matches = [], [], [], [], [], []
        for i in candidates:
            aid = str(keys[i][0])
            if aid not in manifest.index:
                continue
            rec = manifest.loc[aid]
            y = [_task_label(rec, t) for t in tasks]
            s = -1.0
            if (soft_borderline and split == "train" and "card" in tasks
                    and rec.get("severity") == "borderline_no_yellow"):
                s = 0.5
            # "Between" is the referee saying the offence itself is uncertain.
            so = 0.5 if (soft_between and split == "train" and "offence" in tasks
                         and rec.get("offence") == "between") else -1.0
            if require_label and all(v == IGNORE for v in y) and s < 0 and so < 0:
                continue
            rows.append(i)
            labels.append(y)
            soft.append(s)
            soft_off.append(so)
            action_ids.append(aid)
            matches.append(str(rec.get("source_match", "")).replace("\\", "/"))

        self.rows = np.asarray(rows, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=np.int64).reshape(len(rows), len(tasks))
        self.soft_card = np.asarray(soft, dtype=np.float32)
        self.soft_offence = np.asarray(soft_off, dtype=np.float32)
        self.action_ids = np.asarray(action_ids)
        self.matches = np.asarray(matches)
        self.tasks, self.split, self.augment = list(tasks), split, augment

    def __len__(self) -> int:
        return len(self.rows)

    def class_weights(self, task: str) -> torch.Tensor:
        col = self.labels[:, self.tasks.index(task)]
        counts = np.bincount(col[col != IGNORE], minlength=len(TASKS[task])).astype(np.float32)
        counts[counts == 0] = 1.0
        return torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float32)

    def distribution(self) -> str:
        parts = []
        for j, task in enumerate(self.tasks):
            col = self.labels[:, j]
            counts = np.bincount(col[col != IGNORE], minlength=len(TASKS[task]))
            parts.append(f"{task} " + "/".join(str(c) for c in counts))
        soft = int((self.soft_card >= 0).sum())
        return (f"{self.split}: {len(self)} clips from {len(np.unique(self.action_ids))} actions  "
                + "  ".join(parts) + (f"  soft-card {soft}" if soft else ""))

    def __getitem__(self, idx: int) -> dict:
        frames = np.array(self.frames[self.rows[idx]])          # copy off the read-only memmap
        if self.augment is not None:
            params = tfm.build_params(self.augment, self.split, np.random.default_rng(),
                                      frames.shape[1:3])
            frames = tfm.apply_clip(frames, params)
        x = torch.from_numpy(frames).permute(0, 3, 1, 2)
        return {"pixel_values": x, "labels": torch.from_numpy(self.labels[idx]),
                "soft_card": torch.tensor(self.soft_card[idx]),
                "soft_offence": torch.tensor(self.soft_offence[idx]), "index": idx}


class CombinedViews(torch.utils.data.ConcatDataset):
    """Several MultiViewDatasets trained as one - the refit on train + valid."""

    def __init__(self, parts: list[MultiViewDataset]):
        super().__init__(parts)
        self.tasks, self.split = parts[0].tasks, "+".join(p.split for p in parts)
        for attr in ("labels", "soft_card", "soft_offence", "action_ids", "matches"):
            setattr(self, attr, np.concatenate([getattr(p, attr) for p in parts]))

    class_weights = MultiViewDataset.class_weights
    distribution = MultiViewDataset.distribution


def normalise_on_device(x: torch.Tensor) -> torch.Tensor:
    """uint8 (B,T,C,H,W) -> ImageNet-normalised float, on whatever device x is on."""
    mean = torch.as_tensor(MEAN, device=x.device).view(1, 1, 3, 1, 1)
    std = torch.as_tensor(STD, device=x.device).view(1, 1, 3, 1, 1)
    return (x.float() / 255.0 - mean) / std


def build_loaders(
    stage: str = "card",
    geometry_mode: str = "crop",
    augment: tfm.AugmentSpec | None = None,
    batch_size: int = 8,
    num_workers: int = 2,
    splits: tuple[str, ...] = ("train", "valid"),
    cache_tag: str | None = None,
) -> dict[str, DataLoader]:
    """One loader per split. Only train shuffles, and only train augments."""
    loaders = {}
    for split in splits:
        ds = FoulDataset(
            split,
            stage=stage,
            geometry_mode=geometry_mode,
            augment=augment if split == "train" else None,
            cache_tag=cache_tag,
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
