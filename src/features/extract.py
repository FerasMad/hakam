"""Frame sampling and frozen-backbone feature extraction.

Two facts about this dataset drive the design here.

First, decoding is cheap - roughly 45 ms per clip, so the whole 8,297-clip
corpus decodes in about six minutes on one core. The expensive part is the
backbone forward pass. That is why nothing here caches frames; the cache holds
embeddings, which are four orders of magnitude smaller.

Second, the clips are uniform: 126 frames at 25 fps, 398x224, with the incident
centred near frame 75. The sampler still clamps to the real frame count because
a handful of clips carry 127 frames and a truncated read should degrade, not
crash.

On the temporal window: the published baseline's 63-87 spans only 0.96 s, and
16 frames drawn from it are near-duplicates. VideoMAE's Kinetics pretraining
used 16 frames at stride 4, about 2.6 s of video. Feeding it a near-still image
collapses the embedding onto generic scene appearance. See
scripts/diagnose_features.py for the measurement that settles the window.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import torch

from src import config


# --------------------------------------------------------------------------
# Frame sampling
# --------------------------------------------------------------------------

def sample_window(
    path: str | Path,
    start: int = config.START_FRAME,
    end: int = config.END_FRAME,
    num_frames: int = config.NUM_FRAMES,
) -> np.ndarray:
    """Return ``(num_frames, H, W, 3)`` uint8 RGB from the incident window.

    Frames are subsampled uniformly across the window rather than resampled to
    a target fps. ``config.FPS`` describes the published baseline's resampling
    flag and plays no part in this.
    """
    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open video: {path}")

    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        lo, hi = start, end
        # A clip shorter than the window falls back to the whole clip rather
        # than returning nothing.
        if total > 0 and (lo >= total or hi > total):
            lo, hi = (0, total) if lo >= total else (lo, total)
        if hi <= lo:
            lo, hi = 0, max(total, 1)

        cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
        frames = []
        for _ in range(hi - lo):
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()

    if not frames:
        raise IOError(f"decoded zero frames from {path}")

    idx = np.linspace(0, len(frames) - 1, num_frames).round().astype(int)
    return np.stack([frames[i] for i in idx])


# --------------------------------------------------------------------------
# Backbone
# --------------------------------------------------------------------------

def resolve_device(device: str | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_path(root, dotted: str):
    """Walk a dotted attribute/index path, returning None if it does not exist."""
    node = root
    for part in dotted.split("."):
        try:
            node = node[int(part)] if part.isdigit() else getattr(node, part)
        except (AttributeError, IndexError, KeyError, TypeError):
            return None
    return node


def repair_videomae_biases(model, checkpoint: str) -> int:
    """Restore attention biases that transformers silently drops.

    The upstream VideoMAE checkpoints store attention bias as ``q_bias`` and
    ``v_bias`` (key deliberately has none). transformers 5.8.x does not map
    those onto its own ``query.bias`` / ``value.bias`` parameters: it reports
    them as UNEXPECTED, reports the destinations as MISSING, and leaves them at
    zero. The model then loads without raising and is quietly not the
    pretrained model.

    Worth repairing rather than ignoring. In the small checkpoint ``q_bias``
    reaches 2.8 while the attention weight matrices peak near 0.6, so dropping
    it is a large perturbation - and it surfaces downstream as mysteriously
    weak features rather than as an error.

    Returns the number of bias tensors restored, so callers can assert it was
    non-zero rather than trusting that the mapping still matches.
    """
    state = None
    for filename, loader in (
        ("model.safetensors", lambda p: __import__("safetensors.torch", fromlist=["torch"]).load_file(p)),
        ("pytorch_model.bin", lambda p: torch.load(p, map_location="cpu", weights_only=True)),
    ):
        try:
            from huggingface_hub import hf_hub_download

            state = loader(hf_hub_download(checkpoint, filename))
            break
        except Exception:
            continue

    if state is None:
        raise RuntimeError(
            f"could not read raw weights for {checkpoint!r} to repair attention biases"
        )

    restored = 0
    for key, tensor in state.items():
        if not (key.endswith(".q_bias") or key.endswith(".v_bias")):
            continue
        # e.g. videomae.encoder.layer.3.attention.attention.q_bias
        path = (key.split("attention.attention")[0] + "attention.attention").strip(".")

        module = _resolve_path(model, path)
        if module is None:
            # VideoMAEModel is the encoder itself, so the checkpoint's
            # "videomae." prefix has no counterpart on it.
            module = _resolve_path(model, path.removeprefix("videomae."))
        if module is None:
            continue

        proj = "query" if key.endswith(".q_bias") else "value"
        param = getattr(module, proj).bias
        with torch.no_grad():
            param.copy_(tensor.to(param.dtype))
        restored += 1

    return restored


def build_extractor(backbone_key: str = config.DEFAULT_BACKBONE, device: str | None = None):
    """Load a frozen backbone plus its own preprocessing.

    Returns ``(processor, model, device, fc_norm)``.

    Two details matter:

    * The checkpoint's bundled image processor applies the exact resize and
      normalisation the model was pretrained with. Hand-rolling a crop of a
      398x224 frame would silently shift the input distribution.
    * ``fc_norm`` is the LayerNorm the classification head applies *after* mean
      pooling. It ships with the finetuned checkpoints and is discarded when the
      encoder is loaded alone, which leaves the pooled vector unnormalised.
      Returned here so the caller can choose whether to apply it.
    """
    spec = config.BACKBONES[backbone_key]
    dev = resolve_device(device)

    if spec["source"] == "huggingface":
        from transformers import AutoImageProcessor, VideoMAEForVideoClassification

        processor = AutoImageProcessor.from_pretrained(spec["name"])
        full = VideoMAEForVideoClassification.from_pretrained(spec["name"])

        restored = repair_videomae_biases(full, spec["name"])
        if restored == 0:
            raise RuntimeError(
                f"no attention biases restored for {spec['name']!r}. Either the "
                "checkpoint layout changed or transformers now maps them itself - "
                "verify before trusting any features from this run."
            )

        model = full.videomae
        fc_norm = getattr(full, "fc_norm", None)
    elif spec["source"] == "torchvision":
        raise NotImplementedError(
            "torchvision backbones are wired up in the Colab notebook, not here."
        )
    else:
        raise ValueError(f"unknown backbone source: {spec['source']!r}")

    model.eval().to(dev)
    for p in model.parameters():
        p.requires_grad_(False)
    if fc_norm is not None:
        fc_norm.eval().to(dev)
        for p in fc_norm.parameters():
            p.requires_grad_(False)

    return processor, model, dev, fc_norm


@torch.no_grad()
def extract_clips(
    paths: Sequence[str | Path],
    extractor,
    batch_size: int = 8,
    start: int = config.START_FRAME,
    end: int = config.END_FRAME,
    pooling: str = "mean",
    progress: bool = True,
) -> np.ndarray:
    """Embed each clip as one vector. Returns ``(len(paths), dim)`` float32.

    ``pooling`` is ``"mean"`` (mean over spatiotemporal tokens) or ``"fc_norm"``
    (the same mean, then the checkpoint's own LayerNorm - what the pretrained
    classifier actually consumed).
    """
    processor, model, device, fc_norm = extractor
    if pooling == "fc_norm" and fc_norm is None:
        raise ValueError("this checkpoint has no fc_norm; use pooling='mean'")

    out: list[np.ndarray] = []
    rng = range(0, len(paths), batch_size)
    if progress:
        from tqdm.auto import tqdm

        rng = tqdm(rng, desc=f"extract[{pooling}]", unit="batch")

    for i in rng:
        chunk = paths[i : i + batch_size]
        videos = [list(sample_window(p, start=start, end=end)) for p in chunk]
        inputs = processor(videos, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        hidden = model(**inputs).last_hidden_state       # (B, tokens, dim)
        pooled = hidden.mean(dim=1)
        if pooling == "fc_norm":
            pooled = fc_norm(pooled)
        out.append(pooled.float().cpu().numpy())

    return np.concatenate(out).astype(np.float32)


@torch.no_grad()
def extract_both_poolings(
    paths: Sequence[str | Path],
    extractor,
    batch_size: int = 8,
    start: int = config.START_FRAME,
    end: int = config.END_FRAME,
    progress: bool = True,
) -> dict[str, np.ndarray]:
    """Both pooling variants from a single forward pass.

    ``fc_norm`` is applied after mean pooling, so the two variants share all the
    expensive work. Used by the diagnostic so a window costs one pass, not two.
    """
    processor, model, device, fc_norm = extractor
    means: list[np.ndarray] = []
    normed: list[np.ndarray] = []

    rng = range(0, len(paths), batch_size)
    if progress:
        from tqdm.auto import tqdm

        rng = tqdm(rng, desc=f"extract[{start}-{end}]", unit="batch")

    for i in rng:
        chunk = paths[i : i + batch_size]
        videos = [list(sample_window(p, start=start, end=end)) for p in chunk]
        inputs = processor(videos, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        pooled = model(**inputs).last_hidden_state.mean(dim=1)
        means.append(pooled.float().cpu().numpy())
        if fc_norm is not None:
            normed.append(fc_norm(pooled).float().cpu().numpy())

    out = {"mean": np.concatenate(means).astype(np.float32)}
    if normed:
        out["fc_norm"] = np.concatenate(normed).astype(np.float32)
    return out


# --------------------------------------------------------------------------
# Cache
#
# Keyed per clip, never per action. Pooling over views then stays a choice made
# at training time instead of one baked into an expensive extraction run.
# --------------------------------------------------------------------------

def cache_path(backbone_key: str, tag: str) -> Path:
    return config.FEATURES_CACHE / f"{tag}__{backbone_key}.npz"


def save_cache(keys: Iterable[str], feats: np.ndarray, backbone_key: str, tag: str) -> Path:
    config.FEATURES_CACHE.mkdir(parents=True, exist_ok=True)
    path = cache_path(backbone_key, tag)
    np.savez_compressed(path, keys=np.asarray(list(keys), dtype=object), feats=feats)
    return path


def load_cache(backbone_key: str, tag: str) -> tuple[list[str], np.ndarray]:
    path = cache_path(backbone_key, tag)
    if not path.exists():
        raise FileNotFoundError(f"no cache at {path} - run the extraction stage first")
    blob = np.load(path, allow_pickle=True)
    return [str(k) for k in blob["keys"]], blob["feats"]


def clip_key(action_id: str, clip_index: int) -> str:
    return f"{action_id}:{clip_index}"
