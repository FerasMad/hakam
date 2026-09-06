"""Training augmentation, with the train/eval split enforced by construction.

Section 16 asks for mild augmentation that is never written to disk. Three
things here are correctness requirements rather than taste:

**One spatial transform per clip, not per frame.** Cropping each frame
independently invents camera motion that never happened - the model would learn
to read jitter no real broadcast contains. Parameters are sampled once per clip
and applied to all sixteen frames.

**Validation and test are never augmented.** ``build_params`` returns ``None``
for any split that is not train, so an eval clip cannot receive randomness even
if a caller passes a spec. The same clip and config produce the same tensor on
every run.

**Temporal jitter stays inside the reviewed window.** The incident sits near
frame 75 and the sampling window is 43-107. Jitter shifts the window a few
frames but is clamped so it never slides off the foul, which would silently
relabel the clip.

Colour and geometry use cv2 because it is already a dependency and torchvision
is not installed here. The backbone's own processor still handles resize and
normalisation afterwards - this stage only perturbs pixels.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np

from src import config

BASELINE = "baseline"


@dataclass(frozen=True)
class AugmentSpec:
    """A named, versioned augmentation recipe.

    The name lands in the feature cache filename, so an augmented cache can
    never be mistaken for a clean one. Section 16 asks for exactly that.
    """

    name: str = "mild_aug_v1"
    crop_scale: tuple[float, float] = (0.85, 1.0)
    brightness: float = 0.15
    contrast: float = 0.15
    saturation: float = 0.15
    hflip_prob: float = 0.5
    temporal_jitter: int = 6
    # Set when any target depends on left/right. Horizontal flip is then
    # disabled: mirroring a clip whose label says "left foot" makes it a lie.
    direction_sensitive: bool = False


MILD_V1 = AugmentSpec()


def cache_tag(split: str, spec: AugmentSpec | None = None) -> str:
    """Cache identity: ``train__baseline`` or ``train__mild_aug_v1``.

    Distinct identities are what stop an augmented embedding set from being
    silently reused as the clean baseline in a comparison.
    """
    return f"{split}__{BASELINE if spec is None else spec.name}"


def build_params(
    spec: AugmentSpec | None,
    split: str,
    rng: np.random.Generator,
    shape: tuple[int, int],
) -> dict | None:
    """Sample one parameter set for a whole clip, or ``None`` for no-op.

    Returns ``None`` whenever the split is not train, regardless of the spec.
    That is the guard: evaluation cannot be augmented by accident.
    """
    if spec is None or split != "train":
        return None

    height, width = shape
    scale = float(rng.uniform(*spec.crop_scale))
    crop_h, crop_w = int(round(height * scale)), int(round(width * scale))
    top = int(rng.integers(0, height - crop_h + 1))
    left = int(rng.integers(0, width - crop_w + 1))

    return {
        "crop": (top, left, crop_h, crop_w),
        "hflip": (not spec.direction_sensitive)
        and bool(rng.random() < spec.hflip_prob),
        "brightness": 1.0 + float(rng.uniform(-spec.brightness, spec.brightness)),
        "contrast": 1.0 + float(rng.uniform(-spec.contrast, spec.contrast)),
        "saturation": 1.0 + float(rng.uniform(-spec.saturation, spec.saturation)),
    }


def apply_clip(frames: np.ndarray, params: dict | None) -> np.ndarray:
    """Apply one parameter set to every frame of a clip.

    ``frames`` is ``(T, H, W, 3)`` uint8 RGB. Output keeps that shape and dtype;
    the crop is resized back so downstream tensor shapes are unchanged.
    """
    if params is None:
        return frames

    top, left, crop_h, crop_w = params["crop"]
    height, width = frames.shape[1:3]
    out = np.empty_like(frames)

    for i, frame in enumerate(frames):
        img = frame[top : top + crop_h, left : left + crop_w]
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
        if params["hflip"]:
            img = img[:, ::-1]

        img = img.astype(np.float32)
        img *= params["brightness"]
        mean = img.mean()
        img = (img - mean) * params["contrast"] + mean
        grey = img @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        img = grey[..., None] + (img - grey[..., None]) * params["saturation"]

        out[i] = np.clip(img, 0, 255).astype(np.uint8)

    return out


def jittered_window(
    spec: AugmentSpec | None,
    split: str,
    rng: np.random.Generator,
    total_frames: int,
    start: int = config.START_FRAME,
    end: int = config.END_FRAME,
) -> tuple[int, int]:
    """Shift the sampling window a few frames, staying on the incident.

    Clamped to the clip and to a symmetric bound, so the window can never slide
    past the foul and turn the clip into a differently labelled event.
    """
    if spec is None or split != "train" or spec.temporal_jitter <= 0:
        return start, end

    shift = int(rng.integers(-spec.temporal_jitter, spec.temporal_jitter + 1))
    width = end - start
    lo = max(0, min(start + shift, max(0, total_frames - width)))
    return lo, lo + width


def describe(spec: AugmentSpec | None) -> str:
    """One line for the run record, so a cache can be traced to its recipe."""
    if spec is None:
        return "baseline (no augmentation)"
    parts = [
        f"crop_scale={spec.crop_scale}",
        f"brightness=+/-{spec.brightness}",
        f"contrast=+/-{spec.contrast}",
        f"saturation=+/-{spec.saturation}",
        f"hflip_p={0.0 if spec.direction_sensitive else spec.hflip_prob}",
        f"temporal_jitter=+/-{spec.temporal_jitter}",
    ]
    return f"{spec.name}: " + ", ".join(parts)


def direction_safe(spec: AugmentSpec) -> AugmentSpec:
    """The same recipe with horizontal flip disabled.

    Use when a target depends on left/right - body-part detail, for instance -
    rather than dropping augmentation altogether.
    """
    return replace(spec, direction_sensitive=True)
