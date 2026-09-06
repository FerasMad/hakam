"""Pretrained person detection over the incident frame.

This adds *observations* to the contract, never decisions. The detector can say
"four players, two of them in contact range". It cannot say "foul" - that stays
with the cascade, which was trained on referee labels.

Why this exists: the language model currently grounds its Arabic explanation in
categorical labels alone ("standing tackling", "with contact"). A count of
players in the contact zone is a concrete physical fact it can cite, and it
costs no annotation because the detector is pretrained on COCO. Nothing here is
trained by us and nothing here needs labelling.

Three deliberate constraints:

* **Optional by construction.** If ultralytics is not installed, every function
  returns ``None`` and the pipeline runs exactly as before. The detector must
  never become a dependency of the core path.
* **Observations are separated from inferences.** ``players_in_contact_zone``
  is a measurement. It is not evidence of a foul, and the prompt layer must not
  present it as one.
* **Overlay frames are restricted.** An annotated frame is video from a dataset
  under NDA with rectangles drawn on it. Written only when explicitly asked
  for, and only under an ignored directory.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from src import config

# COCO class 0. The footage is football, so every other class is noise here.
PERSON_CLASS = 0
DEFAULT_WEIGHTS = "yolov8n.pt"


@dataclass(frozen=True)
class SceneEvidence:
    """What a detector observed at the incident frame.

    Every field is a measurement with a stated basis, so a generated sentence
    citing it traces back to something that was actually computed.
    """

    frame_index: int
    players_detected: int
    players_in_contact_zone: int
    contact_zone_radius_px: float
    min_pair_distance_px: float | None
    detector: str
    confidence_threshold: float
    annotated_frame: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)

    def citable_values(self) -> set[str]:
        """Facts a generated explanation is permitted to assert.

        Counts only. Pixel distances are diagnostic - a sentence quoting one
        would be unreadable and unverifiable to a referee.
        """
        return {
            f"players_detected={self.players_detected}",
            f"players_in_contact_zone={self.players_in_contact_zone}",
        }


def load_detector(weights: str = DEFAULT_WEIGHTS):
    """Return a pretrained YOLO model, or ``None`` if unavailable.

    Returns ``None`` rather than raising: this feature is additive, and a
    missing optional dependency must degrade the explanation, not break the
    prediction.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        return None

    try:
        return YOLO(weights)
    except Exception:
        # Weight download fails offline. Same reasoning as above.
        return None


def _incident_frame(path: Path, frame_index: int | None) -> tuple[np.ndarray | None, int]:
    """Grab the single frame at the centre of the sampling window."""
    if frame_index is None:
        frame_index = (config.START_FRAME + config.END_FRAME) // 2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None, frame_index
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total > 0:
            frame_index = min(frame_index, total - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
    finally:
        cap.release()
    return (frame if ok else None), frame_index


def _person_boxes(result, conf: float) -> np.ndarray:
    """Person boxes as ``(n, 4)`` xyxy, filtered by class and confidence."""
    if result.boxes is None or len(result.boxes) == 0:
        return np.empty((0, 4), dtype=np.float32)

    cls = result.boxes.cls.cpu().numpy().astype(int)
    scores = result.boxes.conf.cpu().numpy()
    xyxy = result.boxes.xyxy.cpu().numpy()
    keep = (cls == PERSON_CLASS) & (scores >= conf)
    return xyxy[keep].astype(np.float32)


def contact_zone(
    boxes: np.ndarray, radius_factor: float = 1.5
) -> tuple[int, float | None, float]:
    """Count players clustered around the closest pair on screen.

    The closest pair is a reasonable proxy for where the challenge happened: a
    foul involves two players in contact, and everyone else is usually further
    away. Returns ``(count, min_distance, radius_used)``.

    **The radius scales with player size, not pixels.** An absolute threshold
    is meaningless on this dataset: the same 120px covers a whole penalty area
    on a main-camera wide shot and half a torso on a close-up, and every action
    carries both. Measured on action 4, a fixed 120px put nine of fourteen
    players "in contact" on the wide view - visibly wrong. Median detected
    player height is a scale the camera cannot distort, so the radius is
    expressed as a multiple of it.

    Fewer than two players means there is no pair and so no zone; the count
    falls back to however many were detected.
    """
    if len(boxes) < 2:
        return len(boxes), None, 0.0

    heights = boxes[:, 3] - boxes[:, 1]
    radius = float(np.median(heights) * radius_factor)

    centres = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2,
                        (boxes[:, 1] + boxes[:, 3]) / 2], axis=1)
    diff = centres[:, None, :] - centres[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)

    i, j = np.unravel_index(np.argmin(dist), dist.shape)
    min_distance = float(dist[i, j])
    focus = (centres[i] + centres[j]) / 2

    within = np.linalg.norm(centres - focus, axis=1) <= radius
    return int(within.sum()), min_distance, radius


def analyse_clip(
    path: str | Path,
    detector,
    frame_index: int | None = None,
    conf: float = 0.35,
    radius_factor: float = 1.5,
    overlay_path: str | Path | None = None,
) -> SceneEvidence | None:
    """Observe one clip's incident frame. ``None`` when detection is unavailable.

    ``overlay_path`` writes an annotated frame for the demo UI. It contains
    restricted video, so it is opt-in and belongs under an ignored directory.
    """
    if detector is None:
        return None

    path = Path(path)
    frame, index = _incident_frame(path, frame_index)
    if frame is None:
        return None

    result = detector.predict(frame, verbose=False)[0]
    boxes = _person_boxes(result, conf)
    in_zone, min_distance, radius = contact_zone(boxes, radius_factor)

    saved = None
    if overlay_path is not None:
        saved = str(draw_overlay(frame, boxes, Path(overlay_path)))

    return SceneEvidence(
        frame_index=index,
        players_detected=int(len(boxes)),
        players_in_contact_zone=in_zone,
        contact_zone_radius_px=round(radius, 1),
        min_pair_distance_px=min_distance,
        detector=DEFAULT_WEIGHTS,
        confidence_threshold=conf,
        annotated_frame=saved,
    )


def draw_overlay(frame: np.ndarray, boxes: np.ndarray, out: Path) -> Path:
    """Write the frame with person boxes drawn. Demo only.

    Restricted output - a frame of NDA video with rectangles on it.
    """
    canvas = frame.copy()
    for x1, y1, x2, y2 in boxes.astype(int):
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 220, 0), 2)
    cv2.putText(canvas, f"{len(boxes)} players", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)

    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), canvas)
    return out
