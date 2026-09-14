"""Compute a player-centred crop box for every cached clip.

Reads the middle frame straight from the frame cache, so the box is aligned with
exactly the pixels the model will see. Writes ``frame_cache/<split>_boxes.json``:
``{action_id: [x0, y0, x1, y1]}`` in native 398x224 coordinates, or ``null`` when
fewer than two players are found (the dataset then falls back to the full frame).

    python scripts/zoom_boxes.py --splits train valid test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.features import detect

CACHE_DIR = config.PROJECT_ROOT / "frame_cache"


def boxes_for_split(split: str, detector, batch: int = 64) -> None:
    frames = np.load(CACHE_DIR / f"{split}_frames.npy", mmap_mode="r")
    keys = json.loads((CACHE_DIR / f"{split}_keys.json").read_text())
    n, t, h, w, _ = frames.shape
    mid = t // 2

    boxes, found, t0 = {}, 0, time.time()
    for start in range(0, n, batch):
        # Cache is RGB; ultralytics treats numpy input as BGR.
        chunk = [np.ascontiguousarray(frames[i, mid, :, :, ::-1])
                 for i in range(start, min(start + batch, n))]
        for offset, result in enumerate(detector.predict(chunk, verbose=False)):
            person = detect._person_boxes(result, conf=0.35)
            box = detect.zoom_box(detect.contact_focus(person), h, w)
            boxes[keys[start + offset]] = box
            found += box is not None

    (CACHE_DIR / f"{split}_boxes.json").write_text(json.dumps(boxes), encoding="utf-8")
    print(f"[{split}] {found}/{n} clips zoomed ({found/n:.0%}), "
          f"{n-found} fall back to full frame, {time.time()-t0:.0f}s")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    args = ap.parse_args()

    detector = detect.load_detector()
    if detector is None:
        print("ultralytics not available - pip install ultralytics")
        return 1
    for split in args.splits:
        boxes_for_split(split, detector)
    return 0


if __name__ == "__main__":
    sys.exit(main())
