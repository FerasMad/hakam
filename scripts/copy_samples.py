"""Copy the five "Test yourself" clips into the website.

    python scripts/copy_samples.py

The clips are the close-up replays of the five highest-scoring test incidents
(see web/lib/samples.ts and docs/AUDIT.md). They come from the SoccerNet-MVFoul
test split in data/mvfouls/Test and are copied to web/public/samples/, which is
gitignored - the dataset clips are not published in the repository.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = {  # test incident -> close-up clip used by the website
    "88": "clip_1.mp4",
    "183": "clip_2.mp4",
    "144": "clip_2.mp4",
    "212": "clip_2.mp4",
    "281": "clip_1.mp4",
}


def main() -> int:
    target = ROOT / "web" / "public" / "samples"
    target.mkdir(parents=True, exist_ok=True)
    missing = []
    for incident, clip in SAMPLES.items():
        source = ROOT / "data" / "mvfouls" / "Test" / f"action_{incident}" / clip
        if not source.exists():
            missing.append(str(source.relative_to(ROOT)))
            continue
        shutil.copy2(source, target / f"incident-{incident}.mp4")
        print(f"incident {incident}: {source.relative_to(ROOT)} -> web/public/samples/incident-{incident}.mp4")
    if missing:
        print("missing (extract Test.zip into data/mvfouls first):\n  " + "\n  ".join(missing))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
