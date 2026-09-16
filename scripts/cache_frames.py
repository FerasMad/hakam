"""Decode the clearest view of each action once, to a memory-mapped array.

Three decisions here, each measured rather than assumed.

**Last clip only.** Clip order in this dataset is structured: measured across all
2,916 train actions, clip_0 is the live main-camera wide shot 99.97% of the time,
and the last clip is a close-up 95% of the time at median 1.8x replay speed. The
last clip is therefore the closest, slowest, clearest view of the contact. Taking
only it cuts the corpus from 6,621 clips to 2,916 - 2.3x less compute - while
keeping the view a human would choose to review.

**Cached before the processor, at native 224x398.** The backbone's processor
centre-crops 224 from 398, discarding 87px each side. On 57 sampled clips, 12% of
contact points fell outside that crop entirely. Caching the raw decoded frame
means crop-versus-resize becomes a training-time flag on one cache, instead of
two full decode passes - and it is Experiment 2.

**Ephemeral, not on Drive.** ~12.5 GB. It regenerates from the dataset in about
five minutes, and 12 GB of small reads over Drive's FUSE mount is slower than
decoding again. Only checkpoints and metrics are worth persisting.

    python scripts/cache_frames.py --splits train valid test
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
from src.data.annotations import load_split
from src.features import extract as fx

# Native clip geometry, verified on disk: 126 frames, 25 fps, 398x224.
FRAME_H, FRAME_W = 224, 398
CACHE_DIR = config.PROJECT_ROOT / "frame_cache"


def last_clip_per_action(clips) -> list[tuple[str, str, int]]:
    """One row per action: the highest-indexed clip, which is the close-up.

    Sorted by action id so cache row order is deterministic across runs.
    """
    rows = []
    for action_id, group in clips.groupby("action_id"):
        pick = group.loc[group["clip_index"].idxmax()]
        rows.append((str(action_id), str(pick["path"]), int(pick["clip_index"])))
    return sorted(rows, key=lambda r: int(r[0]) if r[0].isdigit() else r[0])


def all_clips(clips) -> list[tuple[str, str, int]]:
    """Every view of every action, grouped by action for view averaging."""
    rows = [(str(r.action_id), str(r.path), int(r.clip_index)) for r in clips.itertuples()]
    return sorted(rows, key=lambda r: (int(r[0]) if r[0].isdigit() else r[0], r[2]))


def cache_split(split: str, force: bool, start: int = config.START_FRAME,
                end: int = config.END_FRAME, tag: str | None = None,
                all_views: bool = False, size: int | None = None,
                limit: int = 0) -> None:
    suffix = f"_{tag}" if tag else ""
    out = CACHE_DIR / f"{split}_frames{suffix}.npy"
    if out.exists() and not force:
        print(f"[{split}] cache exists, skipping ({out.name})")
        return

    split_dir = config.DATA_ROOT / "mvfouls" / config.SPLIT_DIRS[split]
    _, clips = load_split(split_dir)
    if limit:
        ids = sorted(clips["action_id"].unique(), key=lambda a: int(a) if str(a).isdigit() else a)
        clips = clips[clips["action_id"].isin(ids[:limit])]
    rows = all_clips(clips) if all_views else last_clip_per_action(clips)
    frame_h, frame_w = (size, size) if size else (FRAME_H, FRAME_W)

    missing = [p for _, p, _ in rows if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(
            f"[{split}] {len(missing)} clips missing, e.g. {missing[0]}"
        )

    n = len(rows)
    print(f"[{split}] {n:,} clips ({'all views' if all_views else 'last clip each'}), "
          f"{config.NUM_FRAMES} frames at {frame_h}x{frame_w}")

    # Written straight to disk as a memmap: the train array is ~12.5 GB and
    # building it in RAM first would exhaust a Colab runtime.
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    arr = np.lib.format.open_memmap(
        out, mode="w+", dtype=np.uint8,
        shape=(n, config.NUM_FRAMES, frame_h, frame_w, 3),
    )

    t0 = time.time()
    odd = 0
    for i, (_action_id, path, _idx) in enumerate(rows):
        frames = fx.sample_window(path, start=start, end=end)
        if frames.shape[1:3] != (frame_h, frame_w):
            # Native clips are 224x398; --size squashes them here once, which is
            # what the resize geometry did per batch. Other odd sizes are counted.
            import cv2

            odd += frames.shape[1:3] != (FRAME_H, FRAME_W)
            frames = np.stack([cv2.resize(f, (frame_w, frame_h)) for f in frames])
        arr[i] = frames

        if (i + 1) % 250 == 0 or i + 1 == n:
            rate = (i + 1) / (time.time() - t0)
            print(f"  {i+1:>5}/{n}  {rate:5.1f} clips/s  "
                  f"eta {(n - i - 1) / rate / 60:4.1f} min")

    arr.flush()
    del arr

    keys = [[r[0], r[2]] for r in rows] if all_views else [r[0] for r in rows]
    (CACHE_DIR / f"{split}_keys{suffix}.json").write_text(json.dumps(keys), encoding="utf-8")
    (CACHE_DIR / f"{split}_meta{suffix}.json").write_text(
        json.dumps({
            "split": split,
            "n_clips": n,
            "num_frames": config.NUM_FRAMES,
            "height": frame_h,
            "width": frame_w,
            "start_frame": start,
            "end_frame": end,
            "source": "all views, keys are [action_id, clip_index]" if all_views
                      else "last clip per action (a close-up 95% of the time)",
            "resized_clips": odd,
            "note": "raw decoded frames, BEFORE the backbone processor, so crop "
                    "vs resize stays a training-time choice",
        }, indent=2),
        encoding="utf-8",
    )

    gb = out.stat().st_size / 1e9
    print(f"[{split}] {gb:.1f} GB in {(time.time()-t0)/60:.1f} min -> {out.name}"
          + (f"  ({odd} clips resized)" if odd else ""))


def main() -> None:
    ap = argparse.ArgumentParser(description="Cache the last clip of each action")
    ap.add_argument("--splits", nargs="+", default=["train", "valid", "test"],
                    choices=["train", "valid", "test"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--start", type=int, default=config.START_FRAME)
    ap.add_argument("--end", type=int, default=config.END_FRAME)
    ap.add_argument("--tag", default=None,
                    help="write <split>_frames_<tag>.npy, leaving the default cache alone")
    ap.add_argument("--all-views", action="store_true", help="every clip, not just the last")
    ap.add_argument("--size", type=int, default=None, help="store frames resized to size x size")
    ap.add_argument("--limit", type=int, default=0, help="first N actions only (smoke test)")
    args = ap.parse_args()

    print(f"window {args.start}-{args.end} "
          f"({(args.end - args.start)/25:.2f}s at 25fps)\n")
    for split in args.splits:
        cache_split(split, args.force, args.start, args.end, args.tag,
                    args.all_views, args.size, args.limit)
    print(f"\ncache in {CACHE_DIR}")


if __name__ == "__main__":
    main()
