"""Full-corpus feature extraction: every clip in every labelled split.

Produces the **frozen baseline** features. The diagnostic on 2 Sep 2026 showed
frozen Kinetics features encode camera framing (probe 0.94) but barely encode
the foul itself (card probe ~0.47-0.56), so these embeddings are not the final
model - they are the baseline that Experiment 1 compares fine-tuning against,
which section 7 of the brief requires regardless.

Design notes:

* **Per clip, never per action.** View pooling stays a training-time choice
  instead of being baked into an expensive run.
* **Resumable.** A split whose cache already exists is skipped unless --force.
  Colab runtimes disconnect, and losing an hour to a dropped session is
  avoidable.
* **Settings are written beside the cache.** A cache whose window and pooling
  are unknown is not reusable, and this project has already changed the window
  once.

On Colab the dataset stays in the ephemeral runtime and only the cache is copied
to Drive, so no video is ever persisted to Google storage (NDA).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.data.annotations import load_split
from src.features import extract as fx

LABELLED_SPLITS = ["train", "valid", "test"]


def extract_split(
    split: str,
    extractor,
    backbone: str,
    batch_size: int,
    pooling: str,
    force: bool,
) -> None:
    cache = fx.cache_path(backbone, tag=split)
    if cache.exists() and not force:
        print(f"[{split}] cache exists, skipping ({cache.name})")
        return

    split_dir = config.DATA_ROOT / "mvfouls" / config.SPLIT_DIRS[split]
    actions, clips = load_split(split_dir)

    missing = [p for p in clips["path"] if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(
            f"[{split}] {len(missing)} clip files missing, e.g. {missing[0]}"
        )

    expected = config.SPLIT_SIZES.get(split)
    if expected is not None and len(actions) != expected:
        print(f"[{split}] WARNING: {len(actions)} actions, official count is {expected}")

    print(f"[{split}] {len(actions):,} actions -> {len(clips):,} clips")
    t0 = time.time()
    feats = fx.extract_clips(
        clips["path"].tolist(),
        extractor,
        batch_size=batch_size,
        start=config.START_FRAME,
        end=config.END_FRAME,
        pooling=pooling,
    )
    elapsed = time.time() - t0

    keys = [fx.clip_key(a, i) for a, i in zip(clips["action_id"], clips["clip_index"])]
    path = fx.save_cache(keys, feats, backbone, tag=split)

    # Without these settings the cache is not reusable - the window has already
    # changed once in this project.
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "split": split,
                "backbone": backbone,
                "backbone_checkpoint": config.BACKBONES[backbone]["name"],
                "start_frame": config.START_FRAME,
                "end_frame": config.END_FRAME,
                "num_frames": config.NUM_FRAMES,
                "pooling": pooling,
                "n_clips": int(feats.shape[0]),
                "dim": int(feats.shape[1]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"[{split}] {feats.shape} in {elapsed/60:.1f} min "
        f"({elapsed/len(clips)*1000:.0f} ms/clip) -> {path.name}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract frozen features for all splits")
    ap.add_argument("--splits", nargs="+", default=LABELLED_SPLITS, choices=LABELLED_SPLITS)
    ap.add_argument("--backbones", nargs="+", default=["videomae_small"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--pooling", default="mean", choices=["mean", "fc_norm"])
    ap.add_argument("--force", action="store_true", help="re-extract even if cached")
    args = ap.parse_args()

    print(
        f"window {config.START_FRAME}-{config.END_FRAME} "
        f"({(config.END_FRAME - config.START_FRAME) / 25:.2f} s at 25 fps), "
        f"{config.NUM_FRAMES} frames, pooling={args.pooling}"
    )
    print(f"device: {fx.resolve_device()}\n")

    for backbone in args.backbones:
        print(f"=== {backbone} ===")
        extractor = fx.build_extractor(backbone)
        for split in args.splits:
            extract_split(
                split, extractor, backbone, args.batch_size, args.pooling, args.force
            )
        print()

    print(f"caches in {config.FEATURES_CACHE}")


if __name__ == "__main__":
    main()
