"""Preprocessing pipeline: manifests, validation, and audit reports.

Implements the high-value subset of PREPROCESSING.md - the parts that change
what the model trains on, or that act as hard gates:

  * label normalisation and conflict detection (sections 6-8, via labels.py)
  * action and clip manifests (section 5)
  * media existence and decodability validation (section 11)
  * exact-duplicate and cross-split leakage detection (section 12)
  * cascade invariant assertions (section 20)
  * aggregate statistics and audit reports (section 17)

Deliberately deferred, with reasons: sample grids and augmentation cache
identities (section 16 - no augmentation experiment is running yet), the manual
override file (section 13 - nothing to adjudicate, since the dataset contains
zero offence/severity conflicts), and the PyAV rewrite (section 11 - PyAV is not
installed, and the cv2 sampler is already measured and working).

Two deviations from the specification, both deliberate:

  window   The spec pins frames 63-87. Measured on 2 Sep 2026, that window spans
           0.96 s at 25 fps and yields near-duplicate frames, which collapsed
           the embeddings onto generic scene appearance. config.START_FRAME and
           END_FRAME now default to 43-107 (2.56 s), matching VideoMAE's
           pretraining stride. The published window is retained as
           config.PUBLISHED_WINDOW for the baseline comparison.
  decoder  cv2 rather than PyAV, which is in requirements.txt but not installed.

Usage:

    python -m src.data.preprocess --mode metadata          # fast label audit
    python -m src.data.preprocess --mode full              # + decode every clip
    python -m src.data.preprocess --mode full --limit-actions 100
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.data.labels import (
    AmbiguityPolicy,
    Code,
    Status,
    assert_cascade_invariants,
    derive_targets,
)

PIPELINE_VERSION = "1.0"

ACTION_RAW_FIELDS = [
    "UrlLocal", "Offence", "Contact", "Bodypart", "Upper body part",
    "Action class", "Severity", "Multiple fouls", "Try to play",
    "Touch ball", "Handball", "Handball offence",
]

PRIVATE_DIR = config.ARTIFACTS / "preprocessing" / "private"
SUMMARY_DIR = config.ARTIFACTS / "preprocessing" / "summary"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=config.PROJECT_ROOT, timeout=10,
        ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


# --------------------------------------------------------------------------
# Path resolution (section 10.8)
# --------------------------------------------------------------------------

def resolve_clip_path(
    split_dir: Path, action_id: str, clip_index: int, url: str | None
) -> Path:
    """Resolve a clip's local path, preferring the annotated URL.

    The annotations store ``Dataset/Train/action_0/clip_0`` - a packaging-layout
    path with no extension. Its trailing two segments are used rather than
    assuming clip indices are contiguous. Verified 3 Sep 2026: they are
    contiguous throughout this dataset, so the annotation and the rebuilt path
    agree - but trusting the annotation keeps that an observation rather than an
    assumption.
    """
    if url:
        parts = [p for p in str(url).replace("\\", "/").split("/") if p]
        if len(parts) >= 2:
            return split_dir / parts[-2] / f"{parts[-1]}.mp4"
    return split_dir / f"action_{action_id}" / f"clip_{clip_index}.mp4"


# --------------------------------------------------------------------------
# Media validation (section 11)
# --------------------------------------------------------------------------

def probe_clip(path: Path, decode: bool) -> dict:
    """Metadata pass, and optionally a real decode of the required frames."""
    info = {
        "file_exists": path.exists(),
        "file_size_bytes": path.stat().st_size if path.exists() else None,
        "frame_count": None, "fps": None, "width": None, "height": None,
        "duration_seconds": None, "decodable": False, "codes": [],
    }
    if not info["file_exists"]:
        info["codes"].append(Code.MISSING_CLIP)
        return info

    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        info["codes"].append(Code.CORRUPT_CLIP)
        return info

    try:
        info["frame_count"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        info["fps"] = float(cap.get(cv2.CAP_PROP_FPS))
        info["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        info["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if info["fps"] and info["fps"] > 0:
            info["duration_seconds"] = info["frame_count"] / info["fps"]
        else:
            info["codes"].append(Code.INVALID_FPS)
        if not info["width"] or not info["height"]:
            info["codes"].append(Code.INVALID_RESOLUTION)
        if info["frame_count"] < config.END_FRAME:
            info["codes"].append(Code.INSUFFICIENT_FRAMES)

        if not decode:
            info["decodable"] = Code.INSUFFICIENT_FRAMES not in info["codes"]
            return info

        # Metadata lies often enough to be worth checking. Every required frame
        # must actually decode; a short clip is never padded by repeating its
        # last frame, which would fabricate motion.
        wanted = set(
            np.linspace(config.START_FRAME, config.END_FRAME - 1, config.NUM_FRAMES)
            .round().astype(int).tolist()
        )
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(wanted))
        got, position = set(), min(wanted)
        while position <= max(wanted):
            ok, _ = cap.read()
            if not ok:
                break
            if position in wanted:
                got.add(position)
            position += 1

        info["decodable"] = got == wanted
        if not info["decodable"]:
            info["codes"].append(Code.CORRUPT_CLIP)
    finally:
        cap.release()

    return info


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------
# Manifest construction (section 5)
# --------------------------------------------------------------------------

def build_split(
    split: str,
    policy: str,
    decode: bool,
    limit: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    split_dir = config.DATA_ROOT / "mvfouls" / config.SPLIT_DIRS[split]
    ann_path = split_dir / "annotations.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"no annotations.json in {split_dir}")

    with open(ann_path, encoding="utf-8") as fh:
        raw = json.load(fh)
    if "Actions" not in raw or not isinstance(raw["Actions"], dict):
        raise ValueError(f"{ann_path} has no dictionary 'Actions' - malformed schema")

    action_rows, clip_rows, unknown = [], [], []
    items = list(raw["Actions"].items())
    if limit:
        items = items[:limit]

    for action_id, record in items:
        targets = derive_targets(
            record.get("Offence"), record.get("Severity"), policy=policy
        )
        assert_cascade_invariants(targets)

        for code, field_name in (
            (Code.UNKNOWN_OFFENCE, "Offence"),
            (Code.UNKNOWN_SEVERITY, "Severity"),
        ):
            if code in targets.codes:
                unknown.append({
                    "split": split, "action_id": action_id,
                    "field": field_name, "raw_value": record.get(field_name),
                })

        clips = record.get("Clips", [])
        valid_clips = 0
        for idx, clip in enumerate(clips):
            path = resolve_clip_path(split_dir, action_id, idx, clip.get("Url"))
            info = probe_clip(path, decode=decode)

            codes = list(info["codes"])
            speed = clip.get("Replay speed")
            try:
                if speed is not None and float(speed) <= 0:
                    codes.append(Code.INVALID_REPLAY_SPEED)
            except (TypeError, ValueError):
                codes.append(Code.INVALID_REPLAY_SPEED)

            include = bool(info["file_exists"] and info["decodable"])
            valid_clips += int(include)
            clip_rows.append({
                "action_key": f"{split}:{action_id}",
                "action_id": action_id,
                "split": split,
                "clip_index": idx,
                "annotated_url": clip.get("Url"),
                "resolved_path": str(path),
                "path": str(path),                       # legacy column name
                "camera_type": clip.get("Camera type", ""),
                "timestamp_raw": clip.get("Timestamp"),
                "replay_speed": clip.get("Replay speed"),
                "file_exists": info["file_exists"],
                "file_size_bytes": info["file_size_bytes"],
                "decodable": info["decodable"],
                "frame_count": info["frame_count"],
                "fps": info["fps"],
                "width": info["width"],
                "height": info["height"],
                "duration_seconds": info["duration_seconds"],
                "sha256": None,
                "quality_status": Status.VALID if include else Status.INVALID_MEDIA,
                "quality_codes": "|".join(codes),
                "include": include,
            })

        codes = list(targets.codes)
        if valid_clips < 2:
            codes.append(Code.FEWER_THAN_TWO_VALID_VIEWS)

        row = {
            "split": split,
            "action_id": action_id,
            "action_key": f"{split}:{action_id}",
            "source_match": record.get("UrlLocal"),
            "num_clips_raw": len(clips),
            "num_clips_valid": valid_clips,
            "include_multiview": valid_clips >= 2,
        }
        row.update({f"{f}_raw": record.get(f) for f in ACTION_RAW_FIELDS})
        row.update(targets.as_dict())
        row["quality_codes"] = "|".join(codes)
        # Legacy plain-named columns that the existing scripts read.
        row.update({f: record.get(f, "") for f in ACTION_RAW_FIELDS})
        action_rows.append(row)

    return pd.DataFrame(action_rows), pd.DataFrame(clip_rows), unknown


# --------------------------------------------------------------------------
# Duplicates and leakage (section 12)
# --------------------------------------------------------------------------

def detect_duplicates(clips: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Two-stage exact-duplicate detection: group by size, then hash.

    Hashing 8,297 files outright is wasteful when almost none share a size.
    Returns the duplicate report and whether a cross-split leak was found, which
    is a hard failure - a train clip appearing in test invalidates every number
    the project reports.
    """
    present = clips[clips["file_exists"]].copy()
    if present.empty:
        return pd.DataFrame(), False

    sizes = present.groupby("file_size_bytes").size()
    suspect = present[present["file_size_bytes"].isin(sizes[sizes > 1].index)]
    if suspect.empty:
        return pd.DataFrame(), False

    digests: dict[str, list[dict]] = defaultdict(list)
    for row in suspect.to_dict("records"):
        try:
            digests[sha256_file(Path(row["resolved_path"]))].append(row)
        except OSError:
            continue

    rows, leak = [], False
    for digest, group in digests.items():
        if len(group) < 2:
            continue
        splits = {g["split"] for g in group}
        actions = {g["action_key"] for g in group}
        if len(splits) > 1:
            code, leak = Code.CROSS_SPLIT_DUPLICATE, True
        elif len(actions) > 1:
            code = Code.DUPLICATE_ACROSS_ACTIONS
        else:
            code = Code.DUPLICATE_WITHIN_ACTION
        for g in group:
            rows.append({
                "sha256": digest, "reason_code": code, "split": g["split"],
                "action_key": g["action_key"], "clip_index": g["clip_index"],
                "resolved_path": g["resolved_path"],
                "group_size": len(group),
                "splits_involved": "|".join(sorted(splits)),
            })

    return pd.DataFrame(rows), leak


# --------------------------------------------------------------------------
# Reports (section 17)
# --------------------------------------------------------------------------

def write_reports(
    actions: dict[str, pd.DataFrame],
    clips: dict[str, pd.DataFrame],
    unknown: list[dict],
    duplicates: pd.DataFrame,
    settings: dict,
) -> None:
    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    all_actions = pd.concat(actions.values(), ignore_index=True)
    all_clips = pd.concat(clips.values(), ignore_index=True)

    for split, frame in actions.items():
        frame.to_csv(PRIVATE_DIR / f"actions_{split}.csv", index=False)
    for split, frame in clips.items():
        frame.to_csv(PRIVATE_DIR / f"clips_{split}.csv", index=False)

    all_actions[all_actions["quality_status"] == Status.CONFLICT].to_csv(
        PRIVATE_DIR / "conflicts.csv", index=False
    )
    pd.DataFrame(unknown).to_csv(PRIVATE_DIR / "unknown_labels.csv", index=False)
    duplicates.to_csv(PRIVATE_DIR / "duplicate_clips.csv", index=False)

    bad = all_clips[~all_clips["include"]]
    bad.to_csv(PRIVATE_DIR / "missing_or_corrupt_clips.csv", index=False)

    exclusions = []
    for row in all_actions[
        all_actions["quality_status"] == Status.EXCLUDED
    ].to_dict("records"):
        exclusions.append({
            "action_key": row["action_key"], "clip_index": None,
            "reason_code": row["quality_codes"], "detail": "action excluded",
            "detected_at": settings["started_at"],
            "pipeline_version": PIPELINE_VERSION,
        })
    for row in bad.to_dict("records"):
        exclusions.append({
            "action_key": row["action_key"], "clip_index": row["clip_index"],
            "reason_code": row["quality_codes"] or Code.MISSING_CLIP,
            "detail": row["resolved_path"],
            "detected_at": settings["started_at"],
            "pipeline_version": PIPELINE_VERSION,
        })
    pd.DataFrame(exclusions).to_csv(PRIVATE_DIR / "exclusions.csv", index=False)

    # ---- Aggregates, safe to share ----------------------------------------
    split_rows, supervision_rows, distribution_rows = [], [], []
    for split, frame in actions.items():
        cf = clips[split]
        split_rows.append({
            "split": split,
            "actions": len(frame),
            "actions_official": config.SPLIT_SIZES.get(split),
            "clips": len(cf),
            "clips_valid": int(cf["include"].sum()),
            "multiview_actions": int(frame["include_multiview"].sum()),
            "ambiguous": int(frame["ambiguous"].sum()),
            "conflicts": int((frame["quality_status"] == Status.CONFLICT).sum()),
            "excluded": int((frame["quality_status"] == Status.EXCLUDED).sum()),
        })
        for stage in ("offence", "card", "card_colour"):
            supervised = frame[frame[f"supervise_{stage}"]]
            counts = supervised[f"target_{stage}"].value_counts().to_dict()
            supervision_rows.append({
                "split": split, "stage": stage,
                "supervised": len(supervised), "of_total": len(frame),
                **{f"n_{k}": v for k, v in counts.items()},
            })
        for column in ("Offence", "Severity", "Action class"):
            counts = frame[column].replace("", "(empty)").value_counts()
            for value, count in counts.items():
                distribution_rows.append({
                    "split": split, "field": column,
                    "value": value, "count": int(count),
                })

    pd.DataFrame(split_rows).to_csv(SUMMARY_DIR / "split_summary.csv", index=False)
    pd.DataFrame(supervision_rows).to_csv(
        SUMMARY_DIR / "cascade_supervision_counts.csv", index=False
    )
    pd.DataFrame(distribution_rows).to_csv(
        SUMMARY_DIR / "label_distribution.csv", index=False
    )
    (SUMMARY_DIR / "dataset_summary.json").write_text(
        json.dumps({"settings": settings, "splits": split_rows}, indent=2),
        encoding="utf-8",
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Hakam preprocessing pipeline")
    ap.add_argument("--data-root", default=str(config.DATA_ROOT))
    ap.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    ap.add_argument("--mode", default="metadata", choices=["metadata", "full"])
    ap.add_argument(
        "--ambiguity-policy", default=AmbiguityPolicy.STAGEWISE_DROP,
        choices=[AmbiguityPolicy.STAGEWISE_DROP, AmbiguityPolicy.MERGE,
                 AmbiguityPolicy.KEEP],
    )
    ap.add_argument("--start-frame", type=int, default=config.START_FRAME)
    ap.add_argument("--end-frame", type=int, default=config.END_FRAME)
    ap.add_argument("--num-frames", type=int, default=config.NUM_FRAMES)
    ap.add_argument("--limit-actions", type=int, default=None)
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()

    config.DATA_ROOT = Path(args.data_root)
    config.START_FRAME, config.END_FRAME = args.start_frame, args.end_frame
    config.NUM_FRAMES = args.num_frames

    settings = {
        "started_at": _now(),
        "git_commit": _git_commit(),
        "pipeline_version": PIPELINE_VERSION,
        "mode": args.mode,
        "ambiguity_policy": args.ambiguity_policy,
        "window": [args.start_frame, args.end_frame],
        "num_frames": args.num_frames,
        "seed": args.seed,
        "splits": args.splits,
    }
    print(json.dumps(settings, indent=2))

    t0 = time.time()
    actions, clips, unknown, failures = {}, {}, [], []

    for split in args.splits:
        a, c, u = build_split(
            split, args.ambiguity_policy, args.mode == "full", args.limit_actions
        )
        actions[split], clips[split] = a, c
        unknown += u

        expected = config.SPLIT_SIZES.get(split)
        if expected and not args.limit_actions and len(a) != expected:
            failures.append(f"{split}: {len(a)} actions, official count {expected}")
        print(
            f"[{split}] actions={len(a):>5} clips={len(c):>5} "
            f"valid_clips={int(c['include'].sum()):>5} "
            f"ambiguous={int(a['ambiguous'].sum()):>4} "
            f"conflicts={int((a['quality_status'] == Status.CONFLICT).sum()):>3} "
            f"excluded={int((a['quality_status'] == Status.EXCLUDED).sum()):>3}"
        )

    all_clips = pd.concat(clips.values(), ignore_index=True)
    duplicates, leak = detect_duplicates(all_clips)
    if leak:
        failures.append("cross-split exact duplicate detected (CROSS_SPLIT_DUPLICATE)")
    print(f"\nduplicate clip rows: {len(duplicates)}  cross-split leak: {leak}")

    if unknown:
        failures.append(f"{len(unknown)} unknown primary label values")

    settings["finished_at"] = _now()
    settings["elapsed_seconds"] = round(time.time() - t0, 1)
    write_reports(actions, clips, unknown, duplicates, settings)

    print("\ncascade supervision:")
    for split, frame in actions.items():
        parts = [
            f"{stage}={int(frame[f'supervise_{stage}'].sum())}"
            for stage in ("offence", "card", "card_colour")
        ]
        print(f"  {split:6} " + "  ".join(parts))

    print(f"\nprivate manifests -> {PRIVATE_DIR}")
    print(f"summary           -> {SUMMARY_DIR}")

    if failures:
        print("\nQUALITY GATE FAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nall quality gates passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
