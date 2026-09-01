"""Load SoccerNet-MVFoul annotations into flat tables.

The raw file is nested: one record per action, each holding a list of clips.
Two frames come out of it:

  * actions - one row per incident, carrying the labels and the derived
    cascade targets.
  * clips   - one row per video file, carrying the path and camera metadata.

Keeping them separate matters because ``Replay speed`` varies between views of
the same incident, so a clip is not interchangeable with its siblings.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data.labels import AmbiguityPolicy, derive_cascade_targets, is_ambiguous

ACTION_FIELDS = [
    "UrlLocal", "Offence", "Contact", "Bodypart", "Upper body part",
    "Action class", "Severity", "Multiple fouls", "Try to play",
    "Touch ball", "Handball", "Handball offence",
]


def load_split(
    split_dir: Path,
    policy: str = AmbiguityPolicy.DROP,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read one split directory into (actions, clips).

    ``split_dir`` must contain ``annotations.json`` alongside the
    ``action_<id>/`` folders.
    """
    split_dir = Path(split_dir)
    ann_path = split_dir / "annotations.json"
    if not ann_path.exists():
        raise FileNotFoundError(
            f"no annotations.json in {split_dir} - was the zip extracted here?"
        )

    with open(ann_path, encoding="utf-8") as fh:
        raw = json.load(fh)

    split_name = raw.get("Set", split_dir.name)
    action_rows, clip_rows = [], []

    for action_id, record in raw["Actions"].items():
        targets = derive_cascade_targets(
            record.get("Offence", ""), record.get("Severity", ""), policy=policy
        )

        row = {"action_id": action_id, "split": split_name}
        row.update({f: record.get(f, "") for f in ACTION_FIELDS})
        row.update({f"target_{k}": v for k, v in targets.items()})
        row["ambiguous"] = is_ambiguous(
            record.get("Offence", ""), record.get("Severity", "")
        )
        row["num_clips"] = len(record.get("Clips", []))
        action_rows.append(row)

        for idx, clip in enumerate(record.get("Clips", [])):
            clip_rows.append({
                "action_id": action_id,
                "split": split_name,
                "clip_index": idx,
                # Rebuild the path locally rather than trusting the stored Url,
                # which points at the packaging layout, not the extracted one.
                "path": str(split_dir / f"action_{action_id}" / f"clip_{idx}.mp4"),
                "camera_type": clip.get("Camera type", ""),
                "timestamp": clip.get("Timestamp"),
                "replay_speed": clip.get("Replay speed"),
            })

    return pd.DataFrame(action_rows), pd.DataFrame(clip_rows)


def verify_clips_exist(clips: pd.DataFrame) -> pd.DataFrame:
    """Return the rows whose video file is missing on disk.

    Worth running once after extraction: a truncated unzip is easy to miss and
    surfaces much later as confusing training failures.
    """
    missing = clips[~clips["path"].apply(lambda p: Path(p).exists())]
    return missing


def summarise(actions: pd.DataFrame) -> str:
    """Human-readable label distribution, for the data section of the report."""
    lines = [f"split: {actions['split'].iloc[0]}  |  actions: {len(actions)}"]
    lines.append(f"ambiguous labels: {int(actions['ambiguous'].sum())}")
    for col in ("Offence", "Severity", "Action class"):
        counts = actions[col].replace("", "(empty)").value_counts()
        lines.append(f"\n{col}:")
        lines.extend(f"  {v:<28} {c:>5}" for v, c in counts.items())
    lines.append("\ncascade supervision available:")
    for stage in ("offence", "card", "card_colour"):
        col = f"target_{stage}"
        usable = actions[col].notna().sum()
        lines.append(f"  {stage:<14} {usable:>5} / {len(actions)}")
        for v, c in actions[col].dropna().value_counts().items():
            lines.append(f"      {v:<24} {c:>5}")
    return "\n".join(lines)
