"""Why did the pilot's features collapse, and which knob fixes it?

The pilot returned STOP: two views of the same incident were *less* similar to
each other (cosine 0.757) than two random clips (0.767). An embedding that
cannot tell one incident from another cannot support any downstream head.

Three suspects, all cheap to test:

  window   The published baseline's 63-87 spans 0.96 s at 25 fps, so 16 frames
           drawn from it are near-duplicates. VideoMAE's Kinetics pretraining
           used 16 frames at stride 4, about 2.6 s. A near-still input collapses
           the embedding onto generic scene appearance.
  fc_norm  The finetuned checkpoint applies a LayerNorm after mean pooling.
           Loading the encoder alone discards it, leaving the pooled vector
           unnormalised and dominated by a few high-variance dimensions.
  pooling  Averaging every spatiotemporal token washes out a foul that occupies
           a small part of the frame.

``fc_norm`` is applied *after* pooling, so both pooling variants come free from
one forward pass per window. Two windows therefore cost two passes, not four.

Three measurements per configuration:

  view contrast   mean cosine between views of one action, minus mean cosine
                  between random clips. **The primary signal.** Positive means
                  the embedding encodes the incident rather than the sport.
  camera probe    can a linear probe tell a close-up from a main-camera shot?
                  Pure appearance, nothing to do with fouls. This is the real
                  sanity floor - the pilot's `contact` probe was 489/11, so it
                  measured nothing at all.
  card probe      the decision that matters, for reference only. Do not read it
                  until view contrast is clearly positive.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.data.annotations import load_split
from src.features import extract as fx
from src.features import probe as pb

# Narrow: the published baseline. Wide: 64 frames at 25 fps = 2.56 s, centred on
# frame 75, reproducing VideoMAE's 16-frames-at-stride-4 pretraining.
WINDOWS = {
    "narrow_63_87": (config.START_FRAME, config.END_FRAME),
    "wide_43_107": (43, 107),
}

ARTIFACTS = config.ARTIFACTS / "pilot"


def sample(n: int, seed: int = config.SEED):
    from sklearn.model_selection import train_test_split

    actions, clips = load_split(config.DATA_ROOT / "mvfouls" / config.SPLIT_DIRS["train"])
    key = actions["target_card"].fillna("na").astype(str)
    counts = key.value_counts()
    key = key.where(~key.isin(counts[counts < 2].index), "other")
    sub, _ = train_test_split(actions, train_size=n, stratify=key, random_state=seed)
    sub = sub.reset_index(drop=True)
    return sub, clips[clips["action_id"].isin(set(sub["action_id"]))].reset_index(drop=True)


def view_contrast(feats: np.ndarray, clips: pd.DataFrame, keys: list[str]) -> dict:
    """Same-incident similarity minus random-pair similarity."""
    index = {k: i for i, k in enumerate(keys)}
    unit = feats / np.clip(np.linalg.norm(feats, axis=1, keepdims=True), 1e-8, None)

    same = []
    for action_id, group in clips.groupby("action_id"):
        rows = [index[fx.clip_key(action_id, i)] for i in group["clip_index"]
                if fx.clip_key(action_id, i) in index]
        if len(rows) > 1:
            sims = unit[rows] @ unit[rows].T
            same.append(sims[np.triu_indices(len(rows), 1)].mean())

    rng = np.random.default_rng(config.SEED)
    pick = rng.choice(len(unit), min(300, len(unit)), replace=False)
    rand = float((unit[pick] @ unit[pick].T)[np.triu_indices(len(pick), 1)].mean())

    same_mean = float(np.mean(same))
    return {"same": same_mean, "random": rand, "contrast": same_mean - rand}


def camera_probe(feats: np.ndarray, clips: pd.DataFrame, keys: list[str]) -> float:
    """Linear separability of the two dominant camera types. The sanity floor."""
    index = {k: i for i, k in enumerate(keys)}
    top2 = clips["camera_type"].value_counts().head(2).index.tolist()

    rows, labels = [], []
    for action_id, clip_index, cam in zip(
        clips["action_id"], clips["clip_index"], clips["camera_type"]
    ):
        key = fx.clip_key(action_id, clip_index)
        if cam in top2 and key in index:
            rows.append(index[key])
            labels.append(cam)

    if len(set(labels)) < 2:
        return float("nan")
    return pb.linear_probe(feats[rows], np.array(labels)).get(
        "balanced_accuracy", float("nan")
    )


def card_probe(
    feats: np.ndarray, clips: pd.DataFrame, keys: list[str], actions: pd.DataFrame
) -> float:
    index = {k: i for i, k in enumerate(keys)}
    labels = actions.set_index("action_id")["target_card"]

    X, y = [], []
    for action_id, group in clips.groupby("action_id"):
        target = labels.get(action_id)
        if not isinstance(target, str) or target == "":
            continue
        rows = [index[fx.clip_key(action_id, i)] for i in group["clip_index"]
                if fx.clip_key(action_id, i) in index]
        if rows:
            X.append(feats[rows].mean(axis=0))
            y.append(target)

    if len(set(y)) < 2:
        return float("nan")
    return pb.linear_probe(np.stack(X), np.array(y)).get(
        "balanced_accuracy", float("nan")
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnose collapsed video features")
    ap.add_argument("--n", type=int, default=100, help="actions to sample")
    ap.add_argument("--backbone", default=config.PILOT_BACKBONE)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    actions, clips = sample(args.n)
    paths = clips["path"].tolist()
    keys = [fx.clip_key(a, i) for a, i in zip(clips["action_id"], clips["clip_index"])]
    print(f"{len(actions)} actions -> {len(paths)} clips, {len(WINDOWS)} windows\n")

    extractor = fx.build_extractor(args.backbone)
    rows = []

    for wname, (start, end) in WINDOWS.items():
        variants = fx.extract_both_poolings(
            paths, extractor, batch_size=args.batch_size, start=start, end=end
        )
        for pooling, feats in variants.items():
            vc = view_contrast(feats, clips, keys)
            rows.append({
                "window": wname,
                "pooling": pooling,
                "same": vc["same"],
                "random": vc["random"],
                "contrast": vc["contrast"],
                "camera": camera_probe(feats, clips, keys),
                "card": card_probe(feats, clips, keys, actions),
            })
            r = rows[-1]
            print(
                f"{wname:14} {pooling:8} contrast={r['contrast']:+.4f} "
                f"(same={r['same']:.3f} rand={r['random']:.3f})  "
                f"camera={r['camera']:.3f}  card={r['card']:.3f}"
            )

    table = pd.DataFrame(rows).sort_values("contrast", ascending=False)
    best = table.iloc[0]

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Feature diagnosis",
        "",
        f"Sample: {len(actions)} actions / {len(paths)} clips, backbone `{args.backbone}`.",
        "",
        "`contrast` = mean cosine between views of one incident minus mean cosine",
        "between random clips. Positive means the embedding encodes the incident",
        "rather than the sport. This is the number that matters.",
        "",
        table.to_markdown(index=False, floatfmt=".4f"),
        "",
        f"**Best: `{best['window']}` + `{best['pooling']}`, contrast {best['contrast']:+.4f}.**",
        "",
    ]
    if best["contrast"] <= 0.01:
        lines += [
            "No configuration produces a usefully positive contrast. The problem is not",
            "the window or the normalisation. Next suspects: the image processor's",
            "resize/crop of a 398x224 frame, or the backbone itself - try",
            "`videomae_base`, or a backbone pretrained on something closer to sport.",
        ]
    else:
        lines += [
            "Adopt this configuration in `src/config.py` before the full extraction.",
            "The camera probe should sit comfortably above 0.5; if it does not, the",
            "features still fail to encode basic appearance and the card number means",
            "nothing.",
        ]

    out = ARTIFACTS / "diagnose.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nbest: {best['window']} + {best['pooling']}  contrast={best['contrast']:+.4f}")
    print(f"report: {out}")


if __name__ == "__main__":
    main()
