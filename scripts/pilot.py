"""Feasibility pilot: does a frozen backbone encode anything about fouls?

Run before committing to a full 8,297-clip extraction. The design point is that
this is a *ladder* of probes, not a single number. Testing only "card / no card"
gives an uninterpretable result when it fails, because a low score cannot
distinguish "these features are useless" from "this task is genuinely hard".

The ladder, easiest to hardest:

    contact        visually obvious. The sanity floor. If this fails, the
                   problem is preprocessing or the backbone, and no other
                   number on the page means anything.
    action_class   mid-difficulty semantics; a real auxiliary head later.
    offence        cascade stage 1 (88/12).
    card           cascade stage 2 (65/35) - the decision that matters.

So: contact fails -> fix the inputs. contact passes but card fails -> the
features lack fine-grained motion and fine-tuning is mandatory rather than
optional. Both pass -> extract everything.

Staged so it can be stopped early. Stage 0 measures throughput and projects the
cost of the full run before anything expensive happens.

    python scripts/pilot.py --stage 0
    python scripts/pilot.py --stage all
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.data.annotations import load_split
from src.features import extract as fx
from src.features import probe as pb

TOTAL_CORPUS_CLIPS = 8297
ARTIFACTS = config.ARTIFACTS / "pilot"

# Measured from the train annotations: 2241 actions with 2 views, 561 with 3,
# 114 with 4.
AVG_VIEWS_PER_ACTION = 2.27


def artifacts_dir() -> Path:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------

def load_train() -> tuple[pd.DataFrame, pd.DataFrame]:
    split_dir = config.DATA_ROOT / "mvfouls" / config.SPLIT_DIRS["train"]
    return load_split(split_dir)


def sample_actions(actions: pd.DataFrame, n: int, seed: int = config.SEED) -> pd.DataFrame:
    """Stratified subsample on the two cascade targets.

    Stratifying keeps the probe's class balance close to the real one, so the
    pilot's balanced accuracy is comparable to what the full run should give.
    """
    from sklearn.model_selection import train_test_split

    if n >= len(actions):
        return actions.copy()

    key = (
        actions["target_offence"].fillna("na").astype(str)
        + "|"
        + actions["target_card"].fillna("na").astype(str)
    )
    # train_test_split needs at least two members per stratum.
    counts = key.value_counts()
    key = key.where(~key.isin(counts[counts < 2].index), "other")

    sample, _ = train_test_split(actions, train_size=n, stratify=key, random_state=seed)
    return sample.reset_index(drop=True)


# --------------------------------------------------------------------------
# Stage 0 - throughput, before spending anything
# --------------------------------------------------------------------------

def stage0(backbone: str, batch_size: int, n_actions: int) -> None:
    _, clips = load_train()
    paths = clips["path"].tolist()[:20]

    t0 = time.time()
    for p in paths:
        fx.sample_window(p)
    decode_ms = (time.time() - t0) / len(paths) * 1000
    print(f"decode          {decode_ms:6.0f} ms/clip")

    print(f"loading backbone {backbone!r} ...")
    extractor = fx.build_extractor(backbone)
    device = extractor[2]

    # Warm up first - the opening forward pass pays one-off allocation costs.
    fx.extract_clips(paths[:batch_size], extractor, batch_size=batch_size, progress=False)

    t0 = time.time()
    fx.extract_clips(paths, extractor, batch_size=batch_size, progress=False)
    total_ms = (time.time() - t0) / len(paths) * 1000

    pilot_clips = int(n_actions * AVG_VIEWS_PER_ACTION)
    print(f"decode+forward  {total_ms:6.0f} ms/clip   (device: {device})")
    print()
    print(f"pilot   {pilot_clips:>6,} clips  ->  {pilot_clips * total_ms / 60000:6.1f} min")
    print(f"full    {TOTAL_CORPUS_CLIPS:>6,} clips  ->  {TOTAL_CORPUS_CLIPS * total_ms / 60000:6.1f} min")

    if TOTAL_CORPUS_CLIPS * total_ms / 3600000 > 3:
        print("\nWARNING: full extraction projects past 3 hours here. Run it on Colab.")


# --------------------------------------------------------------------------
# Stage 1 - extract and cache
# --------------------------------------------------------------------------

def stage1(backbone: str, batch_size: int, n_actions: int) -> None:
    actions, clips = load_train()
    sample = sample_actions(actions, n_actions)
    sub = clips[clips["action_id"].isin(set(sample["action_id"]))].reset_index(drop=True)

    print(f"sampled {len(sample):,} actions -> {len(sub):,} clips")
    missing = [p for p in sub["path"] if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} clip files missing, e.g. {missing[0]}")

    extractor = fx.build_extractor(backbone)
    feats = fx.extract_clips(sub["path"].tolist(), extractor, batch_size=batch_size)
    keys = [fx.clip_key(a, i) for a, i in zip(sub["action_id"], sub["clip_index"])]

    path = fx.save_cache(keys, feats, backbone, tag=f"pilot{n_actions}")
    out = artifacts_dir()
    sample.to_csv(out / f"pilot{n_actions}_actions.csv", index=False)
    sub.to_csv(out / f"pilot{n_actions}_clips.csv", index=False)
    print(f"cached {feats.shape} -> {path}")


# --------------------------------------------------------------------------
# Stage 2 - the probes
# --------------------------------------------------------------------------

def build_matrices(keys: list[str], feats: np.ndarray, clips: pd.DataFrame):
    """Per-action feature matrices under two view policies.

    Returns ``(action_ids, clip0, mean_pooled, extras)`` where ``extras`` holds
    the replay-speed and camera-type side channels.
    """
    index = {k: i for i, k in enumerate(keys)}
    by_action: dict[str, list[int]] = {}
    for a, i in zip(clips["action_id"], clips["clip_index"]):
        k = fx.clip_key(a, i)
        if k in index:
            by_action.setdefault(a, []).append(index[k])

    cam_types = clips["camera_type"].value_counts().head(4).index.tolist()
    speeds_by_action = {
        a: pd.to_numeric(g["replay_speed"], errors="coerce").fillna(1.0)
        for a, g in clips.groupby("action_id")
    }
    cams_by_action = {a: set(g["camera_type"]) for a, g in clips.groupby("action_id")}

    ids, clip0, pooled, extras = [], [], [], []
    for action_id, rows in by_action.items():
        rows = sorted(rows)
        ids.append(action_id)
        clip0.append(feats[rows[0]])
        pooled.append(feats[rows].mean(axis=0))

        speeds = speeds_by_action[action_id]
        cams = cams_by_action[action_id]
        extras.append(
            [len(rows), float(speeds.max()), float(speeds.mean())]
            + [float(c in cams) for c in cam_types]
        )

    return ids, np.stack(clip0), np.stack(pooled), np.asarray(extras, dtype=np.float32)


PROBE_TARGETS = [
    ("contact", "Contact", "sanity floor - visually obvious"),
    ("action_class", "Action class", "mid-difficulty semantics"),
    ("offence", "target_offence", "cascade stage 1"),
    ("card", "target_card", "cascade stage 2 - the one that matters"),
]


def _clean_labels(labels: pd.DataFrame, ids: list[str], column: str):
    """Return ``(mask, y)`` dropping blank and missing labels."""
    raw = labels.loc[ids, column]
    values = raw.astype(str).values
    mask = raw.notna().values & (values != "") & (values != "nan")
    return mask, values[mask]


def stage2(backbone: str, n_actions: int, permutations: int) -> dict:
    keys, feats = fx.load_cache(backbone, tag=f"pilot{n_actions}")
    out = artifacts_dir()
    sample = pd.read_csv(out / f"pilot{n_actions}_actions.csv", dtype={"action_id": str})
    clips = pd.read_csv(out / f"pilot{n_actions}_clips.csv", dtype={"action_id": str})

    ids, clip0, pooled, extras = build_matrices(keys, feats, clips)
    labels = sample.set_index("action_id")

    results: dict = {
        "backbone": backbone,
        "n_actions": len(ids),
        "feature_dim": int(feats.shape[1]),
        "probes": {},
        "view_policy": {},
        "side_channels": {},
    }

    for name, column, note in PROBE_TARGETS:
        mask, y = _clean_labels(labels, ids, column)
        if len(np.unique(y)) < 2:
            continue

        res = pb.linear_probe(pooled[mask], y)
        res["note"] = note
        res["null"] = pb.permutation_null(pooled[mask], y, n=permutations)
        results["probes"][name] = res
        print(
            f"{name:14} n={res['n']:<5} bacc={res['balanced_accuracy']:.3f}"
            f" +/-{res['std']:.3f}  chance={res['chance']:.2f}"
            f"  p={res['null']['p_value']:.3f}"
        )

    # Side question 1: does pooling views beat the first view alone?
    mask, y = _clean_labels(labels, ids, "target_card")
    for policy, X in (("clip0", clip0), ("mean_views", pooled)):
        r = pb.linear_probe(X[mask], y)
        results["view_policy"][policy] = r["balanced_accuracy"]
        print(f"view:{policy:<11} card bacc={r['balanced_accuracy']:.3f}")

    # Side question 2: do replay speed and camera type carry anything?
    augmented = np.hstack([pooled, extras])
    r = pb.linear_probe(augmented[mask], y)
    results["side_channels"] = {
        "with_metadata": r["balanced_accuracy"],
        "without_metadata": results["view_policy"]["mean_views"],
    }
    print(f"card + metadata   bacc={r['balanced_accuracy']:.3f}")

    return results


# --------------------------------------------------------------------------
# Stage 3 - verdict and report
# --------------------------------------------------------------------------

def verdict(results: dict) -> tuple[str, str]:
    probes = results.get("probes", {})
    contact = probes.get("contact", {}).get("balanced_accuracy", 0.0)
    card = probes.get("card", {}).get("balanced_accuracy", 0.0)
    card_p = probes.get("card", {}).get("null", {}).get("p_value", 1.0)

    if contact < config.PILOT_SANITY_THRESHOLD:
        return "STOP", (
            f"The sanity floor failed: `contact` scored {contact:.3f}, below "
            f"{config.PILOT_SANITY_THRESHOLD}. Whether contact occurred is visually "
            "obvious, so this points at preprocessing or the backbone, not at task "
            "difficulty. Every other number here is uninterpretable until it passes."
        )
    if card >= config.PILOT_GO_THRESHOLD and card_p < 0.05:
        return "GO", (
            f"Stage 2 reached {card:.3f} balanced accuracy (p={card_p:.3f}), clearing "
            f"the {config.PILOT_GO_THRESHOLD} bar against a permutation null. Frozen "
            "features carry decodable signal; proceed to full extraction."
        )
    return "FINE-TUNE", (
        f"The sanity floor passed ({contact:.3f}) but stage 2 reached only {card:.3f} "
        f"(p={card_p:.3f}), short of {config.PILOT_GO_THRESHOLD}. The features work but "
        "lack the fine-grained motion detail this task needs. Fine-tuning is mandatory "
        "rather than optional - budget the Colab GPU run accordingly."
    )


def write_report(results: dict) -> Path:
    out = artifacts_dir()

    # Keep the raw null samples out of the JSON; the figure is generated in the
    # same run and the distributions are large.
    slim = json.loads(json.dumps(results))
    for res in slim.get("probes", {}).values():
        res.get("null", {}).pop("null", None)
    (out / "results.json").write_text(json.dumps(slim, indent=2), encoding="utf-8")

    tag, reasoning = verdict(results)
    nan = float("nan")
    lines = [
        "# Pilot report",
        "",
        f"**Verdict: {tag}**",
        "",
        reasoning,
        "",
        f"- backbone: `{results['backbone']}`",
        f"- actions: {results['n_actions']}",
        f"- feature dim: {results['feature_dim']}",
        "- views: mean-pooled",
        "",
        "## Probe ladder",
        "",
        "| probe | n | classes | chance | balanced acc | null mean | p | note |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, res in results["probes"].items():
        null = res.get("null", {})
        lines.append(
            f"| `{name}` | {res['n']} | {res['n_classes']} | {res['chance']:.2f} | "
            f"**{res['balanced_accuracy']:.3f}** ± {res['std']:.3f} | "
            f"{null.get('null_mean', nan):.3f} | {null.get('p_value', nan):.3f} | "
            f"{res['note']} |"
        )

    vp = results["view_policy"]
    sc = results["side_channels"]
    lines += [
        "",
        "## View pooling (stage-2 target)",
        "",
        f"- first view only: {vp.get('clip0', nan):.3f}",
        f"- mean over views: {vp.get('mean_views', nan):.3f}",
        "",
        "If pooling wins, the full cache must stay **per-clip** so that pooling remains a "
        "training-time choice rather than something baked into an expensive extraction.",
        "",
        "## Replay speed and camera type",
        "",
        f"- embeddings only: {sc.get('without_metadata', nan):.3f}",
        f"- plus metadata: {sc.get('with_metadata', nan):.3f}",
        "",
        "A gain here argues for carrying these fields into `HakamContract`.",
        "",
        "## Caveats",
        "",
        "- Train split only; valid and test were never touched.",
        "- Balanced accuracy throughout - every target here is skewed.",
        "- p-values come from label permutation, not the nominal chance level, so they "
        "reflect this sample size and this CV split.",
        "- A linear probe is a floor, not a ceiling. It bounds what is *linearly* "
        "decodable from frozen features; a trained head may do better.",
    ]
    path = out / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def plot(results: dict) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    probes = results["probes"]
    names = list(probes)
    if not names:
        return None

    obs = np.array([probes[n]["balanced_accuracy"] for n in names])
    null_mean = np.array([probes[n]["null"].get("null_mean", 0.5) for n in names])
    null_p95 = np.array([probes[n]["null"].get("null_p95", 0.5) for n in names])

    fig, ax = plt.subplots(figsize=(7.5, 4))
    x = np.arange(len(names))
    ax.bar(x, obs, 0.55, label="observed", color="#2b6cb0")
    ax.errorbar(
        x, null_mean,
        yerr=[np.zeros(len(names)), null_p95 - null_mean],
        fmt="_", color="#c53030", capsize=10, markersize=18, linewidth=2,
        label="permutation null (mean to p95)",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("balanced accuracy (5-fold CV)")
    ax.set_title(f"Pilot probe ladder - {results['backbone']}")
    ax.legend()
    fig.tight_layout()
    path = artifacts_dir() / "probes.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="Hakam feasibility pilot")
    ap.add_argument("--stage", default="all", choices=["0", "1", "2", "all"])
    ap.add_argument("--n", type=int, default=config.PILOT_N)
    ap.add_argument("--backbone", default=config.PILOT_BACKBONE)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--permutations", type=int, default=config.PILOT_PERMUTATIONS)
    args = ap.parse_args()

    if args.stage in ("0", "all"):
        stage0(args.backbone, args.batch_size, args.n)
    if args.stage in ("1", "all"):
        stage1(args.backbone, args.batch_size, args.n)
    if args.stage in ("2", "all"):
        results = stage2(args.backbone, args.n, args.permutations)
        report = write_report(results)
        fig = plot(results)
        tag, reasoning = verdict(results)
        print(f"\n=== {tag} ===\n{reasoning}\n")
        print(f"report: {report}")
        if fig:
            print(f"figure: {fig}")


if __name__ == "__main__":
    main()
