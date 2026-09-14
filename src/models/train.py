"""Train and evaluate one cascade stage.

Two modes, sharing everything downstream so their numbers are comparable:

  probe     freeze the backbone, extract embeddings once, fit logistic
            regression. Seconds. This is the baseline section 7 requires, and
            the control that makes fine-tuning's gain measurable.
  finetune  unfreeze the later blocks and train end to end. The diagnostic on
            2 Sep showed frozen Kinetics features encode camera framing (0.94)
            and not fouls (0.47), so this is where the real gain has to come
            from.

Metrics are chosen for a screening tool, not a decision-maker. The costly error
is a *miss* - a card-worthy incident waved on - so the headline is recall on the
card class **reported with its review load**. Recall alone is trivially faked by
predicting "card" every time, so the two are never printed apart.

Balanced accuracy is reported too, because it is what the literature quotes and
what the experiment comparison needs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src import config
from src.data import transforms as tfm
from src.features import extract as fx
from src.models.data import build_loaders

RUNS_DIR = config.ARTIFACTS / "runs"


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def balanced_accuracy(labels: np.ndarray, pred: np.ndarray) -> float:
    recalls = []
    for c in np.unique(labels):
        mask = labels == c
        if mask.any():
            recalls.append((pred[mask] == c).mean())
    return float(np.mean(recalls)) if recalls else float("nan")


def threshold_for_recall(
    labels: np.ndarray, scores: np.ndarray, target: float = 0.85
) -> float:
    """Highest threshold still reaching ``target`` recall on the positive class.

    Chosen on validation and then frozen. Tuning it on test would be marking our
    own homework.
    """
    positive = labels == 1
    if not positive.any():
        return 0.5
    best = 0.0
    for t in sorted(np.unique(np.round(scores, 4))):
        if (scores[positive] >= t).mean() >= target:
            best = float(t)
        else:
            break
    return best


def report(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """Everything needed to describe one operating point honestly."""
    pred = (scores >= threshold).astype(int)
    positive, negative = labels == 1, labels == 0

    tp = int(((pred == 1) & positive).sum())
    fp = int(((pred == 1) & negative).sum())
    fn = int(((pred == 0) & positive).sum())

    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)

    # Review load: the share of incidents a human would have to look at. A
    # recall number without this is meaningless - always predicting "card"
    # gives recall 1.0 at load 1.0.
    return {
        "threshold": round(float(threshold), 4),
        "balanced_accuracy": round(balanced_accuracy(labels, pred), 4),
        "recall_card": round(float(recall), 4),
        "precision_card": round(float(precision), 4),
        "miss_rate": round(float(1 - recall), 4),
        "review_load": round(float(pred.mean()), 4),
        "n": int(len(labels)),
        "tp": tp, "fp": fp, "fn": fn,
    }


def selective_accuracy(
    labels: np.ndarray, scores: np.ndarray, coverage: float = 0.6
) -> dict:
    """Accuracy on the most-confident slice - the honest headline number.

    Confidence is distance from 0.5. At 60% coverage this answers: on the cases
    the system chooses to answer, how often is it right?
    """
    confidence = np.abs(scores - 0.5)
    k = max(1, int(round(coverage * len(scores))))
    keep = np.argsort(-confidence)[:k]
    pred = (scores[keep] >= 0.5).astype(int)
    return {
        "coverage": round(float(k / len(scores)), 3),
        "selective_accuracy": round(float((pred == labels[keep]).mean()), 4),
        "selective_balanced_accuracy": round(balanced_accuracy(labels[keep], pred), 4),
    }


def clustered_ci(
    labels: np.ndarray, scores: np.ndarray, matches: np.ndarray | None,
    threshold: float = 0.5, n_boot: int = 1000, seed: int = config.SEED,
) -> list[float] | None:
    """95% bootstrap CI on balanced accuracy, resampling whole matches.

    Incidents from one match share a referee and a broadcast, so they are not
    independent. Resampling matches rather than incidents keeps that correlation,
    which gives an honestly wider interval on a split with few matches.
    """
    if matches is None or len(matches) != len(labels):
        return None
    rng = np.random.default_rng(seed)
    pred = (scores >= threshold).astype(int)
    groups = [np.flatnonzero(matches == m) for m in np.unique(matches)]
    stats = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        if len(np.unique(labels[idx])) < 2:
            continue
        stats.append(balanced_accuracy(labels[idx], pred[idx]))
    if not stats:
        return None
    return [round(float(np.percentile(stats, 2.5)), 4), round(float(np.percentile(stats, 97.5)), 4)]


def _matches_of(loader) -> np.ndarray | None:
    ds = loader.dataset
    if hasattr(ds, "indices"):                       # torch Subset
        base = getattr(ds.dataset, "matches", None)
        return None if base is None else base[np.asarray(ds.indices)]
    return getattr(ds, "matches", None)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

def build_model(backbone_key: str, num_labels: int, freeze_blocks: int,
                head_dropout: float = 0.0):
    """VideoMAE with a fresh classification head.

    ``freeze_blocks`` freezes the patch embedding plus that many encoder layers.
    With 2,916 training actions against 87M parameters, freezing the first half
    is the main guard against memorising the training set.
    """
    from transformers import VideoMAEForVideoClassification

    spec = config.BACKBONES[backbone_key]
    model = VideoMAEForVideoClassification.from_pretrained(
        spec["name"], num_labels=num_labels, ignore_mismatched_sizes=True
    )

    restored = fx.repair_videomae_biases(model, spec["name"])
    if restored == 0:
        raise RuntimeError(
            f"no attention biases restored for {spec['name']!r} - transformers "
            "silently zeroes them; verify before trusting this run."
        )
    print(f"  restored {restored} attention biases")

    if head_dropout > 0:
        model.classifier = nn.Sequential(nn.Dropout(head_dropout), model.classifier)

    if freeze_blocks > 0:
        for p in model.videomae.embeddings.parameters():
            p.requires_grad_(False)
        for layer in model.videomae.encoder.layer[:freeze_blocks]:
            for p in layer.parameters():
                p.requires_grad_(False)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  trainable {trainable/1e6:.1f}M / {total/1e6:.1f}M "
          f"(froze embeddings + {freeze_blocks} blocks)")
    return model


@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Positive-class probability and label for every sample."""
    model.eval()
    scores, labels = [], []
    for batch in loader:
        x = batch["pixel_values"].to(device, non_blocking=True)
        logits = model(pixel_values=x).logits
        scores.append(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy())
        labels.append(batch["label"].numpy())
    return np.concatenate(scores), np.concatenate(labels)


# --------------------------------------------------------------------------
# Frozen probe - the baseline
# --------------------------------------------------------------------------

@torch.no_grad()
def embed(loader, backbone, device) -> tuple[np.ndarray, np.ndarray]:
    backbone.eval()
    feats, labels = [], []
    for batch in loader:
        x = batch["pixel_values"].to(device, non_blocking=True)
        h = backbone(pixel_values=x).last_hidden_state.mean(dim=1)
        feats.append(h.float().cpu().numpy())
        labels.append(batch["label"].numpy())
    return np.concatenate(feats), np.concatenate(labels)


def run_probe(args, loaders, device) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from transformers import VideoMAEForVideoClassification

    spec = config.BACKBONES[args.backbone]
    full = VideoMAEForVideoClassification.from_pretrained(spec["name"])
    if fx.repair_videomae_biases(full, spec["name"]) == 0:
        raise RuntimeError("no attention biases restored - refusing to proceed")
    backbone = full.videomae.to(device).eval()

    print("  embedding train...")
    xtr, ytr = embed(loaders["train"], backbone, device)
    print("  embedding valid...")
    xva, yva = embed(loaders["valid"], backbone, device)

    scaler = StandardScaler().fit(xtr)
    clf = LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=config.SEED
    ).fit(scaler.transform(xtr), ytr)

    return {"scores": clf.predict_proba(scaler.transform(xva))[:, 1], "labels": yva}


# --------------------------------------------------------------------------
# Fine-tuning
# --------------------------------------------------------------------------

def run_finetune(args, loaders, device) -> dict:
    model = build_model(args.backbone, 2, args.freeze_blocks, args.head_dropout).to(device)

    weights = loaders["train"].dataset.class_weights().to(device) \
        if hasattr(loaders["train"].dataset, "class_weights") else None
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)

    params = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    steps = max(1, args.epochs * len(loaders["train"]))
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=steps)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best = {"balanced_accuracy": -1.0}
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total = time.time(), 0.0
        for step, batch in enumerate(loaders["train"], 1):
            x = batch["pixel_values"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = criterion(model(pixel_values=x).logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            optimiser.zero_grad(set_to_none=True)
            schedule.step()
            total += loss.item()

            if step % 50 == 0:
                print(f"    epoch {epoch} step {step}/{len(loaders['train'])} "
                      f"loss {total/step:.4f}")

        scores, labels = predict(model, loaders["valid"], device)
        bacc = balanced_accuracy(labels, (scores >= 0.5).astype(int))
        p_true = np.clip(np.where(labels == 1, scores, 1 - scores), 1e-7, 1.0)
        valid_loss = float(-np.log(p_true).mean())
        train_loss = total / len(loaders["train"])
        history.append({"epoch": epoch, "train_loss": round(train_loss, 4),
                        "valid_loss": round(valid_loss, 4),
                        "valid_balanced_accuracy": round(bacc, 4)})
        print(f"  epoch {epoch}: train loss {train_loss:.4f}  valid loss {valid_loss:.4f}  "
              f"valid balanced acc {bacc:.4f}  ({(time.time()-t0)/60:.1f} min)")

        if bacc > best["balanced_accuracy"]:
            best = {"balanced_accuracy": bacc, "epoch": epoch,
                    "scores": scores, "labels": labels}
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

    if best_state is not None and args.save:
        out = RUNS_DIR / args.name
        out.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, out / "best.pt")

    return {"scores": best["scores"], "labels": best["labels"],
            "best_epoch": best.get("epoch"), "history": history}


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Train one cascade stage")
    ap.add_argument("--mode", default="probe", choices=["probe", "finetune"])
    ap.add_argument("--stage", default="card", choices=["card", "offence"])
    ap.add_argument("--geometry", default="crop", choices=["crop", "resize", "zoom"])
    ap.add_argument("--backbone", default="videomae_small")
    ap.add_argument("--augment", default=None, choices=["mild_aug_v1"])
    ap.add_argument("--freeze-blocks", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--head-dropout", type=float, default=0.3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--target-recall", type=float, default=0.85)
    ap.add_argument("--limit", type=int, default=0, help="smoke test on N samples")
    ap.add_argument("--name", default=None)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    args.name = args.name or f"{args.stage}_{args.mode}_{args.geometry}"
    device = fx.resolve_device()
    augment = tfm.MILD_V1 if args.augment == "mild_aug_v1" else None

    print(f"run={args.name}  device={device}  geometry={args.geometry}")
    print(f"augment: {tfm.describe(augment)}")

    loaders = build_loaders(
        stage=args.stage, geometry_mode=args.geometry, augment=augment,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )
    train_ds = loaders["train"].dataset

    if args.limit:
        from torch.utils.data import DataLoader, Subset

        for split, loader in loaders.items():
            n = min(args.limit, len(loader.dataset))
            loaders[split] = DataLoader(
                Subset(loader.dataset, range(n)), batch_size=args.batch_size
            )
        loaders["train"].dataset.class_weights = train_ds.class_weights
        print(f"  SMOKE TEST: {args.limit} samples per split")

    t0 = time.time()
    result = run_probe(args, loaders, device) if args.mode == "probe" \
        else run_finetune(args, loaders, device)

    scores, labels = result["scores"], result["labels"]
    threshold = threshold_for_recall(labels, scores, args.target_recall)
    recall_key = f"at_recall_{args.target_recall}"

    metrics = {
        "run": args.name,
        "finished_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mode": args.mode, "stage": args.stage, "geometry": args.geometry,
        "backbone": args.backbone, "augment": tfm.describe(augment),
        "freeze_blocks": args.freeze_blocks if args.mode == "finetune" else None,
        "epochs": args.epochs if args.mode == "finetune" else None,
        "best_epoch": result.get("best_epoch"),
        "minutes": round((time.time() - t0) / 60, 1),
        "argmax": report(labels, scores, 0.5),
        recall_key: report(labels, scores, threshold),
        "selective": selective_accuracy(labels, scores, 0.6),
        "balanced_accuracy_ci95": clustered_ci(labels, scores, _matches_of(loaders["valid"])),
        "hyperparameters": {"lr": args.lr, "weight_decay": args.weight_decay,
                            "label_smoothing": args.label_smoothing,
                            "head_dropout": args.head_dropout},
        "history": result.get("history", []),
    }

    out = RUNS_DIR / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    np.savez(out / "scores.npz", scores=scores, labels=labels)

    a, r, s = metrics["argmax"], metrics[recall_key], metrics["selective"]
    print(f"\n=== {args.name} ({metrics['minutes']} min) ===")
    print(f"  stage {args.stage}, argmax     : balanced acc {a['balanced_accuracy']:.3f}"
          f"   recall {a['recall_card']:.3f}   review load {a['review_load']:.0%}")
    print(f"  stage {args.stage}, @recall {args.target_recall}: recall "
          f"{r['recall_card']:.3f}   precision {r['precision_card']:.3f}"
          f"   review load {r['review_load']:.0%}")
    print(f"  selective @{s['coverage']:.0%} coverage: accuracy "
          f"{s['selective_accuracy']:.3f}")
    print(f"  balanced acc 95% CI (match-clustered): {metrics['balanced_accuracy_ci95']}")
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
