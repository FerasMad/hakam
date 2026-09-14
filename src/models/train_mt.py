"""Multi-view, multi-task fine-tuning.

    python -m src.models.train_mt --tasks card offence action_class body_part \
        --backbone videomae_base --epochs 5 --ema 0.999 --soft-borderline --name mv10

Why this exists, from the round 3 learning curves:

- Only VideoMAE-base lowered validation loss; every run peaked at epoch 3-5 and
  then memorised. The model is short of data, not of steps. Each action has 2-4
  views and training used one, so every view becomes a training sample here, and
  the softmax is averaged over an action's views at evaluation.
- One backbone with a head per task lets offence (324 no-offence clips) and the
  contract's attributes learn from every annotated action instead of a subset.
- Best-epoch numbers on 299 actions swing by 3-4 points, so weights are averaged
  (EMA) and runs are compared on the last three epochs.

``--predict-splits test`` writes test scores without computing a single metric.
The test split is scored once, by scripts/ensemble.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src import config
from src.data import transforms as tfm
from src.features import extract as fx
from src.models.data import IGNORE, TASKS, CombinedViews, MultiViewDataset, normalise_on_device
from src.models.train import RUNS_DIR, balanced_accuracy, build_model

LOSS_WEIGHTS = {"card": 1.0, "offence": 1.0, "action_class": 0.5, "body_part": 0.5}


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

class MultiTaskVideoMAE(nn.Module):
    """VideoMAE encoder, the checkpoint's own pooling norm, one linear head per task."""

    def __init__(self, backbone_key: str, tasks: list[str], freeze_blocks: int,
                 dropout: float):
        super().__init__()
        base = build_model(backbone_key, 2, freeze_blocks, 0.0)
        self.videomae = base.videomae
        self.fc_norm = base.fc_norm
        dim = config.BACKBONES[backbone_key]["dim"]
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleDict({t: nn.Linear(dim, len(TASKS[t])) for t in tasks})

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.videomae(pixel_values=x).last_hidden_state
        h = self.fc_norm(h.mean(1)) if self.fc_norm is not None else h[:, 0]
        h = self.dropout(h)
        return {t: head(h) for t, head in self.heads.items()}


def task_loss(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor,
              smoothing: float, soft: torch.Tensor | None = None) -> torch.Tensor:
    """Class-weighted cross-entropy over rows that carry a label.

    ``labels`` uses IGNORE for missing annotations. ``soft`` (card only) is the
    probability of class 1, or negative where there is none.
    """
    n = logits.shape[1]
    mask = labels != IGNORE
    target = torch.zeros(logits.shape, dtype=torch.float32, device=logits.device)
    rows = mask.nonzero().squeeze(1)
    target[rows, labels[rows]] = 1.0
    if soft is not None:
        s = soft >= 0
        target[s, 1] = soft[s]
        target[s, 0] = 1.0 - soft[s]
        mask = mask | s
    if not mask.any():
        return logits.sum() * 0.0
    target = target[mask] * (1.0 - smoothing) + smoothing / n
    per_row = -(target * torch.log_softmax(logits[mask].float(), dim=-1)).sum(-1)
    w = (target * weights).sum(-1)
    return (w * per_row).sum() / w.sum()


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def average_by_action(action_ids: np.ndarray, probs: np.ndarray):
    """Mean probability over each action's views. Returns (ids, probs, first_index)."""
    ids, first, inverse = np.unique(action_ids, return_index=True, return_inverse=True)
    sums = np.zeros((len(ids), probs.shape[1]), dtype=np.float64)
    np.add.at(sums, inverse, probs)
    counts = np.bincount(inverse, minlength=len(ids))[:, None]
    return ids, sums / counts, first


def best_threshold(labels: np.ndarray, p1: np.ndarray) -> float:
    """Threshold on P(class 1) maximising balanced accuracy; ties go nearest 0.5."""
    grid = np.round(np.linspace(0.05, 0.95, 91), 2)
    scores = [balanced_accuracy(labels, (p1 >= t).astype(int)) for t in grid]
    best = max(scores)
    candidates = [t for t, s in zip(grid, scores) if s >= best - 1e-9]
    return float(min(candidates, key=lambda t: abs(t - 0.5)))


def clustered_ci(labels, pred, matches, n_boot: int = 1000, seed: int = config.SEED):
    """95% CI on balanced accuracy, resampling whole matches."""
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(matches == m) for m in np.unique(matches)]
    stats = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        if len(np.unique(labels[idx])) > 1:
            stats.append(balanced_accuracy(labels[idx], pred[idx]))
    if not stats:
        return None
    return [round(float(np.percentile(stats, 2.5)), 4), round(float(np.percentile(stats, 97.5)), 4)]


def task_metrics(task: str, labels: np.ndarray, probs: np.ndarray, matches: np.ndarray,
                 threshold: float | None = None) -> dict:
    """Everything reported for one task on action-level probabilities."""
    keep = labels != IGNORE
    y, p, m = labels[keep], probs[keep], matches[keep]
    classes = TASKS[task]
    binary = len(classes) == 2
    pred = (p[:, 1] >= threshold).astype(int) if (binary and threshold is not None) else p.argmax(1)

    conf = np.where(pred == 1, p[:, 1], p[:, 0]) if binary else p.max(1)
    k = max(1, int(round(0.6 * len(y))))
    top = np.argsort(-conf)[:k]
    nll = -np.log(np.clip(p[np.arange(len(y)), y], 1e-7, 1.0))

    out = {
        "n": int(len(y)),
        "threshold": threshold,
        "balanced_accuracy": round(balanced_accuracy(y, pred), 4),
        "accuracy": round(float((pred == y).mean()), 4),
        "majority_accuracy": round(float(np.bincount(y).max() / len(y)), 4),
        "per_class_recall": {c: round(float((pred[y == i] == i).mean()), 4)
                             for i, c in enumerate(classes) if (y == i).any()},
        "per_class_n": {c: int((y == i).sum()) for i, c in enumerate(classes) if (y == i).any()},
        "selective_accuracy_60": round(float((pred[top] == y[top]).mean()), 4),
        "balanced_nll": round(float(np.mean([nll[y == i].mean() for i in np.unique(y)])), 4),
        "ci95": clustered_ci(y, pred, m),
    }
    if binary and threshold is None:
        out["tuned_threshold"] = best_threshold(y, p[:, 1])
        out["balanced_accuracy_tuned"] = round(
            balanced_accuracy(y, (p[:, 1] >= out["tuned_threshold"]).astype(int)), 4)
    return out


@torch.no_grad()
def predict(model, loader, device, tasks) -> dict[str, np.ndarray]:
    model.eval()
    n = len(loader.dataset)
    probs = {t: np.zeros((n, len(TASKS[t])), dtype=np.float32) for t in tasks}
    for batch in loader:
        x = normalise_on_device(batch["pixel_values"].to(device, non_blocking=True))
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            out = model(x)
        idx = batch["index"].numpy()
        for t in tasks:
            probs[t][idx] = torch.softmax(out[t].float(), dim=-1).cpu().numpy()
    return probs


def action_scores(ds: MultiViewDataset, probs: dict) -> dict:
    """Action-level arrays for saving and scoring: ids, matches, per task probs and labels."""
    out = {}
    for j, t in enumerate(ds.tasks):
        ids, p, first = average_by_action(ds.action_ids, probs[t])
        out["action_ids"], out["matches"] = ids, ds.matches[first]
        out[f"prob_{t}"], out[f"label_{t}"] = p, ds.labels[first, j]
    return out


def evaluate(ds: MultiViewDataset, probs: dict) -> dict:
    s = action_scores(ds, probs)
    return {t: task_metrics(t, s[f"label_{t}"], s[f"prob_{t}"], s["matches"]) for t in ds.tasks}


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def param_groups(model: MultiTaskVideoMAE, lr: float, head_lr: float, llrd: float) -> list[dict]:
    """AdamW groups: heads at ``head_lr``; backbone at ``lr``, or with layer-wise decay.

    With ``llrd`` > 0 block i of n trains at lr * llrd**(n-1-i): the top block at
    ``lr``, earlier blocks progressively slower. That is what lets more blocks be
    unfrozen without wiping the pretrained low-level features.
    """
    head = [p for p in model.heads.parameters() if p.requires_grad]
    seen = {id(p) for p in head}
    groups = [{"params": head, "lr": head_lr}]
    if llrd > 0:
        layers = model.videomae.encoder.layer
        for i, layer in enumerate(layers):
            params = [p for p in layer.parameters() if p.requires_grad]
            if params:
                groups.append({"params": params, "lr": lr * llrd ** (len(layers) - 1 - i)})
                seen.update(id(p) for p in params)
    rest = [p for p in model.parameters() if p.requires_grad and id(p) not in seen]
    if rest:
        groups.append({"params": rest, "lr": lr})
    return groups


def loader(ds, batch_size, workers, train, sampler=None):
    return DataLoader(ds, batch_size=batch_size, shuffle=train and sampler is None,
                      sampler=sampler, drop_last=train, num_workers=workers,
                      pin_memory=torch.cuda.is_available(),
                      persistent_workers=train and workers > 0)


def balanced_sampler(ds: MultiViewDataset, task: str):
    """Draw clips so each class of ``task`` is seen equally often.

    Offence is 724 no-offence clips against 5,687: class weights alone left the
    head predicting "offence" for everything (no-offence recall 0.09 in mv10).
    Clips without a label for the task keep the average weight.
    """
    from torch.utils.data import WeightedRandomSampler

    col = ds.labels[:, ds.tasks.index(task)]
    counts = np.bincount(col[col != IGNORE], minlength=len(TASKS[task])).astype(np.float64)
    per_class = 1.0 / np.maximum(counts, 1)
    w = np.where(col != IGNORE, per_class[np.clip(col, 0, None)], per_class.mean())
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(ds), replacement=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Multi-view multi-task fine-tuning")
    ap.add_argument("--tasks", nargs="+", default=["card"], choices=list(TASKS))
    ap.add_argument("--backbone", default="videomae_base")
    ap.add_argument("--cache-tag", default="mv")
    ap.add_argument("--views", default="all", choices=["all", "last"])
    ap.add_argument("--augment", default="mild_aug_v1", choices=["mild_aug_v1", "none"])
    ap.add_argument("--soft-borderline", action="store_true")
    ap.add_argument("--freeze-blocks", type=int, default=9)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--head-dropout", type=float, default=0.3)
    ap.add_argument("--ema", type=float, default=0.0, help="EMA decay, 0 = off")
    ap.add_argument("--warmup-epochs", type=int, default=1)
    ap.add_argument("--llrd", type=float, default=0.0, help="layer-wise lr decay, 0 = off")
    ap.add_argument("--select", default="final", choices=["final", "best"],
                    help="which epoch's weights score valid/test")
    ap.add_argument("--balance", default=None, choices=list(TASKS),
                    help="sample clips class-balanced for this task instead of weighting its loss")
    ap.add_argument("--loss-weights", nargs="*", default=[], metavar="TASK=W")
    ap.add_argument("--mixup", type=float, default=0.0, help="mixup Beta alpha, 0 = off")
    ap.add_argument("--soft-between", action="store_true",
                    help="train on 'Between' offences as a 0.5 offence target")
    ap.add_argument("--train-splits", nargs="+", default=["train"], choices=["train", "valid"])
    ap.add_argument("--no-eval", action="store_true", help="skip validation (refit on train+valid)")
    ap.add_argument("--stop-epoch", type=int, default=0,
                    help="stop after this epoch while keeping the --epochs lr schedule")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=min(8, os.cpu_count() or 2))
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--predict-splits", nargs="*", default=[], choices=["test"])
    ap.add_argument("--limit", type=int, default=0, help="smoke test on N clips per split")
    ap.add_argument("--save", action="store_true", help="keep the final weights")
    ap.add_argument("--name", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = fx.resolve_device()
    use_amp = device.type == "cuda"
    if use_amp:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    tasks = args.tasks
    primary = "card" if "card" in tasks else tasks[0]
    augment = tfm.MILD_V1 if args.augment == "mild_aug_v1" else None
    out_dir = RUNS_DIR / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"run={args.name}  device={device}  backbone={args.backbone}  tasks={tasks}  "
          f"views={args.views}  ema={args.ema}  soft={args.soft_borderline}  seed={args.seed}")

    parts = []
    for split in args.train_splits:
        part = MultiViewDataset(split, tasks, args.cache_tag, augment, args.views,
                                args.soft_borderline, soft_between=args.soft_between)
        part.split = "train"          # a refit trains on valid clips: augment them like train
        parts.append(part)
    valid_ds = None if args.no_eval else MultiViewDataset("valid", tasks, args.cache_tag, None, args.views)
    for ds in parts + ([valid_ds] if valid_ds is not None else []):
        if args.limit:
            for attr in ("rows", "labels", "soft_card", "soft_offence", "action_ids", "matches"):
                setattr(ds, attr, getattr(ds, attr)[: args.limit])
        print("  " + ds.distribution())
    train_ds = parts[0] if len(parts) == 1 else CombinedViews(parts)

    sampler = balanced_sampler(train_ds, args.balance) if args.balance else None
    train_loader = loader(train_ds, args.batch_size, args.num_workers, True, sampler)
    valid_loader = None if valid_ds is None else loader(valid_ds, args.batch_size, args.num_workers, False)

    model = MultiTaskVideoMAE(args.backbone, tasks, args.freeze_blocks, args.head_dropout).to(device)
    weights = {t: train_ds.class_weights(t).to(device) for t in tasks}
    if args.balance:
        weights[args.balance] = torch.ones_like(weights[args.balance])   # sampling already balances it
    loss_weights = dict(LOSS_WEIGHTS)
    loss_weights.update({k: float(v) for k, v in (kv.split("=") for kv in args.loss_weights)})

    optimiser = torch.optim.AdamW(param_groups(model, args.lr, args.head_lr, args.llrd),
                                  weight_decay=args.weight_decay)
    steps = max(1, args.epochs * len(train_loader))
    warmup = max(1, min(args.warmup_epochs * len(train_loader), steps // 2))

    def lr_factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + np.cos(np.pi * (step - warmup) / max(1, steps - warmup)))

    schedule = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_factor)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    ema = None
    if args.ema > 0:
        from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

        ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(args.ema), use_buffers=True)

    history, best = [], {"ba": -1.0}
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total = time.time(), 0.0
        for step, batch in enumerate(train_loader, 1):
            x = normalise_on_device(batch["pixel_values"].to(device, non_blocking=True))
            y = batch["labels"].to(device, non_blocking=True)
            soft = {"card": batch["soft_card"].to(device, non_blocking=True),
                    "offence": batch["soft_offence"].to(device, non_blocking=True)}
            lam, perm = 1.0, None
            if args.mixup > 0:
                # Blend two clips and both sets of targets. The strongest cheap
                # regulariser for a small video set; it is what lets 50 epochs help.
                lam = float(np.random.beta(args.mixup, args.mixup))
                perm = torch.randperm(x.size(0), device=device)
                x = lam * x + (1.0 - lam) * x[perm]
            with torch.autocast("cuda", enabled=use_amp):
                out = model(x)

            def batch_loss(idx):
                return sum(
                    loss_weights[t] * task_loss(
                        out[t], y[idx, j] if idx is not None else y[:, j], weights[t],
                        args.label_smoothing,
                        None if t not in soft else (soft[t][idx] if idx is not None else soft[t]))
                    for j, t in enumerate(tasks)
                )

            loss = batch_loss(None) if perm is None else lam * batch_loss(None) + (1 - lam) * batch_loss(perm)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            optimiser.zero_grad(set_to_none=True)
            schedule.step()
            if ema is not None:
                ema.update_parameters(model)
            total += loss.item()
            if step % 100 == 0:
                print(f"    epoch {epoch} step {step}/{len(train_loader)} loss {total/step:.4f}")

        eval_model = ema.module if ema is not None else model
        train_loss = total / max(1, len(train_loader))
        if valid_ds is None:
            history.append({"epoch": epoch, "train_loss": round(train_loss, 4)})
            print(f"  epoch {epoch}: train loss {train_loss:.4f}  ({(time.time()-t0)/60:.1f} min)")
            if args.stop_epoch and epoch >= args.stop_epoch:
                break
            continue
        probs = predict(eval_model, valid_loader, device, tasks)
        metrics = evaluate(valid_ds, probs)
        entry = {"epoch": epoch, "train_loss": round(train_loss, 4),
                 "valid_loss": metrics[primary]["balanced_nll"],
                 "valid_balanced_accuracy": metrics[primary]["balanced_accuracy"]}
        entry.update({f"{t}_ba": metrics[t]["balanced_accuracy"] for t in tasks})
        history.append(entry)
        print(f"  epoch {epoch}: train loss {train_loss:.4f}  "
              + "  ".join(f"{t} ba {metrics[t]['balanced_accuracy']:.3f} (nll {metrics[t]['balanced_nll']:.3f})"
                          for t in tasks)
              + f"  ({(time.time()-t0)/60:.1f} min)")

        if metrics[primary]["balanced_accuracy"] > best["ba"]:
            best = {"ba": metrics[primary]["balanced_accuracy"], "epoch": epoch, "metrics": metrics,
                    "probs": probs}
            if args.select == "best":
                best["state"] = {k: v.detach().cpu().clone() for k, v in eval_model.state_dict().items()}
        if args.stop_epoch and epoch >= args.stop_epoch:
            break

    final_model = ema.module if ema is not None else model
    if valid_ds is None:
        for split in args.predict_splits:
            ds = MultiViewDataset(split, tasks, args.cache_tag, None, args.views, require_label=False)
            s = action_scores(ds, predict(final_model, loader(ds, args.batch_size, args.num_workers, False),
                                          device, tasks))
            np.savez(out_dir / f"{split}_scores.npz", **s)
            print(f"  {split}: scored {len(s['action_ids'])} actions (no metrics computed)")
        if args.save:
            torch.save(final_model.state_dict(), out_dir / "final.pt")
        (out_dir / "metrics.json").write_text(json.dumps({
            "run": args.name, "mode": "refit", "stage": primary, "tasks": tasks,
            "train_splits": args.train_splits, "backbone": args.backbone, "epochs": args.epochs,
            "stop_epoch": args.stop_epoch, "minutes": round((time.time() - t_start) / 60, 1),
            "argmax": {"balanced_accuracy": None}, "selective": {"selective_accuracy": None},
            "hyperparameters": vars(args), "history": history,
        }, indent=2, default=str), encoding="utf-8")
        print(f"=== {args.name} refit done ({len(history)} epochs) -> {out_dir}")
        return 0
    if args.select == "best":
        # Long runs overfit after the peak; score valid and test with the checkpoint
        # chosen on valid, never the last epoch.
        final_model.load_state_dict(best["state"])
        probs, metrics = best["probs"], best["metrics"]
        print(f"  selected epoch {best['epoch']} ({primary} ba {best['ba']:.3f})")
    np.savez(out_dir / "valid_scores.npz", **action_scores(valid_ds, probs))
    for split in args.predict_splits:
        ds = MultiViewDataset(split, tasks, args.cache_tag, None, args.views, require_label=False)
        s = action_scores(ds, predict(final_model, loader(ds, args.batch_size, args.num_workers, False),
                                      device, tasks))
        np.savez(out_dir / f"{split}_scores.npz", **s)
        print(f"  {split}: scored {len(s['action_ids'])} actions (no metrics computed)")
    if args.save:
        torch.save(final_model.state_dict(), out_dir / "final.pt")

    tail = [h["valid_balanced_accuracy"] for h in history[-3:]]
    report = {
        "run": args.name,
        "finished_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mode": "multiview", "stage": primary, "tasks": tasks, "views": args.views,
        "backbone": args.backbone, "cache_tag": args.cache_tag,
        "augment": tfm.describe(augment), "freeze_blocks": args.freeze_blocks,
        "epochs": args.epochs, "best_epoch": best.get("epoch"),
        "last3_mean_balanced_accuracy": round(float(np.mean(tail)), 4),
        "minutes": round((time.time() - t_start) / 60, 1),
        "argmax": {"balanced_accuracy": best["metrics"][primary]["balanced_accuracy"]},
        "selective": {"selective_accuracy": best["metrics"][primary]["selective_accuracy_60"]},
        "balanced_accuracy_ci95": best["metrics"][primary]["ci95"],
        "best": best["metrics"], "final": metrics,
        "hyperparameters": {k: getattr(args, k) for k in
                            ("lr", "head_lr", "weight_decay", "label_smoothing", "head_dropout",
                             "ema", "soft_borderline", "batch_size", "seed", "balance",
                             "warmup_epochs", "select", "llrd", "freeze_blocks",
                             "mixup", "soft_between", "epochs")},
        "loss_weights": loss_weights,
        "history": history,
    }
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n=== {args.name} ({report['minutes']} min) ===")
    for t in tasks:
        f = metrics[t]
        print(f"  {t:13s} final ba {f['balanced_accuracy']:.3f} {f['ci95']}  "
              f"tuned {f.get('balanced_accuracy_tuned', float('nan')):.3f}  "
              f"selective@60% {f['selective_accuracy_60']:.3f}  "
              f"majority {f['majority_accuracy']:.3f}  recall {f['per_class_recall']}")
    print(f"  {primary} last-3 mean {report['last3_mean_balanced_accuracy']:.3f}  "
          f"best {best['ba']:.3f} @ epoch {best.get('epoch')}")
    print(f"  -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
