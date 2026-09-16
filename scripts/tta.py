"""Test-time augmentation: re-score saved models on original + horizontally flipped clips.

    python scripts/tta.py --runs mv10_multitask mv14_long50

Needs <run>/final.pt and metrics.json. Writes <run>_tta/{valid,test}_scores.npz in
the same format as training, so scripts/ensemble.py can mix raw and TTA runs.
No metrics are computed here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features import extract as fx
from src.models.data import (IGNORE, LEGACY_CLASSES, TASKS, MultiViewDataset, normalise_on_device,
                             to_current_classes)
from src.models.train import RUNS_DIR
from src.models.train_mt import MultiTaskVideoMAE, action_scores, loader


@torch.no_grad()
def predict_flip(model, dl, device, tasks) -> dict[str, np.ndarray]:
    model.eval()
    n = len(dl.dataset)
    probs = {t: np.zeros((n, model.heads[t].out_features), dtype=np.float32) for t in tasks}
    for batch in dl:
        x = normalise_on_device(batch["pixel_values"].to(device, non_blocking=True))
        idx = batch["index"].numpy()
        acc = {t: 0.0 for t in tasks}
        for view in (x, torch.flip(x, dims=[-1])):
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                out = model(view)
            for t in tasks:
                acc[t] = acc[t] + torch.softmax(out[t].float(), dim=-1)
        for t in tasks:
            probs[t][idx] = (acc[t] / 2).cpu().numpy()
    return probs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--splits", nargs="+", default=["valid", "test"])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=8)
    args = ap.parse_args()
    device = fx.resolve_device()

    for run in args.runs:
        src = RUNS_DIR / run
        if not (src / "final.pt").exists():
            print(f"{run}: no final.pt, skipped")
            continue
        m = json.loads((src / "metrics.json").read_text(encoding="utf-8"))
        hp = m.get("hyperparameters", {})
        tasks = m["tasks"]
        model = MultiTaskVideoMAE(m["backbone"], tasks, m.get("freeze_blocks", 9),
                                  hp.get("head_dropout", 0.3))
        state = torch.load(src / "final.pt", map_location="cpu")
        legacy_heads = []
        for t, legacy in LEGACY_CLASSES.items():
            w = state.get(f"heads.{t}.weight")
            if w is not None and w.shape[0] == len(legacy) != len(TASKS[t]):
                # Head trained on the 8 annotator classes: load it as-is and sum its
                # probabilities into families after the softmax.
                model.heads[t] = torch.nn.Linear(w.shape[1], len(legacy))
                legacy_heads.append(t)
        model.load_state_dict(state)
        model.to(device)
        out = RUNS_DIR / f"{run}_tta"
        out.mkdir(parents=True, exist_ok=True)
        for split in args.splits:
            # Same row filter as training wrote, so action ids line up in the ensemble.
            ds = MultiViewDataset(split, tasks, m.get("cache_tag", "mv"), None, m.get("views", "all"),
                                  require_label=(split == "valid"))
            probs = predict_flip(model, loader(ds, args.batch_size, args.num_workers, False), device, tasks)
            for t in legacy_heads:
                probs[t], _ = to_current_classes(t, probs[t], np.full(len(probs[t]), IGNORE))
            np.savez(out / f"{split}_scores.npz", **action_scores(ds, probs))
        print(f"{run}: TTA scores -> {out}")
        del model
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
