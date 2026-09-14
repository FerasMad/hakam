"""Average seeds, freeze thresholds on valid, then score the test split once.

    python scripts/ensemble.py --runs mv10 mv12_s2 mv12_s3 --name final

Binary thresholds are chosen on valid (max balanced accuracy) before test is read,
and applied unchanged. Writes artifacts/runs/<name>/metrics.json and
test_predictions.csv - one row per test action, the source for real contracts.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.data import IGNORE, TASKS
from src.models.train import RUNS_DIR
from src.models.train_mt import best_threshold, task_metrics


def load_mean(runs: list[str], split: str, task_runs: dict[str, list[str]] | None = None) -> dict:
    """Average each task's probabilities over its runs (``task_runs`` overrides ``runs``)."""
    task_runs = task_runs or {}
    names = list(dict.fromkeys(runs + [r for rs in task_runs.values() for r in rs]))
    loaded = {r: dict(np.load(RUNS_DIR / r / f"{split}_scores.npz", allow_pickle=True)) for r in names}
    ids = loaded[names[0]]["action_ids"]
    for r, s in loaded.items():
        if not np.array_equal(s["action_ids"], ids):
            raise ValueError(f"{r}: {split} actions differ from {names[0]}")
    out = {"action_ids": ids, "matches": loaded[names[0]]["matches"]}
    tasks = {k[len("prob_"):] for s in loaded.values() for k in s if k.startswith("prob_")}
    for t in sorted(tasks, key=list(TASKS).index):
        sources = [r for r in task_runs.get(t, runs) if f"prob_{t}" in loaded[r]]
        if not sources:
            continue
        out[f"prob_{t}"] = np.mean([loaded[r][f"prob_{t}"] for r in sources], axis=0)
        out[f"label_{t}"] = loaded[sources[0]][f"label_{t}"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--task-runs", nargs="*", default=[], metavar="TASK=RUN+RUN",
                    help="use these runs for one task, e.g. offence=mv13_attr")
    ap.add_argument("--name", default="final")
    ap.add_argument("--no-test", action="store_true", help="valid only")
    args = ap.parse_args()
    task_runs = {k: v.split("+") for k, v in (kv.split("=") for kv in args.task_runs)}

    valid = load_mean(args.runs, "valid", task_runs)
    tasks = [k[len("prob_"):] for k in valid if k.startswith("prob_")]
    report = {"runs": args.runs, "task_runs": task_runs, "tasks": tasks,
              "valid": {}, "test": {}, "thresholds": {}}

    print(f"ensemble of {len(args.runs)} runs: {args.runs}\n\nVALID")
    for t in tasks:
        y, p = valid[f"label_{t}"], valid[f"prob_{t}"]
        m = task_metrics(t, y, p, valid["matches"])
        report["valid"][t] = m
        if len(TASKS[t]) == 2:
            keep = y != IGNORE
            report["thresholds"][t] = best_threshold(y[keep], p[keep, 1])
        print(f"  {t:13s} ba {m['balanced_accuracy']:.3f} {m['ci95']}  "
              f"tuned {m.get('balanced_accuracy_tuned', float('nan')):.3f} "
              f"@ {report['thresholds'].get(t)}  recall {m['per_class_recall']}")

    out = RUNS_DIR / args.name
    out.mkdir(parents=True, exist_ok=True)

    if not args.no_test:
        test = load_mean(args.runs, "test", task_runs)
        report["test_scored_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        print("\nTEST (thresholds frozen from valid)")
        rows = {"action_id": test["action_ids"]}
        for t in tasks:
            y, p = test[f"label_{t}"], test[f"prob_{t}"]
            thr = report["thresholds"].get(t)
            m = task_metrics(t, y, p, test["matches"], threshold=thr)
            report["test"][t] = m
            print(f"  {t:13s} ba {m['balanced_accuracy']:.3f} {m['ci95']}  "
                  f"acc {m['accuracy']:.3f} (majority {m['majority_accuracy']:.3f})  "
                  f"selective@60% {m['selective_accuracy_60']:.3f}  recall {m['per_class_recall']}")
            pred = (p[:, 1] >= thr).astype(int) if thr is not None else p.argmax(1)
            rows[f"{t}_pred"] = [TASKS[t][i] for i in pred]
            rows[f"{t}_confidence"] = np.round(p[np.arange(len(pred)), pred], 4)
            rows[f"{t}_label"] = [TASKS[t][i] if i != IGNORE else "" for i in y]
        pd.DataFrame(rows).to_csv(out / "test_predictions.csv", index=False)

    (out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
