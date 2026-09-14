"""Print the comparison table for every training run, and optionally copy runs out.

    python scripts/summarise_runs.py
    python scripts/summarise_runs.py --save /content/drive/MyDrive/hakam_colab/runs
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config

RUNS = config.ARTIFACTS / "runs"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", default=None, help="copy runs to this directory")
    args = ap.parse_args()

    rows, histories, runs, ensembles = [], {}, [], []
    for path in sorted(RUNS.glob("*/metrics.json")):
        m = json.loads(path.read_text(encoding="utf-8"))
        if "runs" in m:
            ensembles.append((path.parent.name, m))
            continue
        if m.get("run", "smoke").startswith("smoke"):
            continue
        runs.append(m)
        rows.append({
            "run": m["run"],
            "balanced_acc": m["argmax"]["balanced_accuracy"],
            "last3_mean": m.get("last3_mean_balanced_accuracy"),
            "ci95": m.get("balanced_accuracy_ci95"),
            "confident_acc@60%": m["selective"]["selective_accuracy"],
            "best_epoch": m.get("best_epoch"),
            "minutes": m["minutes"],
        })
        if m.get("history"):
            histories[m["run"]] = pd.DataFrame(m["history"])

    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    table.to_csv(RUNS / "summary.csv", index=False)

    for run, history in histories.items():
        print(f"\n{run}")
        print(history.to_string(index=False))

    per_task = []
    for m in runs:
        for t, f in (m.get("final") or {}).items():
            per_task.append({"run": m["run"], "task": t, "final_ba": f["balanced_accuracy"],
                             "tuned_ba": f.get("balanced_accuracy_tuned"), "ci95": f["ci95"],
                             "selective@60%": f["selective_accuracy_60"],
                             "majority_acc": f["majority_accuracy"], "recall": f["per_class_recall"]})
    if per_task:
        print("\nper task (final epoch, valid)")
        print(pd.DataFrame(per_task).to_string(index=False))

    for name, m in ensembles:
        if name.startswith("smoke"):
            continue
        print(f"\nensemble {name}: {m['runs']}")
        table = [{"task": t, "split": split, "ba": f["balanced_accuracy"], "ci95": f["ci95"],
                  "accuracy": f["accuracy"], "majority": f["majority_accuracy"],
                  "selective@60%": f["selective_accuracy_60"], "recall": f["per_class_recall"]}
                 for split in ("valid", "test") for t, f in m.get(split, {}).items()]
        print(pd.DataFrame(table).to_string(index=False))

    try:
        import plot_curves

        plot_curves.main()
    except Exception as exc:  # figures are a convenience, never block the summary
        print(f"figures skipped: {exc}")

    if args.save:
        dest = Path(args.save)
        shutil.copytree(RUNS, dest, dirs_exist_ok=True)
        print(f"\nsaved to {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
