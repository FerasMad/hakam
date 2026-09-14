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

    rows, histories = [], {}
    for path in sorted(RUNS.glob("*/metrics.json")):
        m = json.loads(path.read_text(encoding="utf-8"))
        if m["run"].startswith("smoke"):
            continue
        rows.append({
            "run": m["run"],
            "balanced_acc": m["argmax"]["balanced_accuracy"],
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

    if args.save:
        dest = Path(args.save)
        shutil.copytree(RUNS, dest, dirs_exist_ok=True)
        print(f"\nsaved to {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
