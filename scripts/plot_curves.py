"""Learning curves for every run, saved as PNGs for the report.

    python scripts/plot_curves.py

One figure per run (train vs valid loss, valid balanced accuracy per task) and
curves_all.png comparing card runs, in artifacts/runs/figures/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.train import RUNS_DIR


def main() -> int:
    figures = RUNS_DIR / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    runs = []
    for path in sorted(RUNS_DIR.glob("*/metrics.json")):
        m = json.loads(path.read_text(encoding="utf-8"))
        if m.get("history") and not m["run"].startswith("smoke"):
            runs.append(m)

    for m in runs:
        h = m["history"]
        ep = [e["epoch"] for e in h]
        fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6))
        a.plot(ep, [e["train_loss"] for e in h], "-o", label="train loss")
        a.plot(ep, [e["valid_loss"] for e in h], "--s", label="valid loss")
        a.set_xlabel("epoch"); a.set_title("loss"); a.legend(); a.grid(alpha=.3)
        task_keys = [k for k in h[0] if k.endswith("_ba")] or ["valid_balanced_accuracy"]
        for k in task_keys:
            b.plot(ep, [e[k] for e in h], "-o", label=k.replace("_ba", ""))
        b.axhline(0.5, color="grey", lw=.8, ls=":")
        b.axhline(0.6, color="green", lw=.8, ls=":")
        b.set_xlabel("epoch"); b.set_title("valid balanced accuracy"); b.legend(); b.grid(alpha=.3)
        fig.suptitle(f"{m['run']}  ({m.get('backbone', '')})")
        fig.tight_layout()
        fig.savefig(figures / f"{m['run']}.png", dpi=110)
        plt.close(fig)

    card = [m for m in runs if m.get("stage") == "card"]
    if card:
        fig, ax = plt.subplots(figsize=(8, 4))
        for m in card:
            h = m["history"]
            ax.plot([e["epoch"] for e in h], [e["valid_balanced_accuracy"] for e in h], "-o", ms=3,
                    label=m["run"])
        ax.axhline(0.6, color="green", lw=.8, ls=":")
        ax.set_xlabel("epoch"); ax.set_ylabel("valid balanced accuracy (card)")
        ax.legend(fontsize=7); ax.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(figures / "curves_all.png", dpi=110)
        plt.close(fig)

    print(f"figures for {len(runs)} runs -> {figures}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
