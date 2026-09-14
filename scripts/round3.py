"""Round 3 experiments, one variable at a time, in order.

    python scripts/round3.py all --save /content/drive/MyDrive/hakam_colab/runs
    python scripts/round3.py exp6            # a single step

exp5  Exp 3 + head learning rate 1e-3 and warmup (Exp 3 underfit), two schedules:
      a) backbone lr 1e-5, 8 epochs   b) backbone lr 3e-5, 12 epochs
exp6  best schedule + 1.3 s window centred on the foul, stride 2 (VARS, CAS-FD)
exp7  best schedule and window + VideoMAE-base
exp8  best settings on the offence stage (the contract needs it)

Later steps pick their settings from earlier runs by the mean balanced accuracy
of the last three epochs, not the best epoch, which is optimistic on 299 clips.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "artifacts" / "runs"

DENSE = ["--start", "59", "--end", "91", "--tag", "dense"]
COMMON = [
    "--mode", "finetune", "--geometry", "resize", "--augment", "mild_aug_v1",
    "--freeze-blocks", "9", "--head-lr", "1e-3",
    "--weight-decay", "0.1", "--label-smoothing", "0.1", "--head-dropout", "0.3",
    "--batch-size", "8", "--num-workers", "2", "--save",
]
SCHEDULES = {
    "card_exp5a_head_lr": ["--lr", "1e-5", "--epochs", "8"],
    "card_exp5b_longer": ["--lr", "3e-5", "--epochs", "12"],
}


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run([sys.executable, "-u", *cmd], cwd=ROOT, check=True)


def train(stage: str, name: str, extra: list[str] = ()) -> None:
    run(["-m", "src.models.train", "--stage", stage, *COMMON, *extra, "--name", name])


def last3(run_name: str) -> float:
    path = RUNS / run_name / "metrics.json"
    if not path.exists():
        return -1.0
    value = json.loads(path.read_text(encoding="utf-8")).get("last3_mean_balanced_accuracy")
    return -1.0 if value is None else float(value)


def best_schedule() -> tuple[str, list[str]]:
    name = max(SCHEDULES, key=last3)
    print(f"schedule: {name}  ({', '.join(f'{n} {last3(n):.3f}' for n in SCHEDULES)})")
    return name, SCHEDULES[name]


def best_window() -> list[str]:
    exp5, _ = best_schedule()
    dense = last3("card_exp6_dense") > last3(exp5)
    print(f"window: {'dense' if dense else 'default'}  "
          f"({exp5} {last3(exp5):.3f}, exp6 {last3('card_exp6_dense'):.3f})")
    return ["--cache-tag", "dense"] if dense else []


def best_backbone() -> list[str]:
    small = max(last3(n) for n in [*SCHEDULES, "card_exp6_dense"])
    base = last3("card_exp7_base") > small
    print(f"backbone: {'videomae_base' if base else 'videomae_small'}  "
          f"(small {small:.3f}, base {last3('card_exp7_base'):.3f})")
    return ["--backbone", "videomae_base" if base else "videomae_small"]


def exp5() -> None:
    for name, schedule in SCHEDULES.items():
        train("card", name, schedule)


def exp6() -> None:
    run(["scripts/cache_frames.py", "--splits", "train", "valid", "test", *DENSE])
    train("card", "card_exp6_dense", [*best_schedule()[1], "--cache-tag", "dense"])


def exp7() -> None:
    train("card", "card_exp7_base",
          [*best_schedule()[1], "--backbone", "videomae_base", *best_window()])


def exp8() -> None:
    train("offence", "offence_exp8", [*best_schedule()[1], *best_backbone(), *best_window()])


STEPS = {"exp5": exp5, "exp6": exp6, "exp7": exp7, "exp8": exp8}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=[*STEPS, "all"])
    ap.add_argument("--save", default=None, help="copy runs here when done")
    args = ap.parse_args()

    for name in STEPS if args.step == "all" else [args.step]:
        STEPS[name]()

    summary = ["scripts/summarise_runs.py"] + (["--save", args.save] if args.save else [])
    run(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
