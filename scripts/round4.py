"""Round 4: all views, multi-task heads, EMA, seed ensemble, test once.

    python scripts/round4.py all --save /content/drive/MyDrive/hakam_colab/runs
    python scripts/round4.py mv10            # a single step

cache  every view of every action at 224x224
mv9    card only, all views, averaged per action   (is data the bottleneck?)
mv10   all tasks + soft borderline + EMA            (does shared supervision help?)
mv11   mv10 on the SSv2 checkpoint                  (motion vs appearance features)
mv12   best of mv10/mv11 with two more seeds
final  average the three seeds, freeze thresholds on valid, score test once
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "artifacts" / "runs"
EXP7_LAST3 = 0.578          # best round 3 card run, VideoMAE-base on the last clip
ALL_TASKS = ["card", "offence", "action_class", "body_part"]
SEEDS = [7, 13]


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run([sys.executable, "-u", *cmd], cwd=ROOT, check=True)


def last3(name: str) -> float:
    path = RUNS / name / "metrics.json"
    if not path.exists():
        return -1.0
    return float(json.loads(path.read_text(encoding="utf-8"))["last3_mean_balanced_accuracy"])


def views() -> str:
    choice = "all" if last3("mv9_card") >= EXP7_LAST3 or last3("mv9_card") < 0 else "last"
    print(f"views: {choice}  (mv9 {last3('mv9_card'):.3f} vs exp7 {EXP7_LAST3})")
    return choice


def winner() -> tuple[str, str]:
    name = max(["mv10_multitask", "mv11_ssv2"], key=last3)
    backbone = "videomae_base_ssv2" if name == "mv11_ssv2" else "videomae_base"
    print(f"winner: {name}  (mv10 {last3('mv10_multitask'):.3f}, mv11 {last3('mv11_ssv2'):.3f})")
    return name, backbone


def train(name: str, extra: list[str]) -> None:
    run(["-m", "src.models.train_mt", "--epochs", "5", "--name", name, *extra])


def multitask(backbone: str, seed: int | None = None) -> list[str]:
    args = ["--tasks", *ALL_TASKS, "--backbone", backbone, "--soft-borderline",
            "--ema", "0.999", "--views", views(), "--predict-splits", "test", "--save"]
    return args + (["--seed", str(seed)] if seed is not None else [])


def cache() -> None:
    run(["scripts/cache_frames.py", "--splits", "train", "valid", "test",
         "--all-views", "--size", "224", "--tag", "mv"])


def mv9() -> None:
    train("mv9_card", ["--tasks", "card"])


def mv10() -> None:
    train("mv10_multitask", multitask("videomae_base"))


def mv11() -> None:
    train("mv11_ssv2", multitask("videomae_base_ssv2"))


def mv12() -> None:
    _, backbone = winner()
    for seed in SEEDS:
        train(f"mv12_s{seed}", multitask(backbone, seed))


def final() -> None:
    name, _ = winner()
    run(["scripts/ensemble.py", "--runs", name, *[f"mv12_s{s}" for s in SEEDS], "--name", "final"])


STEPS = {"cache": cache, "mv9": mv9, "mv10": mv10, "mv11": mv11, "mv12": mv12, "final": final}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=[*STEPS, "all"])
    ap.add_argument("--save", default=None, help="copy runs here when done")
    args = ap.parse_args()

    for name in STEPS if args.step == "all" else [args.step]:
        STEPS[name]()

    run(["scripts/summarise_runs.py"] + (["--save", args.save] if args.save else []))
    return 0


if __name__ == "__main__":
    sys.exit(main())
