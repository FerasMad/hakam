"""Unattended queue: everything after the 50-epoch runs, saved to Drive after every step.

    python scripts/overnight.py            # on Colab, after mv14_long50 / mv15_llrd50 start
    python scripts/overnight.py --smoke    # local dry run on the smoke cache

Steps (a failure is recorded and the queue moves on):
  wait     until the running train_mt processes finish
  tta      flip test-time augmentation for every saved model
  mixup    the better long recipe + mixup + Between as 0.5 offence, 50 epochs
  select   per task, the ensemble (raw or TTA, which runs) with the best VALID score
  final    score test once with that selection -> runs/final_v2, contracts_v2
  refit    winning single recipe retrained on train+valid, stopped at its best epoch;
           reported separately, never used to pick anything

Everything is decided on validation. artifacts/runs/overnight_report.json says what
ran, what was chosen and why.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ensemble  # noqa: E402
from src.models.data import TASKS  # noqa: E402
from src.models.train_mt import task_metrics  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "artifacts" / "runs"
DRIVE = Path("/content/drive/MyDrive/hakam_colab")
TASK_NAMES = ["card", "offence", "action_class", "body_part"]

cfg: dict = {}
report: dict = {"steps": {}}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd: list[str]) -> int:
    log("$ " + " ".join(cmd))
    return subprocess.run([sys.executable, "-u", *cmd], cwd=ROOT).returncode


def sync() -> None:
    """Copy new or changed files to Drive; skips identical ones so repeated saves stay cheap."""
    if cfg["smoke"] or not DRIVE.exists():
        return
    for src_root, name in ((RUNS, "runs"), (ROOT / "artifacts" / cfg["contracts"], cfg["contracts"]),
                           (ROOT / "logs", "logs")):
        if not src_root.exists():
            continue
        for f in src_root.rglob("*"):
            if f.is_file():
                dst = DRIVE / name / f.relative_to(src_root)
                if not dst.exists() or dst.stat().st_size != f.stat().st_size:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, dst)


def save_report() -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / "overnight_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")


def step(name: str, fn) -> None:
    t0 = time.time()
    log(f"=== step {name}")
    try:
        result = fn()
        report["steps"][name] = {"status": "ok", "minutes": round((time.time() - t0) / 60, 1),
                                 "result": result}
    except Exception as exc:
        traceback.print_exc()
        report["steps"][name] = {"status": f"failed: {exc!r}", "minutes": round((time.time() - t0) / 60, 1)}
    save_report()
    try:
        sync()
    except Exception as exc:
        log(f"drive sync failed: {exc!r}")


def has(run: str, split: str = "valid") -> bool:
    return (RUNS / run / f"{split}_scores.npz").exists()


def metrics_of(run: str) -> dict:
    return json.loads((RUNS / run / "metrics.json").read_text(encoding="utf-8"))


def valid_score(runs: list[str], task: str) -> float:
    runs = [r for r in runs if has(r) and has(r, "test")]
    if not runs:
        return -1.0
    v = ensemble.load_mean(runs, "valid")
    if f"prob_{task}" not in v:
        return -1.0
    m = task_metrics(task, v[f"label_{task}"], v[f"prob_{task}"], v["matches"])
    return float(m.get("balanced_accuracy_tuned", m["balanced_accuracy"]))


def recipe(run: str) -> list[str]:
    m = metrics_of(run)
    hp = m["hyperparameters"]
    args = ["--tasks", *m["tasks"], "--backbone", m["backbone"], "--cache-tag", m.get("cache_tag", "mv"),
            "--epochs", str(hp.get("epochs", m["epochs"])), "--batch-size", str(hp["batch_size"]),
            "--lr", str(hp["lr"]), "--head-lr", str(hp["head_lr"]),
            "--weight-decay", str(hp["weight_decay"]), "--label-smoothing", str(hp["label_smoothing"]),
            "--head-dropout", str(hp["head_dropout"]), "--ema", str(hp["ema"]),
            "--freeze-blocks", str(m["freeze_blocks"]), "--warmup-epochs", str(hp.get("warmup_epochs", 1)),
            "--llrd", str(hp.get("llrd", 0.0))]
    if hp.get("soft_borderline"):
        args.append("--soft-borderline")
    if hp.get("mixup"):
        args += ["--mixup", str(hp["mixup"])]
    if hp.get("soft_between"):
        args.append("--soft-between")
    return args + cfg["overrides"]


# --------------------------------------------------------------------------

def wait() -> str:
    if cfg["smoke"]:
        return "skipped (smoke)"
    while subprocess.run(["pgrep", "-f", "models.trai[n]_mt"], capture_output=True).returncode == 0:
        time.sleep(60)
    return "training processes finished"


def tta() -> list[str]:
    runs = [r for r in cfg["base"] + cfg["attr"] + cfg["long"]
            if (RUNS / r / "final.pt").exists() and not has(f"{r}_tta", "test")]
    if runs:
        run(["scripts/tta.py", "--runs", *runs, "--num-workers", cfg["workers"]])
    return runs


def mixup() -> dict:
    longs = [r for r in cfg["long"] if has(r)]
    if not longs:
        raise RuntimeError("no long run finished; nothing to base the extra run on")
    base = max(longs, key=lambda r: valid_score([r], "card"))
    code = run(["-m", "src.models.train_mt", *recipe(base), *cfg["extra_args"],
                "--select", "best", "--predict-splits", "test", "--save", "--name", cfg["mixup"]])
    if code == 0 and (RUNS / cfg["mixup"] / "final.pt").exists():
        run(["scripts/tta.py", "--runs", cfg["mixup"], "--num-workers", cfg["workers"]])
    return {"based_on": base, "exit": code}


def select() -> dict:
    base, attr, longs, mix = cfg["base"], cfg["attr"], cfg["long"], [cfg["mixup"]]
    pools = {"base": base, "base+long": base + longs, "long": longs, "long+mix": longs + mix,
             "all": base + longs + mix}
    choice = {}
    for task in TASK_NAMES:
        candidates = dict(pools)
        if task in ("action_class", "body_part", "offence"):
            candidates.update({f"{k}+attr": v + attr for k, v in pools.items()})
        scored = {}
        for name, runs in candidates.items():
            for suffix in ("", "_tta"):
                names = [r + suffix for r in runs if has(r + suffix) and has(r + suffix, "test")]
                if names and f"{name}{suffix}" not in scored:
                    scored[f"{name}{suffix}"] = (valid_score(names, task), names)
        scored = {k: v for k, v in scored.items() if v[0] >= 0}
        if not scored:
            continue
        best = max(scored, key=lambda k: scored[k][0])
        choice[task] = {"pool": best, "valid": round(scored[best][0], 4), "runs": scored[best][1],
                        "all": {k: round(v[0], 4) for k, v in scored.items()}}
        log(f"{task}: {best} valid {scored[best][0]:.3f}")
    report["selection"] = choice
    return {t: (c["pool"], c["valid"]) for t, c in choice.items()}


def final() -> dict:
    choice = report.get("selection")
    if not choice or "card" not in choice:
        raise RuntimeError("selection missing")
    task_runs = [f"{t}={'+'.join(c['runs'])}" for t, c in choice.items()]
    code = run(["scripts/ensemble.py", "--runs", *choice["card"]["runs"], "--task-runs", *task_runs,
                "--name", cfg["final"]])
    if code != 0:
        raise RuntimeError(f"ensemble exit {code}")
    run(["scripts/make_contracts.py", "--run", cfg["final"], "--out", cfg["contracts"]])
    m = metrics_of(cfg["final"])
    return {t: m["test"][t]["balanced_accuracy"] for t in m["test"]}


def refit() -> dict:
    singles = [r for r in cfg["base"] + cfg["long"] + [cfg["mixup"]] if has(r) and (RUNS / r / "metrics.json").exists()]
    winner = max(singles, key=lambda r: valid_score([r], "card"))
    m = metrics_of(winner)
    hp = m["hyperparameters"]
    stop = m.get("best_epoch") if hp.get("select") == "best" else m["epochs"]
    name = cfg["refit"]
    code = run(["-m", "src.models.train_mt", *recipe(winner), "--train-splits", "train", "valid",
                "--no-eval", "--stop-epoch", str(stop), "--predict-splits", "test", "--save", "--name", name])
    if code != 0 or not has(name, "test"):
        raise RuntimeError(f"refit exit {code}")
    final_metrics = RUNS / cfg["final"] / "metrics.json"
    thresholds = json.loads(final_metrics.read_text())["thresholds"] if final_metrics.exists() else {}
    t = dict(np.load(RUNS / name / "test_scores.npz", allow_pickle=True))
    out = {"from": winner, "stop_epoch": stop, "note": "trained on valid too; thresholds from final_v2"}
    for task in [k[len("prob_"):] for k in t if k.startswith("prob_")]:
        thr = thresholds.get(task) if len(TASKS[task]) == 2 else None
        out[task] = task_metrics(task, t[f"label_{task}"], t[f"prob_{task}"], t["matches"],
                                 threshold=thr)["balanced_accuracy"]
    report["refit"] = out
    return out


def main() -> int:
    import shlex

    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument("--base", nargs="*", default=["mv10_multitask", "mv12_s7", "mv12_s13"])
    ap.add_argument("--attr", nargs="*", default=["mv13_attr"])
    ap.add_argument("--long", nargs="*", default=["mv14_long50", "mv15_llrd50"])
    ap.add_argument("--extra-name", default="mv16_mixup50")
    ap.add_argument("--extra-args", default="--mixup 0.4 --soft-between",
                    help="flags added to the best long recipe for the extra run (last value wins)")
    ap.add_argument("--final-name", default="final_v2")
    ap.add_argument("--contracts", default="contracts_v2")
    ap.add_argument("--refit-name", default="refit_trainvalid")
    args = ap.parse_args()

    if args.smoke:
        cfg.update(smoke=True, base=["smoke_mv"], attr=["smoke_attr"], long=["smoke_long"],
                   mixup="smoke_mix", final="smoke_final_v2", contracts="smoke_contracts_v2",
                   refit="smoke_refit", workers="0", extra_args=shlex.split(args.extra_args),
                   overrides=["--epochs", "1", "--batch-size", "4", "--num-workers", "0"])
    else:
        cfg.update(smoke=False, base=args.base, attr=args.attr, long=args.long, mixup=args.extra_name,
                   final=args.final_name, contracts=args.contracts, refit=args.refit_name,
                   workers="10", extra_args=shlex.split(args.extra_args), overrides=["--num-workers", "10"])

    for name, fn in (("wait", wait), ("tta", tta), ("mixup", mixup), ("select", select),
                     ("final", final), ("refit", refit)):
        if name not in args.skip:
            step(name, fn)
    if not cfg["smoke"]:
        run(["scripts/summarise_runs.py"])
        sync()
    log("overnight queue done")
    print(json.dumps(report["steps"], indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
