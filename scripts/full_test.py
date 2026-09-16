"""Full test of the final model, retrieval and Arabic ruling on the SoccerNet-MVFoul test split.

    python scripts/full_test.py --weights-dir weights --out artifacts/full_test

For each of the 301 test incidents it runs the final vision model twice:
- **all views**  - every camera angle, averaged (how the reported results were measured);
- **last clip**  - the close-up replay only (what the website does with one upload).

Then it scores the vision heads against the referee's labels, builds the offline
Arabic ruling for every answered decision, checks retrieval covers the Law 12
articles each ruling rests on, and picks ten demo videos.

Writes predictions.csv, metrics.json, demo_picks.csv and report.md to --out.
Needs weights/final.pt + thresholds.json, the Test clips in data/mvfouls/Test and
the preprocessing manifests in artifacts/preprocessing/private.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.inference.contract_builder import CLASSES, from_probabilities
from src.inference.predict import load_predictor, read_clip
from src.models.data import IGNORE, _task_label
from src.models.train import balanced_accuracy
from src.models.train_mt import clustered_ci

TASKS = ["offence", "card", "body_part", "action_class"]
CLIPS_DIR = config.DATA_ROOT / "mvfouls" / "Test"
MANIFEST = config.ARTIFACTS / "preprocessing" / "private" / "actions_test.csv"


# --------------------------------------------------------------------------
# Vision
# --------------------------------------------------------------------------

def clip_paths(action_id: str) -> list[Path]:
    folder = CLIPS_DIR / f"action_{action_id}"
    return sorted(folder.glob("clip_*.mp4"), key=lambda p: int(p.stem.split("_")[1]))


def run_model(predictor, manifest: pd.DataFrame, limit: int = 0) -> pd.DataFrame:
    rows = []
    ids = list(manifest.index)[: limit or None]
    start = time.time()
    for n, action_id in enumerate(ids, 1):
        paths = clip_paths(action_id)
        if not paths:
            continue
        rec = manifest.loc[action_id]
        clips = [read_clip(p) for p in paths]
        t0 = time.time()
        probs_all = predictor.probabilities(clips)
        seconds_all = time.time() - t0
        t0 = time.time()
        probs_last = predictor.probabilities([clips[-1]])
        seconds_last = time.time() - t0

        row = {"action_id": action_id, "match": str(rec.get("source_match", "")).replace("\\", "/"),
               "views": len(paths), "last_clip": str(paths[-1].relative_to(config.PROJECT_ROOT)).replace("\\", "/"),
               "raw_action": rec.get("Action class", ""), "raw_severity": rec.get("Severity", ""),
               "seconds_all": round(seconds_all, 3), "seconds_last": round(seconds_last, 3)}
        for task in TASKS:
            y = _task_label(rec, task)
            row[f"true_{task}"] = CLASSES[task][y] if y != IGNORE else ""
        for mode, probs in (("all", probs_all), ("last", probs_last)):
            contract = from_probabilities(action_id, probs, predictor.thresholds,
                                          predictor.model_version, len(paths) if mode == "all" else 1)
            row[f"{mode}_abstain"] = contract.should_abstain()
            row[f"{mode}_contract"] = contract.to_json(indent=None)
            for task in TASKS:
                p = probs[task]
                row[f"{mode}_p_{task}"] = json.dumps([round(float(v), 4) for v in p])
                if len(CLASSES[task]) == 2:
                    idx = int(p[1] >= predictor.thresholds.get(task, 0.5))
                else:
                    idx = int(np.argmax(p))
                row[f"{mode}_pred_{task}"] = CLASSES[task][idx]
                row[f"{mode}_conf_{task}"] = round(float(p[idx]), 4)
        rows.append(row)
        if n % 25 == 0 or n == len(ids):
            rate = n / (time.time() - start)
            print(f"  {n}/{len(ids)} incidents  {rate:.2f}/s", flush=True)
    return pd.DataFrame(rows)


def vision_metrics(df: pd.DataFrame, mode: str) -> dict:
    out = {}
    for task in TASKS:
        sub = df[df[f"true_{task}"] != ""]
        classes = CLASSES[task]
        y = sub[f"true_{task}"].map(classes.index).to_numpy()
        pred = sub[f"{mode}_pred_{task}"].map(classes.index).to_numpy()
        matches = sub["match"].to_numpy()
        cm = np.zeros((len(classes), len(classes)), dtype=int)
        for t, p in zip(y, pred):
            cm[t, p] += 1
        out[task] = {
            "n": int(len(y)),
            "balanced_accuracy": round(balanced_accuracy(y, pred), 4) if len(y) else None,
            "ci95": clustered_ci(y, pred, matches) if len(np.unique(y)) > 1 else None,
            "accuracy": round(float((y == pred).mean()), 4) if len(y) else None,
            "majority_accuracy": round(float(np.bincount(y).max() / len(y)), 4) if len(y) else None,
            "per_class_recall": {c: round(float((pred[y == i] == i).mean()), 4)
                                 for i, c in enumerate(classes) if (y == i).any()},
            "confusion": {"classes": classes, "matrix": cm.tolist()},
        }
    answered = df[~df[f"{mode}_abstain"]]
    off = answered[answered["true_offence"] != ""]
    out["abstain_rate"] = round(float(df[f"{mode}_abstain"].mean()), 4)
    out["answered_offence_accuracy"] = (round(float((off["true_offence"] == off[f"{mode}_pred_offence"]).mean()), 4)
                                        if len(off) else None)
    card = answered[(answered["true_card"] != "") & (answered[f"{mode}_pred_offence"] == "offence")]
    out["answered_card_balanced_accuracy"] = (round(balanced_accuracy(
        card["true_card"].map(CLASSES["card"].index).to_numpy(),
        card[f"{mode}_pred_card"].map(CLASSES["card"].index).to_numpy()), 4)
        if card["true_card"].nunique() > 1 else None)
    seconds = df[f"seconds_{mode}"] / (df["views"] if mode == "all" else 1)
    out["seconds_per_view"] = round(float(seconds.mean()), 3)
    return out


def card_errors_by_action(df: pd.DataFrame, mode: str) -> dict:
    sub = df[df["true_card"] != ""]
    return {str(a): {"n": int(len(g)),
                     "card_accuracy": round(float((g["true_card"] == g[f"{mode}_pred_card"]).mean()), 3)}
            for a, g in sub.groupby("raw_action")}


# --------------------------------------------------------------------------
# Retrieval and ruling (offline, no API)
# --------------------------------------------------------------------------

def rag_and_ruling(df: pd.DataFrame, mode: str) -> dict:
    from src.contract import HakamContract
    from src.llm.faithfulness import score
    from src.llm.generate import safe_v3
    from src.llm.retrieve import retrieve
    from src.llm.ruling import build_ruling

    needed = {"decision", "restart", "disciplinary", "law", "why", "confidence"}
    totals = {"answered": 0, "six": 0, "clean": 0, "cites": 0, "hedged": 0, "covered": 0}
    overlap, failures = [], []
    for _, row in df[~df[f"{mode}_abstain"]].iterrows():
        contract = HakamContract.from_dict(json.loads(row[f"{mode}_contract"]))
        text, sections, articles = safe_v3(contract)
        check = score(text, contract, articles)
        totals["answered"] += 1
        totals["six"] += needed <= set(sections)
        totals["clean"] += not check["unsupported"]
        totals["cites"] += bool(check["cites_article"])
        totals["hedged"] += bool(check["hedged_low_conf"])
        totals["covered"] += set(build_ruling(contract).article_ids) <= {a["id"] for a in articles}
        if check["unsupported"]:
            failures.append({"action_id": row["action_id"], "unsupported": check["unsupported"]})
        hybrid = [a["id"] for a in retrieve(contract, k=3)]
        os.environ["HAKAM_RETRIEVAL_MODE"] = "tags"
        tags = [a["id"] for a in retrieve(contract, k=3)]
        os.environ["HAKAM_RETRIEVAL_MODE"] = "hybrid"
        overlap.append(len(set(hybrid) & set(tags)) / 3)
    n = max(totals["answered"], 1)
    return {
        "answered": totals["answered"],
        "six_sections_rate": round(totals["six"] / n, 4),
        "no_unsupported_rate": round(totals["clean"] / n, 4),
        "cites_law_rate": round(totals["cites"] / n, 4),
        "hedging_rate": round(totals["hedged"] / n, 4),
        "ruling_articles_retrieved_rate": round(totals["covered"] / n, 4),
        "hybrid_vs_tags_top3_overlap": round(float(np.mean(overlap or [0])), 4),
        "unsupported_examples": failures[:10],
    }


# --------------------------------------------------------------------------
# Demo picks (website mode: one clip, the close-up)
# --------------------------------------------------------------------------

def pick_demo(df: pd.DataFrame, mode: str = "last") -> pd.DataFrame:
    d = df.copy()
    d["answered"] = ~d[f"{mode}_abstain"]
    d["off_ok"] = d["true_offence"] == d[f"{mode}_pred_offence"]
    d["card_ok"] = d["true_card"] == d[f"{mode}_pred_card"]
    d["family"] = d[f"{mode}_pred_action_class"]
    d["strength"] = d[[f"{mode}_conf_offence", f"{mode}_conf_card"]].min(axis=1)
    chosen: list[tuple[str, str]] = []

    def take(frame, reason, count=1, by="strength", distinct_family=False):
        used = {a for a, _ in chosen}
        frame = frame[~frame["action_id"].isin(used)].sort_values(by, ascending=False)
        families = set()
        for _, r in frame.iterrows():
            if sum(1 for _, why in chosen if why == reason) >= count:
                break
            if distinct_family and r["family"] in families:
                continue
            families.add(r["family"])
            chosen.append((r["action_id"], reason))

    ok = d[d["answered"] & d["off_ok"]]
    take(ok[(ok["true_card"] == "card") & ok["card_ok"]], "correct: card", 3, distinct_family=True)
    take(ok[(ok["true_card"] == "no_card") & ok["card_ok"]], "correct: no card", 2)
    take(ok[(ok["true_action_class"] == "hands") & (ok["family"] == "hands")], "correct: hands foul")
    take(ok[ok["true_offence"] == "no_offence"], "correct: no offence", by=f"{mode}_conf_offence")
    take(ok[(ok["raw_severity"].astype(str) == "5.0") & (ok[f"{mode}_pred_card"] == "card")], "correct: red-card incident")
    take(d[~d["answered"]], "refer to human (low confidence)", by=f"{mode}_conf_offence")
    take(d[d["answered"] & (d["true_card"] != "") & ~d["card_ok"]], "honest mistake (card wrong, confident)",
         by=f"{mode}_conf_card")

    picks = []
    for action_id, reason in chosen:
        r = d[d["action_id"] == action_id].iloc[0]
        if r["answered"]:
            model = (f"{r[f'{mode}_pred_offence']} {r[f'{mode}_conf_offence']:.0%} / {r[f'{mode}_pred_card']} "
                     f"{r[f'{mode}_conf_card']:.0%} / {r['family']} / {r[f'{mode}_pred_body_part']}")
        else:
            model = f"refer to human (offence {r[f'{mode}_pred_offence']} {r[f'{mode}_conf_offence']:.0%})"
        picks.append({"action_id": action_id, "why": reason, "clip": r["last_clip"], "views": int(r["views"]),
                      "referee": f"{r['true_offence'] or '?'} / {r['true_card'] or '-'} / {r['raw_action']} / "
                                 f"severity {r['raw_severity']}",
                      "model": model})
    return pd.DataFrame(picks)


# --------------------------------------------------------------------------

def report_md(metrics: dict, picks: pd.DataFrame) -> str:
    lines = ["# Hakam full test", "",
             f"Model {metrics['model_version']} on {metrics['device']} · thresholds {metrics['thresholds']} · "
             f"{metrics['incidents']} test incidents", ""]
    for mode, title in (("all", "All camera views (reported protocol)"), ("last", "Last clip only (website)")):
        m = metrics["vision"][mode]
        lines += [f"## Vision — {title}", "", "| Task | BA | 95% CI | Acc | Majority | Recall per class |",
                  "|---|---|---|---|---|---|"]
        for t in TASKS:
            x = m[t]
            lines.append(f"| {t} | {x['balanced_accuracy']} | {x['ci95']} | {x['accuracy']} | "
                         f"{x['majority_accuracy']} | {x['per_class_recall']} |")
        lines += ["", f"Abstain rate {m['abstain_rate']:.1%} · offence accuracy when answered "
                      f"{m['answered_offence_accuracy']} · card BA when answered "
                      f"{m['answered_card_balanced_accuracy']} · {m['seconds_per_view']} s per view", ""]
    for key, title in (("ruling_offline_all_views", "all views"), ("ruling_offline_last_clip", "last clip")):
        r = metrics[key]
        lines += [f"## Retrieval and Arabic ruling (offline, {title})", "",
                  f"- Answered decisions: {r['answered']}",
                  f"- Six sections present: {r['six_sections_rate']:.1%}",
                  f"- No unsupported claims: {r['no_unsupported_rate']:.1%}",
                  f"- Cites the Law: {r['cites_law_rate']:.1%} · hedging correct: {r['hedging_rate']:.1%}",
                  f"- Ruling's Law 12 articles all retrieved: {r['ruling_articles_retrieved_rate']:.1%}",
                  f"- Hybrid vs tag-only top-3 overlap: {r['hybrid_vs_tags_top3_overlap']:.1%}", ""]
    lines += ["## Ten demo videos (website mode: last clip)", ""]
    if len(picks):
        lines += ["| # | Action | Why | Clip | Referee | Model |", "|---|---|---|---|---|---|"]
        for i, p in enumerate(picks.to_dict("records"), 1):
            lines.append(f"| {i} | {p['action_id']} | {p['why']} | `{p['clip']}` | {p['referee']} | {p['model']} |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights-dir", default=str(config.PROJECT_ROOT / "weights"))
    ap.add_argument("--out", default=str(config.ARTIFACTS / "full_test"))
    ap.add_argument("--limit", type=int, default=0, help="first N incidents only (smoke run)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(MANIFEST, dtype={"action_id": str}).set_index("action_id")
    predictor = load_predictor(args.weights_dir)
    print(f"model {predictor.model_version} on {predictor.device}, thresholds {predictor.thresholds}", flush=True)

    df = run_model(predictor, manifest, args.limit)
    df.to_csv(out / "predictions.csv", index=False)

    metrics = {"model_version": predictor.model_version, "device": str(predictor.device),
               "thresholds": predictor.thresholds, "incidents": int(len(df)),
               "vision": {m: vision_metrics(df, m) for m in ("all", "last")},
               "card_accuracy_by_action": {m: card_errors_by_action(df, m) for m in ("all", "last")}}
    print("vision scored; building offline rulings ...", flush=True)
    metrics["ruling_offline_all_views"] = rag_and_ruling(df, "all")
    metrics["ruling_offline_last_clip"] = rag_and_ruling(df, "last")
    picks = pick_demo(df, "last")
    picks.to_csv(out / "demo_picks.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    report = report_md(metrics, picks)
    (out / "report.md").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
