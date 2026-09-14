"""Turn the ensemble's test predictions into real contracts for the app and the LLM.

    python scripts/make_contracts.py

Reads artifacts/runs/final/test_predictions.csv, writes one HakamContract per test
action to artifacts/contracts/test/<action_id>.json. The cascade is enforced here:
no offence means no card. Ground truth goes to a separate index file and never
into a contract - the language model must not see it.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.contract import HakamContract, Prediction

KEYS = config.PROJECT_ROOT / "frame_cache" / "test_keys_mv.json"


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="final", help="ensemble directory under artifacts/runs")
    ap.add_argument("--out", default="contracts", help="directory under artifacts")
    args = ap.parse_args()
    predictions = config.ARTIFACTS / "runs" / args.run / "test_predictions.csv"
    out = config.ARTIFACTS / args.out / "test"

    df = pd.read_csv(predictions, dtype={"action_id": str}).fillna("")
    views = Counter(str(a) for a, _ in json.loads(KEYS.read_text())) if KEYS.exists() else {}
    out.mkdir(parents=True, exist_ok=True)
    model_version = f"hakam-{args.run}"

    index = []
    for r in df.to_dict("records"):
        offence = Prediction(r["offence_pred"], float(r["offence_confidence"]))
        card = None
        if offence.label == "offence":
            card = Prediction(r["card_pred"], float(r["card_confidence"]))
        attributes = {
            name: Prediction(r[f"{name}_pred"], float(r[f"{name}_confidence"]))
            for name in ("action_class", "body_part") if f"{name}_pred" in r
        }
        contract = HakamContract(
            action_id=r["action_id"], offence=offence, card=card, attributes=attributes,
            model_version=model_version, num_views=views.get(r["action_id"], 0),
        )
        (out / f"{r['action_id']}.json").write_text(contract.to_json(), encoding="utf-8")
        index.append({
            "action_id": r["action_id"], "abstain": contract.should_abstain(),
            "offence": offence.label, "card": card.label if card else None,
            "truth": {k: r.get(f"{k}_label", "") for k in ("offence", "card", "action_class", "body_part")},
        })

    (out.parent / "test_index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    abstain = sum(i["abstain"] for i in index)
    print(f"{len(index)} contracts -> {out}  ({abstain} abstain, {len(index) - abstain} explained)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
