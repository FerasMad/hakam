"""Run the vision model on your own foul clip(s) and print the contract and the Arabic ruling.

    python scripts/try_video.py my_foul.mp4
    python scripts/try_video.py view1.mp4 view2.mp4        # several angles of the same foul

Needs weights/final.pt and weights/thresholds.json (see weights/README.md).
Uses the LLM when OPENAI_API_KEY is set (.env); otherwise prints the ruling built
from the Law 12 rules. Trim the clip so the foul is near the middle - the model
looks at the 2.56 s around the centre of each video.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        print(__doc__)
        return 1
    missing = [p for p in paths if not p.exists()]
    if missing:
        sys.exit(f"file not found: {missing[0]}")

    from src.inference.predict import load_predictor
    from src.llm.faithfulness import score
    from src.llm.generate import explain, safe_v3

    print("loading the model ...", flush=True)
    predictor = load_predictor()
    contract = predictor.predict(paths)
    print(f"\nCONTRACT  ({len(paths)} view(s), {predictor.last_seconds}s on {predictor.device})")
    print(contract.to_json())

    if contract.should_abstain():
        from src.config import ABSTAIN_MESSAGE_AR

        print(f"\nRULING\n{ABSTAIN_MESSAGE_AR}")
        return 0

    engine = "offline (Law 12 rules, no LLM)"
    try:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("no key")
        result = explain(contract)
        text, articles, engine = result.text_ar, result.articles, "live (GPT-5 nano)"
    except Exception:
        text, _, articles = safe_v3(contract)

    print(f"\nRULING  [{engine}]\n{text}")
    print("\nLAWS CITED")
    for a in articles:
        print(f"  - {a['law']} · {a['title_ar']}  [{a['id']}]")
    check = score(text, contract, articles)
    print("\nFAITHFULNESS", json.dumps({k: check[k] for k in ("faithfulness", "unsupported", "cites_article")},
                                      ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
