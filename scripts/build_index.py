"""Build the retrieval index for the Laws of the Game corpus.

    python scripts/build_index.py            # rebuild only if laws/corpus.json changed
    python scripts/build_index.py --force

Writes laws/embeddings.npy (one normalised multilingual-E5 vector per chunk) and
laws/embeddings.json (model + corpus hash, so a stale index is detected), then
runs three sample queries so you can see retrieval working.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.contract import HakamContract, Prediction
from src.llm.retrieve import EMBEDDINGS_META_PATH, build_index, retrieve


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    path = build_index(force=args.force)
    meta = json.loads(EMBEDDINGS_META_PATH.read_text(encoding="utf-8"))
    print(f"index: {path}  ({meta['chunks']} chunks, {meta['model']})")

    samples = {
        "hands + card": {"action_class": "hands", "body_part": "upper_body"},
        "high leg + card": {"action_class": "high leg", "body_part": "under_body"},
        "no offence, tackle": {"action_class": "tackle"},
    }
    for name, attrs in samples.items():
        no_offence = name.startswith("no offence")
        contract = HakamContract(
            action_id=name,
            offence=Prediction("no_offence" if no_offence else "offence", 0.8),
            card=None if no_offence else Prediction("card", 0.7),
            attributes={k: Prediction(v, 0.8) for k, v in attrs.items()},
        )
        print(f"  {name:20s} -> {[a['id'] for a in retrieve(contract)]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
