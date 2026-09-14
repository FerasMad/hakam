#!/usr/bin/env python3
"""Run one grounded Arabic explanation against the built-in mock contract."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.contract import mock_contract
from src.llm.faithfulness import score
from src.llm.generate import explain


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-version", choices=("v1", "v2"), default="v2")
    parser.add_argument(
        "--retrieval-mode",
        choices=("hybrid", "tags"),
        default="hybrid",
        help="Hybrid E5 retrieval is the production default; tags is the light fallback.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["HAKAM_RETRIEVAL_MODE"] = args.retrieval_mode

    # Colour is intentionally absent because that is the realistic CV output.
    contract = mock_contract(colour=None)
    explanation = explain(contract, prompt_version=args.prompt_version)
    faithfulness = score(explanation.text_ar, contract, explanation.articles)

    print("CONTRACT")
    print(contract.to_json())
    print("\nRETRIEVED ARTICLES")
    for rank, article in enumerate(explanation.articles, start=1):
        print(
            f"{rank}. {article['law']} — {article['section']} — "
            f"{article['title_ar']} [{article['id']}]"
        )
    print("\nARABIC EXPLANATION")
    print(explanation.text_ar)
    print("\nFAITHFULNESS")
    print(json.dumps(faithfulness, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
