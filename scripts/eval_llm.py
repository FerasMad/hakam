#!/usr/bin/env python3
"""Evaluate both prompt versions on a broad suite of mock CV contracts."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ARTIFACTS
from src.contract import HakamContract, Prediction, mock_contract
from src.llm.faithfulness import score
from src.llm.generate import explain


ACTION_CLASSES = (
    "tackling",
    "standing tackling",
    "high leg",
    "holding",
    "pushing",
    "elbowing",
    "challenge",
    "dive",
    "dont know",
)


def _case(
    case_id: str,
    *,
    action_class: str,
    offence: str = "offence",
    card: str | None = "card",
    colour: str | None = None,
    offence_confidence: float = 0.88,
    action_confidence: float = 0.81,
    card_confidence: float | None = None,
    contact: str = "with_contact",
    contact_confidence: float = 0.90,
    body_part: str = "under_body",
    try_to_play: str = "no",
    touch_ball: str = "no",
) -> HakamContract:
    contract = mock_contract(
        offence=offence,
        confidence=offence_confidence,
        card=card,
        colour=colour,
    )
    contract.action_id = case_id
    if contract.card is not None and card_confidence is not None:
        contract.card.confidence = card_confidence
    contract.attributes = {
        "action_class": Prediction(action_class, action_confidence),
        "body_part": Prediction(body_part, 0.84),
        "contact": Prediction(contact, contact_confidence),
        "try_to_play": Prediction(try_to_play, 0.78),
        "touch_ball": Prediction(touch_ball, 0.79),
    }
    return contract


def build_cases() -> list[HakamContract]:
    """Twenty-four deterministic cases covering every contract branch."""

    cases = [
        _case(
            f"action-{index:02d}-{action.replace(' ', '-')}",
            action_class=action,
            contact="without_contact" if action in {"dive", "dont know"} else "with_contact",
            body_part="upper_body" if action in {"holding", "pushing", "elbowing"} else "under_body",
            try_to_play="yes" if action in {"tackling", "standing tackling", "challenge"} else "no",
            touch_ball="yes" if action in {"tackling", "challenge"} else "no",
        )
        for index, action in enumerate(ACTION_CLASSES, start=1)
    ]
    cases.extend(
        [
            _case("fair-challenge", action_class="challenge", offence="no_offence", card=None),
            _case("fair-tackle", action_class="tackling", offence="no_offence", card=None),
            _case(
                "no-offence-no-contact",
                action_class="dont know",
                offence="no_offence",
                card=None,
                contact="without_contact",
            ),
            _case(
                "simulation-no-offence",
                action_class="dive",
                offence="no_offence",
                card=None,
                contact="without_contact",
            ),
            _case("foul-no-card", action_class="holding", card="no_card"),
            _case("push-no-card", action_class="pushing", card="no_card", body_part="upper_body"),
            _case("challenge-no-card", action_class="challenge", card="no_card"),
            _case(
                "upper-body-card-colour-unknown",
                action_class="holding",
                body_part="upper_body",
                colour=None,
            ),
            _case("yellow-known", action_class="tackling", colour="yellow"),
            _case("red-known", action_class="high leg", colour="red"),
            _case(
                "low-action-confidence",
                action_class="elbowing",
                action_confidence=0.48,
                body_part="upper_body",
            ),
            _case(
                "low-contact-confidence",
                action_class="standing tackling",
                contact_confidence=0.44,
            ),
            _case(
                "low-card-confidence",
                action_class="pushing",
                card_confidence=0.52,
                body_part="upper_body",
            ),
            _case(
                "abstain-offence",
                action_class="tackling",
                offence_confidence=0.45,
            ),
            _case(
                "abstain-no-offence",
                action_class="challenge",
                offence="no_offence",
                card=None,
                offence_confidence=0.40,
            ),
        ]
    )
    assert len(cases) >= 20
    assert set(ACTION_CLASSES) <= {
        case.attributes["action_class"].label for case in cases
    }
    return cases


VERSIONS = ("v1", "v2", "v3")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summaries(rows: list[dict]) -> dict[str, dict[str, float]]:
    summaries = {}
    for version in VERSIONS:
        selected = [row for row in rows if row["prompt_version"] == version]
        low_conf = [row for row in selected if row["low_confidence_case"]]
        summaries[version] = {
            "mean_faithfulness": statistics.mean(row["faithfulness"] for row in selected),
            "unsupported_rate": statistics.mean(bool(row["unsupported"]) for row in selected),
            "citation_rate": statistics.mean(row["cites_article"] for row in selected),
            "correct_abstention_rate": statistics.mean(
                row["correctly_abstained"] for row in selected
            ),
            "low_conf_hedge_rate": (
                statistics.mean(row["hedged_low_conf"] for row in low_conf)
                if low_conf
                else 1.0
            ),
        }
    return summaries


def _print_table(summaries: dict[str, dict[str, float]]) -> None:
    labels = (
        ("mean faithfulness", "mean_faithfulness", False),
        ("% with unsupported claims", "unsupported_rate", True),
        ("% citing an article", "citation_rate", True),
        ("% correctly abstained", "correct_abstention_rate", True),
        ("% hedging low-confidence fields", "low_conf_hedge_rate", True),
    )
    print("| Metric | " + " | ".join(VERSIONS) + " |")
    print("|---|" + "---:|" * len(VERSIONS))
    for label, key, as_percent in labels:
        values = [summaries[version][key] for version in VERSIONS]
        rendered = [f"{value * 100:.1f}%" if as_percent else f"{value:.3f}" for value in values]
        print(f"| {label} | " + " | ".join(rendered) + " |")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--retrieval-mode",
        choices=("hybrid", "tags"),
        default="hybrid",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ARTIFACTS / "llm",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set. Copy .env.example to .env and paste the key.")
    os.environ["HAKAM_RETRIEVAL_MODE"] = args.retrieval_mode
    cases = build_cases()
    rows: list[dict] = []

    for version in VERSIONS:
        for index, contract in enumerate(cases, start=1):
            print(f"[{version}] {index:02d}/{len(cases)} {contract.action_id}", flush=True)
            explanation = explain(contract, prompt_version=version)
            metrics = score(explanation.text_ar, contract, explanation.articles)
            expected_abstain = contract.should_abstain()
            rows.append(
                {
                    "case_id": contract.action_id,
                    "prompt_version": version,
                    "expected_abstain": expected_abstain,
                    "abstained": explanation.abstained,
                    "correctly_abstained": explanation.abstained == expected_abstain,
                    "low_confidence_case": bool(contract.low_confidence_fields())
                    and not expected_abstain,
                    "faithfulness": metrics["faithfulness"],
                    "unsupported": "|".join(metrics["unsupported"]),
                    "cites_article": metrics["cites_article"],
                    "hedged_low_conf": metrics["hedged_low_conf"],
                    "claimed_labels": "|".join(metrics["claimed_labels"]),
                    "article_ids": "|".join(item["id"] for item in explanation.articles),
                    "contract_json": contract.to_json(indent=None),
                    "explanation_ar": explanation.text_ar,
                }
            )

    _write_csv(args.output_dir / "eval.csv", rows)

    # The automatic run selects a varied set; a human reviewer completes the
    # last three columns after reading every claim in these ten outputs.
    preferred_ids = {
        "action-01-tackling",
        "action-03-high-leg",
        "action-06-elbowing",
        "action-08-dive",
        "fair-challenge",
        "foul-no-card",
        "yellow-known",
        "low-action-confidence",
        "low-card-confidence",
        "abstain-offence",
    }
    audit_rows = []
    for row in rows:
        if row["prompt_version"] == "v3" and row["case_id"] in preferred_ids:
            audit_rows.append(
                {
                    "case_id": row["case_id"],
                    "prompt_version": row["prompt_version"],
                    "contract_json": row["contract_json"],
                    "explanation_ar": row["explanation_ar"],
                    "automatic_unsupported": row["unsupported"],
                    "claim_review": "PENDING HUMAN REVIEW",
                    "verdict": "PENDING",
                    "notes": "",
                }
            )
    _write_csv(args.output_dir / "manual_audit.csv", audit_rows)

    summaries = _summaries(rows)
    print()
    _print_table(summaries)
    print(f"\nEvaluation: {args.output_dir / 'eval.csv'}")
    print(f"Manual-audit sample: {args.output_dir / 'manual_audit.csv'}")


if __name__ == "__main__":
    main()
