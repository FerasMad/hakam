"""The CV -> LLM contract.

This module is the integration point the brief grades in section 10. The rule it
enforces is simple and absolute:

    The language model never sees the video.

It receives only a serialised HakamContract. Every sentence it produces must be
traceable to a field in that object. A multimodal model shown raw frames can
assert visual details it never reliably perceived; this pipeline makes that
structurally impossible rather than discouraging it in a prompt.

That property is measurable — see `explanation_faithfulness` in the evaluation
code — and it is the main difference between this design and X-VARS.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any

from src.config import CONFIDENCE_THRESHOLD


@dataclass
class Prediction:
    """One model output: a label plus how sure the model is about it."""

    label: str
    confidence: float

    def is_confident(self, threshold: float = CONFIDENCE_THRESHOLD) -> bool:
        return self.confidence >= threshold

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be in [0, 1], got {self.confidence!r} for label {self.label!r}"
            )


@dataclass
class HakamContract:
    """Everything the language model is permitted to know about an incident."""

    action_id: str

    # Cascade decisions. Later stages are None when an earlier stage stopped
    # the cascade — no card means no colour to predict.
    offence: Prediction
    card: Prediction | None = None
    card_colour: Prediction | None = None

    # Auxiliary attributes, used to make the explanation concrete.
    attributes: dict[str, Prediction] = field(default_factory=dict)

    # Provenance, so any generated text can be traced back to a specific run.
    model_version: str = "unset"
    num_views: int = 0

    # ------------------------------------------------------------------
    # Confidence handling
    # ------------------------------------------------------------------

    def low_confidence_fields(self, threshold: float = CONFIDENCE_THRESHOLD) -> list[str]:
        """Names of every prediction below the threshold.

        The generator must hedge these rather than stating them flatly.
        """
        weak = []
        for name in ("offence", "card", "card_colour"):
            pred = getattr(self, name)
            if pred is not None and not pred.is_confident(threshold):
                weak.append(name)
        weak.extend(
            name for name, pred in self.attributes.items() if not pred.is_confident(threshold)
        )
        return weak

    def should_abstain(self, threshold: float = CONFIDENCE_THRESHOLD) -> bool:
        """True when the primary decision is too weak to explain at all.

        If we cannot tell whether it was an offence, there is nothing honest to
        say about the sanction.
        """
        return not self.offence.is_confident(threshold)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def citable_values(self) -> set[str]:
        """Every value the generated explanation is allowed to assert.

        Used by the faithfulness check: a claim mentioning anything outside this
        set was invented by the language model.
        """
        values = {self.offence.label}
        for pred in (self.card, self.card_colour):
            if pred is not None:
                values.add(pred.label)
        values.update(pred.label for pred in self.attributes.values())
        return values

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["low_confidence_fields"] = self.low_confidence_fields()
        payload["abstain"] = self.should_abstain()
        return payload

    def to_json(self, indent: int = 2) -> str:
        """The exact string handed to the language model. Nothing else is."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> HakamContract:
        def _pred(raw: dict[str, Any] | None) -> Prediction | None:
            return None if raw is None else Prediction(**raw)

        return cls(
            action_id=payload["action_id"],
            offence=_pred(payload["offence"]),
            card=_pred(payload.get("card")),
            card_colour=_pred(payload.get("card_colour")),
            attributes={k: Prediction(**v) for k, v in payload.get("attributes", {}).items()},
            model_version=payload.get("model_version", "unset"),
            num_views=payload.get("num_views", 0),
        )


def mock_contract(
    offence: str = "offence",
    confidence: float = 0.88,
    card: str | None = "card",
    colour: str | None = "yellow",
) -> HakamContract:
    """Build a contract by hand, with no trained model.

    The entire retrieval and generation half can be developed and evaluated
    against these while waiting for dataset access.
    """
    return HakamContract(
        action_id="mock-0001",
        offence=Prediction(offence, confidence),
        card=None if card is None else Prediction(card, confidence),
        card_colour=None if colour is None else Prediction(colour, confidence),
        attributes={
            "action_class": Prediction("tackling", 0.81),
            "body_part": Prediction("under_body", 0.90),
            "contact": Prediction("with_contact", 0.95),
            "try_to_play": Prediction("no", 0.72),
        },
        model_version="mock",
        num_views=2,
    )
