"""Turn model outputs into a HakamContract.

One place for the rules both the batch contracts (scripts/make_contracts.py) and
live clip inference (src/inference/predict.py) follow:
- the cascade: no card decision without an offence;
- the action family is description, not decision - below the confidence
  threshold it is left out rather than handed to the LLM as a hedged guess.
"""

from __future__ import annotations

import numpy as np

from src.config import ACTION_FAMILIES, CONFIDENCE_THRESHOLD
from src.contract import HakamContract, Prediction

# Class order of each model head. Must match src.models.data.TASKS (tested).
CLASSES = {
    "offence": ["no_offence", "offence"],
    "card": ["no_card", "card"],
    "action_class": list(ACTION_FAMILIES),
    "body_part": ["under_body", "upper_body"],
}


def assemble(action_id: str, offence: Prediction, card: Prediction | None,
             attributes: dict[str, Prediction], model_version: str, num_views: int) -> HakamContract:
    if offence.label != "offence":
        card = None
    attributes = dict(attributes)
    action = attributes.get("action_class")
    if action is not None and not action.is_confident(CONFIDENCE_THRESHOLD):
        del attributes["action_class"]
    return HakamContract(action_id=action_id, offence=offence, card=card, attributes=attributes,
                         model_version=model_version, num_views=num_views)


def _decide(task: str, probs: np.ndarray, thresholds: dict[str, float]) -> Prediction:
    classes = CLASSES[task]
    if len(classes) == 2:
        index = int(probs[1] >= thresholds.get(task, 0.5))
    else:
        index = int(np.argmax(probs))
    return Prediction(classes[index], round(float(probs[index]), 4))


def from_probabilities(action_id: str, probs: dict[str, np.ndarray], thresholds: dict[str, float],
                       model_version: str, num_views: int) -> HakamContract:
    """``probs`` maps each task to its view-averaged class probabilities."""
    offence = _decide("offence", probs["offence"], thresholds)
    card = _decide("card", probs["card"], thresholds) if "card" in probs else None
    attributes = {t: _decide(t, probs[t], thresholds) for t in ("action_class", "body_part") if t in probs}
    return assemble(action_id, offence, card, attributes, model_version, num_views)
