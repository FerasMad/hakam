"""Label vocabulary and cascade target derivation.

The raw annotations encode two kinds of uncertainty as label values:

  * ``Offence`` can be ``Between`` - the annotator could not decide whether the
    challenge was an offence at all.
  * ``Severity`` 2.0 and 4.0 mean "borderline no/yellow" and "borderline
    yellow/red" - the annotator could not decide between two sanctions.

These are not noise to be silently merged away. They are a professional
referee recording genuine ambiguity, and they are the single best evidence for
why published accuracy on this dataset sits near 50%. The default policy drops
them from training and keeps them aside for the limitations analysis.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# Raw vocabularies, verified against VARS model/config/classes.py upstream
# --------------------------------------------------------------------------

ACTION_CLASSES = [
    "Tackling", "Standing tackling", "High leg", "Holding",
    "Pushing", "Elbowing", "Challenge", "Dive", "Dont know",
]

# The upstream file carries both capitalisations of "No offence".
OFFENCE_OFFENCE = {"Offence"}
OFFENCE_NO_OFFENCE = {"No offence", "No Offence"}
OFFENCE_AMBIGUOUS = {"Between"}

SEVERITY_MEANING = {
    "1.0": "no_card",
    "2.0": "borderline_no_yellow",
    "3.0": "yellow",
    "4.0": "borderline_yellow_red",
    "5.0": "red",
}

SEVERITY_AMBIGUOUS = {"2.0", "4.0"}
SEVERITY_NO_CARD = {"1.0"}
SEVERITY_YELLOW = {"3.0"}
SEVERITY_RED = {"5.0"}

BODYPART_CLASSES = ["Upper body", "Under body"]
BINARY_YES_NO = ["Yes", "No"]

# Auxiliary heads. Free supervision - the annotations already carry them.
AUXILIARY_FIELDS = {
    "Action class": ACTION_CLASSES,
    "Bodypart": BODYPART_CLASSES,
    "Contact": ["With contact", "Without contact"],
    "Try to play": BINARY_YES_NO,
    "Touch ball": BINARY_YES_NO,
}


class AmbiguityPolicy:
    """How to treat labels the annotator marked as uncertain."""

    DROP = "drop"        # exclude from training, report separately (default)
    MERGE = "merge"      # fold into the nearest definite class
    KEEP = "keep"        # treat as its own class


def derive_cascade_targets(
    offence: str,
    severity: str,
    policy: str = AmbiguityPolicy.DROP,
) -> dict[str, str | None]:
    """Map raw ``Offence`` and ``Severity`` onto the three cascade stages.

    ``None`` means the sample does not train that stage - either because an
    earlier stage terminated the cascade, or because the label was ambiguous
    under the current policy.

    Note the asymmetry: a sample dropped from stage 2 for an ambiguous severity
    is still perfectly valid supervision for stage 1. Discard per stage, never
    per row, or you throw away usable data.
    """
    offence = (offence or "").strip()
    severity = (severity or "").strip()

    targets: dict[str, str | None] = {
        "offence": None,
        "card": None,
        "card_colour": None,
    }

    # ---- Stage 1: is it an offence? --------------------------------------
    if offence in OFFENCE_OFFENCE:
        targets["offence"] = "offence"
    elif offence in OFFENCE_NO_OFFENCE:
        targets["offence"] = "no_offence"
        return targets                      # nothing further to decide
    elif offence in OFFENCE_AMBIGUOUS:
        if policy == AmbiguityPolicy.MERGE:
            targets["offence"] = "offence"  # Between leans towards an offence
        elif policy == AmbiguityPolicy.KEEP:
            targets["offence"] = "between"
            return targets
        else:
            return targets                  # DROP: no supervision at all
    else:
        return targets                      # unrecognised value

    # ---- Stage 2: does it warrant a card? --------------------------------
    if severity in SEVERITY_NO_CARD:
        targets["card"] = "no_card"
        return targets
    if severity in SEVERITY_YELLOW | SEVERITY_RED:
        targets["card"] = "card"
    elif severity in SEVERITY_AMBIGUOUS:
        if policy == AmbiguityPolicy.MERGE:
            targets["card"] = "no_card" if severity == "2.0" else "card"
            if severity == "2.0":
                return targets
        elif policy == AmbiguityPolicy.KEEP:
            targets["card"] = SEVERITY_MEANING[severity]
            return targets
        else:
            return targets                  # DROP
    else:
        return targets

    # ---- Stage 3: yellow or red? -----------------------------------------
    if severity in SEVERITY_YELLOW:
        targets["card_colour"] = "yellow"
    elif severity in SEVERITY_RED:
        targets["card_colour"] = "red"
    elif severity == "4.0" and policy == AmbiguityPolicy.MERGE:
        targets["card_colour"] = "red"

    return targets


def is_ambiguous(offence: str, severity: str) -> bool:
    """True when the annotator recorded uncertainty on either field."""
    return (offence or "").strip() in OFFENCE_AMBIGUOUS or (
        severity or ""
    ).strip() in SEVERITY_AMBIGUOUS
