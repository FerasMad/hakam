"""Label normalisation, conflict detection, and cascade target derivation.

Implements sections 6-8 of PREPROCESSING.md.

Two principles run through this module.

**Raw values are never overwritten.** Normalisation removes formatting noise and
nothing else. It does not invent a missing label, and it does not resolve a
semantic disagreement between two fields. Every derived value carries reason
codes explaining how it was reached.

**Supervision is decided per stage, never per row.** Discarding a whole incident
because one late stage is ambiguous is the most expensive mistake available
here. ``Offence + severity 4.0`` means "borderline yellow/red" - the colour is
genuinely uncertain, but *that a card is due is certain*, so it must still train
stage 2. There are 56 such actions; the earlier implementation dropped all of
them from stage 2.

Measured against the real dataset on 3 Sep 2026, across all 3,628 labelled
actions in Train/Valid/Test:

    Offence    1.0    1640     Offence    (empty)  39
    Offence    3.0     858     Offence    5.0      35
    Offence    2.0     493     No offence 1.0      17
    No offence (empty) 363     Between    (empty)  11
    Between    1.0     111     Between    2.0       3
    Offence    4.0      56     (empty)    (empty)   1
                               Between    3.0       1

Note what is *absent*: no ``No offence`` is paired with a card-level severity
anywhere in the dataset. The conflict machinery below quarantines nothing today.
It is a guard against schema drift, not a repair.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Raw vocabularies, verified against VARS model/config/classes.py upstream
# --------------------------------------------------------------------------

ACTION_CLASSES = [
    "Tackling", "Standing tackling", "High leg", "Holding",
    "Pushing", "Elbowing", "Challenge", "Dive", "Dont know",
]

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


# --------------------------------------------------------------------------
# Status and reason codes (PREPROCESSING.md section 7)
# --------------------------------------------------------------------------

class Status:
    VALID = "valid"
    USABLE_PARTIAL = "usable_partial"
    AMBIGUOUS = "ambiguous"
    CONFLICT = "conflict"
    INVALID_MEDIA = "invalid_media"
    EXCLUDED = "excluded"


class Code:
    MISSING_OFFENCE = "MISSING_OFFENCE"
    MISSING_SEVERITY = "MISSING_SEVERITY"
    UNKNOWN_OFFENCE = "UNKNOWN_OFFENCE"
    UNKNOWN_SEVERITY = "UNKNOWN_SEVERITY"
    AMBIGUOUS_OFFENCE = "AMBIGUOUS_OFFENCE"
    CONDITIONAL_SANCTION_WITH_AMBIGUOUS_OFFENCE = (
        "CONDITIONAL_SANCTION_WITH_AMBIGUOUS_OFFENCE"
    )
    AMBIGUOUS_CARD_DECISION = "AMBIGUOUS_CARD_DECISION"
    AMBIGUOUS_CARD_COLOUR = "AMBIGUOUS_CARD_COLOUR"
    OFFENCE_SEVERITY_CONFLICT = "OFFENCE_SEVERITY_CONFLICT"
    MISSING_CLIP = "MISSING_CLIP"
    CORRUPT_CLIP = "CORRUPT_CLIP"
    INSUFFICIENT_FRAMES = "INSUFFICIENT_FRAMES"
    INVALID_FPS = "INVALID_FPS"
    INVALID_RESOLUTION = "INVALID_RESOLUTION"
    INVALID_REPLAY_SPEED = "INVALID_REPLAY_SPEED"
    FEWER_THAN_TWO_VALID_VIEWS = "FEWER_THAN_TWO_VALID_VIEWS"
    DUPLICATE_WITHIN_ACTION = "DUPLICATE_WITHIN_ACTION"
    DUPLICATE_ACROSS_ACTIONS = "DUPLICATE_ACROSS_ACTIONS"
    CROSS_SPLIT_DUPLICATE = "CROSS_SPLIT_DUPLICATE"
    AUXILIARY_LABEL_CONFLICT = "AUXILIARY_LABEL_CONFLICT"


# --------------------------------------------------------------------------
# Normalisation (PREPROCESSING.md section 6)
# --------------------------------------------------------------------------

# Matching is exact after trimming and case-folding. Deliberately not fuzzy: a
# new spelling must surface in the unknown-value report and be added here after
# review, rather than being silently absorbed.
OFFENCE_MAP = {
    "offence": "offence",
    "no offence": "no_offence",
    "between": "between",
}

SEVERITY_MAP = {
    1: "no_card",
    2: "borderline_no_yellow",
    3: "yellow",
    4: "borderline_yellow_red",
    5: "red",
}

CARD_LEVEL_SEVERITIES = {"yellow", "borderline_yellow_red", "red"}
AMBIGUOUS_SEVERITIES = {"borderline_no_yellow", "borderline_yellow_red"}


def normalise_offence(raw: str | None) -> tuple[str | None, list[str]]:
    """Canonical offence value plus any reason codes."""
    text = (raw or "").strip()
    if not text:
        return None, [Code.MISSING_OFFENCE]
    canonical = OFFENCE_MAP.get(text.casefold())
    if canonical is None:
        return None, [Code.UNKNOWN_OFFENCE]
    return canonical, []


def normalise_severity(raw: str | float | None) -> tuple[str | None, list[str]]:
    """Canonical severity value plus any reason codes.

    Accepts ``"3.0"``, ``3.0`` and ``3``. Values that are numeric but not one of
    the five defined levels are reported as unknown rather than rounded into the
    nearest class - rounding an unexpected 3.5 into "yellow" would fabricate a
    label.
    """
    if raw is None:
        return None, [Code.MISSING_SEVERITY]
    text = str(raw).strip()
    if not text:
        return None, [Code.MISSING_SEVERITY]

    try:
        number = float(text)
    except ValueError:
        return None, [Code.UNKNOWN_SEVERITY]

    if number != int(number) or int(number) not in SEVERITY_MAP:
        return None, [Code.UNKNOWN_SEVERITY]
    return SEVERITY_MAP[int(number)], []


# --------------------------------------------------------------------------
# Ambiguity policy
# --------------------------------------------------------------------------

class AmbiguityPolicy:
    """How to treat labels the annotator marked as uncertain."""

    STAGEWISE_DROP = "stagewise-drop"  # default: withhold only uncertain stages
    MERGE = "merge"                    # fold into the nearest definite class
    KEEP = "keep"                      # treat uncertainty as its own class

    # Retained so older callers keep working. Stage-wise drop is what DROP
    # always intended, minus the bug that discarded certain stages alongside
    # the uncertain ones.
    DROP = STAGEWISE_DROP


# --------------------------------------------------------------------------
# Cascade derivation (PREPROCESSING.md section 8)
# --------------------------------------------------------------------------

@dataclass
class CascadeTargets:
    """Derived targets plus the provenance of how they were reached."""

    offence: str | None = None            # canonical, pre-cascade
    severity: str | None = None           # canonical, pre-cascade

    target_offence: str | None = None
    target_card: str | None = None
    target_card_colour: str | None = None

    supervise_offence: bool = False
    supervise_card: bool = False
    supervise_card_colour: bool = False

    status: str = Status.EXCLUDED
    codes: list[str] = field(default_factory=list)
    ambiguous: bool = False

    def as_dict(self) -> dict:
        return {
            "offence": self.offence,
            "severity": self.severity,
            "target_offence": self.target_offence,
            "target_card": self.target_card,
            "target_card_colour": self.target_card_colour,
            "supervise_offence": self.supervise_offence,
            "supervise_card": self.supervise_card,
            "supervise_card_colour": self.supervise_card_colour,
            "quality_status": self.status,
            "quality_codes": "|".join(self.codes),
            "ambiguous": self.ambiguous,
        }


def derive_targets(
    offence_raw: str | None,
    severity_raw: str | float | None,
    policy: str = AmbiguityPolicy.STAGEWISE_DROP,
) -> CascadeTargets:
    """Map raw ``Offence`` and ``Severity`` onto cascade targets.

    A target is set only when the annotation actually determines it. The
    ``supervise_*`` flags are what training reads; a target may be present for
    analysis while its supervision flag is false.
    """
    offence, offence_codes = normalise_offence(offence_raw)
    severity, severity_codes = normalise_severity(severity_raw)

    out = CascadeTargets(offence=offence, severity=severity)
    out.codes = list(offence_codes)
    # A missing severity is only worth reporting once the offence root is sound;
    # it is re-added below at the point where it actually costs supervision.
    if severity_codes and severity_codes != [Code.MISSING_SEVERITY]:
        out.codes += severity_codes

    # ---- Root missing or unrecognised: nothing downstream is safe ----------
    if offence is None:
        out.status = Status.EXCLUDED
        return out

    # ---- Conflict: a terminal "no offence" carrying a sanction -------------
    # Absent from the current dataset; kept as a schema-drift guard. Never
    # silently salvaged - we cannot know which of the two fields is wrong.
    if offence == "no_offence" and (
        severity in CARD_LEVEL_SEVERITIES or severity == "borderline_no_yellow"
    ):
        out.status = Status.CONFLICT
        out.codes.append(Code.OFFENCE_SEVERITY_CONFLICT)
        return out

    # ---- Ambiguous root: no cascade stage is safe to supervise -------------
    if offence == "between":
        out.ambiguous = True
        out.status = Status.AMBIGUOUS
        out.codes.append(Code.AMBIGUOUS_OFFENCE)
        if severity in CARD_LEVEL_SEVERITIES:
            # "If this were an offence, this would be the sanction." Not a
            # proven contradiction, but unusable while the root is unresolved.
            out.codes.append(Code.CONDITIONAL_SANCTION_WITH_AMBIGUOUS_OFFENCE)
        if policy == AmbiguityPolicy.KEEP:
            out.target_offence = "between"
            out.supervise_offence = True
        elif policy == AmbiguityPolicy.MERGE:
            out.target_offence = "offence"
            out.supervise_offence = True
        return out

    # ---- Stage 1 ----------------------------------------------------------
    out.target_offence = offence
    out.supervise_offence = True

    if offence == "no_offence":
        out.status = Status.VALID
        return out                      # terminal branch, nothing further

    # ---- Stage 2: does it warrant a card? ---------------------------------
    if severity is None:
        out.status = Status.USABLE_PARTIAL
        out.codes.append(Code.MISSING_SEVERITY)
        return out

    if severity == "no_card":
        out.target_card = "no_card"
        out.supervise_card = True
        out.status = Status.VALID
        return out

    if severity == "borderline_no_yellow":
        # Genuinely uncertain whether a card is due. Stage 1 stays supervised.
        out.ambiguous = True
        out.status = Status.AMBIGUOUS
        out.codes.append(Code.AMBIGUOUS_CARD_DECISION)
        if policy == AmbiguityPolicy.MERGE:
            out.target_card = "no_card"
            out.supervise_card = True
        elif policy == AmbiguityPolicy.KEEP:
            out.target_card = severity
            out.supervise_card = True
        return out

    # yellow, borderline_yellow_red and red all mean a card is due.
    out.target_card = "card"
    out.supervise_card = True

    # ---- Stage 3: which colour? -------------------------------------------
    if severity == "yellow":
        out.target_card_colour = "yellow"
        out.supervise_card_colour = True
        out.status = Status.VALID
    elif severity == "red":
        out.target_card_colour = "red"
        out.supervise_card_colour = True
        out.status = Status.VALID
    else:  # borderline_yellow_red
        # The colour is uncertain; the card is not. This is the case the
        # earlier implementation lost entirely - 56 actions of stage-2
        # supervision.
        out.ambiguous = True
        out.status = Status.AMBIGUOUS
        out.codes.append(Code.AMBIGUOUS_CARD_COLOUR)
        if policy == AmbiguityPolicy.MERGE:
            out.target_card_colour = "red"
            out.supervise_card_colour = True
        elif policy == AmbiguityPolicy.KEEP:
            out.target_card_colour = severity
            out.supervise_card_colour = True

    return out


def assert_cascade_invariants(t: CascadeTargets) -> None:
    """Fail loudly on a target set the cascade could never produce.

    From PREPROCESSING.md section 20. Checked before manifests are written, so a
    derivation bug surfaces at preprocessing time rather than as a confusing
    training result days later.
    """
    if t.target_offence == "no_offence":
        assert t.target_card is None and t.target_card_colour is None, (
            f"no_offence must terminate the cascade, got {t.as_dict()}"
        )
    if t.target_card == "no_card":
        assert t.target_offence == "offence", (
            f"no_card requires an offence, got {t.as_dict()}"
        )
        assert t.target_card_colour is None, (
            f"no_card cannot carry a colour, got {t.as_dict()}"
        )
    if t.target_card_colour in {"yellow", "red"}:
        assert t.target_offence == "offence" and t.target_card == "card", (
            f"a colour requires offence+card, got {t.as_dict()}"
        )


# --------------------------------------------------------------------------
# Backwards-compatible helpers
# --------------------------------------------------------------------------

def derive_cascade_targets(
    offence: str,
    severity: str,
    policy: str = AmbiguityPolicy.STAGEWISE_DROP,
) -> dict[str, str | None]:
    """Older three-key interface, kept so existing callers keep working.

    Returns only supervised targets, so a target withheld from training cannot
    leak into a training set through this path.
    """
    t = derive_targets(offence, severity, policy=policy)
    return {
        "offence": t.target_offence if t.supervise_offence else None,
        "card": t.target_card if t.supervise_card else None,
        "card_colour": t.target_card_colour if t.supervise_card_colour else None,
    }


def is_ambiguous(offence: str, severity: str) -> bool:
    """True when the annotator recorded uncertainty on either field."""
    return derive_targets(offence, severity).ambiguous
