"""Label normalisation and cascade derivation tests (PREPROCESSING.md 19.1).

The cases that matter most are the ones where a naive implementation silently
loses data or fabricates it:

  * ``Offence + 4.0`` must still supervise stage 2. Both outcomes of "borderline
    yellow/red" are cards, so the card decision is certain even though the
    colour is not. The previous implementation dropped all 56 such actions.
  * ``No offence + Red card`` must be quarantined, never silently accepted as a
    normal no-offence row.
  * An out-of-vocabulary severity must be reported, never rounded into the
    nearest class.
"""

from __future__ import annotations

import pytest

from src.data.labels import (
    AmbiguityPolicy,
    Code,
    Status,
    assert_cascade_invariants,
    derive_cascade_targets,
    derive_targets,
    is_ambiguous,
    normalise_offence,
    normalise_severity,
)


# ---------------------------------------------------------------- normalise

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Offence", "offence"),
        ("No offence", "no_offence"),
        ("No Offence", "no_offence"),   # both capitalisations ship upstream
        ("  Offence  ", "offence"),     # whitespace is noise, not meaning
        ("Between", "between"),
    ],
)
def test_offence_normalisation(raw, expected):
    assert normalise_offence(raw)[0] == expected


def test_unknown_offence_is_reported_not_guessed():
    value, codes = normalise_offence("Foul?")
    assert value is None
    assert Code.UNKNOWN_OFFENCE in codes


def test_missing_offence_is_distinct_from_unknown():
    assert normalise_offence("")[1] == [Code.MISSING_OFFENCE]
    assert normalise_offence(None)[1] == [Code.MISSING_OFFENCE]


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1.0", "no_card"), (1.0, "no_card"), (1, "no_card"),
        ("2.0", "borderline_no_yellow"),
        ("3.0", "yellow"), (3, "yellow"), ("  3.0  ", "yellow"),
        ("4.0", "borderline_yellow_red"),
        ("5.0", "red"),
    ],
)
def test_severity_normalisation(raw, expected):
    assert normalise_severity(raw)[0] == expected


@pytest.mark.parametrize("raw", ["3.5", "9.0", "high", "-1"])
def test_out_of_vocabulary_severity_is_not_rounded(raw):
    """Rounding 3.5 into "yellow" would fabricate a label nobody recorded."""
    value, codes = normalise_severity(raw)
    assert value is None
    assert Code.UNKNOWN_SEVERITY in codes


# ------------------------------------------------------------ conflict rules

@pytest.mark.parametrize("severity", ["2.0", "3.0", "4.0", "5.0"])
def test_no_offence_with_sanction_is_quarantined(severity):
    t = derive_targets("No offence", severity)
    assert t.status == Status.CONFLICT
    assert Code.OFFENCE_SEVERITY_CONFLICT in t.codes
    assert not (t.supervise_offence or t.supervise_card or t.supervise_card_colour)


@pytest.mark.parametrize("severity", ["", "1.0"])
def test_no_offence_without_sanction_is_a_valid_terminal_branch(severity):
    t = derive_targets("No offence", severity)
    assert t.status == Status.VALID
    assert t.target_offence == "no_offence"
    assert t.supervise_offence
    assert t.target_card is None and t.target_card_colour is None


# ------------------------------------------------------- stage-wise ambiguity

def test_severity_4_keeps_the_certain_card_and_withholds_only_the_colour():
    """The 56-action case. Both outcomes are cards, so stage 2 is certain."""
    t = derive_targets("Offence", "4.0")
    assert t.supervise_offence and t.target_offence == "offence"
    assert t.supervise_card and t.target_card == "card"
    assert not t.supervise_card_colour
    assert t.ambiguous
    assert Code.AMBIGUOUS_CARD_COLOUR in t.codes


def test_severity_2_supervises_only_offence():
    t = derive_targets("Offence", "2.0")
    assert t.supervise_offence
    assert not t.supervise_card and not t.supervise_card_colour
    assert Code.AMBIGUOUS_CARD_DECISION in t.codes


def test_between_supervises_nothing_under_stagewise_drop():
    t = derive_targets("Between", "1.0")
    assert not (t.supervise_offence or t.supervise_card or t.supervise_card_colour)
    assert t.ambiguous and Code.AMBIGUOUS_OFFENCE in t.codes


def test_between_with_a_card_severity_is_flagged_not_called_a_conflict():
    """"If this were an offence, this would be the sanction" is not a contradiction."""
    t = derive_targets("Between", "3.0")
    assert t.status == Status.AMBIGUOUS
    assert Code.CONDITIONAL_SANCTION_WITH_AMBIGUOUS_OFFENCE in t.codes
    assert Code.OFFENCE_SEVERITY_CONFLICT not in t.codes


def test_offence_with_missing_severity_still_trains_stage_one():
    t = derive_targets("Offence", "")
    assert t.status == Status.USABLE_PARTIAL
    assert t.supervise_offence and not t.supervise_card


def test_empty_offence_is_excluded():
    t = derive_targets("", "")
    assert t.status == Status.EXCLUDED
    assert not t.supervise_offence


# ------------------------------------------------------------ valid branches

@pytest.mark.parametrize(
    "severity,card,colour",
    [("1.0", "no_card", None), ("3.0", "card", "yellow"), ("5.0", "card", "red")],
)
def test_unambiguous_rows_supervise_every_determined_stage(severity, card, colour):
    t = derive_targets("Offence", severity)
    assert t.status == Status.VALID
    assert t.target_card == card
    assert t.target_card_colour == colour
    assert not t.ambiguous


# ---------------------------------------------------------------- policies

def test_merge_folds_borderline_into_the_nearest_definite_class():
    merged_low = derive_targets("Offence", "2.0", policy=AmbiguityPolicy.MERGE)
    merged_high = derive_targets("Offence", "4.0", policy=AmbiguityPolicy.MERGE)
    assert merged_low.target_card == "no_card"
    assert merged_high.target_card_colour == "red"


def test_keep_preserves_uncertainty_as_its_own_class():
    t = derive_targets("Offence", "2.0", policy=AmbiguityPolicy.KEEP)
    assert t.target_card == "borderline_no_yellow" and t.supervise_card


def test_legacy_interface_never_leaks_unsupervised_targets():
    """The old three-key API must not hand back a target training may not use."""
    assert derive_cascade_targets("Offence", "4.0") == {
        "offence": "offence", "card": "card", "card_colour": None
    }
    assert derive_cascade_targets("Between", "1.0") == {
        "offence": None, "card": None, "card_colour": None
    }


def test_is_ambiguous_matches_the_annotator_uncertainty_markers():
    assert is_ambiguous("Offence", "2.0")
    assert is_ambiguous("Offence", "4.0")
    assert is_ambiguous("Between", "1.0")
    assert not is_ambiguous("Offence", "3.0")
    assert not is_ambiguous("No offence", "")


# -------------------------------------------------------------- invariants

@pytest.mark.parametrize(
    "offence,severity",
    [("Offence", "1.0"), ("Offence", "2.0"), ("Offence", "3.0"),
     ("Offence", "4.0"), ("Offence", "5.0"), ("Offence", ""),
     ("No offence", ""), ("No offence", "1.0"), ("No offence", "5.0"),
     ("Between", "1.0"), ("Between", "3.0"), ("Between", ""), ("", "")],
)
def test_every_observed_combination_satisfies_the_cascade_invariants(offence, severity):
    for policy in (AmbiguityPolicy.STAGEWISE_DROP, AmbiguityPolicy.MERGE,
                   AmbiguityPolicy.KEEP):
        assert_cascade_invariants(derive_targets(offence, severity, policy=policy))
