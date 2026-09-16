"""Lexicon-based checks that generated Arabic stays inside its evidence."""

from __future__ import annotations

import re

from src.contract import HakamContract, Prediction
from src.llm import lexicon


_COLOUR_LABELS = {"yellow", "red"}
_CONDITIONAL_MARKERS = (
    "قد",
    "اذا",
    "في حال",
    "عندما",
    "يمكن",
    "احتمال",
    "مشروط",
    "بحسب تصنيف",
)
_HEDGE_MARKERS = (
    "يبدو",
    "علي الارجح",
    "غالبا",
    "ربما",
    "يحتمل",
    "غير مؤكد",
    "بحسب الثقه",
    "قد يكون",
)


def _key(value: str) -> str:
    return str(value).strip().lower().replace("_", " ")


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"[\n.!؟؛]+", text) if part.strip()]


# Line headings of the v2/v3 structure. They name a section, they claim nothing
# about the incident ("العقوبة" is a card form, "مخالفة" an offence form).
_SECTION_HEADINGS = (
    "القرار:",
    "التفسير:",
    "العقوبة الفنية:",
    "العقوبة الانضباطية:",
    "لماذا تُعد مخالفة:",
    "لماذا لا تُعد مخالفة:",
)


def _incident_claim_text(text: str) -> str:
    """Exclude citation and confidence metadata, and section headings, from claims."""

    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(("المادة:", "مستوى الثقة:")):
            continue
        for heading in _SECTION_HEADINGS:
            if stripped.startswith(heading):
                stripped = stripped[len(heading):].strip()
                break
        lines.append(stripped)
    return "\n".join(lines)


def _conditional_colours(text: str) -> set[str]:
    conditional: set[str] = set()
    for raw_sentence in _sentences(text):
        sentence = lexicon.normalise(raw_sentence)
        if not any(marker in sentence for marker in _CONDITIONAL_MARKERS):
            continue
        labels = lexicon.found_labels(raw_sentence)
        conditional.update(labels & _COLOUR_LABELS)
    return conditional


def _false_positive_labels(text: str, contract: HakamContract) -> set[str]:
    """Known surface-form collisions that are not incident-label claims."""

    normalised = lexicon.normalise(text)
    false: set[str] = set()
    # The lexicon form مرفق (elbow) must not match the adjective المرفقة
    # (attached), which appears frequently when referring to supplied articles.
    action = contract.attributes.get("action_class")
    action_key = _key(action.label) if action is not None else ""
    elbow_phrases = {
        lexicon.normalise(form)
        for form in lexicon.arabic_forms("elbowing")
        if len(lexicon.normalise(form).split()) > 1
    }
    if action_key != "elbowing" and not any(form in normalised for form in elbow_phrases):
        # Bare «المرفق» is commonly the adjective “attached”, especially in
        # «النص المرفق», and is not enough on its own to claim elbowing.
        false.add("elbowing")

    if action_key not in {"tackle", "tackling", "standing tackling", "challenge"}:
        specific_tackling = {
            lexicon.normalise(form)
            for form in lexicon.arabic_forms("tackling")
            if lexicon.normalise(form) != "تدخل"
        }
        if not any(form in normalised for form in specific_tackling):
            # The generic noun «تدخل» also means an intervention/decision.
            false.add("tackling")

    if "no" in {_key(value) for value in contract.citable_values()}:
        without_negated_attempts = normalised
        for pattern in (
            r"عدم\s+وجود\s+محاوله\s+لعب\s+الكره",
            r"غياب\s+محاوله\s+لعب\s+الكره",
            r"لا\s+توجد\s+محاوله\s+لعب\s+الكره",
        ):
            without_negated_attempts = re.sub(pattern, " ", without_negated_attempts)
        if "yes" not in lexicon.found_labels(without_negated_attempts):
            false.add("yes")

    # "لون البطاقة غير محدد" describes the explicit null state; it is not the
    # action-class label ``dont know``.
    if contract.card_colour is None and re.search(
        r"لون\s+البطاقه.{0,24}غير\s+محدد", normalised
    ):
        false.add("dont know")

    def only_negated(label: str) -> bool:
        forms = {
            lexicon.normalise(form)
            for form in lexicon.arabic_forms(label)
            if lexicon.normalise(form)
        }
        saw_form = False
        for sentence in _sentences(text):
            sentence = lexicon.normalise(sentence)
            for form in forms:
                start = 0
                while (position := sentence.find(form, start)) >= 0:
                    saw_form = True
                    prefix = sentence[max(0, position - 36) : position]
                    if not any(
                        marker in prefix
                        for marker in ("لا ", "لم ", "ليس", "غير", "دون", "بدون", "عدم", "غياب")
                    ):
                        return False
                    start = position + len(form)
        return saw_form

    if contract.offence.label == "no_offence" and only_negated("offence"):
        false.add("offence")
    if (contract.card is None or _key(contract.card.label) == "no card") and only_negated("card"):
        false.add("card")
    return false


def _prediction_for_field(contract: HakamContract, field: str) -> Prediction | None:
    if field in {"offence", "card", "card_colour"}:
        return getattr(contract, field)
    return contract.attributes.get(field)


def _field_is_hedged_or_absent(text: str, prediction: Prediction) -> bool:
    forms = sorted(
        {lexicon.normalise(form) for form in lexicon.arabic_forms(prediction.label)},
        key=len,
        reverse=True,
    )
    mentioned_sentences = []
    for raw_sentence in _sentences(text):
        sentence = lexicon.normalise(raw_sentence)
        if any(form and form in sentence for form in forms):
            mentioned_sentences.append(sentence)
    if not mentioned_sentences:
        return True
    return all(any(marker in sentence for marker in _HEDGE_MARKERS) for sentence in mentioned_sentences)


def _cites_retrieved_article(text: str, articles: list[dict]) -> bool:
    normalised = lexicon.normalise(text)
    for article in articles:
        law = str(article.get("law", ""))
        number = re.search(r"\d+", law)
        if number and re.search(rf"(?:القانون|law)\s*(?:رقم\s*)?{number.group()}", normalised, re.I):
            return True
        if _key(law) == "glossary" and (
            "مسرد" in normalised or "تعريف" in normalised or "glossary" in normalised
        ):
            return True
    return False


def score(explanation_ar: str, contract: HakamContract, articles: list[dict]) -> dict:
    """Measure label faithfulness, citation and low-confidence hedging.

    Conditional yellow/red wording is treated as a quotation of the Law rather
    than an incident claim when the contract has no colour. The shared Arabic
    form ``التحام`` is accepted for either ``tackling`` or ``challenge``.
    """

    claim_text = _incident_claim_text(explanation_ar)
    claimed = set(lexicon.found_labels(claim_text))
    claimed -= _false_positive_labels(claim_text, contract)
    if contract.card_colour is None:
        claimed -= _conditional_colours(claim_text)

    allowed = {_key(value) for value in contract.citable_values()}
    if contract.should_abstain():
        allowed.add("refer")

    supported: set[str] = set()
    for label in claimed:
        if label in allowed:
            supported.add(label)
        elif label in {"tackling", "challenge"} and allowed & {
            "tackle",
            "tackling",
            "standing tackling",
            "challenge",
        }:
            # The generic forms «تدخل» / «التحام» are how Arabic names a tackle, so
            # they are supported by every tackle-family label, including the legacy
            # ``standing tackling`` value. Naming a specific member of another family
            # (holding for "hands") is not: the model did not claim it.
            supported.add(label)
    unsupported = claimed - supported

    low_fields = contract.low_confidence_fields()
    hedged = all(
        (prediction := _prediction_for_field(contract, field)) is None
        or _field_is_hedged_or_absent(claim_text, prediction)
        for field in low_fields
    )

    return {
        "claimed_labels": sorted(claimed),
        "supported": sorted(supported),
        "unsupported": sorted(unsupported),
        "faithfulness": round(len(supported) / len(claimed), 4) if claimed else 1.0,
        "cites_article": _cites_retrieved_article(explanation_ar, articles),
        "hedged_low_conf": hedged,
    }
