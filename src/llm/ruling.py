"""The full ruling after a clip: restart, disciplinary sanction, the Law, and why.

Every line here is derived from two things only: the contract and Law 12 as
curated in laws/corpus.json. Where the contract does not carry a fact the Law
depends on - the location of the foul (penalty area), whether a raised foot made
contact, the card colour - the line states the Law's rule conditionally instead
of guessing. Each rule records the corpus chunks that support it, so the cited
article is always one the explanation actually rests on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.config import ACTION_FAMILIES, CONFIDENCE_THRESHOLD
from src.contract import HakamContract, Prediction

_FAMILY_OF = {m: f for f, members in ACTION_FAMILIES.items() for m in members}

_ACTION_AR = {
    "tackle": "تدخل على المنافس",
    "hands": "مخالفة باليدين",
    "elbowing": "استخدام المرفق",
    "high leg": "رفع القدم عالياً",
    "dive": "سقوط تمثيلي",
}

_BODY_AR = {"upper_body": "بالجزء العلوي من الجسم", "under_body": "بالجزء السفلي من الجسم"}

# Corpus chunk that states the Law for each action family.
_ACTION_ARTICLE = {
    "tackle": "law12-1-direct-free-kick-actions",
    "hands": "law12-1-direct-free-kick-actions",
    "elbowing": "law12-1-striking-elbow",
    "high leg": "law12-2-dangerous-play-contact",
    "dive": "law12-3-simulation",
}


@dataclass
class Ruling:
    restart: str
    disciplinary: str
    why: str
    article_ids: list[str] = field(default_factory=list)
    why_title: str = "لماذا تُعد مخالفة"


def _weak(pred: Prediction | None) -> bool:
    return pred is not None and pred.confidence < CONFIDENCE_THRESHOLD


def _hedge(pred: Prediction | None, text: str) -> str:
    return f"على الأرجح {text}" if _weak(pred) else text


def _action_text(action: Prediction, family: str) -> str:
    """Arabic for the label the contract actually carries, so no finer or coarser claim is made."""
    from src.llm.prompts import arabic_label

    return arabic_label(action.label) or _ACTION_AR[family]


def action_family(contract: HakamContract) -> str | None:
    """The contract's action as a Law 12 family, whether it carries a family or a fine label."""
    action = contract.attributes.get("action_class")
    if action is None:
        return None
    label = str(action.label).lower()
    if label in ACTION_FAMILIES or label == "dive":
        return label
    return _FAMILY_OF.get(label)


def _restart(contract: HakamContract, family: str | None) -> tuple[str, list[str]]:
    if contract.offence.label == "no_offence":
        return "لا توجد مخالفة، ويستمر اللعب دون ركلة حرة.", ["law12-no-offence-fair-challenge"]

    if family == "dive":
        return ("ركلة حرة غير مباشرة للفريق المنافس، لأن محاولة خداع الحكم لا تُحتسب "
                "لصالح من قام بها."), ["law12-3-simulation"]

    if family == "high leg":
        contact = contract.attributes.get("contact")
        if contact is not None and contact.label == "with_contact":
            return (_hedge(contact, "ركلة حرة مباشرة لأن اللعب الخطر نتج عنه تلامس،")
                    + " أو ركلة جزاء إذا وقعت داخل منطقة جزاء المخالف."), ["law12-2-dangerous-play-contact"]
        if contact is not None and contact.label == "without_contact":
            return _hedge(contact, "ركلة حرة غير مباشرة لأن اللعب الخطر لم ينتج عنه تلامس."), [
                "law12-2-dangerous-play"]
        return ("ركلة حرة مباشرة إذا نتج عن اللعب الخطر تلامس (أو ركلة جزاء إذا وقعت داخل "
                "منطقة جزاء المخالف)، وركلة حرة غير مباشرة إذا لم يحدث تلامس."), [
            "law12-2-dangerous-play-contact", "law12-2-dangerous-play"]

    ids = [_ACTION_ARTICLE[family]] if family in _ACTION_ARTICLE else ["law12-1-direct-free-kick-actions"]
    if family == "hands":
        # A hands foul is holding or pushing; cite both rules, never guess which.
        ids += ["law12-1-holding-opponent", "law12-1-pushing"]
    return ("ركلة حرة مباشرة للفريق المنافس، أو ركلة جزاء إذا وقعت داخل منطقة جزاء المخالف "
            "(مكان الواقعة لا يرد في العقد)."), ids + ["law12-1-contact-direct-free-kick"]


def _disciplinary(contract: HakamContract) -> tuple[str, list[str]]:
    if contract.offence.label == "no_offence":
        return "لا عقوبة انضباطية.", []
    card = contract.card
    if card is None:
        return "لم يحدد العقد قرار البطاقة.", []
    if card.label == "no_card":
        return (_hedge(card, "لا تستوجب بطاقة؛")
                + " فالقانون 12 لا يتطلب إجراءً انضباطياً عندما لا تتجاوز المخالفة درجة الإهمال."), [
            "law12-no-card-careless", "law12-1-careless-definition"]

    colour = contract.card_colour
    if colour is None:
        return (_hedge(card, "تستوجب بطاقة، ولون البطاقة غير محدد:")
                + " إذا اعتُبر التدخل متهوراً فالعقوبة إنذار، وإذا كان باستخدام قوة مفرطة فالعقوبة طرد."), [
            "law12-card-colour-conditional", "law12-3-reckless-caution",
            "law12-1-excessive-force-definition"]
    if colour.label == "yellow":
        return _hedge(colour, "إنذار (بطاقة صفراء)، إذ يعد القانون 12 التهور سلوكاً غير رياضي."), [
            "law12-3-reckless-caution"]
    return _hedge(colour, "طرد (بطاقة حمراء)، إذ يعد القانون 12 استخدام القوة المفرطة موجباً للطرد."), [
        "law12-1-excessive-force-definition"]


def _why(contract: HakamContract, family: str | None) -> tuple[str, str]:
    action = contract.attributes.get("action_class")
    body = contract.attributes.get("body_part")
    body_ar = _BODY_AR.get(body.label) if body is not None else None

    if contract.offence.label == "no_offence":
        sentences = ["اللعب سليم وفق العقد."]
        if family in _ACTION_AR and family != "dive":
            sentences.append(_hedge(action, f"صُنّف الفعل {_action_text(action, family)}."))
        sentences.append("والقانون 12 لا يعد المنافسة على الكرة خطأً لمجرد وجودها، "
                         "ما دامت لا تتضمن فعلاً يعاقب عليه القانون.")
        return "لماذا لا تُعد مخالفة", " ".join(sentences)

    sentences = []
    if family in _ACTION_AR:
        what = f"صُنّف الفعل {_action_text(action, family)}"
        if body_ar and not _weak(body):
            what += f" {body_ar}"
        sentences.append(_hedge(action, what + "."))
        if body_ar and _weak(body):
            sentences.append(f"على الأرجح وقع {body_ar}.")
    else:
        sentences.append("يحدد العقد وجود مخالفة دون تصنيف مؤكد لنوع الفعل.")

    if family == "high leg":
        sentences.append("والقانون 12 يعد اللعب بطريقة تهدد سلامة المنافس لعباً خطراً.")
    elif family == "dive":
        sentences.append("والقانون 12 يعد محاولة خداع الحكم سلوكاً غير رياضي.")
    else:
        sentences.append("والقانون 12 يعد هذا الفعل خطأً يستوجب ركلة حرة مباشرة إذا ارتُكب "
                         "بإهمال أو تهور أو قوة مفرطة.")

    card = contract.card
    maybe = "على الأرجح " if _weak(card) else ""
    if card is not None and card.label == "card":
        sentences.append(f"وبما أن الواقعة {maybe}تستوجب بطاقة، فإن القانون يضعها في درجة التهور على الأقل.")
    elif card is not None and card.label == "no_card":
        sentences.append(f"وبما أن الواقعة {maybe}لا تستوجب بطاقة، فإنها لا تتجاوز درجة الإهمال.")
    return "لماذا تُعد مخالفة", " ".join(sentences)


def build_ruling(contract: HakamContract) -> Ruling:
    family = action_family(contract)
    restart, restart_ids = _restart(contract, family)
    disciplinary, disc_ids = _disciplinary(contract)
    why_title, why = _why(contract, family)
    ids = list(dict.fromkeys(restart_ids + disc_ids))
    return Ruling(restart=restart, disciplinary=disciplinary, why=why,
                  article_ids=ids, why_title=why_title)
