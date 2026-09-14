"""Arabic surface forms for every label the contract can carry.

Without this the faithfulness check does not work at all. ``citable_values()``
in the contract returns English label strings (``"tackling"``, ``"yellow"``)
while the generated explanation is Arabic, so a naive membership test would flag
every true sentence as invented.

Two things make Arabic matching harder than a substring test:

**Orthographic variation.** ``إنذار`` and ``انذار`` are the same word with and
without hamza; ``مخالفه`` and ``مخالفة`` differ only in taa marbuta. Writers and
language models use both. ``normalise`` folds alef and yaa variants and strips
diacritics, so comparison happens on a canonical form.

**Multiple legitimate phrasings.** A yellow card is ``إنذار`` or ``بطاقة صفراء``;
a tackle is ``تدخل`` or ``التحام``. Each label therefore maps to a set, and a
claim counts as supported when any form appears.

The vocabulary mirrors src/data/labels.py. Adding a label there without adding
it here silently weakens the metric, so ``missing_labels()`` exists to catch it.
"""

from __future__ import annotations

import re
import unicodedata

_ALEF = re.compile("[آأإٱ]")     # آ أ إ ٱ
_YAA = re.compile("ى")                          # ى
_TAA_MARBUTA = re.compile("ة")                  # ة
_TATWEEL = re.compile("ـ")                      # ـ
_DIACRITICS = re.compile("[ً-ْٰ]")    # harakat
_NON_WORD = re.compile(r"[^\w\s؀-ۿ]")


def normalise(text: str) -> str:
    """Canonical form for comparison: folded alef/yaa/taa, no diacritics.

    Not for display - only for deciding whether two spellings are the same word.
    """
    text = unicodedata.normalize("NFKC", text)
    text = _DIACRITICS.sub("", text)
    text = _TATWEEL.sub("", text)
    text = _ALEF.sub("ا", text)         # -> ا
    text = _YAA.sub("ي", text)          # -> ي
    text = _TAA_MARBUTA.sub("ه", text)  # -> ه
    text = _NON_WORD.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


# Label value -> the ways a referee or commentator would actually say it.
LEXICON: dict[str, set[str]] = {
    # ---- stage 1: offence ------------------------------------------------
    "offence": {"مخالفة", "خطأ", "لعب خطأ", "مخالفة للقانون"},
    "no_offence": {"لا توجد مخالفة", "ليست مخالفة", "لا مخالفة", "لعب سليم",
                   "تدخل قانوني", "بدون مخالفة"},
    "between": {"غير حاسم", "غير واضح", "محل شك"},

    # ---- stage 2: card ---------------------------------------------------
    "card": {"بطاقة", "عقوبة", "إجراء تأديبي"},
    "no_card": {"بدون بطاقة", "لا تستوجب بطاقة", "دون عقوبة", "بدون عقوبة"},

    # ---- stage 3: colour (cut from the model, still discussed by the LLM)
    "yellow": {"إنذار", "بطاقة صفراء", "الإنذار"},
    "red": {"طرد", "بطاقة حمراء", "الطرد"},

    # ---- action class ----------------------------------------------------
    "tackling": {"تدخل", "التحام", "تدخل على الكرة"},
    "standing tackling": {"تدخل من وضع الوقوف", "تدخل واقف", "تدخل دون انزلاق"},
    "high leg": {"رفع القدم عالياً", "قدم مرفوعة", "لعب خطير بالقدم المرفوعة"},
    "holding": {"إعاقة", "مسك", "شد", "إمساك الخصم"},
    "pushing": {"دفع", "دفع الخصم"},
    "elbowing": {"ضرب بالمرفق", "استخدام المرفق", "مرفق"},
    "challenge": {"التحام", "منازعة", "صراع على الكرة"},
    "dive": {"سقوط تمثيلي", "خداع", "تمثيل", "غطس"},
    "dont know": {"غير محدد", "غير معروف"},

    # ---- body part -------------------------------------------------------
    "upper body": {"الجزء العلوي من الجسم", "الجذع", "أعلى الجسم"},
    "under body": {"الجزء السفلي من الجسم", "الأطراف السفلية", "أسفل الجسم",
                   "الساق", "القدم"},

    # ---- contact ---------------------------------------------------------
    "with contact": {"بتماس", "مع تماس", "احتكاك", "تماس مباشر"},
    "without contact": {"بدون تماس", "دون احتكاك", "لا يوجد تماس"},

    # ---- try to play / touch ball ---------------------------------------
    # The contract emits these attributes as bare yes/no, so they are keyed
    # that way. Separate try_to_play_* / touch_ball_* keys duplicated the same
    # phrases, and a phrase matched under a key the contract never emits was
    # scored as an unsupported claim.
    "yes": {"محاولة لعب الكرة", "حاول لعب الكرة", "قصد الكرة",
            "لمس الكرة", "مس الكرة", "وصل إلى الكرة"},
    "no": {"دون محاولة لعب الكرة", "لم يحاول لعب الكرة",
           "لم يلمس الكرة", "دون لمس الكرة"},

    # ---- abstention ------------------------------------------------------
    "refer": {"يحتاج مراجعة", "مراجعة بشرية", "غير مؤكد", "يحال إلى الحكم"},
}

def _key(label: str) -> str:
    """One spelling for lookups.

    The contract emits underscored values (``with_contact``) while some keys
    here are written with spaces. Both are folded so a key can never become
    unreachable through a punctuation mismatch.
    """
    return str(label).strip().lower().replace("_", " ")


# Pre-normalised once, under folded keys; matching is hot when scoring many
# explanations.
_NORMALISED: dict[str, set[str]] = {
    _key(label): {normalise(form) for form in forms}
    for label, forms in LEXICON.items()
}


def arabic_forms(label: str) -> set[str]:
    """Every accepted Arabic spelling of one label value."""
    for key, forms in LEXICON.items():
        if _key(key) == _key(label):
            return forms
    return set()


_PREFIXES = ("وال", "بال", "فال", "ال", "و", "ب", "ف", "ل")
_SHORT = 3


def _strip_prefix(token: str) -> str:
    for pre in _PREFIXES:
        if token.startswith(pre) and len(token) - len(pre) >= 2:
            return token[len(pre):]
    return token


# Every (form, label) pair, longest first. Longer phrases are matched and masked
# before shorter ones, so "لا تستوجب بطاقة" counts as no_card only - not also as
# card via the "بطاقة" inside it.
_FORMS: list[tuple[str, str]] = sorted(
    ((form, label) for label, forms in _NORMALISED.items() for form in forms if form),
    key=lambda pair: -len(pair[0]),
)


def found_labels(text: str) -> set[str]:
    """Every label the text asserts. Used by the faithfulness check.

    Two rules keep this from over-matching:
    - longest phrase wins, and its span is masked before shorter forms are tried;
    - forms of three letters or fewer must match a whole word (after stripping
      prefixes like ال / و / ب), otherwise "لا" would match inside "الأخطاء".
    """
    haystack = normalise(text)
    found: set[str] = set()
    for form, label in _FORMS:
        if len(form) > _SHORT:
            if form in haystack:
                found.add(label)
                haystack = haystack.replace(form, " " * len(form))
        else:
            tokens = haystack.split(" ")
            hit = False
            for i, tok in enumerate(tokens):
                if tok and (tok == form or _strip_prefix(tok) == form):
                    tokens[i] = " " * len(tok)
                    hit = True
            if hit:
                found.add(label)
                haystack = " ".join(tokens)
    return found


def mentions(text: str, label: str) -> bool:
    """True when ``text`` states ``label`` in any accepted Arabic form."""
    return _key(label) in found_labels(text)


def missing_labels(values) -> set[str]:
    """Label values with no Arabic entry.

    A label the lexicon does not know is invisible to the faithfulness check,
    which would quietly overstate the score. Call this with the contract's
    vocabulary in a test so the gap fails loudly instead.
    """
    return {v for v in values if _key(v) not in _NORMALISED}
