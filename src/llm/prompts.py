"""The two prompt iterations required by the capstone brief."""

from __future__ import annotations

import json

from src.config import CONFIDENCE_THRESHOLD
from src.contract import HakamContract


PROMPT_V1 = """You explain football refereeing decisions in clear Arabic.
Use the supplied incident contract and Laws of the Game articles to explain the
decision. Keep the answer concise and use this structure:

القرار: ...
المادة: ...
التفسير: ...
مستوى الثقة: ...
"""


PROMPT_V2 = """أنت مساعد عربي يشرح قرارات التحكيم وفق قوانين IFAB.

قواعد ملزمة:
1. استخدم فقط الحقائق الموجودة في JSON الخاص بالواقعة والنصوص القانونية المرفقة.
2. لا تصف أي تفصيل بصري غير موجود كحقل في JSON. لا تخترع سرعة أو قفزاً أو مسافة
   عن الكرة أو دقيقة المباراة أو أسماء اللاعبين أو اتجاه الحركة.
3. كل حقل وارد في low_confidence_fields يجب إما حذفه أو صياغته بتحفظ واضح مثل
   «يبدو أن» أو «على الأرجح» أو «بحسب الثقة المتاحة».
4. إذا كانت card_colour تساوي null فلا تؤكد لون بطاقة. يمكنك ذكر العقوبة القانونية
   بصورة شرطية فقط، مثل «قد يستوجب إنذاراً إذا اعتُبر التدخل متهوراً».
5. اذكر صراحة القانون والقسم من إحدى المواد المرفقة. لا تستشهد بمادة غير مرفقة.
6. القرار يأتي من JSON، أما النص القانوني فيشرح معيار القانون ولا يضيف واقعة جديدة.
7. وجود وصف داخل المادة القانونية لا يعني أنه وقع في هذه الواقعة. لا تنسب للواقعة
   كلمات مثل القوة المفرطة أو التهور أو تهديد السلامة أو اتجاه الحركة إلا إذا وردت
   صراحة كقيمة في JSON. استخدم بدلاً من ذلك: «يفرّق القانون بين ...» أو «قد ... إذا».
8. القرار في السطر الأول يجب أن يكون ترجمة موجزة لقيم offence وcard وcard_colour فقط.
   لا تضع تفسيراً قانونياً أو تفاصيل حركة في سطر القرار.
9. افهم القيم حرفيًا: offence تعني «مخالفة»، no_offence تعني «لا مخالفة»،
   card تعني «تستوجب بطاقة»، وno_card تعني «لا تستوجب بطاقة». قيمة null تعني
   «غير محدد» ولا تجعل الحقول الأخرى احتمالية.
10. في مستوى الثقة اذكر نسبة offence.confidence بدقة. لا تقل إن الثقة منخفضة ما
    دام offence غير موجود في low_confidence_fields.
11. لا تنسخ نصوص المواد حرفياً ولا تستخدم الإنجليزية ما دام لها مقابل عربي واضح.
12. أعد النص العربي فقط، دون Markdown أو مقدمة، وبهذه الأسطر الأربعة بالضبط:
13. إذا كانت offence هي no_offence فلا تذكر بطاقة أو إنذاراً أو طرداً، ولا تقل
    «مخالفة» خارج العبارة المنفية «لا توجد مخالفة». استخدم «اللعب سليم» في التفسير.
14. إذا كانت card هي no_card فلا تناقش عقوبات أو ألواناً في التفسير؛ القرار وحده
    يقول «لا تستوجب بطاقة».
15. عندما يكون لون البطاقة null، أي ذكر للإنذار أو الطرد يجب أن يكون في جملة شرطية
    تحتوي كلمة «إذا» صراحة. لا تقل «يستوجب إنذاراً» بوصفها نتيجة لهذه الواقعة.
16. لا تحوّل dont know إلى فعل محدد، ولا تستنتج منع فرصة هدف أو مكان الواقعة أو نية
    اللاعب. لا تستخدم أسماء الحقول أو قيمها الإنجليزية في النص النهائي.
17. إذا كان حقل منخفض الثقة، فكل جملة تذكر قيمته يجب أن تبدأ بـ«على الأرجح»، وإلا
    احذف ذكر هذا الحقل تماماً.
18. ستجد مسودة تفسير آمنة مبنية من العقد. حسّن أسلوبها فقط ولا تنفِ أي حقيقة فيها،
    ولا تضف سبباً أو وصفاً أو عقوبة غير موجودة فيها.

القرار: ...
المادة: القانون ... — ...
التفسير: ...
مستوى الثقة: ...
"""


_ARTICLE_FIELDS = ("id", "law", "section", "title_ar", "text_ar", "text_en")

_DECISION_AR = {
    "offence": "مخالفة",
    "no_offence": "لا توجد مخالفة",
    "card": "تستوجب بطاقة",
    "no_card": "لا تستوجب بطاقة",
    "yellow": "بطاقة صفراء",
    "red": "بطاقة حمراء",
}


_FIELD_AR = {
    "offence": "قرار المخالفة",
    "card": "قرار البطاقة",
    "card_colour": "لون البطاقة",
    "action_class": "تصنيف الفعل",
    "body_part": "موضع التلامس",
    "contact": "وجود التلامس",
    "try_to_play": "محاولة لعب الكرة",
    "touch_ball": "لمس الكرة",
}

_LABEL_AR = {
    "tackle": "تدخل على المنافس",
    "hands": "مخالفة باليدين",
    "tackling": "تدخل على الكرة",
    "standing tackling": "تدخل من وضع الوقوف",
    "high leg": "رفع القدم عالياً",
    "holding": "مسك المنافس",
    "pushing": "دفع المنافس",
    "elbowing": "استخدام المرفق",
    "challenge": "منافسة على الكرة",
    "dive": "سقوط تمثيلي",
    "upper body": "الجزء العلوي من الجسم",
    "under body": "الجزء السفلي من الجسم",
    "with contact": "وجود تلامس",
    "without contact": "عدم وجود تلامس",
}


def arabic_label(label: str) -> str | None:
    """Return the single canonical Arabic wording used in safe explanations."""

    return _LABEL_AR.get(str(label).replace("_", " "))


def grounded_lines(contract: HakamContract, articles: list[dict]) -> tuple[str, str, str]:
    """Return deterministic decision, citation and confidence lines."""

    def decision_text(label: str, confidence: float) -> str:
        text = _DECISION_AR.get(label, label)
        return f"على الأرجح {text}" if confidence < CONFIDENCE_THRESHOLD else text

    parts = [decision_text(contract.offence.label, contract.offence.confidence)]
    if contract.card is not None:
        parts.append(decision_text(contract.card.label, contract.card.confidence))
    if contract.card_colour is not None:
        parts.append(
            decision_text(contract.card_colour.label, contract.card_colour.confidence)
        )
    elif contract.card is not None and contract.card.label == "card":
        parts.append("لون البطاقة غير محدد")

    first = articles[0] if articles else {}
    raw_law = str(first.get("law", "Law 12"))
    law = "المسرد" if raw_law.lower() == "glossary" else raw_law.replace("Law ", "القانون ")
    material = f"{law} — {first.get('title_ar') or 'المادة المسترجعة'}"
    confidence = f"ثقة قرار المخالفة: {contract.offence.confidence:.0%}"
    weak = contract.low_confidence_fields()
    if weak:
        confidence += "؛ مواضع التحفظ: " + "، ".join(
            _FIELD_AR.get(field, "حقل غير محدد") for field in weak
        )
    return " — ".join(parts), material, confidence


def grounded_explanation_draft(contract: HakamContract, articles: list[dict]) -> str:
    """Build an informative ceiling on what the model may say about the case."""

    if contract.offence.label == "no_offence":
        return (
            "يحدد العقد أن اللعب سليم ولا توجد مخالفة. وتوضح المادة المسترجعة "
            "المعيار القانوني لهذا القرار دون إضافة وقائع أخرى."
        )

    sentences = ["يحدد العقد وجود مخالفة."]
    action = contract.attributes.get("action_class")
    if action is not None and action.label != "dont know":
        action_ar = arabic_label(action.label)
        if action_ar:
            prefix = "على الأرجح " if action.confidence < CONFIDENCE_THRESHOLD else ""
            sentences.append(f"{prefix}تصنيف الفعل هو {action_ar}.")
    contact = contract.attributes.get("contact")
    if contact is not None:
        contact_ar = arabic_label(contact.label)
        if contact_ar:
            prefix = "على الأرجح " if contact.confidence < CONFIDENCE_THRESHOLD else ""
            sentences.append(f"{prefix}يذكر العقد {contact_ar}.")

    if contract.card is None:
        pass
    elif contract.card.label == "no_card":
        prefix = "على الأرجح " if contract.card.confidence < CONFIDENCE_THRESHOLD else ""
        sentences.append(f"{prefix}القرار لا يستوجب بطاقة.")
    else:
        prefix = "على الأرجح " if contract.card.confidence < CONFIDENCE_THRESHOLD else ""
        sentences.append(f"{prefix}القرار يستوجب بطاقة.")
        if contract.card_colour is None:
            sentences.append("لا يحدد العقد اللون.")
        else:
            colour = _DECISION_AR.get(contract.card_colour.label, "لون محدد")
            colour_prefix = (
                "على الأرجح "
                if contract.card_colour.confidence < CONFIDENCE_THRESHOLD
                else ""
            )
            sentences.append(f"{colour_prefix}يحدد العقد {colour}.")

    sentences.append(
        "وتوضح المادة المسترجعة المعيار القانوني المرتبط بالقرار دون إضافة وقائع أخرى."
    )
    return " ".join(sentences)


def _fixed_lines(contract: HakamContract, articles: list[dict]) -> str:
    decision, material, confidence = grounded_lines(contract, articles)
    return (
        "انسخ سطر القرار وسطر المادة وسطر مستوى الثقة التاليين حرفياً، واكتب فقط "
        "محتوى سطر التفسير:\n"
        f"القرار: {decision}\n"
        f"المادة: {material}\n"
        f"مستوى الثقة: {confidence}"
    )


def build_prompt(
    contract: HakamContract,
    articles: list[dict],
    prompt_version: str = "v2",
) -> tuple[str, str]:
    """Return ``(system_instructions, user_message)`` for one generation."""

    prompts = {"v1": PROMPT_V1, "v2": PROMPT_V2}
    if prompt_version not in prompts:
        raise ValueError("prompt_version must be 'v1' or 'v2'")
    clean_articles = [
        {key: article.get(key) for key in _ARTICLE_FIELDS}
        for article in articles
    ]
    user_message = (
        "INCIDENT CONTRACT (the complete incident information):\n"
        f"{contract.to_json()}\n\n"
        "RETRIEVED IFAB ARTICLES:\n"
        f"{json.dumps(clean_articles, ensure_ascii=False, indent=2)}"
    )
    if prompt_version == "v2":
        user_message += (
            "\n\nMANDATORY GROUNDED LINES:\n"
            + _fixed_lines(contract, articles)
            + "\n\nSAFE ARABIC EXPLANATION DRAFT — DO NOT CHANGE ITS FACTS:\n"
            + "التفسير: "
            + grounded_explanation_draft(contract, articles)
        )
    return prompts[prompt_version], user_message
