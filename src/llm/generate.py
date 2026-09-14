"""Generate a grounded Arabic explanation from a CV contract and IFAB rules."""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.config import ABSTAIN_MESSAGE_AR, LLM_MAX_TOKENS, LLM_MODEL
from src.contract import HakamContract
from src.llm.lexicon import normalise
from src.llm.prompts import (
    arabic_label,
    build_prompt,
    grounded_explanation_draft,
    grounded_lines,
)
from src.llm.retrieve import retrieve
from src.llm.faithfulness import score


_NON_ARABIC_SCRIPT = re.compile(r"[A-Za-z\u0590-\u05ff]")
_REQUIRED_PREFIXES = ("القرار:", "المادة:", "التفسير:", "مستوى الثقة:")


@dataclass
class Explanation:
    text_ar: str
    articles: list[dict]
    prompt_version: str
    abstained: bool
    model: str


def _extract_explanation(text: str) -> str:
    for raw_line in text.splitlines():
        match = re.match(r"^التفسير\s*:\s*(.*)$", raw_line.strip())
        if match:
            return match.group(1).strip()
    return ""


def _enforce_v2_structure(
    raw_text: str,
    contract: HakamContract,
    articles: list[dict],
) -> str:
    """Keep the generated explanation but render factual lines from evidence."""

    explanation = _extract_explanation(raw_text)
    decision, material, confidence = grounded_lines(contract, articles)
    return (
        f"القرار: {decision}\n"
        f"المادة: {material}\n"
        f"التفسير: {explanation}\n"
        f"مستوى الثقة: {confidence}"
    )


def _safe_v2_fallback(contract: HakamContract, articles: list[dict]) -> str:
    """Return a conservative Arabic explanation after failed model repairs."""

    decision, material, confidence = grounded_lines(contract, articles)
    explanation = grounded_explanation_draft(contract, articles)
    return (
        f"القرار: {decision}\n"
        f"المادة: {material}\n"
        f"التفسير: {explanation}\n"
        f"مستوى الثقة: {confidence}"
    )


def _semantic_risk_reasons(text_ar: str, contract: HakamContract) -> list[str]:
    """Catch high-impact meanings outside the label-based scorer."""

    explanation = normalise(_extract_explanation(text_ar))
    reasons = []
    risky = {
        "عالي الخطوره": "فسر رفع القدم على انه ارتفاع في الخطوره",
        "تسلل": "اضاف تسللا غير موجود في العقد",
        "فرصه هدف": "اضاف فرصه هدف غير موجوده في العقد",
        "هجوم واعد": "اضاف هجوما واعدا غير موجود في العقد",
        "منطقه الجزاء": "اضاف مكان الواقعه",
        "ركله جزاء": "اضاف مكانا او استئنافا لا يثبته العقد",
        "سرعه": "اضاف سرعه غير موجوده في العقد",
        "نيه اللاعب": "اضاف نيه غير موجوده في العقد",
    }
    reasons.extend(message for phrase, message in risky.items() if phrase in explanation)

    action = contract.attributes.get("action_class")
    if action is not None and action.label != "dont know":
        expected_action = arabic_label(action.label)
        if expected_action and normalise(expected_action) not in explanation:
            reasons.append("لم يستخدم الصياغه العربيه القياسيه لنوع الفعل")
    if action is not None and action.label == "dive" and re.search(
        r"(?:لا|ليس|لم|دون|غير).{0,30}(?:خداع|تمثيل)", explanation
    ):
        reasons.append("ناقض تصنيف السقوط التمثيلي الموجود في العقد")

    if contract.offence.label == "no_offence" and any(
        phrase in explanation for phrase in ("تهور", "متهور", "قوه مفرطه")
    ):
        reasons.append("استنتج درجه خطوره غير موجوده في عقد عدم المخالفه")

    if contract.card_colour is not None and re.search(
        r"(?:الواقعه|التدخل|الفعل).{0,24}(?:متهور|قوه مفرطه)", explanation
    ):
        reasons.append("استنتج سبب العقوبه من لون البطاقه")
    return reasons


def _repair_reasons(text_ar: str, contract: HakamContract, articles: list[dict]) -> list[str]:
    metrics = score(text_ar, contract, articles)
    reasons = []
    if metrics["unsupported"]:
        reasons.append("ادعاءات غير موجودة في العقد: " + "، ".join(metrics["unsupported"]))
    if not metrics["cites_article"]:
        reasons.append("لا يوجد استشهاد صريح بالقانون المسترجع")
    if not metrics["hedged_low_conf"]:
        reasons.append("حقل منخفض الثقة ذُكر دون عبارة تحفظ")
    lines = [line.strip() for line in text_ar.splitlines() if line.strip()]
    if len(lines) != 4 or any(
        not line.startswith(prefix) for line, prefix in zip(lines, _REQUIRED_PREFIXES)
    ):
        reasons.append("البنية ليست أربعة أسطر مطابقة")
    elif not lines[2].removeprefix("التفسير:").strip():
        reasons.append("سطر التفسير فارغ")
    if _NON_ARABIC_SCRIPT.search(text_ar):
        reasons.append("النص يحتوي حروفاً إنجليزية أو عبرية")
    reasons.extend(_semantic_risk_reasons(text_ar, contract))
    return reasons


def explain(contract: HakamContract, prompt_version: str = "v2") -> Explanation:
    """Explain one contract, or abstain before constructing an API client."""

    if prompt_version not in {"v1", "v2"}:
        raise ValueError("prompt_version must be 'v1' or 'v2'")
    if contract.should_abstain():
        return Explanation(
            text_ar=ABSTAIN_MESSAGE_AR,
            articles=[],
            prompt_version=prompt_version,
            abstained=True,
            model=LLM_MODEL,
        )

    articles = retrieve(contract)
    instructions, user_message = build_prompt(contract, articles, prompt_version)

    # Imported only after the abstention gate, which makes it impossible for a
    # low-confidence incident to initialise or call the external API.
    from openai import OpenAI

    client = OpenAI()
    text_ar = ""
    request_input = user_message
    reasons: list[str] = []
    for attempt in range(3):
        response = client.responses.create(
            model=LLM_MODEL,
            instructions=instructions,
            input=request_input,
            reasoning={"effort": "minimal"},
            max_output_tokens=LLM_MAX_TOKENS,
            store=False,
        )
        raw_text = (response.output_text or "").strip()
        if not raw_text:
            raise RuntimeError("OpenAI returned an empty explanation")
        if prompt_version != "v2":
            text_ar = raw_text
            break
        text_ar = _enforce_v2_structure(raw_text, contract, articles)
        reasons = _repair_reasons(text_ar, contract, articles)
        if not reasons:
            break
        if attempt < 2:
            request_input = (
                user_message
                + "\n\nالمحاولة السابقة غير مقبولة:\n"
                + text_ar
                + "\n\nأسباب الرفض:\n- "
                + "\n- ".join(reasons)
                + "\nأعد كتابة الأسطر الأربعة من الصفر، وتجنب الكلمات التي سببت "
                "ادعاءات غير مدعومة. لا تضف أي معلومة جديدة."
            )
    if prompt_version == "v2" and reasons:
        text_ar = _safe_v2_fallback(contract, articles)
    return Explanation(
        text_ar=text_ar,
        articles=articles,
        prompt_version=prompt_version,
        abstained=False,
        model=LLM_MODEL,
    )
