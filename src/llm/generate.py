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
    render_v3,
    ruling_sections,
)
from src.llm.retrieve import retrieve
from src.llm.faithfulness import score


_NON_ARABIC_SCRIPT = re.compile(r"[A-Za-z֐-׿]")
_REQUIRED_PREFIXES = {
    "v2": ("القرار:", "المادة:", "التفسير:", "مستوى الثقة:"),
    "v3": ("القرار:", "العقوبة الفنية:", "العقوبة الانضباطية:", "المادة:", "لماذا", "مستوى الثقة:"),
}
V3_ARTICLES = 5


@dataclass
class Explanation:
    text_ar: str
    articles: list[dict]
    prompt_version: str
    abstained: bool
    model: str
    sections: dict | None = None


def _extract_explanation(text: str) -> str:
    """The free-text line: «التفسير» in v2, «لماذا ...» in v3."""
    for raw_line in text.splitlines():
        match = re.match(r"^(?:التفسير|لماذا[^:]*)\s*:\s*(.*)$", raw_line.strip())
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


def v3_articles(contract: HakamContract, k: int = V3_ARTICLES) -> list[dict]:
    """Every article the ruling rests on, topped up with the best retrieved ones to ``k``."""
    from src.llm.retrieve import _load_corpus
    from src.llm.ruling import build_ruling

    corpus = {chunk["id"]: dict(chunk) for chunk in _load_corpus()}
    articles = [corpus[i] for i in build_ruling(contract).article_ids if i in corpus]
    seen = {a["id"] for a in articles}
    for chunk in retrieve(contract, k=k):
        if len(articles) >= k:
            break
        if chunk["id"] not in seen:
            articles.append(chunk)
            seen.add(chunk["id"])
    return articles


def safe_v3(contract: HakamContract, articles: list[dict] | None = None) -> tuple[str, dict, list[dict]]:
    """The full ruling with the deterministic «why» line. Needs no API."""
    articles = v3_articles(contract) if articles is None else articles
    sections = ruling_sections(contract, articles)
    return render_v3(sections), sections, articles


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


def _repair_reasons(
    text_ar: str, contract: HakamContract, articles: list[dict], prompt_version: str = "v2"
) -> list[str]:
    metrics = score(text_ar, contract, articles)
    reasons = []
    if metrics["unsupported"]:
        reasons.append("ادعاءات غير موجودة في العقد: " + "، ".join(metrics["unsupported"]))
    if not metrics["cites_article"]:
        reasons.append("لا يوجد استشهاد صريح بالقانون المسترجع")
    if not metrics["hedged_low_conf"]:
        reasons.append("حقل منخفض الثقة ذُكر دون عبارة تحفظ")
    prefixes = _REQUIRED_PREFIXES[prompt_version]
    lines = [line.strip() for line in text_ar.splitlines() if line.strip()]
    if len(lines) != len(prefixes) or any(
        not line.startswith(prefix) for line, prefix in zip(lines, prefixes)
    ):
        reasons.append(f"البنية ليست {len(prefixes)} أسطر مطابقة")
    elif not _extract_explanation(text_ar):
        reasons.append("سطر التفسير فارغ")
    if _NON_ARABIC_SCRIPT.search(text_ar):
        reasons.append("النص يحتوي حروفاً إنجليزية أو عبرية")
    reasons.extend(_semantic_risk_reasons(text_ar, contract))
    return reasons


def explain(contract: HakamContract, prompt_version: str = "v3") -> Explanation:
    """Explain one contract, or abstain before constructing an API client.

    v3 (default) is the full ruling shown after the clip: decision, restart,
    disciplinary sanction, the quoted Law, why it is (or is not) a violation, and
    confidence. Only the «why» line is written by the model; the rest is rendered
    from the contract and Law 12, so the sanction can never be invented.
    """

    if prompt_version not in {"v1", "v2", "v3"}:
        raise ValueError("prompt_version must be 'v1', 'v2' or 'v3'")
    if contract.should_abstain():
        return Explanation(
            text_ar=ABSTAIN_MESSAGE_AR,
            articles=[],
            prompt_version=prompt_version,
            abstained=True,
            model=LLM_MODEL,
        )

    articles = v3_articles(contract) if prompt_version == "v3" else retrieve(contract)
    instructions, user_message = build_prompt(contract, articles, prompt_version)

    # Imported only after the abstention gate, which makes it impossible for a
    # low-confidence incident to initialise or call the external API.
    from openai import OpenAI

    client = OpenAI()
    text_ar = ""
    sections = None
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
        if prompt_version == "v1":
            text_ar = raw_text
            break
        if prompt_version == "v3":
            why = _extract_explanation(raw_text) or raw_text.splitlines()[0].split(":", 1)[-1]
            sections = ruling_sections(contract, articles, why)
            text_ar = render_v3(sections)
        else:
            text_ar = _enforce_v2_structure(raw_text, contract, articles)
        reasons = _repair_reasons(text_ar, contract, articles, prompt_version)
        if not reasons:
            break
        if attempt < 2:
            request_input = (
                user_message
                + "\n\nالمحاولة السابقة غير مقبولة:\n"
                + text_ar
                + "\n\nأسباب الرفض:\n- "
                + "\n- ".join(reasons)
                + "\nأعد الكتابة من الصفر، وتجنب الكلمات التي سببت "
                "ادعاءات غير مدعومة. لا تضف أي معلومة جديدة."
            )
    if reasons and prompt_version == "v2":
        text_ar = _safe_v2_fallback(contract, articles)
    elif reasons and prompt_version == "v3":
        text_ar, sections, articles = safe_v3(contract, articles)
    return Explanation(
        text_ar=text_ar,
        articles=articles,
        prompt_version=prompt_version,
        abstained=False,
        model=LLM_MODEL,
        sections=sections,
    )
