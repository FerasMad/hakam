from src.contract import Prediction, mock_contract
from src.llm.faithfulness import score
from src.llm.prompts import (
    build_prompt,
    grounded_explanation_draft,
    grounded_lines,
)


def test_v2_prompt_contains_grounding_and_exact_contract_lines():
    contract = mock_contract(colour=None)
    articles = [
        {
            "id": "law12-card-link",
            "law": "Law 12",
            "section": "Disciplinary action",
            "title_ar": "ربط العقوبة بدرجة الخطورة",
            "text_ar": "نص قانوني عربي.",
        }
    ]

    instructions, user_message = build_prompt(contract, articles, "v2")

    assert "لا تخترع" in instructions
    assert "لون البطاقة غير محدد" in user_message
    assert "ثقة قرار المخالفة: 88%" in user_message
    assert "القانون 12" in user_message


def test_v1_and_v2_are_distinct():
    contract = mock_contract()
    article = {
        "id": "law12",
        "law": "Law 12",
        "section": "Fouls",
        "title_ar": "المخالفات",
        "text_ar": "نص.",
    }

    assert build_prompt(contract, [article], "v1") != build_prompt(contract, [article], "v2")


def test_safe_draft_translates_high_leg_without_inventing_high_risk():
    contract = mock_contract(colour=None)
    contract.attributes["action_class"] = Prediction("high leg", 0.81)
    article = {"title_ar": "اللعب الخطر مع تلامس"}

    draft = grounded_explanation_draft(contract, [article])

    assert "رفع القدم عالياً" in draft
    assert "عالي الخطورة" not in draft


def test_safe_draft_hedges_low_action_and_card_confidence():
    article = {
        "law": "Law 12",
        "section": "Direct free kick",
        "title_ar": "الضرب أو استخدام المرفق",
    }
    contracts = []

    low_action = mock_contract(colour=None)
    low_action.attributes["action_class"] = Prediction("elbowing", 0.48)
    contracts.append(low_action)

    low_card = mock_contract(colour=None)
    low_card.card.confidence = 0.52
    contracts.append(low_card)

    for contract in contracts:
        decision, material, confidence = grounded_lines(contract, [article])
        draft = grounded_explanation_draft(contract, [article])
        text = (
            f"القرار: {decision}\n"
            f"المادة: {material}\n"
            f"التفسير: {draft}\n"
            f"مستوى الثقة: {confidence}"
        )
        assert score(text, contract, [article])["hedged_low_conf"] is True
