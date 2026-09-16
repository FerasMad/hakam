from types import SimpleNamespace

import openai

from src.config import ABSTAIN_MESSAGE_AR
from src.contract import Prediction, mock_contract
from src.llm.generate import _semantic_risk_reasons, explain


def test_abstention_returns_fixed_arabic_message_without_api_call(monkeypatch):
    calls = []

    def forbidden_client(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("The API client must not be created while abstaining")

    monkeypatch.setattr(openai, "OpenAI", forbidden_client)
    contract = mock_contract()
    contract.offence.confidence = 0.49

    result = explain(contract)

    assert result.abstained is True
    assert result.text_ar == ABSTAIN_MESSAGE_AR
    assert result.articles == []
    assert calls == []


def test_explain_sends_only_contract_and_retrieved_laws(monkeypatch):
    captured = {}

    class FakeResponses:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                output_text=(
                    "القرار: مخالفة — تستوجب بطاقة — لون البطاقة غير محدد\n"
                    "المادة: القانون 12 — ربط العقوبة بدرجة الخطورة\n"
                    "التفسير: القرار يثبت مخالفة مع تلامس، ولا يحدد لون البطاقة.\n"
                    "مستوى الثقة: ثقة قرار المخالفة: 88%"
                )
            )

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.responses = FakeResponses()

    articles = [
        {
            "id": "law12-card-link",
            "law": "Law 12",
            "section": "Disciplinary action",
            "title_ar": "ربط العقوبة بدرجة الخطورة",
            "text_ar": "لا يحدد التلامس وحده لون البطاقة.",
            "text_en": "Contact alone does not determine card colour.",
            "tags": ["card", "with_contact"],
            "source_pages": {"ar": [105], "en": [108]},
        }
    ]
    monkeypatch.setattr(openai, "OpenAI", FakeClient)
    monkeypatch.setattr("src.llm.generate.retrieve", lambda contract: articles)

    result = explain(mock_contract(colour=None), prompt_version="v2")

    assert result.abstained is False
    assert result.articles == articles
    assert "video" not in captured["input"].lower()
    assert "tackling" in captured["input"]
    assert "law12-card-link" in captured["input"]
    assert captured["store"] is False


def test_v2_repairs_an_unsupported_first_attempt(monkeypatch):
    outputs = iter(
        [
            (
                "القرار: مخالفة — تستوجب بطاقة — لون البطاقة غير محدد\n"
                "المادة: القانون 12 — العقوبة\n"
                "التفسير: تستوجب الواقعة بطاقة حمراء.\n"
                "مستوى الثقة: ثقة قرار المخالفة: 88%"
            ),
                (
                    "القرار: مخالفة — تستوجب بطاقة — لون البطاقة غير محدد\n"
                    "المادة: القانون 12 — العقوبة\n"
                    "التفسير: القرار يثبت وجود مخالفة، وتصنيف الفعل تدخل على الكرة، "
                    "ولا يحدد لون العقوبة.\n"
                    "مستوى الثقة: ثقة قرار المخالفة: 88%"
                ),
        ]
    )
    calls = []

    class FakeResponses:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(output_text=next(outputs))

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.responses = FakeResponses()

    article = {
        "id": "law12-card-link",
        "law": "Law 12",
        "section": "Disciplinary action",
        "title_ar": "العقوبة",
        "text_ar": "لون العقوبة يعتمد على معيار القانون.",
        "text_en": "The sanction depends on the legal criterion.",
        "tags": ["card"],
        "source_pages": {"ar": [105], "en": [108]},
    }
    monkeypatch.setattr(openai, "OpenAI", FakeClient)
    monkeypatch.setattr("src.llm.generate.retrieve", lambda contract: [article])

    result = explain(mock_contract(colour=None), prompt_version="v2")

    assert len(calls) == 2
    assert "بطاقة حمراء" not in result.text_ar
    assert "أسباب الرفض" in calls[1]["input"]
    assert [line.split(":", 1)[0] for line in result.text_ar.splitlines()] == [
        "القرار",
        "المادة",
        "التفسير",
        "مستوى الثقة",
    ]


def test_v2_uses_non_empty_safe_fallback_after_three_bad_attempts(monkeypatch):
    calls = []

    class FakeResponses:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                output_text=(
                    "القرار: مخالفة\n"
                    "المادة: القانون 12\n"
                    "التفسير : \n"
                    "مستوى الثقة: 88%"
                )
            )

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.responses = FakeResponses()

    article = {
        "id": "law12-card-link",
        "law": "Law 12",
        "section": "Disciplinary action",
        "title_ar": "العقوبة",
        "text_ar": "لون العقوبة يعتمد على معيار القانون.",
        "text_en": "The sanction depends on the legal criterion.",
        "tags": ["card"],
        "source_pages": {"ar": [105], "en": [108]},
    }
    monkeypatch.setattr(openai, "OpenAI", FakeClient)
    monkeypatch.setattr("src.llm.generate.retrieve", lambda contract: [article])

    result = explain(mock_contract(colour=None), prompt_version="v2")

    assert len(calls) == 3
    explanation_line = result.text_ar.splitlines()[2]
    assert explanation_line.startswith("التفسير:")
    assert explanation_line.removeprefix("التفسير:").strip()


def test_v2_rejects_a_semantic_contradiction_of_dive(monkeypatch):
    calls = []

    class FakeResponses:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                output_text=(
                    "القرار: مخالفة — تستوجب بطاقة\n"
                    "المادة: القانون 12 — محاولة خداع الحكم\n"
                    "التفسير: لا تشير الواقعة إلى خداع أو تمثيل.\n"
                    "مستوى الثقة: 88%"
                )
            )

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.responses = FakeResponses()

    article = {
        "id": "law12-3-simulation",
        "law": "Law 12",
        "section": "Cautionable offences",
        "title_ar": "محاولة خداع الحكم",
        "text_ar": "محاولة خداع الحكم سلوك غير رياضي.",
        "text_en": "Attempting to deceive the referee is unsporting behaviour.",
        "tags": ["offence", "card", "dive"],
        "source_pages": {"ar": [110], "en": [113]},
    }
    contract = mock_contract(colour=None)
    contract.attributes["action_class"] = Prediction("dive", 0.81)
    contract.attributes["contact"] = Prediction("without_contact", 0.90)
    monkeypatch.setattr(openai, "OpenAI", FakeClient)
    monkeypatch.setattr("src.llm.generate.retrieve", lambda candidate: [article])

    result = explain(contract, prompt_version="v2")

    assert len(calls) == 3
    assert "لا تشير الواقعة إلى خداع" not in result.text_ar
    assert "سقوط تمثيلي" in result.text_ar


def test_semantic_guard_rejects_unproved_severity_and_nonstandard_action_wording():
    fair = mock_contract(offence="no_offence", card=None, colour=None)
    fair.attributes["action_class"] = Prediction("challenge", 0.81)
    fair_text = (
        "القرار: لا توجد مخالفة\n"
        "المادة: القانون 12 — المنافسة القانونية\n"
        "التفسير: طريقة الفعل لا تشير إلى تهور أو قوة مفرطة.\n"
        "مستوى الثقة: 88%"
    )
    assert _semantic_risk_reasons(fair_text, fair)

    holding = mock_contract(card="no_card", colour=None)
    holding.attributes["action_class"] = Prediction("holding", 0.81)
    holding_text = (
        "القرار: مخالفة — لا تستوجب بطاقة\n"
        "المادة: القانون 12 — مسك المنافس\n"
        "التفسير: الواقعة تتضمن مسكنة المنافس.\n"
        "مستوى الثقة: 88%"
    )
    assert _semantic_risk_reasons(holding_text, holding)
