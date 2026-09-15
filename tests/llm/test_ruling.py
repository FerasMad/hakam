from types import SimpleNamespace

import openai
import pytest

from src.contract import HakamContract, Prediction as P, mock_contract
from src.llm.faithfulness import score
from src.llm.generate import _repair_reasons, explain, safe_v3
from src.llm.ruling import build_ruling


@pytest.fixture(autouse=True)
def tag_only_retrieval(monkeypatch):
    monkeypatch.setenv("HAKAM_RETRIEVAL_MODE", "tags")


def _contract(offence="offence", card="card", action="tackle", body="under_body", conf=0.8):
    attrs = {}
    if action:
        attrs["action_class"] = P(action, conf)
    if body:
        attrs["body_part"] = P(body, conf)
    return HakamContract("t", P(offence, 0.8), None if card is None else P(card, conf), None, attrs)


CASES = {
    "tackle card": _contract(),
    "hands no card": _contract(card="no_card", action="hands", body="upper_body"),
    "high leg low card": _contract(action="high leg", conf=0.55),
    "elbowing": _contract(action="elbowing", body="upper_body"),
    "no action": _contract(action=None),
    "no offence": _contract(offence="no_offence", card=None),
    "mock yellow": mock_contract(),
}


@pytest.mark.parametrize("name", list(CASES))
def test_safe_ruling_passes_every_guard(name):
    contract = CASES[name]
    text, sections, articles = safe_v3(contract)
    assert _repair_reasons(text, contract, articles, "v3") == []
    assert score(text, contract, articles)["faithfulness"] == 1.0
    assert set(sections) >= {"decision", "restart", "disciplinary", "law", "why", "confidence"}
    ids = {a["id"] for a in articles}
    assert set(build_ruling(contract).article_ids) <= ids


def test_no_offence_has_no_sanction():
    ruling = build_ruling(CASES["no offence"])
    assert "دون ركلة حرة" in ruling.restart
    assert ruling.disciplinary == "لا عقوبة انضباطية."
    assert ruling.why_title == "لماذا لا تُعد مخالفة"


def test_unknown_colour_is_stated_conditionally():
    ruling = build_ruling(CASES["tackle card"])
    assert "غير محدد" in ruling.disciplinary
    assert "إذا اعتُبر التدخل متهوراً فالعقوبة إنذار" in ruling.disciplinary


def test_penalty_area_is_conditional_because_location_is_unknown():
    assert "إذا وقعت داخل منطقة جزاء المخالف" in build_ruling(CASES["elbowing"]).restart


def test_hands_cites_both_holding_and_pushing():
    ids = build_ruling(CASES["hands no card"]).article_ids
    assert "law12-1-holding-opponent" in ids and "law12-1-pushing" in ids


def test_v3_model_only_writes_the_why_line(monkeypatch):
    captured = {}

    class FakeResponses:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                output_text="لماذا تُعد مخالفة: صُنّف الفعل تدخل على المنافس، والقانون 12 يعده خطأً."
            )

    monkeypatch.setattr(openai, "OpenAI", lambda *a, **k: SimpleNamespace(responses=FakeResponses()))
    contract = CASES["tackle card"]
    result = explain(contract)

    assert result.prompt_version == "v3"
    assert result.sections["why"].startswith("صُنّف الفعل تدخل على المنافس")
    assert result.sections["restart"] == build_ruling(contract).restart
    assert len(result.text_ar.splitlines()) == 6
    assert "video" not in captured["input"].lower()


def test_v3_falls_back_to_safe_ruling_when_the_model_invents(monkeypatch):
    class FakeResponses:
        def create(self, **kwargs):
            return SimpleNamespace(output_text="لماذا تُعد مخالفة: وقعت داخل منطقة الجزاء بسرعة عالية.")

    monkeypatch.setattr(openai, "OpenAI", lambda *a, **k: SimpleNamespace(responses=FakeResponses()))
    contract = CASES["tackle card"]
    result = explain(contract)
    assert result.sections["why"] == build_ruling(contract).why


def test_v3_abstains_without_api(monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError("no API call while abstaining")

    monkeypatch.setattr(openai, "OpenAI", forbidden)
    contract = CASES["tackle card"]
    contract.offence.confidence = 0.4
    result = explain(contract)
    assert result.abstained and result.sections is None
