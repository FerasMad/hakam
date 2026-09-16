from __future__ import annotations

from pathlib import Path

import backend.pipeline as pipeline_module
from backend.pipeline import AnalysisPipeline
from src.contract import HakamContract, Prediction


class StaticPredictor:
    def __init__(self, contract: HakamContract):
        self.contract = contract
        self.paths = None

    def predict(self, paths):
        self.paths = paths
        return self.contract


def _contract(confidence: float) -> HakamContract:
    return HakamContract(
        action_id="test-incident",
        offence=Prediction("offence", confidence),
        card=Prediction("card", 0.73),
        card_colour=None,
        attributes={
            "action_class": Prediction("tackle", 0.82),
            "body_part": Prediction("under_body", 0.76),
        },
        scene=None,
        model_version="hakam-final",
        num_views=1,
    )


def test_abstention_skips_retrieval_and_language_generation(monkeypatch):
    predictor = StaticPredictor(_contract(0.59))

    def forbidden(_contract):
        raise AssertionError("explanation path must not run for abstention")

    monkeypatch.setattr(pipeline_module, "_grounded_explanation", forbidden)
    result = AnalysisPipeline(predictor).analyze(Path("incident.mp4"))

    assert result.status == "abstained"
    assert result.threshold == 0.6
    assert result.messageAr == "الثقة منخفضة — هذه الحالة تحتاج مراجعة بشرية."
    assert predictor.paths == [Path("incident.mp4")]
    assert result.contract.card_colour is None


def test_non_abstention_uses_real_deterministic_ruling_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("HAKAM_RETRIEVAL_MODE", "tags")
    result = AnalysisPipeline(StaticPredictor(_contract(0.91))).analyze(
        Path("incident.mp4")
    )

    assert result.status == "completed"
    assert result.explanation.mode == "safe_fallback"
    assert result.explanation.model == "deterministic-v3"
    assert result.explanation.promptVersion == "v3"
    assert result.explanation.articles
    assert result.explanation.sections.whyTitle == "لماذا تُعد مخالفة"
    assert result.contract.card_colour is None


def test_language_generation_error_preserves_safe_fallback(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-never-sent")
    monkeypatch.setenv("HAKAM_RETRIEVAL_MODE", "tags")

    def failed_explain(*_args, **_kwargs):
        raise RuntimeError("simulated provider failure")

    monkeypatch.setattr(pipeline_module, "explain", failed_explain)
    result = AnalysisPipeline(StaticPredictor(_contract(0.91))).analyze(
        Path("incident.mp4")
    )

    assert result.status == "completed"
    assert result.explanation.mode == "safe_fallback"
    assert result.explanation.model == "deterministic-v3"
