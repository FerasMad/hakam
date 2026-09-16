from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

from src import config
from src.llm.faithfulness import score as faithfulness_score
from src.llm.generate import explain, safe_v3

from backend.errors import InferencePipelineError, InvalidVideoError, RulingPipelineError
from backend.schemas import (
    AbstainedAnalysisPayload,
    AnalysisResponse,
    CompletedAnalysisPayload,
    ExplanationPayload,
    FaithfulnessPayload,
    HakamContractPayload,
    LawArticlePayload,
    RulingSectionsPayload,
    SourcePagesPayload,
)


logger = logging.getLogger(__name__)


def _contract_payload(contract: Any) -> HakamContractPayload:
    raw = contract.to_dict()
    if raw.get("card_colour") is not None:
        raise InferencePipelineError("production inference returned an unsupported card colour")
    if raw.get("scene") is not None:
        raise InferencePipelineError("production inference returned an unsupported scene payload")
    if raw.get("num_views") != 1:
        raise InferencePipelineError("website inference must contain exactly one view")
    return HakamContractPayload.model_validate(raw)


def _article_payload(article: dict) -> LawArticlePayload:
    pages = article.get("source_pages")
    return LawArticlePayload(
        id=str(article.get("id", "")),
        law=str(article.get("law", "")),
        section=str(article.get("section", "")),
        titleAr=str(article.get("title_ar", "")),
        titleEn=article.get("title_en"),
        textAr=str(article.get("text_ar", "")),
        textEn=article.get("text_en"),
        sourcePages=SourcePagesPayload.model_validate(pages) if isinstance(pages, dict) else None,
    )


def _sections_payload(sections: dict) -> RulingSectionsPayload:
    return RulingSectionsPayload(
        decision=sections["decision"],
        restart=sections["restart"],
        disciplinary=sections["disciplinary"],
        law=sections["law"],
        whyTitle=sections["why_title"],
        why=sections["why"],
        confidence=sections["confidence"],
    )


def _faithfulness_payload(text_ar: str, contract: Any, articles: list[dict]) -> FaithfulnessPayload:
    metrics = faithfulness_score(text_ar, contract, articles)
    return FaithfulnessPayload(
        faithfulness=metrics["faithfulness"],
        citesArticle=metrics["cites_article"],
        hedgedLowConfidence=metrics["hedged_low_conf"],
        unsupported=metrics["unsupported"],
    )


def _safe_explanation(contract: Any) -> ExplanationPayload:
    text_ar, sections, articles = safe_v3(contract)
    return ExplanationPayload(
        textAr=text_ar,
        sections=_sections_payload(sections),
        articles=[_article_payload(article) for article in articles],
        promptVersion="v3",
        model="deterministic-v3",
        mode="safe_fallback",
        faithfulness=_faithfulness_payload(text_ar, contract, articles),
    )


def _grounded_explanation(contract: Any) -> ExplanationPayload:
    if not os.getenv("OPENAI_API_KEY", "").strip():
        return _safe_explanation(contract)

    try:
        generated = explain(contract, prompt_version="v3")
        if generated.abstained or generated.sections is None:
            raise RuntimeError("the explanation engine returned no ruling sections")

        safe_text, _, _ = safe_v3(contract, generated.articles)
        mode = "safe_fallback" if generated.text_ar == safe_text else "grounded_llm"
        model = "deterministic-v3" if mode == "safe_fallback" else generated.model
        return ExplanationPayload(
            textAr=generated.text_ar,
            sections=_sections_payload(generated.sections),
            articles=[_article_payload(article) for article in generated.articles],
            promptVersion="v3",
            model=model,
            mode=mode,
            faithfulness=_faithfulness_payload(
                generated.text_ar,
                contract,
                generated.articles,
            ),
        )
    except Exception:
        logger.warning(
            "Grounded language generation failed; using the deterministic v3 fallback",
            exc_info=True,
        )
        return _safe_explanation(contract)


class AnalysisPipeline:
    def __init__(self, predictor: Any):
        self.predictor = predictor

    def analyze(self, video_path: Path) -> AnalysisResponse:
        started = time.perf_counter()
        try:
            contract = self.predictor.predict([video_path])
        except ValueError as exc:
            message = str(exc).lower()
            if "cannot open video" in message or "no frames decoded" in message:
                raise InvalidVideoError("video decoding failed") from exc
            raise InferencePipelineError("vision inference failed") from exc
        except Exception as exc:
            raise InferencePipelineError("vision inference failed") from exc

        contract_payload = _contract_payload(contract)
        elapsed = lambda: round(time.perf_counter() - started, 2)

        # This gate deliberately precedes every retrieval or language-generation call.
        if contract.should_abstain(config.CONFIDENCE_THRESHOLD):
            return AbstainedAnalysisPayload(
                contract=contract_payload,
                processingSeconds=elapsed(),
                threshold=config.CONFIDENCE_THRESHOLD,
                messageAr=config.ABSTAIN_MESSAGE_AR,
            )

        try:
            explanation = _grounded_explanation(contract)
        except Exception as exc:
            raise RulingPipelineError("grounded ruling generation failed") from exc

        return CompletedAnalysisPayload(
            contract=contract_payload,
            processingSeconds=elapsed(),
            explanation=explanation,
        )
