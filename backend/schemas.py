from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PredictionPayload(StrictModel):
    label: str
    confidence: float = Field(ge=0.0, le=1.0)


class HakamContractPayload(StrictModel):
    action_id: str
    offence: PredictionPayload
    card: PredictionPayload | None
    card_colour: None = None
    attributes: dict[str, PredictionPayload]
    scene: None = None
    model_version: str
    num_views: Literal[1]
    low_confidence_fields: list[str]
    abstain: bool


class SourcePagesPayload(StrictModel):
    ar: list[int] | None = None
    en: list[int] | None = None


class LawArticlePayload(StrictModel):
    id: str
    law: str
    section: str
    titleAr: str
    titleEn: str | None = None
    textAr: str
    textEn: str | None = None
    sourcePages: SourcePagesPayload | None = None


class RulingSectionsPayload(StrictModel):
    decision: str
    restart: str
    disciplinary: str
    law: str
    whyTitle: Literal["لماذا تُعد مخالفة", "لماذا لا تُعد مخالفة"]
    why: str
    confidence: str


class FaithfulnessPayload(StrictModel):
    faithfulness: float = Field(ge=0.0, le=1.0)
    citesArticle: bool
    hedgedLowConfidence: bool
    unsupported: list[str]


class ExplanationPayload(StrictModel):
    textAr: str
    sections: RulingSectionsPayload
    articles: list[LawArticlePayload]
    promptVersion: Literal["v3"]
    model: str
    mode: Literal["grounded_llm", "safe_fallback"]
    faithfulness: FaithfulnessPayload


class CompletedAnalysisPayload(StrictModel):
    status: Literal["completed"] = "completed"
    contract: HakamContractPayload
    processingSeconds: float = Field(ge=0.0)
    explanation: ExplanationPayload


class AbstainedAnalysisPayload(StrictModel):
    status: Literal["abstained"] = "abstained"
    contract: HakamContractPayload
    processingSeconds: float = Field(ge=0.0)
    threshold: Literal[0.6] = 0.6
    messageAr: Literal["الثقة منخفضة — هذه الحالة تحتاج مراجعة بشرية."] = (
        "الثقة منخفضة — هذه الحالة تحتاج مراجعة بشرية."
    )


AnalysisResponse = Annotated[
    CompletedAnalysisPayload | AbstainedAnalysisPayload,
    Field(discriminator="status"),
]


class LivenessPayload(StrictModel):
    status: Literal["ok"] = "ok"


class ArtifactStatusPayload(StrictModel):
    finalPt: bool
    thresholdsJson: bool


class ReadinessPayload(StrictModel):
    status: Literal["ready", "not_ready"]
    modelLoaded: bool
    modelVersion: str | None = None
    device: str | None = None
    tasks: list[str]
    thresholds: dict[str, float]
    llmConfigured: bool
    retrievalMode: Literal["hybrid", "tags"]
    artifacts: ArtifactStatusPayload
    reasonCode: str | None = None


class ErrorDetailPayload(StrictModel):
    code: str
    message: str


class ErrorPayload(StrictModel):
    error: ErrorDetailPayload
