import type { AnalysisResult, AnalysisService } from "@/lib/types";

const API_BASE_URL = (
  process.env.NEXT_PUBLIC_HAKAM_API_URL ?? "http://localhost:8000"
).replace(/\/$/, "");

export class AnalysisServiceError extends Error {
  readonly code: string;
  readonly status?: number;

  constructor(message: string, code = "ANALYSIS_UNAVAILABLE", status?: number) {
    super(message);
    this.name = "AnalysisServiceError";
    this.code = code;
    this.status = status;
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function isPrediction(value: unknown, labels: readonly string[]): boolean {
  if (!isRecord(value)) return false;
  return (
    typeof value.label === "string" &&
    labels.includes(value.label) &&
    typeof value.confidence === "number" &&
    Number.isFinite(value.confidence) &&
    value.confidence >= 0 &&
    value.confidence <= 1
  );
}

function isContract(value: unknown): boolean {
  if (!isRecord(value) || !isRecord(value.attributes)) return false;
  if (!isPrediction(value.offence, ["offence", "no_offence"])) return false;
  if (value.card !== null && !isPrediction(value.card, ["card", "no_card"])) return false;
  if ((value.offence as { label: string }).label === "no_offence" && value.card !== null) return false;
  if (value.card_colour !== null || value.scene !== null || value.num_views !== 1) return false;

  const attributeKeys = Object.keys(value.attributes);
  if (attributeKeys.some((key) => key !== "action_class" && key !== "body_part")) return false;

  const action = value.attributes.action_class;
  const bodyPart = value.attributes.body_part;
  if (action !== undefined && !isPrediction(action, ["tackle", "hands", "elbowing", "high leg"])) {
    return false;
  }
  if (bodyPart !== undefined && !isPrediction(bodyPart, ["under_body", "upper_body"])) return false;

  return (
    typeof value.action_id === "string" &&
    typeof value.model_version === "string" &&
    Array.isArray(value.low_confidence_fields) &&
    value.low_confidence_fields.every((field) => typeof field === "string") &&
    typeof value.abstain === "boolean"
  );
}

function isLawArticle(value: unknown): boolean {
  if (!isRecord(value)) return false;
  const requiredText = ["id", "law", "section", "titleAr", "textAr"].every(
    (field) => typeof value[field] === "string" && value[field].length > 0,
  );
  if (!requiredText) return false;
  if (value.titleEn !== undefined && value.titleEn !== null && typeof value.titleEn !== "string") return false;
  if (value.textEn !== undefined && value.textEn !== null && typeof value.textEn !== "string") return false;
  if (value.sourcePages !== undefined && value.sourcePages !== null) {
    if (!isRecord(value.sourcePages)) return false;
    for (const language of ["ar", "en"] as const) {
      const pages = value.sourcePages[language];
      if (pages !== undefined && pages !== null && (!Array.isArray(pages) || !pages.every(Number.isInteger))) {
        return false;
      }
    }
  }
  return true;
}

function isExplanation(value: unknown): boolean {
  if (!isRecord(value) || !isRecord(value.sections) || !isRecord(value.faithfulness)) return false;
  const sections = value.sections;
  const requiredSections = ["decision", "restart", "disciplinary", "law", "why", "confidence"];
  if (!requiredSections.every((field) => typeof sections[field] === "string" && sections[field].length > 0)) {
    return false;
  }
  if (sections.whyTitle !== "لماذا تُعد مخالفة" && sections.whyTitle !== "لماذا لا تُعد مخالفة") {
    return false;
  }
  if (!Array.isArray(value.articles) || value.articles.length === 0 || !value.articles.every(isLawArticle)) {
    return false;
  }
  const faithfulness = value.faithfulness;
  return (
    typeof value.textAr === "string" &&
    value.textAr.length > 0 &&
    value.promptVersion === "v3" &&
    typeof value.model === "string" &&
    (value.mode === "grounded_llm" || value.mode === "safe_fallback") &&
    typeof faithfulness.faithfulness === "number" &&
    Number.isFinite(faithfulness.faithfulness) &&
    faithfulness.faithfulness >= 0 &&
    faithfulness.faithfulness <= 1 &&
    typeof faithfulness.citesArticle === "boolean" &&
    typeof faithfulness.hedgedLowConfidence === "boolean" &&
    Array.isArray(faithfulness.unsupported) &&
    faithfulness.unsupported.every((claim) => typeof claim === "string")
  );
}

function isAnalysisResult(value: unknown): value is AnalysisResult {
  if (!isRecord(value)) return false;
  const candidate = value;
  if (candidate.status !== "completed" && candidate.status !== "abstained") return false;
  if (!isContract(candidate.contract)) return false;
  if (
    typeof candidate.processingSeconds !== "number" ||
    !Number.isFinite(candidate.processingSeconds) ||
    candidate.processingSeconds < 0
  ) {
    return false;
  }

  const contract = candidate.contract as { abstain: boolean };

  if (candidate.status === "abstained") {
    return (
      contract.abstain &&
      candidate.threshold === 0.6 &&
      candidate.messageAr === "الثقة منخفضة — هذه الحالة تحتاج مراجعة بشرية." &&
      candidate.explanation === undefined
    );
  }

  return !contract.abstain && isExplanation(candidate.explanation);
}

async function errorFromResponse(response: Response): Promise<AnalysisServiceError> {
  let code = "ANALYSIS_UNAVAILABLE";
  let message = "Hakam could not analyze this video. Please try again.";

  try {
    const payload = (await response.json()) as {
      error?: { code?: unknown; message?: unknown };
    };
    if (typeof payload.error?.code === "string") code = payload.error.code;
    if (typeof payload.error?.message === "string") message = payload.error.message;
  } catch {
    // Keep the safe public fallback when the response is not JSON.
  }

  return new AnalysisServiceError(message, code, response.status);
}

class HttpAnalysisService implements AnalysisService {
  async analyze({ file, signal }: Parameters<AnalysisService["analyze"]>[0]) {
    const body = new FormData();
    body.append("video", file, file.name);

    let response: Response;
    try {
      response = await fetch(`${API_BASE_URL}/api/analyze`, {
        method: "POST",
        headers: { Accept: "application/json" },
        body,
        signal,
      });
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") throw error;
      throw new AnalysisServiceError(
        "The Hakam backend is unavailable. Confirm that it is running and try again.",
        "BACKEND_UNAVAILABLE",
      );
    }

    if (!response.ok) throw await errorFromResponse(response);

    let payload: unknown;
    try {
      payload = await response.json();
    } catch {
      throw new AnalysisServiceError(
        "The Hakam backend returned an unreadable response.",
        "INVALID_RESPONSE",
        response.status,
      );
    }
    if (!isAnalysisResult(payload)) {
      throw new AnalysisServiceError(
        "The Hakam backend returned an unexpected response.",
        "INVALID_RESPONSE",
        response.status,
      );
    }
    return payload;
  }
}

export const analysisService: AnalysisService = new HttpAnalysisService();
