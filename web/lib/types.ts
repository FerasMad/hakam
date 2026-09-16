export type Language = "en" | "ar";

export type OffenceLabel = "offence" | "no_offence";
export type CardLabel = "card" | "no_card";
export type ActionClassLabel = "tackle" | "hands" | "elbowing" | "high leg";
export type BodyPartLabel = "under_body" | "upper_body";

export interface Prediction<TLabel extends string = string> {
  label: TLabel;
  confidence: number;
}

export interface HakamContract {
  action_id: string;
  offence: Prediction<OffenceLabel>;
  card: Prediction<CardLabel> | null;
  /** The current production model does not predict card colour. */
  card_colour: null;
  attributes: {
    action_class?: Prediction<ActionClassLabel>;
    body_part?: Prediction<BodyPartLabel>;
  };
  scene: null;
  model_version: string;
  num_views: 1;
  low_confidence_fields: string[];
  abstain: boolean;
}

export interface LawArticle {
  id: string;
  law: string;
  section: string;
  titleAr: string;
  titleEn?: string | null;
  textAr: string;
  textEn?: string | null;
  sourcePages?: {
    ar?: number[] | null;
    en?: number[] | null;
  } | null;
}

export interface RulingSections {
  decision: string;
  restart: string;
  disciplinary: string;
  law: string;
  whyTitle: "لماذا تُعد مخالفة" | "لماذا لا تُعد مخالفة";
  why: string;
  confidence: string;
}

export interface FaithfulnessReport {
  faithfulness: number;
  citesArticle: boolean;
  hedgedLowConfidence: boolean;
  unsupported: string[];
}

export type ExplanationMode = "grounded_llm" | "safe_fallback";

export interface ExplanationPayload {
  textAr: string;
  sections: RulingSections;
  articles: LawArticle[];
  promptVersion: "v3";
  model: string;
  mode: ExplanationMode;
  faithfulness: FaithfulnessReport;
}

interface AnalysisBase {
  contract: HakamContract;
  processingSeconds: number;
}

export interface CompletedAnalysis extends AnalysisBase {
  status: "completed";
  explanation: ExplanationPayload;
}

export interface AbstainedAnalysis extends AnalysisBase {
  status: "abstained";
  threshold: 0.6;
  messageAr: "الثقة منخفضة — هذه الحالة تحتاج مراجعة بشرية.";
}

export type AnalysisResult = CompletedAnalysis | AbstainedAnalysis;

export interface AnalyzeIncidentRequest {
  file: File;
  signal?: AbortSignal;
}

export interface AnalysisService {
  analyze(request: AnalyzeIncidentRequest): Promise<AnalysisResult>;
}
