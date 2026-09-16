import { ArabicRuling } from "@/components/arabic-ruling";
import { LawEvidence } from "@/components/law-evidence";
import { TechnicalDetails } from "@/components/technical-details";
import { VisionResult } from "@/components/vision-result";
import type { CompletedAnalysis, Language } from "@/lib/types";

export function NormalResult({
  language,
  result,
  previewUrl,
  fileName,
}: {
  language: Language;
  result: CompletedAnalysis;
  previewUrl: string;
  fileName: string;
}) {
  return (
    <>
      <VisionResult
        language={language}
        contract={result.contract}
        previewUrl={previewUrl}
        fileName={fileName}
      />
      <LawEvidence language={language} articles={result.explanation.articles} />
      <ArabicRuling language={language} sections={result.explanation.sections} />
      <TechnicalDetails language={language} result={result} />
    </>
  );
}
