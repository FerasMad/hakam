import { ConfidenceRail } from "@/components/confidence-rail";
import { ReplaceIcon, ReviewIcon } from "@/components/icons";
import { ReviewFrame } from "@/components/review-frame";
import { TechnicalDetails } from "@/components/technical-details";
import { VideoPanel } from "@/components/video-panel";
import { uiCopy } from "@/lib/i18n";
import type { AbstainedAnalysis, Language } from "@/lib/types";

export function HumanReview({
  language,
  result,
  previewUrl,
  fileName,
  onReplace,
  onReset,
}: {
  language: Language;
  result: AbstainedAnalysis;
  previewUrl: string;
  fileName: string;
  onReplace: () => void;
  onReset: () => void;
}) {
  const copy = uiCopy[language];

  return (
    <section className="pt-10 sm:pt-16" aria-labelledby="human-review-title">
      <VideoPanel language={language} src={previewUrl} fileName={fileName} />

      <ReviewFrame tone="copper" className="mt-8 border border-copper/45 bg-charcoal">
        <div className="grid gap-8 px-6 py-8 sm:px-9 sm:py-10 lg:grid-cols-[1.15fr_0.85fr] lg:items-center lg:gap-14 lg:px-12">
          <div>
            <div className="mb-6 grid h-12 w-12 place-items-center border border-copper/50 text-copper">
              <ReviewIcon className="h-6 w-6" />
            </div>
            <p className="mb-3 text-[0.68rem] font-semibold uppercase tracking-broadcast text-copper">
              {copy.humanReviewEyebrow}
            </p>
            <h1
              id="human-review-title"
              tabIndex={-1}
              className="text-3xl font-semibold tracking-[-0.04em] text-chalk sm:text-5xl"
            >
              {copy.humanReview}
            </h1>
            <p className="mt-4 max-w-xl text-sm leading-7 text-muted sm:text-base">{copy.humanReviewBody}</p>
            <p className="arabic-editorial mt-6 border-s-2 border-copper ps-4 text-xl font-medium leading-9 text-chalk" lang="ar" dir="rtl">
              {result.messageAr}
            </p>
          </div>

          <div className="border-t border-hairline pt-7 lg:border-s lg:border-t-0 lg:ps-10 lg:pt-0">
            <ConfidenceRail
              value={result.contract.offence.confidence}
              label={copy.actualConfidence}
              thresholdLabel={`${copy.requiredThreshold}: 60%`}
              tone="copper"
            />
            <dl className="mt-8 grid grid-cols-2 gap-4 border-t border-hairline pt-5 text-sm">
              <div>
                <dt className="text-xs text-muted">{copy.actualConfidence}</dt>
                <dd className="technical-value mt-1 text-lg text-chalk">
                  {Math.round(result.contract.offence.confidence * 100)}%
                </dd>
              </div>
              <div>
                <dt className="text-xs text-muted">{copy.requiredThreshold}</dt>
                <dd className="technical-value mt-1 text-lg text-chalk">60%</dd>
              </div>
            </dl>
            <div className="mt-7 flex flex-wrap gap-3">
              <button
                type="button"
                onClick={onReplace}
                className="inline-flex min-h-11 items-center gap-2 bg-chalk px-4 text-sm font-semibold text-carbon"
              >
                <ReplaceIcon className="h-4 w-4" />
                {copy.replace}
              </button>
              <button
                type="button"
                onClick={onReset}
                className="min-h-11 border border-hairline px-4 text-sm text-chalk transition-colors hover:border-copper/60"
              >
                {copy.reviewAnother}
              </button>
            </div>
          </div>
        </div>
      </ReviewFrame>

      <TechnicalDetails language={language} result={result} />
    </section>
  );
}
