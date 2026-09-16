import { ReplaceIcon, WarningIcon } from "@/components/icons";
import { ReviewFrame } from "@/components/review-frame";
import { VideoPanel } from "@/components/video-panel";
import { uiCopy } from "@/lib/i18n";
import type { Language } from "@/lib/types";

export function ErrorState({
  language,
  previewUrl,
  fileName,
  message,
  onRetry,
  onReplace,
}: {
  language: Language;
  previewUrl: string;
  fileName: string;
  message: string;
  onRetry: () => void;
  onReplace: () => void;
}) {
  const copy = uiCopy[language];

  return (
    <section className="pt-10 sm:pt-16" aria-labelledby="analysis-error-title">
      <VideoPanel language={language} src={previewUrl} fileName={fileName} />
      <ReviewFrame tone="danger" className="mt-8 border border-danger/45 bg-charcoal">
        <div className="px-6 py-8 sm:px-10 sm:py-10">
          <WarningIcon className="mb-6 h-7 w-7 text-danger" />
          <p className="mb-3 text-[0.68rem] font-semibold uppercase tracking-broadcast text-danger">
            {copy.errorEyebrow}
          </p>
          <h1 id="analysis-error-title" tabIndex={-1} className="text-3xl font-semibold tracking-[-0.04em] sm:text-4xl">
            {copy.errorTitle}
          </h1>
          <p className="mt-4 max-w-2xl text-sm leading-7 text-muted">{copy.errorBody}</p>
          <p role="alert" dir="auto" className="mt-4 max-w-2xl text-sm leading-7 text-danger/95 [unicode-bidi:plaintext]">
            {message}
          </p>
          <div className="mt-7 flex flex-wrap gap-3">
            <button
              type="button"
              onClick={onRetry}
              className="min-h-11 bg-chalk px-4 text-sm font-semibold text-carbon"
            >
              {copy.tryAgain}
            </button>
            <button
              type="button"
              onClick={onReplace}
              className="inline-flex min-h-11 items-center gap-2 border border-hairline px-4 text-sm text-chalk transition-colors hover:border-danger/60"
            >
              <ReplaceIcon className="h-4 w-4" />
              {copy.replace}
            </button>
          </div>
        </div>
      </ReviewFrame>
    </section>
  );
}
