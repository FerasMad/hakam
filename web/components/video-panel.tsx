import { uiCopy } from "@/lib/i18n";
import type { Language } from "@/lib/types";
import { ReviewFrame } from "@/components/review-frame";

export function VideoPanel({
  language,
  src,
  fileName,
  processing = false,
  onReady,
  onError,
}: {
  language: Language;
  src: string;
  fileName: string;
  processing?: boolean;
  onReady?: () => void;
  onError?: () => void;
}) {
  const copy = uiCopy[language];

  return (
    <ReviewFrame className="overflow-hidden border border-hairline bg-black shadow-station">
      <div className="relative flex aspect-video items-center justify-center overflow-hidden bg-black">
        <video
          key={src}
          className="video-element h-full w-full object-contain"
          src={src}
          aria-label={copy.videoPreview}
          controls={!processing}
          preload="metadata"
          onLoadedMetadata={onReady}
          onCanPlay={onReady}
          onError={onError}
        >
          {copy.videoUnsupported}
        </video>

        {processing ? (
          <div className="absolute inset-0 overflow-hidden bg-carbon/75" aria-hidden="true">
            <div className="processing-scan" />
            <div className="absolute inset-0 border-[14px] border-black/15" />
            <div className="absolute start-6 top-6 flex h-7 items-center gap-1 border border-pitch/35 bg-carbon/80 px-2.5">
              <span className="processing-tick h-2.5 w-px bg-pitch" />
              <span className="processing-tick h-3.5 w-px bg-pitch" />
              <span className="processing-tick h-2 w-px bg-pitch" />
            </div>
          </div>
        ) : null}
      </div>

      <div className="border-t border-hairline bg-charcoal px-4 py-3 sm:px-5">
        <div className="text-xs">
          <p className="min-w-0 truncate text-muted">
            <span className="me-2 text-chalk/70">{copy.fileName}</span>
            <bdi className="technical-value">{fileName}</bdi>
          </p>
        </div>
        <div className="mt-4 grid grid-cols-[1fr_auto_1fr] items-center gap-3" aria-label={copy.midpointHelp}>
          <span className="h-px bg-hairline" />
          <span className="flex items-center gap-2 text-[0.65rem] text-muted">
            <span className="h-2.5 w-px bg-pitch" />
            {copy.midpoint}
          </span>
          <span className="h-px bg-hairline" />
        </div>
      </div>
    </ReviewFrame>
  );
}
