import { VideoPanel } from "@/components/video-panel";
import { uiCopy } from "@/lib/i18n";
import type { Language } from "@/lib/types";

export function ProcessingState({
  language,
  previewUrl,
  fileName,
}: {
  language: Language;
  previewUrl: string;
  fileName: string;
}) {
  const copy = uiCopy[language];

  return (
    <section className="mt-10 sm:mt-14" aria-labelledby="processing-title" aria-busy="true">
      <VideoPanel language={language} src={previewUrl} fileName={fileName} processing />
      <div className="mx-auto mt-8 max-w-3xl text-center" aria-live="polite" role="status">
        <div className="mx-auto mb-5 flex h-7 w-10 items-end justify-center gap-1" aria-hidden="true">
          <span className="processing-tick h-4 w-px bg-pitch" />
          <span className="processing-tick h-7 w-px bg-pitch" />
          <span className="processing-tick h-5 w-px bg-pitch" />
        </div>
        <h2 id="processing-title" className="text-2xl font-medium tracking-[-0.03em] sm:text-3xl">
          {copy.processingTitle}
        </h2>
        <p className="mx-auto mt-4 max-w-2xl text-sm leading-7 text-muted sm:text-base">
          {copy.processingBody}
        </p>
        <p className={`mt-5 text-[0.68rem] uppercase leading-6 text-muted ${language === "ar" ? "tracking-normal" : "tracking-[0.13em]"}`}>
          {copy.processingStatus}
        </p>
        <span className="sr-only">{copy.processingA11y}</span>
      </div>
    </section>
  );
}
