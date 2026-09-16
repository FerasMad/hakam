"use client";

import { useState, type DragEvent } from "react";
import { PlayReviewIcon, RemoveIcon, ReplaceIcon, UploadIcon } from "@/components/icons";
import { ReviewFrame } from "@/components/review-frame";
import { VideoPanel } from "@/components/video-panel";
import { uiCopy } from "@/lib/i18n";
import type { Language } from "@/lib/types";

export function UploadStation({
  language,
  previewUrl,
  fileName,
  previewReady,
  validationMessage,
  onFiles,
  onOpenPicker,
  onRemove,
  onAnalyze,
  onPreviewReady,
  onPreviewError,
}: {
  language: Language;
  previewUrl: string | null;
  fileName: string | null;
  previewReady: boolean;
  validationMessage: string | null;
  onFiles: (files: FileList) => void;
  onOpenPicker: () => void;
  onRemove: () => void;
  onAnalyze: () => void;
  onPreviewReady: () => void;
  onPreviewError: () => void;
}) {
  const copy = uiCopy[language];
  const [dragging, setDragging] = useState(false);

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragging(false);
    onFiles(event.dataTransfer.files);
  }

  if (previewUrl && fileName) {
    return (
      <section aria-labelledby="selected-clip-title" className="mt-10 sm:mt-14">
        <div className="mb-5 flex flex-wrap items-end justify-between gap-4">
          <div>
            <p className="mb-2 text-[0.68rem] font-semibold uppercase tracking-broadcast text-pitch">
              {copy.uploadKicker}
            </p>
            <h2 id="selected-clip-title" className="text-xl font-medium text-chalk sm:text-2xl">
              {copy.selectedClip}
            </h2>
          </div>
          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              onClick={onOpenPicker}
              className="inline-flex min-h-11 items-center gap-2 border border-hairline px-4 text-sm text-chalk transition-colors hover:border-pitch/60"
            >
              <ReplaceIcon className="h-4 w-4" />
              {copy.replace}
            </button>
            <button
              type="button"
              onClick={onRemove}
              className="inline-flex min-h-11 items-center gap-2 border border-hairline px-4 text-sm text-muted transition-colors hover:border-danger/60 hover:text-chalk"
            >
              <RemoveIcon className="h-4 w-4" />
              {copy.remove}
            </button>
          </div>
        </div>

        <VideoPanel
          language={language}
          src={previewUrl}
          fileName={fileName}
          onReady={onPreviewReady}
          onError={onPreviewError}
        />

        <div className="mt-4 flex items-start gap-3 border-s border-pitch/50 ps-4 text-sm leading-6 text-muted">
          <span className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-pitch" />
          <p>{copy.uploadGuidance}</p>
        </div>

        <div aria-live="polite" className="mt-5">
          {validationMessage ? (
            <p role="alert" className="border border-danger/45 bg-danger/[0.06] px-4 py-3 text-sm text-chalk">
              {validationMessage}
            </p>
          ) : !previewReady ? (
            <p className="text-sm text-muted">{copy.previewPreparing}</p>
          ) : null}
        </div>

        <div className="mt-7 flex justify-end">
          <button
            type="button"
            onClick={onAnalyze}
            disabled={!previewReady}
            className="inline-flex min-h-12 items-center gap-3 bg-pitch px-6 text-sm font-semibold text-carbon transition-colors hover:bg-[#49D3A4] disabled:cursor-not-allowed disabled:bg-hairline disabled:text-muted"
          >
            <PlayReviewIcon className="h-5 w-5" />
            {copy.analyze}
          </button>
        </div>
      </section>
    );
  }

  return (
    <section className="mt-10 sm:mt-14" aria-labelledby="upload-title">
      <ReviewFrame
        tone={validationMessage ? "danger" : "pitch"}
        className={`border bg-charcoal transition-colors ${
          dragging ? "border-pitch bg-pitch/[0.035]" : validationMessage ? "border-danger/55" : "border-hairline"
        }`}
      >
        <div
          className="flex min-h-[22rem] flex-col items-center justify-center px-6 py-14 text-center sm:min-h-[26rem] sm:px-12"
          onDragEnter={(event) => {
            event.preventDefault();
            setDragging(true);
          }}
          onDragOver={(event) => event.preventDefault()}
          onDragLeave={(event) => {
            if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setDragging(false);
          }}
          onDrop={handleDrop}
        >
          <div className="mb-7 grid h-16 w-16 place-items-center border border-hairline bg-carbon text-pitch">
            <UploadIcon className="h-7 w-7" />
          </div>
          <p className="mb-3 text-[0.68rem] font-semibold uppercase tracking-broadcast text-pitch">
            {copy.uploadKicker}
          </p>
          <h2 id="upload-title" className="max-w-xl text-2xl font-medium tracking-[-0.03em] sm:text-4xl">
            {dragging ? copy.dropActive : copy.uploadTitle}
          </h2>
          <p className="mt-4 max-w-lg text-sm leading-6 text-muted sm:text-base">{copy.uploadBody}</p>
          <button
            type="button"
            onClick={onOpenPicker}
            className="mt-8 min-h-12 border border-chalk/25 bg-chalk px-6 text-sm font-semibold text-carbon transition-colors hover:bg-white"
          >
            {copy.chooseVideo}
          </button>
          <p className="mt-6 max-w-md text-sm leading-6 text-muted">{copy.uploadGuidance}</p>
        </div>
      </ReviewFrame>

      <div aria-live="polite">
        {validationMessage ? (
          <p role="alert" className="mt-4 border-s-2 border-danger bg-danger/[0.06] px-4 py-3 text-sm text-chalk">
            {validationMessage}
          </p>
        ) : null}
      </div>
    </section>
  );
}
