"use client";

import { useState } from "react";
import { PlayReviewIcon } from "@/components/icons";
import { uiCopy } from "@/lib/i18n";
import { sampleIncidents, type SampleIncident } from "@/lib/samples";
import type { Language } from "@/lib/types";

export function TestYourself({
  language,
  onSample,
}: {
  language: Language;
  onSample: (sample: SampleIncident, file: File) => void;
}) {
  const copy = uiCopy[language];
  const [loadingId, setLoadingId] = useState<string | null>(null);
  const [failedId, setFailedId] = useState<string | null>(null);

  async function choose(sample: SampleIncident) {
    if (loadingId) return;
    setLoadingId(sample.id);
    setFailedId(null);
    try {
      const response = await fetch(sample.src);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const blob = await response.blob();
      onSample(sample, new File([blob], `incident-${sample.id}.mp4`, { type: "video/mp4" }));
    } catch {
      setFailedId(sample.id);
    } finally {
      setLoadingId(null);
    }
  }

  return (
    <section className="mt-14 sm:mt-20" aria-labelledby="test-yourself-title">
      <p className="mb-3 text-[0.68rem] font-semibold uppercase tracking-broadcast text-pitch">
        {copy.testYourselfKicker}
      </p>
      <h2 id="test-yourself-title" className="text-2xl font-medium tracking-[-0.03em] sm:text-3xl">
        {copy.testYourselfTitle}
      </h2>
      <p className="mt-3 max-w-2xl text-sm leading-6 text-muted sm:text-base">{copy.testYourselfBody}</p>

      <ul className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
        {sampleIncidents.map((sample) => (
          <li key={sample.id} className="flex flex-col border border-hairline bg-charcoal">
            <div className="aspect-video overflow-hidden bg-black">
              <video
                className="h-full w-full object-contain"
                src={`${sample.src}#t=2.5`}
                preload="metadata"
                muted
                playsInline
                aria-hidden="true"
              />
            </div>
            <div className="flex flex-1 flex-col gap-3 p-4">
              <p className="text-[0.68rem] font-semibold uppercase tracking-broadcast text-muted">
                {copy.sampleIncident} {sample.id}
              </p>
              <div className="text-sm leading-6">
                <p className="text-muted">{copy.refereeDecision}</p>
                <p className="text-chalk">{copy[sample.refereeKey]}</p>
                <p className="text-muted">{copy[sample.detailKey]}</p>
              </div>
              <button
                type="button"
                onClick={() => choose(sample)}
                disabled={Boolean(loadingId)}
                className="mt-auto inline-flex min-h-11 items-center justify-center gap-2 border border-pitch/50 px-3 text-sm font-semibold text-chalk transition-colors hover:bg-pitch hover:text-carbon disabled:cursor-wait disabled:opacity-60"
              >
                <PlayReviewIcon className="h-4 w-4" />
                {loadingId === sample.id ? copy.sampleLoading : copy.sampleAnalyze}
              </button>
              {failedId === sample.id ? (
                <p role="alert" className="text-xs leading-5 text-danger">{copy.sampleUnavailable}</p>
              ) : null}
            </div>
          </li>
        ))}
      </ul>
    </section>
  );
}
