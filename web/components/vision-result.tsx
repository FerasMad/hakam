import { ConfidenceRail } from "@/components/confidence-rail";
import { SectionHeading } from "@/components/section-heading";
import { VideoPanel } from "@/components/video-panel";
import { actionLabels, bodyPartLabels, uiCopy } from "@/lib/i18n";
import type { HakamContract, Language } from "@/lib/types";

export function VisionResult({
  language,
  contract,
  previewUrl,
  fileName,
}: {
  language: Language;
  contract: HakamContract;
  previewUrl: string;
  fileName: string;
}) {
  const copy = uiCopy[language];
  const isOffence = contract.offence.label === "offence";
  const action = contract.attributes.action_class;
  const bodyPart = contract.attributes.body_part;

  return (
    <section className="pt-10 sm:pt-16" aria-labelledby="vision-result-title">
      <SectionHeading marker={copy.visionLayer} title={copy.visionTitle} />

      <div className="grid items-start gap-8 lg:grid-cols-12 lg:gap-10">
        <div className="lg:col-span-7 xl:col-span-8">
          <VideoPanel language={language} src={previewUrl} fileName={fileName} />
        </div>

        <article className="border-t border-pitch pt-6 lg:col-span-5 lg:border-s lg:border-t-0 lg:border-pitch/60 lg:ps-8 lg:pt-0 xl:col-span-4">
          <p className="mb-3 text-[0.68rem] font-semibold uppercase tracking-broadcast text-muted">
            {copy.visionTitle}
          </p>
          <h1
            id="vision-result-title"
            tabIndex={-1}
            className="text-5xl font-semibold leading-[0.95] tracking-[-0.06em] text-chalk sm:text-6xl lg:text-5xl xl:text-6xl"
          >
            {isOffence ? copy.offence : copy.noOffence}
          </h1>

          <div className="mt-8">
            <ConfidenceRail
              value={contract.offence.confidence}
              label={copy.offenceConfidence}
              thresholdLabel={copy.confidenceGate}
            />
          </div>

          <dl className="mt-8 divide-y divide-hairline border-y border-hairline">
            {isOffence && contract.card ? (
              <div className="py-5">
                <dt className="text-xs text-muted">{copy.cardDecision}</dt>
                <dd className="mt-2 flex items-start gap-3">
                  <span className="mt-0.5 h-6 w-4 shrink-0 rounded-[2px] border border-chalk/55" aria-hidden="true" />
                  <span>
                    <strong className="block text-base font-medium text-chalk">
                      {contract.card.label === "card" ? copy.cardWarranted : copy.noCard}
                    </strong>
                    {contract.card.label === "card" ? (
                      <span className="mt-1 block text-xs leading-5 text-muted">
                        {copy.colourNotDetermined}
                      </span>
                    ) : null}
                    <span className="mt-2 block text-xs text-muted">
                      {copy.cardConfidence}: {Math.round(contract.card.confidence * 100)}%
                    </span>
                  </span>
                </dd>
              </div>
            ) : null}

            <div className="grid grid-cols-[minmax(0,0.8fr)_minmax(0,1.2fr)] gap-4 py-5">
              <dt className="text-xs text-muted">{copy.actionFamily}</dt>
              <dd className="text-sm font-medium leading-6 text-chalk">
                {action ? actionLabels[language][action.label] : copy.actionUnavailable}
              </dd>
            </div>

            {bodyPart ? (
              <div className="grid grid-cols-[minmax(0,0.8fr)_minmax(0,1.2fr)] gap-4 py-5">
                <dt className="text-xs text-muted">{copy.bodyPart}</dt>
                <dd className="text-sm font-medium leading-6 text-chalk">
                  {bodyPartLabels[language][bodyPart.label]}
                </dd>
              </div>
            ) : null}
          </dl>
        </article>
      </div>
    </section>
  );
}
