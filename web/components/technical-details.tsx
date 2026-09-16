"use client";

import { useState } from "react";
import { CheckIcon, ChevronIcon, CopyIcon } from "@/components/icons";
import { uiCopy } from "@/lib/i18n";
import type { AnalysisResult, Language } from "@/lib/types";

export function TechnicalDetails({ language, result }: { language: Language; result: AnalysisResult }) {
  const copy = uiCopy[language];
  const [copied, setCopied] = useState(false);
  const contractJson = JSON.stringify(result.contract, null, 2);

  async function copyContract() {
    try {
      await navigator.clipboard.writeText(contractJson);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1600);
    } catch {
      setCopied(false);
    }
  }

  const baseDetails = [
    [copy.modelVersion, result.contract.model_version],
    [copy.processingTime, `${result.processingSeconds.toFixed(2)} s`],
    [copy.numViews, String(result.contract.num_views)],
    [copy.lowConfidence, result.contract.low_confidence_fields.join(", ") || copy.none],
    [copy.abstentionState, result.contract.abstain ? copy.yes : copy.no],
  ];

  return (
    <details className="group mt-16 border-y border-hairline bg-charcoal/45 sm:mt-20">
      <summary className="flex min-h-20 cursor-pointer items-center justify-between gap-5 px-4 py-5 sm:px-6">
        <span>
          <strong className="block text-sm font-medium text-chalk">{copy.technicalDetails}</strong>
          <span className="mt-1 block text-xs text-muted">{copy.technicalHint}</span>
        </span>
        <ChevronIcon className="h-5 w-5 shrink-0 text-muted transition-transform group-open:rotate-180" />
      </summary>

      <div className="border-t border-hairline px-4 py-7 sm:px-6 sm:py-8">
        <dl className="grid gap-x-10 gap-y-5 sm:grid-cols-2 lg:grid-cols-3">
          {baseDetails.map(([label, value]) => (
            <div key={label}>
              <dt className="text-xs text-muted">{label}</dt>
              <dd dir="auto" className="mt-1 text-sm text-chalk [unicode-bidi:isolate]">{value}</dd>
            </div>
          ))}
          {result.status === "completed" ? (
            <>
              <div>
                <dt className="text-xs text-muted">{copy.explanationMode}</dt>
                <dd className="mt-1 text-sm text-chalk">
                  {result.explanation.mode === "grounded_llm" ? copy.groundedLlm : copy.safeFallback}
                </dd>
              </div>
              <div>
                <dt className="text-xs text-muted">{copy.faithfulness}</dt>
                <dd className="technical-value mt-1 text-sm text-chalk">
                  {Math.round(result.explanation.faithfulness.faithfulness * 100)}%
                </dd>
              </div>
              <div>
                <dt className="text-xs text-muted">{copy.citesArticle}</dt>
                <dd className="mt-1 text-sm text-chalk">
                  {result.explanation.faithfulness.citesArticle ? copy.yes : copy.no}
                </dd>
              </div>
              <div>
                <dt className="text-xs text-muted">{copy.hedged}</dt>
                <dd className="mt-1 text-sm text-chalk">
                  {result.explanation.faithfulness.hedgedLowConfidence ? copy.yes : copy.no}
                </dd>
              </div>
            </>
          ) : null}
        </dl>

        {result.status === "completed" ? (
          <div className="mt-8 border-t border-hairline pt-6">
            <h3 className="text-xs text-muted">{copy.retrievedArticles}</h3>
            <ul className="technical-value mt-3 space-y-2 font-mono text-xs leading-5 text-chalk">
              {result.explanation.articles.map((article) => (
                <li key={article.id}>
                  <span className="text-pitch">{article.id}</span>
                  <span className="mx-2 text-muted">—</span>
                  <span lang="ar" dir="rtl" className="font-arabic">{article.titleAr}</span>
                </li>
              ))}
            </ul>
          </div>
        ) : null}

        <div className="mt-8 border-t border-hairline pt-6">
          <div className="mb-3 flex items-center justify-between gap-4">
            <h3 className="text-xs text-muted">{copy.contractJson}</h3>
            <button
              type="button"
              onClick={copyContract}
              className="inline-flex min-h-10 items-center gap-2 border border-hairline px-3 text-xs text-chalk transition-colors hover:border-pitch/50"
            >
              {copied ? <CheckIcon className="h-4 w-4 text-pitch" /> : <CopyIcon className="h-4 w-4" />}
              <span aria-live="polite">{copied ? copy.copied : copy.copyJson}</span>
            </button>
          </div>
          <pre
            dir="ltr"
            className="max-h-[30rem] overflow-auto border border-hairline bg-carbon p-4 text-left font-mono text-[0.7rem] leading-5 text-[#C9D0CD]"
          >
            <code>{contractJson}</code>
          </pre>
        </div>
      </div>
    </details>
  );
}
