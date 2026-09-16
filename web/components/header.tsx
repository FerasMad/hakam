import Image from "next/image";
import type { Language } from "@/lib/types";
import { uiCopy } from "@/lib/i18n";

export function Header({
  language,
  onLanguageChange,
  onNewReview,
  showNewReview,
}: {
  language: Language;
  onLanguageChange: () => void;
  onNewReview: () => void;
  showNewReview: boolean;
}) {
  const copy = uiCopy[language];

  return (
    <header className="border-b border-white/[0.07] bg-carbon/90 backdrop-blur-md">
      <div className="mx-auto flex min-h-[6.25rem] max-w-canvas items-center justify-between gap-5 px-5 sm:px-8 lg:px-12">
        <a href="#main" className="group flex items-center gap-3" aria-label={copy.brandLabel}>
          <span className="relative grid h-12 w-12 shrink-0 place-items-center sm:h-14 sm:w-14">
            <Image
              src="/hakam-robot-referee.png"
              alt=""
              width={56}
              height={56}
              priority
              className="h-full w-full object-contain transition-transform duration-200 group-hover:scale-105"
            />
          </span>
          <span>
            <span className="flex items-baseline gap-2">
              {language === "ar" ? (
                <>
                  <strong className="font-arabic text-lg font-semibold">حكم</strong>
                  <strong className="font-sans text-sm font-semibold tracking-[0.18em]">HAKAM</strong>
                </>
              ) : (
                <>
                  <strong className="text-sm font-semibold tracking-[0.18em]">HAKAM</strong>
                  <strong className="font-arabic text-lg font-semibold">حكم</strong>
                </>
              )}
            </span>
            <span className="mt-0.5 hidden text-[0.68rem] text-muted sm:block">{copy.descriptor}</span>
          </span>
        </a>

        <div className="flex items-center gap-2 sm:gap-3">
          {showNewReview ? (
            <button
              type="button"
              onClick={onNewReview}
              className="min-h-11 border border-hairline px-3 text-xs font-medium text-chalk transition-colors hover:border-white/30 hover:bg-white/[0.03] sm:px-4"
            >
              {copy.newReview}
            </button>
          ) : null}
          <button
            type="button"
            onClick={onLanguageChange}
            aria-label={copy.languageLabel}
            className="min-h-11 min-w-11 border border-hairline px-3 font-sans text-xs font-semibold tracking-[0.12em] text-muted transition-colors hover:border-pitch/50 hover:text-chalk sm:px-4"
          >
            {language === "en" ? "عربي" : "EN"}
          </button>
        </div>
      </div>
    </header>
  );
}
