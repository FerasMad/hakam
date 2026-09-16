import { uiCopy } from "@/lib/i18n";
import type { Language } from "@/lib/types";

export function Footer({ language }: { language: Language }) {
  const copy = uiCopy[language];

  return (
    <footer className="mt-24 border-t border-hairline sm:mt-32">
      <div className="mx-auto max-w-canvas px-5 py-10 sm:px-8 lg:px-12">
        <div className="max-w-xl">
          <p className="mb-2 text-[0.68rem] font-semibold uppercase tracking-broadcast text-muted">
            {copy.researchLabel}
          </p>
          <p className="text-sm leading-6 text-muted">{copy.footerText}</p>
        </div>
      </div>
    </footer>
  );
}
