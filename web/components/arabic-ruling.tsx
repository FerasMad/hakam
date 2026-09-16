import { EyeIcon, PenIcon, ScaleIcon } from "@/components/icons";
import { SectionHeading } from "@/components/section-heading";
import { uiCopy } from "@/lib/i18n";
import type { Language, RulingSections } from "@/lib/types";

export function ArabicRuling({
  language,
  sections,
}: {
  language: Language;
  sections: RulingSections;
}) {
  const copy = uiCopy[language];
  const rows = [
    { label: "العقوبة الفنية", value: sections.restart },
    { label: "العقوبة الانضباطية", value: sections.disciplinary },
    { label: "المادة", value: sections.law },
    { label: sections.whyTitle, value: sections.why },
    { label: "مستوى الثقة", value: sections.confidence },
  ];

  const provenance = [
    { Icon: EyeIcon, text: copy.provenanceVision },
    { Icon: ScaleIcon, text: copy.provenanceLaw },
    { Icon: PenIcon, text: copy.provenanceLanguage },
  ];

  return (
    <section className="mt-24 sm:mt-32" aria-labelledby="ruling-title">
      <SectionHeading
        id="ruling-title"
        marker={copy.rulingLayer}
        title={copy.rulingTitle}
        description={copy.rulingIsArabic}
      />

      <article className="arabic-editorial border-t-2 border-pitch bg-[#0D1011] shadow-station" lang="ar" dir="rtl">
        <header className="border-b border-hairline px-6 py-8 sm:px-10 sm:py-10 lg:px-14">
          <div className="mx-auto max-w-5xl">
            <p className="mb-4 text-xs font-semibold tracking-[0.08em] text-pitch sm:text-sm">القرار</p>
            <h3 className="arabic-ruling-copy max-w-[52rem] text-3xl font-semibold leading-[1.6] tracking-[-0.02em] text-chalk sm:text-4xl lg:text-[2.75rem]">
              {sections.decision}
            </h3>
          </div>
        </header>

        <dl className="mx-auto max-w-5xl px-6 sm:px-10 lg:px-14">
          {rows.map((row, index) => (
            <div
              key={row.label}
              className={`grid gap-4 border-b border-hairline py-8 last:border-b-0 sm:grid-cols-[10rem_minmax(0,1fr)] sm:gap-10 sm:py-10 ${
                index === rows.length - 1 ? "text-muted" : ""
              }`}
            >
              <dt className="pt-1 text-xs font-semibold leading-6 text-pitch sm:text-sm">{row.label}</dt>
              <dd className="arabic-ruling-copy max-w-[52rem] text-lg leading-9 text-chalk sm:text-xl sm:leading-[2.6rem]">
                {row.value}
              </dd>
            </div>
          ))}
        </dl>
      </article>

      <aside className="mt-6 border border-hairline bg-charcoal px-5 py-6 sm:px-7" aria-label={copy.provenanceTitle}>
        <p className="mb-5 text-xs font-medium uppercase tracking-[0.13em] text-muted">
          {copy.provenanceTitle}
        </p>
        <ol className="grid gap-5 md:grid-cols-3 md:gap-0">
          {provenance.map(({ Icon, text }, index) => (
            <li key={text} className="flex gap-3 md:border-e md:border-hairline md:px-5 md:first:ps-0 md:last:border-e-0 md:last:pe-0">
              <Icon className="mt-0.5 h-5 w-5 shrink-0 text-pitch" />
              <span className="text-sm leading-6 text-muted">
                <span className="technical-value me-2 text-[0.65rem] font-semibold text-chalk/70">0{index + 1}</span>
                {text}
              </span>
            </li>
          ))}
        </ol>
      </aside>
    </section>
  );
}
