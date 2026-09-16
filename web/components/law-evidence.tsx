import { ChevronIcon, DocumentIcon } from "@/components/icons";
import { SectionHeading } from "@/components/section-heading";
import { uiCopy } from "@/lib/i18n";
import type { Language, LawArticle } from "@/lib/types";

function PageReferences({ article, language }: { article: LawArticle; language: Language }) {
  const copy = uiCopy[language];
  if (!article.sourcePages?.ar?.length && !article.sourcePages?.en?.length) return null;

  return (
    <p className="mt-5 flex flex-wrap gap-x-5 gap-y-2 text-[0.68rem] text-muted">
      <span>{copy.sourcePages}</span>
      {article.sourcePages.ar?.length ? (
        <span>
          {copy.arabicSource} <bdi className="technical-value">{article.sourcePages.ar.join(", ")}</bdi>
        </span>
      ) : null}
      {article.sourcePages.en?.length ? (
        <span>
          {copy.englishSource} <bdi className="technical-value">{article.sourcePages.en.join(", ")}</bdi>
        </span>
      ) : null}
    </p>
  );
}

function ArticleText({ article, language, compact = false }: { article: LawArticle; language: Language; compact?: boolean }) {
  if (language === "ar") {
    return (
      <>
        <p lang="ar" dir="rtl" className={`arabic-editorial text-chalk ${compact ? "text-base leading-8" : "text-xl leading-10 sm:text-2xl sm:leading-[1.9]"}`}>
          {article.textAr}
        </p>
        {article.textEn ? (
          <p dir="ltr" className="mt-5 border-s border-hairline ps-4 font-sans text-sm leading-7 text-muted">
            {article.textEn}
          </p>
        ) : null}
      </>
    );
  }

  return (
    <>
      {article.textEn ? (
        <p dir="ltr" className={`text-chalk ${compact ? "text-sm leading-7" : "text-lg leading-8 sm:text-xl sm:leading-9"}`}>
          {article.textEn}
        </p>
      ) : null}
      <p lang="ar" dir="rtl" className="arabic-editorial mt-5 border-s border-hairline ps-4 text-base leading-8 text-muted">
        {article.textAr}
      </p>
    </>
  );
}

function ArticleTitle({ article, language, className }: { article: LawArticle; language: Language; className: string }) {
  const useEnglish = language === "en" && Boolean(article.titleEn);
  return (
    <span
      lang={useEnglish ? "en" : "ar"}
      dir={useEnglish ? "ltr" : "rtl"}
      className={`${useEnglish ? "font-sans" : "arabic-editorial"} ${className}`}
    >
      {useEnglish ? article.titleEn : article.titleAr}
    </span>
  );
}

export function LawEvidence({ language, articles }: { language: Language; articles: LawArticle[] }) {
  const copy = uiCopy[language];
  const [primary, ...supporting] = articles;
  if (!primary) return null;

  return (
    <section className="mt-24 sm:mt-32" aria-labelledby="law-evidence-title">
      <SectionHeading
        id="law-evidence-title"
        marker={copy.lawLayer}
        title={copy.lawTitle}
        description={copy.supportsRuling}
      />

      <article className="relative overflow-hidden border border-hairline bg-charcoal">
        <div className="grid lg:grid-cols-[15rem_1fr]">
          <header className="border-b border-hairline p-6 lg:border-b-0 lg:border-e lg:p-8">
            <DocumentIcon className="mb-12 h-7 w-7 text-pitch" />
            <p className="text-[0.65rem] font-semibold uppercase tracking-broadcast text-muted">
              {copy.mainEvidence}
            </p>
            <p className="technical-value mt-3 font-sans text-4xl font-semibold tracking-[-0.06em] text-chalk">
              {primary.law.toUpperCase()}
            </p>
            <p className="technical-value mt-3 font-sans text-xs leading-5 text-muted">{primary.section}</p>
            <p className="technical-value mt-8 break-all font-mono text-[0.65rem] text-muted">{primary.id}</p>
          </header>

          <div className="p-6 sm:p-8 lg:p-10 xl:p-12">
            <h3 className="text-2xl font-semibold leading-10 text-chalk sm:text-3xl">
              <ArticleTitle article={primary} language={language} className="block" />
            </h3>
            <div className="mt-7 border-t border-hairline pt-7">
              <ArticleText article={primary} language={language} />
              <PageReferences article={primary} language={language} />
            </div>
          </div>
        </div>
      </article>

      {supporting.length ? (
        <div className="mt-5 border-y border-hairline">
          <p className="border-b border-hairline py-4 text-xs font-medium uppercase tracking-[0.13em] text-muted">
            {copy.supportingArticles}
          </p>
          {supporting.map((article) => (
            <details key={article.id} className="group border-b border-hairline last:border-b-0">
              <summary className="flex min-h-16 cursor-pointer items-center justify-between gap-5 py-4 text-start transition-colors hover:text-pitch">
                <span className="flex min-w-0 items-center gap-4">
                  <span className="technical-value shrink-0 font-sans text-[0.65rem] uppercase tracking-[0.12em] text-muted">
                    {article.law}
                  </span>
                  <ArticleTitle article={article} language={language} className="truncate text-base font-medium text-chalk" />
                </span>
                <ChevronIcon className="h-4 w-4 shrink-0 text-muted transition-transform group-open:rotate-180" />
              </summary>
              <div className="pb-7 ps-0 sm:ps-20">
                <p className="technical-value mb-4 font-mono text-[0.65rem] text-muted">{article.id}</p>
                <ArticleText article={article} language={language} compact />
                <PageReferences article={article} language={language} />
              </div>
            </details>
          ))}
        </div>
      ) : null}
    </section>
  );
}
