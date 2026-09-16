import { uiCopy } from "@/lib/i18n";
import type { Language } from "@/lib/types";

const teamMembers = [
  {
    name: "Anas Alzahrani",
    linkedin: "https://www.linkedin.com/in/anas-alzahrani-932b38319",
  },
  {
    name: "Feras Madkhali",
    linkedin: "https://www.linkedin.com/in/feras-madkhali-0b639b398",
  },
  {
    name: "Bader Aljoudi",
    linkedin: "https://www.linkedin.com/in/bader-aljoudi-139a62299",
  },
] as const;

export function Footer({ language }: { language: Language }) {
  const copy = uiCopy[language];

  return (
    <footer className="mt-24 border-t border-hairline sm:mt-32">
      <div className="mx-auto max-w-canvas px-5 py-10 sm:px-8 lg:px-12">
        <div className="grid gap-10 lg:grid-cols-[minmax(0,1.1fr)_minmax(20rem,0.9fr)] lg:gap-16">
          <div className="max-w-xl">
            <p className="mb-2 text-[0.68rem] font-semibold uppercase tracking-broadcast text-muted">
              {copy.researchLabel}
            </p>
            <p className="text-sm leading-6 text-muted">{copy.footerText}</p>
          </div>

          <div>
            <p className="mb-2 text-[0.68rem] font-semibold uppercase tracking-broadcast text-muted">
              {copy.teamLabel}
            </p>
            <ul className="grid gap-2 sm:grid-cols-3 lg:grid-cols-1">
              {teamMembers.map((member) => (
                <li key={member.linkedin}>
                  <a
                    href={member.linkedin}
                    target="_blank"
                    rel="noreferrer noopener"
                    aria-label={`${member.name} — LinkedIn`}
                    dir="ltr"
                    className="group flex min-h-11 items-center justify-between gap-3 border-b border-hairline py-2 text-sm text-chalk transition-colors hover:border-copper hover:text-white"
                  >
                    <span>{member.name}</span>
                    <span
                      aria-hidden="true"
                      className="text-copper transition-transform group-hover:-translate-y-0.5 group-hover:translate-x-0.5"
                    >
                      ↗
                    </span>
                  </a>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </footer>
  );
}
