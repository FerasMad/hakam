import type { ReactNode } from "react";

export function ReviewFrame({
  children,
  className = "",
  tone = "pitch",
}: {
  children: ReactNode;
  className?: string;
  tone?: "pitch" | "copper" | "danger";
}) {
  const toneClass =
    tone === "copper" ? "text-copper" : tone === "danger" ? "text-danger" : "text-pitch";

  return (
    <div className={`review-frame ${className}`}>
      <span aria-hidden="true" className={`review-corner review-corner--tl ${toneClass}`} />
      <span aria-hidden="true" className={`review-corner review-corner--tr ${toneClass}`} />
      <span aria-hidden="true" className={`review-corner review-corner--bl ${toneClass}`} />
      <span aria-hidden="true" className={`review-corner review-corner--br ${toneClass}`} />
      {children}
    </div>
  );
}
