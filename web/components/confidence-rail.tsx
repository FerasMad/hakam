export function ConfidenceRail({
  value,
  label,
  thresholdLabel,
  threshold = 0.6,
  tone = "pitch",
}: {
  value: number;
  label: string;
  thresholdLabel: string;
  threshold?: number;
  tone?: "pitch" | "copper";
}) {
  const percent = Math.round(value * 100);
  const fill = tone === "copper" ? "bg-copper" : "bg-pitch";

  return (
    <div>
      <div className="mb-3 flex items-end justify-between gap-4">
        <span className="text-sm text-muted">{label}</span>
        <span className="technical-value text-2xl font-medium text-chalk" aria-label={`${percent}%`}>
          {percent}%
        </span>
      </div>
      <div
        className="relative h-1.5 bg-hairline"
        role="meter"
        aria-label={label}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={percent}
      >
        <span className={`absolute inset-block-0 inset-inline-start-0 ${fill}`} style={{ width: `${percent}%` }} />
        <span
          className="absolute -top-1 h-3.5 w-px bg-chalk/70"
          style={{ insetInlineStart: `${Math.round(threshold * 100)}%` }}
        />
      </div>
      <div className="relative mt-2 h-4 text-[0.65rem] text-muted">
        <span
          className="threshold-label absolute whitespace-nowrap [unicode-bidi:isolate]"
          dir="auto"
          style={{ insetInlineStart: `${Math.round(threshold * 100)}%` }}
        >
          {thresholdLabel}
        </span>
      </div>
    </div>
  );
}
