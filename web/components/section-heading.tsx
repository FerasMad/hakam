export function SectionHeading({
  id,
  marker,
  title,
  description,
}: {
  id?: string;
  marker: string;
  title: string;
  description?: string;
}) {
  return (
    <header className="mb-8 border-b border-hairline pb-5 sm:mb-10 sm:flex sm:items-end sm:justify-between sm:gap-8">
      <div>
        <p className="mb-3 text-[0.68rem] font-semibold uppercase tracking-broadcast text-pitch">
          {marker}
        </p>
        <h2 id={id} className="text-2xl font-medium tracking-[-0.03em] text-chalk sm:text-3xl">{title}</h2>
      </div>
      {description ? (
        <p className="mt-3 max-w-xl text-sm leading-6 text-muted sm:mt-0 sm:text-end">
          {description}
        </p>
      ) : null}
    </header>
  );
}
