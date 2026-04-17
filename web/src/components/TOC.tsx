import type { Heading } from "@/lib/docs";

export function TOC({ headings }: { headings: Heading[] }) {
  const items = headings.filter((h) => h.depth >= 2 && h.depth <= 3);
  if (items.length === 0) return null;
  return (
    <aside className="hidden xl:block w-56 shrink-0 pl-8 py-10 sticky top-0 self-start max-h-screen overflow-y-auto">
      <div className="text-xs font-semibold uppercase tracking-wide text-zinc-500 mb-3">
        On this page
      </div>
      <ul className="space-y-1.5 text-sm">
        {items.map((h, i) => (
          <li key={`${h.id}-${i}`} className={h.depth === 3 ? "pl-3" : ""}>
            <a
              href={`#${h.id}`}
              className="block text-zinc-600 hover:text-orange-600 dark:text-zinc-400 dark:hover:text-orange-400 transition"
            >
              {h.text}
            </a>
          </li>
        ))}
      </ul>
    </aside>
  );
}
