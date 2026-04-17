import Link from "next/link";
import { notFound } from "next/navigation";
import {
  docs,
  sections,
  findDoc,
  docsBySection,
  interactiveBySection,
} from "@/lib/docs";
import { Sidebar } from "@/components/Sidebar";
import { TOC } from "@/components/TOC";

export function generateStaticParams() {
  const paths: { slug: string[] }[] = [];
  for (const s of sections) paths.push({ slug: [s.slug] });
  for (const d of docs) paths.push({ slug: d.slug.split("/") });
  return paths;
}

interface PageProps {
  params: Promise<{ slug: string[] }>;
}

export default async function DocPage({ params }: PageProps) {
  const { slug } = await params;

  // Section index: /mcp, /agent, etc.
  const section = sections.find((s) => s.slug === slug[0]);
  if (slug.length === 1 && section) {
    const sectionDocs = docsBySection(section.slug);
    const viz = interactiveBySection(section.slug);
    return (
      <div className="flex">
        <Sidebar activeSection={section.slug} />
        <main className="flex-1 min-w-0 max-w-4xl px-10 py-12">
          <header className="mb-10 pb-8 border-b border-zinc-200 dark:border-zinc-800">
            <div className="text-xs font-semibold uppercase tracking-wider text-orange-600 dark:text-orange-400 mb-2">
              Section
            </div>
            <h1 className="text-4xl font-bold mb-3">{section.title}</h1>
            <p className="text-zinc-600 dark:text-zinc-400">{section.description}</p>
          </header>

          <section className="mb-12">
            <h2 className="text-sm font-semibold uppercase tracking-wider text-zinc-500 mb-4">
              文档
            </h2>
            <ul className="space-y-2">
              {sectionDocs.map((d) => (
                <li key={d.slug}>
                  <Link
                    href={`/${d.slug}`}
                    className="block p-4 border border-zinc-200 dark:border-zinc-800 rounded-lg hover:border-orange-500 hover:shadow-sm transition"
                  >
                    <div className="font-medium">{d.title}</div>
                    {d.description && (
                      <div className="text-sm text-zinc-500 mt-1">{d.description}</div>
                    )}
                    <div className="text-xs text-zinc-400 mt-1 font-mono">{d.path}</div>
                  </Link>
                </li>
              ))}
            </ul>
          </section>

          {viz.length > 0 && (
            <section>
              <h2 className="text-sm font-semibold uppercase tracking-wider text-zinc-500 mb-4">
                交互笔记（HTML）
              </h2>
              <ul className="grid gap-2 sm:grid-cols-2">
                {viz.map((v) => (
                  <li key={v.file}>
                    <a
                      href={v.href}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="block p-3 border border-zinc-200 dark:border-zinc-800 rounded-lg hover:border-orange-500 transition text-sm"
                    >
                      <div className="font-medium">{v.name}</div>
                      <div className="text-xs text-zinc-500 mt-0.5 font-mono">{v.file}</div>
                    </a>
                  </li>
                ))}
              </ul>
            </section>
          )}
        </main>
      </div>
    );
  }

  // Doc page
  const doc = findDoc(slug);
  if (!doc) notFound();

  return (
    <div className="flex">
      <Sidebar activeSection={doc.section} />
      <main className="flex-1 min-w-0 max-w-4xl px-10 py-12">
        <div className="mb-6 text-xs text-zinc-500 font-mono">{doc.path}</div>
        <article className="prose-doc" dangerouslySetInnerHTML={{ __html: doc.contentHtml }} />
      </main>
      <TOC headings={doc.headings} />
    </div>
  );
}
