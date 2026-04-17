import { notFound } from "next/navigation";
import {
  docs,
  sections,
  findDoc,
  groupSectionDocs,
  interactiveBySection,
} from "@/lib/docs";
import { Sidebar } from "@/components/Sidebar";
import { TOC } from "@/components/TOC";
import { SectionHero } from "@/components/SectionHero";
import { DocCard, InteractiveCard } from "@/components/DocCard";

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
    const { readme, groups, loose } = groupSectionDocs(section.slug);
    const viz = interactiveBySection(section.slug);
    const docCount =
      (readme ? 1 : 0) + loose.length + groups.reduce((n, g) => n + g.docs.length, 0);

    let cardIndex = 0;
    return (
      <div className="flex">
        <Sidebar activeSection={section.slug} />
        <main className="flex-1 min-w-0 max-w-5xl px-8 lg:px-12 py-12">
          <SectionHero
            section={section.slug}
            title={section.title}
            description={section.description}
            docCount={docCount}
            vizCount={viz.length}
          />

          {readme && (
            <article
              className="prose-doc mb-12 pb-10 border-b border-zinc-200 dark:border-zinc-800"
              dangerouslySetInnerHTML={{ __html: readme.contentHtml }}
            />
          )}

          {groups.map((g) => (
            <section key={g.key} className="mb-10">
              <div className="flex items-baseline justify-between mb-4">
                <h2 className="text-lg font-bold">{g.label}</h2>
                <span className="text-xs text-zinc-500">{g.docs.length} 篇</span>
              </div>
              <div className="grid gap-3 md:grid-cols-2">
                {g.docs.map((d) => (
                  <DocCard key={d.slug} doc={d} index={cardIndex++} />
                ))}
              </div>
            </section>
          ))}

          {loose.length > 0 && (
            <section className="mb-10">
              <h2 className="text-lg font-bold mb-4">其他</h2>
              <div className="grid gap-3 md:grid-cols-2">
                {loose.map((d) => (
                  <DocCard key={d.slug} doc={d} index={cardIndex++} />
                ))}
              </div>
            </section>
          )}

          {viz.length > 0 && (
            <section className="mt-14 pt-10 border-t border-zinc-200 dark:border-zinc-800">
              <div className="flex items-baseline justify-between mb-4">
                <h2 className="text-lg font-bold">交互笔记（HTML）</h2>
                <span className="text-xs text-zinc-500">{viz.length} 个</span>
              </div>
              <p className="text-sm text-zinc-500 mb-4">
                浏览器直接打开，含动画、可交互组件，部分主题用动画讲解比文字更直观。
              </p>
              <div className="grid gap-2 md:grid-cols-2">
                {viz.map((v, i) => (
                  <InteractiveCard key={v.file} asset={v} index={i} />
                ))}
              </div>
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
