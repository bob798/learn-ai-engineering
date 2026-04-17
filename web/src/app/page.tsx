import Link from "next/link";
import { sections, docsBySection, interactiveBySection } from "@/lib/docs";

export default function HomePage() {
  return (
    <div className="max-w-5xl mx-auto px-6 py-16">
      <header className="mb-16">
        <h1 className="text-5xl font-bold tracking-tight mb-4">AI Handbook</h1>
        <p className="text-xl text-zinc-600 dark:text-zinc-400">
          AI 工程师的完整学习记录 · 从协议到生态 · 从方法论到实战
        </p>
        <p className="mt-3 text-sm text-zinc-500">
          包含深度追问、真实误解纠错、可交互笔记
        </p>
      </header>

      <div className="grid gap-6 md:grid-cols-2">
        {sections.map((section) => {
          const docCount = docsBySection(section.slug).length;
          const vizCount = interactiveBySection(section.slug).length;
          return (
            <Link
              key={section.slug}
              href={`/${section.slug}`}
              className="group block border border-zinc-200 dark:border-zinc-800 rounded-2xl p-7 hover:border-orange-500 hover:shadow-lg transition"
            >
              <h2 className="text-2xl font-bold mb-2 group-hover:text-orange-600 dark:group-hover:text-orange-400 transition">
                {section.title}
              </h2>
              <p className="text-sm text-zinc-600 dark:text-zinc-400 mb-4">
                {section.description}
              </p>
              <div className="flex gap-4 text-xs text-zinc-500">
                <span>{docCount} 篇文档</span>
                {vizCount > 0 && <span>{vizCount} 个交互笔记</span>}
              </div>
            </Link>
          );
        })}
      </div>
    </div>
  );
}
