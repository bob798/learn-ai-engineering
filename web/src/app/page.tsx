"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import { ArrowRight, BookOpen, Sparkles, Code2, Boxes } from "lucide-react";
import { sections, docsBySection, interactiveBySection } from "@/lib/docs";
import { MCPNPlusM } from "@/components/visualizations/MCPNPlusM";

const SECTION_ICONS: Record<string, typeof Boxes> = {
  mcp: Boxes,
  agent: Sparkles,
  rag: BookOpen,
  "ai-programming": Code2,
};

const SECTION_GRADIENTS: Record<string, string> = {
  mcp: "from-orange-500/10 to-amber-500/5",
  agent: "from-violet-500/10 to-fuchsia-500/5",
  rag: "from-emerald-500/10 to-teal-500/5",
  "ai-programming": "from-amber-500/10 to-yellow-500/5",
};

export default function HomePage() {
  return (
    <main className="overflow-hidden">
      {/* HERO */}
      <section className="relative">
        <div className="absolute inset-0 -z-10 bg-gradient-to-b from-orange-50/60 via-white to-white dark:from-orange-950/20 dark:via-zinc-950 dark:to-zinc-950" />
        <div className="absolute inset-0 -z-10 bg-[radial-gradient(circle_at_top,rgba(234,88,12,0.08),transparent_50%)]" />

        <div className="max-w-6xl mx-auto px-6 pt-24 pb-16">
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="text-center"
          >
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium bg-orange-100/80 dark:bg-orange-950/40 text-orange-700 dark:text-orange-400 mb-6">
              <span className="size-1.5 rounded-full bg-orange-500 animate-pulse" />
              52 docs · 27 interactive · 19 runnable examples
            </div>

            <h1 className="text-5xl md:text-7xl font-bold tracking-tight mb-6">
              Learn AI Engineering
            </h1>

            <p className="text-xl md:text-2xl text-zinc-600 dark:text-zinc-400 italic mb-3">
              by reading someone else&apos;s mistakes.
            </p>

            <p className="text-base text-zinc-500 max-w-2xl mx-auto leading-relaxed">
              AI 工程师的"不整理"笔记本 —— 保留困惑、保留追问、保留误解。
              <br />
              比标准答案更实用，比 awesome list 更体系化。
            </p>

            <div className="mt-10 flex items-center justify-center gap-3 flex-wrap">
              <Link
                href="/mcp"
                className="inline-flex items-center gap-2 px-5 py-2.5 rounded-lg bg-orange-600 hover:bg-orange-700 text-white text-sm font-medium transition shadow-sm"
              >
                从 MCP 开始 <ArrowRight size={16} />
              </Link>
              <a
                href="https://github.com/bob798/learn-ai-engineering"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 px-5 py-2.5 rounded-lg border border-zinc-300 dark:border-zinc-700 hover:bg-zinc-100 dark:hover:bg-zinc-800 text-sm font-medium transition"
              >
                ⭐ Star on GitHub
              </a>
            </div>
          </motion.div>
        </div>
      </section>

      {/* INTERACTIVE: MCP N+M animation */}
      <section className="max-w-5xl mx-auto px-6 pb-20">
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-80px" }}
          transition={{ duration: 0.5 }}
        >
          <div className="text-center mb-8">
            <div className="text-xs uppercase tracking-wider text-orange-600 font-semibold mb-2">
              示例：用动画讲清楚一个抽象概念
            </div>
            <h2 className="text-2xl md:text-3xl font-bold mb-2">
              MCP 为什么是"AI 时代的 USB-C"
            </h2>
            <p className="text-sm text-zinc-500">
              点播放，5 步理解 N×M → N+M 的协议价值
            </p>
          </div>
          <MCPNPlusM />
        </motion.div>
      </section>

      {/* SECTIONS */}
      <section className="max-w-6xl mx-auto px-6 pb-24">
        <div className="text-center mb-12">
          <div className="text-xs uppercase tracking-wider text-orange-600 font-semibold mb-2">
            内容主题
          </div>
          <h2 className="text-3xl font-bold">4 个主题，1 张知识地图</h2>
        </div>

        <div className="grid gap-5 md:grid-cols-2">
          {sections.map((section, i) => {
            const docCount = docsBySection(section.slug).length;
            const vizCount = interactiveBySection(section.slug).length;
            const Icon = SECTION_ICONS[section.slug] ?? Boxes;
            const gradient = SECTION_GRADIENTS[section.slug] ?? "";
            return (
              <motion.div
                key={section.slug}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, margin: "-50px" }}
                transition={{ duration: 0.4, delay: i * 0.08 }}
              >
                <Link
                  href={`/${section.slug}`}
                  className={`group block relative overflow-hidden border border-zinc-200 dark:border-zinc-800 rounded-2xl p-7 hover:border-orange-500 hover:shadow-xl hover:-translate-y-0.5 transition-all bg-gradient-to-br ${gradient}`}
                >
                  <div className="flex items-start justify-between mb-4">
                    <div className="size-10 rounded-lg bg-white/80 dark:bg-zinc-900/80 backdrop-blur flex items-center justify-center text-orange-600 dark:text-orange-400 shadow-sm">
                      <Icon size={20} />
                    </div>
                    <ArrowRight
                      size={18}
                      className="text-zinc-400 group-hover:text-orange-600 group-hover:translate-x-1 transition-all"
                    />
                  </div>
                  <h3 className="text-2xl font-bold mb-2">{section.title}</h3>
                  <p className="text-sm text-zinc-600 dark:text-zinc-400 mb-5 leading-relaxed">
                    {section.description}
                  </p>
                  <div className="flex gap-4 text-xs text-zinc-500">
                    <span>{docCount} 篇文档</span>
                    {vizCount > 0 && <span>{vizCount} 个交互笔记</span>}
                  </div>
                </Link>
              </motion.div>
            );
          })}
        </div>
      </section>

      {/* MANIFESTO */}
      <section className="max-w-4xl mx-auto px-6 pb-24">
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
          className="border-l-4 border-orange-500 pl-6 py-2"
        >
          <div className="text-xs uppercase tracking-wider text-orange-600 font-semibold mb-3">
            写作风格
          </div>
          <p className="text-lg leading-relaxed text-zinc-700 dark:text-zinc-300 mb-4">
            这些笔记<strong>不是</strong>抄来的定义汇总，<strong>不是</strong>给初学者的导论，
            <strong>不是</strong>最佳实践总结 —— 那些只是结果，不是过程。
          </p>
          <p className="text-lg leading-relaxed text-zinc-700 dark:text-zinc-300">
            这些笔记<strong className="text-orange-600 dark:text-orange-400">是</strong>
            一个 AI 工程师从完全不懂到理解的<strong className="text-orange-600 dark:text-orange-400">完整追问链</strong>，
            一个个&quot;本来以为是 X，结果是 Y&quot;的<strong className="text-orange-600 dark:text-orange-400">反转记录</strong>。
          </p>
        </motion.div>
      </section>

      {/* FOOTER */}
      <footer className="border-t border-zinc-200 dark:border-zinc-800 py-10">
        <div className="max-w-6xl mx-auto px-6 flex flex-col md:flex-row items-center justify-between gap-4 text-sm text-zinc-500">
          <div>
            © {new Date().getFullYear()} bob798 ·{" "}
            <a
              href="https://bob798.github.io"
              className="hover:text-orange-600 transition"
            >
              AI 开发日记
            </a>
          </div>
          <div className="flex gap-5">
            <a
              href="https://github.com/bob798/learn-ai-engineering"
              className="hover:text-orange-600 transition"
            >
              GitHub
            </a>
            <a href="/mcp" className="hover:text-orange-600 transition">
              开始阅读
            </a>
          </div>
        </div>
      </footer>
    </main>
  );
}
