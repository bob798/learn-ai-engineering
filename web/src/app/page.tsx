"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import {
  ArrowRight,
  BookOpen,
  Sparkles,
  Code2,
  Boxes,
  Compass,
  CheckCircle2,
  Clock,
  Star,
} from "lucide-react";
import { sections, docsBySection, interactiveBySection } from "@/lib/docs";

/* ── Section visuals ── */
const SECTION_META: Record<
  string,
  { icon: typeof Boxes; gradient: string; color: string }
> = {
  guide: {
    icon: Compass,
    gradient: "from-sky-500/10 to-cyan-500/5",
    color: "text-sky-600 dark:text-sky-400",
  },
  rag: {
    icon: BookOpen,
    gradient: "from-emerald-500/10 to-teal-500/5",
    color: "text-emerald-600 dark:text-emerald-400",
  },
  agent: {
    icon: Sparkles,
    gradient: "from-violet-500/10 to-fuchsia-500/5",
    color: "text-violet-600 dark:text-violet-400",
  },
  mcp: {
    icon: Boxes,
    gradient: "from-orange-500/10 to-amber-500/5",
    color: "text-orange-600 dark:text-orange-400",
  },
  "ai-programming": {
    icon: Code2,
    gradient: "from-amber-500/10 to-yellow-500/5",
    color: "text-amber-600 dark:text-amber-400",
  },
};

/* ── Learning path (condensed) ── */
const PATH_STEPS = [
  {
    num: "0",
    title: "方法论",
    desc: "5D 学习框架 + ATDF 拆解法",
    time: "25 min",
    href: "/guide/learning-path",
  },
  {
    num: "1",
    title: "RAG",
    desc: "让 AI 读懂你的数据",
    time: "5 篇核心",
    href: "/rag",
  },
  {
    num: "2",
    title: "Agent",
    desc: "让 AI 自己行动",
    time: "4 篇核心 + 代码",
    href: "/agent",
  },
  {
    num: "3",
    title: "MCP",
    desc: "标准化的工具接口",
    time: "4 篇核心",
    href: "/mcp",
  },
  {
    num: "4",
    title: "AI 编程",
    desc: "用 AI 工具构建产品",
    time: "4 篇核心",
    href: "/ai-programming",
  },
];

export default function HomePage() {
  return (
    <main className="overflow-hidden">
      {/* ═══════════════ HERO ═══════════════ */}
      <section className="relative">
        <div className="absolute inset-0 -z-10 bg-gradient-to-b from-orange-50/60 via-white to-white dark:from-orange-950/20 dark:via-zinc-950 dark:to-zinc-950" />
        <div className="absolute inset-0 -z-10 bg-[radial-gradient(circle_at_top,rgba(234,88,12,0.08),transparent_50%)]" />

        <div className="max-w-4xl mx-auto px-6 pt-24 pb-20">
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="text-center"
          >
            {/* Tagline */}
            <h1 className="text-4xl md:text-6xl font-bold tracking-tight mb-6">
              <span className="text-orange-600 dark:text-orange-400">
                AI 工程师
              </span>
              的学习手账
            </h1>

            <p className="text-lg md:text-xl text-zinc-600 dark:text-zinc-400 max-w-2xl mx-auto leading-relaxed mb-4">
              一个后端工程师转向 AI 的完整学习记录。
              <br className="hidden md:block" />
              不是专家教程，是<strong>同路人笔记</strong> ——
              保留困惑、保留追问、保留走过的弯路。
            </p>

            {/* Who is this for */}
            <div className="inline-flex flex-col items-start gap-1.5 text-sm text-zinc-500 dark:text-zinc-400 bg-zinc-50 dark:bg-zinc-900/50 border border-zinc-200 dark:border-zinc-800 rounded-xl px-6 py-4 mt-2 mb-8 text-left">
              <div className="text-xs font-semibold text-orange-600 dark:text-orange-400 uppercase tracking-wider mb-1">
                适合你，如果你
              </div>
              <div className="flex items-start gap-2">
                <CheckCircle2
                  size={16}
                  className="text-emerald-500 mt-0.5 shrink-0"
                />
                <span>会写代码（Python / JS / Go），想转向 AI 应用层</span>
              </div>
              <div className="flex items-start gap-2">
                <CheckCircle2
                  size={16}
                  className="text-emerald-500 mt-0.5 shrink-0"
                />
                <span>
                  被 RAG / Agent / MCP / Prompt Engineering 的术语淹没
                </span>
              </div>
              <div className="flex items-start gap-2">
                <CheckCircle2
                  size={16}
                  className="text-emerald-500 mt-0.5 shrink-0"
                />
                <span>不需要训练模型，需要用模型构建产品</span>
              </div>
            </div>

            {/* CTAs */}
            <div className="flex items-center justify-center gap-3 flex-wrap">
              <Link
                href="/guide/learning-path"
                className="inline-flex items-center gap-2 px-6 py-3 rounded-lg bg-orange-600 hover:bg-orange-700 text-white text-sm font-medium transition shadow-sm"
              >
                查看学习路线 <ArrowRight size={16} />
              </Link>
              <a
                href="https://github.com/bob798/learn-ai-engineering"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 px-6 py-3 rounded-lg border border-zinc-300 dark:border-zinc-700 hover:bg-zinc-100 dark:hover:bg-zinc-800 text-sm font-medium transition"
              >
                <Star size={16} /> Star on GitHub
              </a>
            </div>
          </motion.div>
        </div>
      </section>

      {/* ═══════════════ WHY THIS SITE ═══════════════ */}
      <section className="max-w-4xl mx-auto px-6 pb-20">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-60px" }}
          transition={{ duration: 0.4 }}
        >
          <h2 className="text-2xl font-bold text-center mb-10">
            为什么看这里，而不是看教程
          </h2>

          <div className="grid gap-4 md:grid-cols-3">
            {[
              {
                title: "不跳步",
                desc: "专家觉得你应该知道的，你不知道。这里每一篇都是我自己卡住过、查了很久、跑通了之后写的。",
              },
              {
                title: "能跑通",
                desc: "代码给完整命令，不给伪代码。论文给逐句中文注解，不给摘要转述。",
              },
              {
                title: "说不确定",
                desc: "遇到没搞清楚的地方会标注，而不是糊弄过去。踩过的坑比正确答案更有价值。",
              },
            ].map((item, i) => (
              <motion.div
                key={item.title}
                initial={{ opacity: 0, y: 16 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.35, delay: i * 0.08 }}
                className="border border-zinc-200 dark:border-zinc-800 rounded-xl p-6"
              >
                <h3 className="text-lg font-bold mb-2 text-orange-600 dark:text-orange-400">
                  {item.title}
                </h3>
                <p className="text-sm text-zinc-600 dark:text-zinc-400 leading-relaxed">
                  {item.desc}
                </p>
              </motion.div>
            ))}
          </div>
        </motion.div>
      </section>

      {/* ═══════════════ LEARNING PATH ═══════════════ */}
      <section className="max-w-4xl mx-auto px-6 pb-24">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-60px" }}
          transition={{ duration: 0.4 }}
        >
          <div className="text-center mb-10">
            <div className="text-xs uppercase tracking-wider text-orange-600 font-semibold mb-2">
              推荐路线
            </div>
            <h2 className="text-2xl font-bold">
              从 0 到 1 的学习路径
            </h2>
            <p className="text-sm text-zinc-500 mt-2">
              19 篇核心文档，约 6-8 小时。按顺序走，或跳到你需要的章节。
            </p>
          </div>

          {/* Path steps */}
          <div className="relative">
            {/* Connecting line */}
            <div className="absolute left-[23px] top-8 bottom-8 w-px bg-zinc-200 dark:bg-zinc-800 hidden md:block" />

            <div className="space-y-3">
              {PATH_STEPS.map((step, i) => (
                <motion.div
                  key={step.num}
                  initial={{ opacity: 0, x: -12 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.3, delay: i * 0.06 }}
                >
                  <Link
                    href={step.href}
                    className="group flex items-center gap-4 p-4 rounded-xl border border-zinc-200 dark:border-zinc-800 hover:border-orange-400 dark:hover:border-orange-600 hover:shadow-md transition-all"
                  >
                    {/* Number circle */}
                    <div className="size-12 shrink-0 rounded-full bg-orange-50 dark:bg-orange-950/30 border-2 border-orange-200 dark:border-orange-800 flex items-center justify-center text-orange-600 dark:text-orange-400 font-bold text-lg group-hover:bg-orange-100 dark:group-hover:bg-orange-900/40 transition">
                      {step.num}
                    </div>

                    {/* Content */}
                    <div className="flex-1 min-w-0">
                      <div className="font-semibold text-zinc-900 dark:text-zinc-100">
                        {step.title}
                      </div>
                      <div className="text-sm text-zinc-500">
                        {step.desc}
                      </div>
                    </div>

                    {/* Time badge + arrow */}
                    <div className="hidden sm:flex items-center gap-3 shrink-0">
                      <span className="inline-flex items-center gap-1 text-xs text-zinc-400">
                        <Clock size={12} />
                        {step.time}
                      </span>
                      <ArrowRight
                        size={16}
                        className="text-zinc-300 group-hover:text-orange-500 group-hover:translate-x-1 transition-all"
                      />
                    </div>
                  </Link>
                </motion.div>
              ))}
            </div>
          </div>

          <div className="text-center mt-8">
            <Link
              href="/guide/learning-path"
              className="text-sm text-orange-600 hover:text-orange-700 dark:text-orange-400 dark:hover:text-orange-300 font-medium"
            >
              查看完整路线图（含选修 + 待写内容）&rarr;
            </Link>
          </div>
        </motion.div>
      </section>

      {/* ═══════════════ SECTIONS GRID ═══════════════ */}
      <section className="max-w-6xl mx-auto px-6 pb-24">
        <div className="text-center mb-10">
          <div className="text-xs uppercase tracking-wider text-orange-600 font-semibold mb-2">
            全部内容
          </div>
          <h2 className="text-2xl font-bold">按主题浏览</h2>
        </div>

        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {sections
            .filter((s) => s.slug !== "guide")
            .map((section, i) => {
              const docCount = docsBySection(section.slug).length;
              const vizCount = interactiveBySection(section.slug).length;
              const meta = SECTION_META[section.slug] ?? {
                icon: Boxes,
                gradient: "",
                color: "text-zinc-600",
              };
              const Icon = meta.icon;
              return (
                <motion.div
                  key={section.slug}
                  initial={{ opacity: 0, y: 16 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true, margin: "-40px" }}
                  transition={{ duration: 0.35, delay: i * 0.06 }}
                >
                  <Link
                    href={`/${section.slug}`}
                    className={`group block border border-zinc-200 dark:border-zinc-800 rounded-2xl p-6 hover:border-orange-400 dark:hover:border-orange-600 hover:shadow-lg hover:-translate-y-0.5 transition-all bg-gradient-to-br ${meta.gradient}`}
                  >
                    <div className="flex items-start justify-between mb-3">
                      <div
                        className={`size-9 rounded-lg bg-white/80 dark:bg-zinc-900/80 backdrop-blur flex items-center justify-center shadow-sm ${meta.color}`}
                      >
                        <Icon size={18} />
                      </div>
                      <ArrowRight
                        size={16}
                        className="text-zinc-300 group-hover:text-orange-500 group-hover:translate-x-1 transition-all"
                      />
                    </div>
                    <h3 className="text-xl font-bold mb-1">{section.title}</h3>
                    <p className="text-sm text-zinc-500 dark:text-zinc-400 mb-4 leading-relaxed">
                      {section.description}
                    </p>
                    <div className="flex gap-3 text-xs text-zinc-400">
                      <span>{docCount} 篇</span>
                      {vizCount > 0 && <span>{vizCount} 个交互</span>}
                    </div>
                  </Link>
                </motion.div>
              );
            })}
        </div>
      </section>

      {/* ═══════════════ FOOTER ═══════════════ */}
      <footer className="border-t border-zinc-200 dark:border-zinc-800 py-10">
        <div className="max-w-6xl mx-auto px-6 flex flex-col md:flex-row items-center justify-between gap-4 text-sm text-zinc-500">
          <div>
            &copy; {new Date().getFullYear()} bob798 &middot;{" "}
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
            <a href="/guide/learning-path" className="hover:text-orange-600 transition">
              学习路线
            </a>
          </div>
        </div>
      </footer>
    </main>
  );
}
