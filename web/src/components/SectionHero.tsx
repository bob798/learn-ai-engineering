"use client";

import { motion } from "framer-motion";
import { BookOpen, Sparkles, Code2, Boxes, type LucideIcon } from "lucide-react";

const ICONS: Record<string, LucideIcon> = {
  mcp: Boxes,
  agent: Sparkles,
  rag: BookOpen,
  "ai-programming": Code2,
};

const GRADIENTS: Record<string, string> = {
  mcp: "from-orange-500/15 via-amber-500/10 to-transparent",
  agent: "from-violet-500/15 via-fuchsia-500/10 to-transparent",
  rag: "from-emerald-500/15 via-teal-500/10 to-transparent",
  "ai-programming": "from-amber-500/15 via-yellow-500/10 to-transparent",
};

const ICON_COLORS: Record<string, string> = {
  mcp: "text-orange-600 dark:text-orange-400",
  agent: "text-violet-600 dark:text-violet-400",
  rag: "text-emerald-600 dark:text-emerald-400",
  "ai-programming": "text-amber-600 dark:text-amber-400",
};

interface Props {
  section: string;
  title: string;
  description: string;
  docCount: number;
  vizCount: number;
}

export function SectionHero({ section, title, description, docCount, vizCount }: Props) {
  const Icon = ICONS[section] ?? Boxes;
  const gradient = GRADIENTS[section] ?? "";
  const iconColor = ICON_COLORS[section] ?? "text-orange-600";

  return (
    <header className={`relative overflow-hidden rounded-2xl border border-zinc-200 dark:border-zinc-800 px-8 py-10 mb-10 bg-gradient-to-br ${gradient}`}>
      <div className="absolute -top-12 -right-12 size-48 rounded-full bg-gradient-to-br from-current to-transparent opacity-[0.04] pointer-events-none" />
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
      >
        <div className="flex items-start gap-5">
          <div className={`shrink-0 size-14 rounded-xl bg-white/80 dark:bg-zinc-900/80 backdrop-blur flex items-center justify-center ${iconColor} shadow-sm`}>
            <Icon size={26} />
          </div>
          <div className="flex-1 min-w-0">
            <div className="text-xs uppercase tracking-wider text-zinc-500 font-semibold mb-1">
              Section
            </div>
            <h1 className="text-4xl font-bold mb-2">{title}</h1>
            <p className="text-zinc-600 dark:text-zinc-400 text-base">{description}</p>
            <div className="mt-4 flex gap-5 text-sm text-zinc-500">
              <span><strong className="text-zinc-900 dark:text-zinc-100 font-bold">{docCount}</strong> 篇文档</span>
              {vizCount > 0 && (
                <span><strong className="text-zinc-900 dark:text-zinc-100 font-bold">{vizCount}</strong> 个交互笔记</span>
              )}
            </div>
          </div>
        </div>
      </motion.div>
    </header>
  );
}
