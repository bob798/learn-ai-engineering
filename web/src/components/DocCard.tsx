"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import { FileText, ArrowUpRight } from "lucide-react";
import type { DocNode, InteractiveAsset } from "@/lib/docs";

export function DocCard({ doc, index = 0 }: { doc: DocNode; index?: number }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-30px" }}
      transition={{ duration: 0.3, delay: Math.min(index * 0.04, 0.3) }}
    >
      <Link
        href={`/${doc.slug}`}
        className="group block p-4 border border-zinc-200 dark:border-zinc-800 rounded-xl hover:border-orange-500 hover:shadow-md hover:-translate-y-0.5 transition-all bg-white dark:bg-zinc-950"
      >
        <div className="flex items-start gap-3">
          <FileText size={16} className="shrink-0 mt-0.5 text-zinc-400 group-hover:text-orange-500 transition" />
          <div className="flex-1 min-w-0">
            <div className="font-semibold text-sm leading-snug mb-1 group-hover:text-orange-600 dark:group-hover:text-orange-400 transition">
              {doc.title}
            </div>
            {doc.description && (
              <div className="text-xs text-zinc-500 leading-relaxed mb-1.5">
                {doc.description}
              </div>
            )}
            <div className="text-xs text-zinc-400 font-mono truncate">{doc.path}</div>
          </div>
        </div>
      </Link>
    </motion.div>
  );
}

export function InteractiveCard({ asset, index = 0 }: { asset: InteractiveAsset; index?: number }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-30px" }}
      transition={{ duration: 0.3, delay: Math.min(index * 0.04, 0.3) }}
    >
      <a
        href={asset.href}
        target="_blank"
        rel="noopener noreferrer"
        className="group block p-3.5 border border-zinc-200 dark:border-zinc-800 rounded-xl hover:border-violet-500 transition bg-white dark:bg-zinc-950"
      >
        <div className="flex items-start justify-between gap-2">
          <div className="min-w-0 flex-1">
            <div className="text-sm font-medium group-hover:text-violet-600 dark:group-hover:text-violet-400 transition mb-0.5 truncate">
              {asset.name}
            </div>
            <div className="text-xs text-zinc-400 font-mono truncate">{asset.file}</div>
          </div>
          <ArrowUpRight size={14} className="shrink-0 text-zinc-400 group-hover:text-violet-500 group-hover:translate-x-0.5 group-hover:-translate-y-0.5 transition-all" />
        </div>
      </a>
    </motion.div>
  );
}
