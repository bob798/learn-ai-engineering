"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { useRouter } from "next/navigation";
import { docs, sections, type DocNode } from "@/lib/docs";

/* ── Search result type ── */
interface SearchResult {
  doc: DocNode;
  sectionTitle: string;
  /** Snippet around the match in contentMd (empty if title-only match) */
  snippet: string;
}

/* ── Lightweight search (no external deps) ── */
function search(query: string, limit = 12): SearchResult[] {
  if (!query.trim()) return [];
  const q = query.toLowerCase();
  const words = q.split(/\s+/).filter(Boolean);

  const sectionMap = new Map(sections.map((s) => [s.slug, s.title]));

  const scored: { result: SearchResult; score: number }[] = [];

  for (const doc of docs) {
    const titleLower = (doc.title ?? "").toLowerCase();
    const descLower = (doc.description ?? "").toLowerCase();
    const contentLower = doc.contentMd.toLowerCase();

    // Score: title match > description match > content match
    let score = 0;
    let snippet = "";

    for (const w of words) {
      if (titleLower.includes(w)) score += 10;
      if (descLower.includes(w)) score += 5;
      if (contentLower.includes(w)) score += 1;
    }

    if (score === 0) continue;

    // Extract snippet from content around first match
    if (score < 10 * words.length) {
      // Not a pure title match — find snippet
      for (const w of words) {
        const idx = contentLower.indexOf(w);
        if (idx >= 0) {
          const start = Math.max(0, idx - 40);
          const end = Math.min(doc.contentMd.length, idx + w.length + 80);
          snippet =
            (start > 0 ? "..." : "") +
            doc.contentMd.slice(start, end).replace(/\n/g, " ").trim() +
            (end < doc.contentMd.length ? "..." : "");
          break;
        }
      }
    }

    scored.push({
      result: {
        doc,
        sectionTitle: sectionMap.get(doc.section) ?? doc.section,
        snippet,
      },
      score,
    });
  }

  return scored
    .sort((a, b) => b.score - a.score)
    .slice(0, limit)
    .map((s) => s.result);
}

/* ── Highlight matching text ── */
function Highlight({ text, query }: { text: string; query: string }) {
  if (!query.trim()) return <>{text}</>;
  const words = query
    .toLowerCase()
    .split(/\s+/)
    .filter(Boolean)
    .map((w) => w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  if (words.length === 0) return <>{text}</>;
  const re = new RegExp(`(${words.join("|")})`, "gi");
  const parts = text.split(re);
  return (
    <>
      {parts.map((part, i) =>
        re.test(part) ? (
          <mark
            key={i}
            className="bg-orange-200/60 dark:bg-orange-500/30 text-inherit rounded-sm px-0.5"
          >
            {part}
          </mark>
        ) : (
          <span key={i}>{part}</span>
        ),
      )}
    </>
  );
}

/* ── Section badge colors ── */
const SECTION_COLORS: Record<string, string> = {
  guide: "bg-sky-100 text-sky-700 dark:bg-sky-900/40 dark:text-sky-400",
  mcp: "bg-orange-100 text-orange-700 dark:bg-orange-900/40 dark:text-orange-400",
  agent: "bg-violet-100 text-violet-700 dark:bg-violet-900/40 dark:text-violet-400",
  rag: "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-400",
  "ai-programming":
    "bg-amber-100 text-amber-700 dark:bg-amber-900/40 dark:text-amber-400",
};

/* ══════════════════════════════════════════════ */
/* ── SearchDialog Component                  ── */
/* ══════════════════════════════════════════════ */
export function SearchDialog() {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);
  const router = useRouter();

  const results = search(query);

  // Reset selection when results change
  useEffect(() => setSelected(0), [query]);

  // Global Cmd+K / Ctrl+K + custom event from sidebar trigger
  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        setOpen((prev) => !prev);
      }
      if (e.key === "Escape") setOpen(false);
    }
    function onOpenSearch() {
      setOpen(true);
    }
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("open-search", onOpenSearch);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("open-search", onOpenSearch);
    };
  }, []);

  // Focus input when dialog opens
  useEffect(() => {
    if (open) {
      setQuery("");
      setTimeout(() => inputRef.current?.focus(), 50);
    }
  }, [open]);

  // Scroll selected item into view
  useEffect(() => {
    const el = listRef.current?.querySelector(`[data-idx="${selected}"]`);
    el?.scrollIntoView({ block: "nearest" });
  }, [selected]);

  const navigate = useCallback(
    (slug: string) => {
      setOpen(false);
      router.push(`/${slug}`);
    },
    [router],
  );

  // Keyboard navigation inside dialog
  function onInputKeyDown(e: React.KeyboardEvent) {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setSelected((s) => Math.min(s + 1, results.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setSelected((s) => Math.max(s - 1, 0));
    } else if (e.key === "Enter" && results[selected]) {
      navigate(results[selected].doc.slug);
    }
  }

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-[200] flex items-start justify-center pt-[12vh]"
      onClick={() => setOpen(false)}
    >
      <div className="fixed inset-0 bg-black/40 backdrop-blur-sm" />

      {/* Dialog */}
      <div
        className="relative w-full max-w-xl mx-4 bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-700 rounded-2xl shadow-2xl overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Input */}
        <div className="flex items-center gap-3 px-4 border-b border-zinc-200 dark:border-zinc-800">
          <svg
            width="18"
            height="18"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="text-zinc-400 shrink-0"
          >
            <circle cx="11" cy="11" r="8" />
            <path d="m21 21-4.3-4.3" />
          </svg>
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={onInputKeyDown}
            placeholder="搜索文档..."
            className="flex-1 py-4 bg-transparent text-base text-zinc-900 dark:text-zinc-100 placeholder:text-zinc-400 outline-none"
          />
          <kbd className="hidden sm:inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded text-[10px] font-mono text-zinc-400 border border-zinc-200 dark:border-zinc-700">
            ESC
          </kbd>
        </div>

        {/* Results */}
        <div
          ref={listRef}
          className="max-h-[50vh] overflow-y-auto overscroll-contain"
        >
          {query.trim() && results.length === 0 && (
            <div className="px-6 py-10 text-center text-sm text-zinc-400">
              没有找到匹配的文档
            </div>
          )}

          {results.map((r, i) => {
            const colorCls =
              SECTION_COLORS[r.doc.section] ??
              "bg-zinc-100 text-zinc-600 dark:bg-zinc-800 dark:text-zinc-400";
            return (
              <button
                key={r.doc.slug}
                data-idx={i}
                onClick={() => navigate(r.doc.slug)}
                className={`w-full text-left px-4 py-3 flex items-start gap-3 transition-colors ${
                  i === selected
                    ? "bg-orange-50 dark:bg-orange-950/30"
                    : "hover:bg-zinc-50 dark:hover:bg-zinc-800/50"
                }`}
              >
                {/* Section badge */}
                <span
                  className={`mt-0.5 shrink-0 px-1.5 py-0.5 rounded text-[10px] font-semibold ${colorCls}`}
                >
                  {r.sectionTitle}
                </span>

                <div className="min-w-0 flex-1">
                  {/* Title */}
                  <div
                    className={`text-sm font-medium truncate ${
                      i === selected
                        ? "text-orange-700 dark:text-orange-400"
                        : "text-zinc-800 dark:text-zinc-200"
                    }`}
                  >
                    <Highlight text={r.doc.title} query={query} />
                  </div>

                  {/* Snippet */}
                  {r.snippet && (
                    <div className="mt-0.5 text-xs text-zinc-400 line-clamp-2 leading-relaxed">
                      <Highlight text={r.snippet} query={query} />
                    </div>
                  )}
                </div>

                {/* Enter hint on selected */}
                {i === selected && (
                  <kbd className="hidden sm:inline-flex self-center shrink-0 px-1.5 py-0.5 rounded text-[10px] font-mono text-zinc-400 border border-zinc-200 dark:border-zinc-700">
                    Enter
                  </kbd>
                )}
              </button>
            );
          })}
        </div>

        {/* Footer hint */}
        {query.trim() && results.length > 0 && (
          <div className="px-4 py-2 border-t border-zinc-200 dark:border-zinc-800 flex items-center gap-4 text-[11px] text-zinc-400">
            <span>
              <kbd className="px-1 py-0.5 rounded border border-zinc-200 dark:border-zinc-700 font-mono">
                ↑↓
              </kbd>{" "}
              选择
            </span>
            <span>
              <kbd className="px-1 py-0.5 rounded border border-zinc-200 dark:border-zinc-700 font-mono">
                Enter
              </kbd>{" "}
              打开
            </span>
            <span>
              <kbd className="px-1 py-0.5 rounded border border-zinc-200 dark:border-zinc-700 font-mono">
                ESC
              </kbd>{" "}
              关闭
            </span>
          </div>
        )}

        {/* Empty state: popular links */}
        {!query.trim() && (
          <div className="px-4 py-4">
            <div className="text-[11px] font-semibold uppercase tracking-wider text-zinc-400 mb-3">
              快速跳转
            </div>
            {[
              { label: "学习路线图", slug: "guide/learning-path" },
              { label: "ReAct 论文解读", slug: "agent/research/react-paper" },
              { label: "Agent Loop 深度理解", slug: "agent/harness/agent-loop" },
              { label: "MCP 基础", slug: "mcp/01-foundations" },
              { label: "理解 RAG", slug: "rag/01-理解RAG" },
            ].map((link) => (
              <button
                key={link.slug}
                onClick={() => navigate(link.slug)}
                className="w-full text-left px-3 py-2 text-sm text-zinc-600 dark:text-zinc-400 hover:bg-zinc-50 dark:hover:bg-zinc-800/50 rounded-lg transition"
              >
                {link.label}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

/* ══════════════════════════════════════════════ */
/* ── SearchTrigger: button shown in sidebar  ── */
/* ══════════════════════════════════════════════ */
export function SearchTrigger() {
  return (
    <button
      onClick={() => window.dispatchEvent(new Event("open-search"))}
      className="w-full flex items-center gap-2 px-3 py-2 rounded-lg border border-zinc-200 dark:border-zinc-800 text-sm text-zinc-400 hover:border-zinc-300 dark:hover:border-zinc-700 hover:text-zinc-600 dark:hover:text-zinc-300 transition"
    >
      <svg
        width="15"
        height="15"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
        className="shrink-0"
      >
        <circle cx="11" cy="11" r="8" />
        <path d="m21 21-4.3-4.3" />
      </svg>
      <span className="flex-1 text-left">搜索...</span>
      <kbd className="hidden sm:inline text-[10px] font-mono px-1 py-0.5 rounded border border-zinc-200 dark:border-zinc-700">
        ⌘K
      </kbd>
    </button>
  );
}
