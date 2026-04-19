"use client";

import { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  sections,
  buildSectionTree,
  subfolderLabel,
  type TreeNode,
} from "@/lib/docs";
import { SearchTrigger } from "@/components/SearchDialog";

/* ── Icons ── */
const SECTION_ICONS: Record<string, string> = {
  mcp: "M",
  agent: "A",
  rag: "R",
  "ai-programming": "P",
  guide: "G",
};

const SECTION_COLORS: Record<string, string> = {
  mcp: "bg-orange-500/15 text-orange-600 dark:text-orange-400",
  agent: "bg-violet-500/15 text-violet-600 dark:text-violet-400",
  rag: "bg-emerald-500/15 text-emerald-600 dark:text-emerald-400",
  "ai-programming": "bg-amber-500/15 text-amber-600 dark:text-amber-400",
  guide: "bg-sky-500/15 text-sky-600 dark:text-sky-400",
};

/* ── Chevron ── */
function Chevron({ open }: { open: boolean }) {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 16 16"
      className={`shrink-0 transition-transform duration-200 ${open ? "rotate-90" : ""}`}
      fill="currentColor"
    >
      <path d="M6.22 4.22a.75.75 0 0 1 1.06 0l3.25 3.25a.75.75 0 0 1 0 1.06l-3.25 3.25a.75.75 0 0 1-1.06-1.06L8.94 8 6.22 5.28a.75.75 0 0 1 0-1.06Z" />
    </svg>
  );
}

/* ── Collapsible folder ── */
function FolderGroup({
  label,
  children,
  defaultOpen,
  count,
}: {
  label: string;
  children: React.ReactNode;
  defaultOpen: boolean;
  count: number;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div>
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1.5 w-full py-1.5 text-xs font-semibold uppercase tracking-wide text-zinc-500 dark:text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300 transition"
      >
        <Chevron open={open} />
        <span className="truncate">{label}</span>
        <span className="ml-auto text-[10px] font-normal text-zinc-400 tabular-nums">{count}</span>
      </button>
      {open && (
        <ul className="ml-2 mt-0.5 space-y-0.5 border-l border-zinc-200 dark:border-zinc-800 pl-3">
          {children}
        </ul>
      )}
    </div>
  );
}

/* ── Doc link ── */
function DocLink({
  slug,
  title,
  isActive,
}: {
  slug: string;
  title: string;
  isActive: boolean;
}) {
  return (
    <li>
      <Link
        href={`/${slug}`}
        className={`block py-1 text-sm transition truncate ${
          isActive
            ? "text-orange-600 dark:text-orange-400 font-medium border-l-2 border-orange-500 -ml-[13px] pl-[11px]"
            : "text-zinc-600 hover:text-orange-600 dark:text-zinc-400 dark:hover:text-orange-400"
        }`}
        title={title}
      >
        {title}
      </Link>
    </li>
  );
}

/* ── Recursive tree (for deeply nested docs) ── */
function TreeItems({
  nodes,
  activeSlug,
  sectionSlug,
}: {
  nodes: TreeNode[];
  activeSlug: string;
  sectionSlug: string;
}) {
  return (
    <>
      {nodes.map((node) => {
        // Leaf node (has slug = is a doc)
        if (node.slug && node.children.length === 0) {
          return (
            <DocLink
              key={node.slug}
              slug={node.slug}
              title={node.title ?? node.label}
              isActive={activeSlug === node.slug}
            />
          );
        }

        // Branch: folder with children
        const hasActiveChild = hasDescendant(node, activeSlug);
        const folderLabel = subfolderLabel(sectionSlug, node.label);
        const leafCount = countLeaves(node);

        return (
          <li key={node.label}>
            <FolderGroup
              label={folderLabel}
              defaultOpen={hasActiveChild}
              count={leafCount}
            >
              {/* If the folder itself is also a doc (e.g. a README) */}
              {node.slug && (
                <DocLink
                  slug={node.slug}
                  title="Overview"
                  isActive={activeSlug === node.slug}
                />
              )}
              <TreeItems
                nodes={node.children}
                activeSlug={activeSlug}
                sectionSlug={sectionSlug}
              />
            </FolderGroup>
          </li>
        );
      })}
    </>
  );
}

/* ── Helpers ── */
function hasDescendant(node: TreeNode, slug: string): boolean {
  if (node.slug === slug) return true;
  return node.children.some((c) => hasDescendant(c, slug));
}

function countLeaves(node: TreeNode): number {
  if (node.children.length === 0) return node.slug ? 1 : 0;
  return node.children.reduce((n, c) => n + countLeaves(c), node.slug ? 1 : 0);
}

/* ── Main Sidebar ── */
export function Sidebar({ activeSection }: { activeSection?: string }) {
  const pathname = usePathname();
  // Derive activeSlug from pathname: "/agent/research/react-paper" -> "agent/research/react-paper"
  const activeSlug = pathname.replace(/^\//, "");

  return (
    <aside className="w-64 shrink-0 border-r border-zinc-200 dark:border-zinc-800 overflow-y-auto sticky top-0 h-screen">
      {/* Brand */}
      <div className="px-5 pt-6 pb-3">
        <Link href="/" className="block font-bold text-lg hover:text-orange-600 transition">
          AI Handbook
        </Link>
        <p className="text-[11px] text-zinc-400 mt-1">Learn AI Engineering</p>
      </div>

      {/* Search */}
      <div className="px-4 pb-4">
        <SearchTrigger />
      </div>

      {/* Nav sections */}
      <nav className="px-4 pb-8 space-y-1">
        {sections.map((section) => {
          const tree = buildSectionTree(section.slug);
          const isActive = activeSection === section.slug;
          const icon = SECTION_ICONS[section.slug] ?? "·";
          const colorCls = SECTION_COLORS[section.slug] ?? "bg-zinc-500/15 text-zinc-500";

          return (
            <SectionAccordion
              key={section.slug}
              slug={section.slug}
              title={section.title}
              icon={icon}
              colorCls={colorCls}
              tree={tree}
              isActive={isActive}
              activeSlug={activeSlug}
            />
          );
        })}
      </nav>
    </aside>
  );
}

/* ── Section-level accordion ── */
function SectionAccordion({
  slug,
  title,
  icon,
  colorCls,
  tree,
  isActive,
  activeSlug,
}: {
  slug: string;
  title: string;
  icon: string;
  colorCls: string;
  tree: TreeNode;
  isActive: boolean;
  activeSlug: string;
}) {
  const [open, setOpen] = useState(isActive);

  return (
    <div className="mb-1">
      {/* Section header: click left part navigates, click chevron toggles */}
      <div className="flex items-center gap-2 group">
        <Link
          href={`/${slug}`}
          className={`flex items-center gap-2 flex-1 min-w-0 py-2 px-2 rounded-lg text-sm font-semibold transition ${
            isActive
              ? "text-zinc-900 dark:text-zinc-100 bg-zinc-100 dark:bg-zinc-800/60"
              : "text-zinc-700 dark:text-zinc-300 hover:bg-zinc-100 dark:hover:bg-zinc-800/40"
          }`}
        >
          <span
            className={`inline-flex items-center justify-center size-5 rounded text-[10px] font-bold ${colorCls}`}
          >
            {icon}
          </span>
          <span className="truncate">{title}</span>
        </Link>
        <button
          onClick={() => setOpen(!open)}
          className="p-1 rounded text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-300 transition"
          aria-label={open ? "Collapse" : "Expand"}
        >
          <Chevron open={open} />
        </button>
      </div>

      {/* Children */}
      {open && tree.children.length > 0 && (
        <ul className="mt-1 ml-3 space-y-0.5">
          <TreeItems
            nodes={tree.children}
            activeSlug={activeSlug}
            sectionSlug={slug}
          />
        </ul>
      )}
    </div>
  );
}
