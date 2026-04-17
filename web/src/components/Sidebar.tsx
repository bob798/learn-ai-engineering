import Link from "next/link";
import { sections, buildSectionTree, type TreeNode } from "@/lib/docs";

function TreeList({ nodes, level = 0 }: { nodes: TreeNode[]; level?: number }) {
  if (nodes.length === 0) return null;
  return (
    <ul className={level === 0 ? "space-y-1" : "ml-3 mt-1 space-y-0.5 border-l border-zinc-200 dark:border-zinc-800 pl-3"}>
      {nodes.map((node) => (
        <li key={(node.slug ?? node.label) + level}>
          {node.slug ? (
            <Link
              href={`/${node.slug}`}
              className="block py-1 text-sm text-zinc-700 hover:text-orange-600 dark:text-zinc-300 dark:hover:text-orange-400 transition"
            >
              {node.title ?? node.label}
            </Link>
          ) : (
            <span className="block py-1 text-xs font-semibold uppercase tracking-wide text-zinc-500 dark:text-zinc-500">
              {node.label}
            </span>
          )}
          {node.children.length > 0 && <TreeList nodes={node.children} level={level + 1} />}
        </li>
      ))}
    </ul>
  );
}

export function Sidebar({ activeSection }: { activeSection?: string }) {
  return (
    <aside className="w-64 shrink-0 border-r border-zinc-200 dark:border-zinc-800 px-5 py-6 overflow-y-auto">
      <Link href="/" className="block mb-6 font-bold text-lg">
        AI Handbook
      </Link>
      <nav className="space-y-6">
        {sections.map((section) => {
          const tree = buildSectionTree(section.slug);
          const isActive = activeSection === section.slug;
          return (
            <div key={section.slug}>
              <Link
                href={`/${section.slug}`}
                className={`block mb-2 text-sm font-bold ${
                  isActive ? "text-orange-600 dark:text-orange-400" : "text-zinc-900 dark:text-zinc-100"
                }`}
              >
                {section.title}
              </Link>
              <TreeList nodes={tree.children} />
            </div>
          );
        })}
      </nav>
    </aside>
  );
}
