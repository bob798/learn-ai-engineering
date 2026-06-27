"use client";

import { useCallback, useEffect } from "react";
import { useRouter } from "next/navigation";
import { LoopTokenGrowth } from "./visualizations/LoopTokenGrowth";
import { ToolGranularityCompare } from "./visualizations/ToolGranularityCompare";
import { MessageConveyor } from "./visualizations/MessageConveyor";
import { SubagentPipe } from "./visualizations/SubagentPipe";
import { DecisionTree } from "./visualizations/DecisionTree";
import { IntroHook } from "./visualizations/IntroHook";
import { AnalogyBuilder } from "./visualizations/AnalogyBuilder";
import { MindMapCompression } from "./visualizations/MindMapCompression";

const VIZ_REGISTRY: Record<string, React.ComponentType> = {
  LoopTokenGrowth,
  ToolGranularityCompare,
  MessageConveyor,
  SubagentPipe,
  DecisionTree,
  IntroHook,
  AnalogyBuilder,
  MindMapCompression,
};

const VIZ_PLACEHOLDER_RE =
  /<div\s+data-viz="([A-Za-z0-9_-]+)"\s*(?:\/>|>\s*<\/div>)/g;

export function DocBody({ html }: { html: string }) {
  const router = useRouter();
  const basePath = process.env.NEXT_PUBLIC_BASE_PATH || "";

  /** Intercept clicks on internal links to use Next.js router (respects basePath) */
  const handleClick = useCallback(
    (e: React.MouseEvent<HTMLElement>) => {
      const anchor = (e.target as HTMLElement).closest("a");
      if (!anchor) return;
      const href = anchor.getAttribute("href");
      if (!href) return;
      // Skip external links, anchors, and non-http protocols
      if (href.startsWith("http") || href.startsWith("mailto:") || href.startsWith("#")) return;
      // Strip basePath before router.push — router auto-prepends it
      e.preventDefault();
      const routerPath = basePath && href.startsWith(basePath) ? href.slice(basePath.length) || "/" : href;
      router.push(routerPath);
    },
    [router]
  );

  const parts: React.ReactNode[] = [];
  let lastIndex = 0;
  let match: RegExpExecArray | null;
  let key = 0;

  VIZ_PLACEHOLDER_RE.lastIndex = 0;
  while ((match = VIZ_PLACEHOLDER_RE.exec(html)) !== null) {
    const [full, name] = match;
    const start = match.index;

    if (start > lastIndex) {
      parts.push(
        <div
          key={`html-${key++}`}
          dangerouslySetInnerHTML={{ __html: html.slice(lastIndex, start) }}
        />
      );
    }

    const Comp = VIZ_REGISTRY[name];
    if (Comp) {
      parts.push(<Comp key={`viz-${key++}`} />);
    } else {
      parts.push(
        <div
          key={`viz-missing-${key++}`}
          className="my-4 rounded border border-red-300 bg-red-50 dark:bg-red-950/30 dark:border-red-900 p-3 text-sm text-red-700 dark:text-red-300"
        >
          ⚠️ 未注册的可视化组件：<code>{name}</code>
        </div>
      );
    }
    lastIndex = start + full.length;
  }

  if (lastIndex < html.length) {
    parts.push(
      <div
        key={`html-${key++}`}
        dangerouslySetInnerHTML={{ __html: html.slice(lastIndex) }}
      />
    );
  }

  // Render mermaid diagrams after mount
  useEffect(() => {
    const codeBlocks = document.querySelectorAll("pre > code.language-mermaid");
    if (codeBlocks.length === 0) return;

    // Replace code blocks with mermaid containers before loading the library
    const containers: HTMLElement[] = [];
    codeBlocks.forEach((block) => {
      const pre = block.parentElement;
      if (!pre) return;
      const div = document.createElement("div");
      div.className = "mermaid";
      div.textContent = block.textContent || "";
      pre.replaceWith(div);
      containers.push(div);
    });

    // Load mermaid from CDN
    if (!(window as any).mermaid) {
      const script = document.createElement("script");
      script.src = "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js";
      script.onload = () => {
        const m = (window as any).mermaid;
        m.initialize({ startOnLoad: false, theme: "base", fontFamily: "inherit" });
        m.run({ nodes: containers });
      };
      document.head.appendChild(script);
    } else {
      const m = (window as any).mermaid;
      m.run({ nodes: containers });
    }
  }, [html]);

  return <article className="prose-doc" onClick={handleClick}>{parts}</article>;
}
