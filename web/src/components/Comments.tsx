"use client";

import { useEffect, useRef } from "react";

const GISCUS_CONFIG = {
  repo: "bob798/learn-ai-engineering",
  repoId: process.env.NEXT_PUBLIC_GISCUS_REPO_ID || "",
  category: "General",
  categoryId: process.env.NEXT_PUBLIC_GISCUS_CATEGORY_ID || "",
  mapping: "pathname",
  strict: "0",
  reactionsEnabled: "1",
  emitMetadata: "0",
  inputPosition: "bottom",
  theme: "preferred_color_scheme",
  lang: "zh-CN",
};

export function Comments() {
  const ref = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    if (!GISCUS_CONFIG.repoId || !GISCUS_CONFIG.categoryId) return;
    if (el.querySelector("iframe.giscus-frame")) return;

    const script = document.createElement("script");
    script.src = "https://giscus.app/client.js";
    script.async = true;
    script.crossOrigin = "anonymous";
    Object.entries({
      "data-repo": GISCUS_CONFIG.repo,
      "data-repo-id": GISCUS_CONFIG.repoId,
      "data-category": GISCUS_CONFIG.category,
      "data-category-id": GISCUS_CONFIG.categoryId,
      "data-mapping": GISCUS_CONFIG.mapping,
      "data-strict": GISCUS_CONFIG.strict,
      "data-reactions-enabled": GISCUS_CONFIG.reactionsEnabled,
      "data-emit-metadata": GISCUS_CONFIG.emitMetadata,
      "data-input-position": GISCUS_CONFIG.inputPosition,
      "data-theme": GISCUS_CONFIG.theme,
      "data-lang": GISCUS_CONFIG.lang,
      "data-loading": "lazy",
    }).forEach(([k, v]) => script.setAttribute(k, v));

    el.appendChild(script);

    return () => {
      while (el.firstChild) el.removeChild(el.firstChild);
    };
  }, []);

  if (!GISCUS_CONFIG.repoId || !GISCUS_CONFIG.categoryId) {
    return (
      <div className="mt-16 pt-8 border-t border-zinc-200 dark:border-zinc-800">
        <div className="text-xs text-zinc-500 italic">
          💬 评论功能待启用（需要配置 NEXT_PUBLIC_GISCUS_REPO_ID / NEXT_PUBLIC_GISCUS_CATEGORY_ID 环境变量）
        </div>
      </div>
    );
  }

  return (
    <div className="mt-16 pt-8 border-t border-zinc-200 dark:border-zinc-800">
      <h2 className="text-lg font-bold mb-4 text-zinc-900 dark:text-zinc-100">评论</h2>
      <div ref={ref} />
    </div>
  );
}
