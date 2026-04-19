import type { Metadata } from "next";
import { SearchDialog } from "@/components/SearchDialog";
import "./globals.css";

export const metadata: Metadata = {
  title: {
    default: "Learn AI Engineering",
    template: "%s · Learn AI Engineering",
  },
  description:
    "Learn AI engineering by reading someone else's mistakes. AI 工程师的不整理笔记本 — MCP · Agent · RAG · AI Programming.",
  keywords: ["MCP", "AI Agent", "RAG", "LLM", "Claude Code", "AI Engineering", "Function Calling"],
  authors: [{ name: "bob798", url: "https://bob798.github.io" }],
  openGraph: {
    title: "Learn AI Engineering",
    description: "Learn AI engineering by reading someone else's mistakes.",
    type: "website",
    locale: "zh_CN",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh-CN" className="h-full antialiased">
      <body className="min-h-full bg-white dark:bg-zinc-950 text-zinc-900 dark:text-zinc-100">
        <SearchDialog />
        {children}
      </body>
    </html>
  );
}
