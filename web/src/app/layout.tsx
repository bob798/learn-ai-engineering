import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AI Handbook",
  description: "AI 工程师知识手册 · MCP / Agent / RAG / AI Programming",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh-CN" className="h-full antialiased">
      <body className="min-h-full bg-white dark:bg-zinc-950 text-zinc-900 dark:text-zinc-100">
        {children}
      </body>
    </html>
  );
}
