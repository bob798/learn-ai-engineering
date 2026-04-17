import * as fs from "node:fs";
import * as path from "node:path";
import matter from "gray-matter";
import { unified } from "unified";
import remarkParse from "remark-parse";
import remarkGfm from "remark-gfm";
import remarkRehype from "remark-rehype";
import rehypeRaw from "rehype-raw";
import rehypeSlug from "rehype-slug";
import rehypeAutolinkHeadings from "rehype-autolink-headings";
import rehypeHighlight from "rehype-highlight";
import rehypeStringify from "rehype-stringify";

const WEB_DIR = path.resolve(__dirname, "..");
const REPO_ROOT = path.resolve(WEB_DIR, "..");
const CONTENT_DIR = path.join(REPO_ROOT, "content");
const INTERACTIVE_DIR = path.join(REPO_ROOT, "interactive");
const OUT_PATH = path.join(WEB_DIR, "src", "data", "docs.json");

interface Section {
  slug: string;
  dir: string;
  title: string;
  description: string;
}

const SECTIONS: Section[] = [
  { slug: "mcp",            dir: "01-mcp",            title: "MCP",            description: "模型上下文协议 · 从协议到生态" },
  { slug: "agent",          dir: "02-agent",          title: "Agent",          description: "智能体架构 · 方法论 · 生态拆解" },
  { slug: "rag",            dir: "03-rag",            title: "RAG",            description: "检索增强生成 · 从 V1 到企业级" },
  { slug: "ai-programming", dir: "04-ai-programming", title: "AI Programming", description: "AI 编程实战与工具链" },
];

interface Heading { depth: number; text: string; id: string }
interface DocNode {
  slug: string;
  section: string;
  title: string;
  description?: string;
  path: string;
  order: number;
  contentMd: string;
  contentHtml: string;
  headings: Heading[];
}

interface InteractiveAsset {
  section: string;
  file: string;     // relative path under interactive/
  name: string;     // display name (no extension)
  href: string;     // URL from app perspective
}

const processor = unified()
  .use(remarkParse)
  .use(remarkGfm)
  .use(remarkRehype, { allowDangerousHtml: true })
  .use(rehypeRaw)
  .use(rehypeSlug)
  .use(rehypeAutolinkHeadings, { behavior: "append" })
  .use(rehypeHighlight, { detect: true, ignoreMissing: true })
  .use(rehypeStringify);

function walk(dir: string, ext: string): string[] {
  const out: string[] = [];
  if (!fs.existsSync(dir)) return out;
  for (const name of fs.readdirSync(dir)) {
    if (name.startsWith(".")) continue;
    const full = path.join(dir, name);
    const stat = fs.statSync(full);
    if (stat.isDirectory()) out.push(...walk(full, ext));
    else if (name.toLowerCase().endsWith(ext)) out.push(full);
  }
  return out;
}

function extractTitle(md: string, fallback: string): string {
  const m = md.match(/^#\s+(.+)$/m);
  return m ? m[1].trim() : fallback;
}

function slugifyText(s: string): string {
  return s.toLowerCase().trim().replace(/[^\w\u4e00-\u9fa5]+/g, "-").replace(/^-|-$/g, "");
}

function toSlug(rel: string, section: string): string {
  const noExt = rel.replace(/\\/g, "/").replace(/\.md$/i, "");
  if (noExt === "README") return section;
  if (noExt.endsWith("/README")) return `${section}/${noExt.slice(0, -"/README".length)}`;
  return `${section}/${noExt}`;
}

function extractHeadings(md: string): Heading[] {
  const headings: Heading[] = [];
  const seen = new Map<string, number>();
  for (const line of md.split("\n")) {
    const m = line.match(/^(#{1,6})\s+(.+?)\s*#*$/);
    if (!m) continue;
    const depth = m[1].length;
    const text = m[2].replace(/`/g, "").trim();
    let id = slugifyText(text);
    const count = seen.get(id) ?? 0;
    seen.set(id, count + 1);
    if (count > 0) id = `${id}-${count}`;
    headings.push({ depth, text, id });
  }
  return headings;
}

async function renderMarkdown(md: string): Promise<string> {
  const file = await processor.process(md);
  return String(file);
}

async function main() {
  const docs: DocNode[] = [];

  for (const section of SECTIONS) {
    const sectionDir = path.join(CONTENT_DIR, section.dir);
    const files = walk(sectionDir, ".md");
    for (const file of files) {
      const rel = path.relative(sectionDir, file).replace(/\\/g, "/");
      const raw = fs.readFileSync(file, "utf8");
      const { data, content } = matter(raw);
      const html = await renderMarkdown(content);
      const numPrefix = rel.match(/^(\d+)[-_]/)?.[1];
      const fallback = path.basename(rel, path.extname(rel));
      docs.push({
        slug: toSlug(rel, section.slug),
        section: section.slug,
        title: (data.title as string) || extractTitle(content, fallback),
        description: (data.description as string) || undefined,
        path: path.relative(REPO_ROOT, file),
        order: numPrefix ? Number(numPrefix) : 999,
        contentMd: content,
        contentHtml: html,
        headings: extractHeadings(content),
      });
    }
  }

  docs.sort(
    (a, b) =>
      a.section.localeCompare(b.section) ||
      a.order - b.order ||
      a.slug.localeCompare(b.slug)
  );

  // Scan interactive/ for HTML assets (exposed as /viz/<section>/<file>.html via public/)
  const interactive: InteractiveAsset[] = [];
  for (const section of SECTIONS) {
    const dir = path.join(INTERACTIVE_DIR, section.slug);
    const files = walk(dir, ".html");
    for (const file of files) {
      const rel = path.relative(INTERACTIVE_DIR, file).replace(/\\/g, "/");
      interactive.push({
        section: section.slug,
        file: rel,
        name: path.basename(rel, ".html"),
        href: `/viz/${rel}`,
      });
    }
  }

  fs.mkdirSync(path.dirname(OUT_PATH), { recursive: true });
  fs.writeFileSync(
    OUT_PATH,
    JSON.stringify({ sections: SECTIONS, docs, interactive }, null, 2)
  );
  console.log(
    `✓ Wrote ${docs.length} docs + ${interactive.length} interactive HTMLs → ${path.relative(
      REPO_ROOT,
      OUT_PATH
    )}`
  );

  // Expose interactive/ under /viz via symlink in public/
  const PUBLIC_VIZ = path.join(WEB_DIR, "public", "viz");
  fs.mkdirSync(path.dirname(PUBLIC_VIZ), { recursive: true });
  try {
    const stat = fs.lstatSync(PUBLIC_VIZ);
    if (stat.isSymbolicLink() || stat.isDirectory()) {
      fs.rmSync(PUBLIC_VIZ, { recursive: true, force: true });
    }
  } catch { /* not present */ }
  fs.symlinkSync("../../interactive", PUBLIC_VIZ, "dir");
  console.log(`✓ Linked public/viz → ../../interactive`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
