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
  { slug: "guide",          dir: "00-guide",          title: "Guide",          description: "阅读指南 · 给谁看 · 怎么读" },
  { slug: "mcp",            dir: "01-mcp",            title: "MCP",            description: "模型上下文协议 · 从协议到生态" },
  { slug: "agent",          dir: "02-agent",          title: "Agent",          description: "智能体架构 · 方法论 · 生态拆解" },
  { slug: "rag",            dir: "03-rag",            title: "RAG",            description: "检索增强生成 · 从 V1 到企业级" },
  { slug: "ai-programming", dir: "04-ai-programming", title: "AI Programming", description: "AI 编程实战与工具链" },
  { slug: "llm-foundations", dir: "05-llm-foundations", title: "LLM Foundations", description: "模型内部机制 · 术语精度 · 选型判据" },
  { slug: "spring-ai",      dir: "06-spring-ai",      title: "Spring AI",      description: "Java 工程师视角的 AI 应用工程" },
  { slug: "faq",            dir: "07-faq",            title: "FAQ",            description: "AI 协作常见问答 · 为什么会这样 · 怎么办" },
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

/* ── Directory name → section slug mapping ── */
const DIR_TO_SECTION: Record<string, string> = Object.fromEntries(
  SECTIONS.map((s) => [s.dir, s.slug])
);

const SECTION_SLUGS = new Set(SECTIONS.map((s) => s.slug));

/** basePath — mirrors next.config.ts logic */
const BASE_PATH = process.env.NODE_ENV === "production" ? "/learn-ai-engineering" : "";

/** Resolve an absolute content file path to its site slug, or null if outside content/ */
function contentPathToSlug(absPath: string): string | null {
  const rel = path.relative(CONTENT_DIR, absPath).replace(/\\/g, "/");
  if (rel.startsWith("..")) return null;
  const parts = rel.split("/");
  const sectionDir = parts[0];
  const section = DIR_TO_SECTION[sectionDir];
  if (!section) return null;
  const rest = parts.slice(1).join("/");
  return toSlug(rest, section);
}

/**
 * Resolve a relative .html link from a content file to a /viz/ path.
 * e.g. from content/02-agent/methodology/5d-framework.md,
 *   `./interactive.html` → check interactive/agent/interactive.html
 *   `../deep-dives/memgpt-letta/memgpt-letta-guide.html` → check interactive/agent/memgpt-letta-guide.html
 * Returns the viz URL or null if file not found.
 */
function resolveInteractiveHtml(filePath: string, linkPath: string): string | null {
  // Determine which section this content file belongs to
  const relToContent = path.relative(CONTENT_DIR, filePath).replace(/\\/g, "/");
  const sectionDir = relToContent.split("/")[0];
  const section = DIR_TO_SECTION[sectionDir];
  if (!section) return null;

  const fileName = path.basename(linkPath);
  const vizPath = path.join(INTERACTIVE_DIR, section, fileName);
  if (fs.existsSync(vizPath)) {
    return `${BASE_PATH}/viz/${section}/${fileName}`;
  }
  return null;
}

/** Walk hast tree and call fn on every element node */
function walkTree(node: any, fn: (el: any) => void) {
  if (node.type === "element") fn(node);
  if (node.children) {
    for (const child of node.children) walkTree(child, fn);
  }
}

/**
 * Rehype plugin: rewrite internal links for the static site.
 * 1. Relative .md links → site slug paths with basePath
 * 2. Relative .html links → /viz/ paths (interactive files)
 * 3. Absolute internal paths (/agent/...) → prepend basePath
 * 4. Links outside content/ → GitHub blob URL
 */
const IMAGE_EXTS = [".svg", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".avif"];

function rehypeRewriteLinks(filePath: string) {
  return () => (tree: any) => {
    walkTree(tree, (el: any) => {
      // <img> with relative src → copy of content asset under /content-assets/
      if (el.tagName === "img") {
        const src = (el.properties as any)?.src;
        if (!src || typeof src !== "string") return;
        if (src.startsWith("http") || src.startsWith("data:") || src.startsWith("/")) return;
        const resolved = path.resolve(path.dirname(filePath), src.split("#")[0]);
        const rel = path.relative(CONTENT_DIR, resolved).replace(/\\/g, "/");
        if (rel.startsWith("..")) return; // outside content/
        (el.properties as any).src = `${BASE_PATH}/content-assets/${rel}`;
        return;
      }
      if (el.tagName !== "a") return;
      const href = (el.properties as any)?.href;
      if (!href || typeof href !== "string") return;
      if (href.startsWith("http") || href.startsWith("#") || href.startsWith("mailto:")) return;

      const [linkPath, hash] = href.split("#");
      if (!linkPath) return;
      const suffix = hash ? `#${hash}` : "";

      // Case 1: Absolute internal path like /agent/methodology/5d-framework
      if (linkPath.startsWith("/")) {
        const firstSegment = linkPath.split("/")[1];
        if (SECTION_SLUGS.has(firstSegment) || firstSegment === "viz") {
          (el.properties as any).href = `${BASE_PATH}${linkPath}${suffix}`;
        }
        return;
      }

      const fileDir = path.dirname(filePath);
      const resolved = path.resolve(fileDir, linkPath);

      // Case 2: Relative .md link
      if (linkPath.endsWith(".md")) {
        const slug = contentPathToSlug(resolved);
        if (slug) {
          (el.properties as any).href = `${BASE_PATH}/${slug}${suffix}`;
        } else {
          const relToRepo = path.relative(REPO_ROOT, resolved).replace(/\\/g, "/");
          (el.properties as any).href = `https://github.com/bob798/learn-ai-engineering/blob/main/${relToRepo}${suffix}`;
        }
        return;
      }

      // Case 3: Relative .html link → try to resolve to /viz/ path
      if (linkPath.endsWith(".html")) {
        const vizUrl = resolveInteractiveHtml(filePath, linkPath);
        if (vizUrl) {
          (el.properties as any).href = `${vizUrl}${suffix}`;
        }
      }
    });
  };
}

/** Wrap <table> in a scrollable div for mobile */
function rehypeWrapTables() {
  return (tree: any) => {
    function visit(node: any) {
      if (!node.children) return;
      for (let i = 0; i < node.children.length; i++) {
        const child = node.children[i];
        if (child.type === "element" && child.tagName === "table") {
          node.children[i] = {
            type: "element",
            tagName: "div",
            properties: { className: ["table-wrap"] },
            children: [child],
          };
        } else {
          visit(child);
        }
      }
    }
    visit(tree);
  };
}

function createProcessor(filePath: string) {
  return unified()
    .use(remarkParse)
    .use(remarkGfm)
    .use(remarkRehype, { allowDangerousHtml: true })
    .use(rehypeRaw)
    .use(rehypeSlug)
    .use(rehypeAutolinkHeadings, { behavior: "append" })
    .use(rehypeHighlight, { detect: true, ignoreMissing: true })
    .use(rehypeRewriteLinks(filePath))
    .use(rehypeWrapTables)
    .use(rehypeStringify);
}

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

async function renderMarkdown(md: string, filePath: string): Promise<string> {
  const file = await createProcessor(filePath).process(md);
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
      const html = await renderMarkdown(content, file);
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

  // Expose interactive/ under /viz via recursive copy in public/
  // (was symlink; switched to copy for CI/static-export compatibility)
  const PUBLIC_VIZ = path.join(WEB_DIR, "public", "viz");
  fs.mkdirSync(path.dirname(PUBLIC_VIZ), { recursive: true });
  try {
    const stat = fs.lstatSync(PUBLIC_VIZ);
    if (stat.isSymbolicLink() || stat.isDirectory()) {
      fs.rmSync(PUBLIC_VIZ, { recursive: true, force: true });
    }
  } catch { /* not present */ }
  fs.cpSync(INTERACTIVE_DIR, PUBLIC_VIZ, { recursive: true });
  const htmlCount = walk(PUBLIC_VIZ, ".html").length;
  console.log(`✓ Copied interactive/ → public/viz/ (${htmlCount} HTML files)`);

  // Copy content images → public/content-assets/ (mirrors content/ tree)
  // so relative <img src="./images/x.svg"> rewritten to /content-assets/... resolves.
  const PUBLIC_ASSETS = path.join(WEB_DIR, "public", "content-assets");
  fs.rmSync(PUBLIC_ASSETS, { recursive: true, force: true });
  let imgCount = 0;
  for (const ext of IMAGE_EXTS) {
    for (const file of walk(CONTENT_DIR, ext)) {
      const rel = path.relative(CONTENT_DIR, file);
      const dest = path.join(PUBLIC_ASSETS, rel);
      fs.mkdirSync(path.dirname(dest), { recursive: true });
      fs.copyFileSync(file, dest);
      imgCount++;
    }
  }
  console.log(`✓ Copied content images → public/content-assets/ (${imgCount} files)`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
