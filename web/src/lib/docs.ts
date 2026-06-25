import rawData from "@/data/docs.json";

export interface Heading {
  depth: number;
  text: string;
  id: string;
}

export interface DocNode {
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

export interface InteractiveAsset {
  section: string;
  file: string;
  name: string;
  href: string;
}

export interface Section {
  slug: string;
  dir: string;
  title: string;
  description: string;
}

interface DocsData {
  sections: Section[];
  docs: DocNode[];
  interactive: InteractiveAsset[];
}

const data = rawData as DocsData;

export const sections = data.sections;
export const docs = data.docs;
export const interactive = data.interactive;

export function findDoc(slugParts: string[]): DocNode | undefined {
  const slug = slugParts.join("/");
  return docs.find((d) => d.slug === slug);
}

export function docsBySection(section: string): DocNode[] {
  return docs.filter((d) => d.section === section);
}

export function interactiveBySection(section: string): InteractiveAsset[] {
  return interactive.filter((a) => a.section === section);
}

// --- Sub-folder display name mapping ---

const SUBFOLDER_LABELS: Record<string, string> = {
  // guide (no subfolders yet)
  // mcp
  "mcp/01-foundations": "Foundations · 入门基础",
  "mcp/02-core-concepts": "Core Concepts · 核心概念",
  "mcp/03-practical": "Practical · 实战架构",
  "mcp/05-interview": "Interview · 面试与误解",
  // agent
  "agent/concepts": "Concepts · 核心概念",
  "agent/agent-from-scratch": "Agent from Scratch · 从零手写",
  "agent/deep-topics": "Deep Topics · 深入子方向",
  "agent/deep-topics/memory": "Memory · 记忆专题",
  "agent/deep-topics/context-engineering": "Context Engineering · 上下文工程",
  "agent/deep-topics/planning-reasoning": "Planning & Reasoning",
  "agent/deep-topics/papers": "Papers · 论文解读",
  "agent/harness": "Harness · 运行时可靠性",
  "agent/deep-dives": "Deep Dives · 项目深拆",
  "agent/research": "Research · 行业研究",
  // rag
  "rag/mock-interview": "Mock Interview · 模拟面试",
  "rag/research": "Research · 论文参考",
  // ai-programming
  "ai-programming/cases": "Cases · 实战案例",
  "ai-programming/architecture-series": "Architecture Series · 架构系列",
};

export function subfolderLabel(section: string, sub: string): string {
  const key = `${section}/${sub}`;
  if (SUBFOLDER_LABELS[key]) return SUBFOLDER_LABELS[key];
  return sub.replace(/^\d+[-_]/, "").replace(/[-_]/g, " ");
}

export interface DocGroup {
  key: string;          // sub-folder name or "_root"
  label: string;
  docs: DocNode[];
}

export function groupSectionDocs(section: string): {
  readme?: DocNode;
  groups: DocGroup[];
  loose: DocNode[];     // docs at section root (not in a sub-folder)
} {
  const all = docsBySection(section);
  let readme: DocNode | undefined;
  const byGroup = new Map<string, DocNode[]>();
  const loose: DocNode[] = [];

  for (const doc of all) {
    if (doc.slug === section) {
      readme = doc;
      continue;
    }
    const parts = doc.slug.split("/").slice(1); // strip section prefix
    if (parts.length === 1) {
      loose.push(doc);
    } else {
      const sub = parts[0];
      if (!byGroup.has(sub)) byGroup.set(sub, []);
      byGroup.get(sub)!.push(doc);
    }
  }

  const groups: DocGroup[] = Array.from(byGroup.entries())
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([sub, docs]) => ({
      key: sub,
      label: subfolderLabel(section, sub),
      docs,
    }));

  return { readme, groups, loose };
}

// Build a tree of docs grouped by path prefix for sidebar rendering
export interface TreeNode {
  label: string;
  slug?: string;
  title?: string;
  children: TreeNode[];
}

export function buildSectionTree(section: string): TreeNode {
  const sectionDocs = docsBySection(section);
  const root: TreeNode = { label: section, children: [] };

  for (const doc of sectionDocs) {
    const parts = doc.slug.split("/").slice(1); // strip section prefix
    if (parts.length === 0) {
      root.slug = doc.slug;
      root.title = doc.title;
      continue;
    }
    let cur = root;
    for (let i = 0; i < parts.length; i++) {
      const label = parts[i];
      let child = cur.children.find((c) => c.label === label);
      if (!child) {
        child = { label, children: [] };
        cur.children.push(child);
      }
      if (i === parts.length - 1) {
        child.slug = doc.slug;
        child.title = doc.title;
      }
      cur = child;
    }
  }
  return root;
}
