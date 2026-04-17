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
  // mcp
  "mcp/01-foundations": "Foundations · 入门基础",
  "mcp/02-core-concepts": "Core Concepts · 核心概念",
  "mcp/03-practical": "Practical · 实战架构",
  "mcp/05-interview": "Interview · 面试与误解",
  // agent
  "agent/concepts": "Concepts · 核心概念",
  "agent/deep-dives": "Deep Dives · 项目深拆",
  "agent/methodology": "Methodology · 方法论",
  "agent/planning-reasoning": "Planning & Reasoning",
  "agent/research": "Research · 行业研究",
  "agent/templates": "Templates · 可复用模板",
  // rag
  "rag/mock-interview": "Mock Interview · 模拟面试",
  // ai-programming
  "ai-programming/cases": "Cases · 实战案例",
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
