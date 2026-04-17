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
