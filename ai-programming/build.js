#!/usr/bin/env node
/**
 * AI Programming — Build Script
 *
 * Inline CSS + JS from src/ into standalone HTML files in dist/
 *
 * Usage:
 *   node build.js              → build all pages
 *   node build.js tools        → build tools page only
 */

const fs = require('fs');
const path = require('path');

const SRC = path.join(__dirname, 'src');
const DIST = path.join(__dirname, 'dist');

if (!fs.existsSync(DIST)) fs.mkdirSync(DIST);

function read(relPath) {
  return fs.readFileSync(path.join(SRC, relPath), 'utf8');
}

function inlineCSS(html) {
  return html.replace(
    /<link\s+rel="stylesheet"\s+href="([^"]+)"\s*\/?>/g,
    (match, href) => {
      if (href.startsWith('http')) return match;
      try { return `<style>\n/* [inlined: ${href}] */\n${read(href)}\n</style>`; }
      catch (e) { console.warn(`  ⚠ CSS not found: ${href}`); return match; }
    }
  );
}

function inlineJS(html) {
  return html.replace(
    /<script\s+src="([^"]+)"><\/script>/g,
    (match, src) => {
      if (src.startsWith('http') || src.startsWith('//')) return match;
      try { return `<script>\n/* [inlined: ${src}] */\n${read(src)}\n</script>`; }
      catch (e) { console.warn(`  ⚠ JS not found: ${src}`); return match; }
    }
  );
}

// Link rewrite map: src filename → dist filename
const linkMap = {
  'tools.html': 'ai-tools.html',
  'index.html': 'ai-programming.html',
  'omc.html': 'omc-deep-dive.html',
  'cases.html': 'ai-cases.html',
};

function rewriteLinks(html) {
  for (const [src, dist] of Object.entries(linkMap)) {
    html = html.replaceAll(`href="${src}"`, `href="${dist}"`);
  }
  return html;
}

function buildPage(srcFile, distFile) {
  console.log(`📦 ${srcFile} → dist/${distFile}`);
  let html = read(srcFile);
  html = inlineCSS(html);
  html = inlineJS(html);
  html = rewriteLinks(html);
  fs.writeFileSync(path.join(DIST, distFile), html);
  const size = (fs.statSync(path.join(DIST, distFile)).size / 1024).toFixed(1);
  console.log(`   ✅ ${size} KB`);
}

const pages = {
  'tools':     { src: 'tools.html',         dist: 'ai-tools.html' },
  'overview':  { src: 'index.html',          dist: 'ai-programming.html' },
  'omc':       { src: 'omc.html',            dist: 'omc-deep-dive.html' },
  'cases':     { src: 'cases.html',          dist: 'ai-cases.html' },
};

const target = process.argv[2];

if (target && pages[target]) {
  buildPage(pages[target].src, pages[target].dist);
} else if (!target) {
  console.log('Building all pages...\n');
  for (const [key, p] of Object.entries(pages)) {
    try { buildPage(p.src, p.dist); } catch (e) { console.warn(`  ⏭ Skipping ${key}: ${e.message}`); }
  }
  console.log('\n✅ Done. Open dist/ files in browser.');
} else {
  console.log(`Unknown: ${target}\nAvailable: ${Object.keys(pages).join(', ')}`);
  process.exit(1);
}
