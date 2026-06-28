#!/usr/bin/env node
/**
 * 发布用本地服务器
 *
 * 功能：
 * 1. 静态文件服务（替代 npx serve）
 * 2. POST /save-image — 接收 base64 PNG，保存为文件，返回 HTTP URL
 *
 * 用法：
 *   node scripts/publish-server.mjs content/02-agent/concepts
 *   # 然后打开 http://localhost:8787/01-why-agent-is-inevitable-wechat.html
 */

import { createServer } from 'node:http';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { join, extname, resolve } from 'node:path';
import { existsSync } from 'node:fs';

const PORT = parseInt(process.env.PORT || '8787');
const ROOT = resolve(process.argv[2] || '.');

// 确保 _assets 目录存在（存放生成的图片）
const ASSETS_DIR = join(ROOT, '_assets');
if (!existsSync(ASSETS_DIR)) await mkdir(ASSETS_DIR, { recursive: true });

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js':   'application/javascript',
  '.css':  'text/css',
  '.png':  'image/png',
  '.jpg':  'image/jpeg',
  '.svg':  'image/svg+xml',
  '.json': 'application/json',
  '.woff2': 'font/woff2',
};

const server = createServer(async (req, res) => {
  // CORS（WechatSync 可能从扩展上下文请求）
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  if (req.method === 'OPTIONS') { res.writeHead(204); res.end(); return; }

  // POST /save-image — 保存图片并返回 URL
  if (req.method === 'POST' && req.url === '/save-image') {
    try {
      const chunks = [];
      for await (const chunk of req) chunks.push(chunk);
      const body = JSON.parse(Buffer.concat(chunks).toString());
      // body: { name: 'cover.png', data: 'data:image/png;base64,...' }
      const { name, data } = body;
      if (!name || !data) throw new Error('missing name or data');

      const base64 = data.replace(/^data:image\/\w+;base64,/, '');
      const filePath = join(ASSETS_DIR, name.replace(/[^a-zA-Z0-9._\-\u4e00-\u9fff]/g, '_'));
      await writeFile(filePath, Buffer.from(base64, 'base64'));

      const url = `http://localhost:${PORT}/_assets/${encodeURIComponent(name.replace(/[^a-zA-Z0-9._\-\u4e00-\u9fff]/g, '_'))}`;
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ ok: true, url }));
      console.log(`  💾 saved: ${filePath}`);
    } catch (e) {
      res.writeHead(400, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ ok: false, error: e.message }));
    }
    return;
  }

  // GET — 静态文件
  let filePath = join(ROOT, decodeURIComponent(req.url.split('?')[0]));
  if (filePath.endsWith('/')) filePath = join(filePath, 'index.html');

  // 尝试读取文件，找不到则自动补 .html 后缀
  let data, ext;
  try {
    data = await readFile(filePath);
    ext = extname(filePath).toLowerCase();
  } catch (e1) {
    if (e1.code === 'ENOENT' && !extname(filePath)) {
      try {
        filePath = filePath + '.html';
        data = await readFile(filePath);
        ext = '.html';
      } catch (e2) {
        // fall through to error handling below
      }
    }
  }

  if (data) {
    res.writeHead(200, { 'Content-Type': MIME[ext] || 'application/octet-stream' });
    res.end(data);
    return;
  }

  try {
    // 再读一次触发错误走后续逻辑
    await readFile(filePath);
  } catch (e) {
    // 目录列表（简易版）
    if (e.code === 'EISDIR') {
      const { readdir } = await import('node:fs/promises');
      const files = await readdir(filePath);
      const html = files.map(f => `<a href="${f}">${f}</a>`).join('<br>');
      res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
      res.end(`<h2>📁 ${req.url}</h2>${html}`);
      return;
    }
    res.writeHead(404);
    res.end('Not Found: ' + req.url);
  }
});

server.listen(PORT, '0.0.0.0', () => {
  console.log(`\n  🚀 发布服务器已启动\n`);
  console.log(`  根目录:  ${ROOT}`);
  console.log(`  图片目录: ${ASSETS_DIR}`);
  console.log(`  地址:    http://localhost:${PORT}/\n`);
  console.log(`  POST /save-image  — 保存图片\n`);
});
