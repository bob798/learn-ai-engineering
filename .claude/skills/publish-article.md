---
name: publish-article
description: 将markdown文章转为微信公众号发布页面（含封面、多平台文案），提交到GitHub。用于《不可逆的Agent》系列及同类技术博客发布。
---

# 文章发布 Skill

## 触发条件
用户提供一个 markdown 文件路径，要求"发布"、"处理"、"提交"文章。

## 输入
- 一个 markdown 文件（纯正文，可能有/没有 frontmatter）
- 文章所属目录（默认 `content/02-agent/irreversible-agent/`）

## 输出文件
1. `{序号}-{短名}.md` — 带 frontmatter 的正式 markdown
2. `{序号}-{短名}-wechat.html` — 微信公众号发布工具页面

## 流程

### Step 1: 分析文章
- 读取 markdown 源文件
- 提取：标题、副标题、系列编号、核心观点（用于封面/文案）
- 确认目标目录和文件命名

### Step 2: 创建 Markdown（如需）
frontmatter 模板：
```yaml
---
title: {标题}
description: {系列描述，一句话}
abstract: {摘要，2-3句}
status: published
date: {当天日期}
depth: Medium
series: 不可逆的Agent
series_index: {序号}
tags: [{相关标签}]
prev: {上一篇短名}
next: {下一篇短名}
---
```

### Step 3: 创建 Wechat HTML

#### 视觉规范（《不可逆的Agent》系列）
```
横版封面底色: linear-gradient(135deg, #1a0a2e, #2d1b4e, #0f2027)
竖版封面底色: linear-gradient(160deg, #0a0a1a 0%, #1a0a2e 30%, #2d1050 60%, #1a0a2e 100%)
光晕: radial-gradient(ellipse at 30% 50%, rgba(120,80,200,0.15) ...) + rgba(0,180,220,0.1)
标题强调色: linear-gradient(90deg, #ff6b6b, #ffa36b) — 珊瑚橙渐变
XHS标题: linear-gradient(90deg, #ff6b6b, #ffa36b, #ffdb6b)
高亮元素: rgba(255,107,107,0.15) border rgba(255,107,107,0.5) color #ff6b6b
系列标签: border rgba(255,255,255,0.4) 圆角椭圆
```

#### HTML 结构
```
├── 工具栏（固定顶部）
│   ├── 素材区切换
│   ├── ⬇️ 横版封面 / ⬇️ 竖版封面
│   ├── 📋 复制正文 / 𝕏 复制推文 / 📕 小红书
│   └── 📤 公众号同步 / 📤 多平台同步
├── file:// 协议提示
├── 素材区
│   ├── 横版封面 900×500（.cover #coverEl）
│   └── 竖版封面 1080×1440（.cover-xhs #coverXHS）
├── 正文区 <article id="articleContent">
│   ├── 标题 + 系列标记
│   ├── 系列引言（灰底左边框）
│   ├── 各章节（h2 带底线 + 正文段落，内联样式）
│   ├── 关键金句（渐变背景卡片）
│   ├── 黄色提示框（伏笔/重要概念）
│   ├── 参考文献（如有）
│   ├── 系列导航
│   └── 引流尾部（crossPlatformCTA，默认hidden）
└── JavaScript
    ├── html2canvas 封面渲染/下载
    ├── 复制正文/推文/小红书文案
    ├── WechatSync 公众号+多平台同步
    └── file:// 协议检测
```

#### 正文内联样式速查
```css
段落: font-size:16px;color:#333;margin-bottom:16px;line-height:2;text-align:justify;
h2: font-size:20px;font-weight:700;color:#1a1a1a;margin:32px 0 16px;padding-bottom:8px;border-bottom:2px solid #333;
strong: color:#1a1a1a;
code: background:#f5f5f5;padding:2px 6px;border-radius:3px;font-size:14px;color:#c7254e;border:1px solid #e8e8e8;
pre/code块: background:#1e1e1e;color:#d4d4d4;padding:20px;border-radius:6px;font-size:14px;line-height:1.7;
金句卡片: font-size:18px;font-weight:700;text-align:center;padding:20px;background:linear-gradient(135deg,#f5f0ff,#f0f7ff);border-radius:8px;border:1px solid #e0d8f0;
黄色提示: background:#fff8e1;border-left:4px solid #ffb300;padding:16px 20px;font-size:15px;color:#5d4037;
灰色引言: background:#f7f7f7;border-left:4px solid #ddd;padding:12px 16px;font-size:14px;color:#888;
分隔符: text-align:center;color:#ccc;margin:28px 0;font-size:20px;letter-spacing:12px; 内容为 · · ·
互动区: background:#f9f9f9;border-radius:8px;padding:24px;font-weight:600;
```

### Step 4: 启动发布服务器
```bash
# 先检查端口是否被占用
lsof -t -i :8787
# 如果被占用，kill 后重启
kill $(lsof -t -i :8787) 2>/dev/null; sleep 1
node scripts/publish-server.mjs content/02-agent/irreversible-agent
```

### Step 5: 提交到 GitHub
```bash
git add "{md文件}" "{html文件}"
git commit -m "feat: 《不可逆的Agent》系列第{N}篇 — {短标题}"
git push
```
commit 消息格式：`feat:` 新文章，`fix:` 内容优化，`style:` 样式调整

## 推文模板
```
{标题}

{1-2句核心观点}

{3-4个要点，简洁}

#AI #Agent #{相关标签} #AIAgent

《不可逆的Agent》系列第 {N} 篇 | 全文见公众号「我是Bob大叔」
```

## 小红书文案模板
```
{标题}｜AI Agent

{引入，1-2句}

{分点展开，用 emoji 分隔，每点2-3行}

一句话总结：{金句}

这是《不可逆的 Agent》系列第 {N} 篇
完整系列 → 公众号「我是Bob大叔」

#AI #Agent #{标签} #LLM #AIAgent #人工智能 #AI学习 #编程 #技术干货
```

## 注意事项
- 封面设计要体现文章主题，但必须用统一配色
- 正文所有样式必须内联（微信编辑器不支持 class）
- 参考文献用小号灰色字体，不喧宾夺主
- 公众号名固定为「我是Bob大叔」
