---
title: 文章发布流程指南
description: Markdown 源文件 → 公众号/多平台发布的标准化流程，基于 WechatSync 工具链
date: 2026-06-28
---

# 文章发布流程指南

从 Markdown 源文件到公众号及多平台发布的完整流程。

---

## 工具链

| 工具 | 用途 |
|---|---|
| Markdown 源文件 | 内容的唯一真实来源（Single Source of Truth） |
| 发布用 HTML（`*-wechat.html`） | 中间产物：公众号排版 + 图表渲染 + WechatSync 集成 |
| [WechatSync](https://github.com/wechatsync/Wechatsync) | Chrome 扩展，一篇文章同步到 21+ 平台 |

---

## 第一阶段：内容准备（Markdown）

写作阶段只关注内容，不关注排版。在 Markdown 源文件中完成以下检查：

### 必备字段

```yaml
---
title: 文章标题（公众号限 64 字，知乎限 30 字）
description: 技术描述（SEO 用）
abstract: 120 字以内的公众号摘要（用于消息列表预览）
status: published
date: 发布日期
tags: [标签列表]
---
```

### 内容检查清单

- [ ] **开头钩子**：前两段必须通俗易懂，非技术读者也能看进去
- [ ] **图表**：用 Mermaid 语法写（不用 ASCII art，公众号无法渲染）
- [ ] **结尾三件套**：
  - 系列预告 / 下一篇导读
  - 互动问题（引导留言）
  - 关注引导（可选）
- [ ] **参考资料**：
  - 去掉 `https://` 前缀（公众号显示为死链）
  - 加提示语：「请点击阅读原文查看完整版」
- [ ] **status** 改为 `published`

---

## 第二阶段：生成发布用 HTML

### 让 Claude Code 生成

对 Claude Code 说：

> 把 `content/xxx.md` 生成公众号发布用 HTML

Claude Code 会生成 `content/xxx-wechat.html`，包含：

1. **封面图区域**（截图用）
2. **Mermaid 图表预览**（自动渲染为 SVG）
3. **正文区**（`<article>` 标签，内联 CSS）
4. **WechatSync 集成**（一键同步按钮）

### HTML 设计规范

后续所有文章的 `-wechat.html` 都遵循以下结构：

```
┌─────────────────────────┐
│  顶部工具栏（fixed）     │  ← 状态指示 + WechatSync 按钮
├─────────────────────────┤
│  素材区（可折叠）        │  ← 封面图 + Mermaid 图表预览
├─────────────────────────┤
│  <article> 正文区       │  ← WechatSync 提取目标
│    - 全部内联 CSS        │     WeChat 会 strip 掉 <style>
│    - Mermaid → PNG 自动  │     渲染后自动注入 <img>
│    - 表格有内联边框      │
│    - 引用块有内联样式    │
│    - 分隔用 · · ·       │
│    - 无外部链接          │
└─────────────────────────┘
```

### 关键技术点

**为什么正文全部用内联 CSS？**
WeChat 编辑器会 strip 掉 `<style>` 标签和 `class` 属性，只保留 `style` 属性。所以 `<article>` 内的所有元素必须用内联样式。

**Mermaid 图怎么变成图片？**
HTML 加载时，Mermaid 在素材区渲染为 SVG → JS 用 Canvas 转为 PNG data URL → 自动注入正文 `<div id="slot-N">`。全自动。

**封面图怎么生成？**
html2canvas 自动将封面 div 渲染为 PNG，缓存后可下载，同步时通过 `syncPost({ thumb })` 尝试自动上传。

**为什么有两个同步按钮？**
公众号不接受 data URL 图片（base64），必须上传到微信 CDN。所以公众号走「无图同步 + 手动插图」，其他平台走「含图同步」。

---

## 第三阶段：发布（WechatSync）

### 前置条件

1. 安装 [WechatSync Chrome 扩展](https://github.com/wechatsync/Wechatsync)
2. 在 Chrome 中登录所有目标平台（公众号、知乎、掘金 等）
3. 必须通过 HTTP 服务打开（非 `file://`）：`npx serve <目录> -p 8787`

### 发布步骤

```
1. 终端运行：npx serve content/02-agent/concepts -p 8787
2. Chrome 打开 http://localhost:8787/xxx-wechat.html
3. 等状态栏变为 ✅ 就绪
--- 公众号 ---
4a. 点击「📤 公众号同步」（自动去掉正文图片，尝试上传封面）
5a. 同步后进公众号编辑器：
    - 在【图 N】标记处上传下载好的 PNG（点「⬇️ 下载图表」可批量下载）
    - 若封面未自动上传，手动上传封面 PNG
    - 设置「阅读原文」链接
    - 勾选「原创」→ 发布

--- 知乎 / 掘金 / CSDN 等 ---
4b. 点击「📤 多平台同步」（含图，图片以 data URL 嵌入正文）
5b. 选择目标平台，确认标题和摘要，发布
```

### 平台差异说明

| 平台 | 注意事项 |
|---|---|
| 公众号 | 不支持外链、不支持 SVG、需单独上传封面图、需设置「阅读原文」 |
| 知乎 | 支持外链，可选保留参考资料原始 URL |
| 掘金 | 支持 Markdown，可直接从源 .md 发（但图表需替换为图片） |
| CSDN | 支持 Markdown + HTML 混排 |
| 头条号 | 含外链的文章会降级为草稿（WechatSync 已处理） |

---

## 改造清单模板

每篇文章从 Markdown 到发布，需要做以下改造。可以让 Claude Code 一次性完成：

```
### 元信息
- [ ] 加 abstract（120 字）
- [ ] status → published
- [ ] date → 发布日期

### 开头
- [ ] 加通俗钩子（1-2 段）

### 图表
- [ ] ASCII art → Mermaid 语法
- [ ] 确认 Mermaid 在网站上可渲染

### 结尾
- [ ] 系列预告 / 下一篇导读
- [ ] 互动引导问题
- [ ] （可选）关注引导

### 参考资料
- [ ] URL 去掉 https://
- [ ] 加「阅读原文」提示

### 生成 HTML
- [ ] 生成 *-wechat.html
- [ ] 验证：图表渲染正常
- [ ] 验证：WechatSync 能提取正文

### 发布
- [ ] WechatSync 多平台同步
- [ ] 公众号设置封面图 + 阅读原文 + 原创
```

---

## Prompt 模板

后续发文可以直接对 Claude Code 说：

> 按照 `content/publishing-guide.md` 的流程，把 `content/xxx.md` 准备好发布到公众号和多平台。完成 Markdown 改造 + 生成 WechatSync 集成的 HTML。
