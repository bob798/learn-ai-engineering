# CLAUDE.md — ai-handbook

> 这是 Bob 的 AI 知识库项目的 Claude Code 全局上下文文件。
> 所有 Claude Code 会话读取此文件以理解项目背景、规范和当前状态。

---

## 项目身份

**项目名称**：ai-handbook（AI 修炼册）
**GitHub**：bob798/ai-handbook
**定位**：个人 AI 学习路径知识库，六境修炼体系，从 AI 认知到系统架构
**作者背景**：8年 ToB 预售/售前工程师（支付与金融科技），正在转型 AI 应用工程师，目标成为 AI Agent 编排专家

---

## 仓库结构

```
ai-handbook/
├── CLAUDE.md              ← 本文件，Claude Code 全局上下文
├── README.md              ← 项目介绍与修炼路径总览
│
├── methodology/           ← 学习方法论（5D 框架）
│   └── interactive.html
│
├── rag/                   ← L3 筑基境：RAG 知识体系
│   ├── v1/ ~ v3.5/        ← 渐进式 Python 实现
│   └── rag-5d.html        ← 交互式 RAG 学习文档（含冰山知识图）
│
├── mcp/                   ← L4 金丹境：MCP 协议
│   └── mcp-interactive.html
│
├── agent/                 ← L4 金丹境：Agent 架构
│   ├── agent-interactive.html
│   └── planning-reasoning.html（17 Tab）
│
├── llm/                   ← L3/L4：LLM 哲学与原理
│   └── llm-philosophy.html
│
├── ai-programming/        ← L4/L5：AI 编程与团队协同
│   ├── ai-xiulian-ce.html ← 六境修炼全景地图（本次新增）
│   └── cases/
│       └── oh-my-claudecode.md  ← OMC 案例分析
│
└── speakeasy/             ← 实战项目：语言学习 App（应用示例）
```

---

## 当前修炼境界

```
Bob 当前位置：L4 金丹境（Agent 工程）

已完成：L0 凡人境 · L1 入道境 · L2 炼气境 · L3 筑基境
进行中：L4 金丹境 — Agent 架构、多 Agent 编排、Claude Code 工程实践
目标境：L5 元婴境 — 生产级多 Agent 系统架构
```

---

## 核心原则（Claude Code 行为规范）

### 知识生产规范

**分离关注点**：
- Python / JS 代码文件：简洁，注释用英文，不写长篇解释
- HTML 学习文档：深度概念解释住在这里，与代码分开
- CLAUDE.md（本文件）：项目上下文，不写知识内容

**5D 知识框架**（每个主题都按此拆解）：
1. **Decompose**：分层解构，2-4 层，每层关注点清晰
2. **Distinguish**：对比辨析，有 vs 没有，A 方案 vs B 方案
3. **Draw Analogy**：类比迁移，找跨域同构结构
4. **Debug**：错误雷达，主动标出常见误解
5. **Deploy**：实战锚定，Speakeasy / 音曼客服 / 面试场景

**知识深度层次**（L1-L5）：
- L1 记忆 → L2 结构 → L3 迁移 → L4 批判 → L5 创造
- L3（迁移）是"真正理解"的分水岭

### HTML 文档规范

```
设计语言：
- 主色：#F26419（橙色）用于 Agent/RAG 内容
- 副色：#7c3aed（紫色）用于 LLM 哲学内容
- 字体：JetBrains Mono（代码）+ Noto Sans SC（中文）
- 背景：#f5f4f0（暖灰）
- 代码高亮：Dracula 风格

Tab 导航规范：
- 总览 Tab 永远是第一个
- Tab 数量 3-8 个，超过则分文档
- 重要警示用 ⚠️ 标记

颜色语义：
- 橙色边框 callout = 核心结论
- 绿色 gbox = 推荐/正确
- 红色 rbox = 错误/警告
- 蓝色 callout-blue = 补充说明
```

### 文件命名规范

```
目录：单词小写，无连字符（rag/ mcp/ agent/ llm/）
HTML：功能-描述.html（rag-5d.html, ai-xiulian-ce.html）
Python：v{版本号}_描述.py（v1_basic_rag.py）
```

---

## 活跃项目（当前 L4 重点）

### 1. 音曼客服 Agent
```
架构：三角色（接待 / 专家 / 验收）
技术栈：Python + Claude API + MCP
状态：设计阶段，正在研究 Agent 编排方案
参考：oh-my-claudecode 案例分析
```

### 2. AI 编程知识库（本仓库 ai-programming/ 目录）
```
已完成：六境修炼全景地图（ai-xiulian-ce.html）
进行中：OMC 案例深度分析文档
待完成：AI 团队协同规范模板
```

### 3. RAG 课程（v1-v10 渐进式）
```
已完成：v1（基础）~ v3.5（评估基线 Recall@3）
进行中：v4（检索质量优化）
待完成：v8（RAGAS 完整评估框架）
```

### 4. Speakeasy 语言学习 App
```
角色：所有 Agent 和规划概念的一致性应用示例
当前：RAG 词典功能设计（Resource 而非 Tool）
```

---

## Claude Code 行为指引

### 新文档默认位置

```
所有新文档默认整合到 ai-programming/src/ 目录：
- 使用 shared.css（深色主题）+ shared.js（Tab 切换等交互）
- 遵循标准结构：back-nav → hero → nav-bar（Tab）→ .main > .card → footer → drawer
- 复用组件：.card / .dtable / .icard / .code-wrap / .steps / .vs / .feat-grid 等
- 同时更新 src/index.html 的 nav-cards 和知识体系表格，添加新文档入口
- 不要创建独立的浅色主题 HTML 或放在 src/ 以外的位置
```

### 生成代码时

```
- Python 文件：保持简洁，顶部注释说明版本和功能
- HTML 文件：完整独立可在浏览器直接打开，不依赖外部资源
- 向量数据库：使用 ChromaDB（持久化 + metadata 过滤 + 增量 upsert）
- 模型提供商：SiliconFlow / Zhipu AI / Qwen（中国优先），OpenAI（抽象备选）
- 提供商层：始终通过 provider abstraction layer 调用，不硬编码
```

### 生成学习文档时

```
- 每个 HTML 文档必须有总览 Tab（第一个）
- 代码示例：中文注释 + 英文代码
- 必须包含：类比部分（D3）+ 常见误解部分（D4）
- 必须有具体的 Speakeasy 或音曼客服应用场景（D5）
- 冰山知识图：如果主题复杂，添加 iceberg 可视化
```

### 修改已有文件时

```
- 先读文件，理解现有结构，再修改
- 不破坏已有的 Tab 结构和 CSS 变量
- 新增 Tab 插入在倒数第二位（最后一个 Tab 保留给"实战"或"总结"）
```

### 遇到不确定的需求时

```
- 先确认是要"新文档"还是"扩展现有文档"
- 先确认目标境界（L0-L5）和目标读者
- 按 5D 框架列提纲，确认后再生成
```

---

## 关键概念备忘

```
N×M → N+M：
  减少的是开发工作量，不是连接数。这是已纠正的关键误解。

Vector dimensions：
  每个维度不代表具名特征。RGB 颜色类比（组合涌现意义）是正确心智模型。

Agent vs Harness：
  Agent 是模型本身；Harness 是工程师构建的执行框架。

冰山知识图关联关系：
  放在冰山面板底部（下潜后看到），不作为主线弧线显示。

v3.5 范围原则：
  轻量评估基线（Recall@3）属于早期课程；RAGAS 完整框架属于 v8。
```

---

## Push 说明

```bash
# 本地没有 PAT，每次 push 需要在 session 外手动操作
# 文件生成后提交到本地，push 由 Bob 手动执行

git add .
git commit -m "feat: [描述]"
git push origin main
```

---

*最后更新：2026-04 · 当前境界：L4 金丹境（Agent 工程）*
