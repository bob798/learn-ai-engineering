# AI 时代产品原型工具评测：哪个最适合交给 Claude Code？

> 关联 Issue: [#4](https://github.com/bob798/learn-ai-engineering/issues/4)

## 核心问题

在 AI 编程时代，产品从想法到实现的路径变了：

```
传统：想法 → PRD → 设计稿 → 前端开发 → 后端开发 → 联调
AI 时代：想法 → 原型工具生成 → AI 编程工具接手 → 可运行产品
```

**关键选型问题**：哪个原型工具生成的代码，最适合交给 Claude Code 继续迭代？

本文从**代码质量**（AI 可理解性、组件化、可维护性）角度，横向评测 6 个主流工具。

---

## 评测对象

| 工具 | 厂商 | 定位 | 集成方式 |
|------|------|------|----------|
| [v0](https://v0.dev) | Vercel | Prompt → React 组件 | GitHub PR |
| [Bolt.new](https://bolt.new) | StackBlitz | Prompt → 全栈应用 | GitHub 导出 |
| [Lovable](https://lovable.dev) | Lovable | Prompt → SaaS MVP | GitHub 双向同步 |
| [Claude Artifacts](https://claude.ai) | Anthropic | 对话式 UI 原型 | 手动复制代码 |
| [Pencil](https://pencil.dev) | Pencil | IDE 内矢量设计 | MCP 协议原生集成 |
| [Firebase Studio](https://firebase.google.com/docs/studio) | Google | 云端 AI 开发环境 | Zip 导出 |

---

## 评测维度

| 维度 | 权重 | 说明 |
|------|------|------|
| **代码质量** | 35% | 组件化程度、TypeScript、标准库使用 |
| **AI 可迭代性** | 30% | 生成的代码能否被 Claude Code 理解并持续修改 |
| **与 Claude Code 衔接** | 20% | 导出/集成的便捷程度 |
| **输入表达力** | 15% | 支持的输入方式（文本、截图、Figma、手绘等） |

---

## 逐工具深度评测

### 1. v0 (Vercel)

**核心能力**：文本描述 / 截图 / Figma 设计稿 → Next.js 全栈应用

| 维度 | 评分 | 说明 |
|------|------|------|
| 代码质量 | ⭐⭐⭐⭐⭐ | shadcn/ui + Tailwind，组件化最规范 |
| AI 可迭代性 | ⭐⭐⭐⭐⭐ | 标准 React 组件，Claude Code 无缝理解 |
| 衔接方式 | ⭐⭐⭐⭐ | 自动创建 GitHub 分支并提 PR |
| 输入表达力 | ⭐⭐⭐⭐ | 文本 + 截图 + Figma + URL |

**技术栈**：Next.js + TypeScript + Tailwind CSS + shadcn/ui + Radix UI

**定价**：免费 $5/月额度（约 7 条/天），Team $30/人/月

**与 Claude Code 衔接流程**：
```
v0 生成 → 自动 push GitHub 分支 → git clone → Claude Code 接手
```

**优势**：
- 代码规范性最高，直接使用 shadcn/ui 标准组件
- 自动错误检测和修复
- 一键部署到 Vercel

**局限**：
- 锁定 Next.js 技术栈
- 免费额度极少（7 条/天）
- 无 CLI，不能终端直接调用

---

### 2. Bolt.new (StackBlitz)

**核心能力**：文本 / 图片 / Figma / GitHub 导入 → 浏览器内全栈运行

| 维度 | 评分 | 说明 |
|------|------|------|
| 代码质量 | ⭐⭐⭐⭐ | 标准 npm 项目结构，TypeScript |
| AI 可迭代性 | ⭐⭐⭐⭐ | 结构清晰，但大项目 token 消耗后质量下降 |
| 衔接方式 | ⭐⭐⭐⭐ | 导出到 GitHub，标准项目结构 |
| 输入表达力 | ⭐⭐⭐⭐⭐ | 文本 + 图片 + Figma + GitHub 仓库导入 |

**技术栈**：React / Vue / Svelte / Astro / Next.js（可选），Node.js + Express + PostgreSQL + Prisma

**定价**：免费 30 万 token/天，Pro $25/月（1000 万 token/月）

**与 Claude Code 衔接流程**：
```
Bolt 生成 → 导出 GitHub → git clone → Claude Code 接手
```

**优势**：
- **技术栈最灵活**（唯一支持 Vue/Svelte 的工具）
- WebContainer 技术，浏览器内运行完整 Node.js
- AI 模型使用 Claude Sonnet/Opus，代码质量有保障
- 免费额度最慷慨

**局限**：
- WebContainer 不支持原生二进制、Python pip
- 大型项目上下文越大，生成质量越不稳定
- 无法运行需要系统级依赖的后端

---

### 3. Lovable

**核心能力**：Prompt / 截图 / Figma → 带数据库的全栈 Web 应���

| 维度 | 评分 | 说明 |
|------|------|------|
| 代码质量 | ⭐⭐⭐⭐ | React + TypeScript + shadcn/ui，规范性好 |
| AI 可迭代性 | ⭐⭐⭐⭐ | 标准结构，但后端强绑 Supabase |
| 衔接方式 | ⭐⭐⭐⭐⭐ | **GitHub 双向同步**，最流畅 |
| 输入表达力 | ⭐⭐⭐⭐ | Prompt + 截图 + Figma（需插件）+ 可视化编辑器 |

**技术栈**：React + TypeScript + Vite + Tailwind + shadcn/ui，后端 Supabase（PostgreSQL + Auth + Edge Functions）

**定价**：Pro $25/月（100 credits），Business $50/月

**与 Claude Code 衔接流程**：
```
Lovable 生成 → 自动同步 GitHub → git clone → Claude Code 本地开发 → push → Lovable 自动同步回来
```

**优势**：
- **唯一真正的双向 Git 同步** — Claude Code 修改后 push，Lovable 自动感知
- 前后端一体（含数据库、认证、文件存储）
- 适合快速构建 SaaS MVP

**局限**：
- 技术栈完全锁定（无法选 Next.js/Vue）
- 后端强依赖 Supabase，迁移成本高
- Credits 消耗快，复杂项目迭代贵

---

### 4. Claude Artifacts

**核心能力**：对话式生成 → 交互式 React 组件 / HTML 页面 / SVG 可视化

| 维度 | 评分 | 说明 |
|------|------|------|
| 代码质量 | ⭐⭐⭐⭐ | 标准 React Hooks，结构化良好 |
| AI 可迭代性 | ⭐⭐⭐⭐⭐ | Claude 自己生成的代码，Claude Code 100% 理解 |
| 衔接方式 | ⭐⭐⭐ | 手动复制粘贴，无自动化集成 |
| 输入表达力 | ⭐⭐⭐ | 纯文本对话（可附图片） |

**技术栈**：React（单文件）/ HTML+CSS+JS / SVG

**定价**：基础免费，Claude Design 需 Pro（$20/月）

**与 Claude Code 衔接流程**：
```
Claude.ai 对话迭代原型 → 复制代码 → 粘贴到本地项目 → Claude Code 拆分为多文件工程
```

**优势**：
- **对话式迭代最自然** — 说一句改一处，即时预览
- Claude Design 可读取品牌设计系统，输出风格一致的原型
- 与 Claude 生态天然联动（同一上下文无缝过渡）
- **AI 可迭代性最高** — Claude 自己写的代码，自己最懂

**局限**：
- 单文件限制，不适合复杂多页应用
- 无后端、无持久化、无部署
- 衔接靠手动复制粘贴，无 GitHub 集成

---

### 5. Pencil

**核心能力**：IDE 内矢量设计 → 通过 MCP 协议让 AI 直接操作设计 → 导出 React/Tailwind 代码

| 维度 | 评分 | 说明 |
|------|------|------|
| 代码质量 | ⭐⭐⭐⭐ | React + Tailwind + shadcn/ui，支持设计 token |
| AI 可迭代性 | ⭐⭐⭐⭐⭐ | MCP 原生，AI 结构化读写设计意图 |
| 衔接方式 | ⭐⭐⭐⭐⭐ | **MCP 直连 Claude Code，零摩擦** |
| 输入表达力 | ⭐⭐⭐⭐⭐ | 自然语言 + 可视化画布 + Figma 导入 |

**技��栈**：React (JS/TS) + Next.js + Tailwind CSS，支持 shadcn/ui / Radix / Chakra

**定价**：目前完全免费

**MCP 集成三工具**：

| 工具 | 作用 |
|------|------|
| `batch_get` | AI 读取组件层级与元素属性 |
| `batch_design` | AI 增删改移设计元素，直写 .pen 文件 |
| `get_screenshot` | 渲染预览截图，AI 验证修改结果 |

**与 Claude Code 衔接流程**：
```
IDE 内 Pencil 画布设计 → Claude Code 通过 MCP 直接读写 .pen 文件
→ 导出 React 代码 → Claude Code 继续开发
（全程在同一 IDE 内，无需切换工具）
```

**优势**：
- **唯一 MCP 原生集成** — AI 不是"看截图猜"，而是结构化理解设计意图
- .pen 文件进 Git，设计与代码同生命周期
- 消除 Figma ↔ IDE 切换摩擦
- 支持最多 6 个 AI Agent 并发生成方案
- 当前完全免费

**局限**：
- 工具非常新，生态小，社区资源少
- 仅支持 VS Code / Cursor
- 设计能力与 Figma 成熟度有差距
- 复杂场景代码生成质量缺乏公开基准

---

### 6. Firebase Studio (Google)

**核心���力**：云端 AI 开发环境，Gemini 驱动的全栈原型生成

| 维度 | 评分 | 说明 |
|------|------|------|
| 代码质量 | ⭐⭐⭐ | 模板系统完善，但 Gemini 生成质量不如 Claude |
| AI 可迭代性 | ⭐⭐⭐ | 标准结构可导出，但 Gemini 风格代码需适配 |
| 衔接方式 | ⭐⭐⭐ | Zip 导出，无 Git 自动同步 |
| 输入表达力 | ⭐⭐⭐⭐ | 多模态 Prompt + 模板系统 |

**技术栈**：React / Next.js / Angular / Vue / Flutter / React Native + Firebase 后端

**定价**：免费（3 工作区），Premium 按量计费

**与 Claude Code 衔接流程**：
```
Firebase Studio Prototyper 生成 → Zip 下载 → 本地解压 → Claude Code 接手
```

**优势**：
- 技术栈覆盖最广（含 Flutter、React Native 移动端）
- Firebase 后端一键集成（Firestore、Auth、Hosting）
- 无需本地环境配置

**局限**：
- ⚠️ **将于 2027 年 3 月关闭**，2026 年 6 月起不再支持新建工作区
- 迁移方向：Google Antigravity（本地 IDE）
- 开发体验绑定 Gemini，不支持替换为 Claude
- 导出方式原始（Zip），无 Git 集成

---

## 综合对比矩阵

| 工具 | 代码质量 | AI 可迭代性 | 衔接方式 | 输入表达力 | 综合评分 |
|------|---------|-----------|---------|-----------|---------|
| **v0** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | **4.6** |
| **Pencil** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | **4.6** |
| **Bolt.new** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | **4.2** |
| **Lovable** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | **4.2** |
| **Artifacts** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | **3.8** |
| **Firebase Studio** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | **3.2** |

---

## 选型建议：按场景推荐

### 场景 A：快速验证 UI 创意

**推荐：Claude Artifacts → Claude Code**

```
"我想做一个类似 Linear 的任务看板"
→ Claude Artifacts 对话生成原型（3 分钟）
→ 复制代码到项目
→ Claude Code 拆分组件、加路由、接 API
```

适合：想法初期、快速看到效果、无需精细设计

### 场景 B：设计驱动开发

**推荐：Pencil + Claude Code (MCP)**

```
在 VS Code 内 Pencil 画布设计
→ Claude Code 通过 MCP 理解设计结构
→ 一键导出 React 组件
→ Claude Code 继续完善业务逻辑
```

适合：重视设计质量、设计与代码需要同步迭代、团队协作

### 场景 C：生产级前端组件

**推荐：v0 → Claude Code**

```
v0 生成 shadcn/ui 标准组件
→ GitHub PR 自动推送
→ Claude Code review + 集成到项目
```

适合：已有 Next.js 项目、需要高质量可复用组件

### 场景 D：全栈 MVP 原型

**推荐：Lovable → Claude Code**（需要后端）或 **Bolt.new → Claude Code**（灵活技术栈）

```
Lovable: 一句话生成带数据库的完整应用 → GitHub 双向同步 �� Claude Code 迭代
Bolt: 选择任意技术栈快速出原型 → 导出 → Claude Code 重构为生产级
```

适合：需要快速交付完整可运行产品

---

## 关键洞察

### 1. MCP 是未来

Pencil 通过 MCP 协议让 AI **结构化理解设计意图**（而非看截图猜），这代表了 AI 工具协作的未来方向。当更多设计工具支持 MCP，"设计 → 代码"的转换将从"翻译"变为"同构"。

### 2. 代码规范性 > 功能丰富性

对于"交给 Claude Code 继续开发"这个目标，**代码是否规范**比**功能是否完整**更重要。v0 生成的 shadcn/ui 代码几乎不需要重构就能继续迭代；而功能更全的工具如果代码结构混乱，反而增加 AI 理解成本。

### 3. Claude 生态闭环正在形成

```
Claude Artifacts (原型) → Pencil (设计，MCP) → Claude Code (实现) → Claude (Review)
```

Anthropic 生态内的工具链正在形成闭环，上下文和意图在工具间无损传递。

### 4. Firebase Studio 的教训

Google 的 Firebase Studio 即将关闭，说明**平台锁定是真实风险**。选择工具时优先考虑：输出物是否标准（Git + npm）、能否无痛迁移。

---

## 参考资源

- [v0 官方文档](https://v0.app/docs/)
- [Bolt.new 技术架构](https://github.com/stackblitz/bolt.new)
- [Lovable 文档](https://docs.lovable.dev/)
- [Claude Artifacts 帮助](https://support.claude.com/en/articles/9487310)
- [Pencil AI 集成文档](https://docs.pencil.dev/getting-started/ai-integration)
- [Firebase Studio 迁移公告](https://firebase.google.com/docs/studio/migrating-project)
- [Claude Design 发布公告](https://www.anthropic.com/news/claude-design-anthropic-labs)
