---
title: 知识库扩展策略 · LLM 基础与 Spring AI 工程双主题
description: 在现有 MCP/Agent/RAG/AI Programming 四主题之上，新增 05-llm-foundations 与 06-spring-ai 的论证、定位、节奏与执行规范
status: v1 · 2026-05-11 起草
---

# 知识库扩展策略 · LLM 基础与 Spring AI 工程

> **结论先行**：在现有 4 主题之外，新增两个主题——
> - `05-llm-foundations/` 通用 LLM 基础概念（受众宽，跨语言）
> - `06-spring-ai/` Spring AI 工程实战（受众窄，Java 工程师专属）
>
> 两者**独立**而非合并，理由见下文 §3。

---

## 1. 为什么要扩展（动机）

现有 4 主题（MCP / Agent / RAG / AI Programming）覆盖**应用层**，但有两块明显空缺：

### 空缺 ① · LLM 基础概念
- 没有系统讲过 **Inference vs Reasoning 术语区分、prefill/decode、MoE vs Dense、显存模型、激活值与梯度**
- 这些是当前 4 主题暗中依赖的"前置知识"，散落在各篇里被一笔带过
- 国内 AI 教程严重同质化，但**真正讲清"中文都翻译成推理"这种术语坑**的稀缺

### 空缺 ② · Spring AI 工程
- 现有 `code/` 目录只有 Python（RAG / ReAct / MCP demo）
- Java 后端工程师转 AI 是国内开发者的最大转型潮，缺少**针对性中文工程内容**
- 已经积累的 Spring AI 踩坑（parallel_tool_calls、Structured Output、ChatMemory、流式 toolCalls 丢失）值得沉淀

---

## 2. 与现有 4 主题的关系（不重叠论证）

| 主题 | 关注层 | 与新主题的关系 |
|---|---|---|
| `01-mcp/` | 协议层 | 不重叠（MCP 是工具协议，新主题是模型理解 + Java 工程） |
| `02-agent/` | 系统设计层 | 弱重叠（Agentic 概念会引用 `05` 的基础；ReAct 实现会引用 `06`） |
| `03-rag/` | 检索增强层 | 不重叠（向量库/重排是独立栈） |
| `04-ai-programming/` | 编程方法论 | 不重叠（讲怎么用 AI 编程，不讲 AI 模型本身） |
| **`05-llm-foundations/`** ⭐ | **模型理解层** | **新增** |
| **`06-spring-ai/`** ⭐ | **Java 工程层** | **新增** |

**Agent 主题与 05 的边界划分**：
- 凡是"模型本身的能力/限制"（reasoning、MoE、tool calling 行为）→ 归 `05`
- 凡是"Agent 系统设计模式"（ReAct、Plan-Execute、多 agent 协作）→ 归 `02`
- 边界模糊时，从读者搜索意图判断：搜"什么是推理模型"→ `05`，搜"怎么搭 ReAct agent"→ `02`

---

## 3. 为什么 05 / 06 要分开而不合并

考虑过的合并方案：`05-llm-engineering/` 一个主题包含基础概念 + Spring AI。否决理由：

| 维度 | 分开（推荐） | 合并 |
|---|---|---|
| 受众 | 05 跨语言通用 / 06 Java 专属 | 强行绑定，赶走 Python/Go 读者 |
| 发布节奏 | 独立，互不阻塞 | 一边卡住另一边也得等 |
| SEO 检索 | 关键词分离更清晰 | "Spring AI" 关键词被基础概念稀释 |
| 维护成本 | 一个文件一个目的 | 后期分目录拆分代价大 |
| 站点导航 | 主题卡片各占一格 | 卡片标题难取（"基础+Spring AI"？） |

---

## 4. 主题定位语句

### `05-llm-foundations/`
> **"LLM 应用工程师必懂、但没人系统讲清楚的'中间层'概念"**——
> 不教"什么是 LLM"（太基础），不教"怎么搭 RAG"（太上层），专注模型本身的工作原理与术语精度。

**核心差异化**：术语精度 + 追问深度。例：
- 不写"LLM 推理过程"，写"中文都翻译成'推理'，但 Inference 和 Reasoning 是两回事"
- 不写"模型很大要用 GPU"，写"激活值算梯度——为什么训练比推理贵 4 倍显存"

### `06-spring-ai/`
> **"给 Java 后端工程师讲清楚 AI 应用开发——用 Spring AI 当锚点，从一个具体 bug 挖到模型架构"**

**核心差异化**：踩坑驱动 + Java 工程师视角。例：
- 不写"Spring AI 教程"，写"我用 Spring AI 调 o3，被一个 400 报错教做人"
- 不写"Spring AI 工具调用"，写"流式响应丢 toolCalls？别 debug 了，是 Spring AI 的 bug #3366"

---

## 5. 沿用现有写作原则（不另立规则）

继承 README.md 已经声明的写作哲学：
- ✅ **保留误解**：每篇至少一个"我以为是 X，结果是 Y"的反转
- ✅ **保留追问**：核心概念要 5+ 轮深度，不做 wiki 概括
- ✅ **保留体系**：归入 `NN-topic/NN-subtopic/` 的可导航结构
- ✅ **保留交互**：复杂概念配 `interactive/` HTML（条件成熟时）

新增一条**针对快速变化领域的规则**：
- ✅ **保留时效声明**：每篇头部 frontmatter 增加 `verified-on: YYYY-MM-DD` 和 `model-versions: xxx` 字段，超过 6 个月未验证的内容前面挂"⚠️ 未近期验证"提示

---

## 6. 子分类设计（双主题对照）

```
content/05-llm-foundations/
  ├── README.md               主题入口
  ├── PLAN.md                 选题路线图与状态
  ├── 01-runtime/             运行时机制（prefill/decode/RT/显存/带宽）
  ├── 02-architecture/        架构（MoE vs Dense / Reasoning 模型 / 多模态）
  ├── 03-training/            训练范式（forward/backward / RLHF/DPO/RLVR / 蒸馏量化）
  ├── 04-output-control/      输出控制（Structured Output / Constrained Decoding）
  ├── 05-terminology/         术语精度（Inference vs Reasoning / Agent vs Agentic / 命名后缀）
  └── 06-model-zoo/           模型选型（国产对比 / 评测基准）

content/06-spring-ai/
  ├── README.md               主题入口
  ├── PLAN.md                 选题路线图与状态
  ├── 01-pitfalls/            踩坑实录（按时间倒序）
  ├── 02-core-concepts/       核心概念（ChatClient / Advisor / ChatMemory / VectorStore）
  ├── 03-tool-calling/        工具调用（Function / MCP / Parallel / 流式）
  ├── 04-structured-output/   结构化输出（entity / 国产兼容性）
  ├── 05-domestic-models/     国产模型适配（DeepSeek / Qwen / GLM / 豆包配置与差异）
  └── 06-production/          生产化（重试 / 限流 / 可观测 / 成本）
```

**注意**：子分类**初期可以先空着不建**，等有文章了再开。避免一堆空目录污染 `extract-content.ts` 输出。

---

## 7. 配套资源（与现有 code/ interactive/ 对齐）

```
code/spring-ai/                ⭐ 新增 — Java + Spring Boot 可运行示例
  ├── parallel-tool-calls/
  ├── structured-output-vendors/   国产模型 strict 兼容性测试
  ├── chat-memory-modes/
  └── README.md

interactive/llm-foundations/   ⭐ 新增 — 复杂概念可视化
  ├── prefill-vs-decode.html       (条件成熟时做)
  ├── moe-routing.html
  └── thinking-strict-conflict.html
```

**优先级**：先把 `content/` 充实起来，`code/` 等有 3+ 篇核心文章后再补，`interactive/` 等到要做"爆款页"时再做。**不要因形式拖累内容产出**。

---

## 8. 90 天发布节奏

> **铁律**：周更 1 篇 `content/` 文章。低于这个频率，自己会忘，读者也会忘。

### Month 1（基础铺底，建可信度）
- W1：`05/05-terminology/inference-vs-reasoning.md`（差异化最强，**必爆款候选**）
- W2：`06/01-pitfalls/parallel-tool-calls-failures.md`（具体到一个 bug，引流 Spring AI 关键词）
- W3：`05/01-runtime/prefill-vs-decode.md`（底层但实用）
- W4：`05/05-terminology/agent-vs-agentic.md` + `naming-suffix-decoder.md`（短小精悍可合并）

### Month 2（深度展开，证明体系）
- W5：`05/02-architecture/moe-vs-dense.md`
- W6：`06/04-structured-output/vendor-compatibility.md`
- W7：`05/06-model-zoo/domestic-models-2026.md`（**工具型内容长尾流量高**）
- W8：`05/04-output-control/thinking-strict-conflict.md`（差异化点 #2）

### Month 3（系统化收口）
- W9-10：`06/02-core-concepts/` 补 ChatClient / Advisor / ChatMemory 三件套
- W11：`06/03-tool-calling/mcp-integration.md`
- W12：**回顾 + 路线图更新**——看哪些火、哪些没火，调整后 90 天

### 第 90 天的成功指标
- `content/05` + `content/06` 加起来 ≥ 12 篇正文
- 至少 1 篇被站外引用或转载
- 至少 3 篇收到"终于有人讲清楚了"类反馈
- README 总览表格更新到 6 主题

---

## 9. 给作者的工作流建议（高 ROI）

### 工作流 A：踩坑 → 文章（强烈推荐）
1. 日常开发遇到 Spring AI 问题 → 在 `06/01-pitfalls/_inbox/<date>-<problem>.md` 速记 3 要素：现象 / 排查 / 根因
2. 积累 3-5 个速记后，挑一个有"反转"价值的扩写成正文
3. 速记不达标的也别删——后续可能合并成"踩坑合集"

### 工作流 B：对话 → 文章
1. 跟 LLM 深度追问某概念时，**全程保留对话**
2. 把对话里"自己问出的好问题"提取出来作为文章的 H2
3. 文章不是"知识汇总"，是"我当时为什么这样问、回答如何让我反转认知"——天然带追问感

### 工作流 C：术语对齐表（5 分钟产出，长尾流量高）
- 建一个 `05/05-terminology/glossary.md` 持续维护
- 每发现一个易混淆术语就加一行（中英对照 + 一句话区别）
- SEO 友好，会被搜索"X 是什么意思"的人捞到

---

## 10. 必须避免的失败模式

1. ❌ **从"系列文章"膨胀为"写书"**——每篇独立成文，不做严格章节依赖
2. ❌ **追 Qwen/DeepSeek 新版本而打散主线**——热点文章作为"番外"，不编入主序号
3. ❌ **多渠道同时铺**——掘金或公众号选一个稳了再扩
4. ❌ **空目录占位**——子分类没文章就别建目录
5. ❌ **每篇都求完美**——发布后能改，不要在草稿里反复打磨而停发
6. ❌ **代码示例只有伪代码**——要么不放，要放就能 copy-paste 跑通

---

## 11. 何时回看本策略

- 每发 5 篇时回看一次：定位是否漂移、节奏是否跟得上
- 90 天回看：是否启动 `code/spring-ai/`、是否需要新增子分类
- 365 天回看：是否值得做付费小册或视频
