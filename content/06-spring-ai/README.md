---
title: 06 · Spring AI · Java 工程师视角的 AI 应用工程
description: 用 Spring AI 当锚点，从一个具体 bug 挖到模型架构。给 Java 后端工程师的 AI 工程实战
status: v1 · 主题初建 2026-05-11
---

# 06 · Spring AI

> **定位**：给 Java 后端工程师讲清楚 AI 应用开发——
> 用 Spring AI 当锚点，从一个具体 bug 挖到模型架构。

---

## 🎯 受众画像

- Java/Spring Boot 开发者，正考虑或已开始用 Spring AI
- 知道 ChatClient、@Tool、ChatMemory 这些 API 名字
- **但**：踩过 parallel_tool_calls 报 400、entity() 在 DeepSeek 上翻车、流式响应丢 toolCalls 这些坑
- **想知道**：这些坑的根因是 Spring AI 的 bug、模型的限制、还是自己用错了

---

## 💡 与其他 AI 教程的差异化

国内 Spring AI 教程的现状：
- 90% 是"Hello World 翻译"——抄官方文档+换个国产模型
- 几乎没有"踩坑实录"——bug 修了人就走了，不沉淀
- 几乎没有"国产模型适配差异"——都假装跟 OpenAI 完全一样

**本主题反过来做**：
- 📌 **每篇都从一个具体 bug 或一个工程决策点出发**
- 📌 **先讲怎么错了**，再讲怎么对的
- 📌 **国产模型差异是头等公民**——DeepSeek/Qwen/GLM/豆包逐个对照

---

## 🗺️ 子分类

| 目录 | 关注 | 代表内容 |
|---|---|---|
| `01-pitfalls/` | 踩坑实录（按时间倒序） | parallel_tool_calls 失败、流式 toolCalls 丢失 |
| `02-core-concepts/` | 核心抽象 | ChatClient / Advisor 链 / ChatMemory / VectorStore |
| `03-tool-calling/` | 工具调用机制 | @Tool / ToolCallback / 并行执行 / MCP 集成 |
| `04-structured-output/` | 结构化输出 | entity() 原理 / 国产兼容性 / strict 兜底 |
| `05-domestic-models/` | 国产模型适配 | DeepSeek/Qwen/GLM/豆包配置差异 / 已知坑 |
| `06-production/` | 生产化 | 重试 / 限流 / 可观测 / 成本 / Token 治理 |

---

## 🚀 推荐入口

### 新手第一篇
《我用 Spring AI 调 o3，被一个 400 报错教做人》（`01-pitfalls/`）—— 从一个 bug 看清 reasoning 模型的本质。

### 选型决策
《Spring AI 国产模型适配速查表》（`05-domestic-models/`）—— 一张表看清各家配置差异和已知坑。

### 工程化深入
《Spring AI Advisor 链 · 你的 LLM 中间件该怎么写》（`02-core-concepts/`）

---

## 📊 当前进度

| 路径 | 已发布 | 草稿 | 规划 |
|---|:-:|:-:|:-:|
| 01-pitfalls | 0 | 0 | 5 |
| 02-core-concepts | 0 | 0 | 4 |
| 03-tool-calling | 0 | 0 | 4 |
| 04-structured-output | 0 | 0 | 3 |
| 05-domestic-models | 0 | 0 | 5 |
| 06-production | 0 | 0 | 3 |

完整选题清单见 [PLAN.md](./PLAN.md)。

---

## 🛠 配套代码（计划）

```
code/spring-ai/                 (待开)
├── parallel-tool-calls/        complete demo + 各模型表现
├── structured-output-vendors/  国产模型 strict 兼容性测试
├── chat-memory-modes/          短期/长期 memory 三种模式
└── README.md
```

**原则**：先把 `content/` 写到 5+ 篇再开 `code/`，不要因代码工程拖累内容产出。

---

## ✍️ 写作风格（额外约定）

继承根 README 的"保留误解"哲学之外，本主题加 3 条：

1. **每篇标 Spring AI 版本号**：`spring-ai: 2.0.0-M6` 写在 frontmatter，避免半年后误导
2. **bug 类文章必须给最小复现**：5 行内能跑出 bug，不要贴整个项目
3. **国产模型对照必须实测**：不能说"应该可以"，要给 curl 输出或 Java 代码截图

---

## 🔗 与其他主题的边界

- **通用 LLM 概念**（推理模型为什么不支持 parallel）→ 写在 `05-llm-foundations/`，本主题只引用
- **MCP 协议本身** → 写在 `01-mcp/`，本主题只讲 Spring AI 怎么集成
- **Agent 设计模式** → 写在 `02-agent/`，本主题只讲 Spring AI 工程实现
- **RAG 概念与算法** → 写在 `03-rag/`，本主题只讲 Spring AI VectorStore 用法
