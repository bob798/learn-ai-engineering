---
title: "Learning Path · 阅读路线图"
description: 55 篇文档的推荐阅读顺序 — 从 0 到 1 构建 AI 应用的学习路径
order: 2
---

# Learning Path · 阅读路线图

> 这份路线图把本站 55+ 篇文档串联成一条**有顺序的学习路径**。
> 你不需要全部读完 —— 根据你的阶段，选一条线走下去。

---

## 快速分流：你现在在哪

| 你的情况 | 从哪开始 |
|---------|---------|
| 完全没接触过 AI 应用 | 从 [第零章](#第零章--出发前) 开始 |
| 知道 RAG 是什么，但没写过代码 | 从 [第一章](#第一章--rag) 开始 |
| 写过 RAG，想理解 Agent | 从 [第二章](#第二章--agent) 开始 |
| 用过 Agent 工具，想理解 MCP | 从 [第三章](#第三章--mcp) 开始 |
| 都了解，想看实战和工程化 | 直接跳 [第四章](#第四章--ai-编程实战) |

---

## 第零章 · 出发前

建立基础认知框架，理解"AI 工程师"和"ML 研究者"的区别。

| # | 文档 | 一句话 | 时间 |
|---|------|-------|------|
| 0.1 | [5D 学习方法论](/agent/methodology/5d-framework) | 如何快速理解任何新技术领域 | 15 min |
| 0.2 | [ATDF 拆解框架](/agent/methodology/ATDF) | 如何系统评估一个 AI 项目/工具 | 10 min |

**这一章的目标**：你手里有了两个可复用的框架，后面每学一个新主题都能用。

---

## 第一章 · RAG — 让 AI 读懂你的数据

RAG 是最容易上手的 AI 应用模式：把你的文档变成 AI 能查询的知识库。

| # | 文档 | 一句话 | 时间 |
|---|------|-------|------|
| 1.1 | [理解 RAG · 5 集入门](/rag/01-理解RAG) | RAG 是什么、解决什么、不解决什么 | 30 min |
| 1.2 | [概念手册 · 向量与检索](/rag/02-概念手册-向量与检索) | Embedding、向量数据库、相似度检索 | 25 min |
| 1.3 | [代码��解 · V1 & V2](/rag/03-代码讲解-V1V2) | 亲手跑通一个 RAG pipeline | 40 min |
| 1.4 | [工程方法论手册](/rag/04-工程方法论手册) | 从原型到生产的工程考量 | 20 min |
| 1.5 | [RAG 知识地图](/rag/rag-knowledge-map) | 回顾 + 查漏补缺 | 10 min |

**选修 · 深入**：
- [混合检索 RRF 平局陷阱](/rag/mock-interview/05_混合检索RRF平局陷阱专题分析) — 工程细节
- [Embedding 选型参考](/rag/mock-interview/06_embedding选型参考与合成Query) — 模型选型
- [模拟面试题库](/rag/mock-interview/01_基础概念题) — 检验理解

**这一章的目标**：你能从零搭建一个 RAG 系统，知道每一层的 trade-off。

---

## 第二章 · Agent — 让 AI 自己行动

Agent 是 RAG 的"升级"：不只查信息，还能执行动作、做多步决策。

| # | 文档 | 一句话 | 时间 |
|---|------|-------|------|
| 2.1 | [ReAct 论文解读](/agent/papers/react-paper) | 现代 Agent 的起源论文，逐节拆解 | 30 min |
| 2.2 | [ReAct 注解版](viz/agent/react-paper-annotated.html) | 论文原文逐句中英注解 + 词汇追踪 | 交互式 |
| 2.3 | [Agent Loop 深度理解](/agent/harness/agent-loop) | 为什么是循环？60 年血脉 + messages 数组机制 | 40 min |
| 2.4 | [Planning & Reasoning](/agent/planning-reasoning) | ReAct / Plan-and-Execute / Multi-Agent 对比 | 30 min |

**选修 · 生态**：
- [Agent 生态 2026](/agent/research/agent-ecosystem-2026) — 行业全景
- [从 RAG 到 Agent Memory](/agent/memory/rag-to-memory) — RAG 和 Agent 的桥梁
- [Karpathy 路线](/agent/concepts/karpathy-route) — LLM OS → Software 3.0 的思想路线

**动手 · 代码**：
- [ReAct 可运行代码](https://github.com/bob798/learn-ai-engineering/tree/main/code/react-hands-on) — `python run_react.py --idx 0` 跑起来

**这一章的目标**：你理解 Agent = Loop + Tools + LLM，能读懂任何 agent 框架的源码骨架。

---

## 第三章 · MCP — 标准化的工具接口

Agent 需要调用工具，MCP 定义了工具的"USB-C 接口"。

| # | 文档 | 一句话 | 时间 |
|---|------|-------|------|
| 3.1 | [MCP 基础](/mcp/01-foundations) | MCP 是什么、解决什么、为什么重要 | 20 min |
| 3.2 | [Function Calling](/mcp/02-core-concepts/function-calling) | MCP 的底层机制 | 20 min |
| 3.3 | [Tools / Resources / Prompts](/mcp/02-core-concepts/tools-resources-prompts) | MCP 三类能力详解 | 25 min |
| 3.4 | [我理解错的 10 件事](/mcp/05-interview/common-misconceptions) | 真实踩坑 | 15 min |

**选修 · 深入**：
- [Adapter 与 Gateway 设计](/mcp/03-practical/adapter-gateway) — 生产架构
- [MCP 深挖 11 问](/mcp/05-interview/mcp-11q) — 追问到底层
- [MCP 面试题库](/mcp/05-interview/qa) — 检验理解

**这一章的目标**：你理解 MCP 在 Agent 生态中的位置，能给自己的工具写 MCP adapter。

---

## 第四章 · AI 编程实战

用 AI 工具写代码 — 不是"让 AI 帮你写"，而是理解 AI 编程的范式。

| # | 文档 | 一句话 | 时间 |
|---|------|-------|------|
| 4.1 | [AI 编程工具篇](/ai-programming/tools) | Claude Code / Cursor / Aider 等工具对比 | 15 min |
| 4.2 | [OMC 深度拆解](/ai-programming/omc-overview) | 多 agent 编排框架的架构思想 | 30 min |
| 4.3 | [OMC 实战指南](/ai-programming/omc-review-and-discussion-guide) | 从场景出发选对工作流 | 20 min |
| 4.4 | [AI 修炼册](/ai-programming/xiulian-ce) | 长期成长路线 | 15 min |

**这一章的目标**：你从"用 AI 工具"变成"理解 AI 工具的设计"，能判断什么场景该用什么。

---

## 路线图总览

```
第零章 · 出发前          ← 方法论（2 篇，25 min）
    │
    ▼
第一章 · RAG            ← 最容易上手（5 篇核心 + 选修）
    │
    ▼
第二章 · Agent          ← 理解底层原理（4 篇核心 + 选修 + 代码）
    │
    ▼
第三章 · MCP            ← 标准化接口（4 篇核心 + 选修）
    │
    ▼
第四章 · AI 编程实战     ← 工程化落地（4 篇核心）
```

**总计核心路径**：19 篇，约 6-8 小时。
**含选修全量**：55 篇 + 28 个交互笔记。

---

## 还缺什么（坦白说）

| 缺口 | 说明 | 状态 |
|------|------|------|
| **Prompt Engineering 基础** | CoT / Few-shot / System Prompt 的系统讲解 | 待写 |
| **API 调用入门** | 从 HTTP 到 tool_use，对完全不懂 API 的人 | 待写 |
| **端到端项目** | 一个完整的 RAG + Agent + MCP 综合项目 | 待写 |
| **部署与生产** | 成本、延迟、监控、安全 | 待写 |

> 这些"待写"不是承诺，是方向。如果你觉得某个缺口对你特别重要，[开个 Issue](https://github.com/bob798/learn-ai-engineering/issues) 告诉我。
