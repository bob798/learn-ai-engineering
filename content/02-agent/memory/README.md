---
title: "Agent Memory · 记忆专题"
description: Agent 记忆从概念到落地的独立主题——为什么记忆是下一个战场、商用与开源方案怎么拆、调研方法论与动手 Demo
order: 1
---

# Agent Memory · 记忆专题

> 当模型和工具趋同，**记忆**会成为 Agent 最明显的差异化能力：会记住、会学习、会和你一起成长的 Agent 才会胜出。
> 这个专题把散落在概念、研究、方法论里的记忆内容收拢成一条线：**从「为什么需要记忆」→「业界怎么做」→「怎么调研判断」→「怎么动手」**。

---

## 怎么读这个专题

| 阶段 | 读什么 | 一句话 |
|---|---|---|
| **① 为什么** | [从 RAG 到 Agent Memory](rag-to-memory.md) | RAG 被降级为原语，Memory / Context Engineering 是新战场 |
| **① 入门** | [mem0：记忆是什么·为什么·怎么做](mem0-memory-in-agents.md) | mem0 创始人入门文 · 记忆 vs 上下文窗口 vs RAG · 短期/长期记忆分类 |
| **② 业界怎么做（商用）** | [AI 记忆模块调研报告](ai-memory-implementation-survey.md) | OpenAI / Claude Code / Claude 三家方案 8 维度对比 + 实现细节 |
| **② 业界怎么做（开源）** | [开源 AI 记忆项目深度拆解](memory-oss-teardown.md) | Letta / mem0 / Graphiti / Cognee / Memobase / A-MEM 六项目逐维拆解 |
| **③ 怎么调研判断** | [记忆模块调研方法论](memory-module-research-framework.md) | 调研任意记忆模块该看的 8 维度 + 可复用打分表 |
| **④ 实战复盘** | [短期+长期记忆实战](short-term-vs-long-term-memory-in-practice.md) | 读完 mem0 后手搓一个短期+长期记忆层的复盘（pin 系统提示 / 行级隔离 / 写穿透） |
| **④ 实战复盘** | [上下文窗口管理](context-window-management.md) | 从一个「每轮重复摘要」的 bug 说起 · 滚动摘要 / context rot / 取舍 |
| **⑤ 怎么动手** | `code/memory/`（仓库内代码） | 把记忆代码「串进 agent 循环」的 Demo |

---

## 相关内容（不在本目录但强关联）

- [MemGPT / Letta 入门指南](../deep-dives/memgpt-letta/memgpt-letta-guide.md) — 用秘书比喻理解三级记忆（core / recall / archival）
- [Karpathy 路线](../concepts/karpathy-route.md) — LLM OS → LLM Wiki → Agent Memory 的思想脉络
- [Agent 生态 2026](../research/agent-ecosystem-2026.md) — 记忆作为「创新空白」的行业背景

---

## 一句话结论

记忆不是「把更多东西塞进上下文」，而是构建一个**会演化、会取舍、跨会话持久**的内部状态。判断一个记忆方案好不好，本质是回答三件事：**管什么记忆（架构）→ 工程上怎么用（成本 / 可观测）→ 到底比「全塞进上下文」强多少（评估）**。
