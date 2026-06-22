# Agent Research · AI Agent 生态深度拆解

> 用 [ATDF 方法论](methodology/ATDF.md) 系统拆解 AI Agent 生态的产品、架构、商业模式和投入价值。

## 为什么有这个目录

AI Agent 生态在 2025-2026 年爆发式增长——MCP / A2A / ACP 三大协议、Letta/MemGPT 记忆系统、gstack/OMC 编程框架……新东西每周冒出来。

这个目录不是新闻摘要，而是**结构化的深度拆解**：每个主题都按 ATDF 8 维度（定位 · 架构 · 产品 · 业务 · 使用 · 模块 · 生态位 · 迁移）拆到你能判断"值不值得投入时间"的程度。

---

## ATDF 方法论

| 文件 | 说明 |
|---|---|
| [方法论说明](methodology/ATDF.md) | 8 维度 + 3 档深度 + 术语速查 |
| [空白模板](templates/ATDF-template.md) | 复制即用 |

---

## Deep Dives · 深度拆解

| 主题 | 类型 | 一句话结论 | 链接 |
|---|---|---|---|
| **OMC** | Agent 编排框架 | 学架构思想（创作/审核分离 · 智能路由 · 可观测性三支柱），而非绑定具体工具 | [ATDF](deep-dives/omc/omc-atdf.md) |
| **gstack** | AI 编程方法论 | 角色约束 + 流程门控是 AI 编程的标准范式，学 SKILL.md 写法比用它更有价值 | [ATDF](deep-dives/gstack/gstack-atdf.md) |
| **MemGPT / Letta** | Agent 记忆系统 | RAG 的下一站，Memory 的起点 | [入门指南](deep-dives/memgpt-letta/memgpt-letta-guide.html) |

---

## Research · 趋势研究

| 主题 | 范围 | 链接 |
|---|---|---|
| **Agent 生态 2026** | 协议战争 · 被改造领域 · 创新空白 · 工程师机会 | [md](research/agent-ecosystem-2026.md) · [HTML](research/agent-ecosystem-2026.html) |

---

## Memory · 记忆专题

> Agent 记忆从概念到落地的独立主题，[专题总览](memory/README.md)。

| 主题 | 一句话 | 链接 |
|---|---|---|
| **从 RAG 到 Memory** | 传统 RAG 被降级为原语，Memory / Context Engineering 是新战场 | [rag-to-memory.md](memory/rag-to-memory.md) |
| **mem0 入门译文** | 记忆是什么 · 为什么 · 怎么做（记忆 vs 上下文窗口 vs RAG） | [mem0-memory-in-agents.md](memory/mem0-memory-in-agents.md) |
| **商用方案对比** | OpenAI / Claude Code / Claude 三家 8 维度 + 实现细节 | [survey](memory/ai-memory-implementation-survey.md) |
| **开源项目拆解** | Letta / mem0 / Graphiti / Cognee / Memobase / A-MEM | [teardown](memory/memory-oss-teardown.md) |
| **调研方法论** | 8 维度 + 可复用打分表 | [framework](memory/memory-module-research-framework.md) |
| **实战复盘** | 手搓短期+长期记忆层 · 上下文窗口管理 | [实战](memory/short-term-vs-long-term-memory-in-practice.md) · [上下文](memory/context-window-management.md) |

---

## Papers · 论文解读

> 关键论文逐句中文注解 + 可运行复现，[专题总览](papers/README.md)。

| 论文 | 一句话 | 链接 |
|---|---|---|
| **ReAct** | 定义现代 Agent 形态：Thought → Action → Observation 循环 | [react-paper.md](papers/react-paper.md) |

---

## Concepts · 核心概念

| 概念 | 一句话 | 链接 |
|---|---|---|
| **Karpathy 路线** | LLM OS → RAG is a hack → LLM Wiki → Software 3.0 → Agent Memory | [karpathy-route.md](concepts/karpathy-route.md) |

---

## 目录结构

```
02-agent/
├── README.md                ← 本文件
├── methodology/             ← 拆解方法论（ATDF / 5D）
│   ├── ATDF.md
│   └── 5d-framework.md
├── templates/
│   └── ATDF-template.md
├── memory/                  ← ⭐ 记忆专题（独立话题）
│   ├── README.md
│   ├── rag-to-memory.md
│   ├── mem0-memory-in-agents.md
│   ├── ai-memory-implementation-survey.md
│   ├── memory-oss-teardown.md
│   ├── memory-module-research-framework.md
│   ├── short-term-vs-long-term-memory-in-practice.md  ← 实战(原 articles)
│   └── context-window-management.md                   ← 实战(原 articles)
├── papers/                  ← ⭐ 论文解读（独立话题）
│   ├── README.md
│   └── react-paper.md
├── research/
│   └── agent-ecosystem-2026.md
├── deep-dives/
│   ├── omc/ · gstack/
│   └── memgpt-letta/memgpt-letta-guide.md
├── planning-reasoning/
├── harness/
└── concepts/
    └── karpathy-route.md
```

## 如何贡献

1. Fork → 用 [ATDF 模板](templates/ATDF-template.md) 拆解一个你感兴趣的 AI 主题 → PR
2. 对现有拆解有不同判断？开 Issue 讨论
3. 方法论本身的改进建议也欢迎
