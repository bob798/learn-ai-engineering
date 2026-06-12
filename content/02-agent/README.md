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
| **Agent Harness 深度拆解** | 8 问题域框架逆向 Aider / OpenHands / LangGraph / SWE-agent / Claude Code · 从 0 到 1 建造路线图 | [md](research/agent-harness-teardown.md) |

---

## Concepts · 核心概念

| 概念 | 一句话 | 链接 |
|---|---|---|
| **从 RAG 到 Memory** | 传统 RAG 被降级为原语，Memory / Context Engineering 是新战场 | [rag-to-memory.md](concepts/rag-to-memory.md) |
| **Karpathy 路线** | LLM OS → RAG is a hack → LLM Wiki → Software 3.0 → Agent Memory | [karpathy-route.md](concepts/karpathy-route.md) |

---

## 目录结构

```
agent-research/
├── README.md                ← 本文件
├── methodology/
│   └── ATDF.md              ← 拆解方法论
├── templates/
│   └── ATDF-template.md     ← 空白模板
├── research/
│   ├── agent-ecosystem-2026.md
│   └── agent-ecosystem-2026.html
├── deep-dives/
│   ├── omc/
│   │   └── omc-atdf.md
│   ├── gstack/
│   │   └── gstack-atdf.md
│   └── memgpt-letta/
│       └── memgpt-letta-guide.html
└── concepts/
    ├── rag-to-memory.md
    └── karpathy-route.md
```

## 如何贡献

1. Fork → 用 [ATDF 模板](templates/ATDF-template.md) 拆解一个你感兴趣的 AI 主题 → PR
2. 对现有拆解有不同判断？开 Issue 讨论
3. 方法论本身的改进建议也欢迎
