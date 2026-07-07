# Agent · AI Agent 知识体系

> 从"为什么需要 Agent"到"怎么造一个"，再到记忆、规划、运行时可靠性的深度拆解。

## 阅读路径

```
① 理解 Agent       concepts/             ← 从这里开始
       ↓
② 自己造一个       agent-from-scratch/    ← 动手
       ↓
③ 深入子方向       deep-topics/           ← 记忆 / 上下文 / 规划 / 论文
       ↓
④ 运行时可靠性     harness/               ← 真实事故拆解
       ↓
⑤ 拆解业界方案     deep-dives/            ← OMC / gstack / Letta
       ↓
⑥ 行业趋势        research/              ← 生态全景

📝 对外发布系列     irreversible-agent/    ← 《不可逆的Agent》博客连载
```

---

## ① Concepts · 核心概念

> 第一站：搞清楚 Agent 是什么、为什么、怎么工作。

| 主题 | 一句话 | 链接 |
|---|---|---|
| **为什么 AI 必然走向 Agent** | 从第一性原理倒推：LLM 只覆盖知识+推理，剩下四种能力补上去就是 Agent | [01-why-agent-is-inevitable.md](concepts/01-why-agent-is-inevitable.md) |
| **为什么 Agent 必须是循环** | 5 个底层原因 + 60 年血脉 + messages 数组机制 | [02-agent-loop.md](concepts/02-agent-loop.md) |

---

## ② Agent from Scratch · 从零手写

> 第二站：对标 RAG V1-V10，从最小循环打穿到生产级。[路线图](agent-from-scratch/README.md)

| 版本 | 主题 | 状态 |
|---|---|---|
| V1 | 最小 Agent 循环 | ✅ 已完成 |
| V2-V10 | ReAct → 多工具 → 记忆 → 规划 → 评估 → 多Agent → 生产化 | 施工中 |

---

## ③ Deep Topics · 深入子方向

### 记忆专题

> Agent 记忆从概念到落地，[专题总览](deep-topics/memory/README.md)。

| 主题 | 一句话 | 链接 |
|---|---|---|
| **Karpathy 的 Memory 思想脉络** | LLM OS → RAG is a hack → LLM Wiki → Context Engineering → Agent Memory | [karpathy-memory-route.md](deep-topics/memory/karpathy-memory-route.md) |
| **从 RAG 到 Memory** | 传统 RAG 被降级为原语，Memory / Context Engineering 是新战场 | [rag-to-memory.md](deep-topics/memory/rag-to-memory.md) |
| **mem0 入门译文** | 记忆是什么 · 为什么 · 怎么做（记忆 vs 上下文窗口 vs RAG） | [mem0-memory-in-agents.md](deep-topics/memory/mem0-memory-in-agents.md) |
| **商用方案对比** | OpenAI / Claude Code / Claude 三家 8 维度 + 实现细节 | [survey](deep-topics/memory/ai-memory-implementation-survey.md) |
| **开源项目拆解** | Letta / mem0 / Graphiti / Cognee / Memobase / A-MEM | [teardown](deep-topics/memory/memory-oss-teardown.md) |
| **调研方法论** | 8 维度 + 可复用打分表 | [framework](deep-topics/memory/memory-module-research-framework.md) |
| **实战复盘** | 手搓短期+长期记忆层（pin 系统提示 / 行级隔离 / 写穿透） | [实战](deep-topics/memory/short-term-vs-long-term-memory-in-practice.md) |

### 上下文工程

> 把无界历史投影进有界窗口，[专题总览](deep-topics/context-engineering/README.md)。

| 主题 | 一句话 | 链接 |
|---|---|---|
| **窗口管理实战** | 从一个「每轮重复摘要」的 bug 说起 · 滚动摘要 / context rot / 取舍 | [上下文窗口管理](deep-topics/context-engineering/context-window-management.md) |
| **决策框架** | 全量回放 → 三难权衡（保真/成本/稳定）→ 摘要 vs prefix caching → 六大解法家族 | [三难与六大解法](deep-topics/context-engineering/context-compaction-tradeoffs.md) |

### 规划与推理

> [专题总览](deep-topics/planning-reasoning/README.md)

### 论文解读

> 关键论文逐句中文注解 + 可运行复现，[专题总览](deep-topics/papers/README.md)。

| 论文 | 一句话 | 链接 |
|---|---|---|
| **ReAct** | 定义现代 Agent 形态：Thought → Action → Observation 循环 | [react-paper.md](deep-topics/papers/react-paper.md) |

---

## ④ Harness · 运行时可靠性

> Agent loop 跑起来会怎么坏——真实事故的第一性原理拆解。

| 主题 | 一句话 | 链接 |
|---|---|---|
| **沙箱生命周期可靠性** | 从「文件工具全部失败」事故挖起：残留躲不掉，可靠等于残留能被收拾 | [agent-sandbox-lifecycle-reliability.md](harness/agent-sandbox-lifecycle-reliability.md) |
| **Agent 退化循环（doom loop）** | 一个「背景」空转 5 步：缺的不是更高的步数上限，是无进展检测和弃权式终止 | [agent-doom-loop.md](harness/agent-doom-loop.md) |

---

## ⑤ Deep Dives · 业界方案拆解

> 用 [ATDF 方法论](../_framework/methodology/ATDF.md) 拆解 AI Agent 生态的产品与架构。

| 主题 | 类型 | 一句话结论 | 链接 |
|---|---|---|---|
| **OMC** | Agent 编排框架 | 学架构思想（创作/审核分离 · 智能路由 · 可观测性三支柱），而非绑定具体工具 | [ATDF](deep-dives/omc/omc-atdf.md) |
| **gstack** | AI 编程方法论 | 角色约束 + 流程门控是 AI 编程的标准范式，学 SKILL.md 写法比用它更有价值 | [ATDF](deep-dives/gstack/gstack-atdf.md) |
| **MemGPT / Letta** | Agent 记忆系统 | RAG 的下一站，Memory 的起点 | [入门指南](deep-dives/memgpt-letta/memgpt-letta-guide.md) |

---

## ⑥ Research · 行业趋势

| 主题 | 范围 | 链接 |
|---|---|---|
| **Agent 生态 2026** | 协议战争 · 被改造领域 · 创新空白 · 工程师机会 | [md](research/agent-ecosystem-2026.md) |
| **Agent Harness 深度拆解** | 8 问题域框架逆向 Aider / OpenHands / LangGraph / SWE-agent / Claude Code · 从 0 到 1 建造路线图 | [md](research/agent-harness-teardown.md) |

---

## 📝 《不可逆的Agent》博客系列

> 对外发布的连载系列：软件 Agent 六模块 × 物理世界约束 = 具身化的 Agent 工程。核心命题：**物理世界没有 Ctrl+Z**。

| # | 标题 | 阶段 | 状态 |
|---|---|---|---|
| 1 | **一个 Agent 最小需要什么** | 底座篇 | 📝 draft |
| 2 | Agent 怎么"想"——ReAct 到 plan-and-execute | 底座篇 | |
| 3 | Agent 怎么"记"——短期上下文、长期记忆、RAG 的边界 | 底座篇 | |
| 4 | Agent 怎么"碰"世界——function calling 到 MCP | 底座篇 | |
| 5 | 什么时候需要多个 Agent | 底座篇 | |
| 6 | 你怎么知道 Agent 做对了——评估与可观测性 | 底座篇 | |
| 7 | 物理世界没有 Ctrl+Z——不可逆性重塑 Agent 设计 | 崩塌篇 | |
| 8 | 写路径必须进沙箱——动作分级 | 崩塌篇 | |
| 9 | Agent 必须知道世界现在什么样——设备状态模型 | 崩塌篇 | |
| 10 | 不能 replay 的世界怎么测试——仿真先行 | 崩塌篇 | |
| 11 | 灯光归灯光，暖通归暖通——分域 Agent 与冲突仲裁 | 崩塌篇 | |
| 12 | 把 Modbus 翻译给 LLM 听——老协议抽象成 Agent 工具 | 崩塌篇 | |
| 13 | 用 Home Assistant 搭一个带安全包络的灯光 Agent | 实战篇 | |
| 14 | 边缘设备上跑 Agent——GMKtec 上的算力约束 | 实战篇 | |

> 发布建议：先写第 7 篇当开篇宣言（系列的"为什么"），再回头按 1→14 推进。第 7、13 篇最有出圈潜力。
>
> 与内部知识体系的关系：博客系列是**对外叙事线**，自由引用 concepts / agent-from-scratch / deep-topics 的内容，但保持独立阅读体验。

[→ 系列目录](irreversible-agent/)

---

## 目录结构

```
02-agent/
├── README.md                    ← 本文件（总索引 + 阅读路径）
│
├── concepts/                    ← ① 核心概念（入口，不过期的认知基石）
│   ├── 01-why-agent-is-inevitable.md
│   └── 02-agent-loop.md
│
├── agent-from-scratch/          ← ② 从零手写 V1→V10
│   ├── README.md
│   └── 02_代码讲解_V1V2.md
│
├── deep-topics/                 ← ③ 深入子方向
│   ├── memory/                      记忆专题（7 篇）
│   ├── context-engineering/         上下文工程（3 篇）
│   ├── planning-reasoning/          规划与推理
│   └── papers/                      论文解读
│
├── harness/                     ← ④ 运行时可靠性（真实事故）
│   ├── agent-sandbox-lifecycle-reliability.md
│   └── agent-doom-loop.md
│
├── deep-dives/                  ← ⑤ 业界方案拆解
│   ├── omc/
│   ├── gstack/
│   └── memgpt-letta/
│
├── irreversible-agent/           ← 📝《不可逆的Agent》博客连载
│   └── 01-最小agent.md
│
└── research/                    ← ⑥ 行业趋势
    └── agent-ecosystem-2026.md
```

## 如何贡献

1. Fork → 用 [ATDF 模板](../_framework/templates/ATDF-template.md) 拆解一个你感兴趣的 AI 主题 → PR
2. 对现有拆解有不同判断？开 Issue 讨论
