---
title: Agent Loop 深度理解 — shareAI s01 的理论深度补充
description: 你手写过 60 行 agent loop 之后，这里告诉你为什么它必须长这样
---

# Agent Loop 深度理解

## 读前声明

<div data-viz="IntroHook"></div>

> 📎 **本页定位**：[shareAI-lab/learn-claude-code · s01 The Agent Loop](https://github.com/shareAI-lab/learn-claude-code/blob/main/docs/zh/s01-the-agent-loop.md) 的**理论深度补充**。
>
> s01 那里教你**怎么写**（一个 loop + 一个 bash = 一个 agent，60 行代码）。本页告诉你：
> - 为什么**必须**是循环而不是一次调用（5 个底层原因）
> - 这个循环结构是被**谁**发现的（60 年血脉：Shakey → SPA → SOAR → RL → ReAct）
> - messages 数组为什么是 loop 的**唯一**状态载体
> - 如何把 loop 结构**迁移**到任何非 AI 系统（自造类比 4 问法）
>
> **未完成 s01 的请先去那里动手。** 没亲手写过 while 循环的读者，本页的每一句话都会像悬空的道理。

---

# 第一幕 · why · loop 的理论根源

## 📖 为什么必须是 loop

LLM 本身是无状态的一次性函数，而真实任务是"边做边看"的过程。loop 是粘合这两者的唯一方式 —— 五个底层原因：

- **① 信息不全 · 开工时模型什么都不知道**：文件内容、命令输出、测试结果都要"跑一下才知道"。必须 act → observe → 再想。
- **② 条件分支树 · 路径在运行时才展开**："修 bug"展开是：读 → 发现 X → 改 → 跑测试 → 挂了 → 再读报错 → 再改…… 分支数量事先不知道。
- **③ 上下文有限 · 不可能一次塞完整个代码库**：Loop 按需拉信息，用完就丢。这是上下文经济学的唯一解。
- **④ 工具调用即 yield · 控制权必须交还 harness**：模型说"我要 Read 文件"，harness 执行、拿结果、再喂回。yield/resume 就是 loop 的一个迭代。
- **⑤ 错误需要反馈 · 闭着眼写代码 = 必然出错**：编译报错、测试红、命令失败 —— 没有 loop 就没有反馈回路。模型必须看见信号才能修正。

> 五个原因**任何一个**单独都不足以推出 loop 是唯一解，但**五个合起来**让其他方案（一次调用 / 预编排流程 / 树搜索 / ...）都失效。这是 loop 作为"发动机"的必然性。

## 📖 loop 是被谁"发现"的

没有某一个人发明。这是一条多源汇流的演化线 —— 从机器人到认知科学，最后在 2022 年的 ReAct 论文定型。

| 时间 | 里程碑 | 说明 |
|------|--------|------|
| 1960s — 1990s | **远古思想源** · Sense → Plan → Act 的几十年铺垫 | 从 Shakey 机器人到强化学习，"用循环解决序列决策"在 AI 界早已是常识。下一节详讲。 |
| 2021.12 | **WebGPT** (OpenAI) | 最早让 LLM 在浏览器里循环执行动作，拉开 LLM agent 的序幕。 |
| 2022.05 | **MRKL Systems** (AI21) | 提出 LLM + 外部模块路由的范式，Karpas 等人给出概念框架。 |
| 2022.10 ⭐ | **ReAct** — Shunyu Yao 等 (Princeton + Google) | 第一次清晰提出 Thought → Action → Observation 交替循环。这被公认为现代 agent loop 的教科书形态。 |
| 2022.10 — 2023.03 | **工程化爆发** | LangChain（Harrison Chase）把 ReAct 变成人人可调；AutoGPT（Toran Richards）让大众第一次看到全自动 loop；BabyAGI（Yohei Nakajima）把 loop 拆成任务队列。 |
| 2024.12 — 至今 | **Harness 沉淀** | Anthropic《Building Effective Agents》区分 workflow vs agent；Claude Code、Cursor、aider 把 loop 做成基础设施。loop 从论文走进生产。 |

> 你在 shareAI s01 手写的那 60 行代码，是这条演化线的**最新化身**。下一节展开 1960—2000 年代的 6 条源流——ReAct 只是把它们翻译成了 LLM 的语言。

## 📖 远古思想源：loop 的六条血脉

**这一节是本页的灵魂。如果你只读一节，就读它。**

因为它示范的是"**造类比**"的完整过程：同一个结构在 6 个不同领域被 6 批不同的人独立发现。看完这 6 条血脉，你关掉浏览器再遇到陌生系统，就会条件反射地问"这里有没有感知→决策→行动的循环"。

今天的 agent loop 不是 2022 年凭空发明的。从 1960 年代的机器人到 1990 年代的强化学习，"感知→规划→行动"这个结构被反复验证过六次 —— ReAct 只是把它搬到了 LLM 上。理解这六条源流，才能看清 loop 的本质不是"循环调用"，而是**带反馈的决策过程**。

### 1966 — 1972 · Shakey 机器人

**世界第一台"思考型"移动机器人** — Nils Nilsson, Charles Rosen 等（Stanford Research Institute）

拿着摄像头在走廊里推箱子，第一次把"感知-建模-规划-执行"做成可运行系统。它用的 **STRIPS 规划器**（前置条件 + 动作 + 后置效果）至今仍是所有 AI 规划的语法基础。

Shakey 一次完整任务内部就是一个大 loop：拍照 → 更新世界模型 → 求解 STRIPS → 执行动作 → 再拍照。

> `perceive → model → plan → act → perceive ...`

### 1970s · Sense-Plan-Act (SPA) 三段论

**经典 AI 范式** — 经典人工智能学派的通用骨架

把智能体拆成三个独立阶段：**Sense**（传感器读入）→ **Plan**（符号推理生成动作序列）→ **Act**（执行器输出）。每一轮结束后，新感知反哺模型，下一轮重新规划。

这是现代 agent loop 的直系祖先。ReAct 的 Thought/Action/Observation 本质上就是 Plan/Act/Sense 换了个马甲。

> `sense → plan → act → (loop)`

### 1986 · Subsumption Architecture

**反 SPA 的叛逆** — Rodney Brooks (iRobot 创始人, MIT CSAIL)

Brooks 的著名论文 *"Intelligence Without Reason"* 抨击 SPA 太慢太脆 —— 机器人还没想完，老鼠已经跑了。他提出**反应式分层架构**：多个感知-动作 loop 并行运行，底层反射快，高层规划慢，高层压制低层。

这对今天多 agent、subagent、并行 loop 的架构思想影响深远。

> `多个 sense→act loop 并行 · 分层压制`

### 1983 — 至今 · SOAR

**统一认知架构** — Allen Newell, John Laird, Paul Rosenbloom（CMU）

图灵奖得主 Newell 的毕生心血：把所有认知活动统一为一个 **decide cycle**（决策周期）。每个 cycle 包含：阐述状态 → 提议算子 → 评估 → 选择 → 应用。目标栈遇到僵局就自动生成子目标 —— **这就是 subagent 的原型**。

> `elaborate → propose → decide → apply → (impasse → subgoal)`

### 1993 — 至今 · ACT-R

**产生式规则的循环触发** — John R. Anderson (CMU)

认知心理学建模的标杆。核心是**产生式规则**（IF pattern THEN action）不断匹配工作记忆并触发。每一次触发就是一轮 loop，触发结果更新工作记忆，引发下一轮匹配。

ACT-R 明确区分**程序性记忆**（规则）与**陈述性记忆**（事实）—— 这正是今天 agent 的"工具" vs "上下文"。

> `match → select → fire → update memory → (loop)`

### 1998 · Agent-Environment Loop

**loop 的数学定义** — Richard Sutton & Andrew Barto（强化学习奠基作）

《Reinforcement Learning: An Introduction》把 agent 抽象为数学对象：每个 timestep t，agent 观察状态 `sₜ`，选择动作 `aₜ`，环境反馈奖励 `rₜ₊₁` 和新状态 `sₜ₊₁`。目标是最大化累积奖励。

这是今天所有 agent loop 的**数学骨架**。当你把 LLM 的 "Thought/Action/Observation" 映射到 "policy/action/reward"，它们其实是同一个东西。

> `sₜ → π(aₜ|sₜ) → env → rₜ₊₁, sₜ₊₁ → (loop)`

> Yao 等人 2022 写 ReAct 的时候，并不是"发明"了一个新的循环。他们只是把一个已经跑了 60 年的结构，第一次用自然语言 prompt 表达出来，让 LLM 也能加入这个古老的游戏。

---

# 第二幕 · how · loop 本身的机制

## 🔧 控制流 · messages 数组的机制

你在 shareAI s01 写的 `messages[]`，不只是"对话历史"——它是 **loop 的全部状态**。每轮 loop 把 `assistant turn + tool_result` 追加进 messages，整个喂回模型。模型再输出下一个 assistant turn，再追加。**loop 之外没有任何变量在累积状态。**

<div data-viz="MessageConveyor"></div>

这是 agent loop 最反直觉的一点：状态不在程序变量里，不在数据库里，而在 **messages 数组这个数据结构**里，每一轮都整个喂回。

> 模型只负责"**想**下一步"（生成 assistant turn），不负责"**把**下一步跑起来"。harness 做调度、LLM 做决策。这正是 shareAI s01 "Harness 层：循环 —— 模型与真实世界的第一道连接" 的真正含义。

## 🔧 loop 本身会在哪里死

前一节讲了 loop **应该怎么跑**；这一节讲它**怎么跑崩**——**只谈 loop 本身的病**，不谈工具/注入/权限（那些是 harness 其它层的病，属于未来 s02/s05 的深度补充范畴）。

| 症状 | 本质 | 对策 |
|------|------|------|
| 🔴 **死循环** · 反复调同一个工具 | 模型拿到同样结果，却不知道换思路 | 循环检测 / 相似度判重 / 强制跳出 |
| 🟠 **上下文崩塌** · 历史太长忘目标 | 30 轮后模型忘了最初要干什么 | compaction 或重启子 agent（见 s06） |
| 🟠 **过早收敛** · 以为做完其实没做 | 测试没跑、需求没满足就宣告完成 | 独立 verifier 复核 |
| 🔴 **状态漂移** · 中间某轮的 tool_result 错了 | 错误信息喂回模型 → 后续所有决策基于错前提 | 工具返回必须带 schema 校验 + 错误码 |

必须亲手观察过这些病症，才算真正懂 loop。建议你回到 shareAI s01 的代码里故意**制造**每一种（比如把 observation 改成假数据，或强制模型重复调工具），看它在第几轮垮掉。

---

# 元层 · 从 loop 到一切循环

## 如何识别任何系统里的 loop

本节是整页的**目的**。前面所有章节都在训练一件事：给你一个从没见过的系统（HTTP server、游戏 AI、做菜流程），你能不能**主动**拆出它的 loop 结构。这是迁移，不是复述。

### 造类比的 4 步法

识别任何系统是否是 agent loop，问自己 4 个问题：

1. **反馈（feedback）**：什么**事件**驱动下一步？（时钟滴答？用户输入？外部响应？上一步的结果？）
2. **分支树（branching）**：路径是不是**运行时才展开**？（固定流程 ≠ loop · 动态分支 = loop）
3. **状态累积（state accumulation）**：每轮是否有**单调增长**的状态？（累计历史 = 典型 loop · 只读当前帧 = 不是）
4. **调度者（scheduler）**：**谁在想 · 谁在做**？（思考主体和执行主体分离 = 经典 agent-harness 划分）

**4 项都有 → 这是一个 agent loop。** 少一项，就是某种"退化版 loop"（例如只有反馈没有分支树的系统是事件驱动程序，不是 agent）。

### 用下面的工具亲自造一个

选一个你熟悉的系统，填入 4 格——组件会提供参考答案，填完再展开对照。**注意：你的答案没有"正确"，只有"更深还是更浅"。**

<div data-viz="AnalogyBuilder"></div>

> 能做完上面的填空，你就完成了本文的唯一验收：**把 agent loop 从 AI 域迁移出去**。接下来再遇到"XX 能不能做成 agent"的问题，你手里就有了判据，而不是靠直觉。

---

# 出口

## 导航图 · shareAI s01 之后去哪

本页只讲了 s01 的深度补充——loop 结构本身 + 其数学/哲学根源。**真实 agent harness 远不止 loop**。下面是 s01-s12 的完整拔高路径，以及每节对应的"深度补充"规划：

| shareAI 讲 | 核心主题 | 深度补充状态 |
|---|---|---|
| [s01 Agent 循环](https://github.com/shareAI-lab/learn-claude-code/blob/main/docs/zh/s01-the-agent-loop.md) | loop 主循环 + messages[] | ✅ **本页** |
| [s02 Tool Use](https://github.com/shareAI-lab/learn-claude-code/blob/main/docs/zh/s02-tool-use.md) | 工具分发 + 粒度设计 | 🔜 计划中 |
| [s03 Todo Write](https://github.com/shareAI-lab/learn-claude-code/blob/main/docs/zh/s03-todo-write.md) | 任务记忆 | 🔜 计划中 |
| [s04 Subagents](https://github.com/shareAI-lab/learn-claude-code/blob/main/docs/zh/s04-subagent.md) | 嵌套 loop · context 隔离 | 🔜 计划中 |
| [s05 Skill Loading](https://github.com/shareAI-lab/learn-claude-code/blob/main/docs/zh/s05-skill-loading.md) | 能力动态加载 | 🔜 计划中 |
| [s06 Context Compact](https://github.com/shareAI-lab/learn-claude-code/blob/main/docs/zh/s06-context-compact.md) | 上下文压缩 · 对抗 rot | 🔜 计划中 |
| [s07 Task System](https://github.com/shareAI-lab/learn-claude-code/blob/main/docs/zh/s07-task-system.md) | 任务编排 | 🔜 计划中 |
| [s08 Background Tasks](https://github.com/shareAI-lab/learn-claude-code/blob/main/docs/zh/s08-background-tasks.md) | 长期任务 | 🔜 计划中 |
| [s09 Agent Teams](https://github.com/shareAI-lab/learn-claude-code/blob/main/docs/zh/s09-agent-teams.md) | 多 agent 协作 | 🔜 计划中 |
| [s10 Team Protocols](https://github.com/shareAI-lab/learn-claude-code/blob/main/docs/zh/s10-team-protocols.md) | 通信协议 | 🔜 计划中 |
| [s11 Autonomous Agents](https://github.com/shareAI-lab/learn-claude-code/blob/main/docs/zh/s11-autonomous-agents.md) | 自治度 · workflow vs agent | 🔜 计划中 |
| [s12 Worktree Isolation](https://github.com/shareAI-lab/learn-claude-code/blob/main/docs/zh/s12-worktree-task-isolation.md) | 沙盒隔离 | 🔜 计划中 |

### 本页之外的外部资源（仍在 loop 本身范围内）

- **ReAct 原论文** `arxiv 2210.03629` — 读完 s01 + 本页后，论文 §3-4 会有"原来如此"的体感
- **Anthropic《Building Effective Agents》** (2024.12) — 工业视角的 workflow vs agent 分界
- **Sutton & Barto《Reinforcement Learning: An Introduction》** 第 3 章 — 看懂 agent-environment loop 的数学形式

## 一句话抓手

**Loop 本身是 20 行代码**。真正的学问在 loop 周围的四个约束：

<div data-viz="MindMapCompression"></div>

① **context**（能塞多少） · ② **tools**（能做什么） · ③ **planning**（怎么决定下一步） · ④ **safety**（做错了谁兜底）

> **本页专注第 ③ 约束（loop 本身 = planning 层的最小单元）**，另外 3 个约束的深度补充在上面的导航表里——各自对应 shareAI s02 / s04 / s06 / s11，以及未来同名的 `*.md`。
>
> 这四条任何一条松掉，loop 就变成玩具。你在 shareAI 12 讲里写的每个组件，本质上都在服务其中一个约束——先把本页的 loop 吃透，再一讲一讲补。
