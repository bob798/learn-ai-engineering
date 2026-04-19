---
title: "ReAct 论文解读 — Synergizing Reasoning and Acting in Language Models"
description: 一篇论文如何定义了现代 AI Agent 的基本形态：Thought → Action → Observation 循环
order: 2
---

# ReAct 论文解读

> **论文**：*ReAct: Synergizing Reasoning and Acting in Language Models*
> **作者**：Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao
> **机构**：Princeton University + Google Research (Brain Team)
> **发表**：ICLR 2023（首次公开 2022.10）
> **原文**：[arxiv.org/abs/2210.03629](https://arxiv.org/abs/2210.03629)
> **代码**：[github.com/ysymyth/ReAct](https://github.com/ysymyth/ReAct) · [本项目的可运行改造版](../../../code/react-hands-on/HANDS_ON.md)

---

# 一、它是什么

## 一句话

ReAct 是第一篇**明确提出**让 LLM 在同一个生成序列中**交替执行推理（Thought）和行动（Action）**的论文。它把 Chain-of-Thought 的"想"和工具调用的"做"统一进一个循环，奠定了现代 AI Agent 的基本架构。

## 外界的评价

ReAct 不是引用量最高的 LLM 论文，但它的**工程影响力**极其不成比例：

| 维度 | 说明 |
|------|------|
| **学术** | ICLR 2023 oral，Google Scholar 2000+ 引用（截至 2025 年中）。被后续几乎所有 agent 论文引用为 baseline |
| **工程** | LangChain 的第一个 Agent 类型就叫 `zero-shot-react`；AutoGPT、BabyAGI、Claude Code、Cursor 的 agent loop 都是 ReAct 循环的工程化变体 |
| **概念** | "Thought / Action / Observation" 这套术语已经成为 AI Agent 领域的**通用语言**，就像 HTTP 的 "Request / Response" 一样自然 |

> Harrison Chase（LangChain 创始人）在 2023 年初公开说过，LangChain 的 Agent 模块直接受 ReAct 启发。这篇论文是 LLM Agent 从"学术概念"变成"人人可用的工程模式"的转折点。

## 与过去和当下的联系

ReAct 不是凭空发明。它站在两条已有路线的交叉点上：

```
路线 A：推理（Reasoning）                路线 B：行动（Acting）
───────────────────                    ───────────────
Chain-of-Thought (Wei et al. 2022)     WebGPT (OpenAI, 2021.12)
Self-Consistency (Wang et al. 2022)    SayCan (Google, 2022.04)
                                       MRKL (AI21, 2022.05)
                    ↘                ↙
                      ReAct (2022.10)
                    ↙                ↘
            今天的 Agent 生态           今天的 Tool Use 标准
            (Claude Code, Cursor,      (Anthropic tool_use,
             LangChain, AutoGPT)       OpenAI function calling)
```

- **路线 A** 证明了"让模型先想再答"能大幅提升准确率，但模型只会想，不会**做**——它会编造事实（hallucination）。
- **路线 B** 证明了"让模型调用工具"能获取真实信息，但模型只会做，不会**想**——它盲目执行，缺乏策略。

**ReAct 的贡献**：把 A 和 B **交织**（interleave）进同一个生成过程。模型在同一个 prompt 里既输出 Thought（推理），又输出 Action（行动），然后从真实世界拿回 Observation（观察）。这就是现在所有 agent loop 的原型。

### 更远的血脉

如果追溯到 AI 的经典范式，ReAct 的 Thought / Action / Observation 其实就是 **Sense-Plan-Act（SPA）** 换了个名字：

| 经典 AI (1970s) | 强化学习 (1998) | ReAct (2022) | 现代 Agent Loop |
|---|---|---|---|
| Sense | State sₜ | Observation | tool_result |
| Plan | Policy π(aₜ\|sₜ) | Thought | assistant turn（推理） |
| Act | Action aₜ | Action | tool_use call |

> 详细的 6 条历史血脉见 [Agent Loop 深度理解](../harness/agent-loop.md#-远古思想源loop-的六条血脉)。

---

# 二、做了什么

## 核心主张

论文用一句话总结自己的贡献：

> *"We propose ReAct, a general paradigm to **synergize reasoning and acting** with language models for solving diverse language reasoning and decision-making tasks."*

翻译：让 LLM 在**同一个生成过程**中**既推理又行动**，解决需要知识检索和多步决策的任务。

## 四类实验任务

论文在 4 个任务上验证了 ReAct：

| 任务 | 类型 | 数据集 | ReAct 用什么工具 |
|------|------|--------|-----------------|
| **HotpotQA** | 多跳问答 | HotpotQA dev | Wikipedia Search + Lookup |
| **FEVER** | 事实验证 | FEVER dev | Wikipedia Search + Lookup |
| **ALFWorld** | 虚拟家居交互 | ALFWorld (TextWorld) | 游戏内动作（go to, pick up, heat...） |
| **WebShop** | 网页购物 | WebShop 模拟电商 | 搜索、点击、购买 |

前两个是**知识密集型**任务（需要外部信息），后两个是**决策密集型**任务（需要多步操作）。ReAct 在两类任务上都有效。

## 对比了谁

| 方法 | 有 Thought？ | 有 Action？ | 特点 |
|------|:-----------:|:-----------:|------|
| **Standard** | ❌ | ❌ | 直接问直接答 |
| **CoT (Chain-of-Thought)** | ✅ | ❌ | 只推理，不查外部信息 → 会编造事实 |
| **Act-only** | ❌ | ✅ | 只行动，不推理 → 盲目搜索，缺乏策略 |
| **ReAct** | ✅ | ✅ | 推理和行动交替 → 既有策略又有真实信息 |

## 核心发现

1. **ReAct 在知识任务上超越 Act-only**：有 Thought 让模型知道"该搜什么"，而不是乱搜。
2. **ReAct 在决策任务上超越 CoT**：有 Action 让模型拿到真实信息，而不是编造。
3. **ReAct 的 hallucination 率远低于 CoT**：论文发现 CoT 在 FEVER 上的主要失败原因是 hallucination（编造事实），而 ReAct 因为能查 Wikipedia，几乎消除了这类错误。
4. **CoT + Self-Consistency 在纯推理上可能更强**：论文诚实地指出，对于不需要外部信息的纯推理问题，CoT (SC) 的分数可能更高。这不是 ReAct 的短板，而是分工——**ReAct 解决的是需要外部信息的问题**。
5. **ReAct + CoT-SC 组合最强**：论文提出了一个融合策略——先用 ReAct 跑，如果模型不确定，再用 CoT (SC) 投票，取两者中置信度更高的答案。

---

# 三、如何做的

这是论文的技术核心，也是你理解 agent loop 代码的关键。

## 3.1 Prompt 结构

ReAct 没有微调模型，完全靠 **few-shot prompting**。发给模型的 prompt 由 5 部分组成：

```
┌──────────────────────────────────────────────────────┐
│ 1. INSTRUCTION（系统指令）                              │
│    "Solve a question answering task with              │
│     interleaving Thought, Action, Observation..."     │
│    定义三种 Action: Search / Lookup / Finish           │
├──────────────────────────────────────────────────────┤
│ 2. Few-shot 示例（6 道完整示例）                         │
│    每道包含完整 Thought→Action→Observation 循环          │
├──────────────────────────────────────────────────────┤
│ 3. 当前题目                                            │
│    "Question: Were Scott Derrickson and Ed Wood..."    │
├──────────────────────────────────────────────────────┤
│ 4. 已有的推理历史（逐轮累加）                            │
│    Thought 1→Action 1→Observation 1→Thought 2→...     │
├──────────────────────────────────────────────────────┤
│ 5. 当前轮提示前缀                                      │
│    "Thought {i}:" ← 模型从此续写                       │
│    stop_sequences = ["\nObservation {i}:"]             │
└──────────────────────────────────────────────────────┘
```

> 完整的 prompt 原文和逐轮增长过程见 [results/idx-0.md](../../../code/react-hands-on/results/idx-0.md)。

## 3.2 循环机制（The Loop）

这是整篇论文最重要的机制，也是一切 agent loop 的原型：

```
程序 (你的电脑)                         LLM API (远程服务器)
═══════════════                        ═══════════════════

拼接 prompt + "Thought 1:"
            ── HTTP 请求 ──►
            stop = ["\nObservation 1:"]
                                        模型逐 token 生成...
                                        "...Action 1: Search[X]"
                                        "\nObservation 1:" ← 命中 stop!
                                        ⛔ 强制停止
            ◄── HTTP 响应 ──
            返回 Thought + Action

解析 Action → 调 Wikipedia → 拿到真实结果
将 Observation 拼回 prompt
加上 "Thought 2:" → 再发给模型
            ── HTTP 请求 ──►
            ...循环继续...
```

**三个关键设计**：

### ① stop_sequences 拦截

模型看过 few-shot 示例里完整的 Thought → Action → Observation 循环，所以它**有能力自己编造 Observation**。`stop_sequences=["\nObservation {i}:"]` 强制在模型输出到 Observation 之前截断，然后由程序填入**真实的 Wikipedia 搜索结果**。

这是 ReAct 最精妙的设计：**用 few-shot 教模型"格式"，用 stop_sequences 夺回"内容"的控制权**。

### ② 历史累加

每轮的 Thought + Action + Observation 都追加到 prompt 末尾。模型在下一轮看到的是**完整的推理历史**，所以它能基于之前的搜索结果做进一步推理。

对应代码（`run_react.py:193`）：
```python
prompt += step_str  # 本轮结果追加到 prompt，供下一轮使用
```

### ③ 两种退出条件

- **正常退出**：模型输出 `Finish[answer]` → `done=True`，跳出循环
- **强制退出**：跑满 `max_steps`（默认 8 轮）仍未 Finish → 强制 `finish[]`，空答案得 0 分

对应代码（`run_react.py:163-202`）：
```python
for i in range(1, max_steps + 1):
    thought_action = llm(prompt + f"Thought {i}:", stop=[f"\nObservation {i}:"])
    # ...解析 Action，调用 Wikipedia...
    if done:
        break
if not done:
    obs, r, done, info = env_step(env, "finish[]")
```

## 3.3 一个完整的例子

以 HotpotQA idx=0 为例——"Were Scott Derrickson and Ed Wood of the same nationality?"：

| 轮次 | Thought（模型推理） | Action（模型决策） | Observation（程序填入） |
|:----:|---|---|---|
| 1 | 需要搜 Scott Derrickson 和 Ed Wood，找国籍，然后比较 | `Search[Scott Derrickson]` | Scott Derrickson (born July 16, 1966) is an **American** filmmaker... |
| 2 | Scott Derrickson 是美国人。需要再搜 Ed Wood | `Search[Ed Wood]` | Edward Davis Wood Jr. ... was an **American** filmmaker... |
| 3 | Ed Wood 也是美国人。所以他们国籍相同 | `Finish[yes]` | Episode finished, reward = 1 ✅ |

3 轮 LLM 调用，0 次 parse 失败，EM = 1，F1 = 1.00。

> 完整的逐轮 prompt 原文、API 交互图、评分细节见 [results/idx-0.md](../../../code/react-hands-on/results/idx-0.md)。

## 3.4 为什么不用 fine-tune

论文选择 few-shot prompting 而非 fine-tuning，原因有二：

1. **通用性**：few-shot 不需要标注数据，换个任务只需要换几个示例。这让 ReAct 成为一个**通用范式**而非特定任务的解法。
2. **可解释性**：Thought 是模型用自然语言写出的推理过程，人类可以直接阅读、审计、调试。Fine-tuned 模型的推理过程是隐式的，看不见。

---

# 四、对于初学 AI 的我们，需要关注什么

## 4.1 论文层面：读懂这三件事就够了

### ① "交织"是核心词

不是"先想完再做"，也不是"先做完再想"，而是**交替**。每一轮 Thought 基于上一轮的 Observation 做推理，每一轮 Action 基于当前 Thought 做决策。这种交织让模型能**根据新信息修正策略**——这是 ReAct 超越 CoT 和 Act-only 的根本原因。

### ② stop_sequences 是控制权的分界线

整个 ReAct 的"让模型不要编造 Observation"这件事，不是靠提示词实现的，而是靠 **API 参数** `stop_sequences` 硬性截断。这是你第一次看到"程序控制模型行为"的范例——不是求模型听话，而是在 API 层强制它停下。

> 这个思想在今天演化成了 **tool_use API**：Anthropic 和 OpenAI 的现代 API 直接提供结构化的工具调用接口，不再需要靠 stop_sequences 解析文本。但底层思想完全一样——**程序决定什么时候该模型输出，什么时候该程序执行**。

### ③ 论文的诚实之处

ReAct 论文有一个难得的品质：它**诚实地报告了自己不如 CoT (SC) 的场景**。在不需要外部信息的纯推理任务上，CoT + Self-Consistency 更强。论文甚至提出了 "ReAct + CoT-SC" 的融合策略来弥补。

这告诉我们：**没有银弹**。ReAct 解决的是"需要外部信息 + 多步决策"的问题，不是所有问题。

## 4.2 代码层面：动手做这三个实验

读论文不如跑代码。本项目提供了 [可运行的 ReAct 复现代码](../../../code/react-hands-on/HANDS_ON.md)，以下三个实验能帮你把论文理解从"读过"变成"懂了"：

### 实验 1：跑单题，逐轮观察

```bash
cd code/react-hands-on
python run_react.py --idx 0
```

看完整的 Thought → Action → Observation 交替输出。重点观察：
- 模型在 Thought 里**规划了什么**
- 模型在 Action 里**选择了什么工具、传了什么参数**
- Observation 里的**真实信息**如何影响下一轮 Thought

### 实验 2：去掉 Thought，观察分数变化（Ablation）

修改 `run_react.py` 中的 `INSTRUCTION`，去掉 "Thought can reason about the current situation" 这句话，改用 `prompts_naive.json` 里的 Act-only prompt。跑 50 题对比：

```bash
# 原版 ReAct
python run_react.py --n 50 --quiet

# Act-only（需要你修改 prompt）
python run_react.py --n 50 --quiet
```

这就是论文 Figure 2 的核心实验——去掉推理步骤，分数掉多少？

### 实验 3：改工具粒度

论文定义了 3 个工具：`Search`、`Lookup`、`Finish`。试试把 `Search` 和 `Lookup` 合并成一个 `wiki_query`，看 loop 长度和分数的变化。这会让你直观理解**工具粒度**对 agent 行为的影响。

## 4.3 认知层面：建立这两个心智模型

### 心智模型 1：Agent = Loop + Tools + LLM

ReAct 论文用最简洁的方式展示了这个等式：
- **Loop**：Thought → Action → Observation 的循环（`run_react.py` 的 `for i in range(1, max_steps + 1)`）
- **Tools**：Wikipedia Search + Lookup（`wikienv.py`）
- **LLM**：生成 Thought 和 Action 的模型（`make_llm()` 返回的函数）

今天的 Claude Code、Cursor、LangChain Agent **全部是这个等式的工程化放大**。Loop 变复杂了（加了 compaction、subagent、task system），Tools 变多了（文件读写、代码执行、网络请求），LLM 变强了（从 text-davinci-002 到 Claude Opus）。但骨架没变。

### 心智模型 2：Prompt 是程序，stop_sequences 是中断

ReAct 的 prompt 不是"一段话"，而是一个**用自然语言编写的程序**：

- INSTRUCTION = 函数签名（定义输入输出格式）
- Few-shot 示例 = 单元测试（示范期望行为）
- 当前题目 + 历史 = 运行时状态
- stop_sequences = 中断（在特定位置夺回控制权）

当你这样看 prompt，就不会再把它当成"求模型帮忙的一段话"，而是当成**你和模型之间的协议**。这个视角迁移到 tool_use API、function calling、甚至 MCP 都成立。

---

# 出口

## 读完本文之后

| 你想… | 去这里 |
|-------|--------|
| 跑代码 | [ReAct Hands-On 完整指南](../../../code/react-hands-on/HANDS_ON.md) |
| 理解 Loop 的理论深度 | [Agent Loop 深度理解](../harness/agent-loop.md)（60 年血脉、messages 数组机制） |
| 看完整的 Prompt 原文和逐轮 trace | [results/idx-0.md](../../../code/react-hands-on/results/idx-0.md) |
| 读原论文 | [arxiv.org/abs/2210.03629](https://arxiv.org/abs/2210.03629) |
| 了解 ReAct 之后的 Agent 生态演化 | [Agent 生态 2026](agent-ecosystem-2026.md) |

## 一句话带走

**ReAct 的本质不是一个算法，而是一个接口规范**：它定义了 LLM 和外部世界之间"想一步、做一步、看一步"的通信协议。今天所有 agent 系统——不管多复杂——都在执行这个协议的某个变体。读懂 ReAct，就读懂了 agent 的骨架。
