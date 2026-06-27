---
title: Agent 是什么？为什么 AI 必然会走向 Agent
description: 从达特茅斯会议到 GPT 再到 Agent，用第一性原理把 AI 的演进逻辑讲透——带论文依据和观点标识，区分事实、分析与作者判断
status: draft
date: 2026-06-25
depth: Medium
tags: [agent, first-principles, AI-history, LLM, evolution]
---

# Agent 是什么？为什么 AI 必然会走向 Agent

> 从达特茅斯会议到 GPT，再到 Agent，一篇文章看懂 AI 的演进逻辑。

**观点标识说明**：本文区分三类内容——📚 事实（引用论文、教材、历史资料）；🔍 分析（基于多个来源的归纳）；💡 作者观点（第一性原理思考和判断）。

---

## 前言：为什么今天所有 AI 公司都在做 Agent

1956 年，达特茅斯会议提出 AI。2023 年，GPT 爆发，全民聊天。2025 年，几乎所有 AI 公司都在做 Agent。

凭什么是 Agent？它只是「LLM + Tools」吗？

💡 简短的回答：Agent 这个概念在 AI 学科里存在了 30 年，一直没大规模落地，因为缺一个足够强的推理引擎。LLM 补上了这块之后，剩下的能力——感知、行动、记忆、反馈——就变成了可以工程化解决的问题。路径一旦清晰，所有人自然同时转向。这不是跟风，是积累够了才走到这一步。

下面从 AI 的初心开始，一层层把这条因果链讲透。

---

## 第一章 AI 的初心是什么

📚 1955 年，McCarthy、Minsky、Rochester 和 Shannon 提交了达特茅斯会议的申请书。里面有一句话定义了整个学科的目标：

> "Every aspect of learning or any other feature of intelligence can in principle be so precisely described that a machine can be made to simulate it."
>
> ——McCarthy et al.《A Proposal for the Dartmouth Summer Research Project on Artificial Intelligence》(1955) [1]

注意这句话的关键词：simulate intelligence（模拟智能）。不是 generate text，不是 predict next token，是模拟人的智能。

🔍 从 1956 年至今，AI 经历了专家系统、机器学习、深度学习、Transformer、LLM 几个阶段。每一代的技术手段不同，但达特茅斯定下的目标没变过。

```
专家系统 (1970s-80s)    人写规则，机器照做
       ↓
机器学习 (1990s-2000s)  给数据，让机器自己找规律
       ↓
深度学习 (2012-)        更大的网络、更多的数据，效果起飞
       ↓
Transformer (2017-)     注意力机制，序列建模质变
       ↓
LLM (2020-)             大语言模型，能对话、能推理
       ↓
Agent (2024-)           能自己规划、调工具、完成任务
```

那么问题来了：什么叫「智能」？模拟智能到底要模拟什么？

---

## 第二章 什么是智能

顺着「智能」这个词往下钻，问五层就见底。

**什么是智能？** 不是知道很多东西，是能用所知道的去达成目标。一个人背了整本百科全书但不会用，没人说他「智能」。

**达成目标需要什么？** 需要推理。环境存在未知，得根据已有信息想出下一步该干嘛。

**光推理够不够？** 不够。知道答案并不能改变世界。想出来还得做出来，目标才能达成。

**做完就行了？** 不行。第一次决策可能错误。做完得知道做得对不对，看到结果，判断离目标近了还是远了，然后调整。

**调整靠什么？** 靠记住之前干过什么、结果怎样。经验能提升未来决策。没有记忆，每次都从零开始，调整无从谈起。

五层问完，「智能」可以拆成六个具体能力：

| 能力 | 做什么 |
|---|---|
| 目标 | 知道自己要干嘛 |
| 感知 | 接收外部信息 |
| 推理 | 根据信息想出下一步 |
| 行动 | 把想的变成做的 |
| 反馈 | 看到结果，判断对不对 |
| 学习 | 记住经验，下次做得更好 |

🔍 这六条不是某一篇论文的原文，是基于控制论 [3]、认知科学和现代 AI 教材 [2] 的抽象总结。控制论创始人 Wiener 在 1948 年就提出：智能系统的核心是感知-行动-反馈的闭环，不是单向输出。这个洞察直到今天仍然是 Agent 设计的理论根基。

---

## 第三章 Agent 并不是今天才出现

很多人觉得 Agent 是 2024 年随 GPT 一起火起来的新概念。不是。

📚 Russell 和 Norvig 在 AI 领域最权威的教科书里，从第一版（1995 年）就把 Agent 作为核心概念：

> "An agent is anything that can be viewed as perceiving its environment through sensors and acting upon that environment through actuators."
>
> ——Russell & Norvig《Artificial Intelligence: A Modern Approach》[2]

这个定义说的是：只要一个系统能感知环境并对环境做出动作，它就是 Agent。跟 LLM 没关系，跟深度学习没关系——这是 AI 学科从 1995 年就确立的基本概念。

🔍 过去 30 年，Agent 一直活在教科书和学术论文里，没大规模落地。不是概念不对，是缺少足够强的基础模型来支撑「推理」这一环。一个 Agent 要能理解自然语言指令、拆解任务、决定下一步调什么工具——这些能力在 LLM 之前都做不好。LLM 补上了推理能力，Agent 才从理论变成了工程。

---

## 第四章 LLM 为什么不够

📚 LLM 的技术基础是 Transformer（Vaswani et al., 2017 [4]），核心能力由 GPT-3（Brown et al., 2020 [5]）展示：给定上下文，预测下一个 token。

这个机制带来了两个强能力：知识表达（训练数据里的）和推理（根据上下文推导出下一步）。这两条做得很好，好到让人觉得「AI 已经很强了」。

🔍 但拿第二章的六条能力去对 LLM，缺口一眼就看出来：

| 能力 | LLM 的现状 |
|---|---|
| 感知 | 只能读文本（后来加了多模态，但仍是被动接收） |
| 行动 | 只能输出文本，不能操作任何外部系统 |
| 记忆 | 上下文窗口用完就忘，没有跨会话记忆 |
| 反馈 | 不知道自己的回答是对是错，没有验证环节 |

LLM 本质上是一个「只能回答，不能做事」的系统。你问它「帮我订明天的机票」，它会告诉你怎么订，但它订不了。你问它「这段代码有 bug 吗」，它会分析，但它跑不了、测不了、改不了。

这不是模型不够大或训练数据不够多能解决的。这是架构层面的缺失——LLM 的输入是文本，输出也是文本，它跟外部世界之间没有通路。

---

## 第五章 Agent 如何补齐这些能力

Agent 不是一个全新的东西，它是在 LLM 外面包了一层，把缺的那几条能力补上。这些能力不是凭空造出来的，每一项背后都有对应的研究工作：

📚 近几年的关键工作：

| 缺失的能力 | 怎么补的 | 代表工作 |
|---|---|---|
| 行动（Tool Use） | 让模型调用外部工具 | Toolformer（Schick et al., 2023 [6]）、OpenAI Function Calling |
| 推理 + 行动闭环 | 交替进行思考和行动 | ReAct（Yao et al., 2023 [7]） |
| 规划 | 先拆解任务再逐步执行 | Plan-and-Execute、BabyAGI（开源实践） |
| 反思 | 根据失败结果自我修正 | Reflexion（Shinn et al., 2023 [8]） |
| 记忆 | 分层记忆（工作记忆 + 长期存储） | MemGPT（Packer et al., 2023 [9]）、Mem0 |

把这些能力串起来，就是一个循环：

```
拿到目标
  ↓
想（推理 + 规划）
  ↓
做（调工具）
  ↓
看（观察结果）
  ↓
判断：目标达成了吗？
  ├── 没有 → 回到「想」
  └── 达成 → 结束
```

这个循环就是 Agent 的核心。如果你读过 ReAct 论文 [7]，会认出这就是 Thought → Action → Observation 循环。不是学术界凭空造了一个模式，而是「要完成任务」这个需求本身就要求这样的结构。

---

## 第六章 从论文到生产：2024-2026 的验证

第五章的引用全部停在 2023 年。接下来两年发生的事情，刚好可以检验前面的推导是不是纸上谈兵。

### 能力逐项兑现

🔍 把第五章的论文和 2024-2026 的产品对照着看：

| 能力 | 2023 论文 | 2024-2026 生产 |
|---|---|---|
| 行动 | Toolformer [6] | MCP [10]、Agents SDK [11] |
| 推理 + 行动 | ReAct [7] | Claude Code、Cursor |
| 感知 | 多模态论文 | Computer Use [12] |
| 规划 | BabyAGI | Devin |
| 记忆 | MemGPT [9] | CLAUDE.md + Memory 系统 |
| 多 Agent | 学术框架 | A2A 协议 [13]、CrewAI |

几个值得展开的：

📚 **MCP（Model Context Protocol）**[10]：Anthropic 在 2024 年发布的工具接入标准协议。之前每个 Agent 自己对接 API，每换一个工具就重写一次适配层。MCP 把工具描述、调用、返回的格式统一了，Agent 生态的碎片化问题从协议层面解决。OpenAI 随后发布 Agents SDK [11]，走的是同一个方向。

📚 **Computer Use**[12]：Anthropic 在 2024 年让模型直接操作屏幕和鼠标。Agent 的「行动」能力从调 API 扩展到操作 GUI，感知从「读文本」扩展到「看屏幕」。这一步补的是第二章里「感知」和「行动」两条能力的最后一块拼图。

📚 **A2A（Agent-to-Agent Protocol）**[13]：Google 在 2025 年发布的多 Agent 互操作标准。当单个 Agent 的能力稳定之后，下一个工程问题自然是多个 Agent 之间怎么协作。这个顺序跟第二章的能力依赖链一致：先把单体能力补齐，再解决协作。

### 工程重心已经转移

📚 Karpathy 在 2025 年提出 Context Engineering [14]：Agent 的核心工程问题不再是「怎么调工具」，而是「怎么管上下文」。模型的推理窗口有限，Agent 跑多步之后，历史信息怎么压缩、什么该留什么该丢、怎么避免上下文腐烂，成了比工具调用更难的问题。

🔍 这个信号说明 Agent 的基本架构（推理 + 工具 + 记忆 + 反馈）已经稳定，不再是研究问题。工程重心在向更深层的可靠性移动——而这恰恰是一个技术从论文走向生产的标志。

---

## 第七章 为什么 Agent 是 AI 的必然演进

💡 以下是我的分析。

回到开头的问题：为什么 AI 一定会走向 Agent？

把前面几章串起来看：

1. AI 的目标从 1956 年起就是模拟完整智能（第一章）
2. 完整智能需要六种能力：目标、感知、推理、行动、反馈、学习（第二章）
3. Agent 作为概念在 AI 里存在了 30 年，一直缺推理引擎（第三章）
4. LLM 补上了推理，但只覆盖了六种能力中的两种（第四章）
5. 剩下四种能力通过工具调用、反思、记忆等工程手段逐一补齐（第五章）
6. 2024-2026 的工程实践验证了这条演进路径（第六章）

这些能力补齐的顺序也不是随意的——它们之间有依赖关系：

```
只能生成文本         → LLM
能调工具             → LLM + Tool Use
能根据结果调整       → LLM + Tool Use + 反馈循环
能记住跨会话经验     → LLM + Tool Use + 反馈循环 + Memory
能自己拆解复杂任务   → LLM + Tool Use + 反馈循环 + Memory + Planning
```

每加一层，离「完整智能」就近一步。加到最后，就是 Agent。这条路不存在岔道——你不可能跳过「行动」直接到「学习」，也不可能不要「反馈」就做好「规划」。

💡 所以 Agent 不是谁发明了一个概念然后大家跟风。是 AI 的能力积累到 LLM 阶段之后，下一步该补什么，倒推出来就是 Agent 这个形态。Agent 不是一个独立的新物种，而是 AI 各项能力逐渐补齐后的系统形态。

---

## 第八章 Agent 之后还会怎么走

🔍 Agent 补齐了单个智能体的能力，但离「完整智能」还有距离。从当前研究和工程实践看，下一步大致有两个方向：

**多 Agent 协作**：一个人能做的事有限，一群各有专长的人协作能做更多。现在已经有很多 Multi-Agent 框架在做这件事——让不同 Agent 分工、对话、互相审核。

**自主学习**：目前的 Agent 虽然有记忆，但还不能真正从失败中自我改进。它记住的是「上次用户说了什么」，而不是「上次我哪里做错了、下次该怎么改」。当 Agent 能做到后者，才算真正闭环。

```
Single Agent        一个人干活
     ↓
Multi-Agent         一群人协作
     ↓
Self-learning       从错误中自我改进
     ↓
AGI                 通用人工智能
```

Agent 不是终点，是 AI 从「语言模型」变成「智能系统」这条路上，走到现在的位置。

---

## 我的思考

💡 回顾全文，有三个判断我认为值得单独拎出来：

**AI 的发展不是从 Chat 走向 Agent，是从「生成答案」重新回到「完成目标」。** 达特茅斯的初心就是模拟智能、完成任务，GPT 时代的对话反而是一个中间状态——模型够强了但还没接上手脚。Agent 不是新方向，是回到正轨。

**LLM 改变了 AI 的能力边界，Agent 改变了 AI 与现实世界的交互方式。** LLM 解决了「能不能想明白」的问题，Agent 解决了「想明白之后能不能做到」的问题。两者是互补的，不是替代的。

**能力之间的依赖关系决定了演进顺序几乎是定死的。** 没有推理就没有规划，没有行动就没有反馈，没有反馈就没有学习。这不是事后总结，而是可以用来预测下一步的框架——当前 Agent 最薄弱的环节是自主学习，那么下一波突破大概率在这里。

---

## 参考资料

### 经典理论

[1] McCarthy, J., Minsky, M. L., Rochester, N., & Shannon, C. E. (1955). *A Proposal for the Dartmouth Summer Research Project on Artificial Intelligence*.

[2] Russell, S. & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach* (4th ed.). Pearson.

[3] Wiener, N. (1948). *Cybernetics: Or Control and Communication in the Animal and the Machine*. MIT Press.

### LLM 基础

[4] Vaswani, A. et al. (2017). Attention Is All You Need. *NeurIPS 2017*.

[5] Brown, T. et al. (2020). Language Models are Few-Shot Learners. *NeurIPS 2020*.

### Agent 关键工作（2023）

[6] Schick, T. et al. (2023). Toolformer: Language Models Can Teach Themselves to Use Tools. *NeurIPS 2023*.

[7] Yao, S. et al. (2023). ReAct: Synergizing Reasoning and Acting in Language Models. *ICLR 2023*.

[8] Shinn, N. et al. (2023). Reflexion: Language Agents with Verbal Reinforcement Learning. *NeurIPS 2023*.

[9] Packer, C. et al. (2023). MemGPT: Towards LLMs as Operating Systems. *arXiv:2310.08560*.

### Agent 工程落地（2024-2026）

[10] Anthropic. (2024). *Model Context Protocol (MCP) Specification*. https://modelcontextprotocol.io

[11] OpenAI. (2025). *OpenAI Agents SDK*. https://github.com/openai/openai-agents-python

[12] Anthropic. (2024). *Developing a computer use model*. https://www.anthropic.com/research/developing-computer-use

[13] Google. (2025). *Agent2Agent Protocol (A2A)*. https://github.com/google/A2A

[14] Karpathy, A. (2025). *Context Engineering*. 提出 Agent 的核心工程问题是上下文管理而非工具调用。
