---
title: 为什么 Agent 必须是循环
description: 上一篇推导出 Agent 需要六种能力。这些能力组装起来，为什么一定是一个循环，而不是一次调用？5 个底层原因 + 60 年历史血脉 + messages 数组机制
status: draft
date: 2026-06-27
depth: Medium
tags: [agent, agent-loop, ReAct, first-principles, control-theory]
prev: 01-why-agent-is-inevitable
next: 03-karpathy-route
---

# 为什么 Agent 必须是循环

> [上一篇](01-why-agent-is-inevitable.md) 推导出完整智能需要六种能力，LLM 只覆盖了推理，剩下的靠 Agent 补齐。这篇回答下一个问题：这些能力组装起来，为什么一定是一个循环？

**观点标识说明**：📚 事实；🔍 分析；💡 作者观点。

---

## 第一章 为什么不能一次调用

很多人第一次接触 Agent 会想：让模型一次性把所有步骤想好，然后依次执行，不行吗？

不行。有五个底层原因，每个单独都不致命，合在一起让「一次调用」彻底失效。

🔍 五个原因：

**信息不全。** 开工时模型什么都不知道。文件内容、命令输出、测试结果，都要「跑一下才知道」。不跑就不知道该想什么。

**分支在运行时展开。**「修 bug」展开是：读代码 → 发现问题 → 改 → 跑测试 → 挂了 → 读报错 → 再改。分支数量事先不可能预知，无法一次规划完。

**上下文有限。** 不可能把整个代码库一次塞进窗口。循环按需拉信息、用完释放，是上下文经济学的唯一解。

**工具调用必须交还控制权。** 模型说「我要读文件」，harness 执行、拿结果、再喂回。每一次工具调用都是一个 yield/resume，这本身就是循环的一个迭代。

**错误需要反馈。** 闭着眼写代码必然出错。编译报错、测试红、命令失败——没有循环就没有反馈回路，模型看不见信号就没法修正。

💡 五条合在一起，让预编排流程、一次调用、树搜索等方案都失效。循环不是一种可选的实现方式，而是感知-行动-反馈闭环在工程上的唯一表达。

---

## 第二章 循环的 60 年血脉

📚 今天的 Agent 循环不是 2022 年凭空发明的。从 1960 年代的机器人到 1990 年代的强化学习，「感知→规划→行动」这个结构被六批不同的人在六个不同领域独立发现。ReAct [1] 只是把它搬到了 LLM 上。

### 六条源流

**1966-1972 · Shakey 机器人** [5]

📚 世界第一台「思考型」移动机器人（Nilsson, Rosen 等，Stanford Research Institute）。拿着摄像头在走廊里推箱子，第一次把「感知-建模-规划-执行」做成可运行系统。它用的 STRIPS 规划器（前置条件→动作→后置效果）至今仍是 AI 规划的语法基础。

**1970s · Sense-Plan-Act（SPA）**

📚 经典 AI 范式的通用骨架。把智能体拆成三个阶段：Sense（传感器读入）→ Plan（符号推理生成动作序列）→ Act（执行器输出）。每轮结束后新感知反哺模型，下一轮重新规划。ReAct 的 Thought/Action/Observation 本质上就是 Plan/Act/Sense 换了个名字。

**1986 · Subsumption Architecture** [6]

📚 Brooks（iRobot 创始人，MIT）的论文 *Intelligence Without Reason* 抨击 SPA 太慢太脆——机器人还没想完，老鼠已经跑了。他提出反应式分层架构：多个感知-动作循环并行运行，底层反射快、高层规划慢，高层压制低层。这是今天多 Agent、subagent 的架构思想源头。

**1983-至今 · SOAR** [2]

📚 图灵奖得主 Newell 的统一认知架构（Newell, Laird, Rosenbloom，CMU）。所有认知活动统一为一个决策周期：阐述状态 → 提议算子 → 评估 → 选择 → 应用。目标栈遇到僵局就自动生成子目标——这是 subagent 的原型。

**1993-至今 · ACT-R** [3]

📚 认知心理学建模的标杆（Anderson，CMU）。核心是产生式规则（IF pattern THEN action）不断匹配工作记忆并触发。每一次触发就是一轮循环，结果更新工作记忆，引发下一轮匹配。ACT-R 明确区分程序性记忆（规则）和陈述性记忆（事实），这正是今天 Agent 里「工具」和「上下文」的分离。

**1998 · Agent-Environment Loop** [4]

📚 Sutton 和 Barto 把 Agent 抽象为数学对象：每个 timestep t，Agent 观察状态 sₜ，选择动作 aₜ，环境反馈奖励 rₜ₊₁ 和新状态 sₜ₊₁。目标是最大化累积奖励。这是今天所有 Agent 循环的数学骨架。

### 汇流

🔍 六条源流的循环表达对照：

| 源流 | 年代 | 循环结构 |
|---|---|---|
| Shakey | 1966 | perceive → model → plan → act |
| SPA | 1970s | sense → plan → act |
| Subsumption | 1986 | 多个 sense→act 并行，分层压制 |
| SOAR | 1983 | elaborate → propose → decide → apply |
| ACT-R | 1993 | match → select → fire → update memory |
| RL | 1998 | sₜ → π(aₜ\|sₜ) → env → rₜ₊₁, sₜ₊₁ |

📚 2022 年 Yao 等人写 ReAct [1] 时，不是发明了一个新循环，而是把已经跑了 60 年的结构用自然语言 prompt 表达出来，让 LLM 加入了这个古老的游戏。

---

## 第三章 循环怎么跑：messages 数组

🔍 理解了为什么必须是循环，下一个问题是：循环的状态存在哪里？

答案是 messages 数组。它不只是「对话历史」，而是循环的全部状态载体。每轮循环把 assistant turn（模型的思考和工具调用请求）+ tool_result（工具执行结果）追加进 messages，整个喂回模型。模型再输出下一个 assistant turn，再追加。循环之外没有任何变量在累积状态。

```python
messages = [system_prompt]

while not done:
    response = llm(messages)                    # 模型想下一步
    messages.append(response)                    # 追加 assistant turn

    if response.has_tool_call:
        result = execute(response.tool_call)     # harness 执行
        messages.append(tool_result(result))      # 追加结果
    else:
        done = True                              # 不再调工具，循环结束
```

💡 这段代码的核心设计决策：模型只负责「想下一步」（生成 assistant turn），不负责「把下一步跑起来」。harness 做调度，LLM 做决策。两者通过 messages 数组交换信息。这就是 Agent 架构里区分 Agent（模型）和 Harness（框架）的原因——前者是大脑，后者是手脚。

---

## 第四章 循环怎么坏

🔍 理解循环怎么跑，还要理解它怎么跑崩。以下四种故障都出在循环本身，不涉及工具权限、注入攻击等外部问题。

| 故障 | 本质 | 对策 |
|---|---|---|
| 死循环 | 模型拿到同样结果却不知道换思路，反复调同一个工具 | 循环检测 / 相似度判重 / 强制跳出 |
| 上下文崩塌 | 30 轮后 messages 太长，模型忘了最初目标 | compaction（压缩历史）或重启子 Agent |
| 过早收敛 | 测试没跑、需求没满足就宣告完成 | 独立 verifier 复核 |
| 状态漂移 | 中间某轮的 tool_result 是错的，后续决策基于错误前提 | 工具返回带 schema 校验 + 错误码 |

💡 四种故障有一个共同规律：反馈链断了。死循环是反馈没有新信息，上下文崩塌是反馈被遗忘，过早收敛是没做反馈就结束了，状态漂移是反馈本身是错的。回到上一篇的六种能力框架——「反馈」是最容易出问题的一环，因为它依赖所有其他环节都正确工作。

> 真实案例拆解见 [Agent 退化循环（doom loop）](../harness/agent-doom-loop.md) 和 [沙箱生命周期可靠性](../harness/agent-sandbox-lifecycle-reliability.md)。

---

## 我的思考

💡 三个判断：

**循环不是实现细节，是智能的结构性要求。** 第二章的六条血脉说明，不管技术栈怎么换（机器人/认知科学/强化学习/LLM），只要目标是「在不确定环境中完成任务」，最终都会收敛到感知→决策→行动→反馈的循环。这不是某个团队的设计选择，是问题本身的结构决定的。

**识别循环是可迁移的能力。** 理解了 Agent 循环的本质是「带反馈的决策过程」，就能在任何系统里认出它：HTTP server 的请求-处理-响应，游戏 AI 的感知-决策-行动，甚至做菜的「尝味道→调味→再尝」。判断标准四条：有反馈、有运行时分支、有状态累积、思考和执行分离。四条都满足就是 Agent 循环。

**循环的 20 行代码不难，难的是四个约束。** 循环本身就是一个 while 加几行 append。真正的工程难度在循环周围：上下文能塞多少（context）、能调什么工具（tools）、怎么决定下一步（planning）、做错了谁兜底（safety）。四个约束任何一个松掉，循环就变成玩具。

---

## 参考资料

### Agent 循环核心

[1] Yao, S. et al. (2023). ReAct: Synergizing Reasoning and Acting in Language Models. *ICLR 2023*.

[2] Laird, J., Newell, A., & Rosenbloom, P. (1987). SOAR: An Architecture for General Intelligence. *Artificial Intelligence*, 33(1), 1-64.

[3] Anderson, J. R. et al. (2004). An Integrated Theory of the Mind. *Psychological Review*, 111(4), 1036-1060.

[4] Sutton, R. & Barto, A. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.

### 历史源流

[5] Nilsson, N. (1984). *Shakey the Robot*. SRI International Technical Note 323.

[6] Brooks, R. (1991). Intelligence Without Reason. *IJCAI-91*.

### 延伸阅读

- [上一篇：为什么 AI 必然走向 Agent](01-why-agent-is-inevitable.md)
- [下一篇：Karpathy 路线](03-karpathy-route.md)
- [Agent 退化循环（doom loop）](../harness/agent-doom-loop.md)
- [沙箱生命周期可靠性](../harness/agent-sandbox-lifecycle-reliability.md)
