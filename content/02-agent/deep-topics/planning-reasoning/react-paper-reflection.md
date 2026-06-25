# ReAct 论文读后感 —— 从图灵机视角看 Agent 纪元

> 基于与 Claude 的深度对话整理，核心视角：**ReAct 不是"加了工具"的量变，而是计算模型的质变 —— 它把 LLM 从"函数"升格为"图灵机的决策核心"**。

---

## 一、原始读后感（出发点）

### 模式的改变
从之前单一的动作转为连续性动作。

### 内容的变化
给它连续的上下文信息（叠加每次对话的上下文及结果）。

### 外部能力
给 LLM 提供了获取外部信息能力 —— wiki API。

### harness 雏形
当下 harness 架构：model、harness、context。

ReAct 场景映射：
1. LLM = model
2. 外部能力 = harness
3. 连续的上下文信息 = context

---

## 二、升维视角：本质突破

### 1. 从"函数"到"图灵机"

ReAct 之前的 LLM 是纯函数：`f(prompt) = output`，一次性生成。
ReAct 让 LLM 变成了**有状态的决策循环**：

```
while not done:
    thought = reason(context)
    action = decide(thought)
    observation = execute(action)
    context = context + [thought, action, observation]
```

这不是"加了工具"的量变，而是**计算模型**的质变 —— LLM 获得了"读写外部状态"的能力，逼近图灵完备。

### 2. 原始抽象里缺了一个关键要素：**Loop（控制流）**

原始模型：`model + harness + context`（静态三要素）
更完整的是：`model + harness + context + **loop**`

Loop 才是 Agent 与 Chain 的分水岭：
- **Chain**：固定步数、线性展开（Prompt Chaining）
- **Agent**：LLM 自己决定是否继续、下一步做什么（ReAct 首次实现）

谁掌握"是否终止"的决策权 —— 这是架构的灵魂。ReAct 把这个权力交给了 model，所以它才叫 Agent 的起点。

### 3. Reason + Act 的互补性（论文真正的 insight）

- **只 Act 不 Reason**（纯工具调用）→ 容易陷入无意义的动作循环，错误不可恢复
- **只 Reason 不 Act**（纯 CoT）→ 幻觉无法被外部事实校准
- **Reason + Act 交错** → Reasoning 引导 Action 的方向，Action 的 Observation 校准 Reasoning

这对应认知科学里的 **元认知（metacognition）** —— 不只是做，还要"想自己在做什么"。

### 4. 从 ReAct 看后续演化的必然性

今天所有 Agent 架构的痛点，ReAct 已经埋下伏笔：

| ReAct 原始问题 | 现代解法 |
|---|---|
| Context 无限膨胀 | Memory 分层（short/long term） |
| Observation 太长噪音大 | Context Engineering / Summarization |
| LLM 不知道有什么工具可用 | Function Calling / MCP |
| 错误无法回滚 | Plan-and-Execute / Reflexion |
| 单 Agent 能力边界 | Multi-Agent / Swarm |

**一句话提炼**：ReAct 是把"生成模型"改造成"具身智能体（embodied agent）"的最小可行原型。读 ReAct 其实是在读 Agent 纪元的创世纪。

---

## 三、深化：为什么说"计算模型的质变、逼近图灵机"

### 参照系：计算能力分层

计算理论里，"能算什么"是分层的（Chomsky 层级）：

```
表达力 低 ─────────────────────────────────────→ 高
  正则语言  <  上下文无关  <  上下文相关  <  图灵完备
  (regex)    (括号匹配)      (自然语言)      (任意程序)
```

**核心判据**：一个计算模型能不能算某件事，取决于它有没有三样东西：
1. **状态**（记住"我走到哪了"）
2. **可读写的外部存储**（不被输入长度限制）
3. **条件循环**（能根据状态决定跳转）

### 原始 LLM：强大的"查表函数"

一次 forward pass 的 LLM 本质是：
```
output = f(prompt)     # 固定输入长度 → 固定计算量
```

它很强，但它是**有界计算（bounded computation）**：
- 输入长度有上限（context window）
- 计算步数 = 层数 × token 数，**固定**
- 没有真正的"循环"，只有"展开"
- 不能在推理过程中回去修改已经算出的东西

类比：它像一个超级复杂的 Excel 公式。公式再长，也算不了"给我列出所有素数直到我喊停"这种任务 —— 因为 Excel 公式里没有 `while`。

**数学上**：单次 forward 的 Transformer 是 `TC⁰` 复杂度类（可并行的浅层电路），严格弱于图灵机。这不是工程限制，是**理论天花板**。

### ReAct 把 LLM 嵌进了图灵机的框架

图灵机的定义：
```
纸带（无限存储） + 读写头 + 状态机 + 转移规则
```

ReAct 的映射：

| 图灵机组件 | ReAct 对应物 |
|---|---|
| 纸带（tape） | 外部世界 + 累积的 context |
| 读头（read） | `Action: Search[...]` 拉外部信息 |
| 写头（write） | `Observation` 写回 context |
| 状态机 | LLM 本身（每轮重新"看一遍纸带再决策"） |
| 转移规则 | `Thought → Action` 的生成逻辑 |
| 停机 | LLM 自己决定输出 `Finish[...]` |

关键在于**循环 + 外部读写**这两件事一起出现。单有任何一个都不够：
- 只有循环、没有外部读写 = 有限状态机（困在自己脑子里）
- 只有外部读写、没有循环 = 单次函数调用（tool use 早期形态）
- 循环 + 外部读写 = **图灵完备的架子**

### "逼近"而非"等于"的原因

说的是"逼近"，几个保留条件：

1. **Context 仍然有限**：纸带不是无限的，超了就遗忘 → 现实里用 Memory/RAG 补
2. **LLM 决策不保证正确**：图灵机的转移规则是确定性的，LLM 是概率的 → 可能陷死循环
3. **终止不可证**：图灵机的停机问题已经够难，LLM 自己判断停机更不可靠

所以工业上我们说 "Agent = LLM + Loop + Tools + Memory"，就是在补齐图灵机的这四块拼图。

### 一句话总结

**ReAct 之前**：LLM 是个"一锤子买卖"的函数 `y = f(x)`，算力有理论上限。
**ReAct 之后**：LLM 变成**图灵机的"决策核心"**，纸带是外部世界，能算的事情原则上无上限。

这就是为什么 2022 年 ReAct 一出来，所有人都意识到"Agent 纪元开始了" —— 不是因为它解决了什么具体任务，而是因为它**跨越了一个计算理论层级的门槛**。

---

## 四、工程视角：Claude Code / Cursor 在补图灵机的哪几块拼图

先把 ReAct 留下的"裸"图灵机画出来 —— 它的问题是：纸带会满、读写头只有一个、状态机会走神、没有停机保证。现代 agent harness 本质上就是在解决这四个工程问题。

### 拼图 1：**纸带扩容与分层**（Context/Memory）

裸 ReAct 的"纸带" = 单个对话的 context window，爆了就死。

harness 的补法：

| 层级 | Claude Code 实现 | 图灵机类比 |
|---|---|---|
| L1 工作区 | 当前 context | 读写头附近的纸带 |
| L2 会话级 | TodoWrite / Plan | 暂存区（scratchpad） |
| L3 项目级 | `CLAUDE.md` / `AGENTS.md` | 预加载的"公理" |
| L4 跨会话 | `memory/` 持久化目录 | 永久纸带 |
| L5 压缩 | Auto-compact | 纸带重写（保留摘要丢弃细节） |
| L6 外挂 | RAG / Context Hub | 按需 page-in 的虚拟内存 |

**关键洞察**：现代 harness 实际上实现了**操作系统的虚拟内存机制** —— context window 是 RAM，文件系统是硬盘，compaction 是 swap。LLM 以为自己有无限纸带，其实是 harness 在做分页。

### 拼图 2：**多读写头 + 权限分区**（Tools + Permissions）

裸 ReAct 只有一个 `Search` 工具，一次只能干一件事。

harness 的补法：
- **多工具并行** = 多个读头同时工作（Claude Code 支持单轮多 tool call）
- **工具分类** = 纸带分区（Read/Write/Bash/Web 各有权限）
- **permission mode** = 纸带的"只读/可写"保护位
- **sandbox / worktree** = 隔离的副本纸带（改坏了能回滚）

这对应计算机体系结构里的 **MMU（内存管理单元）** —— 不是让 LLM 能读写更多，而是让它**安全地**读写。

### 拼图 3：**状态机加固**（Control Flow）

裸 ReAct 的状态机就是 LLM 自己，会幻觉、会走神、会死循环。

harness 的补法：

```
            ┌─────────────────────────┐
            │  Hooks (前/后置拦截)     │  ← 硬编码的转移规则
            ├─────────────────────────┤
            │  Subagents (递归调用)    │  ← 子图灵机
            ├─────────────────────────┤
            │  Plan Mode (先规划)      │  ← 离线编译转移图
            ├─────────────────────────┤
            │  Slash Commands (宏)     │  ← 预定义状态序列
            ├─────────────────────────┤
            │  LLM (概率转移核心)      │  ← ReAct 原本只有这层
            └─────────────────────────┘
```

对应关系：
- **Hooks** = 确定性的 `if-then` 规则，不信任 LLM 的地方就用硬规则兜底
- **Subagents** = 子程序调用（带独立 context，防止主纸带被污染）
- **Plan Mode** = 先画状态转移图再执行，避免"边想边做"的漂移
- **Verifier/Reviewer** = 独立的验证图灵机（写审分离）

**核心思想**：**不要指望 LLM 是可靠的状态机，而要用确定性结构约束它的转移路径。**

### 拼图 4：**停机判定**（Termination）

裸 ReAct 停机靠 LLM 自己喊 `Finish[]`，不可靠。

harness 的补法：
- **显式 TODO 列表** = 所有 task 完成才停（Claude Code 的 TodoWrite）
- **测试/构建通过** = 外部客观信号（tests pass, build green）
- **verifier 独立判断** = 第二个 LLM 做停机裁判（不能自己判自己）
- **max iterations / budget** = 硬上限兜底（防死循环）
- **Ralph 模式** = 反过来：明确不停机，直到外部信号

### 全景对照

```
     图灵机组件              ReAct (2022)           Claude Code (2026)
──────────────────────────────────────────────────────────────────────
  纸带                    单个 context           分层 memory + 虚存
  读/写头                 一个 Search API        并行多工具 + 权限
  状态机                  纯 LLM                 LLM + Hooks + Subagent
  转移规则                Thought→Action        Plan + Slash + 硬规则
  停机                    LLM 自判               Verifier + Tests + 预算
  可靠性                  能跑 demo              能跑生产
```

---

## 五、真正的顿悟

ReAct 证明了 **LLM + Loop + Tools 理论上图灵完备**。
但"理论完备"和"工程可用"差了十万八千里 —— 就像"图灵机理论上能算一切"和"x86 + Linux 能跑业务"之间的距离。

现代 agent harness 做的事，**本质上是在给 LLM 建一套操作系统**：

| OS 概念 | Agent Harness 对应 |
|---|---|
| 虚拟内存 | Memory 管理（分层 + 压缩 + RAG） |
| 系统调用 | Tool 调度 |
| 进程/线程 | Subagent |
| 中断处理 | Hooks |
| 权限系统 | Permission mode |
| 看门狗 | Verifier |

所以会发现：**OS 设计里学到的所有东西，在 agent harness 设计里都会再用一遍。** 这不是巧合 —— 只要你想让一个概率性的图灵核心稳定干活，你就会独立地重新发明操作系统。

---

## 六、可继续深挖的方向

1. **从 OS 视角反推 agent harness 还缺什么** —— 真正的"进程间通信"、"文件系统语义"、"故障恢复"。
2. **为什么是 2022 年出现而不是更早** —— base model 能力阈值 + instruct-tuning 成熟的交汇点。
3. **ReAct 对今天 harness 设计（如 Claude Code 自己）的直接影响** —— `<thinking>` 标签、tool_use 交错、context 管理策略的血缘关系。
4. **Reasoning model (o1/R1) 对 ReAct 范式的冲击** —— 当 Reason 本身变成内生能力，Act 的边界会怎么移动。
