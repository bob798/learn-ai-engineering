# Agent Harness 深度拆解 · 从 0 到 1 自建者的逆向工程笔记

> 生成日期：2026-06-12
> 起源：从 0 到 1 搭建 agent + harness 前，先把主流开源项目拆开看清楚
> 方法：领域 → 问题域 → 子问题域 → 根因 → 各产品方案（落到源码/机制层）→ 最佳实践 → 可迁移启示
> 拆解对象：**Aider · OpenHands · LangGraph · SWE-agent/mini-swe-agent**，以 Claude Code/Agent SDK 与 Manus 的公开工程实践作横向参照

---

## 〇、先把词定义清楚：什么是 Harness

**Agent = 模型 + 循环 + 工具。Harness = 围绕这个循环的一切工程脚手架。**

模型厂商交付的是一个"会推理、会发起工具调用"的大脑；但大脑不会自己：

- 管理自己的上下文窗口（满了就失忆）
- 安全地执行代码（一条 `rm -rf` 就是生产事故）
- 从崩溃中恢复（进程挂了，几小时的工作归零）
- 把编辑准确写进文件（LLM 输出的 diff 经常对不上行号）
- 知道自己干得对不对（没有验证就没有可靠性）

Harness 就是补齐这些能力的**运行时容器**：上下文管理器、工具执行器、沙箱、状态持久化、验证回路、可观测性。2025 年以来业界形成一个重要共识——**模型每代变强，harness 里的"补丁式智能"就贬值一轮，但"基础设施式能力"（沙箱、持久化、验证）反而升值**。拆解下面四个产品时，请始终带着这个透镜：它做的事是"替模型补智商"还是"给模型修跑道"？

---

## 一、深化后的拆解框架：8 个问题域

用户视角的框架是"领域→问题域→原因→方案→最佳实践"。基于对这些项目的逆向，我把 agent harness 这个领域固化为 **8 个问题域**，每个都给出根因分类——根因只有三类：

| 根因类型 | 含义 | 例子 |
|---|---|---|
| **M（模型特性）** | LLM 本质上是无状态的、概率的、上下文有限的 | 失忆、幻觉行号、lazy coding |
| **E（工程特性）** | 分布式系统、进程、文件系统的经典问题 | 崩溃恢复、并发、隔离 |
| **$（经济特性）** | token 计费、KV-cache、延迟成本 | 上下文越长越贵越慢 |

### 问题域总览

| # | 问题域 | 核心子问题 | 根因 |
|---|---|---|---|
| P1 | **Agent Loop 控制** | 何时停？卡死检测？迭代预算？步内/步间状态？ | M+E |
| P2 | **上下文与记忆管理** | 窗口溢出怎么办？什么进上下文？长期记忆放哪？ | M+$ |
| P3 | **工具调用与 ACI 设计** | 给模型什么样的接口？粒度多粗？反馈怎么裁剪？ | M |
| P4 | **代码/文件编辑可靠性** | 模型输出的编辑怎么可靠落盘？ | M |
| P5 | **沙箱与执行环境** | 在哪执行？怎么隔离？环境怎么复现？ | E |
| P6 | **错误恢复与持久化** | 进程崩了怎么续？人要介入怎么暂停？ | E |
| P7 | **验证与反馈回路** | agent 怎么知道自己做对了？ | M |
| P8 | **可观测性与评估** | 怎么调试一条 50 步的轨迹？怎么衡量改进？ | E |

四个拆解对象恰好各自把其中 2-3 个问题域做到了极致，这正是选它们的原因：

- **Aider** → P3/P4（编辑可靠性的教科书）+ P2（repo-map 是上下文选择的开创性方案）
- **OpenHands** → P5（沙箱与运行时的工业级答案）+ P1（事件溯源的 loop 架构）
- **LangGraph** → P6（持久化与人机协同的运行时原语）+ P1（显式图控制流）
- **SWE-agent** → P3（ACI 是它发明的词，有论文级证据）+ 其续作 mini-swe-agent 是对整个领域的"反转测试"

---

## 二、产品拆解 ① Aider —— 编辑可靠性与上下文选择的教科书

### 2.1 领域定位

终端里的 AI 结对编程工具（2023 年至今，Paul Gauthier 个人主导，持续活跃维护，最近一次 PyPI 发布在 2026 年 2 月）。它**不是**自治 agent——人始终在回路里，每轮人发指令、模型改代码、自动 git commit。正因为砍掉了自治性，它把"单轮编辑的可靠性"打磨到了极致。

### 2.2 它主攻的问题域

#### P4 · 编辑可靠性：edit format 体系

**子问题**：LLM 怎么把"我想改这段代码"可靠地变成磁盘上的文件变更？

**根因（M）**：三个模型特性叠加——
1. 模型对**行号没有可靠感知**（tokenizer 看不见"第 47 行"），所以经典 unified diff 的 `@@ -47,6 +47,8 @@` 头是幻觉重灾区；
2. 模型有 **lazy coding** 倾向（输出 `# ... rest of the code ...` 偷懒省略）；
3. 整文件重写虽然可靠，但 token 成本随文件大小线性爆炸（$）。

**Aider 的方案**：一套可插拔的 edit format 体系，源码在 `aider/coders/` 下，每种格式一个 Coder 类，全部继承 `base_coder.py`：

| 格式 | Coder 类 | 机制 | 适用 |
|---|---|---|---|
| `whole` | WholeFileCoder | 整文件重写 | 弱模型、小文件，最可靠最贵 |
| `diff` | EditBlockCoder | **SEARCH/REPLACE block**：模型输出"原文片段→替换片段"，harness 用精确匹配+容错匹配定位 | 主力格式，绝大多数强模型默认 |
| `udiff` | UnifiedDiffCoder | 简化版 unified diff（**不要求行号正确**，hunk 头被忽略，只用上下文行定位） | 专为治 GPT-4 Turbo 的 lazy coding 设计 |
| `editor-diff/editor-whole` | — | 给 architect 模式的执行模型用的精简变体 | 双模型流水线 |

**这里最值得抄的设计**：SEARCH/REPLACE 的本质是**把"定位"这个模型不擅长的活，从坐标空间（行号）转移到内容空间（原文匹配）**——模型擅长复述见过的内容，不擅长数数。harness 侧再配模糊匹配（容忍空白差异）兜底。这个思想后来被几乎所有编程 agent 继承（Claude Code 的 Edit 工具的 old_string/new_string 是同构设计）。

**失败处理**：如果 SEARCH 块匹配不到，Aider 不是静默丢弃，而是把"哪个块没匹配上+最接近的文件片段"作为错误反馈塞回对话，让模型重试——**编辑失败是对话的一部分，不是异常**。

#### P2 · 上下文选择：repo-map

**子问题**：仓库几十万行，上下文窗口装不下，每轮该给模型看什么？

**根因（M+$）**：全量塞入既不可能也不经济；让模型自己 `ls`/`grep` 探索（agentic search）在 2023 年的模型上太慢太贵。

**Aider 的方案**（[官方博客](https://aider.chat/2023/10/22/repomap.html)，至今仍是其文档核心）：

1. **tree-sitter** 解析全仓库（130+ 语言），抽出每个文件的**符号定义与引用**；
2. 构建有向图：节点是文件/符号，边是"引用"关系；
3. 用 NetworkX 跑 **personalized PageRank**——personalization 向量偏向"当前在聊天里的文件 + 用户消息里提到的标识符"；
4. 按排名贪心选取符号签名，**装进默认约 1k token 的预算**（可调，无文件加入聊天时会自动放大）；
5. 产出一份"地图"：高相关文件只展示关键签名，不展示函数体。

**本质**：这是**静态分析驱动的检索**，与 RAG（embedding 相似度）和 agentic search（让模型自己翻）三足鼎立的第三条路线。它的优势是确定性、零额外推理成本、天然尊重代码结构；劣势是对"语义相关但符号无关"的内容失明。

#### P1/P7 · 极简 loop + 本地验证

Aider 的 loop 极短：用户输入 → LLM 输出编辑 → 应用编辑 → （可选）自动 lint/test → **若 lint/test 失败，错误回灌，触发 reflection 重试（`max_reflections=3` 硬上限）** → git auto-commit。

两个值得注意的硬设计：
- **git 即检查点**：每次 AI 编辑自动 commit（带描述性消息），`/undo` 就是 `git revert`。不自己发明持久化，复用人类已有的最强版本控制——P6 的零成本解法。
- **reflection 上限 3 次**：防 loop 失控的最朴素答案，承认"3 次修不好的，第 4 次也修不好"。

### 2.3 Aider 没解决什么

无沙箱（直接跑在你机器上）、无长期记忆、无多 agent、自治性弱。**architect 模式**（[2024-09 博客](https://aider.chat/2024/09/26/architect.html)）是它对编排的唯一让步：强模型负责推理出改法（自然语言），弱模型负责转成 edit format——双模型流水线，用 $ 换质量，实测显著提分。

### 2.4 可迁移启示

1. **edit format 是 harness 的地基**，比 loop 设计更先决——编辑不可靠，一切上层都是空中楼阁。直接抄 SEARCH/REPLACE 思想。
2. **验证回路要廉价且自动**（lint→test→reflect，带硬上限）。
3. **git 是免费的 checkpoint 系统**，先用它，别自己造。
4. **基准测试驱动开发**：Aider 的每个 edit format 决策都有自家 benchmark（225 题 Exercism polyglot）数据支撑。自建 harness 第一周就该有自己的 20 题小基准。

---

## 三、产品拆解 ② OpenHands —— 沙箱与事件溯源的工业级答案

### 3.1 领域定位

开源自治软件工程 agent 平台（原 OpenDevin，2024-03 起步，All Hands AI 公司化运营），平台论文 [arXiv 2407.16741](https://arxiv.org/abs/2407.16741)。2025 年 11 月完成 **V1/Agent SDK 大重构**（论文 [arXiv 2511.03690](https://arxiv.org/abs/2511.03690)，[官方博客](https://www.openhands.dev/blog/the-path-to-openhands-v1)）。它与 Aider 是光谱两端：完全自治、长任务、必须沙箱。

### 3.2 它主攻的问题域

#### P5 · 沙箱与执行环境

**子问题**：自治 agent 会执行任意 bash/任意代码，怎么隔离？环境怎么按项目定制？远程执行怎么做？

**根因（E）**：自治性与安全性的根本张力——agent 能力越强，爆炸半径越大。且真实软件任务需要"装依赖、跑服务、开浏览器"的全功能环境，不是一个 Python 解释器就够。

**OpenHands 的方案**（V0 经典架构）：

```
AgentController (宿主进程)          Runtime 沙箱 (Docker 容器)
┌──────────────────────┐          ┌──────────────────────────┐
│  agent.step(state)    │ Action  │  Action Execution Server  │
│  LLM 决策下一个动作    │ ───────→ │  (REST/WS 接收动作)        │
│                       │          │  ├── bash (持久 tmux 会话) │
│                       │ Observ. │  ├── IPython (Jupyter)     │
│  事件流追加            │ ←─────── │  ├── 文件读写              │
└──────────────────────┘          │  └── 浏览器 (BrowserGym)   │
                                   └──────────────────────────┘
```

关键机制：
- **agent 决策与动作执行物理分离**：宿主进程里的 controller 只做 LLM 调用；所有副作用动作序列化后发给容器内的 action execution server 执行，观察结果序列化传回。这条边界天然支持把 runtime 换成远程 K8s Pod（其商业版 Remote Runtime 即按此横向扩展）。
- **沙箱是全功能 Linux**：持久 bash 会话（环境变量、cwd、后台进程跨步保持）、Jupyter 内核、Chromium。环境用自定义 Docker 镜像按项目定制。
- **V1 的修正**：V0 把"必须起 Docker"作为默认，开发体验沉重。V1 SDK 改为**默认进程内 `LocalWorkspace`，需要隔离时换 `DockerWorkspace`，agent 代码零改动**——沙箱从"强制架构"降级为"可插拔依赖注入"。这是踩坑后的重要回摆：隔离级别应是部署期决策，不是开发期决策。

#### P1 · Agent Loop：事件溯源（event sourcing）

**子问题**：50+ 步的长任务，loop 的状态怎么表示？怎么重放调试？怎么断点续跑？

**根因（E+M）**：把状态散在内存变量里，崩溃即归零、调试无轨迹；而 LLM 对话本身就是 append-only 的，天然适合事件日志。

**OpenHands 的方案**：**整个会话 = 一条不可变事件流**。每个事件要么是 `Action`（agent 想做什么：CmdRunAction、FileEditAction、BrowseAction…）要么是 `Observation`(世界回了什么：CmdOutputObservation、ErrorObservation…)，统一带 ID、时间戳、来源，类型安全序列化（V1 用 discriminated union）。推论：

- **崩溃恢复 = 重放事件流**（P6 免费获得）；
- **调试 = 读事件日志**（P8 免费获得）；
- **上下文构建 = 事件流的一个视图**（见下）。

loop 控制上：iteration budget（最大步数预算）+ **stuck detection**（检测最近 N 步重复同样的 action/observation 对，判定原地打转则中断）——承认 M 根因：模型会死循环，harness 必须有外部裁判。

#### P2 · 上下文管理：Condenser

**子问题**：事件流会无限增长，LLM 窗口不会。

**方案**：**Condenser 抽象**——上下文不是"对话历史"本身，而是"事件流经过一个压缩器后的投影"。内置策略：滑动窗口截断、LLM 总结压缩（把旧事件折叠成摘要事件插回流中）等，可组合可替换。**把"压缩策略"做成一等抽象而非 loop 里的 if-else**，是它比多数自研 harness 干净的地方。

#### P3 · CodeAct：用代码统一动作空间

OpenHands 的默认 agent（CodeActAgent）来自 [CodeAct 论文（arXiv 2402.01030）](https://arxiv.org/abs/2402.01030)：与其定义 50 个 JSON 工具，不如让 agent 输出**可执行代码（bash/Python）作为统一动作**——代码天然可组合（循环、条件、管道），一步顶 JSON 工具调用十步。这是"工具粒度"光谱上的极端立场，与 mini-swe-agent 殊途同归（见第五节）。

### 3.3 V1 重构的教训（对自建者最值钱的部分）

V0 是单体仓库，agent/工具/沙箱/服务端耦合。V1 拆成四个独立包：**sdk（agent 定义）/ tools（动作处理器）/ workspace（执行环境）/ agent_server（托管 API）**。官方陈述的动机：单体导致测试慢（工具测试拖累核心迭代）、依赖重（嵌入场景只想要 sdk）、发布绑死。——**这意味着从 0 到 1 时你可以直接按这四个边界切包，跳过他们花 18 个月才换来的教训。**

### 3.4 可迁移启示

1. **事件溯源是 agent loop 的最佳状态模型**：一条 append-only 的 Action/Observation 日志，持久化、调试、上下文构建三个问题一次解决。
2. **决策与执行分离**：LLM 调用进程和副作用执行环境之间放一条序列化边界，沙箱/远程/扩容全在这条边界上做文章。
3. **沙箱可插拔，不要强制**：Local 起步，Docker 隔离，接口不变。
4. **压缩器做成一等抽象**（Condenser 模式）。
5. **包边界**：sdk / tools / workspace / server，第一天就这么切。

---

## 四、产品拆解 ③ LangGraph —— 持久化运行时与显式控制流

### 4.1 领域定位

LangChain 旗下的 agent 编排运行时，2025 年 10 月发布 1.0（langchain 1.0 的 agent 抽象也改为基于其运行时构建）。它不是一个"成品 agent"，而是**造 agent 的图运行时**——前两个产品是"住宅"，它是"钢筋水泥"。选它入列，是因为它把 P6（持久化/恢复/人机协同）做成了**运行时原语**而非应用层补丁。

### 4.2 它主攻的问题域

#### P1 · 显式控制流：StateGraph + Pregel

**子问题**：纯 while-loop agent 的控制流全在 prompt 里（隐式、不可静态分析）；企业流程需要"这步之后必须走审批"这种**硬保证**。

**根因（M）**：prompt 是软约束，模型可以不听话；监管/合规场景需要图结构这种硬约束。

**方案**：开发者用 `StateGraph` 显式声明节点（计算单元）与边（含条件边），共享一个带 **reducer** 的类型化 State（如 `messages` 通道用 append 语义合并）。执行引擎是 **Pregel 风格的 BSP（Bulk Synchronous Parallel）模型**：每个 **super-step** 并行执行所有被激活的节点 → 节点写 channel → channel 更新激活下游节点 → 进入下一个 super-step。"agent loop"在这里退化为"图里的一条环形边"——LangGraph 的世界观是：**循环只是图的特例**。

#### P6 · 持久化、时间旅行、人机协同（它的王牌）

**子问题**：长任务跨进程存活？人审批要等三天，线程不能挂着？错了想回滚到第 7 步重来？

**根因（E）**：这些是工作流引擎（Temporal 等）的经典问题，LLM agent 只是新瓶装旧酒。

**方案**：三个咬合的原语——

1. **Checkpointer**（`BaseCheckpointSaver` 接口，SQLite/Postgres/Redis 实现）：**每个 super-step 结束自动快照全部图状态**，按 `thread_id` 归档。
2. **interrupt() / Command**：节点内调用 `interrupt(payload)`，图暂停、状态落盘、进程可以死；三天后客户端带 `Command(resume=value)` 回来，从断点继续——**人机协同 = 持久化的一个推论**，不是独立功能。
3. **Time-travel**：取任意历史 checkpoint，改 state 后 fork 出新分支重放。调试 50 步轨迹时这是杀手级能力。

**诚实的边界**（社区共识，如 Diagrid 的批评文章）：checkpoint 粒度是 super-step，**节点内部不是事务性的**——节点里发了邮件然后崩溃，重放会再发一次。真正的 exactly-once 语义仍需节点内幂等设计。自建者注意：**checkpoint ≠ durable execution 的全部**。

#### P2 · 记忆分层

短期记忆 = thread 内的 State（随 checkpoint 持久化）；长期记忆 = 独立的 **Store 接口**（跨 thread 的 KV+语义检索）。把"对话内状态"和"跨对话知识"在类型上拆开，是值得抄的清晰边界。

#### P8 · 可观测性

与 LangSmith 深度集成：每个 super-step、每次 LLM 调用、每个 state diff 自动成 trace。生态绑定换来开箱即用——自建时可用 OpenTelemetry GenAI 语义约定 + Langfuse 复刻同等能力。

### 4.3 LangGraph 没解决什么

它不回答 P3/P4/P5（工具怎么设计、编辑怎么落盘、沙箱怎么隔离全是你的事）；图的前期设计成本高，对探索式任务（不知道流程长什么样）反而是枷锁——这正是 Anthropic "Building effective agents" 说的 **workflow（已知流程，要可靠）vs agent（未知流程，要自由）** 分界：LangGraph 的甜区在前者，以及"自由 loop 外面包一层硬流程"的混合体。

### 4.4 可迁移启示

1. **持久化要做在运行时层**，每步自动快照，而不是应用层手动 save——HITL、恢复、time-travel 全是它的推论。
2. **State 用类型化 schema + reducer**，别用裸 dict 到处 mutate。
3. **interrupt 原语**：暂停 = 状态落盘 + 进程退出，恢复 = 注入值续跑。自建版本哪怕只支持"暂停到文件"也值得。
4. **节点幂等**是你自己的责任，checkpoint 救不了副作用。

---

## 五、产品拆解 ④ SWE-agent → mini-swe-agent —— ACI 的科学证据与极简反转

这一节是整个拆解的"思想实验"：同一个团队（Princeton），两年内给出了两个方向相反的答案，而两个都是对的——因为模型变了。

### 5.1 SWE-agent（2024）：ACI 是一门设计学科

[NeurIPS 2024 论文（arXiv 2405.15793）](https://arxiv.org/abs/2405.15793)提出 **ACI（Agent-Computer Interface）**：人类有 HCI，agent 需要专门设计的接口——**不是给模型人类的工具，而是给模型为它定制的工具**。

**实验证据**（这是少有的有对照实验的 harness 研究）：
- 直接给 LM 一个 Linux shell（人类接口），它会被冗长输出淹没、被 vim 这类交互式程序卡死；
- 换成定制 ACI 后，SWE-bench 解决率从 RAG 基线的 3.8% 提到 12.5%（GPT-4 时代数据）。

**ACI 四原则与对应机制**：

| 原则 | 机制 | 治什么根因 |
|---|---|---|
| 动作要简洁紧凑 | 少量复合命令（`find_file`、`search_dir`、`edit`），不是全量 bash | M：动作空间越大越易错 |
| 反馈要信息致密 | **文件查看器一次只显示 100 行窗口**（带当前位置/总行数元信息），`scroll_down` 翻页 | M：长输出稀释注意力（context rot） |
| 防错 guardrail | **edit 命令内置 linter**：改完语法不合法 → 拒绝写入并回显错误，让模型先修 | M：错误会级联，第 10 步的错让后 40 步全废 |
| 结果要去噪 | 搜索命中过多时不列全量，只返回"命中 N 个文件 + 提示缩小范围" | M+$：垃圾结果占窗口 |

### 5.2 mini-swe-agent（2025）：把 harness 砍到 100 行

同团队后来发布 [mini-swe-agent](https://github.com/SWE-agent/mini-SWE-agent)：**整个 agent 约 100 行 Python，唯一的工具是 bash（`subprocess.run`），没有专用 file viewer、没有 lint guardrail、没有任何花活**——在 Claude Sonnet 4 级模型上 SWE-bench Verified 跑到 65%+，后续版本报告 >74%，被 Meta/NVIDIA 等用作模型评估基线。

**为什么反转成立**：2024 年的 ACI 精巧设计，本质是在**替弱模型代偿**（它不会翻文件，我给它造翻页器）。2025+ 的模型经过大量 agentic 训练，原生会用 bash 导航仓库、自己控制输出量——代偿层从"加分项"变成"摩擦项"。

**但注意三件没有被反转的事**（自建者最容易误读这里）：
1. **沙箱没有被砍掉**——mini-swe-agent 依然在隔离环境里跑评估；
2. **观察裁剪没有被砍掉**——输出超长仍会截断；
3. **它的场景是基准评估**（一次性、无人值守、不计交互体验），生产级交互产品（如 Claude Code）依然保留 Read/Edit/Grep 等结构化工具，因为权限控制、UI 渲染、编辑可靠性需要结构化接口。

**正确结论不是"harness 无用"，而是：harness 中"代偿模型能力"的部分半衰期极短，"提供模型没有的东西"（隔离、持久化、权限、验证）的部分长期保值。** 这正好回到第〇节的透镜。

### 5.3 可迁移启示

1. 起点就用 **bash + 最小文件工具**，别先造 20 个工具——让基准数据告诉你哪里需要加结构。
2. 但凡加一个定制工具，对照 ACI 四原则自查：紧凑动作？致密反馈？防错？去噪？
3. **guardrail 加在副作用入口**（edit/exec），不是加在 prompt 里。
4. 永远留一条 mini 基线：你的 fancy harness 如果跑不过 100 行 bash-only 基线，说明你的"智能"是负资产。

---

## 六、横向对比矩阵：8 问题域 × 4 产品

| 问题域 | Aider | OpenHands | LangGraph | SWE-agent / mini |
|---|---|---|---|---|
| P1 loop 控制 | 人驱动短 loop + reflection≤3 | 事件流 + 步数预算 + 卡死检测 | 图/super-step，循环=环形边 | 线性 loop + 步数上限 |
| P2 上下文 | repo-map（静态分析+PageRank+token 预算） | Condenser 一等抽象（截断/LLM 摘要） | State schema + checkpoint；长期记忆走 Store | 100 行窗口查看器 / 模型自理 |
| P3 工具/ACI | 无工具，纯 edit format | CodeAct：代码即动作 | 不管（BYO tools） | 定制 ACI（2024）→ 纯 bash（2025） |
| P4 编辑可靠性 | ★ SEARCH/REPLACE + udiff + 失败回灌 | 沙箱内结构化 FileEdit | 不管 | edit+linter 拒绝写入 |
| P5 沙箱 | 无（本地直跑） | ★ Docker runtime + 动作执行服务，V1 可插拔 | 不管 | Docker（评估场景） |
| P6 恢复/持久化 | git auto-commit 即检查点 | 事件流重放 | ★ checkpointer + interrupt + time-travel | 无（一次性任务） |
| P7 验证 | lint/test 自动回灌 | agent 在沙箱里自己跑测试 | 图里显式加验证节点 | 评估 harness 外置 |
| P8 可观测/评估 | 自家 polyglot 基准驱动开发 | 事件日志天然可重放 | LangSmith 全链路 trace | SWE-bench（它定义了评估标准本身） |

★ = 该问题域的业界参考答案。

**横向参照系**（公开工程资料，补足矩阵盲区）：

- **Claude Code / Agent SDK**（Anthropic 工程博客）：loop 公式化为 **gather context → take action → verify work**；上下文靠 **compaction**（接近窗口上限时高保真总结重开窗口）+ **subagents**（子 agent 烧几万 token 探索，只把 1-2k token 的蒸馏结论带回主上下文）+ agentic search 优先于 RAG；验证强调可插拔（lint 规则/截图视觉反馈/LLM-as-judge）。其 2025-11 的 "Effective harnesses for long-running agents" 进一步主张：长任务用**初始化 agent 准备环境与结构化状态文件，编码 agent 增量推进并自验**。
- **Manus**（[创始人博客](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Manus)，2025-07）：生产 agent 第一指标是 **KV-cache 命中率**（缓存与非缓存 token 有 10 倍价差）→ 推论：prompt 前缀必须稳定、上下文只追加不修改、**工具不要动态增删而是用 logits masking 屏蔽**；**文件系统是终极上下文**（无限、持久、agent 可自操作）；**把错误留在上下文里**（模型看见自己的失败才不会重蹈覆辙）；用 todo.md 反复"背诵"目标对抗长任务漂移（recitation）。

---

## 七、最佳实践总纲（按问题域收口）

1. **P1**：loop 本体保持极简（mini-swe-agent 证明 100 行够了）；复杂性放进可插拔组件。外部裁判必须有：步数预算、卡死检测、reflection 上限。
2. **P2**：分层打法——窗口内靠稳定前缀+只追加（保 KV-cache）；接近上限触发 compaction；探索性消耗甩给 subagent 隔离；持久知识落文件系统/Store。**错误轨迹保留，不要清洗**。
3. **P3**：从 bash+读写文件起步；每加一个工具用 ACI 四原则审查；工具集保持稳定（masking 优于增删）。
4. **P4**：SEARCH/REPLACE 内容定位，永远不要行号；编辑失败信息回灌重试。
5. **P5**：执行与决策之间放序列化边界；隔离级别可插拔（Local→Docker→microVM/远程），按部署需求选，开发期不强制。
6. **P6**：先白嫖 git；要跨进程存活就上 step 级自动快照 + interrupt 原语；副作用节点自己保证幂等。
7. **P7**：验证回路决定可靠性上限——lint/test 自动化回灌是最高 ROI 的一条；UI 任务加视觉反馈；主观质量加 LLM-as-judge。
8. **P8**：第一天就建 20 题自有小基准 + 全轨迹日志（事件溯源使其免费）；每个 harness 改动跑基准，不凭感觉迭代。

## 八、从 0 到 1 的建造路线图

```
里程碑 1（~3 天）· mini 骨架
  while-loop + bash 工具 + 步数上限 + Docker 跑
  └── 对标 mini-swe-agent；建 20 题自有基准，记下分数 —— 这是你的地平线

里程碑 2（~1 周）· 编辑与验证
  SEARCH/REPLACE 编辑工具 + 匹配失败回灌 + lint/test 自动验证 + reflection≤3 + git auto-commit
  └── 对标 Aider；基准分应显著上涨

里程碑 3（~2 周）· 事件溯源与上下文
  loop 状态改造为 Action/Observation 事件流 + Condenser 式压缩接口 + compaction + 观察裁剪
  └── 对标 OpenHands；此时崩溃恢复和调试免费获得

里程碑 4（按需）· 运行时硬化
  step 级 checkpoint + interrupt/resume + subagent 上下文隔离 + OTel trace
  └── 对标 LangGraph / Claude Agent SDK；只在真有长任务/HITL/并行需求时做

反模式清单：先设计 20 个工具｜prompt 里塞流程硬约束｜动态增删工具破坏 KV-cache｜
          清洗错误轨迹｜没有基准就调参｜在 LLM 进程里直接 exec
```

---

## 九、主要参考源

- OpenHands：平台论文 [arXiv 2407.16741](https://arxiv.org/abs/2407.16741) · V1 SDK 论文 [arXiv 2511.03690](https://arxiv.org/abs/2511.03690) · [The Path to OpenHands v1](https://www.openhands.dev/blog/the-path-to-openhands-v1) · CodeAct [arXiv 2402.01030](https://arxiv.org/abs/2402.01030)
- Aider：[repo-map 博客](https://aider.chat/2023/10/22/repomap.html) · [repo-map 文档](https://aider.chat/docs/repomap.html) · [architect 模式](https://aider.chat/2024/09/26/architect.html) · [polyglot 基准](https://aider.chat/2024/12/21/polyglot.html) · [GitHub](https://github.com/Aider-AI/aider)
- LangGraph：[GitHub](https://github.com/langchain-ai/langgraph) · [Durable execution 文档](https://docs.langchain.com/oss/python/langgraph/durable-execution) · 1.0 发布（2025-10）· 批评视角：[Diagrid - Checkpoints Are Not Durable Execution](https://www.diagrid.io/blog/checkpoints-are-not-durable-execution-why-langgraph-crewai-google-adk-and-others-fall-short-for-production-agent-workflows)
- SWE-agent：[arXiv 2405.15793](https://arxiv.org/abs/2405.15793)（NeurIPS 2024）· [ACI 文档](https://swe-agent.com/0.7/background/aci/) · [mini-swe-agent](https://github.com/SWE-agent/mini-SWE-agent)
- Anthropic 工程博客：[Building agents with the Claude Agent SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk) · [Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) · [Effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents) · Building effective agents（2024-12）
- Manus：[Context Engineering for AI Agents: Lessons from Building Manus](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Manus)

> 核实说明：以上架构事实均经 2026-06 网络检索交叉核实；少数源码级细节（如 Aider `max_reflections=3`、OpenHands stuck detection 的具体阈值）来自项目源码的既有知识，标记为"机制存在已核实、具体数值以当前源码为准"。
