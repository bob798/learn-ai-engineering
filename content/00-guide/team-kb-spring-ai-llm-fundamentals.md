---
title: 团队知识库 · Spring AI 踩坑与 LLM 基础概念问答合集
description: 一场从 Spring AI parallel_tool_calls 报错出发，逐层挖到模型架构、训练范式、术语精度的完整问答记录。可作为团队内部速查与正文素材来源。
status: v1 · 2026-05-11 整理
verified-on: 2026-05-11
model-versions: Spring AI 2.0.0-M6 / OpenAI o3 / DeepSeek V3.1 / Qwen3 / GLM-4.6
---

# 团队知识库 · Spring AI 踩坑与 LLM 基础概念问答合集

> **怎么读这份文档**
> 1. 每节都是"一个真实问题 → 简洁结论 → 必要展开"。优先看结论，需要细节再看展开。
> 2. 每节末尾的「📎 延伸阅读」指向后续将扩写的正文位置（`content/05-llm-foundations/` 或 `content/06-spring-ai/`）。
> 3. 本文档**不是最终文章**——是 Q&A 速记 + 素材库，供团队成员快速对齐 + 后续文章取材。
> 4. 涉及版本号的结论请回看 frontmatter 的 `verified-on`，超 6 个月请重新验证。

---

## 目录

**Part A · Spring AI 工程层**
1. [parallel_tool_calls 在哪些场景下会失败](#1)
2. [parallel_tool_call 到底有什么价值](#2)
3. [Responses API + 内置工具是什么意思](#3)
4. [推理模型为什么默认关闭 parallel](#4)
5. [ReAct 场景适合用什么模型](#5)
6. [国产模型对 parallel_tool_calls 的支持现状](#6)
7. [Structured Output 在国产模型上怎么落地](#7)
8. [Spring AI 流式响应丢 toolCalls 问题](#8)

**Part B · LLM 基础概念层**
9. [LLM RT 是什么](#9)
10. [模型都有哪些类型](#10)
11. [Chat 模型 vs Reasoning 模型](#11)
12. [`reasoningEffort` 是什么](#12)
13. [前向计算 vs 后向计算](#13)
14. [Prefill vs Decode](#14)
15. [激活值算梯度——具体在干什么](#15)
16. [MoE vs Dense](#16)
17. [显存到底装了什么](#17)
18. [量化是什么](#18)
19. [BFCL 等评测榜单怎么看](#19)

**Part C · 训练与术语层**
20. [RL 是什么](#20)
21. [Agent vs Agentic 的区别](#21)
22. [Agentic 数据指什么](#22)
23. [Claude 是什么模式](#23)
24. [Inference vs Reasoning · 中文都叫"推理"](#24)
25. [Strict 模式是什么 / 中文怎么翻译](#25)
26. [为什么 Thinking 模式与 Structured Output 互斥](#26)
27. [Qwen-Max vs Qwen3 的命名区别](#27)
28. [VL / Coder / Distill 等后缀](#28)

**Part D · 学习路径**
29. [面向 Java 工程师的 AI 学习路线](#29)

---

# Part A · Spring AI 工程层

<a id="1"></a>
## 1. parallel_tool_calls 在哪些场景下会失败

**结论**：至少 6 种场景会触发 400 / 静默失败。

| 场景 | 现象 | 根因 |
|---|---|---|
| **使用 reasoning 模型** (o1/o3/o4-mini) 同时打开 `parallel_tool_calls=true` | HTTP 400 `parallel_tool_calls is not supported with this model` | reasoning 模型 API 显式拒绝该参数 |
| 使用 **Responses API + built-in tools**（web_search_preview / file_search 等）同时设 `parallel_tool_calls=true` | 400 | 内置工具不允许并行调用 |
| `tool_choice` 设为 `"required"` 但模型决定只调一个工具 | 行为不一致；某些版本报错 | required + parallel 语义冲突 |
| 模型返回 parallel tool_calls 但客户端逐个串行执行（issue #5195） | 性能不及预期，但**不报错** | Spring AI `DefaultToolCallingManager` 是顺序执行的 |
| 流式响应中工具调用片段丢失（issue #3366） | toolCalls 字段为空 / 部分缺失 | 流式合并逻辑 bug |
| 使用早期 DeepSeek-V2 等不支持 parallel 的国产模型 | 400 或参数被忽略 | 模型本身未支持 |

**📎 延伸阅读**：`06/01-pitfalls/parallel-tool-calls-failures.md`（P0-#1）

---

<a id="2"></a>
## 2. parallel_tool_call 到底有什么价值

**结论**：核心是 **降低 Agent 链路总 RT**，次要是降低 token 成本。

具体收益：
- **场景示例**：用户问"帮我查北京、上海、深圳今天的天气"
  - 关闭 parallel：3 次串行 LLM round-trip × ~3s = 9s
  - 打开 parallel：1 次 LLM round-trip 生成 3 个 tool_call → 3 个工具**应用层并发执行** → 1 次 LLM 汇总 = ~5s
- **token 节省**：少 2 轮 system+context 重复传输
- **副作用**：模型可能误判"应并行"为"应串行"，复杂依赖场景需要降级

**关键边界**：parallel 只是"模型一次返回多个 tool_call"——**实际并发执行是客户端的责任**（Spring AI `DefaultToolCallingManager` 当前是顺序执行，需要自己用 `CompletableFuture` 包装）。

**📎 延伸阅读**：`06/03-tool-calling/parallel-execution-issue-5195.md`（P2-#10）

---

<a id="3"></a>
## 3. Responses API + 内置工具是什么意思

**结论**：OpenAI 在 Chat Completions API 之外推出的**新一代 API**，把"工具"内置到模型侧。

| 维度 | Chat Completions API | Responses API |
|---|---|---|
| 工具来源 | 全部由客户端提供 functions/tools | OpenAI 提供 `web_search_preview`、`file_search`、`code_interpreter` 等内置工具 |
| 工具执行位置 | 客户端 | OpenAI 服务侧 |
| 状态管理 | 客户端维护 messages | 服务端可维护 conversation state |
| 并发工具调用 | 支持（部分模型） | 内置工具**默认禁用 parallel** |

**为什么内置工具不让 parallel**：内置工具是 OpenAI 自己跑的，并发会冲击其内部限速；同时这些工具往往有强依赖（先 web_search 再 file_search），不是真正的"独立任务"。

**📎 延伸阅读**：`05/02-architecture/api-evolution.md`（待规划）

---

<a id="4"></a>
## 4. 推理模型为什么默认关闭 parallel

**结论**：reasoning 模型的"思考"过程**本身就是串行链式推理**，并行工具调用与其内部 CoT 范式冲突。

展开：
- reasoning 模型（o1/o3/DeepSeek-R1）在生成最终回答前会先生成大量**内部 reasoning tokens**（chain-of-thought）
- 这个过程是**严格因果**的——上一步的结论决定下一步思考方向
- 让它"一次性决定调 N 个工具" 等于绕过它的核心能力——它会变成普通 chat 模型
- 所以 OpenAI 直接在 API 层禁止：`parallel_tool_calls + reasoning model = 400`

**对应工程决策**：
- 需要并行工具 → 用 Chat 模型（GPT-4o / Claude Sonnet / DeepSeek-V3）
- 需要深度推理 + 工具 → 用 reasoning 模型，但接受**串行多轮 round-trip**

**📎 延伸阅读**：`05/02-architecture/chat-vs-reasoning.md`（P1-#7）

---

<a id="5"></a>
## 5. ReAct 场景适合用什么模型

**结论**：**Chat 模型优先**，除非任务真的需要深度推理。

判断框架：

| 任务特征 | 推荐 |
|---|---|
| 工具调用频繁、链路长、要 parallel | Chat（GPT-4o / Claude Sonnet 4 / DeepSeek-V3） |
| 单步要做复杂数学/代码/逻辑推导 | Reasoning（o3 / DeepSeek-R1） |
| 既要推理又要工具（混合 Agent） | Reasoning 模型 + 串行 ReAct，或拆成"规划用 reasoning + 执行用 chat"两段 |

**反直觉点**：很多人以为"agent 任务越复杂越该用 reasoning 模型"——实际上**多数 agent 的瓶颈不是单步推理深度，而是工具调用次数 + RT 总和**。reasoning 模型每步思考都更慢，对 agent 总 RT 是放大效应。

**📎 延伸阅读**：`05/02-architecture/chat-vs-reasoning.md`（P1-#7）

---

<a id="6"></a>
## 6. 国产模型对 parallel_tool_calls 的支持现状

**结论**（2026-05 验证）：

| 模型 | parallel_tool_calls | 备注 |
|---|---|---|
| DeepSeek-V3.1 | ✅ 支持 | 兼容 OpenAI 协议 |
| DeepSeek-R1 | ❌ 不支持 | reasoning 模型，与 OpenAI 同 |
| Qwen3 (非 thinking) | ✅ 支持 | DashScope 协议 + OpenAI 兼容 |
| Qwen3 (thinking 模式) | ❌ 不支持 | thinking 与 parallel 冲突 |
| GLM-4.6 | ✅ 支持 | 智谱官方 starter |
| Kimi K2 | ✅ 支持 | |
| 豆包 1.5 / 1.6 | ✅ 支持 | |
| Hunyuan-Large | ⚠️ 部分版本支持 | 需查最新文档 |

**客观评测**：
- **BFCL** (Berkeley Function-Calling Leaderboard) ——业界主流
- **τ-Bench** ——多轮工具调用
- **SuperCLUE-Agent** ——中文专项

国产前列模型在 BFCL parallel/multiple 子项上 2025 下半年已接近 GPT-4o 水平，但**实测**与榜单常有差距——务必自己跑用例。

**📎 延伸阅读**：`06/05-domestic-models/compatibility-matrix.md`（P0-#2）

---

<a id="7"></a>
## 7. Structured Output 在国产模型上怎么落地

**结论**：分清**三层**，国产支持度递减。

| 层级 | 含义 | 国产支持 |
|---|---|---|
| L1 · **JSON Mode** | 强制输出合法 JSON（不保结构） | 几乎全部支持 |
| L2 · **JSON Schema** | 提供 schema 让模型"参考" | 大部分支持，但不保严格符合 |
| L3 · **Strict Mode** | 服务侧 constrained decoding，**100% 符合 schema** | 仅 OpenAI / 部分 DeepSeek / Qwen3 部分版本 |

**Spring AI `entity()` 在国产模型上为什么会翻车**：
- `entity()` 底层依赖 strict 级 JSON 输出
- 当国产模型只有 L1/L2 支持时，会返回**不完全合规的 JSON**（多字段/少字段/类型错）
- Jackson 反序列化失败 → 异常抛出

**应对**：
- 用 `BeanOutputConverter` 但**加重试 + 自校验**
- Prompt 里写死 schema，加 few-shot 示例
- 关键场景**降级到字符串 + 后置解析**

**📎 延伸阅读**：
- `05/04-output-control/structured-output-3-levels.md`（P0-#3）
- `06/04-structured-output/entity-vs-vendors.md`（P0-#3）

---

<a id="8"></a>
## 8. Spring AI 流式响应丢 toolCalls 问题

**结论**：不是用错，是 Spring AI 的 bug（[issue #3366](https://github.com/spring-projects/spring-ai/issues/3366)）。

现象：
- 用 `ChatClient.stream()` + 工具调用
- 累积 chunk 时 `toolCalls` 字段拼接逻辑有问题
- 最终聚合的 `ChatResponse` 中 `toolCalls` 为 null 或不完整

临时应对：
- 关键链路**避免流式 + 工具**，改用 block
- 或自己实现 Flux 聚合 logic，绕过 Spring AI 默认聚合器

**📎 延伸阅读**：`06/01-pitfalls/streaming-toolcalls-lost.md`（P0-#4）

---

# Part B · LLM 基础概念层

<a id="9"></a>
## 9. LLM RT 是什么

**结论**：Round-Trip Time——一次完整的"输入 prompt → 拿到完整输出"耗时。

拆解：
```
LLM RT = TTFT (Time To First Token) + (Output_tokens - 1) × TPOT (Time Per Output Token)
```

- **TTFT**：首字延迟，主要由 **prefill 阶段** 决定（compute-bound）
- **TPOT**：每个后续 token 的延迟，由 **decode 阶段** 决定（memory-bandwidth-bound）

**对 Agent 的意义**：Agent 总 RT = Σ(每步 LLM RT + 工具 RT)。**LLM RT 才是大头**，工具 RT 通常 < 1s 但 LLM 一次 1-5s。

**📎 延伸阅读**：`05/01-runtime/llm-rt.md`（P1-#6）

---

<a id="10"></a>
## 10. 模型都有哪些类型

**结论**：从应用工程师视角，有用的分类是这些：

### 按"输出范式"
- **Chat 模型**：输入→直接输出（GPT-4o, Claude 3.5/4, Qwen3, DeepSeek-V3）
- **Reasoning 模型**：输入→先生成内部 CoT→再输出（o1, o3, o4-mini, DeepSeek-R1, Qwen3-thinking）

### 按"模态"
- 纯文本（多数）
- VL / 多模态（GPT-4o, Qwen3-VL, Claude）
- Coder 专用（Qwen3-Coder, DeepSeek-Coder）

### 按"架构"
- Dense（传统稠密，全部参数都参与每个 token 计算）
- MoE（Mixture of Experts，每个 token 只激活部分专家）

### 按"训练阶段产物"
- Base（预训练后未对齐）
- Instruct / Chat（SFT + RLHF 后）
- Distill（蒸馏版）
- Thinking / Reasoning（带 reasoning trace 训练）

**📎 延伸阅读**：`05/05-terminology/naming-suffix-decoder.md`（P2-#11）

---

<a id="11"></a>
## 11. Chat 模型 vs Reasoning 模型

**结论**：**思考过程是否被显式拉到训练目标里**，是核心区别。

| 维度 | Chat | Reasoning |
|---|---|---|
| 训练 | SFT + RLHF | 在 RLHF 基础上做大规模 **RLVR**（带可验证奖励的强化学习） |
| 推理时 | 直接生成 answer | 先生成 reasoning tokens（CoT），再生成 answer |
| API 参数 | temperature, top_p... | 额外 `reasoning_effort` (low/med/high) |
| 价格 | 单价低 | 单价高 + reasoning tokens 也计费 |
| 适合场景 | 工具调用密集、对话、生成 | 数学、代码、长链推导 |

**反直觉**：reasoning 模型**不是更"聪明"**——是把"思考"显性化、训练化。对于不需要深度推理的任务，它只是**更慢更贵**。

**📎 延伸阅读**：`05/02-architecture/chat-vs-reasoning.md`（P1-#7）

---

<a id="12"></a>
## 12. `reasoningEffort` 是什么

**结论**：reasoning 模型独有的参数，控制模型"思考多久"。

取值：`low` / `medium` / `high`

| effort | reasoning tokens 量 | 用途 |
|---|---|---|
| low | 数百 | 简单分类、提取 |
| medium | 几千 | 一般编码、分析 |
| high | 数万 | 数学竞赛、复杂证明 |

**计费陷阱**：reasoning tokens 也按输出 token 计费——high effort 一次可能花数美元。

**Spring AI 配置**：`OpenAiChatOptions.builder().reasoningEffort("medium")`

**📎 延伸阅读**：`05/02-architecture/chat-vs-reasoning.md`（P1-#7）

---

<a id="13"></a>
## 13. 前向计算 vs 后向计算

**结论**：
- **Forward Pass（前向）**：input → layer1 → layer2 → ... → output。**推理时只做这一步**。
- **Backward Pass（后向）**：output 的 loss → 反向传播 → 计算每层参数的梯度。**只有训练才做**。

Java 工程师类比：
- Forward = 调用栈从 controller 一路 call 到 dao 然后返回
- Backward = 假设你能"从返回值反推每层方法该改哪个 if-else"

**为什么重要**：
- 推理只跑 forward → 显存只需要"参数 + 输入 + 中间激活的瞬时值（可丢弃）"
- 训练要跑 backward → 必须**保留 forward 的所有激活值**用于计算梯度 → 显存爆涨 4-5 倍

**📎 延伸阅读**：`05/03-training/forward-backward.md`（P3-#16）

---

<a id="14"></a>
## 14. Prefill vs Decode

**结论**：LLM 推理两个阶段，性能特性完全不同。

| | Prefill | Decode |
|---|---|---|
| 干什么 | 把整个 prompt 一次性过一遍模型，得到 KV cache | 逐个生成 output token |
| 并行度 | 整个 prompt 并行 | 强串行（一次一个 token） |
| 瓶颈 | **算力**（compute-bound） | **显存带宽**（memory-bandwidth-bound） |
| 决定的延迟 | TTFT | TPOT |

**为什么第一个字慢**：prefill 把 1000 个 prompt tokens 一次过完，可能 500ms-2s；之后每个 token 只要几十 ms。

**工程意义**：
- 长 prompt 优化 TTFT → 减 prompt 长度、用 prompt cache
- 长 output 优化 TPOT → 用更小的模型 / 投机解码 / vllm 等推理框架

**📎 延伸阅读**：`05/01-runtime/prefill-vs-decode.md`（P1-#5）

---

<a id="15"></a>
## 15. 激活值算梯度——具体在干什么

**结论**：

**激活值（Activation）**：每一层的输入经过该层计算后的**输出**。Forward 时产生。

**梯度（Gradient）**：loss 对每个参数的偏导数。Backward 时计算。

**关系**：根据链式法则，**计算第 N 层参数的梯度，需要用到第 N 层的激活值**。所以训练时必须**把 forward 的所有激活值保留在显存里**，等 backward 用。

显存占比（70B 模型 batch=1 训练近似）：
- 参数：140GB（FP16）
- 梯度：140GB
- 优化器状态（Adam）：560GB
- 激活值：随序列长度 × batch 暴涨，常常 100-500GB

**这就是为什么训练贵 4-5 倍显存**。

**省显存技巧**：
- Gradient Checkpointing：丢掉中间激活，backward 时重新 forward 一遍换显存
- ZeRO / FSDP：参数/梯度/优化器分片到多卡
- LoRA：只训练小矩阵，参数 + 梯度 + 激活都大幅缩小

**📎 延伸阅读**：`05/03-training/activation-and-gradient.md`（P1-#10）

---

<a id="16"></a>
## 16. MoE vs Dense

**结论**：
- **Dense**：每个 token 都经过**全部参数**计算
- **MoE**：模型由 N 个专家组成，每个 token 只路由到**部分专家**（典型 top-2 / top-8）

**典型 MoE 标注**：`DeepSeek-V3 671B-A37B`
- 总参数 671B
- 每 token 实际激活 37B

**为什么国产偏爱 MoE**：
- 推理时只算激活参数 → **算力需求按 37B 算**，而不是 671B
- 但显存仍需装下全部 671B（参数都得在 GPU 上）→ **显存成本高，但推理速度快**
- 训练复杂度高（路由负载均衡）但中国厂商已突破

**对应概念**：
- Dense → 全员加班
- MoE → 分工 + 路由调度，每人专精一块

**📎 延伸阅读**：`05/02-architecture/moe-vs-dense.md`（P1-#8）

---

<a id="17"></a>
## 17. 显存到底装了什么

**结论**：

**推理时**：
```
显存 ≈ 模型参数 + KV cache + 当前激活值
```
- 70B FP16 参数 ≈ 140GB
- KV cache 随上下文长度暴涨

**训练时**：
```
显存 ≈ 参数 + 梯度 + 优化器状态 + 激活值 + 临时 buffer
```
- 总和约 5-8 倍参数大小（FP16 训练 Adam）

**HBM 带宽**：显存到算力单元的传输速度。H100 HBM3 约 3TB/s。decode 阶段瓶颈就在这——每生成一个 token，要把全部参数从 HBM 读到 SM 至少一遍。

**📎 延伸阅读**：`05/01-runtime/vram.md`（P1-#9）

---

<a id="18"></a>
## 18. 量化是什么

**结论**：把 FP16 / BF16 的参数压成更小的整数类型，省显存 + 提速。

| 类型 | 精度 | 显存压缩比 | 质量损失 |
|---|---|---|---|
| FP16 / BF16 | 16 bit | 1x | baseline |
| INT8 | 8 bit | 2x | 几乎无 |
| INT4 / GGUF Q4 | 4 bit | 4x | 轻微 |
| INT2 | 2 bit | 8x | 明显 |

工程意义：消费级 GPU 跑 70B 模型靠的就是 INT4 量化。

**📎 延伸阅读**：`05/01-runtime/quantization.md`（P3-#20）

---

<a id="19"></a>
## 19. BFCL 等评测榜单怎么看

**结论**：

| 榜单 | 测什么 | 用法 |
|---|---|---|
| **BFCL** (Berkeley Function-Calling) | 工具调用（single/multiple/parallel/relevance） | 国产模型最有用的参考 |
| **τ-Bench** | 多轮工具调用 + 用户模拟 | 接近真实 agent 场景 |
| **SuperCLUE-Agent** | 中文 agent 综合 | 中文场景 |
| **MMLU / GPQA** | 通用知识 / 科学问答 | 看模型"博学度" |
| **HumanEval / SWE-Bench** | 代码 | Coder 模型必看 |

**关键提醒**：**榜单仅供初筛**。同分模型在自己业务上常常差异巨大——务必自己跑 10-20 个真实 case。

**📎 延伸阅读**：`05/06-model-zoo/bfcl-and-friends.md`（P2-#14）

---

# Part C · 训练与术语层

<a id="20"></a>
## 20. RL 是什么

**结论**：Reinforcement Learning，强化学习。

LLM 后训练阶段用到的几种：

| 缩写 | 全称 | 用法 |
|---|---|---|
| **RLHF** | RL from Human Feedback | 人类标注偏好 → 训练 reward model → 用它指导模型 |
| **RLAIF** | RL from AI Feedback | 用更强的 AI 代替人类标注 |
| **RLVR** | RL with Verifiable Rewards | 能验证对错的任务（数学/代码）直接用规则给奖励 |
| **DPO** | Direct Preference Optimization | 跳过 reward model，直接用偏好数据训练 |
| **PPO** | Proximal Policy Optimization | RL 算法（OpenAI RLHF 早期用的） |
| **GRPO** | Group Relative Policy Optimization | DeepSeek-R1 用的算法 |

**为什么 reasoning 模型崛起**：RLVR 让数学/代码能力可被低成本大规模优化——不需要人类标注，规则就能给奖励信号。

**📎 延伸阅读**：`05/03-training/rlhf-dpo-rlvr.md`（P3-#17）

---

<a id="21"></a>
## 21. Agent vs Agentic 的区别

**结论**：
- **Agent**（名词）：一个具体的系统/实体——"调 LLM + 工具 + 记忆 + 规划的程序"
- **Agentic**（形容词）："具备 agent 特性的"——可作用于数据、工作流、能力等任何名词

例：
- "我们在做一个 Agent" → 在做一个具体系统
- "我们在准备 agentic 数据" → 准备用来训练 Agent 能力的数据
- "agentic workflow" → 自带规划/调度/反思能力的工作流

**为什么这个区别重要**：中文经常把两个词都翻成"智能体"——但语义层级完全不同。Agentic 是"特性"，Agent 是"实体"。

**📎 延伸阅读**：`05/05-terminology/agent-vs-agentic.md`（P0-#4）

---

<a id="22"></a>
## 22. Agentic 数据指什么

**结论**：用于训练模型"具备 agent 能力"的数据，**核心是 trajectory（轨迹）**。

典型 trajectory：
```
user: 帮我查北京天气并对比上海
assistant: <think>需要调 weather 工具两次</think>
           tool_call: weather(city="北京")
tool: {"temp": 18, ...}
assistant: tool_call: weather(city="上海")
tool: {"temp": 22, ...}
assistant: 北京 18°C，上海 22°C，上海高 4°C
```

整段都是训练数据——模型学的是**何时调工具、调什么、如何整合结果、如何输出**。

**为什么 agentic 能力难训**：
- 高质量轨迹稀缺，需要 expert agent 生成
- 工具的真实反馈难以模拟
- 长链路上下文窗口压力大

**📎 延伸阅读**：`05/03-training/agentic-data.md`（P3-#19）

---

<a id="23"></a>
## 23. Claude 是什么模式

**结论**：Claude 是 **Chat 模型 + 可选 extended thinking 模式**的混合。

- Sonnet 4 / Opus 4：默认 Chat 模式，可开启 `thinking` 参数进入显式 reasoning
- 没有像 OpenAI 那样把"chat 系列"和"o 系列"完全分开
- thinking 量可控（类似 reasoning_effort）

**架构据公开信息推测**：Dense（非 MoE）。Anthropic 历来对架构细节保密。

**📎 延伸阅读**：`05/02-architecture/claude-architecture.md`（待规划）

---

<a id="24"></a>
## 24. Inference vs Reasoning · 中文都叫"推理"

**结论**：这是中文社区最大的术语坑。

| 英文 | 含义 | 中文最佳翻译 |
|---|---|---|
| **Inference** | 模型 forward 一次得到输出的**计算过程** | **推断 / 推理（运行时义）** |
| **Reasoning** | 模型"思考"，生成 CoT 链做逻辑推导的**能力** | **推理（能力义）** |

例：
- "推理速度慢" → 一般指 inference（计算太慢）
- "推理能力强" → 一般指 reasoning（逻辑能力）
- "推理模型" → reasoning models（能力含义）

**为什么是大坑**：
- "推理模型推理慢" 中两个"推理"是不同英文词
- 一个是模型分类（reasoning），一个是运行时（inference）

**如何避免**：
- 写文档时**首次出现都标英文**：推理（reasoning）、推断（inference）
- 团队内统一约定：能力义用"推理"，过程义用"推断"或"前向"

**📎 延伸阅读**：`05/05-terminology/inference-vs-reasoning.md`（**P0-#1，本主题第一篇必爆款**）

---

<a id="25"></a>
## 25. Strict 模式是什么 / 中文怎么翻译

**结论**：

- **英文**：Strict
- **中文翻译**：「严格模式」/「严约束输出」
- **含义**：服务侧用 **constrained decoding**（受约束解码）强制每一步生成的 token 必须符合给定 JSON Schema

实现机制：
- 把 JSON Schema 编译成 **CFG（上下文无关文法）**
- decode 每步时，根据当前生成状态计算"允许的下一个 token 集合"
- 对模型 logits 做 mask——不在允许集合的 token 概率置 -inf
- 因此**100% 保证输出符合 schema**

**支持情况**：
- OpenAI：`response_format: {type: "json_schema", strict: true}`
- DeepSeek：部分支持
- Qwen3：部分支持
- 大多数国产模型 only L1/L2，需要后置校验

**📎 延伸阅读**：`05/04-output-control/structured-output-3-levels.md`（P0-#3）

---

<a id="26"></a>
## 26. 为什么 Thinking 模式与 Structured Output 互斥

**结论**：两者都要**接管 decode 阶段**，而 decode 只有一份控制权。

具体冲突：
- **Thinking** 要求：模型先生成 reasoning tokens（思考过程），再生成 answer。reasoning 阶段输出**自然语言**，不是 JSON
- **Strict** 要求：从第一个 token 起就走 schema 文法引擎，每步 logit mask 到 JSON 允许的 token

如果硬要叠加，会出现：
- thinking 阶段被 strict 引擎卡住——"think 这两个字不在 schema 里" → 模型说不下去
- 或反过来：thinking 阶段绕过 schema → 最终输出不合 schema → strict 失效

所以厂商 API 层直接禁止两者同开：
- Qwen3 thinking 模式：禁用 strict output
- OpenAI o1/o3：本来就不支持 strict json schema
- 其他厂商类似

**应对**：
- 需要思考 + 结构化 → 用 reasoning 模型 + JSON Mode（L1）+ 后置 schema 校验
- 或两段式：第一段 reasoning 模型生成思考，第二段 chat 模型把结果格式化

**📎 延伸阅读**：`05/04-output-control/thinking-strict-conflict.md`（P0-#2）

---

<a id="27"></a>
## 27. Qwen-Max vs Qwen3 的命名区别

**结论**：阿里 Qwen 命名是**两个维度叠加**——容易混淆。

### 维度 A · 档位（参考阿里云体系）
- Qwen-**Max**：旗舰，能力上限
- Qwen-**Plus**：均衡
- Qwen-**Turbo**：速度优先
- Qwen-**Flash**：极速 / 低价

档位是 **DashScope 接入名**——商业产品视角。

### 维度 B · 模型代际
- Qwen / Qwen1.5 / Qwen2 / Qwen2.5 / **Qwen3** ：开源模型版本号

代际是 **开源/学术视角**——模型架构、训练数据迭代版本。

### 两者关系
- Qwen-Max 在不同时期会"切到不同代际"——例如 2024 年的 Qwen-Max 可能是基于 Qwen2.5；2025 后切到 Qwen3
- 开源社区下载到的 `Qwen3-72B-Instruct`，对应阿里云的某个 Max/Plus 档位
- **同名不同实**：用 API 调用 Qwen-Max ≠ 用 vllm 跑 Qwen3-72B

**应对**：
- 看模型卡 / 文档里**显式标注的代际号**
- 关注 frontmatter 里的 `model-versions` 字段，避免引用过期信息

**📎 延伸阅读**：`05/06-model-zoo/qwen-naming.md`（P2-#12）

---

<a id="28"></a>
## 28. VL / Coder / Distill 等后缀

**结论**：常见后缀速查。

| 后缀 | 全称 | 含义 |
|---|---|---|
| **VL** | Vision-Language | 视觉 + 语言多模态 |
| **Coder** | - | 代码专项微调 |
| **Math** | - | 数学专项 |
| **Instruct** | - | SFT + 对齐后的版本（能跟指令） |
| **Chat** | - | 同 Instruct，部分厂商用此名 |
| **Base** | - | 预训练完未对齐（开发者用） |
| **Distill** | Distillation | 蒸馏小模型 |
| **MoE** | Mixture of Experts | MoE 架构 |
| **A37B** | Activated 37B | MoE 激活参数量 |
| **AWQ / GPTQ / GGUF** | - | 量化格式 |
| **Pro / Max / Plus / Turbo / Flash / Nano** | - | 商业产品档位 |

**📎 延伸阅读**：`05/05-terminology/naming-suffix-decoder.md`（P2-#11）

---

# Part D · 学习路径

<a id="29"></a>
## 29. 面向 Java 工程师的 AI 学习路线

**结论**：四层金字塔，从下往上。

```
                  ┌────────────────────┐
                  │  L4 · 训练与 RL     │  (可选，做研究才学)
                  ├────────────────────┤
                  │  L3 · 模型架构      │   MoE / Reasoning / 多模态
                  ├────────────────────┤
                  │  L2 · 运行时        │   prefill/decode / 显存 / RT
                  ├────────────────────┤
                  │  L1 · API + 工程    │   Spring AI / MCP / Agent 模式
                  └────────────────────┘
```

推荐路径：
1. **先打通 L1**：能用 Spring AI 调 OpenAI + 国产模型，会工具调用 + RAG + 简单 Agent
2. **碰到坑追问 L2/L3**：当 parallel_tool_calls 报错时去理解 reasoning 模型；当模型慢时去理解 prefill/decode
3. **L4 一般不需要**——除非要做模型微调

**误区**：
- ❌ 从"transformer 架构图"开始学——离应用太远，劝退率高
- ❌ 啃论文——读完忘——**遇到具体问题时再读对应论文章节**
- ✅ **以工程问题为锚点，向下挖深度**

**📎 本知识库的呈现**：
- L1 → `01-mcp/` `02-agent/` `03-rag/` `04-ai-programming/` `06-spring-ai/`
- L2-L3 → `05-llm-foundations/`
- L4 → `05/03-training/`（少量）

---

# 附录 · 后续维护建议

## 这份文档的定位
- ✅ **团队内速查 + 文章素材库**
- ❌ **不是面向公众的文章**——节奏太密、跳跃太大

## 谁来更新 / 何时更新
- 每场类似的"深度追问对话"结束后，参与人**补一节**到本文档
- 每发表一篇正文，把对应章节的 📎 链接更新为已发布状态
- **季度回看**：哪些章节被引用次数最多 → 优先扩成完整正文

## 与 PLAN.md 的关系
- 本文档 = 内容池
- `05/PLAN.md` `06/PLAN.md` = 文章路线图
- 每个 📎 链接都是从内容池捞素材的指针

## 验证机制
- 涉及版本号 / API 行为的章节，每 3 个月跑一次实测
- 实测脚本统一放 `code/spring-ai/verification/`（待建）
- 失效内容**前面加 ⚠️ 标记**，不直接删除（保留误解的痕迹）
