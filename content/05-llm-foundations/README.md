---
title: 05 · LLM Foundations · 应用工程师必懂的"中间层"
description: 不教"什么是 LLM"，不教"怎么搭 RAG"，专注模型本身的工作原理、术语精度与选型判据
status: v1 · 主题初建 2026-05-11
---

# 05 · LLM Foundations

> **这个主题不教什么**：
> ❌ "什么是大模型" / "GPT 是什么" — 太基础，wiki 已有
> ❌ "怎么搭 RAG / 怎么写 Prompt" — 见 `03-rag/` 和 `04-ai-programming/`
>
> **这个主题专注什么**：
> ✅ LLM 应用工程师**每天都在用、但没人系统讲清楚**的"中间层"概念
> ✅ 中文社区**术语翻译错位**导致的认知坑
> ✅ 选型时**真正决定决策**的底层判据（不是榜单分数）

---

## 🎯 受众画像

- 已经能调通 OpenAI/Claude/DeepSeek 的 API
- 能跑起来 LangChain / Spring AI 的 Hello World
- **但**：被 reasoning 模型 400、被 thinking 与 strict 互斥、被 MoE 的"671B 跑得飞快"困惑过
- **想知道**：这些现象背后的真正机制，而不是抄答案

---

## 🗺️ 子分类

| 目录 | 关注 | 代表问题 |
|---|---|---|
| `01-runtime/` | 模型运行时机制 | prefill 和 decode 谁更慢？为什么第一个字慢？ |
| `02-architecture/` | 模型架构 | MoE 和 Dense 区别？为什么 671B 跟 37B 一样快？ |
| `03-training/` | 训练范式 | 为什么训练比推理贵 4 倍显存？RLHF/DPO/RLVR 区别？ |
| `04-output-control/` | 输出约束机制 | Structured Output 三层？为什么 thinking 与 strict 互斥？ |
| `05-terminology/` | 术语精度 | Inference vs Reasoning 都翻译成"推理"怎么办？Agent vs Agentic？ |
| `06-model-zoo/` | 模型选型 | 国产模型怎么选？BFCL 怎么看？命名后缀解码 |

> 子目录会**按需创建**——有文章时才 mkdir，避免空目录污染站点构建。

---

## 📚 推荐阅读路径

### 路径 A · 术语先行（最适合通用读者）
1. 《Inference vs Reasoning · 中文都叫"推理"的术语坑》（`05-terminology/`）
2. 《Agent vs Agentic · 名词与形容词的鸿沟》（`05-terminology/`）
3. 《模型命名后缀解码大全》（`05-terminology/`）

### 路径 B · 架构溯源（适合做选型的）
1. 《Chat 模型 vs Reasoning 模型 · 何时用哪个》（`02-architecture/`）
2. 《MoE vs Dense · 为什么国产偏爱 MoE》（`02-architecture/`）
3. 《国产模型选型表 2026》（`06-model-zoo/`）

### 路径 C · 性能直觉（适合做工程优化的）
1. 《prefill vs decode · 为什么第一个字慢》（`01-runtime/`）
2. 《显存到底装了什么》（`01-runtime/`）
3. 《激活值算梯度 · 训练为什么贵 4 倍》（`03-training/`）

---

## 📊 当前进度

| 路径 | 已发布 | 草稿 | 规划 |
|---|:-:|:-:|:-:|
| 01-runtime | 0 | 0 | 4 |
| 02-architecture | 0 | 0 | 3 |
| 03-training | 0 | 0 | 4 |
| 04-output-control | 0 | 0 | 3 |
| 05-terminology | 0 | 0 | 3 |
| 06-model-zoo | 0 | 0 | 3 |

完整选题清单见 [PLAN.md](./PLAN.md)。

---

## ✍️ 写作风格（继承根 README）

- **保留误解**：每篇至少一个"我以为是 X，结果是 Y"
- **保留追问**：核心概念 5+ 轮深度，不做 wiki 概括
- **保留时效**：frontmatter 标 `verified-on` + `model-versions`
- **首选证据**：能引官方文档/源码就引，避免凭记忆
- **配图节制**：能用对比表说清就别画图，要画就画时序/结构图

---

## 🔗 与其他主题的边界

- 涉及 **Agent 系统设计模式**（ReAct、Plan-Execute）→ 写在 `02-agent/`
- 涉及 **MCP 协议** → 写在 `01-mcp/`
- 涉及 **检索增强** → 写在 `03-rag/`
- 涉及 **Spring AI 具体踩坑** → 写在 `06-spring-ai/`，本主题只讲通用机制
