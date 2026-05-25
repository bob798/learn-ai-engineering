---
title: 05 · LLM Foundations 选题路线图
description: 全部选题、优先级、状态、依赖关系
status: v1 · 持续维护
---

# 05-llm-foundations · PLAN

状态图例：`📝 草稿中` `✅ 已发` `⏸️ 暂缓` `🔥 高优先级` `🌱 待孵化`

---

## P0 · 立即开写（差异化最强 + 流量潜力大）

| # | 拟标题 | 子目录 | 状态 | 备注 |
|---|---|---|:-:|---|
| 1 | Inference vs Reasoning · 中文都叫"推理"的术语坑 | 05-terminology | 🔥🌱 | **必爆款候选**，差异化点 #1 |
| 2 | Thinking 模式 vs Strict Output · 为什么互斥 | 04-output-control | 🔥🌱 | 差异化点 #2，配时序图 |
| 3 | Structured Output 三层 · JSON Mode/Schema/Strict | 04-output-control | 🌱 | Constrained Decoding 原理 |
| 4 | Agent vs Agentic · 名词与形容词 | 05-terminology | 🌱 | 短小精悍 |

---

## P1 · 第一批跟进（系统性铺底）

| # | 拟标题 | 子目录 | 状态 | 依赖 |
|---|---|---|:-:|---|
| 5 | Prefill vs Decode · 为什么第一个字慢 | 01-runtime | 🌱 | TTFT/TPOT 概念 |
| 6 | LLM RT 是什么 · Agent 延迟的真正瓶颈 | 01-runtime | 🌱 | 引用 #5 |
| 7 | Chat 模型 vs Reasoning 模型 · 何时用哪个 | 02-architecture | 🌱 | reasoningEffort 参数 |
| 8 | MoE vs Dense · 为什么国产偏爱 MoE | 02-architecture | 🌱 | 显存/带宽前置 |
| 9 | 显存到底装了什么 · 推理 vs 训练 | 01-runtime | 🌱 | 引出 #10 |
| 10 | 激活值算梯度 · 训练为什么贵 4 倍 | 03-training | 🌱 | 依赖 forward/backward |

---

## P2 · 工具型内容（长尾流量）

| # | 拟标题 | 子目录 | 状态 | 备注 |
|---|---|---|:-:|---|
| 11 | 模型命名后缀解码大全（VL/Coder/Distill/A37B…） | 05-terminology | 🌱 | 备查类文章 |
| 12 | Qwen-Max vs Qwen3 · 档位 vs 代际两维度命名 | 06-model-zoo | 🌱 | 蹭 Qwen 关键词 |
| 13 | 国产模型选型表 2026 · 按场景对比 | 06-model-zoo | 🌱 | 定期更新版本 |
| 14 | BFCL 榜单怎么看 · 工具调用真实能力评估 | 06-model-zoo | 🌱 | 引现有评测 |
| 15 | 术语对照表（持续维护） | 05-terminology | 🌱 | glossary.md |

---

## P3 · 进阶（拉拢已入门读者）

| # | 拟标题 | 子目录 | 状态 |
|---|---|---|:-:|
| 16 | Forward Pass vs Backward Pass · Java 工程师视角 | 03-training | 🌱 |
| 17 | RLHF / DPO / RLVR · 模型怎么"被调教"的 | 03-training | 🌱 |
| 18 | Constrained Decoding 原理 · CFG 文法引擎 | 04-output-control | 🌱 |
| 19 | Agentic 数据 · Agent 能力的真正来源 | 03-training | 🌱 |
| 20 | 量化与显存 · INT8/INT4/GGUF 速查 | 01-runtime | 🌱 |

---

## 🚧 依赖关系（写作顺序约束）

```
05/05-terminology/inference-vs-reasoning.md  (P0-#1)
        ↓ 被引用
05/02-architecture/chat-vs-reasoning.md     (P1-#7)

05/01-runtime/vram.md                       (P1-#9)
        ↓ 被引用
05/03-training/activation-and-gradient.md   (P1-#10)
05/02-architecture/moe-vs-dense.md          (P1-#8)

05/04-output-control/structured-output-3-levels.md  (P0-#3)
        ↓ 被引用
05/04-output-control/thinking-strict-conflict.md    (P0-#2)
```

---

## 🎯 验收标准

每篇正文必须满足：
- [ ] 至少 1 个"我以为是 X，结果是 Y"的反转
- [ ] 至少 1 个对比表（不是配图就是表格）
- [ ] frontmatter 标 `verified-on` + `model-versions`
- [ ] 至少 1 个外链（官方文档 / 论文 / GitHub Issue）
- [ ] 字数 1500-3500（不要超 4000，超了拆篇）

---

## 🔄 路线图更新规则

- 每发 1 篇，更新本 PLAN 状态
- 每发 5 篇，回看是否需要补 P3 或调 P0
- 不在 PLAN 里的临时选题→ 先加到本文末尾的"🌱 待孵化"区，至少留一周再决定是否纳入

## 🌱 待孵化

(此处持续记录灵感，不承诺写作)
