---
title: "Papers · 论文解读"
description: 把奠定现代 AI Agent 的关键论文逐节拆开读——不做摘要转述,而是逐句中文注解 + 可运行代码复现
order: 1
---

# Papers · 论文解读

> 论文给**逐句中文注解**，不给摘要转述；能配代码的，配**可运行的复现**。
> 目标不是「读过」，而是能用自己的话讲清楚这篇论文**解决了什么问题、怎么解决、留下了什么坑**。

---

## 已收录

| 论文 | 一句话 | 链接 |
|---|---|---|
| **ReAct** (ICLR 2023) | 定义了现代 Agent 的基本形态：Thought → Action → Observation 循环 | [解读](react-paper.md) · [逐句注解版](../../../interactive/agent/react-paper-annotated.html) · [可运行复现](../../../code/react-hands-on/HANDS_ON.md) |

---

## 延伸

- [ReAct 模式反思](../planning-reasoning/react-paper-reflection.md) — 放在 Planning & Reasoning 专题里，从「三种推理模式」的角度再看 ReAct
- [Agent Loop 的 60 年血脉](../../concepts/agent-loop.md) — ReAct 循环的工程实现与历史脉络

---

## 想加一篇论文解读？

按这个结构写：

1. **一句话定位** — 这篇论文在 Agent 演化里的位置
2. **原文逐节拆解** — 关键段落逐句注解，保留困惑和追问
3. **能跑通** — 有代码的复现核心实验，给完整命令
4. **留下的坑** — 它没解决什么 / 后续论文怎么补
