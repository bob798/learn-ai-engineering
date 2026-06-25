---
title: "Context Engineering · 上下文工程"
description: 把无界历史投影进有界、昂贵、可缓存的固定窗口——窗口管理的根因、三难权衡与六大解法谱系
order: 2
---

# Context Engineering · 上下文工程

> Anthropic 把这一整片叫 **Context Engineering**——「在推理时,策划与维护那组最优 token 的策略集合」,比 prompt engineering 更宽(涵盖 system prompt、工具定义、外部数据、对话历史)。
>
> 这个专题聚焦其中最硬的子问题:**把「无界、有状态、会一直增长的历史」,投影进一个「有界、昂贵、且最好能复用(可缓存)的固定窗口」**。它和[记忆专题](../memory/README.md)互为表里——**上下文是「现在」(模型这次推理唯一能看到的 token),记忆是「管理过去如何变成现在」的机制**;本专题讲前者怎么管,记忆专题讲后者怎么存。

---

## 怎么读这个专题

| 角度 | 读什么 | 一句话 |
|---|---|---|
| **实战复盘** | [上下文窗口管理](context-window-management.md) | 从一个「每轮重复摘要」的 bug 说起 · 滚动摘要 / context rot / 开源实践 / 设计取舍 |
| **决策框架** | [上下文压缩的三难与六大解法](context-compaction-tradeoffs.md) | 全量回放 → 三难权衡(保真/成本/稳定)→ 摘要 vs prefix caching → 六大解法家族 → 怎么选 |

两篇互补:**先用决策框架建立地图,再用实战复盘看一个真实 bug 怎么落地。**

---

## 相关内容(强关联)

- [Agent · 记忆专题](../memory/README.md) — 上下文是工作记忆(L1),记忆专题覆盖短期(L2)/长期(L3)与跨会话召回
- [MemGPT / Letta 入门指南](../deep-dives/memgpt-letta/memgpt-letta-guide.md) — 「LLM 即操作系统」:context=RAM,外存=disk,记忆=换页管理器

---

## 一句话结论

上下文管理不是「把更多东西塞进窗口」,而是在 **保真 / 成本 / 稳定** 这个不可能三角里,根据「未来的访问模式」选一个落点。记忆系统之所以存在,正是因为上下文有限且昂贵——**是 context 的有限性,逼出了记忆这门学问。**
