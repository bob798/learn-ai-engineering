# ATDF · AI 主题拆解框架

> AI Topic Decomposition Framework — 用 8 个维度系统拆解任何 AI 主题

## 为什么需要这个

AI 领域新东西每周冒出来。面对一个新产品/框架/协议/论文，多数人的反应是"收藏→积灰"。

ATDF 强制你回答 8 个维度的具体问题，30 分钟到 3 小时内产出一份**可以判断"值不值得投入时间"**的结构化笔记。

## 8 个维度

| # | 维度 | 核心问题 |
|---|---|---|
| ① | **定位** | 它是什么、替代了什么、没有它怎么做 |
| ② | **架构** | 核心组件、数据流、关键机制、依赖 |
| ③ | **产品** | 谁在用、怎么分发、定价、上手难度 |
| ④ | **业务** | 竞品、护城河、融资、5 年生存预测 |
| ⑤ | **使用** | 10 行最小示例、3 个坑、用/不用场景 |
| ⑥ | **模块拆解** | 选一个核心模块完整展示工作机制 |
| ⑦ | **生态位** | 商业分层位置、平台吞噬风险、个人机会 |
| ⑧ | **实战迁移** | 哪些模式可以迁移到自己的项目 |

## 三档深度

| 档位 | 时间 | 覆盖维度 | 输出 |
|---|---|---|---|
| 🟢 Scan | 15 分钟 | ①②⑦ | 一段话笔记 |
| 🟡 Deep | 1-2 小时 | 全 8 维 | 一页结构化笔记 |
| 🔴 Hands-on | 半天 | 全 8 维 + 代码 | 一页笔记 + 可跑的 repo |

## 附加模块

### 📖 术语速查

每篇笔记开头放一张术语表：英文术语 + 直译 + 本主题含义 + 生活化比喻。先建词汇再读正文。

### Mermaid 架构图

用 Mermaid 画系统总览、数据流、执行流程、商业分层——让架构"看得见"。

## 模板

→ [ATDF 空白模板](../templates/ATDF-template.md)

## 已完成的拆解

| 主题 | 深度 | 链接 |
|---|---|---|
| oh-my-claudecode (OMC) | Deep | [omc-atdf.md](../../02-agent/deep-dives/omc/omc-atdf.md) |
| gstack (Garry Tan) | Deep | [gstack-atdf.md](../../02-agent/deep-dives/gstack/gstack-atdf.md) |
| MemGPT / Letta | 入门指南 | [memgpt-letta-guide.md](../../02-agent/deep-dives/memgpt-letta/memgpt-letta-guide.md) |

## 如何贡献

1. 复制 [ATDF 模板](../templates/ATDF-template.md)
2. 按 8 个维度填写你感兴趣的 AI 主题
3. 提 PR 到 `02-agent/deep-dives/<主题名>/` 目录
