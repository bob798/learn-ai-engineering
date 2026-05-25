---
title: 06 · Spring AI 选题路线图
description: 全部选题、优先级、状态、Spring AI 版本依赖
status: v1 · 持续维护
---

# 06-spring-ai · PLAN

状态图例：`📝 草稿中` `✅ 已发` `⏸️ 暂缓` `🔥 高优先级` `🌱 待孵化`
默认 Spring AI 版本：`2.0.0-M6`（不同篇可覆盖）

---

## P0 · 立即开写（流量钩子 + 真实踩坑）

| # | 拟标题 | 子目录 | 状态 | 备注 |
|---|---|---|:-:|---|
| 1 | 我用 Spring AI 调 o3，被一个 400 报错教做人 | 01-pitfalls | 🔥🌱 | parallel_tool_calls + reasoning 模型限制 |
| 2 | Spring AI 国产模型适配速查表（DeepSeek/Qwen/GLM/豆包） | 05-domestic-models | 🔥🌱 | 工具型，长尾流量高 |
| 3 | Spring AI 的 entity() 在 DeepSeek 上为什么会翻车 | 04-structured-output | 🌱 | 引申到 strict 兼容性 |
| 4 | 流式响应丢 toolCalls？别 debug 了，是 #3366 | 01-pitfalls | 🌱 | 短文，引官方 issue |

---

## P1 · 工程化基础（系统性补齐）

| # | 拟标题 | 子目录 | 状态 |
|---|---|---|:-:|
| 5 | Spring AI ChatClient · 一行代码背后的 5 层抽象 | 02-core-concepts | 🌱 |
| 6 | Spring AI Advisor 链 · 你的 LLM 中间件该怎么写 | 02-core-concepts | 🌱 |
| 7 | Spring AI ChatMemory · 短期/长期/混合三种模式 | 02-core-concepts | 🌱 |
| 8 | Spring AI VectorStore · 切换 PG/Milvus/Qdrant 的零成本抽象 | 02-core-concepts | 🌱 |

---

## P2 · 工具调用深度

| # | 拟标题 | 子目录 | 状态 |
|---|---|---|:-:|
| 9 | Spring AI @Tool 注解 · 把任意 Java 方法暴露成 LLM 工具 | 03-tool-calling | 🌱 |
| 10 | Spring AI 工具调用串行执行的真相（issue #5195） | 03-tool-calling | 🌱 |
| 11 | 用 CompletableFuture 给 Spring AI 加并发工具执行 | 03-tool-calling | 🌱 |
| 12 | Spring AI + MCP · 5 分钟接入 100 个工具 | 03-tool-calling | 🌱 |

---

## P3 · 国产模型适配（垂直深入）

| # | 拟标题 | 子目录 | 状态 |
|---|---|---|:-:|
| 13 | Spring AI 接 DeepSeek · 配置 + 已知坑全集 | 05-domestic-models | 🌱 |
| 14 | Spring AI 接 Qwen（DashScope）· thinking 与 strict 互斥 | 05-domestic-models | 🌱 |
| 15 | Spring AI 接 GLM-4.6 · 智谱的官方 starter 实测 | 05-domestic-models | 🌱 |
| 16 | Spring AI 接豆包 · 火山引擎 endpoint 配置 | 05-domestic-models | 🌱 |
| 17 | 国产模型 Structured Output 兼容性矩阵（实测） | 04-structured-output | 🌱 |

---

## P4 · 生产化（拉高深度）

| # | 拟标题 | 子目录 | 状态 |
|---|---|---|:-:|
| 18 | Spring AI 重试 + Fallback · LLM 抖动是常态 | 06-production | 🌱 |
| 19 | Spring AI Micrometer 可观测 · Token 成本怎么追踪 | 06-production | 🌱 |
| 20 | Spring AI Prompt Caching · 给国产模型省一半钱 | 06-production | 🌱 |

---

## 🚧 依赖关系

```
06/01-pitfalls/parallel-tool-calls-failures.md   (P0-#1)
        ↓ 引用
05/02-architecture/chat-vs-reasoning.md          (跨主题，05 的 P1)

06/04-structured-output/entity-vs-vendors.md     (P0-#3)
        ↓ 引用
05/04-output-control/structured-output-3-levels.md  (跨主题，05 的 P0)

06/03-tool-calling/parallel-execution-issue-5195.md  (P2-#10)
        ↓ 配套
06/03-tool-calling/completablefuture-wrapper.md      (P2-#11) 给方案
```

跨主题引用是预期的——05 讲"为什么"，06 讲"在 Spring AI 里怎么应对"。

---

## 🎯 验收标准

每篇正文必须满足：
- [ ] frontmatter 标 `spring-ai-version` + `verified-on` + `model-versions`
- [ ] 至少 1 段可 copy-paste 跑通的 Java 代码（不是伪代码）
- [ ] 如果是 bug 类，给最小复现（≤ 20 行）
- [ ] 如果涉及国产模型，至少 2 家实测对比
- [ ] 字数 1500-4000

---

## 🔥 高 ROI 持续工作流

### 「Spring AI 踩坑日记」工作流
1. 日常开发遇到任何 Spring AI 异常 → `01-pitfalls/_inbox/<date>-<slug>.md` 速记
2. 速记格式：
   ```
   ## 现象
   ## Spring AI 版本 / 模型
   ## 复现步骤
   ## 排查过程
   ## 根因
   ## 解决方案
   ```
3. 速记积累 5 个 → 挑 1 个有反转价值的扩写
4. 扩写时**保留排查过程**，不要只写结论——读者真正想看的是"你怎么排查的"

### 「国产模型差异矩阵」持续维护
- 建一个 `05-domestic-models/compatibility-matrix.md`，长表格
- 每接入新厂商 / 新版本，加一行
- 字段：模型 / 版本 / Spring AI starter / parallel_tool_calls / structured output strict / 流式 / 已知坑链接

---

## 🌱 待孵化（不承诺写）

(灵感记录区)
