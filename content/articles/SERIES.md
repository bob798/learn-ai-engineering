# AI Engineering Architecture Series

这个目录用于沉淀 AI 系统架构系列内容，重点关注如何把 AI 能力工程化为可调用、可治理、可观测、可评估的平台能力。

## 内容索引

1. [Skill 模块架构设计方法论](./01-skill-architecture-methodology.md)
2. [如何和 AI 协作完成架构设计](./02-collaborating-with-ai-for-architecture-design.md)
3. [AI 不是不懂，是会"忘"——一次 4 轮迭代的真实复盘](./03-ai-collaboration-rule-decay.md)
4. [记忆不是塞进上下文——短期+长期记忆的实战复盘](./04-short-term-vs-long-term-memory-in-practice.md)
5. [上下文窗口管理——从一个"每轮重复摘要"的 bug 说起](./05-context-window-management.md)

## 系列定位

本系列不只讨论模型调用或 Prompt 编写，而是从 SaaS、平台工程、LLMOps 和安全治理角度，整理 AI 系统在生产环境中需要解决的核心架构问题。

重点主题包括：

- AI 引擎平台架构
- Skill / Agent / Workflow 能力抽象
- Prompt、工具、知识库、模型的运行时编排
- 多租户、权限、审计、成本和限流
- AI 结果的评估、回滚、灰度和可观测性
- AI 能力如何以平台 API 的方式被业务系统稳定调用

## 基本设计原则

- 先定义领域边界，再定义数据结构。
- 先定义输入输出契约，再实现内部逻辑。
- 区分配置态、运行态、观测态。
- 把模型、工具、知识库作为外部依赖，通过接口隔离。
- 不相信 AI 输出天然可靠，必须用 schema、校验、评估和日志工程化。
- 权限、租户隔离、审计、成本控制要前置设计。
