# Skill 模块架构设计方法论

## 1. Skill 的核心定义

在 SaaS AI 引擎平台中，`Skill` 不应该只是 Prompt 模板。更合适的定义是：

> Skill 是一个可被 AI 平台调用的能力单元，包含输入输出契约、执行策略、依赖资源、权限边界、版本管理、评估机制和运行日志。

Skill 可以包装：

- Prompt 模板
- 模型参数
- RAG 检索能力
- 外部工具或 API
- 多步骤 Agent 行为
- 工作流编排
- 人工审批节点
- 输出解析与校验逻辑

因此，Skill 的目标不是“管理一段提示词”，而是把不稳定的 AI 能力封装成 SaaS 系统里可调用、可治理、可追溯、可评估的稳定能力。

## 2. 通用架构设计方法论

设计这类 AI 平台模块时，推荐采用下面的顺序：

```text
1. 定义业务边界
2. 建立领域模型
3. 明确模块契约
4. 设计运行时链路
5. 设计生命周期
6. 设计权限和治理
7. 设计可观测性
8. 设计评估和发布机制
9. 最后再设计表结构、接口和技术实现
```

核心原则是：

> 先定义边界，再定义契约，再定义运行时，再定义治理能力，最后才选技术实现。

不要一开始就陷入数据库表、Prompt 模板或某个模型 SDK。AI 系统真正复杂的地方通常不是“怎么调用模型”，而是“如何让 AI 能力在生产环境里稳定、可控、可追溯地运行”。

## 3. 方法论来源

这类设计通常会综合多种成熟工程方法。

### 3.1 领域驱动设计 DDD

先识别业务领域里的核心对象、生命周期和边界，而不是先设计表。

Skill 场景中的领域对象可能包括：

```text
Skill
SkillVersion
SkillRun
ToolBinding
KnowledgeBinding
ModelConfig
PermissionPolicy
EvaluationConfig
ExecutionLog
```

需要先回答：

```text
Skill 是实体还是配置？
SkillVersion 是否是独立实体？
SkillRun 是执行记录、审计日志，还是事件？
Tool 是 Skill 内部对象，还是外部平台能力？
KnowledgeBase 是 Skill 管理，还是由知识库模块管理？
```

如果领域语言不清晰，后续代码会逐渐变成混乱的配置拼接。

### 3.2 契约优先设计

Skill 会被多个调用方使用：

```text
Chat
Agent
Workflow
Open API
后台测试面板
业务系统
```

所以需要先定义稳定契约：

```text
Skill 创建契约
Skill 发布契约
Skill 执行契约
Skill 输出契约
Skill 错误契约
Skill 权限契约
```

一个典型执行请求可以是：

```json
{
  "tenant_id": "t_001",
  "skill_code": "customer_summary",
  "version": "latest",
  "input": {
    "customer_id": "c_001"
  },
  "context": {
    "user_id": "u_001",
    "workspace_id": "w_001"
  }
}
```

典型返回结果：

```json
{
  "run_id": "run_001",
  "status": "success",
  "output": {
    "summary": "..."
  },
  "usage": {
    "tokens": 1234,
    "cost": 0.02,
    "latency_ms": 1800
  },
  "trace_id": "trace_001"
}
```

契约稳定后，内部实现可以不断演进，外部调用方不用频繁改造。

### 3.3 分层架构

Skill 平台应避免一个大服务包办所有职责。推荐拆成几个清晰层次：

```text
管理面：Skill 创建、编辑、版本、发布
运行面：Skill 执行、模型调用、工具调用、输出校验
治理面：权限、租户隔离、审计、限流、成本
观测面：日志、trace、指标、错误分析
评估面：测试集、评分、A/B、人工反馈
```

这可以避免 `SkillService` 同时负责 CRUD、Prompt 渲染、模型调用、权限校验、日志计费和效果评估。

### 3.4 六边形架构 / 端口适配器

Skill Runtime 不应该直接绑定具体模型供应商、向量库或业务 API。

它应该依赖抽象接口：

```text
ModelProvider
ToolExecutor
KnowledgeRetriever
PermissionChecker
EventLogger
CostMeter
OutputValidator
```

具体实现可以替换：

```text
OpenAI / Claude / 本地模型
ElasticSearch / Milvus / pgvector
CRM API / ERP API / 工单系统
```

这样平台不会被某个模型、某个向量数据库或某个业务系统绑定死。

### 3.5 CQRS 思想

Skill 系统要区分三类状态：

```text
配置态：Skill、SkillVersion、Prompt、ModelConfig、ToolBinding
运行态：SkillRun、上下文、模型调用、工具调用
观测态：日志、trace、成本、质量反馈、评估结果
```

这些状态的查询方式、写入频率、数据生命周期和性能要求都不同，不应该强行放在同一个模型里。

### 3.6 平台工程

Skill 模块本质上是平台能力，而不是单点业务功能。

平台工程关注：

```text
标准化
自助配置
复用
权限治理
可观测
配额和成本
低接入成本
稳定 API
```

因此 Skill 应该像平台 API 一样设计，而不是像某个业务页面后面的内部函数。

### 3.7 LLMOps

AI 系统和普通业务系统不同，模型输出有不确定性，因此需要内置 LLMOps 能力：

```text
Prompt 版本化
模型参数记录
测试集
A/B 测试
LLM-as-judge
人工反馈
输出 schema 校验
fallback model
成本控制
质量回归测试
```

Skill 版本升级时，要能回答：

```text
v2 是否比 v1 更好？
好在哪里？
差在哪里？
是否可以发布？
能否快速回滚？
```

### 3.8 安全工程

Skill 可能调用工具、读取知识库、访问业务数据，因此必须遵循安全前置设计：

```text
最小权限原则
服务端强校验
租户隔离
敏感数据脱敏
审计日志
工具调用白名单
参数校验
限流和配额
```

模型可以参与推理，但不能成为权限决策的最终主体。

## 4. Skill 模块的推荐领域模型

推荐把 Skill 拆成以下对象：

```text
Skill
├── SkillVersion
├── SkillInputSchema
├── SkillOutputSchema
├── PromptTemplate
├── ToolBinding
├── KnowledgeBinding
├── ModelConfig
├── PermissionPolicy
├── RuntimeConfig
├── EvaluationConfig
└── ExecutionLog
```

### 4.1 Skill

表示能力的稳定身份。

```text
id
tenant_id
name
code
description
category
status: draft / published / archived
owner
created_at
updated_at
```

### 4.2 SkillVersion

真正执行时应该绑定版本，而不是直接执行 Skill。

```text
id
skill_id
version
prompt_config
model_config
tools_config
knowledge_config
input_schema
output_schema
runtime_config
status: draft / testing / released / deprecated
created_by
created_at
```

发布后的版本建议不可变。需要修改时创建新版本。

### 4.3 SkillRun

记录每次执行。

```text
id
tenant_id
skill_id
skill_version_id
caller_type: chat / workflow / api / agent
caller_id
input
output
status
model
tokens_in
tokens_out
cost
latency_ms
error_code
error_message
trace_id
created_at
```

### 4.4 ToolBinding

描述 Skill 允许调用哪些工具。

```text
skill_version_id
tool_id
tool_name
permission_scope
timeout_ms
retry_policy
```

### 4.5 KnowledgeBinding

描述 Skill 绑定哪些知识库，以及检索策略。

```text
skill_version_id
knowledge_base_id
retrieval_top_k
similarity_threshold
rerank_enabled
```

## 5. Skill 的运行时链路

推荐设计统一 Runtime：

```text
调用方
  ↓
Skill Gateway
  ↓
权限校验
  ↓
加载 SkillVersion
  ↓
输入 Schema 校验
  ↓
上下文构建
  ↓
知识检索，可选
  ↓
Prompt 渲染
  ↓
模型调用
  ↓
工具调用，可选
  ↓
输出解析与校验
  ↓
日志、计费、指标
  ↓
返回结果
```

业务系统不应该直接拼 Prompt 或裸调模型，而是调用统一入口：

```ts
skillRuntime.execute({
  tenantId,
  skillCode: "sales_email_generator",
  version: "latest",
  input: {},
  context: {}
})
```

## 6. Skill 类型分层

MVP 不建议直接做完整 Agent 平台。可以从简单到复杂分层扩展。

### 6.1 Prompt Skill

```text
输入 → Prompt 模板 → 模型 → 输出
```

适合：

```text
文案生成
总结
分类
信息抽取
翻译
```

### 6.2 RAG Skill

```text
输入 → 检索知识库 → Prompt → 模型 → 输出
```

适合：

```text
客服问答
文档助手
内部知识查询
政策解释
```

### 6.3 Tool Skill

```text
输入 → 模型决策 → 调用工具/API → 汇总输出
```

适合：

```text
查订单
查 CRM
创建工单
查询数据库
执行业务动作
```

### 6.4 Workflow Skill

```text
多个 Skill + 条件分支 + 审批 + 外部调用
```

适合：

```text
销售跟进自动化
合同审查
报表分析
多步骤业务 Agent
```

## 7. Skill 场景的特殊注意点

### 7.1 Skill 不是 Prompt

Prompt 只是 Skill 的一种实现材料。Skill 更像一个能力单元：

```text
输入契约
执行策略
依赖资源
输出契约
权限
评估
版本
日志
```

如果只做 Prompt 管理，后续接 RAG、工具调用、工作流和 Agent 都会很痛苦。

### 7.2 输入输出必须结构化

AI 输出天然不稳定，所以 Skill 必须有 schema：

```text
input_schema
output_schema
output_parser
validation_policy
fallback_policy
```

Skill 结果常常会被业务系统继续消费，不能只返回一段自然语言。

### 7.3 版本必须强约束

每次线上执行都要记录：

```text
skill_id
skill_version_id
prompt_version
model
model_params
tool_versions
knowledge_snapshot
```

否则用户问“为什么昨天和今天结果不一样”，系统无法解释。

### 7.4 权限不能交给模型判断

模型只能提出它想调用某个工具，真正是否允许必须由系统判断。

需要校验：

```text
用户是否有权限
租户是否有权限
Skill 是否绑定该工具
工具参数是否合法
数据范围是否允许
调用频率是否超限
```

### 7.5 要设计失败路径

AI Skill 的失败不是异常情况，而是常态之一。

需要提前定义：

```text
模型超时怎么办
JSON 解析失败怎么办
工具调用失败怎么办
知识库无结果怎么办
输出置信度低怎么办
命中敏感内容怎么办
成本超限怎么办
```

常见策略：

```text
retry
fallback_model
fallback_prompt
return_partial_result
human_review
fail_fast
```

### 7.6 上下文是核心资产

Skill 的效果高度依赖上下文。

上下文来源包括：

```text
用户输入
业务对象
会话历史
知识库检索
系统变量
租户配置
权限范围
工具结果
```

还要控制上下文大小、优先级、脱敏规则和注入风险。

### 7.7 观测指标不能只看调用成功率

AI Skill 至少要看：

```text
成功率
延迟
成本
token
模型错误
工具错误
解析错误
用户采纳率
人工修改率
召回质量
输出合规率
```

否则无法判断 Skill 是真的产生价值，还是只是“能跑”。

### 7.8 评估体系要内置

Skill 发布前后都要评估。

建议支持：

```text
测试集
期望输出
规则评分
人工评分
LLM-as-judge
A/B 对比
回归测试
线上反馈
```

## 8. 推荐 MVP 范围

第一阶段不要直接做完整 Agent 平台。建议先做：

```text
Skill CRUD
SkillVersion
Prompt Skill 执行
输入输出 Schema
模型配置
执行日志
基础权限
测试运行面板
```

第一版核心 API：

```http
POST /skills
POST /skills/{id}/versions
POST /skills/{id}/publish
POST /skill-runs
GET  /skill-runs/{id}
```

后续再扩展：

```text
RAG Skill
Tool Skill
Workflow Skill
Agent Skill
Evaluation Center
Cost Center
Governance Center
```

## 9. 一句话总结

Skill 模块的架构核心不是“Prompt 怎么管理”，而是“如何把不稳定的 AI 能力封装成 SaaS 系统里可调用、可治理、可追溯、可评估的稳定能力单元”。

