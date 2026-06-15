---
title: AI 记忆模块调研报告 · OpenAI / Claude Code / Claude（含实现细节）
description: 用 8 维度方法论对比三家记忆方案，深挖每个维度的具体实现方式（数据格式 / API 签名 / 注入机制），附分析过程与分级引用
status: growing
topic: research
---

# AI 记忆模块调研报告：OpenAI vs Claude Code vs Claude

> 方法论见 [[memory-module-research-framework]]。本报告分三部分：**① 结论与横向对比 → ② 每个维度的具体实现方式（深挖）→ ③ 分析过程与分级引用**。
> 关联：[[rag-to-memory]]、[[mem0-short-term-vs-long-term]]、记忆动手 Demo（`code/memory/`）。
> 时间基准：2026 年 6 月。**引用按【官方】/【第三方逆向】/【媒体/利益相关方】分级，写公开内容前请按文末清单核对。**

---

## 一句话结论

**三家代表"记忆"的三种哲学，且都没走 mem0 式的"自动事实抽取 + 向量记忆层"：**
- **OpenAI**：消费端（ChatGPT）有真正的托管记忆系统，但封闭不可编程；**API 侧刻意只给"对话状态"不给"记忆层"**。
- **Claude Code**：记忆 = 人工维护的分层 markdown + 轻量自动笔记，**刻意不做向量化**。
- **Claude（API）**：把"记忆是架构问题"做成产品——提供从**自管文件**（memory tool）到**托管 store**（Managed Agents memory stores）的梯度，配套 context editing / compaction 减压。

三家的检索**全部是"文本/文件 + 模型理解"，没有一家把记忆做成向量召回**——这正是 mem0 等第三方记忆层的生存空间。

---

## 横向对比（方法论 8 维度浓缩）

| 维度 | OpenAI | Claude Code | Claude (API) |
|---|---|---|---|
| **1 类型边界** | ChatGPT：事实条目(bio)+画像合成；API：纯原始消息堆叠 | 分层 markdown + 自动笔记，**无抽取** | 多档：自管文件 / 托管 store / 上下文压缩 |
| **2 写入** | ChatGPT 自动(dreaming)+用户触发；API 全靠开发者 | 人工编辑为主 + Claude 偶尔自动记笔记 | 模型决定(memory tool)/双向(stores)/服务端自动(compaction) |
| **3 检索** | **非向量**，作为 system-prompt section 静态注入 | **非向量**，启动时拼接进上下文 | **非向量**，文件遍历 + 模型理解 |
| **4 存储** | 全托管，不可自管、不可移植 | 本地文件 / git，无中央服务器 | 自管(memory tool) **或** 托管(memory stores) |
| **5 遗忘** | 无 TTL；可手动删但"复现"争议 + 诉讼保留 | 无自动过期，靠人工审计 + /compact | TTL/redact/版本快照（stores 有审计） |
| **6 工程/可观测** | 消费端无 API；API 透明但浅 | 透明、git-friendly、`/memory` `/context` | memory tool 高度可编程；stores 企业级审计 |
| **7 评估** | 官方无 benchmark（仅第三方用 LOCOMO 对比） | 无量化指标，靠人工观察 | memory tool 可编程评估；stores 可导出审计 |
| **8 成熟度** | ChatGPT 记忆 GA；Assistants/Threads **2026-08 关停** | CLAUDE.md / Auto Memory GA | memory tool / memory stores 均 Beta |

---

## 二、每个维度的具体实现方式（深挖）

> 下面按"维度 × 三家"展开实现机制。⚠️ 标注表示该细节来自第三方/逆向或需核实。

### 维度 1：记忆类型与边界

**OpenAI（ChatGPT）** —— 两套并存：
- **Saved Memories**：`bio` 工具写入的显式事实，存储为**带日期戳的编号文本行**：
  ```
  # Model Set Context
  1. [2024-04-26]. User loves dogs.
  3. [2025-05-02]. The user likes ice cream and cookies.
  ```
  平台会自动合并/去重相关条目。⚠️【第三方逆向：Embrace The Red / TheBigPromptLibrary】
- **Reference chat history**：通过名为 **"dreaming"** 的后台进程合成的**多层用户画像**（非原始堆叠），见维度 2。

**OpenAI（API）** —— **只有"对话状态"，没有"记忆层"**：`previous_response_id` 链式 / Conversations 对象 / （已废弃的）Threads，本质都是**原始消息时间序列**，无任何自动摘要或事实抽取。【官方】确认 OpenAI 不提供 mem0 式的事实抽取记忆抽象。

**Claude Code** —— **分层人工 markdown + 自动笔记双轨**：
- CLAUDE.md 四层：managed(企业) → user(`~/.claude/CLAUDE.md`) → project(`./CLAUDE.md`) → local(`./CLAUDE.local.md`)；外加 `.claude/rules/*.md`（路径作用域规则）。
- Auto Memory：Claude 自己的笔记，存 `~/.claude/projects/<repo>/memory/`。
- **完全无向量、无事实抽取**，是"人写的指令 + 模型自记的笔记"。【官方】

**Claude（API）** —— 多档梯度：
- memory tool：`/memories` 目录下的**自定义文件结构**，开发者定义格式。
- memory stores：工作区级持久记忆，对象模型 `memstore_*`(库) / `mem_*`(单条) / `memver_*`(版本)。单条 ≤100KB，每 session 最多 attach 8 个 store。⚠️(单 store 上限 2000 条为第三方整理，需核实)
- Projects(claude.ai)：项目指令 + 知识库(向量检索) + Auto Memory(账户级画像)。【官方 + 部分需核实】

---

### 维度 2：写入路径

**OpenAI（ChatGPT）**：
- **bio 写入**：模型通过**命名空间通道** `to=bio` 发送（如 `to=bio I love dogs`），平台拦截落库；底层是否包装成标准 function call **未公开**。⚠️【第三方逆向】
- **dreaming（画像合成）**：**后台异步批处理**，回看历史对话自动萃取画像，非实时。精确周期未公开；可参照官方 ChatGPT Pulse 为 nightly（每日级）。2026-06 升级为 "Dreaming V3"（画像变可编辑摘要）。【官方术语 + 媒体】
- 冲突处理（新旧事实矛盾如何取舍）**未公开**。

**OpenAI（API）**：完全由开发者显式控制——`store=true/false` 决定是否服务端保存，链式靠传 `previous_response_id`。无模型自主写入、无事实冲突概念。【官方】

**Claude Code**：
- CLAUDE.md：人工编辑 / `/init` 生成。
- `#` 快捷或"remember that…"：Claude **自动决定**是否写入 Auto Memory（不是每次都写）。
- `/init`：扫描代码库（结构、依赖、构建命令）生成 `./CLAUDE.md`。【官方；⚠️ `CLAUDE_CODE_NEW_INIT` 等具体环境变量需核实】

**Claude（API）**：
- **memory tool（模型决定）**：系统提示强制 *"ALWAYS VIEW YOUR MEMORY DIRECTORY BEFORE DOING ANYTHING ELSE"*，Claude 自主调用 6 个命令；应用端执行并持久化。
- **memory stores（双向）**：Agent 写 `/mnt/memory/`（FUSE 挂载）持久化，开发者也可用 `memories.create/update` 直接编程修改，支持 `precondition: content_sha256` 乐观并发。
- **compaction（服务端自动）**：Claude 自动生成摘要替换历史。【官方】

---

### 维度 3：检索路径（最核心）

**全部三家都是"文本/文件 + 模型理解"，没有一家做向量召回记忆。**

**OpenAI（ChatGPT）** —— **多个命名 section 拼接进 system prompt，每条消息重新注入**⚠️【第三方逆向】：

| Section | 内容 | 是否含 assistant 回复 |
|---|---|---|
| `Model Set Context` | bio 显式记忆（全量） | — |
| `Assistant Response Preferences` | 推断的回答风格（带 Confidence） | — |
| `Notable Past Conversation Topic Highlights` | 早期话题摘要 | — |
| `Helpful User Insights` | 画像数据点 | — |
| `Recent Conversation Content` | 最近 ~15–40 个会话 | **否，只含用户消息** |
| `User Interaction Metadata` | 用量/设备/活动模式 | — |

关键：**不是全文 RAG 检索**——ChatGPT 无法搜索你的历史原文；记忆是 dreaming 预计算后**静态注入**。`Recent Conversation Content` 设计上**排除助手回复**以降数据量+降注入风险。

**OpenAI（API）**：`previous_response_id` 链式是**全量重放**（历史 input token 每轮都计费），无选择性召回。检索能力来自独立的 **vector store / file search**（默认 `text-embedding-3-large`@256 维，chunk 800/overlap 400，hybrid 检索 + metadata 过滤），但那是 RAG 不是记忆系统。【官方】

**Claude Code** —— **无向量，启动时按层拼接进上下文（作为用户消息）**：
- 加载顺序：managed → user → project → local → 子目录规则 → Auto Memory(MEMORY.md 前 200 行/25KB)。
- 目录树**向上遍历**到根；子目录 CLAUDE.md / 路径作用域规则**懒加载**（Claude 读匹配文件时才注入）。
- Auto Memory 超出 200 行/25KB 的部分需 Claude 主动 Read。【官方；⚠️ 注入为"用户消息"而非"系统提示"这一点来自第三方分析，需核实】

**Claude（API）**：
- memory tool / memory stores：Claude 调 `view` 命令**遍历文件目录**读取相关文件进上下文，纯文件遍历 + 模型理解，**非向量**。
- memory stores 支持 `memories.list(path_prefix, depth)` 按路径过滤。【官方】

---

### 维度 4：存储与后端

| | 后端 | 可自管 | 向量化 |
|---|---|---|---|
| OpenAI ChatGPT | 全托管、账户内 | ❌ | 注入是静态 section，**不确认有向量库** |
| OpenAI API 状态 | Response 存 30 天 / Conversation 无 TTL；`store=false` 自管 | 部分 | 否（vector store 另算，$0.10/GB/天，首 1GB 免费） |
| Claude Code | 本地文件 / git，无中央服务器 | ✅ 完全 | 否 |
| Claude memory tool | **开发者自建**(文件/DB/S3/加密) | ✅ 完全 | 否 |
| Claude memory stores | **Anthropic 托管**，FUSE 挂载抽象 | ❌(零维护) | 否 |

Claude Code 细节：Auto Memory 路径由 git 仓库推导（`~/.claude/projects/<repo-hash>/memory/`），同仓库所有 worktree 共享；不跨机器同步。【官方】

---

### 维度 5：遗忘 / 生命周期

**OpenAI**：
- ChatGPT：可在 Settings 逐条删或全清；可说 "forget X" 即时移除。**无 TTL**。⚠️ 但有"已删记忆数年后复现"报道【媒体】，且因 NYT 诉讼**已删会话被法院要求无限期保留**【媒体】。删除某次对话**不连带删**它产生的记忆。
- API：`store=true` response 默认 30 天 TTL；Conversation 对象无 TTL，需 `DELETE /v1/conversations/{id}` 显式删（删 conversation **不连带删** items）。【官方】

**Claude Code**：CLAUDE.md / Auto Memory **无自动过期**，靠人工审计（`/memory`）。`/compact` 时先清工具输出再摘要；⚠️ auto-compact 触发阈值（~95% 容量等具体数字）需核实。压缩后**项目根 CLAUDE.md 与 Auto Memory 会从磁盘重新加载**，但子目录 CLAUDE.md / 路径规则**不会**，要等 Claude 再次读匹配文件。【官方核心 + ⚠️ 阈值细节需核实】

**Claude（API）**：
- memory tool：开发者自行实现 TTL / 删除。
- memory stores：多层控制——开发者 API 删 / agent 删 / **redact**（清内容保审计日志，GDPR）/ 版本 30 天（活跃版本不过期）；每次改动生成 `memver_*` 不可变快照。【官方】

---

### 维度 6：工程化与可观测

**OpenAI**：ChatGPT 记忆**无开发者 API**、不可移植；用户可在 Settings 查看 saved memories 全列表（透明），但 dreaming 画像不完全可见。API 侧状态是明文消息列表，透明但浅，无"记忆条目"概念。【官方 + ⚠️逆向】

**Claude Code**：高度透明、git-friendly。命令：`/init`(生成) `/memory`(查看编辑各层) `/context`(可视化上下文占用) `/compact`。⚠️ `InstructionsLoaded` hook 及其 JSON 字段、`claudeMdExcludes`/`autoMemoryDirectory` 等设置项**部分需核实**（可能为第三方整理）。【官方核心 + ⚠️】

**Claude（API）**：
- memory tool：6 命令清晰，全 SDK 支持，所有 memory 调用在响应中可见可审计；SDK 提供抽象基类（Python `BetaAbstractMemoryTool` / TS `betaMemoryTool`）接客户端存储。
- context editing：响应返回 `applied_edits`（清了几个 tool use、省了多少 token）。
- memory stores：企业级——版本控制 + 审计（记录操作类型/操作者 session_id/时间戳）+ `precondition` 乐观并发 + `read_only` 挂载防注入。【官方】

---

### 维度 7：评估能力

- **OpenAI**：官方**无记忆 benchmark**。唯一公开对比是**第三方 mem0** 用 **LOCOMO**（长对话基准）测得"比 OpenAI 内置记忆准确率 +26% / 延迟 -91%"——⚠️【利益相关方自报数据，且针对 API 场景非 ChatGPT 消费端】。
- **Claude Code**：无量化指标（无 recall/precision），靠人工观察 + `/memory` 验证指令是否加载。【官方】
- **Claude（API）**：memory tool 可编程评估（记录命令历史、检索频率、A/B 不同记忆结构）；memory stores 可导出审计、查版本历史。【官方】

---

### 维度 8：生态与成熟度

| | 状态 |
|---|---|
| OpenAI ChatGPT 记忆 | GA；saved memories 早期，reference chat history 2025-04，Dreaming V3 2026-06 |
| OpenAI Assistants/Threads | **已废弃**：2025-08-26 公告，**2026-08-26 关停**，无自动迁移工具（Thread→Conversation） |
| OpenAI Responses/Conversations | Responses 2025-03 发布主推；Conversations GA/beta 标签⚠️需核实 |
| OpenAI vector store/file search | GA |
| Claude Code CLAUDE.md / Auto Memory | GA（Auto Memory 约 v2.1.59+）⚠️版本号需核实 |
| Claude memory tool | Beta（`memory_20250818`，2025-09） |
| Claude memory stores | Public Beta（`managed-agents-2026-04-01`，2026-04） |

---

## 关键实现速查（Claude API，可直接用 —— 与官方 SDK 文档一致）

```python
# 1) Memory Tool —— 自管文件记忆
tools=[{"type": "memory_20250818", "name": "memory"}]
# 6 命令：view / create / str_replace / insert / delete / rename，路径限定 /memories/
# SDK 抽象基类：Python BetaAbstractMemoryTool，TS betaMemoryTool

# 2) Context Editing —— 清理过期 tool result / thinking
# beta header: context-management-2025-06-27
context_management={"edits": [
  {"type": "clear_tool_uses_20250919",
   "trigger": {"type": "input_tokens", "value": 100000},  # 默认 100k
   "keep": {"type": "tool_uses", "value": 3},
   "exclude_tools": ["web_search"]},
  {"type": "clear_thinking_20251015", "keep": {"type": "thinking_turns", "value": 2}},
]}

# 3) Compaction —— 服务端摘要（长会话）
# beta header: compact-2026-01-12
context_management={"edits": [{"type": "compact_20260112",
   "trigger": {"type": "input_tokens", "value": 150000}}]}  # 默认 150k，最小 50k
# 关键：必须把 response.content（含 compaction block）回传，否则丢状态

# 4) Managed Agents Memory Stores —— 托管跨会话记忆
# beta header: managed-agents-2026-04-01
store = client.beta.memory_stores.create(name="...", description="...")
client.beta.memory_stores.memories.create(store.id, path="/x.md", content="...")
# attach（仅 session 创建时）：resources=[{"type":"memory_store","memory_store_id":store.id,"access":"read_write"}]
# FUSE 挂载到容器 /mnt/memory/<name>/；版本审计 memory_versions.list / .redact
```

---

## 二·补：三个专题深挖

### 专题 A：Claude Code Auto Memory 的写入逻辑

**核心定性：实时记笔记，不是 dreaming 式离线合成。**【官方】文档原文 "Claude reads and writes memory files **during your session**"。

| 问题 | 实现 | 来源 |
|---|---|---|
| 何时写 | 会话**进行中持续决策**，非会话结束批量合成；看到 UI "Writing memory" 即在写 | 官方 |
| 写什么 | "It decides what's worth remembering based on whether the information would be useful in a future conversation" | 官方 |
| 决策算法 | **未公开** —— 纠正/确认/构建命令/调试发现是否为硬触发信号官方未说明 | ⚠️ 未公开 |
| 显式触发 | 说 "remember X" → Claude 明确存入 **Auto Memory**（不是 CLAUDE.md） | 官方 |
| 用什么工具写 | **标准 Read/Write/Edit**，非专用 memory 命令 | 官方 |
| 文件结构 | `MEMORY.md`(索引，每会话载前 200 行/25KB) + 主题文件(`debugging.md` 等，按需读取)；超限自动分流到主题文件 | 官方 |
| 多 worktree | 同 repo 共享一个 `~/.claude/projects/<repo>/memory/` | 官方 |
| 自维护/去重 | 是否主动清理过时条目、去重、覆盖 vs 追加、冲突检测 —— **均未公开** | ⚠️ 未公开 |
| 条目格式 | **官方无示例**；推测纯 markdown(标题/摘要 + 指向主题文件链接)，建议 `/memory` 打开实测 | ⚠️ 无示例 |
| 与 CLAUDE.md 分工 | CLAUDE.md=**你写**的指令规则；Auto Memory=**Claude 写**的学习与模式(构建命令/调试洞察/风格偏好)。`#` 默认进 Auto Memory | 官方 |

> 要点：Auto Memory 的"写什么值得记"是**模型自主、算法不公开**；它**没有**OpenAI dreaming 那种"会话后离线再合成画像"的阶段。

### 专题 B：各家 Dreaming / 离线合成对比（实时 vs 离线提炼）

> "Dreaming" = 后台离线读历史、重新合成记忆。**只有 OpenAI 和 Anthropic Managed Agents 真正有；Claude Code / claude.ai / Claude memory tool 都是实时，无离线合成。**

**OpenAI ChatGPT「Dreaming」**【官方公告 2026-06-04 "Dreaming V3"，但一手页 403，引文经多家媒体转引】
- **定义**：*"automatically curate memories in the background by referencing chat history"* —— 异步后台进程，跨多年对话合成用户画像。
- **机制**：**离线异步批处理**，"learns while users are away"。**精确周期未公开**（勿引用具体数字）；与 ChatGPT Pulse（确证 nightly）共享记忆基座，但"dreaming=nightly"属推断。
- **产物**：Memory Summary Page（可编辑摘要）；V3 新增**时间感知**（行程结束自动把"将去新加坡"改写为"已于 2026-07 去过"）。
- **V3 变化**：从"should I remember this?"显式确认 → **全自动合成**；dreaming 取代 saved-memories 列表成为**主基座**。
- **冲突解决算法**：**未公开**（只有时间衰减改写，非"新旧矛盾如何覆盖/合并"）。
- **透明度争议**：UI 控制变多，但**完整可审计性反而下降**（摘要不含全部记忆、删对话不删衍生记忆、删除日志留 ≤30 天）。⚠️【媒体 + CHI 2026 研究】

**Anthropic「Dreams」**（Managed Agents，**唯一**官方离线合成）【官方 beta，研究预览】
- **唯一存在处**：Managed Agents API 的 `Dreams`，**手动触发**(`POST /v1/dreams`)，非自动周期。
- **输入**：现有 memory store + 1–100 个 session 完整转录；可带 `instructions`(≤4096 字)控制策略。
- **产物**：**生成新的重组 store**（合并重复、用最新值覆盖矛盾、表层化新洞察），**原 store 永不修改** —— 审阅后选择采用/丢弃。
- **状态**：研究预览，需申请；支持 opus-4-8/4-7、sonnet-4-6；按 token 计费。
- ⚠️ beta header 调研给出 `managed-agents-2026-04-21`，与 memory stores 的 `managed-agents-2026-04-01` 不同，**需核实**（Dreams 未出现在本仓库已加载的 claude-api skill 文档中，属较新功能）。

**实时（无离线合成）的三者**：
- **Claude Code Auto Memory**：实时记笔记。网上 "Claude Code auto dream" 文章是**第三方/社区命名**，非官方机制（官方文档零提及）。⚠️
- **claude.ai Projects / Auto Memory**：项目指令 + 知识库向量检索 + 会话内记忆，**无 dreaming 式后台画像合成**。
- **Claude API memory tool**：纯客户端实时文件读写，无后台合成。

| | 后台离线合成 | 触发 | 产物 |
|---|---|---|---|
| OpenAI ChatGPT | ✅ Dreaming(自动) | 自动(周期未公开) | 可编辑画像摘要 |
| Anthropic Managed Agents | ✅ Dreams(手动) | `POST /v1/dreams` | **新 store**(原 store 不动) |
| Claude Code Auto Memory | ❌ 实时 | 会话内 | 更新 MEMORY.md |
| claude.ai / Claude memory tool | ❌ 实时 | — | — |

### 专题 C：记忆记录的 Schema 设计

**mem0（事实抽取式记忆的范本）**【官方 docs + 源码】
- **对外 API 返回的单条记忆**：`id`(UUID) / `memory`(事实文本) / `metadata` / `categories[]` / `created_at` / `updated_at` / `score`(仅 search，0–1)。
- **内部向量库 payload**(源码 `main.py`)：文本键内部叫 `data`(对外是 `memory`)、`hash`(**MD5** 去重)、`user_id`/`agent_id`/`run_id`(三作用域)、`actor_id`、`role`、`text_lemmatized`(BM25 用)。⚠️ `hash`/`user_id` 是内部/过滤字段，**不在对外返回对象里**。
- **embedding 单独存**：向量由向量库本身持有，payload 只是附着元数据。
- **三层存储**：① 向量库(Qdrant 默认，语义检索) ② 图存储(Neo4j 等，实体-关系，可选) ③ 历史 DB(默认 **SQLite**，审计 `old_memory`/`new_memory`/`event`/时间戳)。
- **`add()` 返回**：`{"results":[{"id","memory","event":"ADD|UPDATE|DELETE|NONE"}], "relations":[]}` —— LLM 决策对已有记忆的操作，而非简单插入。

**各家最小 schema 对比**

| 系统 | 存储单元 | 关键字段 | 结构化 | 去重 hash |
|---|---|---|---|---|
| mem0 | 结构化记录 | id/memory/metadata/categories/作用域/时间/score + 三层 | 高 | MD5 |
| OpenAI ChatGPT bio | 一行自然语言 | `N. [date]. fact`(⚠️第三方逆向约定，无官方 schema)，无向量 | 极低 | — |
| OpenAI vector store | file + attributes | attributes ≤16 key，用于 filter | 中 | — |
| Claude memory tool | 纯文件 | path + 文本，无结构化字段 | 极低 | — |
| Claude memory stores | `mem_` 对象 | id/type/path/content/`content_sha256`(SHA256 并发控制)/created_at/version | 中 | SHA256 |

> 注意：mem0 用 **MD5 去重内容**，Claude memory stores 用 **SHA256 做并发/版本校验**，目的不同。

**推荐的"理想长期记忆记录" schema（可直接建表）**
```sql
CREATE TABLE memory_record (
  id            UUID PRIMARY KEY,
  content       TEXT NOT NULL,          -- 记忆文本
  content_hash  CHAR(64),              -- 去重/幂等
  embedding     VECTOR(1536),          -- 放向量库
  user_id TEXT, agent_id TEXT, run_id TEXT, actor_id TEXT,  -- 多租户/会话作用域
  memory_type   TEXT,                  -- episodic | semantic | procedural
  categories    TEXT[],
  source        TEXT,                  -- 溯源(来源消息/文档)
  confidence    REAL,                  -- 抽取置信度
  created_at TIMESTAMPTZ NOT NULL, updated_at TIMESTAMPTZ,
  last_accessed_at TIMESTAMPTZ,        -- 支持近因/衰减
  ttl           INTERVAL,              -- 遗忘(短期记忆尤其需要)
  is_deleted    BOOLEAN DEFAULT FALSE,
  metadata      JSONB
);
```
存储分层范式（与 mem0 一致）：① 短期(rolling summary/run_id 作用域)放 Redis；长期(稳定事实)落向量库+关系库 ② 检索 = 向量相似 **AND** 元数据过滤(user_id/type/时间) ③ 审计层独立记 ADD/UPDATE/DELETE ④ 可选图层。

---

## 三、分析过程（方法可复现）

1. **沉淀方法论**：先把"调研记忆模块该看哪些维度"固化为 8 维度 + 打分表（`content/02-agent/methodology/memory-module-research-framework.md`）。
2. **第一轮概览调研**：并行派出 3 个调研 agent（OpenAI 用 web-search-agent；Claude Code / Claude 用 claude-code-guide，可联网核对官方文档），各自按 8 维度结构化输出 + 来源 + 不确定标注。
3. **第二轮实现深挖**：针对"每个维度的具体实现方式"再并行派出 3 个 agent，要求带**数据格式 / API 签名 / 注入机制 / beta header / 代码片段**，并强制标注来源是【官方】还是【逆向】。
4. **交叉校验**：Claude API 的 type 字符串与 beta header（`memory_20250818`、`clear_tool_uses_20250919`、`clear_thinking_20251015`、`compact_20260112`、`managed-agents-2026-04-01`）与本仓库已加载的 `claude-api` skill 文档一致，标为【已验证】；ChatGPT 内部 section 名 / bio 格式 / 会话轮数全部来自逆向，标为⚠️。
5. **综合成文**：横向对比 → 逐维度实现 → 速查代码 → 引用分级。

> 本报告由多 agent 调研综合而成，部分实现细节（尤其 ChatGPT 内部结构、Claude Code 部分环境变量/hook 字段）来自第三方，**未逐字核对官方原页**。引用公开内容前请按下方清单核实。

---

## 四、分级引用

**【官方】**
- OpenAI: [Memory and new controls](https://openai.com/index/memory-and-new-controls-for-chatgpt/)、[Memory FAQ](https://help.openai.com/en/articles/8590148-memory-faq)、[Dreaming](https://openai.com/index/chatgpt-memory-dreaming/)、[Conversation state](https://developers.openai.com/api/docs/guides/conversation-state)、[Assistants migration](https://developers.openai.com/api/docs/assistants/migration)、[Deprecations](https://developers.openai.com/api/docs/deprecations)、[Retrieval / file search](https://developers.openai.com/api/docs/guides/retrieval)、[API Pricing](https://openai.com/api/pricing/)
- Claude Code: [memory.md](https://code.claude.com/docs/en/memory.md)、[context-window.md](https://code.claude.com/docs/en/context-window.md)、[how-claude-code-works.md](https://code.claude.com/docs/en/how-claude-code-works.md)、[settings.md](https://code.claude.com/docs/en/settings.md)、[hooks-guide.md](https://code.claude.com/docs/en/hooks-guide.md)
- Claude API: [Memory tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool)、[Context editing](https://platform.claude.com/docs/en/build-with-claude/context-editing)、[Compaction](https://platform.claude.com/docs/en/build-with-claude/compaction)、[Managed Agents memory](https://platform.claude.com/docs/en/managed-agents/memory)、[Projects help](https://support.claude.com/en/articles/9519177)

**【官方 · 专题深挖新增】**
- Dreaming/Dreams: [OpenAI Dreaming 公告](https://openai.com/index/chatgpt-memory-dreaming/)、[Claude Managed Agents Dreams](https://platform.claude.com/docs/en/managed-agents/dreams)（beta，⚠️header 待核实）
- mem0 schema: [Get Memories](https://docs.mem0.ai/api-reference/memory/get-memories)、[Search](https://docs.mem0.ai/api-reference/memory/search-memories)、[main.py](https://github.com/mem0ai/mem0/blob/main/mem0/memory/main.py)、[storage.py](https://github.com/mem0ai/mem0/blob/main/mem0/memory/storage.py)、[add() 操作](https://mem0.ai/blog/understanding-mem0-s-add()-operation)

**【第三方逆向】**（ChatGPT 内部结构的唯一来源，谨慎引用）
- [Embrace The Red — How ChatGPT Remembers You](https://embracethered.com/blog/posts/2025/chatgpt-how-does-chat-history-memory-preferences-work/)
- [TheBigPromptLibrary — chatgpt bio & memory](https://github.com/0xeb/TheBigPromptLibrary/blob/main/Articles/chatgpt-bio-tool-and-memory/chatgpt-bio-and-memory.md)
- [llmrefs — Reverse Engineering ChatGPT Memory](https://llmrefs.com/blog/reverse-engineering-chatgpt-memory)

**【媒体 / 利益相关方】**（引用须标注性质）
- mem0 LOCOMO 对比数据（mem0 自报）：[mem0.ai](https://mem0.ai/) / [GitHub](https://github.com/mem0ai/mem0)
- "已删记忆复现" / NYT 诉讼保留 / Dreaming V3：各科技媒体报道

---

## 五、写公开内容前的核对清单

1. ChatGPT 内部 section 名、bio 条目格式、"~15–40 会话"——**仅逆向**，OpenAI 未公开，引用务必标注"第三方逆向"。
2. mem0 "+26%/-91%" ——**利益相关方自报**，且针对 API 非 ChatGPT 消费端。
3. "已删记忆复现"、NYT 诉讼保留 ——**媒体报道**。
4. Claude Code 的 `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE` / `InstructionsLoaded` hook 字段 / Auto Memory 版本号 / "注入为用户消息"——**部分需核对官方原页**。
5. memory stores 单 store 上限 2000 条、Conversations API 的 GA/beta 标签、OpenAI 单条 response 是否有 delete 端点——**需核实**。
6. 各 beta header / type 字符串以**官方最新文档**为准（会随版本更新）。
