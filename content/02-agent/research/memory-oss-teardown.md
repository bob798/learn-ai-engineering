---
title: 开源 AI 记忆项目深度拆解 · Graphiti / Cognee / Memobase / A-MEM
description: 用 8 维度方法论逐项拆解四个代表不同范式的开源记忆项目，含实现机制、schema、评分、可借鉴点与局限
status: growing
topic: research
---

# 开源 AI 记忆项目深度拆解

> 方法论见 [[memory-module-research-framework]]；三大商用方案对比见 [[ai-memory-implementation-survey]]。候选清单见同目录 OSS candidates。
> 本篇用 8 维度逐项拆解 6 个**代表不同范式**的开源项目。时间基准 2026-06。评分 1-5；⚠ 表示不确定/需核实。所有字段名尽量取自源码。

| 项目 | 范式 | 一句话 | 综合 |
|---|---|---|---|
| **Letta / MemGPT** | OS 式分层 | core/recall/archival 分层 + LLM 自管换页 + stateful server | ~4.6 |
| **mem0** | 事实抽取式 | LLM 抽事实 + ADD/UPDATE/DELETE/NOOP 决策(赛道事实标准) | ~3.9(争议见维度7) |
| **Graphiti / Zep** | 时序知识图谱 | 双时态图 + 软遗忘 + 检索期零 LLM | ~4.5 |
| **Cognee** | 图谱构建层(GraphRAG) | ECL 管线把任意数据抽成可推理图 | ~4.1 |
| **Memobase** | 结构化用户画像 | profile 槽位(topic/sub_topic/content) + event 时间线 | ~4.4 |
| **A-MEM** | 自组织(Zettelkasten) | atomic note + LLM 记忆进化/自链接 | ~3.25(学术满分/工程偏弱) |

> Letta、mem0 另有独立专文：[[memgpt-letta-guide]]、mem0 译文 [[mem0-short-term-vs-long-term]] / [[rag-to-memory]]；此处按统一 8 维度补入以便横向对比。

---

## 1. Graphiti / Zep —— 时序知识图谱（27k★ Apache-2.0，v0.29.2）

**范式定位**：把对话/数据抽成 **节点-边-节点** 三元组，边上挂"事实句"，用**四时间戳双时态**做软遗忘；检索是纯算法 hybrid（向量+BM25+图遍历+RRF/MMR/cross-encoder），**检索期零 LLM 调用**。

| 维度 | 实现要点 | 分 |
|---|---|---|
| 1 类型边界 | 三层子图：Episode(原始)/Entity(抽取实体+事实边)/Community(聚类摘要)，原始与抽取都留 | 5 |
| 2 写入 | `add_episode()` 增量摄取 → extract_nodes → **实体消解 resolve**(embedding+全文找候选,LLM 判同一实体合并) → extract_edges → **每条新边一次 LLM 失效判断** | 5 |
| 3 检索 | hybrid + RRF 融合 + 可选 rerank(mmr/cross_encoder/node_distance/episode_mentions)；16 个预设 recipe；`search(query, center_node_uuid=)` | 5 |
| 4 存储 | 图库可插拔(Neo4j 主力/FalkorDB/Neptune/Kuzu弃)，向量+全文索引由图库本身承载，非独立向量库 | 4 |
| **5 遗忘** | **核心**：矛盾时旧边 `invalid_at ← 新边 valid_at`(现实时间轴关闭)，系统作废设 `expired_at`(事务时间轴)，**历史永不物理删除**，支持时间点回溯 | 5 |
| 6 工程/可观测 | `graphiti-core` + MCP server + REST；**不绑 Zep 云**可自托管；边带 `episodes` 可溯源到原文，可解释性强 | 4 |
| 7 评估 | DMR 94.8%(略胜 MemGPT 93.4%)；LongMemEval(gpt-4o) 71.2% vs 60.2%，上下文 115k→1.6k token、延迟降~90%（**厂商自评论文**） | 4 |
| 8 生态 | 27.4k★、196 releases、迭代极频繁、Apache-2.0 | 5 |

**`EntityEdge` 核心字段**（边即"事实"）：
```python
fact: str                  # 自然语言事实句
fact_embedding: list[float]
episodes: list[str]        # 产生此事实的 episode uuid(溯源)
valid_at / invalid_at      # 现实时间轴：事实何时真/失效
expired_at / created_at    # 事务时间轴：系统何时作废/知道
group_id                   # 图分区键(多租户隔离)
```

**总评**：需要"知识会过时、且失效要可追溯"的长期 agent 记忆（个人助理偏好、CRM/客服业务状态、时态推理）——双时态软遗忘是同类开源最完整的。**局限**：写入 LLM 密集(每边一次失效判断)成本/延迟高；时态正确性仍有边界 bug(时区比较 #920、回填 #1489)；失效判断依赖 LLM 质量；评估为厂商自评。

---

## 2. Cognee —— 图谱构建层 / GraphRAG（18k★ Apache-2.0，v1.1.2）

**范式定位**：**"LLM 驱动的知识图谱构建层 + 混合检索"**，对外包装成 memory。核心不是记忆条目存储，而是把任意数据通过 **ECL 管线（Extract-Cognify-Load）** 抽成实体-关系图 + 向量索引。

> ⚠ v1.1.x 有两层 API：高层 `remember/recall/forget/improve`(营销叙事) 与底层 `add/cognify/search/memify`(真正核心，教程/源码主用)。以底层为准。

| 维度 | 实现要点 | 分 |
|---|---|---|
| 1 类型边界 | 偏"知识图谱构建层(GraphRAG-as-memory)"，存实体+边+chunk+摘要+向量；非对话 scratchpad | 4 |
| **2 写入(ECL)** | **核心**：`add()`(ingest) → `cognify()`(LLM 抽实体关系成图)。默认 Task DAG：classify→chunk→**extract_graph_and_summarize**→add_data_points→fk_edges；`run_tasks()` 可完全自定义 pipeline | 5 |
| 3 检索 | `search(query, query_type=SearchType.X)`：GRAPH_COMPLETION(默认)/RAG_COMPLETION/CHUNKS/INSIGHTS(三元组)/CYPHER/NATURAL_LANGUAGE/TEMPORAL 等十余种 | 5 |
| 4 存储 | **三件套可插拔**：图库(Kùzu/NetworkX 默认,Neo4j/FalkorDB/Neptune) + 向量(LanceDB 默认,pgvector/Qdrant/Milvus) + 关系(SQLite/Postgres) | 5 |
| 5 遗忘 | `prune.prune_data()/prune_system()` 全量重置 + 数据集级 `forget()`；`incremental_loading` + `source_content_hash` 去重；**细粒度单实体失效/TTL 弱** | 3 |
| 6 工程/本体 | `DataPoint`(pydantic 基类，含 `index_fields`/`source_content_hash`/`version`) 可自定义本体；`cognify(graph_model=YourModel)` 换本体 | 4 |
| 7 评估 | arXiv 2505.24478 Dreamify(HPO 超参搜索)把 HotpotQA F1 0.169→0.840，但**无横向对手对比 + 训练测试同分布**，可信度打折 | 3 |
| 8 生态 | 17.8k★、v1.1.2 活跃、Apache-2.0、集成丰富(MCP/Redis/Qdrant/Neptune)；⚠ 融资金额未在官方查到 | 4 |

**`DataPoint` 关键字段**：`id/created_at/updated_at/version/source_content_hash(去重)/metadata{index_fields}` —— `index_fields` 决定哪些字段进向量库。

**总评**：把异构文档/库转成可推理知识图谱 + 混合检索的 GraphRAG 底座（多跳问答、实体溯源、自定义本体、图库/向量库可插拔）。**局限**：细粒度遗忘弱(prune 偏全量重置)；cognify 每次靠 LLM 抽取，高频小增量成本高；benchmark 含超参过拟合且自评；双层 API 易混淆。

---

## 3. Memobase —— 结构化用户画像（2.8k★ Apache-2.0）

**范式定位**：以**结构化 user profile 为一等公民**，把对话抽成 `topic/sub_topic/content` 三段槽位（而非自由文本片段），另存 event 时间线。**画像检索 embedding-free（纯 SQL，<100ms）**；event 走 pgvector。

| 维度 | 实现要点 | 分 |
|---|---|---|
| **1 类型边界** | **Profile**(三段式槽位，内置 8 topic:basic_info/contact/education/demographics/work/interest/psychological/life_event，可自定义) + **Event**(summary/tags/profile_delta/time) | 5 |
| 2 写入 | `insert(ChatBlob)` → buffer 缓冲 → flush 触发(满 1024 token / 隔 3600s / 手动) → LLM 抽取+合并(**APPEND/UPDATE/ABORT** 三选一) | 5 |
| 3 检索 | `profile()` 纯 SQL 读结构化 JSON(<100ms,免 embedding)；`context()` 返回可直接拼 prompt 的字符串；event 走向量 search | 5 |
| 4 存储 | FastAPI+Postgres(+pgvector)+Redis；`user_profiles` 表 `content TEXT`+`attributes JSONB`(**无向量列**)，`user_events` 才有 `Vector` 列 | 4 |
| 5 遗忘 | 默认 UPDATE 覆盖；sub_topic 的 `update_description` 指导保留策略；超 token 触发 re-summarize、超 15 sub_topic 触发 reorganize；**无 TTL/衰减** | 4 |
| 6 工程 | Python/Node/Go SDK + REST；docker 自托管 / 云；多租户 `project_id`；可观测文档薄 | 4 |
| 7 评估 | repo 内 LoCoMo(LLM Judge)：overall 75.78% > mem0 66.88% > Zep 65.99%；**Temporal 85% 大幅领先**，但 **Multi-Hop 输给 mem0**(自报数字) | 4 |
| 8 生态 | 2.7k★、Apache-2.0、多语言 SDK；⚠ 近半年 push 放缓、无正式 release notes | 4 |

**自定义 profile schema**：
```yaml
additional_user_profiles:
  - topic: "gaming"
    sub_topics: [{name: "FPS"}, {name: "RPG", update_description: "保留最高段位"}]
```

**总评**：长期 1:1 对话的陪伴/教育/客服/角色扮演 chatbot——结构化画像可直接喂 prompt、可审计、可人工修正，产品化体验接近 ChatGPT memory，延迟敏感场景友好。**局限**：强依赖 LLM 抽取/合并质量；画像偏静态人口学属性，Multi-Hop 跨片段推理弱；"<100ms"仅画像路径(event search 回到 500ms+)；无 TTL；⚠ 你之前提的 "<80ms" 官方未证实(仅见 <100ms)。

---

## 4. A-MEM —— 自组织 Zettelkasten（1k★ MIT，NeurIPS 2025，⚠维护近停滞）

**范式定位**：把卡片盒笔记法落地到 agent 记忆——每条是含结构化属性的 **atomic note**，写入时 LLM 自动生成 keywords/tags/context，并与历史 note **动态建链 + 反向更新旧 note 属性（记忆进化）**。

| 维度 | 实现要点 | 分 |
|---|---|---|
| **1 类型边界** | 单一 `MemoryNote`：content/keywords/tags/`context`(一句话描述)/links/timestamp/last_accessed/retrieval_count/evolution_history | 5 |
| **2 写入(进化)** | **核心**：`analyze_content()` LLM 抽 keywords/context/tags → `process_memory()` 找 k 近邻 → 进化 LLM 返回 `should_evolve/actions[strengthen,update_neighbor]/suggested_connections/new_context_neighborhood`，**反向覆盖邻居旧 note 的 context/tags** | 5 |
| 3 检索 | `search()` 纯 embedding；`search_agentic()` = embedding + 命中 note 的 links 邻居。⚠ **已知 bug**：ChromaDB 用 L2 当 cosine(#24)、score 语义反转(#23) 未修 | 3 |
| 4 存储 | ChromaDB(既向量库又元数据库)，embedding=all-MiniLM-L6-v2；note 属性序列化进 metadata(读时 literal_eval) | 4 |
| 5 遗忘 | 进化 `update_neighbor` 就地覆盖旧 context/tags(有损)；有 `delete()` 但**无衰减/TTL**，只增不减 | 2 |
| 6 工程 | 研究原型：每条写入 ≥2 次 LLM 调用(慢且贵，p50 检索 0.668s)；无可观测；bug 积压 | 2 |
| 7 评估 | LoCoMo 6 模型 5 类；Temporal 45.85 vs 基线 18.41 大幅领先；但 **多份第三方论文显示 mem0 多数类别反超 A-MEM**，且 LoCoMo 赛道有可复现争议 | 3 |
| 8 生态 | 1k★、MIT、⚠ Issue#16"项目死了吗"无回复、核心 bug 长期未修、近纯学术 | 2 |

**记忆进化伪代码**：
```python
def add_note(content):
    attrs = LLM.analyze_content(content)            # keywords/context/tags
    note  = MemoryNote(content, **attrs)
    neighbors = find_related_memories(note.content, k=5)   # embedding 近邻
    resp = LLM(EVOLVE_PROMPT(note, neighbors))      # should_evolve / actions / ...
    if resp["should_evolve"]:
        if "strengthen" in resp["actions"]:
            note.links += resp["suggested_connections"]
        if "update_neighbor" in resp["actions"]:
            for i, nb in enumerate(neighbors):
                nb.context = resp["new_context_neighborhood"][i]   # 反向覆盖旧 note
```

**总评**：**最值得借鉴的是 schema 设计 + 记忆进化 prompt 协议**(可抄进自己系统)，向量+link 混合检索方向也对。**作为生产库不可用**：写入重度依赖 LLM(慢贵)、检索有未修的度量 bug、无生命周期管理、维护停滞——建议只借鉴其 note schema 与 evolution 逻辑，存储/检索层自行重写（对照 mem0 工程化）。

---

## 5. Letta / MemGPT —— OS 式分层（23k★ Apache-2.0，v0.16.8）

**范式定位**：把上下文窗口当 OS 的 RAM，用**分页/换页**管理记忆。in-context 的 core memory(LLM 可自改) + out-of-context 的 recall(对话历史)和 archival(向量库),全部由 **stateful agent server** 持久化到 Postgres。是 MemGPT 论文的工程化产品形态(2024 改名 Letta)。

| 维度 | 实现要点 | 分 |
|---|---|---|
| **1 类型边界** | **核心**：core(RAM,persona/human block,常驻 prompt,字符 limit) / recall(对话历史,可搜) / archival(主动写入知识,向量,无上限)；main context = system + working context + FIFO queue | 5 |
| 2 写入 | LLM 用工具自管：`core_memory_append/replace`、`archival_memory_insert(content,tags)`；**换出触发**：prompt 达上下文 70% 注入警告→LLM 主动归档，100% flush ~50% queue 并递归摘要 | 5 |
| 3 检索 | `archival_memory_search(query,tags,page)` 向量检索、`conversation_search` 搜 recall；结果进 FIFO queue 即"page-in"，LLM 再决定固化到 core | 4 |
| 4 存储 | stateful server，state 全在 DB(可多实例水平扩展)；pip 默认 SQLite(不支持迁移)，生产用 Postgres+pgvector(`LETTA_PG_URI`) | 5 |
| 5 遗忘 | core 满靠 LLM `replace` 压缩；recall 递归摘要;archival **无上限单调增长**；**Sleep-time agents**(idle 时后台重组记忆)是相对论文的演进 | 3.5 |
| 6 工程/可观测 | **ADE**(可视化 IDE，实时看/改 core blocks、上下文层级、工具调用)；REST + Py/TS SDK；每 agent 持久化为对象 | 5 |
| 7 评估 | MemGPT 论文 DMR：+GPT-4 Turbo 93.4%(裸 35.3%)；nested-KV 全程 100%；但 DMR 已饱和(Zep 报 94.8%) | 4 |
| 8 生态 | 23.3k★、177 releases、Apache-2.0、迭代活跃 | 5 |

**memory block 结构**：`{label, description, value(整块覆盖), limit(字符上限), read_only}` + 元数据 `chars_current/chars_limit`。

**总评**：需要跨会话长期个性化 + 记忆状态**可视化可审计** + agent 自维护画像的有状态助手；想开箱即用 stateful server + ADE 而非自拼 RAG 栈。**局限**：写入/换页强依赖 LLM 主动发起(召回漏检 + token/延迟成本)；archival 无淘汰单调增长；复杂关系记忆不如图记忆精准；生产必须上 Postgres+pgvector。

---

## 6. mem0 —— 事实抽取式（58k★ Apache-2.0，赛道事实标准）

**范式定位**：不存原始对话,LLM **抽取候选事实** → 与已有记忆比对做 **ADD/UPDATE/DELETE/NOOP** LLM 决策。开源 Agent 记忆事实标准之一,但 benchmark 数字有公开争议。

| 维度 | 实现要点 | 分 |
|---|---|---|
| 1 类型边界 | 事实抽取(丢原文)；四级作用域 `user_id/run_id/agent_id/app_id`；base(向量) vs **mem0^g**(Neo4j 图,实体-关系,时序更强) | 4 |
| **2 写入(add)** | **核心**：①抽取候选事实 ②对每条取 top-k 相似旧记忆,LLM function-call 选 **ADD/UPDATE/DELETE/NOOP**(冲突合并交给 LLM 而非规则)；`infer=false` 可逐字原样存；⚠ v3 平台改异步 ADD-only(返回 PENDING+event_id 轮询) | 4.5 |
| 3 检索 | 向量相似 + 元数据过滤(作用域)，返回 `score`；排序 user→session→raw；v3 平台升级 hybrid(向量+BM25+实体) | 4 |
| 4 存储 | 三层:向量库(Qdrant 默认,可换) + 图(Neo4j,可选) + 历史 DB(SQLite 审计) | 4.5 |
| 5 遗忘 | `update/delete/delete_all/history`(审计轨迹)；DELETE 决策=隐式遗忘；`run_id` 称自动过期但**无明确 TTL 配置** | 3.5 |
| 6 工程/可观测 | OSS 自托管 vs Platform；Py/TS SDK + CLI；API add/search/get_all/update/delete/history;深度可观测需自建 | 4 |
| **7 评估** | ⚠⚠ **重点争议**：自报 LoCoMo "比 OpenAI 内置 +26%/p95 -91%/token -90%"(新版自报 91.6)，但 **Zep 公开反驳**(指其集成 Zep 有误、full-context 反而 ~73% > mem0 ~68%)+ **官方 issue #3944 复现仅 ~0.20**(时间戳 bug)。**别把数字当既定事实** | 2.5 |
| 8 生态 | ~58.6k★(头部)、Apache-2.0、活跃、ECAI 2025、常作对比基准 | 4.5 |

**add() 返回(v1.1)**：`{"results":[{"id","memory","event":"ADD|UPDATE|DELETE|NONE","previous_memory"?}]}`。

**总评**：长期个性化助手/多轮 Agent，需跨 session 记偏好且 token 敏感(抽取压缩上下文)；想开箱即用、Apache-2.0、双 SDK、多租户隔离。**局限**：① **benchmark 可信度有实质争议**(Zep 反驳 + #3944 复现失败),选型勿轻信自报数字；② 抽取丢原文 + 每次 add 有 LLM 成本,时序处理出过 bug；③ 无内建 TTL/衰减；④ v3 异步化增集成复杂度。

---

## 横向对比矩阵

| 维度 \ 项目 | Letta | mem0 | Graphiti | Cognee | Memobase | A-MEM |
|---|---|---|---|---|---|---|
| 范式 | OS 分层 | 事实抽取 | 时序图谱 | 图谱构建层 | 用户画像 | 自组织 note |
| 记忆单元 | core/recall/archival | 抽取的事实 | 节点+事实边 | 实体+边+chunk | profile 槽位+event | atomic note |
| 检索 | archival 向量+换页 | 向量+过滤(score) | hybrid(零 LLM) | 图/向量/NL→Cypher | profile=SQL,event=向量 | embedding+link |
| 遗忘机制 | 递归摘要+sleep-time | UPDATE/DELETE 决策(无TTL) | **双时态软失效**⭐ | 全量 prune | UPDATE 覆盖+重组 | 进化覆盖(无TTL) |
| 写入成本 | 高(LLM 自管) | 中高(抽取+决策 LLM) | 高(每边 LLM) | 高(cognify LLM) | 中(flush 批量) | 高(每条≥2 LLM) |
| schema 亮点 | memory block | event ADD/UPDATE/DELETE | 边挂 fact+四时间戳 | DataPoint+本体 | topic/sub_topic/content | note+evolution |
| 评估可信 | DMR 已饱和 | ⚠**有复现争议** | 厂商自评 | 含超参过拟合 | 自报(Temporal强) | 第三方反超 |
| 生产可用 | ✅ 成熟 | ✅ 成熟(争议) | ✅ 较成熟 | ✅ 较成熟 | ✅ 中上 | ❌ 研究原型 |
| 综合 | ~4.6 | ~3.9 | ~4.5 | ~4.1 | ~4.4 | ~3.25 |

## 检索实现对比（各家 search 到底怎么做）

> 共性：除 Graphiti(hybrid)外，多数核心检索仍是"向量相似 + 过滤"，图/link 是增强。

| 项目 | 检索机制 | 过滤维度 | rerank/融合 | 返回 |
|---|---|---|---|---|
| **Letta** | archival 向量(embedding 相似)；recall 走 `conversation_search` | tags、page | — | passage 进 FIFO queue(page-in)，LLM 再决定固化 |
| **mem0** | query 向量化 → ANN(Qdrant) | user/agent/run_id + metadata + `threshold` | v3 hybrid(向量+BM25+实体)；base 无 | `memory` + `score` |
| **Graphiti** | **hybrid**：向量 + BM25 + 图遍历 | group_id + SearchFilters | **RRF 融合** + 可选 mmr/cross_encoder/**node_distance**/episode_mentions | EntityEdge 列表(检索期**零 LLM**) |
| **Cognee** | `SearchType` 切换 | dataset/node_set | 依类型 | GRAPH_COMPLETION(图+LLM 生成)/RAG(向量)/CHUNKS/INSIGHTS(三元组)/CYPHER/NATURAL_LANGUAGE |
| **Memobase** | profile=**纯 SQL 拼接**(免 embedding,<100ms)；event=pgvector 向量 | only_topics/prefer_topics | profile 可选 LLM 语义重排 | `profile()` 结构化 JSON / `context()` 可直接拼 prompt 的串 |
| **A-MEM** | `search()` 纯 embedding(ChromaDB)；`search_agentic()`=embedding + 命中 note 的 links 邻居 | — | ⚠ 实现用 L2 当 cosine(bug #24) | note + metadata |

**mem0 `search()` 详细流程**（最常被问）：
```text
search(query, user_id=, limit=, threshold=, filters=)
 1. query → embedding(默认 text-embedding-3-small)
 2. 向量库 ANN 相似检索取候选
 3. 按 user_id/agent_id/run_id + metadata 过滤(多租户隔离)
 4. threshold(相似度下限) + limit 截断；隐式排序 user→session→raw
 5. 返回 [{id, memory(事实文本), score, metadata, created_at, updated_at}]
```
mem0 检索的是 **add() 阶段已抽取/去重/去冲突后的"事实"**，不是原始对话——所以只需召回少量高质量事实即可，这是它省 token 的根因。开启 mem0^g(Neo4j) 后额外用实体-关系图遍历增强时序/跨事实推理。

---

## 记忆记录 schema 对照 + 推荐自建 schema

**6 家记忆单元字段对照**

| 项目 | 记忆单元 | 关键字段 | 去重/版本 | 是否带向量 |
|---|---|---|---|---|
| Letta | memory block / archival passage | `label/value/limit/description/read_only` | — | archival 带 |
| mem0 | 事实记录 | `id/memory/hash(MD5)/metadata/categories/user_id,agent_id,run_id/created_at/updated_at/score` | MD5 去重 | 是 |
| Graphiti | EntityEdge(边=事实) | `fact/fact_embedding/episodes/valid_at/invalid_at/expired_at/created_at/group_id` | 时态版本 | 是 |
| Cognee | DataPoint | `id/created_at/updated_at/version/source_content_hash/metadata{index_fields}/ontology_valid` | content_hash + version | index_fields 决定 |
| Memobase | profile 槽 + event | profile:`content TEXT`+`attributes JSONB`(topic/sub_topic)；event:`event_data JSONB`+`Vector` | UPDATE 覆盖 | 仅 event 带 |
| A-MEM | MemoryNote | `content/keywords/tags/context/links/timestamp/last_accessed/retrieval_count/evolution_history` | 进化覆盖 | ChromaDB |

**推荐自建 schema（融合 6 家优点，可直接建表）**
```sql
CREATE TABLE memory_record (
  id            UUID PRIMARY KEY,
  content       TEXT NOT NULL,          -- 抽取后的事实(mem0/Memobase 思路,非原文堆叠)
  content_hash  CHAR(64),              -- 去重(mem0=MD5)
  embedding     VECTOR(1536),          -- 放向量库
  user_id TEXT, agent_id TEXT, run_id TEXT,  -- 四级作用域(mem0)
  memory_type   TEXT,                  -- episodic|semantic|procedural(A-MEM/Letta 分层思想)
  topic TEXT, sub_topic TEXT,          -- 结构化画像槽(Memobase)
  categories    TEXT[], keywords TEXT[], links UUID[],  -- 标签+自链接(A-MEM)
  source        TEXT, confidence REAL, -- 溯源(Graphiti episodes)+置信度
  valid_at TIMESTAMPTZ, invalid_at TIMESTAMPTZ,  -- 双时态软遗忘(Graphiti)⭐
  created_at TIMESTAMPTZ NOT NULL, updated_at TIMESTAMPTZ, last_accessed_at TIMESTAMPTZ,
  ttl INTERVAL, is_deleted BOOLEAN DEFAULT FALSE,
  metadata JSONB
);
-- 配套:历史审计表(mem0 history:old/new/event) + 可选图层(实体-关系)
```
> 完整 schema 设计推导见 [[ai-memory-implementation-survey]] 专题 C。

---

## 离线合成 / 记忆巩固对比（实时 vs 后台提炼）

> 对照 survey 里"各家 Dreaming"：商用侧 OpenAI(Dreaming)、Anthropic Managed Agents(Dreams)、**Claude Code(autoDream，扒二进制 v2.1.177 证实：≥24h+≥5 会话、夜间 1–5am 后台巩固 memory，服务端灰度+官方未文档化)** 都有后台离线合成；OSS 这边——

| 项目 | 写入时机 | 离线/批量巩固 |
|---|---|---|
| **Letta** | 实时(LLM 工具) | ✅ **Sleep-time agents**：idle 时后台重组/精炼 block(consolidation) |
| **A-MEM** | 实时(每条触发邻居进化) | ✅ `consolidate_memories()` 每 `evo_threshold`(默认100) 次重建 collection |
| **Memobase** | insert 进 buffer | ✅ 半离线：异步 flush 批量抽取 + re-summarize/reorganize 巩固 |
| **Cognee** | `add→cognify` | ◐ cognify 可 `run_in_background`；`memify()` 后处理 enrichment |
| **mem0** | add 同步 AUDN | ◐ v3 平台改异步队列；无专门"巩固"阶段 |
| **Graphiti** | 实时增量(每边 LLM 失效判断) | ❌ 无离线巩固——失效是**实时**判定的(写入即处理) |

要点：**Letta sleep-time / A-MEM evolution 是 OSS 里最接近"dreaming 式离线提炼"的**；Graphiti 走相反路线（实时失效，无需离线整理）。

---

## Benchmark landscape（评测体系 + 可复现争议）

| Benchmark | 测什么 | 现状 |
|---|---|---|
| **DMR**(MemGPT/MSC) | 跨会话事实回忆 | 已饱和(90%+,MemGPT 93.4%/Zep 94.8%)，区分度低 |
| **LoCoMo** | 长对话 QA,5 类(single/multi-hop/temporal/open-domain/adversarial)，~16–26k token | 主流但**被批太弱**(现代上下文可直接塞下)+ adversarial 类缺 ground truth |
| **LongMemEval** | 更难、更贴企业长程 | 新主力,区分度更好 |
| **BEAM(1M)** | 超长上下文记忆 | 较新 |

**⚠ 可复现争议（选型必读）**：
- mem0 自报 LoCoMo "+26%/-91%/91.6"，但 **Zep 公开反驳**(指其集成 Zep 有误、full-context baseline ~73% 反而高于 mem0 ~68%)。
- **mem0 官方 issue #3944** 复现仅 ~0.20(时间戳 bug)。
- Zep 自己 84% 也被第三方修正到 58.44%。
- 多份第三方论文显示 **mem0 在多数 LoCoMo 类别反超 A-MEM**。

**结论：所有厂商自报记忆 benchmark 都应标注"自评 + 有复现争议"，认真选型须自己用固定模型版本跑 ≥5 次取均值。**

---

## 能直接借鉴到自己记忆层的点

- **遗忘**：抄 Graphiti 的 **双时态**（`valid_at/invalid_at` 区分"世界变了"，`expired_at` 区分"认知改了"）——比硬删除强得多。
- **schema**：抄 Memobase 的 **topic/sub_topic/content 画像槽位**(产品化友好) + A-MEM 的 **note 富属性**(keywords/context/links)。
- **写入合并**：抄 mem0/Memobase 的 **ADD/UPDATE/DELETE(ABORT)** LLM 决策，避免重复堆叠。
- **检索**：抄 Graphiti 的 **hybrid + RRF + 检索期零 LLM**(成本/延迟最优)。
- **写入管线**：抄 Cognee 的 **Task DAG**(ECL 可组合)做工程化抽取。

## ⚠ 核对清单（写公开内容前）
1. 各项目 star/release/最近 push 为 2026-06 数量级，以当日 GitHub 为准。
2. 所有 benchmark 数字多为**项目自报**(LoCoMo 赛道有公认可复现争议：Zep 反驳 mem0、mem0 #3944 复现失败)——引用须标注自报性质。
3. Graphiti 时态 bug(#920/#1489)、A-MEM 检索 bug(#23/#24) 状态可能已变。
4. Memobase "<80ms" 未证实(官方 <100ms)；Cognee 融资金额未查到；Cognee `SearchType` 枚举跨版本有出入。
5. 字段名取自当时源码，版本更新可能变化。

## 来源（按项目）
- **Graphiti**：[repo](https://github.com/getzep/graphiti)、[Zep 论文 arXiv:2501.13956](https://arxiv.org/abs/2501.13956)、[edges.py](https://github.com/getzep/graphiti/blob/main/graphiti_core/edges.py)、[DeepWiki](https://deepwiki.com/getzep/graphiti)
- **Cognee**：[repo](https://github.com/topoteretes/cognee)、[cognify.py](https://github.com/topoteretes/cognee/blob/v1.1.0/cognee/api/v1/cognify/cognify.py)、[DataPoint.py](https://github.com/topoteretes/cognee/blob/main/cognee/infrastructure/engine/models/DataPoint.py)、[Dreamify arXiv:2505.24478](https://arxiv.org/abs/2505.24478)
- **Memobase**：[repo](https://github.com/memodb-io/memobase)、[database.py](https://github.com/memodb-io/memobase/blob/main/src/server/api/memobase_server/models/database.py)、[LoCoMo README](https://github.com/memodb-io/memobase/blob/main/docs/experiments/locomo-benchmark/README.md)
- **A-MEM**：[repo](https://github.com/agiresearch/A-mem)、[论文 arXiv:2502.12110](https://arxiv.org/abs/2502.12110)、[memory_system.py](https://github.com/agiresearch/A-mem/blob/main/agentic_memory/memory_system.py)、[Zep 对 LoCoMo 的反驳](https://blog.getzep.com/lies-damn-lies-statistics-is-mem0-really-sota-in-agent-memory/)
