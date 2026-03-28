#!/usr/bin/env python3
"""
v3_vector_db.py — 向量数据库：持久化 + 元数据过滤 + 增量入库
===============================================================
v2 遗留的三个问题，v3 全部解决：

  ✗ v2 问题 1: chunks 只在内存，程序退出就消失
  ✓ v3 解决:   ChromaDB 把向量写到磁盘，重启直接加载，不重算

  ✗ v2 问题 2: 每次运行重算所有 embedding（慢 + 费钱）
  ✓ v3 解决:   入库一次，检索 N 次，embedding 只算一遍

  ✗ v2 问题 3: 无法增量更新，新文档 = 全量重算
  ✓ v3 解决:   upsert 操作，只算新增文档的 embedding

本文件新增的核心概念：
  Collection   = 命名空间，隔离不同知识库（部门/项目/版本）
  HNSW         = 为什么百万向量能毫秒检索（不是暴力遍历）
  Metadata     = 每个 chunk 的标签，支持过滤检索
  upsert       = 有则更新，无则插入（增量入库的安全操作）

依赖：pip install openai numpy chromadb
运行：python v3_vector_db.py
      第一次运行: 建库 + 入库 + 检索
      第二次运行: 直接加载（不重算 embedding）+ 演示增量入库
"""

import os
import time
import re
import chromadb
from rag_provider import embed, chat, cosine_sim, model_info

# ──────────────────────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────────────────────

DB_PATH        = "./rag_v3_db"          # ChromaDB 数据目录，持久化在磁盘
COLLECTION_NAME = "rag_course_v3"       # Collection 名称，类似数据库的"表名"
TOP_K          = 3                      # 检索返回数量


# ──────────────────────────────────────────────────────────────
# 知识库文档（模拟两个来源：技术文档 + 产品手册）
# ──────────────────────────────────────────────────────────────

# 文档 A：技术文档（来自工程团队）
DOC_A = {
    "source": "tech_spec",
    "title":  "RAG 系统技术规范 v2.0",
    "content": """RAG系统技术规范

一、检索架构
RAG系统采用两阶段检索架构。第一阶段使用向量检索（Dense Retrieval）召回候选集，通过Embedding模型将查询和文档块转换为高维向量，计算余弦相似度排序。第二阶段使用Reranker模型对候选集精排，Cross-encoder架构逐对计算查询与文档的相关性分数，返回最终Top-K结果。

二、分块策略
生产环境推荐分块参数：chunk_size=300到500字符，overlap=50到100字符（约chunk_size的15%到20%）。技术文档使用句子感知分块，代码文档按函数边界分块，PDF报告按段落分块。分块质量直接决定检索质量上限，需通过Recall@K指标量化评估。

三、向量数据库选型
本地开发推荐ChromaDB，零配置，支持持久化。生产环境推荐Qdrant，支持payload过滤、HNSW索引优化、水平扩展。HNSW（Hierarchical Navigable Small World）算法通过构建多层导航图实现近似最近邻搜索，在百万级向量规模下可达毫秒级检索延迟，查询复杂度为O(log N)。

四、评估体系
核心检索指标：Recall@K（前K个结果中正确答案的占比）和MRR（平均倒数排名）。生成指标使用RAGAS框架评估Faithfulness（忠实度）和Answer Relevancy（相关性）。生产部署前必须在Golden Dataset上完成评估，Golden Dataset需包含真实用户查询，手工标注相关文档对。"""
}

# 文档 B：产品手册（来自产品团队）
DOC_B = {
    "source": "product_manual",
    "title":  "企业 AI 知识库产品手册",
    "content": """企业AI知识库产品手册

一、产品价值
企业AI知识库基于RAG技术，解决大模型两大核心痛点：一是知识截止日期问题，企业文档实时更新，AI回答同步更新；二是私有知识缺失问题，将企业内部资料纳入AI的"视野"。相比传统搜索，AI知识库能理解语义、跨文档综合推理，而非简单的关键词匹配。

二、适用场景
客服知识库：将产品手册、FAQ、政策文件入库，客服可用自然语言查询。合规审查：法规文件入库后，AI可快速定位相关条款，给出引用来源。研发文档：代码注释、设计文档、架构决策记录入库，新成员快速上手。销售赋能：产品文档、竞品分析、案例库入库，售前可快速组织方案。

三、权限与隔离
多租户架构下，不同部门的知识库通过Collection隔离，销售部门只能检索销售Collection，研发只能检索研发Collection。Metadata字段支持细粒度权限控制，例如按文档密级（公开/内部/机密）过滤检索结果。

四、知识库维护
文档更新流程：新文档上传后自动触发解析、分块、Embedding计算，增量写入向量库，不影响在线检索服务。旧文档删除：根据document_id精确删除对应chunks，无需重建整个知识库。版本管理：通过Metadata中的version字段标记文档版本，检索时可指定版本范围。"""
}

DOCUMENTS = [DOC_A, DOC_B]


# ──────────────────────────────────────────────────────────────
# 分块函数（复用 v2 的句子感知分块，这里简化版）
# ──────────────────────────────────────────────────────────────

def chunk_by_sentence(text: str, max_chars: int = 280) -> list[str]:
    """句子感知分块，复用 v2 的策略"""
    sentences = re.split(r'(?<=[。！？；\n])', text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s) > 5]
    chunks, current = [], ""
    for s in sentences:
        if len(current) + len(s) <= max_chars:
            current += s
        else:
            if current:
                chunks.append(current.strip())
            current = s
    if current.strip():
        chunks.append(current.strip())
    return [c for c in chunks if len(c) > 20]


# ──────────────────────────────────────────────────────────────
# ChromaDB 工具函数
# ──────────────────────────────────────────────────────────────

def get_collection(db_path: str, name: str):
    """
    创建或加载 ChromaDB Collection。

    Collection 是什么？
    → 向量库里的"命名空间"，类似数据库里的"表"。
    → 不同 Collection 完全隔离，适合多部门/多项目场景。
    → 同一个 Collection 里的 chunk 共享 embedding 维度（必须一致）。

    chromadb.PersistentClient 会把数据写到磁盘，程序退出后数据不丢失。
    下次启动直接 get_or_create_collection，已有数据自动加载。
    """
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(
        name=name,
        # 使用余弦相似度（和 v1/v2 一致）
        # ChromaDB 内部存向量，计算相似度时自动处理归一化
        metadata={"hnsw:space": "cosine"}
    )
    return collection


def upsert_document(collection, doc: dict) -> int:
    """
    将一篇文档分块后入库。

    upsert = update + insert：
    → 如果 ID 已存在 → 更新（文档修订时用）
    → 如果 ID 不存在 → 插入（新文档）
    → 比 insert 安全：重复运行不会报错或产生重复数据

    每个 chunk 存入：
    → id:        全局唯一标识，"来源_序号" 格式
    → embedding: 1024 维向量（由 rag_provider.embed() 计算）
    → document:  chunk 原文（用于生成回答时展示）
    → metadata:  标签，支持后续过滤检索
    """
    chunks = chunk_by_sentence(doc["content"])
    if not chunks:
        return 0

    ids, embeddings, documents, metadatas = [], [], [], []

    for i, chunk in enumerate(chunks):
        chunk_id = f"{doc['source']}_{i:04d}"
        emb = embed(chunk)

        ids.append(chunk_id)
        embeddings.append(emb.tolist())     # ChromaDB 接受 list，不接受 np.array
        documents.append(chunk)
        metadatas.append({
            "source":  doc["source"],       # 来源标识（过滤用）
            "title":   doc["title"],        # 文档标题（展示用）
            "chunk_i": i,                   # 块序号（调试用）
        })

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )
    return len(chunks)


def query_collection(collection, query: str, top_k: int = 3,
                     where: dict | None = None) -> list[dict]:
    """
    检索向量库。

    where 参数 = Metadata 过滤条件（在 ANN 检索结果里做二次过滤）：
      where=None                      → 检索全库
      where={"source": "tech_spec"}   → 只检索技术文档
      where={"source": "product_manual"} → 只检索产品手册

    ChromaDB 的 HNSW 索引：
    → 不是暴力遍历所有向量，而是建了一个多层"导航图"
    → 从顶层粗粒度导航到底层精细候选，O(log N) 复杂度
    → 百万向量 < 10ms，暴力遍历可能需要几秒
    """
    q_emb = embed(query)

    kwargs = {
        "query_embeddings": [q_emb.tolist()],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)

    # ChromaDB 返回嵌套列表（支持批量查询），这里只取第一个查询的结果
    docs      = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]     # cosine distance，不是相似度！

    return [
        {
            "text":     docs[i],
            "source":   metas[i]["source"],
            "title":    metas[i]["title"],
            "score":    round(1 - distances[i], 4),  # distance → similarity
        }
        for i in range(len(docs))
    ]


# ──────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────

def main():
    LINE = "─" * 62
    info = model_info()

    print(f"\n{'═'*62}")
    print(f" V3 · 向量数据库持久化")
    print(f" 提供商: {info['provider']}  |  Embedding: {info['embed_model']}")
    print(f"{'═'*62}")

    collection = get_collection(DB_PATH, COLLECTION_NAME)
    existing_count = collection.count()

    # ────────────────────────────────────────────────────────
    # STEP 1: 判断是否需要入库
    # 关键体验：第二次运行直接跳过入库，不重算 embedding
    # ────────────────────────────────────────────────────────
    print(f"\n{'═'*62}")
    print(" STEP 1 ／ 检查知识库状态")
    print(f"{'═'*62}")
    print(f"\n  数据库路径: {DB_PATH}")
    print(f"  Collection:  {COLLECTION_NAME}")
    print(f"  当前 chunk 数: {existing_count}")

    if existing_count == 0:
        # ── 第一次运行：入库 ──
        print(f"\n  → 知识库为空，开始入库（会调用 Embedding API）...\n")
        total_chunks = 0
        t0 = time.time()

        for doc in DOCUMENTS:
            print(f"  处理文档: 「{doc['title']}」")
            n = upsert_document(collection, doc)
            print(f"    → 分成 {n} 个 chunk，已 upsert 到向量库")
            total_chunks += n

        elapsed = round(time.time() - t0, 2)
        print(f"\n  ✅ 入库完成")
        print(f"     总 chunk 数: {collection.count()}")
        print(f"     耗时: {elapsed}s（含 Embedding API 调用）")
        print(f"\n  💡 下次运行，这一步会被跳过，直接从磁盘加载。")
        print(f"     无论跑多少次，Embedding API 只调用这一次。")

    else:
        # ── 第二次以后运行：直接加载 ──
        print(f"\n  → 知识库已有 {existing_count} 个 chunk，直接加载（不重算 embedding）")
        print(f"  ✅ 加载完成，0 次 API 调用")
        print(f"\n  💡 这就是向量数据库持久化的价值：")
        print(f"     入库 1 次 → 检索 N 次，API 费用只花一遍。")
        print(f"     加一篇新文档，只算新文档的 embedding，旧数据不动。")

    # ────────────────────────────────────────────────────────
    # STEP 2: 基础语义检索（全库）
    # 和 v1/v2 对比：现在检索的是持久化的向量库，不是内存数组
    # ────────────────────────────────────────────────────────
    print(f"\n{'═'*62}")
    print(" STEP 2 ／ 基础检索（全库）")
    print(f"{'═'*62}")

    QUERY_1 = "HNSW 是什么，为什么向量检索这么快"

    print(f"\n  问题: {QUERY_1}\n")
    results_1 = query_collection(collection, QUERY_1, top_k=TOP_K)

    print(f"  Top-{TOP_K} 检索结果：")
    print(LINE)
    for i, r in enumerate(results_1):
        print(f"  [{i+1}] 相似度={r['score']}  来源={r['source']}")
        print(f"       {r['text'][:80]}...")
    print(LINE)

    print(f"\n  💡 注意：结果来自两个不同来源（tech_spec + product_manual）。")
    print(f"     STEP 3 会演示「只查某个来源」的 Metadata 过滤。")

    # ────────────────────────────────────────────────────────
    # STEP 3: Metadata 过滤检索（核心新能力）
    # 场景：企业多部门知识库，不同角色只看自己权限范围内的内容
    # ────────────────────────────────────────────────────────
    print(f"\n{'═'*62}")
    print(" STEP 3 ／ Metadata 过滤检索（企业级核心能力）")
    print(f"{'═'*62}")

    QUERY_2 = "多租户和权限隔离怎么做"

    print(f"\n  问题: {QUERY_2}\n")

    # 无过滤：全库检索
    print(f"  场景 A: 全库检索（无过滤）")
    results_all = query_collection(collection, QUERY_2, top_k=2)
    for r in results_all:
        print(f"  ✦ [{r['score']}][{r['source']:15s}] {r['text'][:65]}...")

    print()

    # 只查技术文档
    print(f"  场景 B: 只查技术文档 where source=tech_spec")
    results_tech = query_collection(
        collection, QUERY_2, top_k=2,
        where={"source": "tech_spec"}
    )
    for r in results_tech:
        print(f"  ✦ [{r['score']}][{r['source']:15s}] {r['text'][:65]}...")

    print()

    # 只查产品手册
    print(f"  场景 C: 只查产品手册 where source=product_manual")
    results_prod = query_collection(
        collection, QUERY_2, top_k=2,
        where={"source": "product_manual"}
    )
    for r in results_prod:
        print(f"  ✦ [{r['score']}][{r['source']:15s}] {r['text'][:65]}...")

    print(f"""
  💡 Metadata 过滤的工程价值：
     · 多租户隔离：销售 Collection 和研发 Collection 物理隔离
     · 权限控制：where {{"clearance": "public"}} 过滤出公开文档
     · 版本管理：where {{"version": "2.0"}} 只检索最新版本
     · 时间过滤：where {{"updated_at": {{"$gte": "2024-01-01"}}}} 过滤旧文档
     这些能力用内存数组（v2）完全实现不了，是向量数据库的核心价值。""")

    # ────────────────────────────────────────────────────────
    # STEP 4: 增量入库（upsert 演示）
    # 场景：产品手册更新了新章节，只需要入库新内容
    # ────────────────────────────────────────────────────────
    print(f"\n{'═'*62}")
    print(" STEP 4 ／ 增量入库：新文档只算新 embedding")
    print(f"{'═'*62}")

    # 模拟"新增一篇竞品分析文档"
    NEW_DOC = {
        "source": "competitive_analysis",
        "title":  "RAG 竞品分析报告 2025",
        "content": """RAG竞品分析报告

一、主流方案对比
目前市场上主流RAG产品包括：LangChain（开源框架，灵活但需自行工程化）、LlamaIndex（文档处理能力强，适合复杂文档结构）、Dify（低代码平台，非技术人员友好）、RAGFlow（专注文档解析，PDF处理能力强）。

二、技术差异
在检索质量方面，RAGFlow的PDF解析基于视觉模型，能处理表格和图文混排，优于基于文本的解析方案。在评估体系方面，只有少数产品内置了RAGAS评估框架。在向量数据库集成方面，各产品对Chroma、Qdrant、Pinecone的支持程度不一。

三、选型建议
学习和原型阶段推荐LangChain或LlamaIndex，有完整的教程生态。企业生产部署推荐基于开源组件自建，避免供应商锁定。如果团队没有工程师，Dify是最快上线的选择。"""
    }

    count_before = collection.count()
    print(f"\n  当前 chunk 数: {count_before}")
    print(f"  新增文档: 「{NEW_DOC['title']}」")
    print(f"  （只会调用新文档的 Embedding API，旧数据不动）\n")

    t1 = time.time()
    n_new = upsert_document(collection, NEW_DOC)
    elapsed_new = round(time.time() - t1, 2)

    count_after = collection.count()
    print(f"  ✅ 增量入库完成")
    print(f"     新增 {n_new} 个 chunk  |  耗时 {elapsed_new}s")
    print(f"     库内 chunk 数: {count_before} → {count_after}")

    # 验证新文档可以被检索到
    print(f"\n  验证：搜索新文档内容")
    QUERY_3 = "RAGFlow 和 LangChain 有什么区别，怎么选"
    results_new = query_collection(collection, QUERY_3, top_k=2)
    for r in results_new:
        icon = "🆕" if r["source"] == "competitive_analysis" else "  "
        print(f"  {icon} [{r['score']}][{r['source']:22s}] {r['text'][:60]}...")

    print(f"\n  💡 增量入库 = 向量数据库最重要的工程特性之一")
    print(f"     旧数据：touch 0 次  |  新数据：只算新文档的 embedding")
    print(f"     企业知识库每天都有新文档，这个特性决定了运营成本。")

    # ────────────────────────────────────────────────────────
    # STEP 5: RAG 完整回答（把 v1 的 Prompt 注入接上来）
    # ────────────────────────────────────────────────────────
    print(f"\n{'═'*62}")
    print(" STEP 5 ／ 完整 RAG 回答（检索 + 生成）")
    print(f"{'═'*62}")

    QUERY_FINAL = "企业部署 RAG 知识库，向量数据库该怎么选型？"
    print(f"\n  问题: {QUERY_FINAL}\n")

    contexts = query_collection(collection, QUERY_FINAL, top_k=3)

    print(f"  检索到的上下文：")
    ctx_text = ""
    for i, c in enumerate(contexts):
        print(f"  [{i+1}][{c['source']}] {c['text'][:60]}...")
        ctx_text += f"[{i+1}] {c['text']}\n"

    prompt = f"""你是一个严谨的技术顾问。请仅根据以下参考资料回答问题，不要添加资料之外的内容。

参考资料：
{ctx_text}
问题：{QUERY_FINAL}
回答："""

    print(f"\n  🤖 AI 回答（基于检索内容）：")
    answer = chat(prompt)
    print(f"  ┌{'─'*56}")
    for line in answer.split('\n'):
        print(f"  │ {line}")
    print(f"  └{'─'*56}")

    # ────────────────────────────────────────────────────────
    # 总结
    # ────────────────────────────────────────────────────────
    print(f"\n{'═'*62}")
    print(" V3 总结")
    print(f"{'═'*62}")
    print(f"""
  你现在理解的概念：
    ✓ Collection：向量库的命名空间，隔离不同知识库
    ✓ 持久化：入库一次，检索 N 次，重启不丢数据
    ✓ upsert：有则更新，无则插入，增量入库的安全操作
    ✓ Metadata 过滤：按来源/版本/权限精确检索
    ✓ HNSW：O(log N) 近似最近邻，百万向量毫秒级检索

  V3 的局限（V4 要解决）：
    ✗ 用的是通用 Embedding 模型（text-embedding-3-small / BGE-M3）
    ✗ 中文专业术语、行业词汇的检索质量可能不够好
    ✗ 没有对比不同 Embedding 模型的效果差异
    → V4 解决：Embedding 模型选型，领域适配，用 Recall@K 量化对比

  目前进度：v1 ✓  v2 ✓  v3 ✓  v3.5 ✓（黄金数据集）
  下一步：v4 Embedding 模型选型与量化对比
    """)

    print(f"  数据库文件位置: {os.path.abspath(DB_PATH)}/")
    print(f"  Collection 名: {COLLECTION_NAME}")
    print(f"  当前总 chunk 数: {collection.count()}")
    print(f"\n  提示：删除 {DB_PATH}/ 目录可清空知识库，下次重新入库。\n")


if __name__ == "__main__":
    main()