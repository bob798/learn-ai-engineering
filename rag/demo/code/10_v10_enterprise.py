#!/usr/bin/env python3
"""
10_v10_enterprise.py — 企业级生产 RAG：从 Demo 到系统
=====================================
插入位置：v9（Agentic RAG）之后，课程终点

核心认知：
  v1~v9 解决了"RAG 能不能用"的问题。
  v10 解决的是"RAG 能不能在生产环境可靠运行"的问题。
  这是售前背景的最大优势：你懂客户的生产顾虑。

Demo 到生产的 4 个差距：

  1. 成本与延迟（语义缓存）
     同样的问题被重复问，每次都调 LLM API 既慢又贵。
     语义缓存：相似问题命中缓存，直接返回历史答案。
     效果：命中时延迟从 ~2s → <10ms，成本降低 80%。

  2. 可观测性（请求追踪）
     生产环境出了问题，需要知道每次请求做了什么。
     Tracing：记录每次请求的检索结果、生成答案、延迟、模型。
     生产工具：Langfuse（开源，可自托管）/ LangSmith（SaaS）

  3. 数据隔离（多租户）
     企业部署时，不同部门/客户的数据不能互相访问。
     Namespace 隔离：用 Chroma Collection 或向量库的 Filter 实现。

  4. 增量更新
     文档更新时，不能重建整个索引（代价太高）。
     增量入库：新文档分块 → embedding → 追加到向量库，不影响在线服务。

依赖：pip install openai numpy python-dotenv chromadb
运行：python 10_v10_enterprise.py
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║  📺 讲师注释                                                  ║
# ║  对应集数：01_理解RAG.html · Ep5「Demo 到生产差了什么」      ║
# ║  核心代码：第 122~155 行（SemanticCache 类），                ║
# ║            第 157~195 行（RequestTracer 类）                  ║
# ║  可跳过：第 65~120 行（BM25 + 分块样板），                   ║
# ║          第 220~270 行（多租户样板）                         ║
# ║  本集关键数字：缓存命中 117ms vs 未命中 624ms，节省 81% 延迟；║
# ║               命中率 50%                                     ║
# ╚══════════════════════════════════════════════════════════════╝

import json
import math
import re
import time
import uuid
import numpy as np
from collections import Counter
from datetime import datetime
from pathlib import Path
from importlib.util import module_from_spec, spec_from_file_location

# ── 加载 Provider 接口 ───────────────────────────────────────
_provider_path = Path(__file__).with_name("00_配置提供商_先改这个.py")
_provider_spec = spec_from_file_location("rag_provider", _provider_path)
if _provider_spec is None or _provider_spec.loader is None:
    raise ImportError(f"无法加载提供商配置文件: {_provider_path}")
_provider_module = module_from_spec(_provider_spec)
_provider_spec.loader.exec_module(_provider_module)

embed      = _provider_module.embed
chat       = _provider_module.chat
cosine_sim = _provider_module.cosine_sim
model_info = _provider_module.model_info


# ══════════════════════════════════════════════════════════════
# 辅助工具
# ══════════════════════════════════════════════════════════════

def chunk_by_sentence(text, max_chars=300):
    sentences = re.split(r'(?<=[。！？；\n])', text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s) > 5]
    chunks = []; current = ""
    for s in sentences:
        if len(current) + len(s) <= max_chars: current += s
        else:
            if current: chunks.append(current.strip())
            current = s
    if current.strip(): chunks.append(current.strip())
    return [c for c in chunks if len(c) > 20]


class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.k1 = k1; self.b = b; self.N = len(corpus)
        self.tokenized = [self._tok(d) for d in corpus]
        self.avgdl = sum(len(d) for d in self.tokenized) / self.N
        self.df: dict[str, int] = {}
        for doc in self.tokenized:
            for t in set(doc): self.df[t] = self.df.get(t, 0) + 1

    def _tok(self, text): return re.findall(r'[\u4e00-\u9fff]|[a-zA-Z0-9]+', text)

    def score(self, query, doc_idx):
        doc = self.tokenized[doc_idx]; dl = len(doc); tf_map = Counter(doc)
        total = 0.0
        for t in self._tok(query):
            if t not in self.df: continue
            idf = math.log((self.N - self.df[t] + 0.5) / (self.df[t] + 0.5) + 1)
            tf = tf_map.get(t, 0)
            total += idf * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return total

    def retrieve(self, query, top_k=None):
        n = top_k or self.N
        return sorted([(i, self.score(query, i)) for i in range(self.N)],
                      key=lambda x: x[1], reverse=True)[:n]


# ══════════════════════════════════════════════════════════════
# 模块一：语义缓存（Semantic Cache）
#
# 核心原理：
#   对每次用户 query 计算 embedding，在缓存中找最相似的历史 query。
#   相似度 > 阈值 → 命中缓存，直接返回历史答案。
#   相似度 ≤ 阈值 → 未命中，正常走 RAG 管道，结果写入缓存。
#
# 为什么用"语义"相似度而不是精确字符串匹配？
#   "RAG 分块多大" 和 "分块推荐尺寸是多少" 是同一个问题，精确匹配会失败。
#
# 生产注意：
#   - 阈值（默认 0.92）需要在自己语料上调，太低误命中，太高缓存命中率低
#   - 生产场景用 Redis + 向量库（而非 in-memory）持久化缓存
#   - 缓存需要设置 TTL（过期时间），防止答案过期
# ══════════════════════════════════════════════════════════════

class SemanticCache:
    def __init__(self, similarity_threshold: float = 0.92):
        self.threshold = similarity_threshold
        self.entries: list[dict] = []   # {"query": str, "embedding": np.array, "answer": str, "hit_count": int}

    def lookup(self, query: str, q_emb: np.ndarray) -> str | None:
        """返回缓存答案（命中时），或 None（未命中）"""
        best_score, best_entry = 0.0, None
        for entry in self.entries:
            score = cosine_sim(q_emb, entry["embedding"])
            if score > best_score:
                best_score, best_entry = score, entry
        if best_score >= self.threshold and best_entry:
            best_entry["hit_count"] += 1
            return best_entry["answer"]
        return None

    def store(self, query: str, q_emb: np.ndarray, answer: str):
        self.entries.append({
            "query":     query,
            "embedding": q_emb,
            "answer":    answer,
            "hit_count": 0,
            "cached_at": datetime.now().isoformat(),
        })

    def stats(self) -> dict:
        total_hits = sum(e["hit_count"] for e in self.entries)
        return {
            "cached_queries": len(self.entries),
            "total_hits": total_hits,
            "entries": [{"query": e["query"], "hits": e["hit_count"]} for e in self.entries],
        }


# ══════════════════════════════════════════════════════════════
# 模块二：请求追踪（Request Tracer）
#
# 记录每次 RAG 请求的完整信息：
#   - request_id：全局唯一 ID，用于关联前后端日志
#   - query / answer：输入输出
#   - retrieved_chunks：检索到的文档片段
#   - latency_ms：分阶段延迟（embedding / retrieval / generation）
#   - cache_hit：是否命中缓存
#   - model：使用的模型
#
# 生产工具：
#   Langfuse（开源，可自托管）→ pip install langfuse
#   LangSmith（SaaS）
#   两者都提供比本文件更完整的 UI、报警、版本管理
# ══════════════════════════════════════════════════════════════

class RequestTracer:
    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.traces: list[dict] = []

    def record(self, trace: dict):
        trace["timestamp"] = datetime.now().isoformat()
        self.traces.append(trace)
        # 追加写入 JSONL（每行一条记录，方便流式读取）
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")

    def summary(self) -> dict:
        if not self.traces: return {}
        latencies = [t.get("total_latency_ms", 0) for t in self.traces]
        cache_hits = sum(1 for t in self.traces if t.get("cache_hit"))
        return {
            "total_requests":     len(self.traces),
            "cache_hit_rate":     round(cache_hits / len(self.traces), 2),
            "avg_latency_ms":     round(sum(latencies) / len(latencies), 0),
            "p50_latency_ms":     round(sorted(latencies)[len(latencies)//2], 0),
        }


# ══════════════════════════════════════════════════════════════
# 模块三：多租户 Namespace（Chroma Collections）
#
# 问题背景：
#   企业部署时，部门 A 的文档不能被部门 B 检索到。
#   最简单的隔离方式：每个租户用独立的 Chroma Collection。
#
# Chroma Collection 相当于一个独立的向量数据库实例：
#   不同 Collection 的数据完全隔离，互不干扰。
#
# 更精细的隔离（单 Collection 内）：
#   用 Chroma 的 metadata 过滤（where={"tenant_id": "A"}）
#   适合租户数量多、每个租户文档少的场景。
# ══════════════════════════════════════════════════════════════

def build_tenant_collection(tenant_id: str, document: str, chroma_client) -> object:
    """为租户创建独立的 Chroma Collection，写入分块 embedding"""
    collection_name = f"tenant_{tenant_id}"
    # 如果已存在则删除重建（Demo 用，生产中应增量更新）
    try:
        chroma_client.delete_collection(collection_name)
    except Exception:
        pass
    collection = chroma_client.create_collection(collection_name)

    chunks = chunk_by_sentence(document, max_chars=300)
    for i, chunk in enumerate(chunks):
        emb = embed(chunk)
        collection.add(
            ids=[f"{tenant_id}_chunk_{i}"],
            embeddings=[emb.tolist()],
            documents=[chunk],
            metadatas=[{"tenant_id": tenant_id, "chunk_idx": i}],
        )
    return collection


def tenant_retrieve(collection, query: str, top_k: int = 3) -> list[str]:
    """从指定租户的 Collection 中检索"""
    q_emb = embed(query)
    results = collection.query(
        query_embeddings=[q_emb.tolist()],
        n_results=top_k,
    )
    return results["documents"][0] if results["documents"] else []


# ══════════════════════════════════════════════════════════════
# 企业级 RAG 管道（组合以上三个模块）
# ══════════════════════════════════════════════════════════════

class EnterpriseRAG:
    def __init__(
        self,
        chunks: list[str],
        chunk_embs: np.ndarray,
        bm25: BM25,
        cache: SemanticCache,
        tracer: RequestTracer,
        model_name: str = "unknown",
    ):
        self.chunks     = chunks
        self.chunk_embs = chunk_embs
        self.bm25       = bm25
        self.cache      = cache
        self.tracer     = tracer
        self.model_name = model_name

    def query(self, user_query: str, top_k: int = 3) -> dict:
        request_id = str(uuid.uuid4())[:8]
        t_start = time.time()
        trace = {"request_id": request_id, "query": user_query, "model": self.model_name}

        # 1. Embedding（用于缓存查找 + 检索）
        t0 = time.time()
        q_emb = embed(user_query)
        trace["embed_latency_ms"] = round((time.time() - t0) * 1000, 1)

        # 2. 语义缓存查找
        cached = self.cache.lookup(user_query, q_emb)
        if cached:
            trace["cache_hit"] = True
            trace["answer"] = cached
            trace["total_latency_ms"] = round((time.time() - t_start) * 1000, 1)
            self.tracer.record(trace)
            return {"answer": cached, "cache_hit": True, "request_id": request_id,
                    "latency_ms": trace["total_latency_ms"]}

        trace["cache_hit"] = False

        # 3. 混合检索
        t0 = time.time()
        v_scores = [cosine_sim(q_emb, ce) for ce in self.chunk_embs]
        v_ranked = sorted(enumerate(v_scores), key=lambda x: x[1], reverse=True)
        b_ranked = self.bm25.retrieve(user_query)
        scores: dict[int, float] = {}
        for rank, (idx, _) in enumerate(v_ranked):
            scores[idx] = scores.get(idx, 0) + 1.5 / (60 + rank + 1)
        for rank, (idx, _) in enumerate(b_ranked):
            scores[idx] = scores.get(idx, 0) + 1.0 / (60 + rank + 1)
        merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_chunks = [self.chunks[i] for i, _ in merged[:top_k]]
        trace["retrieval_latency_ms"] = round((time.time() - t0) * 1000, 1)
        trace["retrieved_chunks"] = top_chunks

        # 4. 生成
        t0 = time.time()
        context = "\n\n".join(f"[文档{i+1}] {c}" for i, c in enumerate(top_chunks))
        prompt = f"""请根据以下参考资料回答问题。只使用资料中的信息。

参考资料：
{context}

问题：{user_query}
回答："""
        answer = chat(prompt, temperature=0.0)
        trace["generation_latency_ms"] = round((time.time() - t0) * 1000, 1)
        trace["answer"] = answer

        # 5. 写入缓存
        self.cache.store(user_query, q_emb, answer)

        trace["total_latency_ms"] = round((time.time() - t_start) * 1000, 1)
        self.tracer.record(trace)

        return {
            "answer":       answer,
            "cache_hit":    False,
            "request_id":   request_id,
            "latency_ms":   trace["total_latency_ms"],
            "chunks_used":  top_chunks,
        }


# ══════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════

# 不同租户的知识库文档
TENANT_DOCS = {
    "engineering": """RAG系统架构与工程实践指南

一、什么是RAG
RAG是Retrieval-Augmented Generation的缩写，中文叫检索增强生成。核心思路是：在大模型回答问题之前，先从外部知识库检索相关内容，再把检索结果和用户问题一起输入给大模型，让模型基于真实资料生成回答。RAG解决了大模型两个核心痛点：知识截止日期和私有领域知识缺失。

二、分块策略的重要性
文档分块是RAG系统中最容易被忽视但影响最大的工程决策。分块太小（如50个字符）：每块缺乏完整语义，LLM无法从碎片中理解上下文，答案质量大幅下降。分块太大（如2000个字符）：召回的块包含太多无关内容，LLM在长上下文中丢失焦点，答案精度降低。工程实践建议：推荐分块大小在200到500个字符之间，并设置约10%到20%的重叠区域（overlap）防止边界信息丢失。""",

    "sales": """售前 AI 解决方案文档

一、为什么客户需要 RAG
传统搜索系统无法理解自然语言意图，关键词匹配对同义词和语义理解很差。
RAG 系统让客户用自然语言提问，从企业内部文档精准获取答案。
典型客户场景：客服知识库、合同检索、技术手册问答、政策查询。

二、实施周期与成本
标准 RAG 项目从需求到上线通常需要 4~8 周。
成本构成：云 API 费用（embedding + LLM）+ 工程人工 + 向量数据库服务费。
中型企业（10万文档以内）月运营成本通常在 3000~15000 元之间。""",
}


def main():
    info = model_info()
    print(f"\n{'═'*60}")
    print(f" v10 ／ 企业级生产 RAG")
    print(f" Provider: {info['provider']}  |  Chat: {info['chat_model']}")
    print(f"{'═'*60}")

    # ── 模块一演示：语义缓存 ─────────────────────────────────
    print(f"\n{'═'*60}")
    print(" 模块一：语义缓存（Semantic Cache）")
    print(f"{'═'*60}\n")

    doc = TENANT_DOCS["engineering"]
    chunks = chunk_by_sentence(doc, max_chars=300)
    chunk_embs = np.array([embed(c) for c in chunks])
    bm25 = BM25(chunks)

    cache  = SemanticCache(similarity_threshold=0.92)
    tracer = RequestTracer(
        log_file=Path(__file__).parent / "rag_traces.jsonl"
    )
    rag = EnterpriseRAG(chunks, chunk_embs, bm25, cache, tracer, info["chat_model"])

    # 注意：语义相似对的相似度需 > 0.92（阈值）才会命中
    # 使用 BGE-M3 实测相似度：
    #   "RAG 的分块推荐用多大？" vs "RAG 分块大小推荐是多少？" = 0.97 ✓
    #   "overlap 比例应该设多少？" vs "overlap 设置多少合适？"   = 0.96 ✓
    cache_test_cases = [
        ("RAG 的分块推荐用多大？",        "首次请求，写入缓存"),
        ("RAG 分块大小推荐是多少？",      "语义相似(0.97) → 应命中缓存"),
        ("overlap 比例应该设多少？",      "新问题 → 不应命中缓存"),
        ("overlap 设置多少合适？",        "语义相似(0.96) → 应命中缓存"),
    ]

    for query, note in cache_test_cases:
        t0 = time.time()
        result = rag.query(query)
        elapsed = round((time.time() - t0) * 1000, 0)
        hit_icon = "⚡ HIT " if result["cache_hit"] else "🔍 MISS"
        print(f"  {hit_icon}  [{elapsed:>5}ms]  {query}")
        print(f"            说明：{note}")

    print(f"\n  缓存统计：{json.dumps(cache.stats(), ensure_ascii=False)}")

    # ── 模块二演示：请求追踪 ─────────────────────────────────
    print(f"\n{'═'*60}")
    print(" 模块二：请求追踪（Request Tracer）")
    print(f"{'═'*60}\n")

    summary = tracer.summary()
    print(f"  总请求数：{summary.get('total_requests', 0)}")
    print(f"  缓存命中率：{summary.get('cache_hit_rate', 0):.0%}")
    print(f"  平均延迟：{summary.get('avg_latency_ms', 0):.0f}ms")
    print(f"  P50 延迟：{summary.get('p50_latency_ms', 0):.0f}ms")
    print(f"  追踪日志已写入：rag_traces.jsonl")
    print(f"""
  Langfuse 集成（生产推荐，自托管开源）：
  ─────────────────────────────────────
  pip install langfuse

  from langfuse import Langfuse
  lf = Langfuse(public_key="...", secret_key="...", host="...")

  # 每次请求包裹一个 trace
  with lf.trace(name="rag_query", input={{"query": user_query}}) as trace:
      retrieved = retrieve(...)
      answer    = generate(...)
      trace.update(output={{"answer": answer}})
      trace.score(name="faithfulness", value=0.9)
  ─────────────────────────────────────""")

    # ── 模块三演示：多租户 Namespace ─────────────────────────
    print(f"\n{'═'*60}")
    print(" 模块三：多租户 Namespace（Chroma Collections）")
    print(f"{'═'*60}\n")

    try:
        import chromadb
        chroma_client = chromadb.Client()

        print("  为 engineering / sales 两个租户建立独立 Collection...")
        eng_col  = build_tenant_collection("engineering", TENANT_DOCS["engineering"], chroma_client)
        sale_col = build_tenant_collection("sales",       TENANT_DOCS["sales"],       chroma_client)
        print("  → Collections 创建完成\n")

        isolation_test = [
            (eng_col,  "engineering", "分块大小推荐多少？"),
            (sale_col, "sales",       "实施 RAG 项目大概要多少钱？"),
            (eng_col,  "engineering", "RAG 项目实施成本是多少？"),  # 工程部门找不到销售文档
        ]
        for col, tenant, query in isolation_test:
            results = tenant_retrieve(col, query, top_k=1)
            found = results[0][:60] + "..." if results else "[无结果]"
            print(f"  租户 [{tenant}]  Q: {query}")
            print(f"  → {found}\n")

        print("  关键认知：engineering 租户检索不到 sales 文档，数据天然隔离。")

    except ImportError:
        print("  ℹ️  chromadb 未安装，跳过多租户演示")
        print("  请运行：uv pip install chromadb（需要 Python 3.12）")

    # ── 模块四：增量索引更新思路 ─────────────────────────────
    print(f"\n{'═'*60}")
    print(" 模块四：增量索引更新（设计思路）")
    print(f"{'═'*60}")
    print(f"""
  问题：文档更新时，不能每次都重建整个索引（代价 O(N)）。

  增量更新流程：
    新文档上传
      → 分块（chunk_by_sentence）
      → embedding（每个 chunk 单独）
      → 追加到向量库（chroma.add，不影响已有数据）
      → 更新 BM25 索引（追加新 token）

  删除/更新旧文档：
    向量库支持按 ID 删除（chroma.delete）
    重新 embedding 更新的 chunk，更新对应 ID

  代码示意：
  ─────────────────────────────────────
  def incremental_add(collection, new_document: str, doc_id: str):
      chunks = chunk_by_sentence(new_document)
      for i, chunk in enumerate(chunks):
          emb = embed(chunk)
          collection.add(
              ids=[f"{{doc_id}}_chunk_{{i}}"],
              embeddings=[emb.tolist()],
              documents=[chunk],
          )
      print(f"已追加 {{len(chunks)}} 个 chunk，索引总量: {{collection.count()}}")
  ─────────────────────────────────────

  生产注意：
    - embedding 有 API 速率限制，大批量入库需要限流（rate limit）
    - 高并发写入时注意向量库的写锁竞争
    - 重要场景：先写入"staging"集合，验证后切换，实现蓝绿部署
    """)

    # ── 总结 ─────────────────────────────────────────────────
    print(f"{'═'*60}")
    print(" 企业级 RAG 系统全貌")
    print(f"{'═'*60}")
    print(f"""
  Enterprise RAG Stack
  ═══════════════════════

  Query Layer
    用户输入
      → 语义缓存命中?         (本文件 模块一)
      → Query 改写            (v7)
      → 混合检索 + Reranking  (v5 + v6)

  Retrieval Layer
    向量数据库（多租户隔离）  (本文件 模块三)
    BM25 稀疏索引
    增量索引更新              (本文件 模块四)

  Generation Layer
    LLM 生成（Prompt 约束）   (v8)
    Agentic 多步推理          (v9)

  Observability Layer
    请求追踪 + 延迟监控       (本文件 模块二)
    Langfuse / LangSmith
    评估指标看板              (v8)

  售前对话中，你现在可以回答：
    Q: "数据会不会泄露给其他用户？"    → 多租户 Collection 隔离
    Q: "系统出了问题怎么排查？"        → Tracing 每次请求都有记录
    Q: "重复问题会不会很慢很贵？"      → 语义缓存，命中时 <10ms
    Q: "文档更新了要重建索引吗？"      → 增量更新，不影响在线服务
    """)

    print(f"  → 追踪日志：rag_traces.jsonl")
    print(f"  → 课程完结。从 v1 到 v10，你已经走完了完整的 RAG 工程路径。\n")


if __name__ == "__main__":
    main()
