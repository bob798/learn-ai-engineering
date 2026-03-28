#!/usr/bin/env python3
"""
06_v6_reranking.py — Reranking：两阶段检索（粗排 + 精排）
=====================================
插入位置：v5（混合检索）之后、v7（Query 变换）之前

核心认知：
  v5 的混合检索已经把"召回质量"做得很好。
  但 Top-K 里还存在一个问题：排名不一定是最优的。
  第 1 名不一定比第 2 名更相关——Embedding 是向量距离，不是精准的相关性判断。

  Reranking 是"精排"：先用便宜的方法（向量检索）快速召回 Top-N，
  再用昂贵但精准的方法（Cross-encoder）对 Top-N 重新排序，取最终 Top-K。

两种 Encoder 架构对比：
  Bi-encoder（向量检索）：
    query → Encoder → 向量 q
    doc   → Encoder → 向量 d
    相似度 = cosine(q, d)   ← 分开编码，可以预计算 doc 向量，检索快
    缺点：query 和 doc 之间没有"交叉注意力"，理解有限

  Cross-encoder（Reranker）：
    [query, doc] → Encoder → 相关性分数（0-1）  ← 一起输入
    优点：query 和 doc 充分交互，理解更深，排名更准
    缺点：每对 (query, doc) 都要单独推理，不能预计算，不适合大规模召回

  工程结论：
    大规模召回用 Bi-encoder（快），精排用 Cross-encoder（准）——这是信息检索的经典两阶段架构。

工具选型：
  BAAI/bge-reranker-base   ← 本文件使用，中英文，~270MB，开源本地运行
  BAAI/bge-reranker-v2-m3  ← 更强，多语言，~560MB
  Cohere Rerank API         ← 托管服务，无需本地部署，按次计费
  cross-encoder/ms-marco    ← 英文最强，不适合中文

依赖：pip install sentence-transformers openai numpy python-dotenv
      ⚠️ 首次运行会下载模型（约 270MB），需要网络。
运行：python 06_v6_reranking.py
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║  📺 讲师注释                                                  ║
# ║  对应集数：01_理解RAG.html · Ep3 延伸（Reranking）           ║
# ║  核心代码：第 134~141 行（加权 rrf_merge，w_bm25=1.5 修复    ║
# ║            平局），第 183 行起（load_reranker + rerank）      ║
# ║  可跳过：第 104~131 行（BM25，v5 已讲）                      ║
# ║  本集关键数字：等权 RRF MRR=0.57 → 加权 RRF=0.92            ║
# ║               → Reranker=1.00                                ║
# ╚══════════════════════════════════════════════════════════════╝

import json
import math
import re
import time
import numpy as np
from collections import Counter
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

# ── 加载 Provider 接口 ───────────────────────────────────────
_provider_path = Path(__file__).with_name("00_配置提供商_先改这个.py")
_provider_spec = spec_from_file_location("rag_provider", _provider_path)
if _provider_spec is None or _provider_spec.loader is None:
    raise ImportError(f"无法加载提供商配置文件: {_provider_path}")
_provider_module = module_from_spec(_provider_spec)
_provider_spec.loader.exec_module(_provider_module)

embed      = _provider_module.embed
cosine_sim = _provider_module.cosine_sim
model_info = _provider_module.model_info


# ══════════════════════════════════════════════════════════════
# 知识库文档（同 v3.5 ~ v5）
# ══════════════════════════════════════════════════════════════

DOCUMENT = """RAG系统架构与工程实践指南

一、什么是RAG
RAG是Retrieval-Augmented Generation的缩写，中文叫检索增强生成。核心思路是：在大模型回答问题之前，先从外部知识库检索相关内容，再把检索结果和用户问题一起输入给大模型，让模型基于真实资料生成回答。RAG解决了大模型两个核心痛点：知识截止日期和私有领域知识缺失。

二、分块策略的重要性
文档分块是RAG系统中最容易被忽视但影响最大的工程决策。分块太小（如50个字符）：每块缺乏完整语义，LLM无法从碎片中理解上下文，答案质量大幅下降。分块太大（如2000个字符）：召回的块包含太多无关内容，LLM在长上下文中丢失焦点，答案精度降低。工程实践建议：推荐分块大小在200到500个字符之间，并设置约10%到20%的重叠区域（overlap）防止边界信息丢失。不同文档类型有不同的最优分块策略：技术文档适合句子级分块，代码文档按函数边界分块，PDF报告按段落分块。

三、向量检索原理
每个文本块通过Embedding模型转换为高维向量，通常为1536维或3072维。向量间的语义相似度通过余弦相似度衡量：两向量夹角越小，语义越接近，余弦值越接近1。检索时将用户问题也转为向量，在向量库中找最近邻的文档向量，返回Top-K个结果。常用向量数据库包括Chroma（本地开发）、Qdrant（云端生产）、Pinecone（托管服务）。

四、RAG评估指标
评估RAG系统需要同时关注检索和生成两个维度。检索维度：召回率（Recall）衡量相关文档有没有被找到，精确率（Precision）衡量找到的文档中有多少真正相关。生成维度：忠实度（Faithfulness）衡量答案是否忠实于检索资料而没有幻觉，相关性（Answer Relevancy）衡量答案是否切中用户问题。构建Golden Dataset（黄金测试集）是评估的前提，通常需要人工标注20到100条高质量的问答对。
"""

QUERIES = [
    {"id": "Q001", "query": "RAG 的分块推荐用多大？overlap 比例是多少？",
     "relevant_text": "推荐分块大小在200到500个字符之间，并设置约10%到20%的重叠区域（overlap）防止边界信息丢失", "difficulty": "medium"},
    {"id": "Q002", "query": "RAG 解决了大模型哪些核心问题？",
     "relevant_text": "RAG解决了大模型两个核心痛点：知识截止日期和私有领域知识缺失", "difficulty": "easy"},
    {"id": "Q003", "query": "不同类型的文档应该用什么分块策略？",
     "relevant_text": "不同文档类型有不同的最优分块策略：技术文档适合句子级分块，代码文档按函数边界分块，PDF报告按段落分块", "difficulty": "medium"},
    {"id": "Q004", "query": "Chroma 和 Qdrant 各适合什么场景？",
     "relevant_text": "常用向量数据库包括Chroma（本地开发）、Qdrant（云端生产）、Pinecone（托管服务）", "difficulty": "easy"},
    {"id": "Q005", "query": "为什么分块太大会影响 RAG 质量，具体原因是什么？",
     "relevant_text": "分块太大（如2000个字符）：召回的块包含太多无关内容，LLM在长上下文中丢失焦点，答案精度降低", "difficulty": "hard"},
    {"id": "Q006", "query": "Faithfulness 指标衡量的是什么？",
     "relevant_text": "忠实度（Faithfulness）衡量答案是否忠实于检索资料而没有幻觉", "difficulty": "medium"},
    {"id": "Q007", "query": "向量维度是多少？用什么衡量两个向量的相似度？",
     "relevant_text": "每个文本块通过Embedding模型转换为高维向量，通常为1536维或3072维。向量间的语义相似度通过余弦相似度衡量", "difficulty": "easy"},
]


# ══════════════════════════════════════════════════════════════
# BM25 + RRF（复用 v5）
# ══════════════════════════════════════════════════════════════

class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.k1 = k1; self.b = b; self.N = len(corpus)
        self.tokenized = [self._tok(d) for d in corpus]
        self.avgdl = sum(len(d) for d in self.tokenized) / self.N
        self.df: dict[str, int] = {}
        for doc in self.tokenized:
            for t in set(doc):
                self.df[t] = self.df.get(t, 0) + 1

    def _tok(self, text):
        return re.findall(r'[\u4e00-\u9fff]|[a-zA-Z0-9]+', text)

    def score(self, query, doc_idx):
        doc = self.tokenized[doc_idx]; dl = len(doc); tf_map = Counter(doc)
        total = 0.0
        for t in self._tok(query):
            if t not in self.df: continue
            idf = math.log((self.N - self.df[t] + 0.5) / (self.df[t] + 0.5) + 1)
            tf = tf_map.get(t, 0)
            tf_norm = tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            total += idf * tf_norm
        return total

    def retrieve(self, query, top_k=None):
        n = top_k or self.N
        return sorted([(i, self.score(query, i)) for i in range(self.N)],
                      key=lambda x: x[1], reverse=True)[:n]


def rrf_merge(vector_ranked, bm25_ranked, k=60, w_vector=1.0, w_bm25=1.5):
    scores: dict[int, float] = {}
    for rank, (idx, _) in enumerate(vector_ranked):
        scores[idx] = scores.get(idx, 0) + w_vector / (k + rank + 1)
    for rank, (idx, _) in enumerate(bm25_ranked):
        scores[idx] = scores.get(idx, 0) + w_bm25 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


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


def recall_at_k(retrieved, relevant_text, k):
    key = relevant_text[:15].strip()
    return 1.0 if any(key in c for c in retrieved[:k]) else 0.0

def mrr_score(retrieved, relevant_text):
    key = relevant_text[:15].strip()
    for rank, chunk in enumerate(retrieved, start=1):
        if key in chunk: return 1.0 / rank
    return 0.0


# ══════════════════════════════════════════════════════════════
# Cross-encoder Reranker
#
# 架构说明：
#   Bi-encoder：query 和 doc 分开编码 → 向量 → 余弦相似度
#     优势：doc 向量可预计算，检索 O(1)（ANN 查找）
#     局限：query 和 doc 之间无直接交互，准确度有上限
#
#   Cross-encoder：query + doc 拼接后一起输入 Transformer
#     → 直接输出相关性分数（0-1）
#     优势：充分建模 query-doc 交互，准确度高
#     局限：每对 (query, doc) 都需完整推理，不能预计算
#
#   结论：两阶段架构——Bi-encoder 召回 Top-N（快），Cross-encoder 精排（准）
# ══════════════════════════════════════════════════════════════

def load_reranker():
    """
    加载 Cross-encoder 模型。
    首次运行会从 HuggingFace 下载（约 270MB）。
    国内用户如下载慢，可设置镜像：
      export HF_ENDPOINT=https://hf-mirror.com
    """
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        raise ImportError(
            "\n[错误] 未安装 sentence-transformers\n"
            "请运行：pip install sentence-transformers\n"
            "（或 uv pip install sentence-transformers）"
        )

    print("  加载 Cross-encoder 模型（首次运行需下载 ~270MB）...")
    t0 = time.time()
    model = CrossEncoder("BAAI/bge-reranker-base")
    print(f"  → 模型加载完成，耗时 {time.time() - t0:.1f}s")
    return model


def rerank(model, query: str, candidates: list[str], top_k: int) -> list[tuple[str, float]]:
    """
    用 Cross-encoder 对候选 chunks 重新打分并排序。
    返回 [(chunk, score), ...] 降序。
    """
    pairs = [[query, chunk] for chunk in candidates]
    t0 = time.time()
    scores = model.predict(pairs)
    latency_ms = (time.time() - t0) * 1000
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k], latency_ms


# ══════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════

def main():
    info = model_info()
    print(f"\n{'═'*60}")
    print(f" v6 ／ Reranking（Bi-encoder 召回 + Cross-encoder 精排）")
    print(f" Provider: {info['provider']}  |  Embed: {info['embed_model']}")
    print(f"{'═'*60}")

    # ── STEP 1：加载基线 ─────────────────────────────────────
    baseline_path = Path(__file__).parent / "baseline.json"
    if not baseline_path.exists():
        print("  ⚠️  未找到 baseline.json，请先运行 03_v3.5_检索基线.py")
        return
    with open(baseline_path, encoding="utf-8") as f:
        baseline = json.load(f)
    print(f"\n  基线 Recall@3 : {baseline['recall_at_3']}")
    print(f"  基线 MRR      : {baseline['mrr']}")

    # ── STEP 2：建立双路索引 ─────────────────────────────────
    print(f"\n{'═'*60}")
    print(" STEP 2 ／ 建立双路索引（Bi-encoder 向量 + BM25）")
    print(f"{'═'*60}\n")
    chunks = chunk_by_sentence(DOCUMENT, max_chars=300)
    print(f"  分块数: {len(chunks)}")
    chunk_embs = np.array([embed(c) for c in chunks])
    bm25 = BM25(chunks)
    print(f"  → Bi-encoder 索引就绪（{chunk_embs.shape}）")
    print(f"  → BM25 词典大小: {len(bm25.df)} tokens")

    # ── STEP 3：加载 Cross-encoder ──────────────────────────
    print(f"\n{'═'*60}")
    print(" STEP 3 ／ 加载 Cross-encoder（bge-reranker-base）")
    print(f"{'═'*60}\n")
    reranker = load_reranker()

    # ── STEP 4：架构演示（一条 Query 的完整两阶段流程）─────
    print(f"\n{'═'*60}")
    print(" STEP 4 ／ 两阶段架构演示（Q001）")
    print(f"{'═'*60}")

    demo = QUERIES[0]  # Q001 是典型的难例
    print(f"\n  Query: {demo['query']}")

    # 第一阶段：Bi-encoder 召回 Top-10
    q_emb = embed(demo["query"])
    v_scores = [cosine_sim(q_emb, ce) for ce in chunk_embs]
    v_ranked = sorted(enumerate(v_scores), key=lambda x: x[1], reverse=True)
    b_ranked = bm25.retrieve(demo["query"])
    top10_items = rrf_merge(v_ranked, b_ranked)[:10]
    top10_chunks = [chunks[i] for i, _ in top10_items]

    print(f"\n  阶段一：Bi-encoder（混合 RRF）召回 Top-10")
    for i, chunk in enumerate(top10_chunks[:5]):
        marker = " ← 正确答案" if demo["relevant_text"][:15] in chunk else ""
        print(f"    [{i+1}] {chunk[:60]}...{marker}")
    if len(top10_chunks) > 5:
        print(f"    ... （共 10 个候选）")

    # 第二阶段：Cross-encoder 精排取 Top-3
    top3_reranked, latency_ms = rerank(reranker, demo["query"], top10_chunks, top_k=3)
    print(f"\n  阶段二：Cross-encoder 精排 → Top-3  （打分耗时: {latency_ms:.0f}ms）")
    for i, (chunk, score) in enumerate(top3_reranked):
        marker = " ← 正确答案" if demo["relevant_text"][:15] in chunk else ""
        print(f"    [{i+1}] score={score:.4f}  {chunk[:60]}...{marker}")

    # ── STEP 5：全量对比评估 ─────────────────────────────────
    print(f"\n{'═'*60}")
    print(" STEP 5 ／ 三路对比：等权RRF vs 加权RRF vs Cross-encoder Reranker")
    print(f"{'═'*60}")
    print("""
  为什么要三路对比？
    等权 RRF（1:1）：v5 发现的平局问题在这里复现，MRR<1
    加权 RRF（1:1.5）：v5 的修复方案，靠调权重绕过平局
    Reranker：完全绕开融合层，直接判断 query-文档相关性
  这样才能清楚看出 Reranker 解决的是哪个层次的问题。
    """)

    RECALL_N = 10   # 召回候选数
    TOP_K    = 3    # 最终取 Top-K

    results_eq     = []   # 等权 RRF Top-3（有平局问题）
    results_weighted = [] # 加权 RRF Top-3（v5 修复版）
    results_rerank = []   # Cross-encoder 精排
    total_rerank_ms = 0.0

    for item in QUERIES:
        diff_icon = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}.get(item["difficulty"], "⚪")

        q_emb = embed(item["query"])
        v_scores = [cosine_sim(q_emb, ce) for ce in chunk_embs]
        v_ranked = sorted(enumerate(v_scores), key=lambda x: x[1], reverse=True)
        b_ranked = bm25.retrieve(item["query"])

        # 等权 RRF（复现平局问题）
        eq_merged   = rrf_merge(v_ranked, b_ranked, w_vector=1.0, w_bm25=1.0)
        eq_chunks   = [chunks[i] for i, _ in eq_merged[:TOP_K]]
        r_eq = recall_at_k(eq_chunks, item["relevant_text"], TOP_K)
        m_eq = mrr_score(eq_chunks,   item["relevant_text"])

        # 加权 RRF（v5 修复版，作为 Reranker 的召回基础）
        w_merged    = rrf_merge(v_ranked, b_ranked, w_vector=1.0, w_bm25=1.5)
        topN_chunks = [chunks[i] for i, _ in w_merged[:RECALL_N]]
        w_chunks    = topN_chunks[:TOP_K]
        r_w  = recall_at_k(w_chunks, item["relevant_text"], TOP_K)
        m_w  = mrr_score(w_chunks,   item["relevant_text"])

        # Cross-encoder Reranking（基于加权 RRF 的 Top-10 候选）
        rr_result, ms = rerank(reranker, item["query"], topN_chunks, top_k=TOP_K)
        total_rerank_ms += ms
        rr_chunks = [c for c, _ in rr_result]
        r_rr = recall_at_k(rr_chunks, item["relevant_text"], TOP_K)
        m_rr = mrr_score(rr_chunks,   item["relevant_text"])

        print(f"  {diff_icon} [{item['id']}]  "
              f"等权={m_eq:.2f}  加权={m_w:.2f}  Reranker={m_rr:.2f}")
        print(f"       {item['query']}")

        results_eq.append(      {"recall": r_eq, "mrr": m_eq})
        results_weighted.append({"recall": r_w,  "mrr": m_w})
        results_rerank.append(  {"recall": r_rr, "mrr": m_rr})

    # ── STEP 6：汇总 ────────────────────────────────────────
    def avg(lst, key):
        return round(sum(x[key] for x in lst) / len(lst), 3)

    eq_recall  = avg(results_eq,       "recall")
    w_recall   = avg(results_weighted, "recall")
    rr_recall  = avg(results_rerank,   "recall")
    eq_mrr     = avg(results_eq,       "mrr")
    w_mrr      = avg(results_weighted, "mrr")
    rr_mrr     = avg(results_rerank,   "mrr")
    avg_rerank_ms = total_rerank_ms / len(QUERIES)

    print(f"\n{'═'*60}")
    print(" STEP 6 ／ 汇总")
    print(f"{'═'*60}\n")

    rows = [
        ("[基线] 纯向量 v3.5",             baseline["recall_at_3"], baseline["mrr"]),
        ("混合 等权RRF 1:1（平局问题）",    eq_recall,               eq_mrr),
        ("混合 加权RRF 1:1.5（v5修复）",   w_recall,                w_mrr),
        (f"加权RRF Top-{RECALL_N} + 精排", rr_recall,               rr_mrr),
    ]
    print(f"  {'方案':<32}  {'Recall@3':>9}  {'MRR':>7}  {'vs 基线':>8}")
    print(f"  {'─'*60}")
    best_mrr = max(m for _, _, m in rows)
    for name, recall, mrr_v in rows:
        delta = mrr_v - baseline["mrr"]
        ds = f"+{delta:.3f}" if delta > 0 else (f"{delta:.3f}" if delta < 0 else "  ─   ")
        mark = " ← 最佳" if mrr_v == best_mrr and mrr_v > baseline["mrr"] else ""
        print(f"  {name:<32}  {recall:>9.3f}  {mrr_v:>7.3f}  {ds:>8}{mark}")

    print(f"\n  Cross-encoder 精排延迟：平均 {avg_rerank_ms:.0f}ms / query（{RECALL_N} 个候选）")
    print(f"""
  关键认知：
    ✓ Reranker 解决的是"召回对了但排序错了"的问题
    ✓ Recall 不变（召回阶段决定）→ MRR 变化（排序阶段决定）
    ✓ 两阶段架构是信息检索的经典范式（不是 RAG 专利）
    ✓ 延迟权衡：精排增加 {avg_rerank_ms:.0f}ms，对实时场景需评估是否可接受
    ✓ bge-reranker-v2-m3（~560MB）比 base 更强，可按需升级

    下一步（v7）：
      精排解决了"排序"问题，但如果用户的 query 本身就写得不好怎么办？
      Query 变换（改写/扩展）从 query 侧提升检索质量。
    """)

    # ── 保存结果 ─────────────────────────────────────────────
    result = {
        "strategy":     f"hybrid_rrf_top{RECALL_N}_then_bge_reranker_top{TOP_K}",
        "equal_rrf":    {"recall_at_3": eq_recall, "mrr": eq_mrr},
        "weighted_rrf": {"recall_at_3": w_recall,  "mrr": w_mrr},
        "reranker":     {"recall_at_3": rr_recall, "mrr": rr_mrr},
        "baseline":     {"recall_at_3": baseline["recall_at_3"], "mrr": baseline["mrr"]},
        "avg_rerank_latency_ms": round(avg_rerank_ms, 1),
    }
    out = Path(__file__).parent / "v6_reranking_result.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  → 已保存到 v6_reranking_result.json")
    print(f"  → 下一步：python 07_v7_query变换.py\n")


if __name__ == "__main__":
    main()
