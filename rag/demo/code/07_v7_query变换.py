#!/usr/bin/env python3
"""
07_v7_query变换.py — Query 变换与扩展：从 query 侧提升检索质量
=====================================
插入位置：v6（Reranking）之后、v8（评估框架）之前

核心认知：
  v6 解决了"召回到了但排序错了"。
  还有另一个问题：用户的 query 本身就写得不好。
  - 太短：一句话问了多件事（"分块多大？overlap 呢？"）
  - 带噪音：主题词"RAG"让向量偏向文档介绍而非答案
  - 措辞偏：用户说"编造"，文档写"幻觉"——同义词但词汇不重叠

  Query 变换：在把 query 送给检索引擎之前，先用 LLM 对它进行处理。
  不改变用户意图，改变查询的表达形式。

本文件实现三种技术：

  1. Multi-Query（多路查询）
     LLM 把一个 query 改写为 3 个不同角度的 query，
     分别检索后对结果集取并集（RRF 融合去重）。
     好处：一次召回的覆盖面更广，减少遗漏。

  2. HyDE（Hypothetical Document Embedding，假设文档嵌入）
     LLM 先"假设"一段回答文档，用这段文档的 embedding 做检索。
     原理：答案域的向量比问题域的向量更接近文档域。
     好处：绕开"问题-文档"语义鸿沟，尤其对短 query 有效。

  3. Step-back Prompting（后退一步提问）
     LLM 把具体问题抽象为更通用的问题（"后退一步"）。
     例："RAG 的 overlap 推荐多少" → "文本分块的工程实践原则是什么"
     好处：召回更广泛的背景知识，适合需要上下文的推理问题。

依赖：pip install openai numpy python-dotenv（无额外依赖）
运行：python 07_v7_query变换.py
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║  📺 讲师注释                                                  ║
# ║  对应集数：独立专题（Query 变换三种方式）                    ║
# ║  核心代码：第 131~150 行（rrf_merge_multi，N路融合），        ║
# ║            multi_query_expand / hyde_embed / step_back 三个  ║
# ║            函数                                              ║
# ║  可跳过：第 101~129 行（BM25，已讲）                         ║
# ║  本集关键数字：4路对比——原始 Query vs Multi-Query vs         ║
# ║               HyDE vs Step-back                              ║
# ╚══════════════════════════════════════════════════════════════╝

import json
import math
import re
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
chat       = _provider_module.chat
cosine_sim = _provider_module.cosine_sim
model_info = _provider_module.model_info


# ══════════════════════════════════════════════════════════════
# 知识库文档（同 v3.5 ~ v6）
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
# BM25 + RRF（复用 v5/v6）
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


def rrf_merge_multi(ranked_lists: list[list[tuple[int, float]]], k=60) -> list[tuple[int, float]]:
    """
    支持 N 路结果的 RRF 融合（泛化版，Multi-Query 使用）。
    每路权重相等，适合多个改写 query 的结果合并。
    """
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, (idx, _) in enumerate(ranked):
            scores[idx] = scores.get(idx, 0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


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

def hybrid_retrieve(query, chunk_embs, chunks, bm25, top_k):
    """标准混合检索（复用，v6 同款）"""
    q_emb = embed(query)
    v_scores = [cosine_sim(q_emb, ce) for ce in chunk_embs]
    v_ranked = sorted(enumerate(v_scores), key=lambda x: x[1], reverse=True)
    b_ranked = bm25.retrieve(query)
    merged = rrf_merge(v_ranked, b_ranked)
    return [chunks[i] for i, _ in merged[:top_k]], [i for i, _ in merged[:top_k]]


# ══════════════════════════════════════════════════════════════
# Query 变换策略
# ══════════════════════════════════════════════════════════════

def multi_query_expand(query: str, n: int = 3) -> list[str]:
    """
    Multi-Query：LLM 生成 n 个不同角度的查询语句。
    策略：从不同维度重述同一问题，提高覆盖面。
    """
    prompt = f"""请将下面的问题改写为 {n} 个不同角度的搜索查询，每个查询从不同角度探索同一个问题。
要求：
- 每行输出一个查询
- 不要编号，不要解释
- 保持与原问题相同的意图，但用不同措辞

原始问题：{query}

改写后的查询（{n} 个）："""
    result = chat(prompt, temperature=0.3)
    queries = [line.strip() for line in result.strip().split("\n") if line.strip()]
    return queries[:n]


def hyde_embed(query: str) -> np.ndarray:
    """
    HyDE：LLM 生成假设答案文档，用它的 embedding 做检索。

    为什么这样做有效？
      向量空间中的三角关系：
        问题 → 答案 → 相关文档
      答案文本与文档文本在语义空间中更接近（都在"答案"域）。
      用假设答案的向量代替问题向量，等于"换了个起点"，
      更容易找到真实答案文档。
    """
    prompt = f"""请用两三句话直接回答下面的问题，像在写技术文档，给出具体数字和结论：

问题：{query}

直接回答："""
    hypothetical = chat(prompt, temperature=0.0)
    return embed(hypothetical), hypothetical


def step_back(query: str) -> str:
    """
    Step-back Prompting：把具体问题抽象为更通用的原则性问题。
    用于：需要背景知识才能回答的问题，或问题太具体导致召回面太窄。

    例：
      具体："RAG 系统里 overlap 推荐多少比例"
      后退："文本分块的工程设计原则是什么"
    """
    prompt = f"""请将下面的具体问题抽象成一个更通用、原则性的问题（"后退一步"）。
不要改变意图，但要更抽象、更宽泛，便于检索更多相关背景信息。
只输出改写后的问题，不要解释。

具体问题：{query}

更通用的问题："""
    return chat(prompt, temperature=0.0).strip()


# ══════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════

def main():
    info = model_info()
    print(f"\n{'═'*60}")
    print(f" v7 ／ Query 变换（Multi-Query + HyDE + Step-back）")
    print(f" Provider: {info['provider']}  |  Chat: {info['chat_model']}")
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

    # ── STEP 2：建立索引 ─────────────────────────────────────
    chunks = chunk_by_sentence(DOCUMENT, max_chars=300)
    chunk_embs = np.array([embed(c) for c in chunks])
    bm25 = BM25(chunks)

    # ── STEP 3：技术演示（Q001 的三种变换）──────────────────
    print(f"\n{'═'*60}")
    print(" STEP 3 ／ 技术演示（以 Q001 为例）")
    print(f"{'═'*60}")

    demo = QUERIES[0]
    print(f"\n  原始 Query：{demo['query']}\n")

    # Multi-Query
    print("  【Multi-Query】")
    sub_queries = multi_query_expand(demo["query"], n=3)
    for i, q in enumerate(sub_queries, 1):
        print(f"    改写 {i}：{q}")

    # 对 3 个 sub-query 分别检索，RRF 融合
    all_ranked: list[list[tuple[int, float]]] = []
    for sq in sub_queries:
        _, ranked_idx = hybrid_retrieve(sq, chunk_embs, chunks, bm25, top_k=len(chunks))
        all_ranked.append([(i, 0.0) for i in ranked_idx])
    mq_merged = rrf_merge_multi(all_ranked)
    mq_chunks = [chunks[i] for i, _ in mq_merged[:3]]
    mq_mrr = mrr_score(mq_chunks, demo["relevant_text"])
    print(f"    → Multi-Query Top-3 MRR: {mq_mrr:.3f}")

    # HyDE
    print("\n  【HyDE】")
    hyde_emb, hypothetical = hyde_embed(demo["query"])
    print(f"    假设文档：{hypothetical[:80]}...")
    hyde_scores = [cosine_sim(hyde_emb, ce) for ce in chunk_embs]
    hyde_ranked = sorted(range(len(hyde_scores)), key=lambda i: hyde_scores[i], reverse=True)
    hyde_chunks = [chunks[i] for i in hyde_ranked[:3]]
    hyde_mrr = mrr_score(hyde_chunks, demo["relevant_text"])
    print(f"    → HyDE Top-3 MRR: {hyde_mrr:.3f}")

    # Step-back
    print("\n  【Step-back】")
    sb_query = step_back(demo["query"])
    print(f"    抽象后：{sb_query}")
    sb_chunks, _ = hybrid_retrieve(sb_query, chunk_embs, chunks, bm25, top_k=3)
    sb_mrr = mrr_score(sb_chunks, demo["relevant_text"])
    print(f"    → Step-back Top-3 MRR: {sb_mrr:.3f}")

    # ── STEP 4：全量四路对比 ─────────────────────────────────
    print(f"\n{'═'*60}")
    print(" STEP 4 ／ 全量对比（7 条 Query）")
    print(f"{'═'*60}\n")

    TOP_K = 3
    results = {"original": [], "multi_query": [], "hyde": [], "step_back": []}

    for item in QUERIES:
        diff_icon = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}.get(item["difficulty"], "⚪")

        # 原始混合检索
        orig_chunks, orig_ranked = hybrid_retrieve(item["query"], chunk_embs, chunks, bm25, TOP_K)
        r_o = recall_at_k(orig_chunks, item["relevant_text"], TOP_K)
        m_o = mrr_score(orig_chunks,   item["relevant_text"])

        # Multi-Query
        subs = multi_query_expand(item["query"], n=3)
        all_r = []
        for sq in subs:
            _, ridx = hybrid_retrieve(sq, chunk_embs, chunks, bm25, len(chunks))
            all_r.append([(i, 0.0) for i in ridx])
        mq_res = rrf_merge_multi(all_r)
        mq_c = [chunks[i] for i, _ in mq_res[:TOP_K]]
        r_mq = recall_at_k(mq_c, item["relevant_text"], TOP_K)
        m_mq = mrr_score(mq_c,   item["relevant_text"])

        # HyDE
        h_emb, _ = hyde_embed(item["query"])
        h_scores = [cosine_sim(h_emb, ce) for ce in chunk_embs]
        h_ranked = sorted(range(len(h_scores)), key=lambda i: h_scores[i], reverse=True)
        h_chunks = [chunks[i] for i in h_ranked[:TOP_K]]
        r_h = recall_at_k(h_chunks, item["relevant_text"], TOP_K)
        m_h = mrr_score(h_chunks,   item["relevant_text"])

        # Step-back
        sb_q = step_back(item["query"])
        sb_c, _ = hybrid_retrieve(sb_q, chunk_embs, chunks, bm25, TOP_K)
        r_sb = recall_at_k(sb_c, item["relevant_text"], TOP_K)
        m_sb = mrr_score(sb_c,   item["relevant_text"])

        # 找出本 query 最佳
        best = max(m_o, m_mq, m_h, m_sb)
        def mark(v): return " ★" if v == best and v > m_o else ""

        print(f"  {diff_icon} [{item['id']}]  原始={m_o:.2f}  MQ={m_mq:.2f}{mark(m_mq)}  "
              f"HyDE={m_h:.2f}{mark(m_h)}  SB={m_sb:.2f}{mark(m_sb)}")
        print(f"       {item['query']}")

        results["original"].append(  {"recall": r_o,  "mrr": m_o})
        results["multi_query"].append({"recall": r_mq, "mrr": m_mq})
        results["hyde"].append(       {"recall": r_h,  "mrr": m_h})
        results["step_back"].append(  {"recall": r_sb, "mrr": m_sb})

    # ── STEP 5：汇总 ─────────────────────────────────────────
    def avg(lst, key):
        return round(sum(x[key] for x in lst) / len(lst), 3)

    print(f"\n{'═'*60}")
    print(" STEP 5 ／ 汇总")
    print(f"{'═'*60}\n")

    strategy_rows = [
        ("[基线] 纯向量 v3.5",  baseline["recall_at_3"],           baseline["mrr"]),
        ("原始混合检索（v5）",  avg(results["original"],  "recall"), avg(results["original"],  "mrr")),
        ("Multi-Query",         avg(results["multi_query"],"recall"), avg(results["multi_query"],"mrr")),
        ("HyDE",                avg(results["hyde"],       "recall"), avg(results["hyde"],       "mrr")),
        ("Step-back",           avg(results["step_back"],  "recall"), avg(results["step_back"],  "mrr")),
    ]
    best_mrr = max(m for _, _, m in strategy_rows)
    print(f"  {'策略':<24}  {'Recall@3':>9}  {'MRR':>7}  {'vs 基线':>8}")
    print(f"  {'─'*54}")
    for name, recall, mrr_v in strategy_rows:
        delta = mrr_v - baseline["mrr"]
        ds = f"+{delta:.3f}" if delta > 0 else (f"{delta:.3f}" if delta < 0 else "  ─   ")
        mark = " ← 最佳" if mrr_v == best_mrr and mrr_v > baseline["mrr"] else ""
        print(f"  {name:<24}  {recall:>9.3f}  {mrr_v:>7.3f}  {ds:>8}{mark}")

    print(f"""
  关键认知：
    ✓ Multi-Query：覆盖面广，对覆盖不全的召回有效；LLM 调用 ×3
    ✓ HyDE：对"问题-文档"语义鸿沟大的场景效果突出；额外 1 次 LLM + embedding
    ✓ Step-back：对需要背景知识的问题有效；简单问题可能反而变差
    ✓ 三种方法可组合（先 Multi-Query 扩展，再 HyDE 补充）
    ✓ 每种变换额外成本：1~3 次 LLM call（~0.5~2s），要评估是否值得

    开销对比：
      Multi-Query：3 次 LLM + 3×embedding 检索
      HyDE：1 次 LLM + 1 次 embedding
      Step-back：1 次 LLM

    下一步（v8）：
      v3.5~v7 都在优化"检索"。
      评估框架（RAGAS）让你用数据说话：这些优化加在一起，RAG 系统究竟到了什么水平？
    """)

    # ── 保存结果 ─────────────────────────────────────────────
    result = {
        "strategies": {
            k: {"recall_at_3": avg(v, "recall"), "mrr": avg(v, "mrr")}
            for k, v in results.items()
        },
        "baseline": {"recall_at_3": baseline["recall_at_3"], "mrr": baseline["mrr"]},
    }
    out = Path(__file__).parent / "v7_query_transform_result.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  → 已保存到 v7_query_transform_result.json")
    print(f"  → 下一步：python 08_v8_评估框架.py\n")


if __name__ == "__main__":
    main()
