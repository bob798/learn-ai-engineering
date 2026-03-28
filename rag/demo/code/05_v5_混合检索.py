#!/usr/bin/env python3
"""
05_v5_混合检索.py — BM25 + 向量 + RRF 混合检索
=====================================
插入位置：v4（Embedding 选型）之后、v6（Reranking）之前

核心问题：
  纯向量检索有一个致命弱点：遇到精确关键词匹配时会失效。
  例如：用户问「Qdrant」，文档里就有「Qdrant」这个词，
  但向量检索可能把「Chroma」的块排更靠前（语义相似）。

本文件做三件事：
  1. 演示向量检索的"精确词盲区"——什么情况下纯向量会失败
  2. 用 BM25 做稀疏检索，补向量检索的短板
  3. 用 RRF（互惠排名融合）合并两路结果，对比 baseline 数字

不依赖外部库：BM25 用纯 Python 实现（约30行），理解原理比调包重要。

核心认知：
  - 向量检索：擅长语义理解（换种说法也能找到）
  - BM25：擅长精确匹配（关键词一字不差地出现）
  - 混合检索：取长补短，不是随意叠加，而是有理论支撑的融合
  - RRF 的魔法：不需要归一化两路得分，直接按排名合并，鲁棒性强

依赖：pip install openai numpy python-dotenv
运行：python 05_v5_混合检索.py
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║  📺 讲师注释                                                  ║
# ║  对应集数：01_理解RAG.html · Ep3「检索为什么失败」           ║
# ║  核心代码：第 63~131 行（BM25 类），                         ║
# ║            第 133~170 行（rrf_merge，本文件最重要的函数）    ║
# ║  可跳过：第 185~233 行（eval 样板，与 v3.5 相同）            ║
# ║  本集关键数字：MRR 0.71 → 0.85（纯向量 → 混合检索）         ║
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
cosine_sim = _provider_module.cosine_sim
model_info = _provider_module.model_info


# ══════════════════════════════════════════════════════════════
# BM25 实现（纯 Python，不依赖外部库）
#
# BM25 是什么？
#   最经典的关键词检索算法，Google 最早用的就是它的变体。
#   核心思想：一个词在这段文字里出现得越多、在其他文字里越少见，
#   这段文字和查询的相关性就越高。
#
# 两个关键参数：
#   k1 = 1.5：控制词频饱和度（同一个词重复出现，边际效益递减）
#   b  = 0.75：控制文档长度惩罚（长文档里出现一个词，含金量低于短文档）
# ══════════════════════════════════════════════════════════════

class BM25:
    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b  = b
        self.N  = len(corpus)
        self.tokenized = [self._tokenize(doc) for doc in corpus]
        self.avgdl = sum(len(d) for d in self.tokenized) / self.N

        # 文档频率：每个词出现在多少个 chunk 里
        self.df: dict[str, int] = {}
        for doc in self.tokenized:
            for token in set(doc):
                self.df[token] = self.df.get(token, 0) + 1

    def _tokenize(self, text: str) -> list[str]:
        """
        中文分词：字符级切分
        为什么用字级而不是词级？
          词级分词（jieba）更精准，但引入额外依赖。
          对中文 RAG 场景，字级 BM25 已经能捕捉关键词，够用。
          生产环境可以替换成 jieba 分词提升精度。
        """
        # 保留中文字符和英文单词，去掉标点
        tokens = re.findall(r'[\u4e00-\u9fff]|[a-zA-Z0-9]+', text)
        return tokens

    def score(self, query: str, doc_idx: int) -> float:
        """计算 query 和第 doc_idx 个文档的 BM25 得分"""
        doc    = self.tokenized[doc_idx]
        dl     = len(doc)
        tf_map = Counter(doc)
        q_tokens = self._tokenize(query)

        total = 0.0
        for token in q_tokens:
            if token not in self.df:
                continue
            # IDF：词越罕见，权重越高
            idf = math.log(
                (self.N - self.df[token] + 0.5) / (self.df[token] + 0.5) + 1
            )
            # TF（归一化）：词频越高越好，但有饱和上限；长文档有惩罚
            tf  = tf_map.get(token, 0)
            tf_norm = tf * (self.k1 + 1) / (
                tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            )
            total += idf * tf_norm
        return total

    def retrieve(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """返回 [(doc_idx, score), ...] 按得分降序"""
        scores = [(i, self.score(query, i)) for i in range(self.N)]
        return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]


# ══════════════════════════════════════════════════════════════
# RRF — 互惠排名融合（Reciprocal Rank Fusion）
#
# 为什么不直接加权平均两路得分？
#   向量检索的 cosine 值范围是 0~1，
#   BM25 得分范围是 0~几十（取决于语料）。
#   直接加权 → BM25 得分把向量得分压死，失去意义。
#
# RRF 的解法：
#   不管原始得分是多少，只看排名。
#   每路检索各贡献 1/(k + rank) 分（k=60 是经验默认值）。
#   两路排名加在一起，就是最终排名。
#   优点：不需要归一化，鲁棒性强，已被学术界广泛验证。
# ══════════════════════════════════════════════════════════════

def rrf_merge(
    vector_ranked: list[tuple[int, float]],
    bm25_ranked:   list[tuple[int, float]],
    k: int = 60,
    w_vector: float = 1.0,
    w_bm25:   float = 1.0,
) -> list[tuple[int, float]]:
    """
    加权 RRF：合并两路检索结果，返回 [(doc_idx, rrf_score), ...] 降序

    参数：
      k        : 平滑因子（Cormack 2009 推荐 60，适合大语料）
      w_vector : 向量检索的权重（默认 1.0）
      w_bm25   : BM25 检索的权重（默认 1.0）

    为什么加权？
      当两路排名对同一 query 完全互换时（item A 向量第1/BM25第2，
      item B 向量第2/BM25第1），unweighted RRF 得分精确相等，
      排名由 Python dict 插入顺序决定，结果不可控。
      加权后 w_bm25 > w_vector 时，BM25 rank-1 的贡献更大，
      对精确词匹配场景（如 Q001 含 "overlap"）给出正确排名。

    典型设置：
      中文精确词较多 → w_bm25=1.5, w_vector=1.0
      语义理解为主   → w_vector=1.5, w_bm25=1.0
      不确定         → 1.0 / 1.0（等权，观察后再调）
    """
    scores: dict[int, float] = {}
    for rank, (idx, _) in enumerate(vector_ranked):
        scores[idx] = scores.get(idx, 0) + w_vector / (k + rank + 1)
    for rank, (idx, _) in enumerate(bm25_ranked):
        scores[idx] = scores.get(idx, 0) + w_bm25 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ══════════════════════════════════════════════════════════════
# 辅助函数
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


def recall_at_k(retrieved_chunks, relevant_text, k):
    key = relevant_text[:15].strip()
    return 1.0 if any(key in c for c in retrieved_chunks[:k]) else 0.0


def mrr_score(retrieved_chunks, relevant_text):
    key = relevant_text[:15].strip()
    for rank, chunk in enumerate(retrieved_chunks, start=1):
        if key in chunk:
            return 1.0 / rank
    return 0.0


# ══════════════════════════════════════════════════════════════
# 知识库文档（同 v3.5）
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


def main():
    info = model_info()
    print(f"\n{'═'*60}")
    print(f" v5 ／ 混合检索（BM25 + 向量 + RRF）")
    print(f" Provider: {info['provider']}  |  Embed: {info['embed_model']}")
    print(f"{'═'*60}")

    # ── STEP 1：加载 baseline ──────────────────────────────
    print("\n STEP 1 ／ 加载基线\n")
    baseline_path = Path(__file__).parent / "baseline.json"
    if not baseline_path.exists():
        print("  ⚠️  未找到 baseline.json，请先运行 03_v3.5_检索基线.py")
        return
    with open(baseline_path, encoding="utf-8") as f:
        baseline = json.load(f)
    print(f"  基线 Recall@3 : {baseline['recall_at_3']}")
    print(f"  基线 MRR      : {baseline['mrr']}")

    # ── STEP 2：建索引 ─────────────────────────────────────
    print(f"\n{'═'*60}")
    print(" STEP 2 ／ 建立双路索引")
    print(f"{'═'*60}\n")

    chunks = chunk_by_sentence(DOCUMENT, max_chars=300)
    print(f"  分块数: {len(chunks)}")

    # 向量索引
    print("  建立向量索引（Embedding）...")
    chunk_embs = np.array([embed(c) for c in chunks])
    print(f"  → 向量矩阵: {chunk_embs.shape}")

    # BM25 索引
    print("  建立 BM25 索引（关键词）...")
    bm25 = BM25(chunks)
    print(f"  → 词典大小: {len(bm25.df)} 个 token")

    # ── STEP 3：演示两种检索的互补性 ──────────────────────
    print(f"\n{'═'*60}")
    print(" STEP 3 ／ 互补性演示")
    print(f"{'═'*60}")

    demo_cases = [
        {
            "desc": "精确词匹配场景（BM25 优势）",
            "query": "Qdrant 是什么",
            "note": "文档里有「Qdrant」这个词，向量可能因语义聚类把 Chroma/Pinecone 排更前",
        },
        {
            "desc": "语义理解场景（向量优势）",
            "query": "如何让模型基于事实而不是凭空编造",
            "note": "没有「幻觉」「Faithfulness」原词，纯关键词匹配会失败",
        },
    ]

    for case in demo_cases:
        q = case["query"]
        q_emb = embed(q)

        # 向量检索
        v_scores = [cosine_sim(q_emb, ce) for ce in chunk_embs]
        v_ranked = sorted(enumerate(v_scores), key=lambda x: x[1], reverse=True)

        # BM25 检索
        b_ranked = bm25.retrieve(q, top_k=len(chunks))

        print(f"\n  【{case['desc']}】")
        print(f"  Query : {q}")
        print(f"  说明  : {case['note']}")
        print(f"\n  向量 Top-3:")
        for rank, (idx, score) in enumerate(v_ranked[:3], 1):
            print(f"    [{rank}] cos={score:.4f}  {chunks[idx][:55]}...")
        print(f"\n  BM25  Top-3:")
        for rank, (idx, score) in enumerate(b_ranked[:3], 1):
            print(f"    [{rank}] bm25={score:.2f}  {chunks[idx][:55]}...")

    # ── STEP 4：四路对比评估 ───────────────────────────────
    print(f"\n{'═'*60}")
    print(" STEP 4 ／ 四路对比：纯向量 vs 纯BM25 vs 等权RRF vs 加权RRF")
    print(f"{'═'*60}")
    print(f"""
  为什么加等权 RRF 还不够？
    当两路排名对同一 query 完全互换时（A:向量第1/BM25第2，B:向量第2/BM25第1），
    unweighted RRF 得分精确相等，由 Python dict 插入顺序决定，结果不可控。
    加权 RRF（w_bm25 > w_vector）使 BM25 rank-1 贡献更大，打破对称平局。
  本次设置：w_vector=1.0, w_bm25=1.5（中文精确词场景 BM25 权重更高）
    """)

    TOP_K = 3
    # 加权 RRF 参数（中文语料中精确词匹配重要，BM25 权重略高）
    W_VECTOR = 1.0
    W_BM25   = 1.5

    results = {"vector": [], "bm25": [], "hybrid_eq": [], "hybrid_w": []}

    for item in QUERIES:
        q_emb = embed(item["query"])

        # 向量检索
        v_scores = [cosine_sim(q_emb, ce) for ce in chunk_embs]
        v_ranked = sorted(enumerate(v_scores), key=lambda x: x[1], reverse=True)
        v_chunks = [chunks[i] for i, _ in v_ranked]

        # BM25 检索
        b_ranked = bm25.retrieve(item["query"], top_k=len(chunks))
        b_chunks = [chunks[i] for i, _ in b_ranked]

        # 等权 RRF（w=1:1）
        eq_merged = rrf_merge(v_ranked, b_ranked, w_vector=1.0, w_bm25=1.0)
        eq_chunks = [chunks[i] for i, _ in eq_merged]

        # 加权 RRF（w_bm25 > w_vector）
        w_merged = rrf_merge(v_ranked, b_ranked, w_vector=W_VECTOR, w_bm25=W_BM25)
        w_chunks = [chunks[i] for i, _ in w_merged]

        r_v  = recall_at_k(v_chunks,  item["relevant_text"], TOP_K)
        r_b  = recall_at_k(b_chunks,  item["relevant_text"], TOP_K)
        r_eq = recall_at_k(eq_chunks, item["relevant_text"], TOP_K)
        r_w  = recall_at_k(w_chunks,  item["relevant_text"], TOP_K)

        m_v  = mrr_score(v_chunks,  item["relevant_text"])
        m_b  = mrr_score(b_chunks,  item["relevant_text"])
        m_eq = mrr_score(eq_chunks, item["relevant_text"])
        m_w  = mrr_score(w_chunks,  item["relevant_text"])

        diff_icon = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}.get(item["difficulty"], "⚪")
        # 标记加权后是否改善
        improved = " ★ 加权改善" if m_w > m_eq else ""
        print(f"  {diff_icon} [{item['id']}]  "
              f"向量={r_v:.0f}/{m_v:.2f}  "
              f"BM25={r_b:.0f}/{m_b:.2f}  "
              f"等权={r_eq:.0f}/{m_eq:.2f}  "
              f"加权={r_w:.0f}/{m_w:.2f}{improved}")
        print(f"       {item['query']}")

        results["vector"].append({"recall": r_v,  "mrr": m_v})
        results["bm25"].append(  {"recall": r_b,  "mrr": m_b})
        results["hybrid_eq"].append({"recall": r_eq, "mrr": m_eq})
        results["hybrid_w"].append( {"recall": r_w,  "mrr": m_w})

    # ── STEP 5：汇总 ──────────────────────────────────────
    def avg(lst, key):
        return round(sum(x[key] for x in lst) / len(lst), 3)

    v_recall  = avg(results["vector"],    "recall")
    b_recall  = avg(results["bm25"],      "recall")
    eq_recall = avg(results["hybrid_eq"], "recall")
    w_recall  = avg(results["hybrid_w"],  "recall")
    v_mrr     = avg(results["vector"],    "mrr")
    b_mrr     = avg(results["bm25"],      "mrr")
    eq_mrr    = avg(results["hybrid_eq"], "mrr")
    w_mrr     = avg(results["hybrid_w"],  "mrr")

    print(f"\n{'═'*60}")
    print(" STEP 5 ／ 汇总")
    print(f"{'═'*60}\n")

    rows = [
        ("[基线] 纯向量（v3.5）",            baseline["recall_at_3"], baseline["mrr"]),
        ("纯向量（本次重跑）",                v_recall,                v_mrr),
        ("纯 BM25",                          b_recall,                b_mrr),
        ("混合 RRF（等权 1:1）",             eq_recall,               eq_mrr),
        (f"混合 RRF（加权 {W_VECTOR}:{W_BM25}）", w_recall,           w_mrr),
    ]
    print(f"  {'方案':<32}  {'Recall@3':>9}  {'MRR':>7}  {'vs 基线':>8}")
    print(f"  {'─'*62}")
    best_mrr = max(mrr_val for _, _, mrr_val in rows)
    for name, recall, mrr_val in rows:
        delta = mrr_val - baseline["mrr"]
        delta_str = f"+{delta:.3f}" if delta > 0 else (f"{delta:.3f}" if delta < 0 else "  ─   ")
        marker = " ← 最佳 MRR" if mrr_val == best_mrr and mrr_val > baseline["mrr"] else ""
        print(f"  {name:<32}  {recall:>9.3f}  {mrr_val:>7.3f}  {delta_str:>8}{marker}")

    print(f"""
  关键认知：
    ✓ 向量检索对同义词/换说法鲁棒，对精确词匹配弱
    ✓ BM25 对精确词匹配强，对语义理解弱
    ✓ 等权 RRF 在两路互换排名时会产生平局，结果不可控
    ✓ 加权 RRF（w_bm25 > w_vector）让 BM25 在精确词场景有发言权
    ✓ 权重是超参，需要在你自己的语料上评测后设定，不要照搬

    更根本的解法（v6）：
      Q001 的问题是 query 里带了主题词"RAG"导致向量干扰。
      Query 改写（剥离主题词 → 聚焦子问题）才是真正的根治方案。
    """)

    # ── 保存结果 ───────────────────────────────────────────
    v5_result = {
        "strategy":             f"hybrid_rrf_weighted(w_vector={W_VECTOR}, w_bm25={W_BM25}, k=60)",
        "recall_at_3":          w_recall,
        "mrr":                  w_mrr,
        "baseline_recall_at_3": baseline["recall_at_3"],
        "breakdown": {
            "vector":    {"recall_at_3": v_recall,  "mrr": v_mrr},
            "bm25":      {"recall_at_3": b_recall,  "mrr": b_mrr},
            "hybrid_eq": {"recall_at_3": eq_recall, "mrr": eq_mrr},
            "hybrid_w":  {"recall_at_3": w_recall,  "mrr": w_mrr},
        },
    }
    output_path = Path(__file__).parent / "v5_hybrid_result.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(v5_result, f, ensure_ascii=False, indent=2)

    print(f"  → 已保存到 v5_hybrid_result.json")
    print(f"  → 下一步：python 06_v6_reranking.py\n")


if __name__ == "__main__":
    main()
