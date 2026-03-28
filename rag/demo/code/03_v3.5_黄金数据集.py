#!/usr/bin/env python3
"""
03_v3.5_检索基线.py — 建立检索基线
=====================================
插入位置：V2（分块策略）之后、V4（Embedding 选型）之前

核心目的：
  在开始任何优化之前，先用一把固定的尺子量出当前系统的真实水平。
  这个数字就是基线——v4 到 v7 的每次改动，都要和它对比。

本文件只做三件事：
  1. 建立 7 条手标 query（覆盖不同难度，真实用户措辞）
  2. 用 V2 的最优分块策略跑一次检索，记录 Recall@3 + MRR
  3. 把结果存成 baseline.json，供后续版本对比

不在本文件做的事（留给后续版本）：
  - LLM 合成 query → v6
  - 多策略横向对比 → v2 已做
  - 开发集/测试集拆分 → v8
  - RAGAS 自动化评估 → v8

依赖：pip install openai numpy
运行：python 03_v3.5_检索基线.py
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║  📺 讲师注释                                                  ║
# ║  对应集数：01_理解RAG.html · Ep4「怎么知道系统好不好」前置  ║
# ║  核心代码：第 64~122 行（8条人工标注 Query，黄金数据集），   ║
# ║            第 146~174 行（recall_at_k + mrr 函数）           ║
# ║  可跳过：第 124~145 行（分块样板，v2 已讲）                  ║
# ║  本集关键数字：基线 MRR=0.71，Recall@3=0.75                  ║
# ║               （后续 v4~v7 优化的起点）                      ║
# ╚══════════════════════════════════════════════════════════════╝

import json
import numpy as np
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

# 复用统一接口
_provider_path = Path(__file__).with_name("00_配置提供商_先改这个.py")
_provider_spec = spec_from_file_location("rag_provider", _provider_path)
if _provider_spec is None or _provider_spec.loader is None:
    raise ImportError(f"无法加载提供商配置文件: {_provider_path}")
_provider_module = module_from_spec(_provider_spec)
_provider_spec.loader.exec_module(_provider_module)

embed = _provider_module.embed
cosine_sim = _provider_module.cosine_sim


# ══════════════════════════════════════════════════════════════
# 知识库文档（同 V2）
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


# ══════════════════════════════════════════════════════════════
# STEP 1：手标 Query 集合
#
# 设计原则：
#   - 用真实用户措辞，不照抄文档原文
#   - 覆盖 easy / medium / hard 三个难度
#   - relevant_text 必须是文档原文片段（检索命中的判断标准）
# ══════════════════════════════════════════════════════════════

QUERIES = [
    {
        "id": "Q001",
        "query": "RAG 的分块推荐用多大？overlap 比例是多少？",
        "relevant_text": "推荐分块大小在200到500个字符之间，并设置约10%到20%的重叠区域（overlap）防止边界信息丢失",
        "difficulty": "medium",
    },
    {
        "id": "Q002",
        "query": "RAG 解决了大模型哪些核心问题？",
        "relevant_text": "RAG解决了大模型两个核心痛点：知识截止日期和私有领域知识缺失",
        "difficulty": "easy",
    },
    {
        "id": "Q003",
        "query": "不同类型的文档应该用什么分块策略？",
        "relevant_text": "不同文档类型有不同的最优分块策略：技术文档适合句子级分块，代码文档按函数边界分块，PDF报告按段落分块",
        "difficulty": "medium",
    },
    {
        "id": "Q004",
        "query": "Chroma 和 Qdrant 各适合什么场景？",
        "relevant_text": "常用向量数据库包括Chroma（本地开发）、Qdrant（云端生产）、Pinecone（托管服务）",
        "difficulty": "easy",
    },
    {
        "id": "Q005",
        "query": "为什么分块太大会影响 RAG 质量，具体原因是什么？",
        "relevant_text": "分块太大（如2000个字符）：召回的块包含太多无关内容，LLM在长上下文中丢失焦点，答案精度降低",
        "difficulty": "hard",
    },
    {
        "id": "Q006",
        "query": "Faithfulness 指标衡量的是什么？",
        "relevant_text": "忠实度（Faithfulness）衡量答案是否忠实于检索资料而没有幻觉",
        "difficulty": "medium",
    },
    {
        "id": "Q007",
        "query": "向量维度是多少？用什么衡量两个向量的相似度？",
        "relevant_text": "每个文本块通过Embedding模型转换为高维向量，通常为1536维或3072维。向量间的语义相似度通过余弦相似度衡量",
        "difficulty": "easy",
    },
]


# ══════════════════════════════════════════════════════════════
# 分块策略：使用 V2 评估中表现最优的句子感知分块
# ══════════════════════════════════════════════════════════════

import re

def chunk_by_sentence(text, max_chars=300):
    """句子感知分块：在句子边界切割，保留完整语义单元"""
    sentences = re.split(r'(?<=[。！？；\n])', text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s) > 5]
    chunks = []
    current = ""
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


# ══════════════════════════════════════════════════════════════
# 评估指标
# ══════════════════════════════════════════════════════════════

def recall_at_k(retrieved: list[str], relevant_text: str, k: int) -> float:
    """
    Recall@K：前 K 个结果里有没有命中正确答案？
    返回 1.0（命中）或 0.0（未命中）

    这是检索系统最重要的指标：
    答案没被召回 → 不管 Prompt 多好、LLM 多强，都没用。
    """
    key = relevant_text[:15].strip()
    return 1.0 if any(key in c for c in retrieved[:k]) else 0.0


def mrr(retrieved: list[str], relevant_text: str) -> float:
    """
    MRR（Mean Reciprocal Rank）：相关文档排在第几位？
    rank=1 → 1.0，rank=2 → 0.5，rank=3 → 0.33，找不到 → 0.0

    比 Recall@K 更细：不只看"找没找到"，还看"排多靠前"。
    """
    key = relevant_text[:15].strip()
    for rank, chunk in enumerate(retrieved, start=1):
        if key in chunk:
            return 1.0 / rank
    return 0.0


# ══════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════

def main():
    print(f"\n{'═'*60}")
    print(" v3.5 ／ 建立检索基线")
    print(f"{'═'*60}")

    # ── STEP 1：分块 + 建索引 ──────────────────────────────
    print("\n STEP 1 ／ 分块 + 建 Embedding 索引\n")
    chunks = chunk_by_sentence(DOCUMENT, max_chars=300)
    print(f"  策略：句子感知分块（max_chars=300）")
    print(f"  块数：{len(chunks)}")
    for i, c in enumerate(chunks):
        print(f"  [{i}] {c[:60]}...")

    chunk_embs = []
    print("\n  建立向量索引...")
    for c in chunks:
        chunk_embs.append(embed(c))
    chunk_embs = np.array(chunk_embs)
    print(f"  → 索引矩阵: {chunk_embs.shape}")

    # ── STEP 2：逐条 Query 检索，记录结果 ─────────────────
    print(f"\n{'═'*60}")
    print(" STEP 2 ／ 逐条检索，记录 Recall@3 + MRR")
    print(f"{'═'*60}\n")

    TOP_K = 3
    records = []

    for item in QUERIES:
        q_emb = embed(item["query"])
        scores = [cosine_sim(q_emb, ce) for ce in chunk_embs]
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        retrieved = [chunks[i] for i in ranked_idx]

        r = recall_at_k(retrieved, item["relevant_text"], k=TOP_K)
        m = mrr(retrieved, item["relevant_text"])

        icon = "✅" if r == 1.0 else "❌"
        diff = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}.get(item["difficulty"], "⚪")
        print(f"  {icon} {diff} [{item['id']}] Recall@3={r:.0f}  MRR={m:.3f}")
        print(f"       Query : {item['query']}")
        print(f"       Top-1 : {retrieved[0][:60]}...")
        print()

        records.append({
            "id":       item["id"],
            "query":    item["query"],
            "difficulty": item["difficulty"],
            "recall_at_3": r,
            "mrr":      round(m, 3),
            "top1":     retrieved[0][:80],
        })

    # ── STEP 3：汇总基线数字 ───────────────────────────────
    recall_avg = sum(r["recall_at_3"] for r in records) / len(records)
    mrr_avg    = sum(r["mrr"] for r in records) / len(records)

    print(f"{'═'*60}")
    print(" STEP 3 ／ 基线数字")
    print(f"{'═'*60}")
    print(f"""
  分块策略 : 句子感知分块 max_chars=300
  Query 数 : {len(QUERIES)} 条（手标，覆盖 easy/medium/hard）
  ─────────────────────────────
  Recall@3 : {recall_avg:.3f}   ← {recall_avg*100:.0f}% 的问题在 Top-3 里找到了答案
  MRR      : {mrr_avg:.3f}   ← 平均命中位置约第 {1/mrr_avg:.1f} 位
  ─────────────────────────────

  这是基线。v4 起的每次优化，都和这两个数字比。
  数字变大 = 真的变好了。数字不变 = 优化没用。
    """)

    # ── 保存基线文件 ───────────────────────────────────────
    baseline = {
        "strategy": "chunk_by_sentence(max_chars=300)",
        "n_queries": len(QUERIES),
        "recall_at_3": round(recall_avg, 3),
        "mrr": round(mrr_avg, 3),
        "details": records,
    }
    output_path = Path(__file__).parent / "baseline.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(baseline, f, ensure_ascii=False, indent=2)
    print(f"  → 基线已保存到 baseline.json")
    print(f"  → 下一步：python 04_v4_embedding选型.py\n")


if __name__ == "__main__":
    main()
