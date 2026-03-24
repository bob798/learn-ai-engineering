#!/usr/bin/env python3
"""
03_v3.5_黄金数据集.py — 黄金数据集
=====================================
插入位置：V3（向量数据库）之后、V4（Embedding 选型）之前

核心思想（来自视频）：
  先有标准，再有系统，再有优化。
  黄金数据集是整个项目的"尺子"——
  没有它，后续所有优化都是凭直觉，而不是凭数据。

本文件做三件事：
  1. 展示如何构建黄金数据集（手标 + LLM 合成两种路线）
  2. 定义 Recall@K 和 MRR 两个检索评估指标
  3. 用黄金数据集给 V2 的三种分块策略打分，得出量化结论

依赖：pip install openai numpy
运行：export OPENAI_API_KEY="sk-xxx" && python 03_v3.5_黄金数据集.py
"""

import json
import numpy as np
from openai import OpenAI

client = OpenAI()
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL  = "gpt-4o-mini"


# ══════════════════════════════════════════════════════════════
# 知识库文档（同 V2，模拟企业内部 Wiki）
# ══════════════════════════════════════════════════════════════

DOCUMENT = """RAG系统架构与工程实践指南

一、什么是RAG
RAG是Retrieval-Augmented Generation的缩写，中文叫检索增强生成。核心思路是：在大模型回答问题之前，先从外部知识库检索相关内容，再把检索结果和用户问题一起输入给大模型，让模型基于真实资料生成回答。RAG解决了大模型两个核心痛点：知识截止日期和私有领域知识缺失。

二、分块策略的重要性
文档分块是RAG系统中最容易被忽视但影响最大的工程决策。分块太小（如50个字符）：每块缺乏完整语义，LLM无法从碎片中理解上下文，答案质量大幅下降。分块太大（如2000个字符）：召回的块包含太多无关内容，LLM在长上下文中丢失焦点，答案精度降低。工程实践建议：推荐分块大小在200到500个字符之间，并设置约10%到20%的重叠区域（overlap）防止边界信息丢失。不同文档类型有不同的最优分块策略：技术文档适合句子级分块，代码文档按函数边界分块，PDF报告按段落分块。

三、向量检索原理
每个文本块通过Embedding模型转换为高维向量，通常为1536维或3072维。向量间的语义相似度通过余弦相似度衡量：两向量夹角越小，语义越接近，余弦值越接近1。检索时将用户问题也转为向量，在向量库中找最近邻的文档向量，返回Top-K个结果。常用向量数据库包括Chroma（本地开发）、Qdrant（云端生产）、Pinecone（托管服务）。

四、RAG评估指标
评估RAG系统需要同时关注检索和生成两个维度。检索维度：召回率（Recall）衡量相关文档有没有被找到，精确率（Precision）衡量找到的文档中有多少真正相关。生成维度：忠实度（Faithfulness）衡量答案是否忠实于检索资料而没有幻觉，相关性（Answer Relevancy）衡量答案是否切中用户问题。RAGAS是目前最常用的RAG评估框架，支持上述全部指标的自动化评估。构建Golden Dataset（黄金测试集）是评估的前提，通常需要人工标注20到100条高质量的问答对。
"""


# ══════════════════════════════════════════════════════════════
# 黄金数据集结构
# 每条记录 = { query, relevant_text_fragment, difficulty, type }
# ══════════════════════════════════════════════════════════════

# 路线一：手动标注（效果最好，最真实）
# 规则：relevant_text 必须是文档原文的一段，query 是真实用户会问的问题
GOLDEN_DATASET_MANUAL = [
    {
        "id": "M001",
        "query": "RAG 的分块推荐用多大？overlap 比例是多少？",
        "relevant_text": "推荐分块大小在200到500个字符之间，并设置约10%到20%的重叠区域（overlap）防止边界信息丢失",
        "difficulty": "medium",
        "type": "factual",          # 事实性——有明确答案
        "source": "manual",
    },
    {
        "id": "M002",
        "query": "RAG 解决了大模型哪些核心问题？",
        "relevant_text": "RAG解决了大模型两个核心痛点：知识截止日期和私有领域知识缺失",
        "difficulty": "easy",
        "type": "factual",
        "source": "manual",
    },
    {
        "id": "M003",
        "query": "不同类型的文档应该用什么分块策略？",
        "relevant_text": "不同文档类型有不同的最优分块策略：技术文档适合句子级分块，代码文档按函数边界分块，PDF报告按段落分块",
        "difficulty": "medium",
        "type": "conceptual",       # 概念性——需要理解
        "source": "manual",
    },
    {
        "id": "M004",
        "query": "Chroma 和 Qdrant 各适合什么场景？",
        "relevant_text": "常用向量数据库包括Chroma（本地开发）、Qdrant（云端生产）、Pinecone（托管服务）",
        "difficulty": "easy",
        "type": "factual",
        "source": "manual",
    },
    {
        "id": "M005",
        "query": "为什么分块太大会影响 RAG 质量，具体原因是什么？",
        "relevant_text": "分块太大（如2000个字符）：召回的块包含太多无关内容，LLM在长上下文中丢失焦点，答案精度降低",
        "difficulty": "hard",
        "type": "conceptual",
        "source": "manual",
    },
    {
        "id": "M006",
        "query": "用什么工具可以自动化评估 RAG 的质量？",
        "relevant_text": "RAGAS是目前最常用的RAG评估框架，支持上述全部指标的自动化评估",
        "difficulty": "easy",
        "type": "factual",
        "source": "manual",
    },
    {
        "id": "M007",
        "query": "Faithfulness 指标衡量的是什么？",
        "relevant_text": "忠实度（Faithfulness）衡量答案是否忠实于检索资料而没有幻觉",
        "difficulty": "medium",
        "type": "factual",
        "source": "manual",
    },
]


def generate_synthetic_queries(chunks: list[str], n_per_chunk: int = 1) -> list[dict]:
    """
    路线二：LLM 合成查询

    何时使用：
    - 文档太多，手动标注成本太高
    - 需要覆盖更多文档段落
    - 快速搭建初始数据集（但质量不如手标）

    注意事项：
    - 合成查询偏向文档原文措辞，可能低估真实用户的检索难度
    - 应与手标数据集混合使用，不能完全替代
    - 生产系统上线后，要用真实用户 query 逐步替换合成 query
    """
    synthetic = []
    for i, chunk in enumerate(chunks[:4]):   # 只处理前4块，控制 API 成本
        prompt = f"""请根据以下文档片段，生成1个真实用户可能会问的问题。
要求：
- 问题要能从这段文字中找到答案
- 问题要自然，像真人在用知识库时会问的
- 不要直接复述文档原文，要换一种表达方式
- 只输出问题本身，不要任何解释

文档片段：
{chunk[:300]}

问题："""
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        query = resp.choices[0].message.content.strip()
        synthetic.append({
            "id": f"S{i+1:03d}",
            "query": query,
            "relevant_text": chunk[:200],
            "difficulty": "medium",
            "type": "synthetic",
            "source": "llm_generated",
        })
    return synthetic


# ══════════════════════════════════════════════════════════════
# 评估指标
# ══════════════════════════════════════════════════════════════

def embed(text: str) -> np.ndarray:
    resp = client.embeddings.create(input=text, model=EMBED_MODEL)
    return np.array(resp.data[0].embedding)

def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def recall_at_k(retrieved_chunks: list[str], relevant_text: str, k: int) -> float:
    """
    Recall@K：在前 K 个结果里，有没有包含正确答案？

    计算方式：
    - 遍历前 K 个 chunk
    - 如果任何一个 chunk 包含 relevant_text 的关键词 → 命中
    - 返回 1.0（命中）或 0.0（未命中）

    这是检索系统最重要的指标：
    "正确答案有没有被捞到？"
    如果 Recall@K 低，不管 Prompt 多好、LLM 多强，都没用。

    注意：这里用简单的文本包含判断。
    生产环境可以用 embedding 相似度阈值来判断"相关"。
    """
    top_k = retrieved_chunks[:k]
    # 提取 relevant_text 的核心关键词（取前 15 个字）
    key = relevant_text[:15].strip()
    for chunk in top_k:
        if key in chunk:
            return 1.0
    return 0.0


def mrr(retrieved_chunks: list[str], relevant_text: str) -> float:
    """
    MRR（Mean Reciprocal Rank）：相关文档排在第几位？

    计算方式：
    - 找到第一个包含相关内容的 chunk 的位置 rank
    - 返回 1 / rank
    - rank=1 → MRR=1.0（最好），rank=3 → MRR=0.33，找不到 → 0.0

    MRR 不仅看"有没有找到"，还看"排在哪里"。
    RAG 里 Top-1 的文档质量比 Top-5 影响更大，因为 Top-1 更可能被放进 Prompt。
    """
    key = relevant_text[:15].strip()
    for rank, chunk in enumerate(retrieved_chunks, start=1):
        if key in chunk:
            return 1.0 / rank
    return 0.0


def evaluate_retrieval(chunks: list[str], golden: list[dict], top_k: int = 3) -> dict:
    """
    对一套 chunks（一种分块策略的结果）跑完整评估
    返回：recall@k, mrr, 以及每条记录的详情
    """
    # 预计算所有 chunk 的 embedding（真实系统里这步在入库时做）
    chunk_embs = [embed(c) for c in chunks]

    results = []
    for item in golden:
        q_emb = embed(item["query"])
        scores = [cosine_sim(q_emb, ce) for ce in chunk_embs]
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        retrieved = [chunks[i] for i in ranked_idx]

        r_at_k = recall_at_k(retrieved, item["relevant_text"], k=top_k)
        mrr_val = mrr(retrieved, item["relevant_text"])

        results.append({
            "id":       item["id"],
            "query":    item["query"][:40] + "...",
            "recall":   r_at_k,
            "mrr":      round(mrr_val, 3),
            "top1":     retrieved[0][:50] + "...",
        })

    recall_avg = sum(r["recall"] for r in results) / len(results)
    mrr_avg    = sum(r["mrr"] for r in results) / len(results)

    return {
        "recall_at_k": round(recall_avg, 3),
        "mrr":         round(mrr_avg, 3),
        "details":     results,
    }


# ══════════════════════════════════════════════════════════════
# 从 V2 复制三种分块策略（这里只保留函数，不重复注释）
# ══════════════════════════════════════════════════════════════

import re

def chunk_fixed_no_overlap(text, chunk_size=120):
    text = text.strip(); chunks = []; start = 0
    while start < len(text):
        c = text[start:start+chunk_size].strip()
        if c: chunks.append(c)
        start += chunk_size
    return chunks

def chunk_fixed_with_overlap(text, chunk_size=200, overlap=40):
    text = text.strip(); chunks = []; start = 0
    while start < len(text):
        end = start + chunk_size
        c = text[start:end].strip()
        if c: chunks.append(c)
        next_start = end - overlap
        if next_start <= start: next_start = start + 1
        start = next_start
        if start >= len(text): break
    return [c for c in chunks if len(c) > 20]

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


# ══════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════

def main():
    LINE = "─" * 60

    # ────────────────────────────────────────────────────────
    # STEP 1: 建立黄金数据集（手标 + 合成）
    # ────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(" STEP 1 ／ 建立黄金数据集")
    print(f"{'═'*60}")

    # 先用句子感知分块，生成合成 query 用的 chunk
    chunks_for_synth = chunk_by_sentence(DOCUMENT, max_chars=300)

    print(f"\n  [手动标注] {len(GOLDEN_DATASET_MANUAL)} 条（覆盖不同难度和类型）")
    for item in GOLDEN_DATASET_MANUAL:
        diff_icon = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}.get(item["difficulty"], "⚪")
        print(f"  {diff_icon} [{item['id']}][{item['type']:12s}] {item['query'][:45]}...")

    print(f"\n  [LLM 合成] 正在生成...")
    synthetic = generate_synthetic_queries(chunks_for_synth)
    for item in synthetic:
        print(f"  🤖 [{item['id']}] {item['query'][:45]}...")

    # 合并数据集
    full_dataset = GOLDEN_DATASET_MANUAL + synthetic
    print(f"\n  → 合并后：{len(full_dataset)} 条（手标 {len(GOLDEN_DATASET_MANUAL)} + 合成 {len(synthetic)}）")

    # 拆分：开发集（优化用）+ 测试集（最终评估用，不参与调参）
    split_idx    = int(len(full_dataset) * 0.8)
    dev_set      = full_dataset[:split_idx]
    test_set     = full_dataset[split_idx:]
    print(f"  → 开发集: {len(dev_set)} 条  |  测试集（封存）: {len(test_set)} 条")
    print(f"\n  ⚠️  测试集一旦划分，在最终评估之前不能用于调参。")
    print(f"     如果用测试集来调参，你就在为特定数据集过拟合，")
    print(f"     报出的指标是虚高的，上生产后会打脸。")

    # ────────────────────────────────────────────────────────
    # STEP 2: 用黄金数据集给三种分块策略打分
    # ────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(" STEP 2 ／ 用 Recall@3 + MRR 给 V2 的三种分块策略打分")
    print(f"{'═'*60}")
    print(f"  使用开发集评估 | 问题数: {len(dev_set)}")

    strategies = [
        ("A｜固定120字无overlap",       chunk_fixed_no_overlap(DOCUMENT, 120)),
        ("B｜固定200字+40字overlap",     chunk_fixed_with_overlap(DOCUMENT, 200, 40)),
        ("C｜句子感知max=300",           chunk_by_sentence(DOCUMENT, 300)),
    ]

    scores = []
    for name, chunks in strategies:
        print(f"\n  评估中: {name} ({len(chunks)} 块)...")
        result = evaluate_retrieval(chunks, dev_set, top_k=3)
        scores.append((name, result, len(chunks)))

        print(f"  Recall@3: {result['recall_at_k']:.3f}  |  MRR: {result['mrr']:.3f}")
        print(f"  {'─'*50}")
        for d in result["details"]:
            icon = "✅" if d["recall"] == 1.0 else "❌"
            print(f"  {icon} [{d['id']}] MRR={d['mrr']}  {d['query']}")

    # ────────────────────────────────────────────────────────
    # STEP 3: 汇总对比，给出有数据支撑的结论
    # ────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(" STEP 3 ／ 汇总：现在有数字了")
    print(f"{'═'*60}")
    print(f"\n  {'策略':<25}  {'Recall@3':>9}  {'MRR':>7}  {'块数':>5}")
    print(f"  {'─'*50}")
    best_recall = max(s[1]['recall_at_k'] for s in scores)
    for name, result, n_chunks in scores:
        marker = " ← 最佳" if result['recall_at_k'] == best_recall else ""
        print(f"  {name:<25}  {result['recall_at_k']:>9.3f}  {result['mrr']:>7.3f}  {n_chunks:>5}{marker}")

    print(f"""
  关键认知：
    ✓ 现在你有数字了，可以说"策略 C 的 Recall@3 是 X，比 A 高 Y%"
    ✓ 不同难度 / 类型的 query，命中率可能差异很大
    ✓ MRR 告诉你的信息比 Recall@K 更细：
      - Recall@K=1 只说"找到了"
      - MRR=1.0 说"排第一就找到"，MRR=0.33 说"排第三才找到"

  数据集的局限（要诚实面对）：
    ✗ 手标 7 条太少，统计意义弱——生产系统要 50~100 条
    ✗ 合成 query 措辞接近文档，实际用户语言更口语化
    ✗ relevant_text 用简单字符串匹配判断，有误判风险
    → 这是 v3.5 的起点，评估体系会在 v8 进一步完善

  现在你完成了：
    v1  最小 RAG 循环
    v2  分块策略
    v3  （下一步）向量数据库持久化
    v3.5 黄金数据集 + 检索评估 ← 你在这里

  v4 开始的每次优化，都用这把尺子量一下。
    """)

    # 可选：保存数据集到文件
    with open("golden_dataset.json", "w", encoding="utf-8") as f:
        json.dump({"dev": dev_set, "test": test_set}, f, ensure_ascii=False, indent=2)
    print("  → 已保存到 golden_dataset.json（测试集封存，不参与后续调参）")


if __name__ == "__main__":
    main()
