#!/usr/bin/env python3
"""
04_v4_embedding选型.py — 跨厂商 Embedding 模型横向对比
=====================================
插入位置：v3.5（检索基线）之后、v5（混合检索）之前

核心问题：
  同一批 Query，不同厂商、不同模型的 Recall@3 差多少？凭什么选？

本文件做三件事：
  1. 加载 v3.5 的 baseline.json，作为对比起点
  2. 同时调用多个厂商的 Embedding API，对同一批 Query 跑评估
  3. 逐条对比，找出哪些 Query 因换模型而改变结果，给出选型建议

设计原则：
  - 每个 Candidate 携带完整的 provider 配置（api_key_env / base_url / sdk）
  - 跳过未配置 API Key 的 Candidate，不报错，只打印提示
  - run_eval 接受 embed_fn 参数，不依赖全局状态

核心认知：
  - Embedding 模型不是越大越好，是越适合你的语料越好
  - 中文语料 → 优先选中文专项模型（BGE-large-zh、BGE-M3）
  - 专业术语语料（医疗/法律）→ 需要领域微调的模型
  - 评测过再选，不要凭直觉，不要凭厂商宣传

依赖：pip install openai numpy python-dotenv zhipuai
运行：python 04_v4_embedding选型.py
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║  📺 讲师注释                                                  ║
# ║  对应集数：独立专题（Embedding 选型）                        ║
# ║  核心代码：第 127~170 行（build_embed_fn，多模型动态切换），  ║
# ║            第 198~220 行（run_eval 评估循环）                 ║
# ║  可跳过：第 37~126 行（provider 配置和 ZhipuAI 样板）        ║
# ║  本集关键数字：7条 Query 样本量不足，统计意义不强——          ║
# ║               见 mock-interview/06_embedding选型参考.md 的分析║
# ╚══════════════════════════════════════════════════════════════╝

import json
import os
import re
import numpy as np
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

# ── 加载 Provider 基础工具（只取 cosine_sim，不依赖当前 Provider 配置）──
_provider_path = Path(__file__).with_name("00_配置提供商_先改这个.py")
_provider_spec = spec_from_file_location("rag_provider", _provider_path)
if _provider_spec is None or _provider_spec.loader is None:
    raise ImportError(f"无法加载提供商配置文件: {_provider_path}")
_provider_module = module_from_spec(_provider_spec)
_provider_spec.loader.exec_module(_provider_module)

cosine_sim = _provider_module.cosine_sim


# ══════════════════════════════════════════════════════════════
# 候选模型列表（跨厂商）
#
# 每个 Candidate 是一个独立单元，包含完整的接入信息：
#   - api_key_env : .env 中的环境变量名
#   - base_url    : API 地址（None = 使用官方默认）
#   - sdk         : "openai_compat" | "zhipu"
#   - model       : 模型 ID
#
# 添加新模型：直接在列表里追加一个 dict，无需改其他代码。
# 跳过逻辑：api_key_env 对应的环境变量未设置时自动跳过。
# ══════════════════════════════════════════════════════════════

CANDIDATES = [
    {
        "name":        "SiliconFlow · BGE-M3",
        "provider":    "siliconflow",
        "model":       "BAAI/bge-m3",
        "base_url":    "https://api.siliconflow.cn/v1",
        "api_key_env": "SILICONFLOW_API_KEY",
        "sdk":         "openai_compat",
        "note":        "中英双语，1024维，国内学习首选",
    },
    {
        "name":        "SiliconFlow · BGE-large-zh",
        "provider":    "siliconflow",
        "model":       "BAAI/bge-large-zh-v1.5",
        "base_url":    "https://api.siliconflow.cn/v1",
        "api_key_env": "SILICONFLOW_API_KEY",
        "sdk":         "openai_compat",
        "note":        "中文专项，1024维，纯中文语料通常更准",
    },
    {
        "name":        "ZhipuAI · embedding-3",
        "provider":    "zhipu",
        "model":       "embedding-3",
        "base_url":    None,
        "api_key_env": "ZHIPU_API_KEY",
        "sdk":         "zhipu",
        "note":        "2048维，中文理解强，智谱出品",
    },
    {
        "name":        "OpenAI · text-embedding-3-small",
        "provider":    "openai",
        "model":       "text-embedding-3-small",
        "base_url":    None,
        "api_key_env": "OPENAI_API_KEY",
        "sdk":         "openai_compat",
        "note":        "1536维，速度快，cost 低",
    },
    {
        "name":        "OpenAI · text-embedding-3-large",
        "provider":    "openai",
        "model":       "text-embedding-3-large",
        "base_url":    None,
        "api_key_env": "OPENAI_API_KEY",
        "sdk":         "openai_compat",
        "note":        "3072维，OpenAI 最高精度，cost 约 3x",
    },
    {
        "name":        "Qwen · text-embedding-v3",
        "provider":    "qwen",
        "model":       "text-embedding-v3",
        "base_url":    "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key_env": "DASHSCOPE_API_KEY",
        "sdk":         "openai_compat",
        "note":        "1024/2048维可选，阿里出品",
    },
]


# ══════════════════════════════════════════════════════════════
# 每个 Candidate 独立建 Client
#
# 为什么不复用 00_配置文件 的 _get_client()？
#   00 里的 client 是单例，绑定到当前 PROVIDER。
#   这里需要同时访问多个厂商，每个 Candidate 需要独立的 client 实例。
# ══════════════════════════════════════════════════════════════

def build_embed_fn(candidate: dict):
    """
    为指定 Candidate 构建一个 embed(text) -> np.ndarray 函数。
    如果 API Key 未配置，返回 None（调用方负责跳过）。
    """
    api_key = os.getenv(candidate["api_key_env"], "")
    if not api_key:
        return None

    sdk  = candidate["sdk"]
    model = candidate["model"]

    if sdk == "openai_compat":
        from openai import OpenAI
        kwargs: dict = {"api_key": api_key}
        if candidate.get("base_url"):
            kwargs["base_url"] = candidate["base_url"]
        client = OpenAI(**kwargs)

        def embed_fn(text: str) -> np.ndarray:
            resp = client.embeddings.create(input=text, model=model)
            return np.array(resp.data[0].embedding)

    elif sdk == "zhipu":
        try:
            from zhipuai import ZhipuAI
        except ImportError:
            print(f"  ⚠️  未安装 zhipuai，跳过 {candidate['name']}（pip install zhipuai）")
            return None
        client = ZhipuAI(api_key=api_key)

        def embed_fn(text: str) -> np.ndarray:
            resp = client.embeddings.create(input=text, model=model)
            return np.array(resp.data[0].embedding)

    else:
        raise ValueError(f"不支持的 SDK 类型: {sdk}")

    return embed_fn


# ══════════════════════════════════════════════════════════════
# 评估函数
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


def recall_at_k(retrieved, relevant_text, k):
    key = relevant_text[:15].strip()
    return 1.0 if any(key in c for c in retrieved[:k]) else 0.0


def mrr_score(retrieved, relevant_text):
    key = relevant_text[:15].strip()
    for rank, chunk in enumerate(retrieved, start=1):
        if key in chunk:
            return 1.0 / rank
    return 0.0


def run_eval(chunks: list[str], embed_fn, queries: list[dict], top_k: int = 3) -> tuple:
    """
    用给定的 embed_fn 对 chunks 建索引，对 queries 跑评估。
    返回 (records, recall_avg, mrr_avg)
    """
    chunk_embs = np.array([embed_fn(c) for c in chunks])
    records = []
    for item in queries:
        q_emb = embed_fn(item["query"])
        scores = [cosine_sim(q_emb, ce) for ce in chunk_embs]
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        retrieved = [chunks[i] for i in ranked_idx]
        r = recall_at_k(retrieved, item["relevant_text"], k=top_k)
        m = mrr_score(retrieved, item["relevant_text"])
        records.append({
            "id": item["id"],
            "query": item["query"],
            "difficulty": item["difficulty"],
            "recall_at_3": r,
            "mrr": round(m, 3),
            "top1": retrieved[0][:60],
        })
    recall_avg = sum(r["recall_at_3"] for r in records) / len(records)
    mrr_avg    = sum(r["mrr"] for r in records) / len(records)
    return records, round(recall_avg, 3), round(mrr_avg, 3)


# ══════════════════════════════════════════════════════════════
# 知识库文档 + Query（和 v3.5 保持一致）
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
# 主流程
# ══════════════════════════════════════════════════════════════

def main():
    print(f"\n{'═'*60}")
    print(f" v4 ／ Embedding 模型跨厂商对比")
    print(f"{'═'*60}")

    # ── STEP 1：加载 baseline ──────────────────────────────
    print("\n STEP 1 ／ 加载 v3.5 基线\n")
    baseline_path = Path(__file__).parent / "baseline.json"
    if not baseline_path.exists():
        print("  ⚠️  未找到 baseline.json，请先运行 03_v3.5_检索基线.py")
        return
    with open(baseline_path, encoding="utf-8") as f:
        baseline = json.load(f)
    print(f"  基线 Recall@3 : {baseline['recall_at_3']}")
    print(f"  基线 MRR      : {baseline['mrr']}")
    print(f"  基线策略      : {baseline['strategy']}")

    # ── STEP 2：建 Chunk（固定，和 v3.5 相同）──────────────
    chunks = chunk_by_sentence(DOCUMENT, max_chars=300)

    # ── STEP 3：逐个 Candidate 评估 ───────────────────────
    print(f"\n{'═'*60}")
    print(" STEP 2 ／ 逐模型评估（跳过未配置 API Key 的）")
    print(f"{'═'*60}")

    all_results = []

    for cand in CANDIDATES:
        print(f"\n  {'─'*50}")
        print(f"  模型 : {cand['name']}")
        print(f"  说明 : {cand['note']}")

        embed_fn = build_embed_fn(cand)
        if embed_fn is None:
            print(f"  ⏭  跳过（{cand['api_key_env']} 未设置）")
            continue

        print(f"  建立向量索引 + 评估中...")
        try:
            records, recall_avg, mrr_avg = run_eval(chunks, embed_fn, QUERIES)
        except Exception as e:
            print(f"  ✗  调用失败: {e}")
            continue

        # 向量维度从第一条记录的 embed 结果推断
        dim = len(embed_fn(chunks[0]))
        all_results.append({
            "name":        cand["name"],
            "provider":    cand["provider"],
            "model":       cand["model"],
            "dim":         dim,
            "recall_at_3": recall_avg,
            "mrr":         mrr_avg,
            "details":     records,
        })
        print(f"  ✓  维度={dim}  Recall@3={recall_avg:.3f}  MRR={mrr_avg:.3f}")

    if not all_results:
        print("\n  ⚠️  没有可用的 Candidate，请在 .env 中至少配置一个 API Key")
        return

    # ── STEP 4：逐条差异分析（任意两个模型间） ────────────
    if len(all_results) >= 2:
        print(f"\n{'═'*60}")
        print(" STEP 3 ／ 逐条差异分析")
        print(f"{'═'*60}\n")

        # 找出任意 Query 在不同模型间结果不一致的情况
        for q_idx, item in enumerate(QUERIES):
            row = [(r["name"], r["details"][q_idx]["recall_at_3"],
                    r["details"][q_idx]["mrr"]) for r in all_results]
            recalls = [v for _, v, _ in row]
            if len(set(recalls)) == 1:
                continue  # 所有模型结果相同，跳过

            diff_icon = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}.get(item["difficulty"], "⚪")
            print(f"  {diff_icon} [{item['id']}] {item['query']}")
            for name, recall, mrr_val in row:
                icon = "✅" if recall == 1.0 else "❌"
                print(f"    {icon} {name:<35} Recall={recall:.0f}  MRR={mrr_val:.3f}")
            print()

        # 检查是否所有 Query 都一致
        all_same = all(
            len({r["details"][i]["recall_at_3"] for r in all_results}) == 1
            for i in range(len(QUERIES))
        )
        if all_same:
            print("  所有模型在本数据集上 Recall@3 完全一致。")
            print("  → 语料较短时模型差异不明显；真实生产场景（专业术语/长文档）差距会放大。")

    # ── STEP 5：汇总排名 ───────────────────────────────────
    print(f"\n{'═'*60}")
    print(" STEP 4 ／ 汇总排名")
    print(f"{'═'*60}\n")

    sorted_results = sorted(all_results, key=lambda r: (r["recall_at_3"], r["mrr"]), reverse=True)

    print(f"  {'模型':<38}  {'维度':>5}  {'Recall@3':>9}  {'MRR':>7}  {'vs 基线':>8}")
    print(f"  {'─'*72}")
    print(f"  {'[基线]':<38}  {'─':>5}  {baseline['recall_at_3']:>9.3f}  {baseline['mrr']:>7.3f}  {'─':>8}")

    best_recall = baseline["recall_at_3"]
    best_model  = None
    for r in sorted_results:
        delta     = r["recall_at_3"] - baseline["recall_at_3"]
        delta_str = f"+{delta:.3f}" if delta > 0 else (f"{delta:.3f}" if delta < 0 else "  ─   ")
        marker    = " ← 最佳" if r == sorted_results[0] and r["recall_at_3"] >= best_recall else ""
        print(f"  {r['name']:<38}  {r['dim']:>5}  {r['recall_at_3']:>9.3f}  {r['mrr']:>7.3f}  {delta_str:>8}{marker}")
        if r["recall_at_3"] > best_recall or (r["recall_at_3"] == best_recall and r["mrr"] > baseline["mrr"]):
            best_recall = r["recall_at_3"]
            best_model  = r

    print(f"""
  选型原则（按优先级）：
    1. Recall@3 更高的优先
    2. Recall@3 相同时，MRR 更高的优先（命中排名更靠前）
    3. 指标相同时，选维度更小的（存储省，检索快）
    4. 同等条件下，优先国内厂商（延迟低，合规）
    """)

    winner = best_model or sorted_results[0]
    print(f"  → 本次推荐：{winner['name']}")
    if best_model:
        print(f"     比基线提升：Recall@3 {winner['recall_at_3'] - baseline['recall_at_3']:+.3f}  MRR {winner['mrr'] - baseline['mrr']:+.3f}")
    else:
        print(f"     与基线持平，无明显提升")

    # ── 保存结果 ───────────────────────────────────────────
    v4_result = {
        "best_model":             winner["model"],
        "best_model_name":        winner["name"],
        "best_provider":          winner["provider"],
        "recall_at_3":            winner["recall_at_3"],
        "mrr":                    winner["mrr"],
        "baseline_recall_at_3":   baseline["recall_at_3"],
        "n_candidates_evaluated": len(all_results),
        "all_candidates": [
            {
                "name":        r["name"],
                "provider":    r["provider"],
                "model":       r["model"],
                "dim":         r["dim"],
                "recall_at_3": r["recall_at_3"],
                "mrr":         r["mrr"],
            }
            for r in sorted_results
        ],
    }
    output_path = Path(__file__).parent / "v4_embedding_result.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(v4_result, f, ensure_ascii=False, indent=2)

    print(f"\n  → 已保存到 v4_embedding_result.json")
    print(f"  → 下一步：python 05_v5_混合检索.py\n")


if __name__ == "__main__":
    main()
