#!/usr/bin/env python3
"""
02_v2_文档分块策略.py — 文档分块策略
=====================================
目标：理解 Chunking 是 RAG 质量的第一道关卡
核心问题：
  - chunk_size 太小 → 语义碎片，检索到的内容不完整
  - chunk_size 太大 → 噪声多，LLM 答案精度下降
  - 无 overlap → 关键信息可能落在块边界被切断
  - 有 overlap → 边界信息保留，块数增加，存储略增

本文件复用 v1 的 embed / retrieve / chat 函数，
聚焦展示：分块决策 → 检索结果 → 最终回答质量的因果链

依赖：pip install openai numpy python-dotenv
运行：python 02_v2_文档分块策略.py
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║  📺 讲师注释                                                  ║
# ║  对应集数：01_理解RAG.html · Ep2「RAG 怎么运作」延伸         ║
# ║  核心代码：第 119~145 行（chunk_by_sentence，三种策略中的    ║
# ║            胜者），第 168~192 行（retrieve）                  ║
# ║  可跳过：第 61~116 行（固定分块变体，理解对比思路即可）      ║
# ║  本集关键数字：句子感知分块 Recall@3 最高，                  ║
# ║               chunk_size=300 为推荐起点                      ║
# ╚══════════════════════════════════════════════════════════════╝

import re
import numpy as np
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

# 复用 v1 的统一接口
_provider_path = Path(__file__).with_name("00_配置提供商_先改这个.py")
_provider_spec = spec_from_file_location("rag_provider", _provider_path)
if _provider_spec is None or _provider_spec.loader is None:
    raise ImportError(f"无法加载提供商配置文件: {_provider_path}")
_provider_module = module_from_spec(_provider_spec)
_provider_spec.loader.exec_module(_provider_module)

embed = _provider_module.embed
chat = _provider_module.chat


# ══════════════════════════════════════════════════════════════
# 测试文档：一篇关于 RAG 的技术文档（模拟企业内部 Wiki）
# 这是真实场景中你会处理的材料类型
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
# 三种分块策略（从简单到智能）
# ══════════════════════════════════════════════════════════════

def chunk_fixed_no_overlap(text: str, chunk_size: int = 150) -> list[str]:
    """
    策略 A：固定大小分块，无重叠

    实现：按字符数切割，走到哪切到哪，不看语义边界。
    优点：实现最简单，块大小完全可控。
    缺点：
      1. 可能在句子中间切断（"推荐分块大小在200到500" 可能被切成两半）
      2. 边界处的信息无法被任何一块完整捕获
    适用：快速原型验证，对精度要求不高的场景。
    """
    text = text.strip()
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size
    return chunks


def chunk_fixed_with_overlap(text: str, chunk_size: int = 200, overlap: int = 40) -> list[str]:
    """
    策略 B：固定大小分块，有重叠

    关键机制：
      - 每个新块的起点 = 上一个块的终点 - overlap
      - 即：相邻两块有 overlap 个字符是重复的

    为什么需要 overlap？
    想象一个关键句子："建议使用200到500字符，并设置重叠防止信息丢失"
    如果它刚好跨越两个块的边界：
      块A：...建议使用200到500字
      块B：符，并设置重叠防止信息丢失...
    两个块都拿不到完整的这句话，检索命中但内容残缺。
    
    有了 overlap=40，这句话会完整出现在某个块里，因为边界前移了。

    overlap 推荐值：chunk_size 的 10%~20% ？ 为什么
    """
    text = text.strip()
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        next_start = end - overlap          # ← overlap 的核心：回退 overlap 个字符
        if next_start <= start:             # 防止死循环
            next_start = start + 1
        start = next_start
        if start >= len(text):
            break
    return [c for c in chunks if len(c) > 20]


def chunk_by_sentence(text: str, max_chars: int = 300) -> list[str]:
    """
    策略 C：句子感知分块

    思路：
      1. 先按标点符号分句（中文：。！？；换行）
      2. 把句子累积到 max_chars，不超过时合并，超过时切块开新块

    优点：
      - 不会在句子中间切断，语义最完整
      - 每块是若干完整句子，检索结果更易阅读
    缺点：
      - 块大小不固定（有些块可能只有一个长句子）
      - 实现略复杂

    适用：技术文档、新闻、规范、说明书等结构良好的文本。
    """
    # 按中文标点分句，保留标点（用 lookahead 不消耗标点）
    sentence_pattern = r'(?<=[。！？；\n])'
    sentences = re.split(sentence_pattern, text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]

    chunks = []
    current_chunk = ""

    for sent in sentences:
        # 如果加上这句不超限，继续累积
        if len(current_chunk) + len(sent) <= max_chars:
            current_chunk += sent
        else:
            # 当前块已满，保存并开新块
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sent

    if current_chunk.strip():              # 别忘了最后一块
        chunks.append(current_chunk.strip())

    return [c for c in chunks if len(c) > 20]


# ══════════════════════════════════════════════════════════════
# 复用 V1 的 embedding / 检索 / 生成函数
# ══════════════════════════════════════════════════════════════

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def retrieve(query: str, chunks: list[str], top_k: int = 2) -> list[dict]:
    """从 chunks 列表中检索 Top-K 最相关的块"""
    q_emb = embed(query)
    chunk_embs = [embed(c) for c in chunks]     # 注意：生产中这步在入库时做，不在查询时做
    scores = [cosine_sim(q_emb, ce) for ce in chunk_embs]
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    return [{"text": chunks[i], "score": round(s, 4)} for i, s in ranked]


def rag_answer(query: str, contexts: list[dict]) -> str:
    """用检索到的上下文生成回答"""
    ctx = "\n".join(f"[{i+1}] {c['text']}" for i, c in enumerate(contexts))
    prompt = f"""请仅根据以下参考资料回答问题。若资料中没有相关信息，请说"资料未提及"。

参考资料：
{ctx}

问题：{query}
回答："""
    return chat(prompt, temperature=0)


# ══════════════════════════════════════════════════════════════
# 主流程：三个实验，每个都有数据支撑
# ══════════════════════════════════════════════════════════════

def print_chunks_preview(name: str, chunks: list[str]):
    avg_len = sum(len(c) for c in chunks) // len(chunks) if chunks else 0
    print(f"  策略: {name}")
    print(f"  结果: 共 {len(chunks)} 块  |  平均 {avg_len} 字符/块")
    for i, c in enumerate(chunks[:3]):
        preview = c[:55].replace('\n', ' ')
        print(f"  Chunk[{i}]({len(c)}字): \"{preview}...\"")
    if len(chunks) > 3:
        print(f"  ... 还有 {len(chunks)-3} 块")
    print()


def main():
    LINE = "─" * 60

    # ────────────────────────────────────────────────────────
    # 实验 0：展示三种分块策略的分块结果
    # ────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(" STEP 1 ／ 对同一文档，应用三种分块策略，看分块结果")
    print(f"{'═'*60}\n")
    print(f"  文档总长度: {len(DOCUMENT)} 字符  ({len(DOCUMENT)//2} 个汉字约）\n")
    print(LINE)

    chunks_A = chunk_fixed_no_overlap(DOCUMENT, chunk_size=120)
    chunks_B = chunk_fixed_with_overlap(DOCUMENT, chunk_size=200, overlap=40)
    chunks_C = chunk_by_sentence(DOCUMENT, max_chars=300)

    print_chunks_preview("A｜固定120字，无 overlap（太小，演示问题）", chunks_A)
    print_chunks_preview("B｜固定200字 + 40字 overlap（推荐基础配置）", chunks_B)
    print_chunks_preview("C｜句子感知，max=300字（语义最完整）", chunks_C)

    # ────────────────────────────────────────────────────────
    # 实验 1：可视化 overlap 的作用
    # ────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(" STEP 2 ／ Overlap 的作用：保护边界信息")
    print(f"{'═'*60}")

    # 这句话是文档中的关键信息，让我们看它有没有被切断
    KEY_SENTENCE = "推荐分块大小在200到500个字符之间，并设置约10%到20%的重叠区域"
    print(f"\n  目标关键句（含具体建议数值）:")
    print(f"  「{KEY_SENTENCE}」\n")

    # 检查关键句在三种策略中是否完整
    found_A = any(KEY_SENTENCE in c for c in chunks_A)
    found_B = any(KEY_SENTENCE in c for c in chunks_B)
    found_C = any(KEY_SENTENCE in c for c in chunks_C)

    print(f"  策略A（120字无overlap）：关键句完整存在？ {'✅ 是' if found_A else '❌ 被切断了'}")
    print(f"  策略B（200字有overlap）：关键句完整存在？ {'✅ 是' if found_B else '❌ 被切断了'}")
    print(f"  策略C（句子感知）：      关键句完整存在？ {'✅ 是' if found_C else '❌ 被切断了'}")

    if not found_A:
        # 展示关键句是怎么被切断的
        print(f"\n  📌 策略A中，关键句被切断的情况：")
        for i, c in enumerate(chunks_A):
            if "200到500" in c or "推荐分块" in c or "重叠区域" in c:
                print(f"  Chunk[{i}]: \"{c}\"")
        print(f"\n  → 看到了吗？这句话被切成两截，")
        print(f"    「数字建议」和「重叠说明」分散在两个块里，")
        print(f"    任何一个块单独来看都是信息残缺的。")

    # ────────────────────────────────────────────────────────
    # 实验 2：核心对比——不同分块 → 不同检索 → 不同回答质量
    # ────────────────────────────────────────────────────────
    QUERY = "RAG 的分块策略有什么工程建议？推荐用多大的分块和 overlap？"

    print(f"\n{'═'*60}")
    print(" STEP 3 ／ 关键实验：分块策略 → 检索质量 → 回答质量")
    print(f"{'═'*60}")
    print(f"\n  问题: {QUERY}\n")
    print(f"  （这个问题的答案在文档「二、分块策略」章节中有具体数值）\n")

    # 策略 A：块太小
    print(LINE)
    print("  策略 A｜固定 120 字，无 overlap（块太小）")
    print(LINE)
    results_A = retrieve(QUERY, chunks_A, top_k=2)
    print("  命中的 Top-2 块：")
    for r in results_A:
        preview = r['text'][:80].replace('\n', ' ')
        print(f"  [{r['score']}] \"{preview}...\"")
    ans_A = rag_answer(QUERY, results_A)
    print(f"\n  🤖 回答:\n  {ans_A}\n")

    # 策略 B：合理分块 + overlap
    print(LINE)
    print("  策略 B｜固定 200 字 + 40 字 overlap（推荐配置）")
    print(LINE)
    results_B = retrieve(QUERY, chunks_B, top_k=2)
    print("  命中的 Top-2 块：")
    for r in results_B:
        preview = r['text'][:80].replace('\n', ' ')
        print(f"  [{r['score']}] \"{preview}...\"")
    ans_B = rag_answer(QUERY, results_B)
    print(f"\n  🤖 回答:\n  {ans_B}\n")

    # 策略 C：句子感知
    print(LINE)
    print("  策略 C｜句子感知分块（语义最完整）")
    print(LINE)
    results_C = retrieve(QUERY, chunks_C, top_k=2)
    print("  命中的 Top-2 块：")
    for r in results_C:
        preview = r['text'][:80].replace('\n', ' ')
        print(f"  [{r['score']}] \"{preview}...\"")
    ans_C = rag_answer(QUERY, results_C)
    print(f"\n  🤖 回答:\n  {ans_C}\n")

    # ────────────────────────────────────────────────────────
    # 总结
    # ────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(" V2 总结：分块决策框架")
    print(f"{'═'*60}")
    print(f"""
  分块工程决策表：

  ┌──────────────┬──────────┬────────────┬────────────────┐
  │ 策略          │ 实现复杂度│ 语义完整性  │ 推荐场景        │
  ├──────────────┼──────────┼────────────┼────────────────┤
  │ 固定大小无重叠 │ 极低     │ 低（易切断）│ 快速原型        │
  │ 固定大小有重叠 │ 低       │ 中          │ 通用文档，生产基线│
  │ 句子感知      │ 中       │ 高          │ 技术文档/规范    │
  │ 语义分块      │ 高       │ 最高        │ 高精度场景（v3+）│
  └──────────────┴──────────┴────────────┴────────────────┘

  参数推荐：
    chunk_size  →  200~500 字符（视文档类型调整）
    overlap     →  chunk_size 的 10%~20%
    chunk_size 和 overlap 没有通用最优解，需要用 v8 的评估框架来度量。

  你现在理解的概念：
    ✓ chunk_size：决定每块的信息密度，影响检索粒度
    ✓ overlap：防止边界信息丢失，代价是块数量略增
    ✓ 句子感知分块：牺牲块大小一致性，换取语义完整性
    ✓ "分块太小" 和 "分块太大" 造成的问题是不同的

  ⚠️  此刻的问题：我们是靠"肉眼看输出"判断哪种策略更好。
  这是不可靠的——你的直觉可能是错的，换一个文档结论就变了。
  → V3.5 要解决：构建黄金数据集，用 Recall@K 数字说话，
    让每次分块决策都有可对比的量化依据。

  V2 的局限（V3 要解决）：
    ✗ chunks 只存在内存里，程序一退出就消失
    ✗ 每次都要重新计算所有 chunk 的 embedding（慢且费钱）
    → V3 把 chunks + embeddings 持久化到向量数据库（ChromaDB）
    """)


if __name__ == "__main__":
    main()
