#!/usr/bin/env python3
"""
01_v1_最小RAG循环.py — 最小 RAG 循环（支持国内模型）
=====================================
修改提供商：编辑 00_配置提供商_先改这个.py 中的 PROVIDER 变量

快速启动：
  # 1. 安装依赖
  pip install openai numpy python-dotenv

  # 2. 配置（复制模板并编辑）
  cp .env.example .env
  # 编辑 .env 文件填入 API Key 和选择提供商

  # 3. 运行
  python 01_v1_最小RAG循环.py
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║  📺 讲师注释                                                  ║
# ║  对应集数：01_理解RAG.html · Ep2「RAG 怎么运作」             ║
# ║  核心代码：第 48~65 行（retrieve + build_prompt），          ║
# ║            STEP 3&4（有/无 RAG 对比）                        ║
# ║  可跳过：第 37~47 行（provider 加载样板）                    ║
# ║  本集关键数字：有 RAG vs 无 RAG 的回答质量对比               ║
# ║               （肉眼判断，无量化指标）                        ║
# ╚══════════════════════════════════════════════════════════════╝

import numpy as np
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_provider_path = Path(__file__).with_name("00_配置提供商_先改这个.py")
_provider_spec = spec_from_file_location("rag_provider", _provider_path)
if _provider_spec is None or _provider_spec.loader is None:
    raise ImportError(f"无法加载提供商配置文件: {_provider_path}")
_provider_module = module_from_spec(_provider_spec)
_provider_spec.loader.exec_module(_provider_module)

embed = _provider_module.embed
chat = _provider_module.chat
cosine_sim = _provider_module.cosine_sim
model_info = _provider_module.model_info

DOCUMENTS = [
    "RAG（检索增强生成）在生成答案前先检索外部文档，让回答有据可查，有效减少大模型幻觉。",
    "向量数据库将文本转为高维向量存储，检索时通过余弦相似度找到语义最接近的文档块。",
    "LangChain 是主流 LLM 应用框架，封装了文档加载、分块、检索、生成的完整流程链路。",
    "Transformer 的核心是注意力机制（Attention），让模型在生成时动态关注输入的不同部分。",
    "RAG 系统的瓶颈在检索而非生成：召回了错误文档，生成再好也没用——垃圾进垃圾出。",
]
# 【假设】这 5 条文本已经是分好块的 chunk。
# 真实场景里它们来自 PDF/Word/Wiki 经过分块处理后的结果。
# 分块策略（chunk_size、overlap、按句子切割）直接影响这里的质量。
# → V2 专门解决"如何分块"这个问题。


def retrieve(query: str, doc_embeddings: np.ndarray, top_k: int = 2) -> list[dict]:
    q_emb = embed(query)
    scores = [cosine_sim(q_emb, de) for de in doc_embeddings]
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    return [{"text": DOCUMENTS[i], "score": round(s, 4)} for i, s in ranked]


def build_prompt(query: str, contexts: list[dict]) -> str:
    ctx = "\n".join(f"  [{i+1}] {c['text']}" for i, c in enumerate(contexts))
    return f"""你是一个严谨的知识助手。请仅根据以下参考资料回答问题。
若参考资料中没有相关信息，请明确说"资料中未提及"，不要猜测。

参考资料：
{ctx}

问题：{query}
回答："""


def main():
    info = model_info()
    print(f"\n{'═'*60}")
    print(f" 当前提供商: {info['provider']}  |  Embedding: {info['embed_model']}")
    print(f"{'═'*60}")

    # STEP 1: 为知识库建立 Embedding 索引
    print("\n STEP 1 ／ 为知识库建立 Embedding 索引\n")
    doc_embeddings = []
    for i, doc in enumerate(DOCUMENTS):
        emb = embed(doc)
        doc_embeddings.append(emb)
        print(f"  Doc[{i}] {doc[:32]}...")
        print(f"         维度={emb.shape[0]}  示例(前4维)={emb[:4].round(5)}")

    doc_embeddings = np.array(doc_embeddings)
    print(f"\n  → 知识库矩阵形状: {doc_embeddings.shape}  [文档数 × 向量维度]")
    print(f"\n  💡 为什么要变成向量？为什么用余弦不用欧氏距离？")
    print(f"     → 详见《概念手册》../docs/01_概念手册_向量与检索.html 第1节、第2节")

    # STEP 2: 语义检索
    QUERY = "RAG 系统里最关键的环节是什么？"
    print(f"\n{'═'*60}")
    print(" STEP 2 ／ 语义检索")
    print(f"{'═'*60}")
    print(f"  用户问题: {QUERY}\n")

    q_emb = embed(QUERY)
    scores = [cosine_sim(q_emb, de) for de in doc_embeddings]

    print("  全库相似度（热力条）：")
    for i, (doc, score) in enumerate(zip(DOCUMENTS, scores)):
        bar = "█" * int(score * 40)
        print(f"  [{i}] {score:.4f}  {bar}")
        print(f"       {doc[:50]}...")

    results = retrieve(QUERY, doc_embeddings, top_k=2)
    print(f"\n  → Top-2 命中：")
    for r in results:
        print(f"     ✦ [{r['score']}]  {r['text']}")

    # STEP 3 & 4: RAG Prompt + 对比实验
    prompt_rag = build_prompt(QUERY, results)

    print(f"\n{'═'*60}")
    print(" STEP 3 & 4 ／ 有 RAG vs 无 RAG 对比")
    print(f"{'═'*60}")

    print("\n  ❌ 无 RAG（直接问 LLM）：")
    ans_raw = chat(QUERY)
    for line in ans_raw.split('\n'):
        print(f"  │ {line}")

    print("\n  ✅ 有 RAG（基于检索内容）：")
    ans_rag = chat(prompt_rag)
    for line in ans_rag.split('\n'):
        print(f"  │ {line}")

    print(f"\n{'═'*60}")
    print(" V1 核心概念回顾")
    print(f"{'═'*60}")
    print("""
  ✓ Embedding：文本 → N 维向量（语义的数字表示）
  ✓ 余弦相似度：衡量两向量的方向相似程度（0~1）
  ✓ Top-K 检索：从全库找最相关的 K 个文档
  ✓ Prompt 注入：把检索结果"喂"给 LLM

  → 深度原理：../docs/01_概念手册_向量与检索.html
  → 下一步：python 02_v2_文档分块策略.py
    """)


if __name__ == "__main__":
    main()
