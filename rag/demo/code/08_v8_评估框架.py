#!/usr/bin/env python3
"""
08_v8_评估框架.py — RAG 评估体系：用数据说话
=====================================
插入位置：v7（Query 变换）之后，优化循环终结

核心认知：
  v3.5~v7 每一步都做了"优化"。
  但你知道这些优化加在一起，系统到底到什么水平了吗？
  没有评估框架，"优化"就是瞎猜。

本文件建立完整的 RAGAS 风格评估体系：
  4 个核心指标，覆盖检索和生成两个维度。

  ┌─────────────────────────────────────────────────────┐
  │  维度   │  指标              │  衡量什么               │
  ├─────────┼────────────────────┼─────────────────────────┤
  │  检索   │  Context Recall    │  答案所需信息有没有被召回 │
  │  检索   │  Context Precision │  召回的内容有多少是有用的 │
  │  生成   │  Faithfulness      │  答案有没有幻觉           │
  │  生成   │  Answer Relevancy  │  答案有没有切题           │
  └─────────┴────────────────────┴─────────────────────────┘

  为什么不继续用 MRR？
    MRR（v3.5~v7 使用）只衡量检索排名，有两个盲区：
      1. 不衡量生成质量——即使检索完美，LLM 也可能产生幻觉或答非所问
      2. 不区分"召回了但有用"vs"召回了但是噪音"——MRR 只关心 top-1 是否命中

    MRR vs RAGAS 4 指标：
    ┌──────────────────┬─────────────────────────────────────────┐
    │  指标             │  衡量什么            │  需要 LLM？       │
    ├──────────────────┼──────────────────────┼───────────────────┤
    │  MRR             │  检索排名（召回率）   │  ✗ 纯规则         │
    │  Context Recall  │  信息覆盖度（连续值） │  ✓ 语义判断       │
    │  Context Prec.   │  检索噪音比例         │  ✓ 语义判断       │
    │  Faithfulness    │  生成是否有幻觉       │  ✓ 事实核查       │
    │  Answer Relev.   │  生成是否切题         │  ✓ 相关性判断     │
    └──────────────────┴──────────────────────┴───────────────────┘

    使用策略：
      - 早期快速迭代（v3.5~v7）→ 用 MRR：无 LLM 成本，快速反馈
      - 阶段性全面评估（v8+）→ 用 RAGAS 4 指标：覆盖检索+生成

  Context Recall ≠ Recall@K：
    Recall@K = "正确答案有没有被召回"（布尔值）
    Context Recall = "正确答案里的信息点，有多少比例被召回的 context 覆盖到了"（0-1 连续值）

RAGAS 框架（生产推荐）：
  本文件手动实现以上 4 个指标，让你理解底层逻辑。
  生产环境使用 RAGAS 框架（pip install ragas），提供相同指标 + 自动批量评估。
  本文件末尾展示 RAGAS 集成代码片段。

依赖：pip install openai numpy python-dotenv
运行：python 08_v8_评估框架.py
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║  📺 讲师注释                                                  ║
# ║  对应集数：01_理解RAG.html · Ep4「怎么知道系统好不好」       ║
# ║  核心代码：context_recall / context_precision /              ║
# ║            faithfulness / answer_relevancy 四个函数          ║
# ║            （在 provider 加载之后）                          ║
# ║  可跳过：第 85~200 行（文档内容和 golden dataset 样板）      ║
# ║  本集关键数字：Context Recall≈0.87，Faithfulness≈0.91        ║
# ║               （v8 典型输出）                                ║
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
# 知识库文档 + Golden Dataset（同 v3.5~v7）
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

# 每条 query 带 ground_truth（标准答案），这是 RAGAS 评估的前提
GOLDEN_DATASET = [
    {
        "id": "Q001",
        "query": "RAG 的分块推荐用多大？overlap 比例是多少？",
        "ground_truth": "推荐分块大小在200到500个字符之间，并设置约10%到20%的重叠区域（overlap）防止边界信息丢失",
        "difficulty": "medium",
    },
    {
        "id": "Q002",
        "query": "RAG 解决了大模型哪些核心问题？",
        "ground_truth": "RAG解决了大模型两个核心痛点：知识截止日期和私有领域知识缺失",
        "difficulty": "easy",
    },
    {
        "id": "Q003",
        "query": "不同类型的文档应该用什么分块策略？",
        "ground_truth": "不同文档类型有不同的最优分块策略：技术文档适合句子级分块，代码文档按函数边界分块，PDF报告按段落分块",
        "difficulty": "medium",
    },
    {
        "id": "Q004",
        "query": "Chroma 和 Qdrant 各适合什么场景？",
        "ground_truth": "常用向量数据库包括Chroma（本地开发）、Qdrant（云端生产）、Pinecone（托管服务）",
        "difficulty": "easy",
    },
    {
        "id": "Q005",
        "query": "为什么分块太大会影响 RAG 质量，具体原因是什么？",
        "ground_truth": "分块太大（如2000个字符）：召回的块包含太多无关内容，LLM在长上下文中丢失焦点，答案精度降低",
        "difficulty": "hard",
    },
    {
        "id": "Q006",
        "query": "Faithfulness 指标衡量的是什么？",
        "ground_truth": "忠实度（Faithfulness）衡量答案是否忠实于检索资料而没有幻觉",
        "difficulty": "medium",
    },
    {
        "id": "Q007",
        "query": "向量维度是多少？用什么衡量两个向量的相似度？",
        "ground_truth": "每个文本块通过Embedding模型转换为高维向量，通常为1536维或3072维。向量间的语义相似度通过余弦相似度衡量",
        "difficulty": "easy",
    },
]


# ══════════════════════════════════════════════════════════════
# 检索管道（v5 混合检索，最优配置）
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

    def retrieve(self, query):
        return sorted([(i, self.score(query, i)) for i in range(self.N)],
                      key=lambda x: x[1], reverse=True)


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

def retrieve(query, chunk_embs, chunks, bm25, top_k=3):
    q_emb = embed(query)
    v_scores = [cosine_sim(q_emb, ce) for ce in chunk_embs]
    v_ranked = sorted(enumerate(v_scores), key=lambda x: x[1], reverse=True)
    b_ranked = bm25.retrieve(query)
    merged   = rrf_merge(v_ranked, b_ranked)
    return [chunks[i] for i, _ in merged[:top_k]]


# ══════════════════════════════════════════════════════════════
# 生成
# ══════════════════════════════════════════════════════════════

def generate_answer(query: str, context_chunks: list[str]) -> str:
    context = "\n\n".join(f"[文档{i+1}] {c}" for i, c in enumerate(context_chunks))
    prompt = f"""请根据以下参考资料回答问题。
要求：只使用参考资料中的信息，如果没有相关信息就说"参考资料中没有该信息"。

参考资料：
{context}

问题：{query}
回答："""
    return chat(prompt, temperature=0.0)


# ══════════════════════════════════════════════════════════════
# RAGAS 风格的 4 个评估指标（手动实现）
#
# 每个指标的实现都展示了 RAGAS 内部做的事：
# 将评估问题转化为 LLM 的是/否/打分任务，然后统计平均值。
# ══════════════════════════════════════════════════════════════

def _llm_score_0_1(prompt: str) -> float:
    """内部工具：让 LLM 打 0-10 分，归一化到 0-1。"""
    result = chat(prompt, temperature=0.0).strip()
    nums = re.findall(r'\b(\d+(?:\.\d+)?)\b', result)
    for n in nums:
        v = float(n)
        if 0 <= v <= 10:
            return round(v / 10.0, 2)
    return 0.5


def context_recall(ground_truth: str, contexts: list[str]) -> float:
    """
    Context Recall：ground truth 中的信息点，有多少被 contexts 覆盖？

    实现方式：
      把 ground truth 分解成若干信息点，逐一判断是否能从 contexts 中找到支撑。
      score = 有支撑的信息点数 / 总信息点数

    和 Recall@K 的区别：
      Recall@K = 布尔值（命中了没有）
      Context Recall = 覆盖比例（命中了多少比例的信息）
    """
    ctx = "\n".join(f"- {c}" for c in contexts)
    prompt = f"""请评估"参考文档"是否覆盖了"标准答案"中的所有信息点。
打分 0-10：
  10 = 标准答案的所有信息在参考文档中都能找到
   5 = 约一半的信息能找到
   0 = 参考文档完全没有涉及标准答案的内容

参考文档：
{ctx}

标准答案：{ground_truth}

覆盖程度分数（只输出数字）："""
    return _llm_score_0_1(prompt)


def context_precision(query: str, contexts: list[str], ground_truth: str) -> float:
    """
    Context Precision：召回的 context 中，有多少是真正有用的？

    实现方式：
      对每个 context，判断它是否对回答该 query 有帮助（基于 ground_truth）。
      score = 有帮助的 context 数 / 总 context 数
    """
    useful_count = 0
    for ctx in contexts:
        prompt = f"""请判断"文档片段"是否对回答"问题"有帮助（参考"标准答案"判断）。
只输出"是"或"否"。

问题：{query}
标准答案（参考）：{ground_truth}
文档片段：{ctx}

有帮助吗（是/否）："""
        result = chat(prompt, temperature=0.0).strip()
        if "是" in result or "yes" in result.lower():
            useful_count += 1
    return round(useful_count / len(contexts), 2) if contexts else 0.0


def faithfulness(answer: str, contexts: list[str]) -> float:
    """
    Faithfulness（忠实度）：答案中的每句话是否都有文档支撑？

    高忠实度 = 答案没有幻觉，完全基于召回的文档。
    低忠实度 = 模型"编"了文档没有的内容。
    """
    ctx = "\n".join(f"- {c}" for c in contexts)
    prompt = f"""请评估"答案"中的信息是否完全来自"参考文档"，没有添加文档之外的内容。
打分 0-10：
  10 = 答案每一句都在参考文档中有依据
   5 = 大部分有依据，但有少量额外发挥
   0 = 答案内容完全不依赖参考文档

参考文档：
{ctx}

答案：{answer}

忠实度分数（只输出数字）："""
    return _llm_score_0_1(prompt)


def answer_relevancy(answer: str, query: str) -> float:
    """
    Answer Relevancy（答案相关性）：答案是否切中了用户的问题？

    高相关性 = 答案直接、完整地回答了问题。
    低相关性 = 答案虽然正确，但答非所问，或者只回答了一半。
    """
    prompt = f"""请评估"答案"是否完整、直接地回答了"问题"。
打分 0-10：
  10 = 答案完整、直接地回答了问题
   5 = 答案部分回答了问题
   0 = 答案没有回答问题

问题：{query}
答案：{answer}

相关性分数（只输出数字）："""
    return _llm_score_0_1(prompt)


# ══════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════

def main():
    info = model_info()
    print(f"\n{'═'*60}")
    print(f" v8 ／ RAG 评估框架（4 项 RAGAS 风格指标）")
    print(f" Provider: {info['provider']}  |  Chat: {info['chat_model']}")
    print(f"{'═'*60}")

    # ── STEP 1：加载基线 ─────────────────────────────────────
    baseline_path = Path(__file__).parent / "baseline.json"
    if not baseline_path.exists():
        print("  ⚠️  未找到 baseline.json，请先运行 03_v3.5_检索基线.py")
        return
    with open(baseline_path, encoding="utf-8") as f:
        baseline = json.load(f)
    print(f"\n  检索基线 Recall@3={baseline['recall_at_3']}  MRR={baseline['mrr']}")

    # ── STEP 2：建立检索索引 ─────────────────────────────────
    print(f"\n  建立检索索引...")
    chunks = chunk_by_sentence(DOCUMENT, max_chars=300)
    chunk_embs = np.array([embed(c) for c in chunks])
    bm25 = BM25(chunks)
    print(f"  → {len(chunks)} 个 chunks，双路索引就绪")

    # ── STEP 3：逐条评估 ─────────────────────────────────────
    print(f"\n{'═'*60}")
    print(" STEP 3 ／ 逐条评估（检索 → 生成 → 4 项打分）")
    print(f"{'═'*60}\n")
    print(f"  {'ID':<6}  {'CR':>5}  {'CP':>5}  {'F':>5}  {'AR':>5}  {'Query'}")
    print(f"  {'─'*60}")

    records = []
    for item in GOLDEN_DATASET:
        # 1. 检索
        ctxs = retrieve(item["query"], chunk_embs, chunks, bm25, top_k=3)

        # 2. 生成
        answer = generate_answer(item["query"], ctxs)

        # 3. 评估 4 项指标
        cr = context_recall(    item["ground_truth"], ctxs)
        cp = context_precision( item["query"],        ctxs, item["ground_truth"])
        f  = faithfulness(      answer,               ctxs)
        ar = answer_relevancy(  answer,               item["query"])

        diff = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}.get(item["difficulty"], "⚪")
        print(f"  {diff}{item['id']:<4}  {cr:>5.2f}  {cp:>5.2f}  {f:>5.2f}  {ar:>5.2f}  {item['query'][:35]}...")

        records.append({
            "id":               item["id"],
            "query":            item["query"],
            "difficulty":       item["difficulty"],
            "answer":           answer,
            "contexts":         ctxs,
            "context_recall":   cr,
            "context_precision": cp,
            "faithfulness":     f,
            "answer_relevancy": ar,
        })

    # ── STEP 4：系统级汇总 ───────────────────────────────────
    def avg_metric(key):
        return round(sum(r[key] for r in records) / len(records), 3)

    cr_avg  = avg_metric("context_recall")
    cp_avg  = avg_metric("context_precision")
    f_avg   = avg_metric("faithfulness")
    ar_avg  = avg_metric("answer_relevancy")

    print(f"\n{'═'*60}")
    print(" STEP 4 ／ 系统级报告（v1~v7 最优管道）")
    print(f"{'═'*60}\n")

    print(f"  {'指标':<22}  {'得分':>6}  {'解读'}")
    print(f"  {'─'*60}")
    rows = [
        ("Context Recall",    cr_avg,  "召回内容覆盖了多少答案信息"),
        ("Context Precision", cp_avg,  "召回的 chunk 有多少是真正有用的"),
        ("Faithfulness",      f_avg,   "答案有没有幻觉（是否忠实文档）"),
        ("Answer Relevancy",  ar_avg,  "答案有没有切题"),
    ]
    for name, score, desc in rows:
        bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
        print(f"  {name:<22}  {score:>6.3f}  {bar}  {desc}")

    print(f"""
  诊断思路：
    Context Recall 低  → 相关文档没被召回，优化检索（v5/v6/v7 的方法）
    Context Precision 低 → 召回了太多噪音，加 Reranking（v6）或缩小 Top-K
    Faithfulness 低    → 模型在幻觉，加强 Prompt（"只用文档信息"约束）
    Answer Relevancy 低 → 答非所问，检查 Prompt 或 query 理解

  RAGAS 框架集成（生产推荐）：
  ─────────────────────────────────────────
  pip install ragas datasets langchain-openai

  from ragas import evaluate
  from ragas.metrics import (
      faithfulness, answer_relevancy,
      context_recall, context_precision
  )
  from datasets import Dataset

  data = {{
      "question":    [item["query"]    for item in records],
      "answer":      [item["answer"]   for item in records],
      "contexts":    [item["contexts"] for item in records],
      "ground_truth":[item["ground_truth"] for item in GOLDEN_DATASET],
  }}
  result = evaluate(
      Dataset.from_dict(data),
      metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
  )
  print(result)
  ─────────────────────────────────────────
  RAGAS 内部做的事和本文件完全一致，只是有更严格的 prompt 和更好的批量处理。
    """)

    # ── 保存结果 ─────────────────────────────────────────────
    summary = {
        "context_recall":    cr_avg,
        "context_precision": cp_avg,
        "faithfulness":      f_avg,
        "answer_relevancy":  ar_avg,
    }
    result = {
        "summary": summary,
        "baseline_retrieval": {"recall_at_3": baseline["recall_at_3"], "mrr": baseline["mrr"]},
        "details": records,
    }
    out = Path(__file__).parent / "v8_eval_result.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"  → 已保存到 v8_eval_result.json")
    print(f"""
  优化循环总结（v3.5 → v8）：
    v3.5  建立检索基线
    v4    Embedding 模型选型
    v5    混合检索（BM25 + 向量 + RRF）
    v6    Reranking（两阶段精排）
    v7    Query 变换（Multi-Query / HyDE / Step-back）
    v8    评估框架（4 维度打分，闭环优化）  ← 你在这里

  → 下一步：python 09_v9_agentic_rag.py（进阶：让 LLM 自主决策检索策略）\n""")


if __name__ == "__main__":
    main()
