#!/usr/bin/env python3
"""
09_v9_agentic_rag.py — Agentic RAG：让 LLM 自主决策检索
=====================================
插入位置：v8（评估框架）之后、v10（企业级生产）之前

传统 RAG vs Agentic RAG：

  传统 RAG（Pipeline 式，v1~v8）：
    用户提问 → 检索 → 生成 → 返回答案
    流程固定，不管问题简单还是复杂，都执行同样的步骤。
    局限：
      简单问题浪费检索资源
      复杂问题（需要多步推理）一次检索不够

  Agentic RAG（有状态循环）：
    用户提问 → LLM 自主决策 → 是否检索？检索什么？ → 继续还是结束？
    LLM 扮演"智能体"：能使用工具，能根据上下文调整策略。

本文件实现三种模式：

  模式一：Self-RAG（自我检索判断）
    LLM 先判断"这个问题需要检索吗？"
    不需要检索的问题（常识问题）直接回答，节省延迟和成本。

  模式二：Agentic Retrieval（自主调用检索工具）
    LLM 通过 Tool Calling 主动调用 search_rag() 工具。
    LLM 自己决定检索什么、要不要再检索一次。

  模式三：Multi-hop（多步推理检索）
    复杂问题需要多次检索：
    第一次检索 → 发现还需要更多信息 → 以检索结果为线索发起第二次检索 → 综合回答

技术基础：Tool Calling（函数调用）
  OpenAI 兼容 API 的标准功能。
  LLM 返回一个 JSON 结构（不是普通文字），告诉我们"调用哪个函数、用什么参数"。
  我们执行函数，把结果塞回给 LLM，它继续推理。

依赖：pip install openai numpy python-dotenv（无额外依赖）
运行：python 09_v9_agentic_rag.py
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║  📺 讲师注释                                                  ║
# ║  对应集数：独立专题（Agentic RAG）                           ║
# ║  核心代码：第 141~210 行（search_rag 工具函数 +              ║
# ║            run_tool_call），第 211~285 行（agentic_loop       ║
# ║            主循环）                                          ║
# ║  可跳过：第 93~140 行（BM25 + 分块样板）                     ║
# ║  本集关键数字：Self-RAG 工具调用不稳定（7B 模型限制），       ║
# ║               需要 32B+ 才可靠——见                           ║
# ║               mock-interview/07_agentic_rag_模型选型.md      ║
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

# Tool Calling 需要直接操作消息数组，使用 Provider 的原始客户端
_client     = _provider_module._get_client()
_cfg        = _provider_module.get_config()


# ══════════════════════════════════════════════════════════════
# 知识库（同 v3.5~v8，复用）
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
# 检索工具（Agentic RAG 的"工具"）
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


# 全局索引（工具调用时使用）
_chunks: list[str] = []
_chunk_embs: np.ndarray = None
_bm25: BM25 = None


def search_rag(query: str, top_k: int = 3) -> str:
    """
    检索工具函数。
    这个函数会被 LLM 通过 Tool Calling 调用。
    返回格式化的检索结果字符串（方便 LLM 阅读）。
    """
    global _chunks, _chunk_embs, _bm25
    q_emb = embed(query)
    v_scores = [cosine_sim(q_emb, ce) for ce in _chunk_embs]
    v_ranked = sorted(enumerate(v_scores), key=lambda x: x[1], reverse=True)

    from collections import defaultdict
    scores: dict[int, float] = {}
    for rank, (idx, _) in enumerate(v_ranked):
        scores[idx] = scores.get(idx, 0) + 1.5 / (60 + rank + 1)
    for rank, (idx, _) in enumerate(_bm25.retrieve(query)):
        scores[idx] = scores.get(idx, 0) + 1.0 / (60 + rank + 1)
    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for i, (idx, _) in enumerate(merged[:top_k]):
        results.append(f"[片段{i+1}] {_chunks[idx]}")
    return "\n\n".join(results)


# ══════════════════════════════════════════════════════════════
# Tool Calling 辅助函数
#
# Tool Calling 流程：
#   1. 把工具定义（JSON Schema）传给 LLM
#   2. LLM 如果想调用工具，返回 tool_calls（不是普通文字）
#   3. 我们执行对应的 Python 函数
#   4. 把执行结果塞回消息历史，LLM 继续
#   5. LLM 不再调用工具时，返回最终答案
# ══════════════════════════════════════════════════════════════

SEARCH_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "search_rag",
        "description": "从知识库中检索与问题相关的文档片段。当你需要查找特定信息时调用此工具。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "用于检索的查询语句，应该是具体的技术问题或关键词"
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回的文档片段数量，默认 3",
                    "default": 3
                }
            },
            "required": ["query"]
        }
    }
}


def run_tool_call(tool_name: str, tool_args: dict) -> str:
    """执行工具调用，返回结果字符串"""
    if tool_name == "search_rag":
        return search_rag(
            query=tool_args.get("query", ""),
            top_k=tool_args.get("top_k", 3)
        )
    return f"[未知工具: {tool_name}]"


def agentic_loop(
    user_query: str,
    system_prompt: str = None,
    tools: list = None,
    max_iterations: int = 5,
    verbose: bool = True
) -> tuple[str, list]:
    """
    Agentic 对话循环：LLM 可以多次调用工具，直到给出最终答案。

    返回：(最终答案, 执行步骤列表)
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_query})

    steps = []
    tools = tools or [SEARCH_TOOL_DEF]

    for iteration in range(max_iterations):
        # 调用 LLM
        resp = _client.chat.completions.create(
            model=_cfg["chat_model"],
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.0,
        )
        msg = resp.choices[0].message

        # LLM 决定不调用工具 → 直接返回答案
        if not msg.tool_calls:
            answer = (msg.content or "").strip()
            steps.append({"type": "answer", "content": answer})
            if verbose:
                print(f"    [回答] {answer[:100]}...")
            messages.append({"role": "assistant", "content": answer})
            return answer, steps

        # LLM 决定调用工具
        messages.append(msg)  # 把包含 tool_calls 的消息追加到历史
        for tc in msg.tool_calls:
            tool_name = tc.function.name
            tool_args = json.loads(tc.function.arguments)

            if verbose:
                print(f"    [调用工具] {tool_name}({tool_args})")

            tool_result = run_tool_call(tool_name, tool_args)
            steps.append({
                "type": "tool_call",
                "tool": tool_name,
                "args": tool_args,
                "result_preview": tool_result[:100],
            })

            if verbose:
                print(f"    [工具结果] {tool_result[:80]}...")

            # 把工具结果返回给 LLM
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": tool_result,
            })

    return "[超过最大迭代次数]", steps


# ══════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════

def main():
    global _chunks, _chunk_embs, _bm25

    info = model_info()
    print(f"\n{'═'*60}")
    print(f" v9 ／ Agentic RAG（Self-RAG + Multi-hop）")
    print(f" Provider: {info['provider']}  |  Chat: {info['chat_model']}")
    print(f"{'═'*60}")

    # 检查是否支持 Tool Calling
    if _cfg.get("sdk") == "zhipu":
        print("\n  ℹ️  ZhipuAI SDK 的 Tool Calling 接口与 OpenAI 有差异。")
        print("  建议切换到 siliconflow 或 openai provider 运行本文件。\n")

    # ── STEP 1：建立检索索引 ─────────────────────────────────
    print(f"\n{'═'*60}")
    print(" STEP 1 ／ 建立检索索引")
    print(f"{'═'*60}\n")
    _chunks = chunk_by_sentence(DOCUMENT, max_chars=300)
    _chunk_embs = np.array([embed(c) for c in _chunks])
    _bm25 = BM25(_chunks)
    print(f"  → {len(_chunks)} 个 chunks 就绪")

    # ── STEP 2：模式一 — Self-RAG（先判断是否需要检索）──────
    print(f"\n{'═'*60}")
    print(" STEP 2 ／ 模式一：Self-RAG（LLM 自主判断是否需要检索）")
    print(f"{'═'*60}")

    self_rag_system = """你是一个智能问答助手，拥有知识库检索工具。

判断规则：
- 如果问题是通用常识（常见概念、基础知识），直接回答，不需要检索。
- 如果问题涉及特定的参数、具体数字、私有文档内容，调用 search_rag 工具检索后再回答。

回答要简洁准确。"""

    self_rag_cases = [
        ("什么是 Python？",                "常识问题 → 不应检索"),
        ("RAG 的分块推荐用多大？",          "知识库问题 → 应检索"),
        ("Chroma 和 Qdrant 各适合什么场景？", "知识库问题 → 应检索"),
    ]

    for query, expected in self_rag_cases:
        print(f"\n  问题：{query}")
        print(f"  预期：{expected}")
        answer, steps = agentic_loop(
            query, system_prompt=self_rag_system, verbose=True
        )
        tool_calls = [s for s in steps if s["type"] == "tool_call"]
        print(f"  工具调用次数：{len(tool_calls)}  ({'符合预期' if (len(tool_calls) == 0) == ('不应检索' in expected) else '⚠️ 不符预期'})")

    # ── STEP 3：模式二 — Agentic Retrieval ───────────────────
    print(f"\n{'═'*60}")
    print(" STEP 3 ／ 模式二：Agentic Retrieval（LLM 自主决定检索策略）")
    print(f"{'═'*60}")

    agentic_system = """你是一个 RAG 技术专家。
当需要查找特定技术细节时，使用 search_rag 工具检索知识库。
你可以多次检索以获取足够的信息。根据检索结果给出准确、详细的回答。"""

    agentic_cases = [
        "RAG 系统里 embedding 的推荐维度是多少？用什么指标衡量语义相似度？",
        "评估 RAG 系统需要关注哪些指标？检索和生成各有什么指标？",
    ]

    for query in agentic_cases:
        print(f"\n  问题：{query}")
        answer, steps = agentic_loop(query, system_prompt=agentic_system, verbose=True)
        tool_calls = [s for s in steps if s["type"] == "tool_call"]
        print(f"  共检索 {len(tool_calls)} 次")

    # ── STEP 4：模式三 — Multi-hop 推理 ─────────────────────
    print(f"\n{'═'*60}")
    print(" STEP 4 ／ 模式三：Multi-hop（多步推理，复杂问题需要多次检索）")
    print(f"{'═'*60}")
    print("""
  Multi-hop 场景：问题本身不复杂，但答案分散在多个 chunk，
  需要 LLM 先检索第一部分，再根据第一部分的信息决定下一步检索什么。
    """)

    multihop_system = """你是一个技术专家。请按步骤思考：
1. 先检索与问题直接相关的信息
2. 如果信息不完整，基于已有结果再次检索补充细节
3. 汇总所有检索结果，给出完整回答

每次检索后评估信息是否足够，不够则继续检索（最多 3 次）。"""

    multihop_query = (
        "我想搭建一个 RAG 系统：应该选哪个向量数据库？分块多大合适？"
        "评估效果用什么指标？"
    )
    print(f"  复合问题：{multihop_query}\n")
    answer, steps = agentic_loop(
        multihop_query,
        system_prompt=multihop_system,
        max_iterations=8,
        verbose=True
    )
    tool_calls = [s for s in steps if s["type"] == "tool_call"]
    print(f"\n  共发起 {len(tool_calls)} 次检索，查询词：")
    for tc in tool_calls:
        print(f"    - {tc['args'].get('query', '')}")

    # ── STEP 5：架构总结 ─────────────────────────────────────
    print(f"\n{'═'*60}")
    print(" STEP 5 ／ Agentic RAG 架构总结")
    print(f"{'═'*60}")
    print(f"""
  传统 RAG（固定管道）vs Agentic RAG（有状态循环）：

    传统 RAG                     Agentic RAG
    ─────────────────────────    ────────────────────────────────
    检索次数固定（1次）          LLM 自主决定检索次数（0~N次）
    检索 query = 用户 query      检索 query = LLM 自主生成
    流程线性、可预测             流程动态，难以预测
    延迟稳定（~1s）              延迟波动（0.5~10s）
    成本固定                     成本随问题复杂度变化

  适用场景：
    Agentic RAG 适合：复杂多跳问题、需要推理链的问题、问题本身模糊
    传统 RAG 适合：  问题明确、延迟敏感、成本敏感、结果可解释性要求高

  生产框架：
    LangGraph：用有向图描述 Agentic 流程，支持条件分支、循环、并发
    本文件展示的是最简单的实现，LangGraph 提供更完整的状态管理和错误处理

  下一步（v10）：从 Demo 到生产级系统，差距在工程。
    """)

    # ── 保存结果 ─────────────────────────────────────────────
    result = {
        "patterns_demonstrated": ["self_rag", "agentic_retrieval", "multi_hop"],
        "multi_hop_example": {
            "query": multihop_query,
            "retrieval_count": len(tool_calls),
            "search_queries": [tc["args"].get("query") for tc in tool_calls],
        },
    }
    out = Path(__file__).parent / "v9_agentic_result.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  → 已保存到 v9_agentic_result.json")
    print(f"  → 下一步：python 10_v10_enterprise.py\n")


if __name__ == "__main__":
    main()
