#!/usr/bin/env python3
"""
11_模拟面试官.py — 基于 ai-handbook 知识库的文字版模拟面试官
=====================================
首次运行：扫描 ai-handbook 内容文件 → 向量化 → 存入 ChromaDB
后续运行：直接加载已有知识库 → 开始面试

面试官能力：
  - 从 interview_qa.json 有序出题（RAG / MCP / Agent）
  - 根据候选人回答追问，调用知识库核实答案
  - 每轮独立评估：对照 key_points 打分，写入 logs/ JSONL
  - 全程保留对话历史，面试官记得每一轮的问答

依赖：pip install chromadb openai numpy python-dotenv langchain-text-splitters bs4 prompt_toolkit
运行：python 11_模拟面试官.py
退出：Ctrl+C
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║  📺 配置说明                                                  ║
# ║  - 使用 .env 中的 PROVIDER 配置（同其他脚本）                ║
# ║  - 需要 Tool Calling 支持                                    ║
# ║    推荐：siliconflow（Qwen2.5-72B）/ openai                  ║
# ║    不推荐：zhipu（Tool Calling 接口与 OpenAI 有差异）         ║
# ╚══════════════════════════════════════════════════════════════╝

import datetime
import json
import random
import re
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

try:
    from prompt_toolkit import prompt as pt_prompt
    from prompt_toolkit.history import InMemoryHistory
    _PT_AVAILABLE = True
except ImportError:
    _PT_AVAILABLE = False

try:
    from langchain_text_splitters import HTMLHeaderTextSplitter, RecursiveCharacterTextSplitter
except ImportError:
    raise ImportError("请先安装：pip install langchain-text-splitters")

try:
    import chromadb
except ImportError:
    raise ImportError("请先安装：pip install chromadb")

import numpy as np  # noqa: F401（embed() 返回 np.ndarray，需要 import）

# ── 加载 Provider 接口 ───────────────────────────────────────
_provider_path = Path(__file__).with_name("00_配置提供商_先改这个.py")
_provider_spec = spec_from_file_location("rag_provider", _provider_path)
if _provider_spec is None or _provider_spec.loader is None:
    raise ImportError(f"无法加载提供商配置文件: {_provider_path}")
_provider_module = module_from_spec(_provider_spec)
_provider_spec.loader.exec_module(_provider_module)

embed      = _provider_module.embed
model_info = _provider_module.model_info
_cfg       = _provider_module.get_config()
_client    = _provider_module._get_client()


# ══════════════════════════════════════════════════════════════
# 知识库配置
# ══════════════════════════════════════════════════════════════

# rag/code/ → rag/ → ai-handbook/
HANDBOOK_ROOT   = Path(__file__).parent.parent.parent
DB_PATH         = Path(__file__).parent / "interview_kb"
COLLECTION_NAME = "ai_handbook"
# 分块策略版本：改变分块方式时递增，触发自动重建知识库
KB_VERSION      = "v2_header_split"

# 明确列出高价值文件，跳过纯导航/索引页
_EXPLICIT_SOURCES = [
    ("rag",   "rag/docs/rag-5d.html"),                              # 综合 Q&A，质量最高
    ("rag",   "rag/docs/01_理解RAG.html"),                          # 原理讲解
    ("rag",   "rag/docs/02_概念手册_向量与检索.html"),               # 技术深度
    ("rag",   "rag/docs/03_代码讲解_V1V2.html"),                    # 代码理解
    ("rag",   "rag/docs/04_工程方法论手册.html"),                    # 工程实践
    ("mcp",   "mcp/05-interview/qa.md"),                            # MCP 面试题库
    ("mcp",   "mcp/05-interview/common-misconceptions.md"),         # MCP 常见误解
    ("mcp",   "mcp/02-core-concepts/function-calling.md"),          # MCP 核心概念
    ("mcp",   "mcp/02-core-concepts/tools-resources-prompts.md"),   # MCP 核心概念
    ("agent", "agent/agent-5d-v3.html"),                            # Agent 框架
]

def _build_kb_sources() -> list[tuple[str, Path]]:
    sources = []
    for topic, rel in _EXPLICIT_SOURCES:
        p = HANDBOOK_ROOT / rel
        if p.exists():
            sources.append((topic, p))
    # 面试题库（全部 mock-interview/*.md）
    mock_dir = HANDBOOK_ROOT / "rag/code/mock-interview"
    for md in sorted(mock_dir.glob("*.md")):
        sources.append(("interview", md))
    return sources

KB_SOURCES = _build_kb_sources()


# ══════════════════════════════════════════════════════════════
# 文本提取与分块
#
# HTML：HTMLHeaderTextSplitter（LangChain）
#   - 按 h1/h2/h3 切块，标题自动注入每个 chunk（保留层级上下文）
#   - 再用 RecursiveCharacterTextSplitter 二次切割过长块
#   - 显著优于手写 parser：标题与正文有关联，检索相关性更高
#
# Markdown：句子感知分块（沿用 v3 策略）
# ══════════════════════════════════════════════════════════════

_HTML_SPLITTER = HTMLHeaderTextSplitter(
    headers_to_split_on=[("h1", "h1"), ("h2", "h2"), ("h3", "h3")],
)
_CHAR_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
)


def chunk_html(path: Path) -> list[str]:
    """
    HTML → 层级感知分块。
    每个 chunk 前缀注入面包屑标题（h1 > h2 > h3），
    让 embedding 能感知所属章节，提升检索准确率。
    """
    html_text = path.read_text(encoding="utf-8", errors="replace")
    docs = _HTML_SPLITTER.split_text(html_text)

    chunks: list[str] = []
    for doc in docs:
        # 构造标题面包屑
        breadcrumb = " > ".join(
            v for v in [doc.metadata.get("h1"), doc.metadata.get("h2"), doc.metadata.get("h3")]
            if v
        )
        content = doc.page_content.strip()
        if not content:
            continue
        # 注入面包屑（面包屑不重复出现时才加）
        enriched = f"[{breadcrumb}]\n{content}" if breadcrumb else content

        # 二次切割：避免单块过长
        sub = _CHAR_SPLITTER.split_text(enriched)
        chunks.extend(s for s in sub if len(s) > 30)

    return chunks


def chunk_by_sentence(text: str, max_chars: int = 500) -> list[str]:
    """Markdown / 纯文本分块（句子感知）"""
    sentences = re.split(r'(?<=[。！？；\n])', text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s) > 5]
    chunks, current = [], ""
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
# 知识库构建与加载（ChromaDB 持久化）
# ══════════════════════════════════════════════════════════════

def _get_collection():
    """创建或加载 ChromaDB Collection（持久化到磁盘）"""
    client = chromadb.PersistentClient(path=str(DB_PATH))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine", "kb_version": KB_VERSION},
    )


def get_or_build_kb():
    """
    首次运行：扫描 ai-handbook 文件 → 向量化 → upsert 到 ChromaDB
    后续运行：版本匹配时直接加载；版本不匹配时自动重建
    """
    collection = _get_collection()

    if collection.count() > 0:
        stored_ver = collection.metadata.get("kb_version", "")
        if stored_ver == KB_VERSION:
            print(f"  → 知识库已就绪（{collection.count()} 个 chunks），直接加载")
            return collection
        print(f"  → 分块策略已更新（{stored_ver} → {KB_VERSION}），重建知识库...")
        import shutil
        shutil.rmtree(DB_PATH)
        collection = _get_collection()

    print(f"  首次运行，开始构建知识库（共 {len(KB_SOURCES)} 个文件）...\n")
    total_chunks = 0

    for topic, path in KB_SOURCES:
        # HTML：层级感知分块；Markdown/纯文本：句子感知分块
        if path.suffix == ".html":
            chunks = chunk_html(path)
        else:
            raw_text = path.read_text(encoding="utf-8", errors="replace")
            chunks = chunk_by_sentence(raw_text)
        if not chunks:
            print(f"  ⚠  跳过（无有效内容）: {path.name}")
            continue

        source = path.name
        ids, embeddings, documents, metadatas = [], [], [], []

        for i, chunk in enumerate(chunks):
            ids.append(f"{source}_{i:04d}")
            embeddings.append(embed(chunk).tolist())
            documents.append(chunk)
            metadatas.append({
                "topic":   topic,
                "source":  source,
                "chunk_i": i,
            })

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        total_chunks += len(chunks)
        print(f"  ✓ [{topic:10s}] {source:<45} {len(chunks):>4} chunks")

    print(f"\n  → 入库完成，共 {total_chunks} 个 chunks，已持久化到 {DB_PATH.name}/")
    return collection


# ══════════════════════════════════════════════════════════════
# 检索工具（面试官通过 Tool Calling 调用）
# ══════════════════════════════════════════════════════════════

_collection = None   # 由 main() 初始化后注入


def search_kb(query: str, topic: str = "", top_k: int = 5) -> str:
    """
    检索 ai-handbook 知识库，返回格式化文本供 LLM 阅读。

    topic 过滤：rag / mcp / agent / interview（面试题库）/ 空字符串=全库
    """
    where = {"topic": topic} if topic else None

    q_emb = embed(query)
    kwargs: dict = {
        "query_embeddings": [q_emb.tolist()],
        "n_results":        top_k,
        "include":          ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where

    results   = _collection.query(**kwargs)
    docs      = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]

    if not docs:
        return "未在知识库中找到相关内容。"

    parts = []
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances), 1):
        score = round(1 - dist, 3)
        parts.append(
            f"[片段{i}] 来源:{meta['source']}  相关度:{score}\n{doc}"
        )
    return "\n\n".join(parts)


SEARCH_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "search_kb",
        "description": (
            "检索 ai-handbook 知识库，获取 RAG、MCP、Agent 相关知识点。"
            "出题前调用获取题目素材；候选人回答后调用核实答案正确性。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type":        "string",
                    "description": "检索关键词或问题，越具体越准确",
                },
                "topic": {
                    "type":        "string",
                    "enum":        ["rag", "mcp", "agent", "interview", ""],
                    "description": "限定主题范围（空字符串=不限制）",
                    "default":     "",
                },
            },
            "required": ["query"],
        },
    },
}


# ══════════════════════════════════════════════════════════════
# 面试官 System Prompt
# ══════════════════════════════════════════════════════════════

INTERVIEWER_SYSTEM_PROMPT = """\
你是一位资深 AI 工程师面试官，专门考察 RAG、MCP、Agent 方向的技术深度。

【核心原则：绝对禁止泄题】
- 提问时只说问题本身，禁止透露答案、参考答案、评分要点或任何提示
- search_kb 检索结果只能用于内部核实，不能说给候选人听
- 不要以"根据知识库…"开头解释背景，直接问题

【行为规则】
1. 收到 [下一题] 指令：只做一件事——用自然语气把题目问出来，不加任何背景说明
2. 候选人回答后：调用 search_kb 核实 → 给 1-2 句简评（说对了什么/漏了什么）→ 问下一题
3. 评价聚焦缺失点，不要重复候选人已经说对的部分
4. 发现明显错误认知时直接纠正，引用知识库原文

【题目切换】
- [下一题] <题目>：直接提问，首题无需评价上一题
- [继续出题]：调用 search_kb 找一个知识点，自行出一道未问过的题\
"""


# ══════════════════════════════════════════════════════════════
# 面试对话（保留跨轮历史）
#
# 关键设计：messages 作为外部状态跨轮传递。
# v9 的 agentic_loop() 每次重置 messages，不适合多轮面试。
# 这里改为 interview_turn()：每轮追加消息，历史完整保留。
# 面试官因此能记得之前问了什么、候选人怎么回答的。
# ══════════════════════════════════════════════════════════════

def interview_turn(user_content: str, messages: list) -> tuple[str, list]:
    """
    一轮面试：
      1. 将用户内容追加到 messages
      2. 调用 LLM（携带完整历史 + tool 定义）
      3. 若 LLM 调用工具 → 执行 search_kb → 结果追加 → 再调用 LLM
      4. LLM 不再调用工具 → 返回回答，更新 messages

    返回: (面试官回复文本, 更新后的 messages)
    """
    messages.append({"role": "user", "content": user_content})

    for _ in range(6):   # 最多 6 次工具调用（一轮通常 1~2 次即可）
        resp = _client.chat.completions.create(
            model=_cfg["chat_model"],
            messages=messages,
            tools=[SEARCH_TOOL_DEF],
            tool_choice="auto",
            temperature=0.3,
        )
        msg = resp.choices[0].message

        # LLM 不调用工具 → 直接输出回答
        if not msg.tool_calls:
            content = (msg.content or "").strip()
            messages.append({"role": "assistant", "content": content})
            return content, messages

        # LLM 调用工具 → 执行后把结果追加回 messages
        messages.append(msg)
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            # 部分 provider 会二次序列化，args 可能是字符串
            if isinstance(args, str):
                args = json.loads(args)
            result = search_kb(
                query=args.get("query", ""),
                topic=args.get("topic", ""),
            )
            messages.append({
                "role":         "tool",
                "tool_call_id": tc.id,
                "content":      result,
            })

    # 超过最大迭代次数（极少发生）
    fallback = "（工具调用超出上限，请重新提问）"
    messages.append({"role": "assistant", "content": fallback})
    return fallback, messages


# ══════════════════════════════════════════════════════════════
# QA 数据集：加载、评估、记录
# ══════════════════════════════════════════════════════════════

QA_PATH = Path(__file__).parent / "interview_qa.json"
LOG_DIR  = Path(__file__).parent / "logs"


_DIFFICULTY_ORDER = {"easy": 0, "medium": 1, "hard": 2}


def load_qa_dataset(shuffle: bool = True) -> list[dict]:
    """
    加载 interview_qa.json。
    顺序：easy → medium → hard，同难度内随机打乱。
    保证候选人从简单题热身，逐步加深难度。
    """
    if not QA_PATH.exists():
        print(f"  ⚠  未找到 {QA_PATH.name}，面试官将自主从知识库选题")
        return []
    with open(QA_PATH, encoding="utf-8") as f:
        qa_list = json.load(f)

    if shuffle:
        buckets: dict[str, list] = {"easy": [], "medium": [], "hard": []}
        for q in qa_list:
            buckets.setdefault(q["difficulty"], []).append(q)
        for bucket in buckets.values():
            random.shuffle(bucket)
        qa_list = buckets["easy"] + buckets["medium"] + buckets["hard"]

    diff_counts = {d: sum(1 for q in qa_list if q["difficulty"] == d) for d in _DIFFICULTY_ORDER}
    print(f"  → 题库加载完成（{len(qa_list)} 道题）  {diff_counts}")
    return qa_list


def _extract_json(content: str) -> dict | None:
    """从 LLM 输出中提取 JSON（兼容 ``` 包裹和裸 JSON）"""
    content = content.strip()
    if "```" in content:
        parts = content.split("```")
        content = parts[1].lstrip("json").strip() if len(parts) > 1 else content
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return None


def evaluate_answer(qa: dict, user_answer: str) -> dict:
    """
    P0 评估方案：QAG 语义覆盖 + Prometheus CoT 推理 + 错误扣分

    改进点（相比旧方案）：
    1. 先推理再打分（Prometheus 模式）：强制 LLM 逐条分析后再出分，一致性更高
    2. 语义等价判断（QAG 模式）：不要求原文，换个说法说对了也算命中
    3. 错误惩罚：明显错误陈述扣分，不只看覆盖率
    4. analysis 字段：每条要点有独立推理，可追溯

    不加入面试对话历史，不影响面试官上下文。
    """
    n = len(qa["key_points"])
    key_points_numbered = "\n".join(
        f"{i+1}. {kp}" for i, kp in enumerate(qa["key_points"])
    )

    prompt = f"""你是一位严格的面试评估员。

面试题：{qa['question']}
参考答案（满分示例）：{qa['reference_answer']}

评分要点（共 {n} 条，每条 1 分）：
{key_points_numbered}

候选人回答：{user_answer}

【第一步：逐条分析】
对每条评分要点，判断候选人回答是否表达了该要点的核心含义。
判断标准：语义等价即可，不要求原文，换个说法说对了也算命中。
候选人额外说的正确内容不影响得分。

【第二步：错误检测】
候选人是否有与参考答案明显矛盾的错误陈述？（答"不会"或"不知道"不算错误，只是未覆盖）

【第三步：计算得分】
score = 命中要点数 - 错误陈述条数（最低 0，最高 {n}）

以 JSON 格式返回（只返回 JSON，不要其他文字）：
{{
  "analysis": [
    {{"point": "要点原文", "hit": true, "reason": "候选人说了...等价于此"}},
    {{"point": "要点原文", "hit": false, "reason": "未提及"}}
  ],
  "errors": ["错误陈述1（如有）"],
  "key_points_hit":    ["命中的要点原文"],
  "key_points_missed": ["未命中的要点原文"],
  "score":     <命中数 - 错误数，最低0>,
  "max_score": {n},
  "feedback":  "1-2句中文点评：说对了什么、关键遗漏是什么"
}}"""

    resp = _client.chat.completions.create(
        model=_cfg["chat_model"],
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    content = (resp.choices[0].message.content or "").strip()
    result = _extract_json(content)

    if result is None:
        return {
            "score":             0,
            "max_score":         n,
            "analysis":          [],
            "errors":            [],
            "key_points_hit":    [],
            "key_points_missed": qa["key_points"],
            "feedback":          content[:200],
        }

    # 保证 errors 字段存在（旧格式兼容）
    result.setdefault("errors",   [])
    result.setdefault("analysis", [])
    return result


def record_turn(
    turn: int,
    qa: dict,
    user_answer: str,
    eval_result: dict,
    log_path: Path,
) -> dict:
    """将一轮面试记录追加到 JSONL 日志文件"""
    record = {
        "turn":              turn,
        "question_id":       qa["id"],
        "topic":             qa["topic"],
        "difficulty":        qa["difficulty"],
        "question":          qa["question"],
        "user_answer":       user_answer,
        "key_points_hit":    eval_result.get("key_points_hit",    []),
        "key_points_missed": eval_result.get("key_points_missed", []),
        "errors":            eval_result.get("errors",            []),
        "analysis":          eval_result.get("analysis",          []),
        "score":             eval_result.get("score",     0),
        "max_score":         eval_result.get("max_score", len(qa["key_points"])),
        "feedback":          eval_result.get("feedback",  ""),
        "reference_answer":  qa["reference_answer"],
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def print_session_summary(records: list[dict], log_path: Path) -> None:
    """Ctrl+C 时输出本次面试汇总"""
    if not records:
        return
    total_score = sum(r["score"]     for r in records)
    total_max   = sum(r["max_score"] for r in records)
    pct = round(total_score / total_max * 100) if total_max else 0

    print(f"\n{'═'*60}")
    print(f" 本次面试汇总（共 {len(records)} 题）")
    print(f"{'═'*60}")
    print(f" 总得分：{total_score} / {total_max}  （{pct}%）\n")

    for r in records:
        bar = "█" * r["score"] + "░" * (r["max_score"] - r["score"])
        print(f"  [{r['question_id']}] {bar} {r['score']}/{r['max_score']}")
        print(f"    Q: {r['question'][:50]}...")
        if r["key_points_missed"]:
            print(f"    遗漏: {', '.join(r['key_points_missed'][:2])}")

    print(f"\n  详细记录已保存：{log_path}")
    print(f"{'═'*60}\n")


# ══════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════

def main():
    global _collection

    info = model_info()
    print(f"\n{'═'*60}")
    print(f" v11 ／ 模拟面试官（ai-handbook 知识库）")
    print(f" Provider: {info['provider']}  |  Chat: {info['chat_model']}")
    print(f"{'═'*60}")

    # ZhipuAI Tool Calling 兼容性检查
    if _cfg.get("sdk") == "zhipu":
        print("\n  ⚠️  ZhipuAI SDK 的 Tool Calling 接口与 OpenAI 有差异，面试官无法正常工作。")
        print("  请将 .env 中的 PROVIDER 改为 siliconflow 或 openai 后重新运行。\n")
        return

    # ── STEP 1：知识库 ───────────────────────────────────────
    print(f"\n{'═'*60}")
    print(" STEP 1 ／ 知识库")
    print(f"{'═'*60}\n")
    _collection = get_or_build_kb()

    # ── STEP 2：题库 ─────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(" STEP 2 ／ 题库")
    print(f"{'═'*60}\n")
    qa_list = load_qa_dataset(shuffle=True)
    qa_idx  = 0

    # 日志文件
    LOG_DIR.mkdir(exist_ok=True)
    ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"interview_{ts}.jsonl"

    # ── STEP 3：面试 ─────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(" STEP 3 ／ 面试开始（Ctrl+C 退出）")
    print(f"{'═'*60}\n")

    # 初始化对话历史（system prompt 只加一次）
    messages: list  = [{"role": "system", "content": INTERVIEWER_SYSTEM_PROMPT}]
    session_records: list[dict] = []

    # 面试官出第一题
    current_qa = qa_list[qa_idx] if qa_list else None
    if current_qa:
        first_directive = f"[下一题] {current_qa['question']}"
    else:
        first_directive = "请开始面试，从知识库中选一道中等难度的 RAG 基础题。"

    print("  [面试官正在准备第一题...]\n")
    opening, messages = interview_turn(first_directive, messages)
    print(f"面试官：{opening}\n")

    # 交互循环
    _ansi_escape = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')
    _pt_history   = InMemoryHistory() if _PT_AVAILABLE else None

    def read_answer() -> str:
        """
        读取用户输入。
        - 有 prompt_toolkit：支持历史翻阅(↑↓)、行编辑(Ctrl+A/E)、中文输入
        - 无 prompt_toolkit：降级到 input()，过滤 ANSI 转义字符
        """
        if _PT_AVAILABLE:
            return pt_prompt("你：", history=_pt_history).strip()
        raw = input("你：")
        return _ansi_escape.sub('', raw).strip()

    turn = 0
    try:
        while True:
            answer = read_answer()
            if not answer:
                continue
            turn += 1
            print()

            # ── 独立评估（不影响面试对话上下文）────────────
            if current_qa:
                eval_result = evaluate_answer(current_qa, answer)
                record = record_turn(turn, current_qa, answer, eval_result, log_path)
                session_records.append(record)
                score, max_s = record["score"], record["max_score"]
                bar = "█" * score + "░" * (max_s - score)
                print(f"  [得分 {score}/{max_s}  {bar}]\n")

            # ── 切换到下一题 ──────────────────────────────
            qa_idx += 1
            if qa_list and qa_idx < len(qa_list):
                # QA 题库还有题：注入结构化题目
                current_qa  = qa_list[qa_idx]
                next_q_hint = f"\n\n[下一题] {current_qa['question']}"
            else:
                # QA 题库耗尽：转向向量库自由出题
                current_qa  = None
                next_q_hint = "\n\n[继续出题] 请用 search_kb 检索一个知识点，出一道未问过的新题继续面试"
            user_msg = answer + next_q_hint

            # ── 面试官回应（含下一题）────────────────────
            response, messages = interview_turn(user_msg, messages)
            print(f"面试官：{response}\n")

    except KeyboardInterrupt:
        print_session_summary(session_records, log_path)


if __name__ == "__main__":
    main()
