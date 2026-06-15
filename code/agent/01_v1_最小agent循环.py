#!/usr/bin/env python3
"""
01_v1_最小agent循环.py — 从零手写最小 Agent 循环（无任何框架）
=====================================
对标 rag/code/01_v1_最小RAG循环.py。

  快速启动：
    pip install openai python-dotenv     # 真实运行需要
    cp .env.example .env                  # 填 API Key、选 PROVIDER
    python 01_v1_最小agent循环.py

  离线验证循环逻辑（不需要 API Key、不需要装 openai）：
    python 01_v1_最小agent循环.py --selftest
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║  📺 讲师注释                                                  ║
# ║  配套讲解：content/02-agent/agent-from-scratch/              ║
# ║            02_代码讲解_V1V2.md（V1 部分）                    ║
# ║  核心代码：run_agent()——那个 while 循环就是 Agent 的全部    ║
# ║  保留的困惑：「Agent 不是模型能力，是外面那个循环」          ║
# ║  本集关键对比：无循环（模型心算，常算错）vs 有循环（调       ║
# ║               计算器，精确）——和 RAG 的「有无 RAG」对称     ║
# ╚══════════════════════════════════════════════════════════════╝

from __future__ import annotations

import re
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Callable

# ── 加载同源 provider（文件名以数字开头，import 不了，用 importlib）──
def _load_provider():
    p = Path(__file__).with_name("00_配置提供商.py")
    spec = spec_from_file_location("agent_provider", p)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载提供商配置: {p}")
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ══════════════════════════════════════════════════════════════
# 1. 工具（V1 只给一个工具——最小化）
# ══════════════════════════════════════════════════════════════
# 为什么是计算器？因为它「确定性、可验证、模型常算错」——
# 最能干净地演示「循环带来了模型本身没有的能力」。

def tool_calculator(expr: str) -> str:
    """安全的四则运算。只允许数字和 + - * / ( ) . 和空格。"""
    expr = expr.strip()
    if not re.fullmatch(r"[\d\.\+\-\*\/\(\)\s]+", expr):
        return f"错误：表达式含非法字符，只支持数字和 + - * / ( )，收到的是 {expr!r}"
    try:
        # 受限 eval：禁掉所有 builtins，只能做算术
        result = eval(expr, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:  # noqa: BLE001
        return f"错误：无法计算 {expr!r} —— {e}"


TOOLS = {"calculator": tool_calculator}


# ══════════════════════════════════════════════════════════════
# 2. System Prompt —— 用一个极简文本协议教模型怎么「调工具」
# ══════════════════════════════════════════════════════════════
# V1 故意不用原生 function calling（那是 V3），也不用 ReAct 的
# Thought/Action/Observation 三段式（那是 V2）。
# V1 只教模型回两种格式之一，让「循环 + 解析」这件事赤裸地暴露出来。
SYSTEM_PROMPT = """你是一个能调用工具的助手。你有一个工具：

  calculator(<算术表达式>)  —— 精确计算四则运算，例如 calculator(123*456)

每一步，你只能回复以下两种格式之一，且只回一行：

  ACTION: calculator(<表达式>)     # 当你需要算数时
  ANSWER: <最终答案>               # 当你已经能给出最终答案时

规则：
- 不要自己心算复杂算术，一律交给 calculator。
- 我会把工具结果作为「观察」发回给你，你再决定下一步。
- 给出 ANSWER 后对话结束。"""


# ══════════════════════════════════════════════════════════════
# 3. 解析模型输出
# ══════════════════════════════════════════════════════════════
def parse_step(text: str) -> tuple[str, str]:
    """把模型这一步的输出解析成 (类型, 内容)。类型 ∈ {'action','answer','unknown'}。"""
    for line in text.splitlines():
        line = line.strip()
        m = re.match(r"ACTION:\s*calculator\((.*)\)\s*$", line, re.IGNORECASE)
        if m:
            return "action", m.group(1)
        m = re.match(r"ANSWER:\s*(.*)$", line, re.IGNORECASE)
        if m:
            return "answer", m.group(1).strip()
    return "unknown", text.strip()


# ══════════════════════════════════════════════════════════════
# 4. Agent 循环 —— 这 ~20 行就是 Agent 的全部
# ══════════════════════════════════════════════════════════════
def run_agent(question: str, llm: Callable[[list], str], max_steps: int = 6,
              verbose: bool = True) -> str:
    """
    感知 → 决策 → 行动 → 把观察喂回去 → 再决策……直到 ANSWER 或步数耗尽。

    llm: 一个 (messages) -> str 的函数。真实运行传 provider.chat；
         --selftest 传一个脚本化的假模型。把 LLM 做成可注入的，
         循环逻辑才能离线验证——这本身就是 Agent 工程的好习惯。
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    for step in range(1, max_steps + 1):
        out = llm(messages)
        kind, content = parse_step(out)
        if verbose:
            print(f"  ┌─ Step {step} ─ 模型决策：{out}")

        if kind == "answer":
            if verbose:
                print(f"  └─ ✅ 终止：ANSWER\n")
            return content

        if kind == "action":
            observation = TOOLS["calculator"](content)
            if verbose:
                print(f"  │   ⚙️  调用 calculator({content})  →  {observation}")
                print(f"  └─ 把观察回灌，继续循环\n")
            # 把「模型的决策」和「工具观察」都追加进历史——这就是「记忆」的雏形
            messages.append({"role": "assistant", "content": out})
            messages.append({"role": "user", "content": f"观察结果: {observation}"})
            continue

        # unknown：模型没按协议回，给一次纠正机会（V7 会专门处理错误恢复）
        if verbose:
            print(f"  └─ ⚠️  未识别格式，要求模型重答\n")
        messages.append({"role": "assistant", "content": out})
        messages.append({"role": "user", "content": "请严格用 ACTION: 或 ANSWER: 开头重答一行。"})

    return "（已达最大步数仍未给出答案——这正是 V7『反思与自纠』『死循环防护』要解决的失败模式）"


# ══════════════════════════════════════════════════════════════
# 5. 对比实验：无循环 vs 有循环（对标 RAG 的「无 RAG vs 有 RAG」）
# ══════════════════════════════════════════════════════════════
QUESTION = "一个仓库有 1234 个文件，删掉了 567 个，又新增了原来数量 3 倍的文件，现在共有多少个文件？"
# 正确答案：(1234-567) + 1234*3 = 667 + 3702 = 4369


def demo_real():
    provider = _load_provider()
    info = provider.model_info()
    print(f"\n{'═'*64}")
    print(f" v1 ／ 最小 Agent 循环")
    print(f" Provider: {info['provider']}  |  Chat: {info['chat_model']}")
    print(f"{'═'*64}")
    print(f"\n  问题：{QUESTION}")
    print(f"  正确答案：4369   （(1234-567) + 1234*3）")

    print(f"\n{'─'*64}\n  ❌ 无 Agent 循环（直接问 LLM，模型自己心算）：")
    raw = provider.chat_text(QUESTION)
    for ln in raw.splitlines():
        print(f"  │ {ln}")

    print(f"\n{'─'*64}\n  ✅ 有 Agent 循环（模型自己决定调 calculator）：")
    final = run_agent(QUESTION, llm=provider.chat, verbose=True)
    print(f"  → Agent 最终答案：{final}")

    print(f"\n{'═'*64}\n V1 核心概念回顾\n{'═'*64}")
    print("""
  ✓ Agent 循环：感知→决策→行动→观察回灌→再决策（run_agent 那 ~20 行）
  ✓ 工具：模型不具备的确定性能力（这里是精确计算）
  ✓ 文本协议：V1 用 ACTION:/ANSWER: 两种格式，最小化暴露「解析」这一步
  ✓ messages 历史：把决策和观察追加进去 = 记忆的雏形（V4 会专门做）

  💡 保留的困惑：
     「我以为 Agent 是更聪明的模型，其实 Agent = 同一个模型 + 外面那个 while 循环。
       聪明的不是模型，是循环让它能『查』和『算』。」

  → 配套讲解：content/02-agent/agent-from-scratch/02_代码讲解_V1V2.md
  → 下一步：02_v2_react模式.py（把决策拆成 Thought/Action/Observation）
    """)


# ══════════════════════════════════════════════════════════════
# 6. 离线自测：用脚本化假模型验证「循环 + 解析 + 工具」三件事正确
# ══════════════════════════════════════════════════════════════
def selftest():
    print("【selftest】用假 LLM 离线验证循环逻辑（不需 API Key / openai）\n")

    # 1) 解析器
    assert parse_step("ACTION: calculator(1+2)") == ("action", "1+2")
    assert parse_step("ANSWER: 42") == ("answer", "42")
    assert parse_step("我不知道")[0] == "unknown"
    print("  ✓ parse_step 三种格式解析正确")

    # 2) 工具：正确 + 安全（拒绝非法字符 / 禁用 builtins）
    assert tool_calculator("(1234-567) + 1234*3") == "4369"
    assert "错误" in tool_calculator("__import__('os').system('ls')")
    print("  ✓ calculator 计算正确且拒绝注入")

    # 3) 完整循环：脚本化假模型，先要求算两步、再给答案
    script = [
        "ACTION: calculator(1234-567)",        # → 667
        "ACTION: calculator(667 + 1234*3)",     # → 4369
        "ANSWER: 现在共有 4369 个文件",
    ]
    calls = {"i": 0}

    def fake_llm(messages):
        # 真实 agent 会「看」messages 决策；这里假模型按脚本走，只为验证循环
        i = calls["i"]
        calls["i"] += 1
        return script[i]

    result = run_agent(QUESTION, llm=fake_llm, verbose=True)
    assert "4369" in result, f"循环结果不含正确答案: {result}"
    assert calls["i"] == 3, f"应恰好 3 次模型调用，实际 {calls['i']}"
    print("  ✓ run_agent 完整跑通：2 次工具调用 + 1 次 ANSWER，得到 4369")

    # 4) 死循环保护：假模型永远只回 ACTION，不给 ANSWER
    def stuck_llm(messages):
        return "ACTION: calculator(1+1)"
    capped = run_agent(QUESTION, llm=stuck_llm, max_steps=3, verbose=False)
    assert "最大步数" in capped
    print("  ✓ max_steps 兜底：模型死循环时安全退出（V7/V10 会强化）")

    print("\n✅ selftest 全部通过——V1 循环逻辑成立。")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        demo_real()
