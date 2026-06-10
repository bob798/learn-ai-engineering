#!/usr/bin/env python3
"""
03_写穿透_记忆编排.py — 短期 + 长期：写穿透架构 + 异步整合
=====================================
对应译文章节：《AI 智能体如何结合短期记忆与长期记忆？》

把 01（短期）和 02（长期）拼成译文里的「写穿透（write-through）」五步：
  1. Ingest      用户发消息
  2. 短期写入     追加到会话缓冲
  3. Check       查长期索引，把相关记忆注入上下文
  4. Generate    模型回应
  5. Consolidate 异步抽取「值得记住的事实」写入长期索引

这就是 Mem0 这类「记忆层」替你做的编排——这里手搓一遍，看清每一步。

需要 .env + 联网。运行：python 03_写穿透_记忆编排.py
"""
from __future__ import annotations

import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

# 复用 01 / 02 的构件
_here = Path(__file__).parent


def _load(name: str, attrs: list[str]):
    p = _here / name
    mod_name = "mem_" + name[:2]  # 唯一且稳定的模块名
    spec = spec_from_file_location(mod_name, p)
    assert spec and spec.loader
    mod = module_from_spec(spec)
    # 登记进 sys.modules：否则 01/02 里的 dataclass 解析注解时找不到所属模块
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return [getattr(mod, a) for a in attrs]

(chat,) = _load("00_配置提供商.py", ["chat"])
(ShortTermMemory,) = _load("01_短期记忆_会话缓冲.py", ["ShortTermMemory"])
(LongTermMemory,) = _load("02_长期记忆_向量存储.py", ["LongTermMemory"])


# ── Step 5：整合——让 LLM 从对话里抽取值得长期记住的事实 ──
EXTRACT_PROMPT = """你是记忆整合器。从下面这轮对话中，抽取「值得长期记住」的稳定事实
（用户偏好、身份、长期约束等），忽略寒暄和一次性内容。

对话：
{turn}

只输出 JSON 数组，每个元素是一句中文事实；没有则输出 []。
示例：["用户叫 Sarah", "用户偏好深色主题"]"""


def consolidate(turn: str) -> list[str]:
    raw = chat(EXTRACT_PROMPT.format(turn=turn))
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        facts = json.loads(raw)
        return [f for f in facts if isinstance(f, str)] if isinstance(facts, list) else []
    except json.JSONDecodeError:
        return []


class MemoryAgent:
    """把短期 + 长期 + 整合编排到一起的最小智能体。"""

    def __init__(self, user_id: str) -> None:
        self.user_id = user_id
        self.stm = ShortTermMemory(budget=400)
        self.ltm = LongTermMemory()
        self.stm.append("system", "你是用户的私人助理，回答简洁，善用已知的用户记忆。")

    def turn(self, user_msg: str) -> str:
        # 1 + 2. Ingest & 短期写入
        self.stm.append("user", user_msg)

        # 3. Check：检索长期记忆，注入上下文
        hits = self.ltm.search(user_msg, user_id=self.user_id, k=3)
        recalled = [it.content for score, it in hits if score > 0.3]
        memory_block = "（已知用户记忆）\n" + "\n".join(f"- {m}" for m in recalled) if recalled else "（暂无相关长期记忆）"

        # 4. Generate
        prompt = f"{memory_block}\n\n{self.stm.render()}\n\n请只回复 assistant 的下一句话："
        reply = chat(prompt)
        self.stm.append("assistant", reply)

        # 5. Consolidate（生产里这步是异步 worker；这里同步演示）
        facts = consolidate(f"user: {user_msg}\nassistant: {reply}")
        for f in facts:
            self.ltm.add(f, user_id=self.user_id, kind="semantic")

        return reply, recalled, facts


def demo() -> None:
    agent = MemoryAgent(user_id="sarah")

    script = [
        "你好，我叫 Sarah，平时只写 Python，别给我推荐 Java 方案。",
        "帮我推荐一个轻量的 Web 框架。",
    ]

    for i, msg in enumerate(script, 1):
        print("=" * 64)
        print(f"第 {i} 轮 · 用户: {msg}")
        print("=" * 64)
        reply, recalled, facts = agent.turn(msg)
        print(f"  检索到的长期记忆: {recalled or '无'}")
        print(f"  助理回复: {reply}")
        print(f"  本轮整合出的新事实 → 写入长期记忆: {facts}\n")

    print("观察：第 1 轮把「叫 Sarah / 只写 Python / 别推 Java」整合进长期记忆；")
    print("      第 2 轮推荐框架时，这些事实被检索回来注入上下文——跨轮记忆生效。")
    print("      Mem0 / LangMem 等记忆层，本质就是把这套写穿透编排做成一个 API。")


if __name__ == "__main__":
    demo()
