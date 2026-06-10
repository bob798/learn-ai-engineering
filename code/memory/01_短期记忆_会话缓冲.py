#!/usr/bin/env python3
"""
01_短期记忆_会话缓冲.py — 短期记忆：滑动窗口 + token 预算 + 固定系统提示
=====================================
对应译文章节：《短期记忆》

短期记忆 = LLM 的"工作记忆"，活在推理的热路径上：
  - 受上下文窗口约束（这里用 token 预算模拟）
  - 同步、易失：会话一结束就没了，除非显式保存
  - 每一个 token 每次推理都要花钱

本文件不调用任何 API，纯逻辑演示，直接 python 运行即可。

╔══════════════════════════════════════════════════════════════╗
║  本集关键演示                                                ║
║  1. 朴素截断会删掉系统提示 → "灾难性遗忘"（反例）            ║
║  2. 固定（pin）系统提示 + 滑动窗口 → 正确做法                ║
╚══════════════════════════════════════════════════════════════╝
"""
from __future__ import annotations

from dataclasses import dataclass, field


# ── token 估算：演示用，约定「1 个汉字/单词 ≈ 1 token」──
def count_tokens(text: str) -> int:
    return max(1, len(text))


@dataclass
class Message:
    role: str          # system | user | assistant
    content: str

    @property
    def tokens(self) -> int:
        return count_tokens(self.content)


@dataclass
class ShortTermMemory:
    """会话缓冲区：滑动窗口 + token 预算，系统提示永远固定保留。"""
    budget: int = 60                       # 上下文窗口预算（token）
    _messages: list[Message] = field(default_factory=list)

    def append(self, role: str, content: str) -> None:
        self._messages.append(Message(role, content))
        self._evict()

    def _evict(self) -> None:
        """淘汰策略：固定 system 提示，其余按 FIFO（最旧先出）裁剪到预算内。"""
        pinned = [m for m in self._messages if m.role == "system"]
        rest = [m for m in self._messages if m.role != "system"]

        used = sum(m.tokens for m in pinned)
        kept: list[Message] = []
        # 从最新往回保留，直到预算用完
        for m in reversed(rest):
            if used + m.tokens > self.budget:
                break
            kept.append(m)
            used += m.tokens
        kept.reverse()
        self._messages = pinned + kept

    @property
    def used_tokens(self) -> int:
        return sum(m.tokens for m in self._messages)

    def render(self) -> str:
        return "\n".join(f"  [{m.role:9}] {m.content}" for m in self._messages)


# ── 反例：朴素截断（直接砍开头）会把系统提示一起删掉 ──
def naive_truncate(messages: list[Message], budget: int) -> list[Message]:
    """常见错误：从头部砍。结果系统提示先被删，模型瞬间"失忆"。"""
    out = list(messages)
    while sum(m.tokens for m in out) > budget and out:
        out.pop(0)   # ← 砍最旧的，但系统提示恰恰排在最前
    return out


def demo() -> None:
    system = Message("system", "你是一个严谨的电商客服，回答必须基于订单事实，禁止编造。")

    convo = [
        system,
        Message("user", "我上周买的耳机还没到"),
        Message("assistant", "已帮您查询，物流显示在途，预计明天送达。"),
        Message("user", "帮我看下能不能改地址"),
        Message("assistant", "在途订单暂不支持改地址，可签收后申请换货。"),
        Message("user", "那帮我催一下快递"),
    ]

    print("=" * 64)
    print("【反例】朴素截断 —— 从头部砍，系统提示被删 → 灾难性遗忘")
    print("=" * 64)
    bad = naive_truncate(convo, budget=60)
    has_system = any(m.role == "system" for m in bad)
    for m in bad:
        print(f"  [{m.role:9}] {m.content}")
    print(f"\n  系统提示是否还在: {'是' if has_system else '❌ 否——模型已忘记自己是客服、忘记禁止编造'}\n")

    print("=" * 64)
    print("【正确】固定系统提示 + 滑动窗口")
    print("=" * 64)
    stm = ShortTermMemory(budget=60)
    for m in convo:
        stm.append(m.role, m.content)
    print(stm.render())
    has_system = any(m.role == "system" for m in stm._messages)
    print(f"\n  系统提示是否还在: {'✓ 是——人设与红线始终在场' if has_system else '否'}")
    print(f"  当前占用: {stm.used_tokens}/{stm.budget} token\n")

    print("要点：短期记忆是有限资源。淘汰时务必 pin 住系统提示，")
    print("      否则一旦对话变长，模型最先忘掉的就是它的身份和规则。")


if __name__ == "__main__":
    demo()
