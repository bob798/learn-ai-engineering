#!/usr/bin/env python3
"""
02_长期记忆_向量存储.py — 长期记忆：向量存储 + 检索 + 陈旧更新
=====================================
对应译文章节：《长期记忆》

长期记忆 = 异步、可索引、存在于推理链路之外的持久知识。
生产里它是 Qdrant / Pinecone；这里用纯内存向量库演示同样的三件事：
  1. 写入（向量化后存入，带结构化元数据）
  2. 检索（余弦相似度 Top-K，只在需要时拉进上下文）
  3. 更新（用户信息变了 → 旧 embedding 不更新 = "数据陈旧"，模型会自信地答错）

需要配置 .env 并联网（调用 embedding）。运行：python 02_长期记忆_向量存储.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np

# ── 加载 provider（embed / cosine_sim）──
# 注意：spec 加载的模块要登记进 sys.modules，否则被它定义的 dataclass
# 在解析注解时找不到所属模块（配合 from __future__ import annotations 会报错）。
_p = Path(__file__).with_name("00_配置提供商.py")
_spec = spec_from_file_location("mem_provider", _p)
assert _spec and _spec.loader
_mod = module_from_spec(_spec)
sys.modules["mem_provider"] = _mod
_spec.loader.exec_module(_mod)
embed, cosine_sim, model_info = _mod.embed, _mod.cosine_sim, _mod.model_info


@dataclass
class MemoryItem:
    """每条长期记忆都包成结构化对象，而不是裸文本（对应译文「数据模型与 Schema」）。"""
    content: str
    user_id: str
    kind: str                       # episodic | semantic | procedural
    embedding: np.ndarray = field(repr=False, default=None)
    created_at: str = "2026-06-10T00:00:00Z"


class LongTermMemory:
    """内存版向量库。换成 Qdrant 时，只有 add/search 的内部实现要改。"""

    def __init__(self) -> None:
        self._items: list[MemoryItem] = []

    def add(self, content: str, user_id: str, kind: str = "semantic") -> None:
        item = MemoryItem(content=content, user_id=user_id, kind=kind)
        item.embedding = embed(content)
        self._items.append(item)

    def search(self, query: str, user_id: str, k: int = 3) -> list[tuple[float, MemoryItem]]:
        """检索：先按 user_id 做行级隔离（隐私），再算相似度 Top-K。"""
        qv = embed(query)
        scored = [
            (cosine_sim(qv, it.embedding), it)
            for it in self._items
            if it.user_id == user_id        # ← 用户 A 永远拿不到用户 B 的记忆
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:k]

    def update(self, user_id: str, old_substr: str, new_content: str, kind: str = "semantic") -> bool:
        """
        处理"陈旧"：用户信息变了，必须连 embedding 一起换，否则旧向量仍会被检索到。
        返回是否命中并更新。
        """
        for i, it in enumerate(self._items):
            if it.user_id == user_id and old_substr in it.content:
                self._items[i] = MemoryItem(content=new_content, user_id=user_id, kind=kind)
                self._items[i].embedding = embed(new_content)
                return True
        return False


def demo() -> None:
    info = model_info()
    print(f"提供商: {info['provider']} | embedding: {info['embed_model']}\n")

    ltm = LongTermMemory()
    # 给用户 alice 写入几条记忆（语义 + 情景）
    ltm.add("用户 alice 偏好用 Python，不喜欢 Java。", "alice", kind="semantic")
    ltm.add("用户 alice 的收货地址是上海市浦东新区。", "alice", kind="semantic")
    ltm.add("alice 昨天上传了一份 RAG 评估报告 PDF。", "alice", kind="episodic")
    # 噪声：另一个用户的记忆，用来验证行级隔离
    ltm.add("用户 bob 的收货地址是北京市海淀区。", "bob", kind="semantic")

    print("=" * 64)
    print("【检索】alice 问：我的快递寄到哪？")
    print("=" * 64)
    for score, it in ltm.search("我的收货地址在哪里", user_id="alice", k=3):
        print(f"  {score:.3f}  [{it.kind:9}] {it.content}")
    print("  → 注意 bob 的北京地址不会出现：user_id 行级隔离生效\n")

    print("=" * 64)
    print("【陈旧】alice 搬家到广州，但若不更新 embedding 会怎样？")
    print("=" * 64)
    before = ltm.search("我的收货地址在哪里", user_id="alice", k=1)[0][1]
    print(f"  更新前 Top-1: {before.content}")
    ok = ltm.update("alice", old_substr="收货地址", new_content="用户 alice 的收货地址是广州市天河区。")
    after = ltm.search("我的收货地址在哪里", user_id="alice", k=1)[0][1]
    print(f"  更新成功: {ok}")
    print(f"  更新后 Top-1: {after.content}")
    print("\n要点：检索精度是硬瓶颈，数据陈旧是无声杀手。")
    print("      信息变更必须连 embedding 一起替换，否则模型会自信地答出旧地址。")


if __name__ == "__main__":
    demo()
