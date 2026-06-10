#!/usr/bin/env python3
"""
04_用mem0库.py — A 路线：直接用 mem0 开源库（Apache-2.0）
=====================================
前面 01~03 是手搓「写穿透」编排，为的是看清原理。
生产里你通常不重复造轮子，直接用 mem0 这类记忆层——它把
  短期会话 + 长期向量库 + 图记忆 + 事实抽取/整合
封装成 add() / search() 两个调用。

mem0 是 Apache-2.0 开源，可本地自建、可商用（见项目 README 的版权说明）。

依赖：pip install mem0ai
注意：mem0 默认走 OpenAI。国内可在 config 里把 llm / embedder 指到
      OpenAI 兼容网关（如硅基流动），下面给了示例。

本文件默认「干跑」：未安装 mem0 时只打印说明，不报错。
"""
from __future__ import annotations

import os


def build_local_mem0():
    """
    构造一个指向 OpenAI 兼容网关的 mem0 实例（以硅基流动为例）。
    真正联网需要：pip install mem0ai + .env 里配好 SILICONFLOW_API_KEY。
    """
    from mem0 import Memory  # 延迟导入：未安装时不影响文件被读

    api_key = os.getenv("SILICONFLOW_API_KEY", "")
    base_url = "https://api.siliconflow.cn/v1"

    config = {
        "llm": {
            "provider": "openai",
            "config": {
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "api_key": api_key,
                "openai_base_url": base_url,
            },
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": "BAAI/bge-m3",
                "api_key": api_key,
                "openai_base_url": base_url,
            },
        },
        # 向量库默认用内存版 / Chroma；生产可换 Qdrant：
        # "vector_store": {"provider": "qdrant", "config": {"host": "localhost", "port": 6333}},
    }
    return Memory.from_config(config)


def demo() -> None:
    try:
        import mem0  # noqa: F401
    except ImportError:
        print("未安装 mem0ai。这是「A 路线」的对照示例，安装后即可运行：\n")
        print("    pip install mem0ai\n")
        print("典型用法（和我们 03 手搓的写穿透等价，但只要两行）：\n")
        print('    m = build_local_mem0()')
        print('    m.add("我叫 Sarah，只写 Python，别推 Java", user_id="sarah")')
        print('    print(m.search("推荐个 web 框架", user_id="sarah"))')
        print("\n对比 03_写穿透_记忆编排.py：mem0 把『短期写入→检索→整合』全包了。")
        return

    m = build_local_mem0()
    m.add("我叫 Sarah，平时只写 Python，别给我推荐 Java。", user_id="sarah")
    hits = m.search("帮我推荐一个轻量 web 框架", user_id="sarah")
    print("mem0 检索结果:")
    print(hits)


if __name__ == "__main__":
    demo()
