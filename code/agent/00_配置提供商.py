#!/usr/bin/env python3
"""
00_配置提供商.py — Agent 系列模型提供商抽象层
=====================================
与 rag/code/00_配置提供商_先改这个.py 同源，但面向 Agent：
  - chat(messages)   接收完整对话历史（agent 循环需要多轮消息）
  - chat_text(prompt) 单轮便捷封装
  - embed(text)      向量化（V4/V5 记忆版本才会用到，这里先备好）

⚠️ 所有第三方依赖（openai / numpy）都「懒加载」——
   只有真正调用到 chat()/embed() 时才 import，
   因此没装依赖也能 `import` 本模块跑 01 的 --selftest。

配置方式：
  1. cp .env.example .env
  2. 填入 API Key，设置 PROVIDER（siliconflow | zhipu | qwen | openai）
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai import OpenAI
    from zhipuai import ZhipuAI

# ──────────────────────────────────────────────────
# 加载 .env（优先当前目录）
# ──────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    _env = Path(__file__).parent / ".env"
    if _env.exists():
        load_dotenv(_env)
except ImportError:
    pass

PROVIDER = os.getenv("PROVIDER", "siliconflow")

# 各提供商默认配置（与 RAG 保持一致，可被 .env 覆盖）
PROVIDER_CONFIG = {
    "siliconflow": {
        "base_url":    "https://api.siliconflow.cn/v1",
        "api_key_env": "SILICONFLOW_API_KEY",
        "embed_model": "BAAI/bge-m3",
        "chat_model":  "Qwen/Qwen2.5-7B-Instruct",
        "sdk":         "openai_compat",
    },
    "zhipu": {
        "api_key_env": "ZHIPU_API_KEY",
        "embed_model": "embedding-3",
        "chat_model":  "glm-4-flash",
        "sdk":         "zhipu",
    },
    "qwen": {
        "base_url":    "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key_env": "DASHSCOPE_API_KEY",
        "embed_model": "text-embedding-v3",
        "chat_model":  "qwen-plus",
        "sdk":         "openai_compat",
    },
    "openai": {
        "base_url":    None,
        "api_key_env": "OPENAI_API_KEY",
        "embed_model": "text-embedding-3-small",
        "chat_model":  "gpt-4o-mini",
        "sdk":         "openai_compat",
    },
}


def get_config() -> dict:
    cfg = PROVIDER_CONFIG[PROVIDER].copy()
    if os.getenv("EMBED_MODEL"):
        cfg["embed_model"] = os.environ["EMBED_MODEL"]
    if os.getenv("CHAT_MODEL"):
        cfg["chat_model"] = os.environ["CHAT_MODEL"]
    return cfg


_client = None  # 单例


def _get_client():
    global _client
    if _client is not None:
        return _client
    cfg = get_config()
    api_key = os.getenv(cfg["api_key_env"], "")
    if not api_key:
        raise EnvironmentError(
            f"\n[错误] 未找到 API Key 环境变量: {cfg['api_key_env']}\n"
            f"请编辑 code/agent/.env 配置 API Key，或用 --selftest 离线跑循环逻辑。"
        )
    if cfg["sdk"] == "openai_compat":
        from openai import OpenAI
        import httpx
        kwargs = {"api_key": api_key}
        if cfg.get("base_url"):
            kwargs["base_url"] = cfg["base_url"]
        kwargs["http_client"] = httpx.Client(trust_env=False, timeout=30.0)
        _client = OpenAI(**kwargs)
    elif cfg["sdk"] == "zhipu":
        from zhipuai import ZhipuAI
        _client = ZhipuAI(api_key=api_key)
    else:
        raise ValueError(f"不支持的 SDK: {cfg['sdk']}")
    return _client


# ──────────────────────────────────────────────────
# 统一接口
# ──────────────────────────────────────────────────

def chat(messages: list[dict], temperature: float = 0.0) -> str:
    """
    Agent 循环的核心调用：传入完整 messages 历史，返回模型这一步的输出文本。

    messages 形如：
      [{"role": "system", "content": ...},
       {"role": "user", "content": ...},
       {"role": "assistant", "content": ...},   # 模型上一步的决策
       {"role": "user", "content": "观察结果: ..."}]  # 工具回灌
    """
    cfg = get_config()
    client = _get_client()
    temp = temperature if cfg["sdk"] != "zhipu" else max(temperature, 0.01)
    resp = client.chat.completions.create(
        model=cfg["chat_model"],
        messages=messages,
        temperature=temp,
    )
    return (resp.choices[0].message.content or "").strip()


def chat_text(prompt: str, temperature: float = 0.0) -> str:
    """单轮便捷封装。"""
    return chat([{"role": "user", "content": prompt}], temperature)


def embed(text: str):
    """文本向量化（V4/V5 记忆版本使用）。numpy 懒加载。"""
    import numpy as np
    cfg = get_config()
    client = _get_client()
    resp = client.embeddings.create(input=text, model=cfg["embed_model"])
    return np.array(resp.data[0].embedding)


def model_info() -> dict:
    cfg = get_config()
    return {"provider": PROVIDER, "chat_model": cfg["chat_model"], "embed_model": cfg["embed_model"]}


if __name__ == "__main__":
    info = model_info()
    print(f"\n当前提供商: {info['provider']}  |  Chat: {info['chat_model']}")
    print("测试 chat()...")
    print("  回答:", chat_text("用一句话说清楚：Agent 和直接调用 LLM 的区别是什么？"))
    print("\n✅ 连通性验证通过，可以运行 01_v1_最小agent循环.py")
