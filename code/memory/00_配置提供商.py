#!/usr/bin/env python3
"""
00_配置提供商.py — 模型提供商抽象层（memory 模块专用精简版）
=====================================
和 rag/ 模块同款接口：embed() / chat() / cosine_sim()。
本模块所有 demo 都通过这三个函数调用模型，换提供商只改 .env，不动业务代码。

快速开始：
  1. cp .env.example .env
  2. 填入 API Key，设置 PROVIDER（siliconflow | zhipu | qwen | openai）
  3. pip install -r requirements.txt
  4. python 00_配置提供商.py   # 测试连通性
"""
from __future__ import annotations

import os
import numpy as np
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai import OpenAI
    from zhipuai import ZhipuAI

try:
    from dotenv import load_dotenv
    _env = Path(__file__).parent / ".env"
    if _env.exists():
        load_dotenv(_env)
except ImportError:
    pass

PROVIDER = os.getenv("PROVIDER", "siliconflow")

# 各提供商默认配置（可被 .env 中的 EMBED_MODEL / CHAT_MODEL 覆盖）
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


def _build_client():
    cfg = get_config()
    api_key = os.getenv(cfg["api_key_env"], "")
    if not api_key:
        raise EnvironmentError(
            f"\n[错误] 未找到 API Key 环境变量: {cfg['api_key_env']}\n"
            f"请编辑 code/memory/.env 配置你的 API Key\n"
        )
    if cfg["sdk"] == "openai_compat":
        from openai import OpenAI
        import httpx
        kwargs: dict = {"api_key": api_key}
        if cfg.get("base_url"):
            kwargs["base_url"] = cfg["base_url"]
        # trust_env=False：跳过系统代理，国内 API 走代理反而握手超时
        kwargs["http_client"] = httpx.Client(trust_env=False, timeout=30.0)
        return OpenAI(**kwargs)
    elif cfg["sdk"] == "zhipu":
        try:
            from zhipuai import ZhipuAI
        except ImportError:
            raise ImportError("未安装 zhipuai，请 pip install zhipuai 或切换 PROVIDER")
        return ZhipuAI(api_key=api_key)
    raise ValueError(f"不支持的 SDK 类型: {cfg['sdk']}")


_client = None


def _get_client():
    global _client
    if _client is None:
        _client = _build_client()
    return _client


def embed(text: str) -> np.ndarray:
    """文本 → 向量。长期记忆靠它把记忆条目向量化后存进向量库。"""
    cfg = get_config()
    resp = _get_client().embeddings.create(input=text, model=cfg["embed_model"])
    return np.array(resp.data[0].embedding)


def chat(prompt: str, temperature: float = 0.0) -> str:
    """调用 LLM。整合阶段用它从对话里抽取「值得记住的事实」。"""
    cfg = get_config()
    temp = temperature if cfg["sdk"] == "openai_compat" else max(temperature, 0.01)
    resp = _get_client().chat.completions.create(
        model=cfg["chat_model"],
        messages=[{"role": "user", "content": prompt}],
        temperature=temp,
    )
    return (resp.choices[0].message.content or "").strip()


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """余弦相似度：长期记忆检索时用它给候选记忆打分。"""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def model_info() -> dict:
    cfg = get_config()
    return {"provider": PROVIDER, "embed_model": cfg["embed_model"], "chat_model": cfg["chat_model"]}


if __name__ == "__main__":
    info = model_info()
    print(f"\n当前提供商: {info['provider']}")
    print(f"Embedding 模型: {info['embed_model']}")
    print(f"Chat 模型:      {info['chat_model']}\n")
    print("测试 embed()...")
    v = embed("AI 记忆分短期和长期")
    print(f"  ✓ 向量维度: {v.shape}")
    print("测试 chat()...")
    print(f"  ✓ 模型回复: {chat('用一句话说明什么是长期记忆')}\n")
