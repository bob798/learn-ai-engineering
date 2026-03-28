#!/usr/bin/env python3
"""
00_配置提供商_先改这个.py — 模型提供商抽象层
=====================================
支持国内主流模型，统一 embed() 和 chat() 接口。

⚠️ 配置已迁移到 .env 文件，请先编辑 .env 配置你的 API Key 和选择提供商。
⚠️ 使用前请运行: pip install python-dotenv

配置方式：
  1. 复制 .env.example 为 .env
  2. 填入你的 API Key（对应你选择的提供商）
  3. 设置 PROVIDER（siliconflow | zhipu | qwen | openai）
  4. 可选：自定义 embed_model 和 chat_model

国内推荐：
  硅基流动（SiliconFlow）→ 免费额度大，BGE-M3 embedding 质量好
  智谱 AI（ZhipuAI）     → 国内最稳，embedding-3 支持中文
  通义千问（Qwen）        → 阿里系，DashScope API

OpenAI 兼容（只换 base_url，代码几乎不变）：
  DeepSeek               → 价格极低，推理能力强，无 embedding（用 SF 补）
  硅基流动               → 同时提供 embedding + chat，一个 key 搞定
  月之暗面（Moonshot）   → Kimi 背后的模型，中文理解好
"""

from __future__ import annotations

import os
import numpy as np
from pathlib import Path
from typing import TYPE_CHECKING

# 类型注解专用导入（运行时不会实际导入这些包）
if TYPE_CHECKING:
    from openai import OpenAI
    from zhipuai import ZhipuAI

# ──────────────────────────────────────────────────
# 加载 .env 文件（优先从当前目录）
# ──────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # 如果没有 python-dotenv，继续使用系统环境变量

# ═══════════════════════════════════════════════════
# ★ 配置已移至 .env 文件 ★
# 请编辑 rag/demo/code/.env 文件修改以下配置：
#   - PROVIDER: "siliconflow" | "zhipu" | "qwen" | "openai"
#   - 对应的 API Key 环境变量
#   - 可选：EMBED_MODEL 和 CHAT_MODEL 自定义模型
# ═══════════════════════════════════════════════════
PROVIDER = os.getenv("PROVIDER", "siliconflow")


# ──────────────────────────────────────────────────
# 各提供商默认配置（可被 .env 中的变量覆盖）
# ──────────────────────────────────────────────────

PROVIDER_CONFIG = {
    "siliconflow": {
        # 硅基流动：OpenAI 兼容协议，一个 key 同时支持 embedding + chat
        # 注册：https://siliconflow.cn  → 有免费额度
        # 环境变量：SILICONFLOW_API_KEY（已在 .env 中配置）
        "base_url":     "https://api.siliconflow.cn/v1",
        "api_key_env":  "SILICONFLOW_API_KEY",
        "embed_model":  "BAAI/bge-m3",           # 中英文最佳，1024维
        "chat_model":   "Qwen/Qwen2.5-7B-Instruct",
        "sdk":          "openai_compat",
    },
    "zhipu": {
        # 智谱 AI：需要安装 pip install zhipuai
        # 注册：https://open.bigmodel.cn  → 有免费额度
        # 环境变量：ZHIPU_API_KEY（已在 .env 中配置）
        "api_key_env":  "ZHIPU_API_KEY",
        "embed_model":  "embedding-3",            # 2048维，中文强
        "chat_model":   "glm-4-flash",            # 免费额度充足
        "sdk":          "zhipu",
    },
    "qwen": {
        # 通义千问：OpenAI 兼容协议，需要申请 DashScope API
        # 注册：https://dashscope.aliyun.com
        # 环境变量：DASHSCOPE_API_KEY（已在 .env 中配置）
        "base_url":     "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key_env":  "DASHSCOPE_API_KEY",
        "embed_model":  "text-embedding-v3",      # 1024/2048维可选
        "chat_model":   "qwen-plus",
        "sdk":          "openai_compat",
    },
    "openai": {
        # 保留 OpenAI 作为对照
        # 环境变量：OPENAI_API_KEY（已在 .env 中配置）
        "base_url":     None,                     # 官方默认
        "api_key_env":  "OPENAI_API_KEY",
        "embed_model":  "text-embedding-3-small",
        "chat_model":   "gpt-4o-mini",
        "sdk":          "openai_compat",
    },
}

# ──────────────────────────────────────────────────
# 获取当前配置（支持 .env 覆盖模型名称）
# ──────────────────────────────────────────────────

def get_config() -> dict:
    """获取当前提供商配置，优先使用 .env 中的自定义模型名称"""
    cfg = PROVIDER_CONFIG[PROVIDER].copy()
    
    # 从 .env 覆盖模型名称（如果已设置）
    embed_model_env = os.getenv("EMBED_MODEL")
    chat_model_env = os.getenv("CHAT_MODEL")
    
    if embed_model_env:
        cfg["embed_model"] = embed_model_env
    if chat_model_env:
        cfg["chat_model"] = chat_model_env
    
    return cfg


# ──────────────────────────────────────────────────
# 客户端初始化（根据 PROVIDER 自动选择）
# ──────────────────────────────────────────────────
def _build_client() -> "OpenAI":
    """构建并返回 API 客户端"""
    cfg = get_config()
    api_key = os.getenv(cfg["api_key_env"], "")
    if not api_key:
        raise EnvironmentError(
            f"\n[错误] 未找到 API Key 环境变量: {cfg['api_key_env']}\n"
            f"请编辑 rag/demo/code/.env 文件配置你的 API Key\n"
            f"申请地址见 .env 文件中 {PROVIDER} 的注释"
        )
    if cfg["sdk"] == "openai_compat":
        from openai import OpenAI
        import httpx
        kwargs: dict = {"api_key": api_key}
        if cfg.get("base_url"):
            kwargs["base_url"] = cfg["base_url"]
        # trust_env=False：跳过系统代理（如 Clash/V2Ray）
        # 国内 API（siliconflow/qwen）不需要代理，走代理反而 TLS 握手超时
        # 如果你使用 OpenAI 官方且需要代理，请改为 trust_env=True
        kwargs["http_client"] = httpx.Client(trust_env=False, timeout=30.0)
        return OpenAI(**kwargs)
    elif cfg["sdk"] == "zhipu":
        try:
            from zhipuai import ZhipuAI
        except ImportError:
            raise ImportError(
                f"\n[错误] 未安装 zhipuai 包\n"
                f"请运行: pip install zhipuai\n"
                f"或者切换到其他提供商（siliconflow | qwen | openai）\n"
                f"修改 rag/demo/code/.env 文件中的 PROVIDER 配置"
            )
        return ZhipuAI(api_key=api_key)
    else:
        raise ValueError(f"不支持的 SDK 类型: {cfg['sdk']}")

_client: "OpenAI | ZhipuAI" = None  # type: ignore[assignment]

def _get_client() -> "OpenAI | ZhipuAI":
    """获取或创建 API 客户端（单例模式）"""
    global _client
    if _client is None:
        _client = _build_client()
    return _client


# ──────────────────────────────────────────────────
# 统一接口（所有版本调用这两个函数）
# ──────────────────────────────────────────────────

def embed(text: str) -> np.ndarray:
    """
    把文本转成向量。
    
    为什么要转向量、为什么用余弦相似度 → 见《概念手册》
    这里只关注"怎么调用"。
    """
    cfg = get_config()
    client = _get_client()

    if cfg["sdk"] == "openai_compat":
        resp = client.embeddings.create(
            input=text,
            model=cfg["embed_model"],
        )
        return np.array(resp.data[0].embedding)

    elif cfg["sdk"] == "zhipu":
        resp = client.embeddings.create(
            input=text,
            model=cfg["embed_model"],
        )
        return np.array(resp.data[0].embedding)
    else:
        raise ValueError(f"不支持的 SDK 类型: {cfg['sdk']}")


def chat(prompt: str, temperature: float = 0.0) -> str:
    """
    调用 LLM 生成回答。
    temperature=0 保证输出稳定，便于对比实验。
    """
    cfg = get_config()
    client = _get_client()

    if cfg["sdk"] == "openai_compat":
        resp = client.chat.completions.create(
            model=cfg["chat_model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()

    elif cfg["sdk"] == "zhipu":
        resp = client.chat.completions.create(
            model=cfg["chat_model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=max(temperature, 0.01),   # 智谱不支持 temperature=0
        )
        return (resp.choices[0].message.content or "").strip()
    else:
        raise ValueError(f"不支持的 SDK 类型: {cfg['sdk']}")


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """
    余弦相似度 = (a · b) / (|a| × |b|)
    为什么用余弦而不是欧氏距离 → 见《概念手册》
    """
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def model_info() -> dict:
    """返回当前使用的模型信息，方便打印确认"""
    cfg = get_config()
    return {
        "provider":    PROVIDER,
        "embed_model": cfg["embed_model"],
        "chat_model":  cfg["chat_model"],
    }


# ──────────────────────────────────────────────────
# 快速验证（直接运行此文件测试连通性）
# ──────────────────────────────────────────────────

if __name__ == "__main__":
    info = model_info()
    print(f"\n当前提供商: {info['provider']}")
    print(f"Embedding 模型: {info['embed_model']}")
    print(f"Chat 模型:      {info['chat_model']}\n")

    print("测试 embed()...")
    vec = embed("RAG 是一种检索增强生成技术")
    print(f"  向量维度: {vec.shape[0]}")
    print(f"  前4维:    {vec[:4].round(5)}")
    print(f"  向量模长: {np.linalg.norm(vec):.4f}")

    print("\n测试 chat()...")
    ans = chat("用一句话解释什么是 RAG")
    print(f"  回答: {ans}")

    print("\n测试 cosine_sim()...")
    v1 = embed("猫是一种动物")
    v2 = embed("狗是一种宠物")
    v3 = embed("量子力学是物理学的分支")
    print(f"  相似句对 ('猫是动物' vs '狗是宠物'):      {cosine_sim(v1, v2):.4f}")
    print(f"  不相关对 ('猫是动物' vs '量子力学'):       {cosine_sim(v1, v3):.4f}")
    print("\n✅ 连通性验证通过，可以运行 01_v1_最小RAG循环.py")
