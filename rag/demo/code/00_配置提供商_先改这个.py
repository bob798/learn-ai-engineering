#!/usr/bin/env python3
"""
00_配置提供商_先改这个.py — 模型提供商抽象层
=====================================
支持国内主流模型，统一 embed() 和 chat() 接口。
V1 / V2 / V3.5 等所有版本只需修改这一个文件中的 PROVIDER。

国内推荐：
  硅基流动（SiliconFlow）→ 免费额度大，BGE-M3 embedding 质量好
  智谱 AI（ZhipuAI）     → 国内最稳，embedding-3 支持中文
  通义千问（Qwen）        → 阿里系，DashScope API

OpenAI 兼容（只换 base_url，代码几乎不变）：
  DeepSeek               → 价格极低，推理能力强，无 embedding（用 SF 补）
  硅基流动               → 同时提供 embedding + chat，一个 key 搞定
  月之暗面（Moonshot）   → Kimi 背后的模型，中文理解好
"""

import os
import numpy as np

# ═══════════════════════════════════════════════════
# ★ 只需修改这一行，切换提供商 ★
# 选项: "siliconflow" | "zhipu" | "qwen" | "openai"
# ═══════════════════════════════════════════════════
PROVIDER = "siliconflow"


# ──────────────────────────────────────────────────
# 各提供商配置
# ──────────────────────────────────────────────────

PROVIDER_CONFIG = {
    "siliconflow": {
        # 硅基流动：OpenAI 兼容协议，一个 key 同时支持 embedding + chat
        # 注册：https://siliconflow.cn  → 有免费额度
        # 环境变量：export SILICONFLOW_API_KEY="sf-xxx"
        "base_url":     "https://api.siliconflow.cn/v1",
        "api_key_env":  "SILICONFLOW_API_KEY",
        "embed_model":  "BAAI/bge-m3",           # 中英文最佳，1024维
        "chat_model":   "Qwen/Qwen2.5-7B-Instruct",
        "sdk":          "openai_compat",
    },
    "zhipu": {
        # 智谱 AI：需要安装 pip install zhipuai
        # 注册：https://open.bigmodel.cn  → 有免费额度
        # 环境变量：export ZHIPU_API_KEY="xxx"
        "api_key_env":  "ZHIPU_API_KEY",
        "embed_model":  "embedding-3",            # 2048维，中文强
        "chat_model":   "glm-4-flash",            # 免费额度充足
        "sdk":          "zhipu",
    },
    "qwen": {
        # 通义千问：OpenAI 兼容协议，需要申请 DashScope API
        # 注册：https://dashscope.aliyun.com
        # 环境变量：export DASHSCOPE_API_KEY="sk-xxx"
        "base_url":     "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key_env":  "DASHSCOPE_API_KEY",
        "embed_model":  "text-embedding-v3",      # 1024/2048维可选
        "chat_model":   "qwen-plus",
        "sdk":          "openai_compat",
    },
    "openai": {
        # 保留 OpenAI 作为对照
        # 环境变量：export OPENAI_API_KEY="sk-xxx"
        "base_url":     None,                     # 官方默认
        "api_key_env":  "OPENAI_API_KEY",
        "embed_model":  "text-embedding-3-small",
        "chat_model":   "gpt-4o-mini",
        "sdk":          "openai_compat",
    },
}


# ──────────────────────────────────────────────────
# 客户端初始化（根据 PROVIDER 自动选择）
# ──────────────────────────────────────────────────

def _build_client():
    cfg = PROVIDER_CONFIG[PROVIDER]
    api_key = os.getenv(cfg["api_key_env"], "")
    if not api_key:
        raise EnvironmentError(
            f"\n[错误] 未找到 API Key 环境变量: {cfg['api_key_env']}\n"
            f"请运行: export {cfg['api_key_env']}='你的key'\n"
            f"申请地址见 00_配置提供商_先改这个.py 中 {PROVIDER} 的注释"
        )
    if cfg["sdk"] == "openai_compat":
        from openai import OpenAI
        kwargs = {"api_key": api_key}
        if cfg.get("base_url"):
            kwargs["base_url"] = cfg["base_url"]
        return OpenAI(**kwargs)
    elif cfg["sdk"] == "zhipu":
        from zhipuai import ZhipuAI
        return ZhipuAI(api_key=api_key)

_client = None

def _get_client():
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
    cfg = PROVIDER_CONFIG[PROVIDER]
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


def chat(prompt: str, temperature: float = 0.0) -> str:
    """
    调用 LLM 生成回答。
    temperature=0 保证输出稳定，便于对比实验。
    """
    cfg = PROVIDER_CONFIG[PROVIDER]
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
        return resp.choices[0].message.content.strip()


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """
    余弦相似度 = (a · b) / (|a| × |b|)
    为什么用余弦而不是欧氏距离 → 见《概念手册》
    """
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def model_info() -> dict:
    """返回当前使用的模型信息，方便打印确认"""
    cfg = PROVIDER_CONFIG[PROVIDER]
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
