# Agent 从零手写 · 可运行代码

对标 `code/rag/` 的 V1-V10。每个版本一个文件，从最小循环打穿到生产级。
配套讲解在 [`content/02-agent/agent-from-scratch/`](../../content/02-agent/agent-from-scratch/README.md)。

## 版本进度

| 文件 | 版本 | 状态 |
|---|---|---|
| `00_配置提供商.py` | 模型抽象层（chat/embed，懒加载） | ✅ |
| `01_v1_最小agent循环.py` | V1 最小 Agent 循环 + 一个工具 + 离线自测 | ✅ |
| `02_v2_react模式.py` | V2 ReAct（Thought/Action/Observation） | ⬜ |
| `…` | V3-V10 见主线施工图 | ⬜ |

## 快速开始

```bash
cd code/agent

# 离线验证循环逻辑（不需 API Key、不需装 openai）
python 01_v1_最小agent循环.py --selftest

# 真实运行
uv venv --python 3.12 && source .venv/bin/activate
uv pip install openai python-dotenv
cp .env.example .env          # 填 API Key，选 PROVIDER
python 00_配置提供商.py        # 验证连通性
python 01_v1_最小agent循环.py  # 跑「无循环 vs 有循环」对比
```

## 设计约定

- **provider 懒加载**：`import` 本模块不触发 `openai`，所以 `--selftest` 零依赖可跑。
- **LLM 可注入**：`run_agent(question, llm=...)` 的 `llm` 是 `(messages)->str`，
  真实跑传 `provider.chat`，自测传脚本化假模型——循环逻辑因此可离线验证。
- **每个版本自带 `--selftest`**：把「循环/解析/工具」三件事的正确性钉死，不依赖 API。
