# RAG 实战课程

售前转 AI 应用工程师 · 面向学习、实战和后续知识库建设的 RAG 学习仓库

---

## 先看什么

现在 `rag/docs` 已经不是“几篇并列文档”，而是一套带统一入口的知识体系。

推荐从这里开始：

- 统一入口：`rag/docs/index.html`
- 系统学习：`rag/docs/00_课程路线图.html`
- 快速回查：`rag/docs/ai-knowledge-hub.html`
- 做项目：`rag/docs/03_工程方法论手册.html`
- 讲解 / 面试：`rag/docs/rag-5d.html`
- 看全貌：`rag/docs/rag-knowledge-map.html`

如果你是第一次学，不要从“快速回查”开始；先打开 `index.html`，再按任务分流。

---

## 当前结构

### `docs/`：统一入口 + 知识页面

| 文件 | 角色 | 适合什么时候打开 |
|------|------|------------------|
| `index.html` | 统一入口，只负责任务分流 | 第一次进入 docs 时先打开 |
| `00_课程路线图.html` | 学习顺序，只负责“先学什么、后学什么” | 系统学习时打开 |
| `ai-knowledge-hub.html` | 快速回查，只负责术语、参数、问题定位、知识对象 | 已经知道要查什么时打开 |
| `01_概念手册_向量与检索.html` | 概念直觉，只负责 embedding / 向量 / 余弦等底层理解 | 跑 V1 代码前打开 |
| `02_代码讲解_V1V2.html` | 代码数据流，只负责解释 V1 / V2 怎么工作 | 准备运行或复习代码时打开 |
| `03_工程方法论手册.html` | 项目落地，只负责评估、badcase、实验、边界 | 开始做真实项目时打开 |
| `rag-knowledge-map.html` | 系统全貌，只负责模块关系和结构感 | 复习整体链路或做方案讲解时打开 |
| `rag-5d.html` | 横向辨析，只负责对比、场景判断、讲解表达、面试复习 | 要讲清楚 RAG 或做面试准备时打开 |
| `knowledge-updates.html` | 更新日志，只负责记录知识库结构和内容变更 | 想看最近做了什么调整时打开 |

### `demo/code/`：可运行代码

| 文件 | 内容 | 依赖 |
|------|------|------|
| `00_配置提供商_先改这个.py` | 切换模型提供商，配置 API Key | `openai`, `numpy` |
| `01_v1_最小RAG循环.py` | embedding + 余弦检索 + Prompt 注入 + 有无 RAG 对比 | 依赖 `00` |
| `02_v2_文档分块策略.py` | 3 种分块策略对比实验 | 依赖 `00` |
| `03_v3.5_黄金数据集.py` | 手标 + 合成数据集 + Recall@K + MRR 评估 | 依赖 `00` |

---

## 推荐路径

### 1. 系统学习

```text
docs/index.html
  ↓
docs/00_课程路线图.html
  ↓
docs/01_概念手册_向量与检索.html
  ↓
docs/02_代码讲解_V1V2.html#v1
  ↓
demo/code/01_v1_最小RAG循环.py
  ↓
docs/02_代码讲解_V1V2.html#v2
  ↓
demo/code/02_v2_文档分块策略.py
  ↓
docs/03_工程方法论手册.html
  ↓
demo/code/03_v3.5_黄金数据集.py
```

### 2. 做项目

```text
docs/index.html
  ↓
docs/03_工程方法论手册.html
  ↓
docs/rag-knowledge-map.html
  ↓
docs/rag-5d.html
  ↓
docs/ai-knowledge-hub.html
```

### 3. 快速回查

```text
docs/ai-knowledge-hub.html
  ↓
按主题 / 按问题 / 标准知识对象
  ↓
跳到概念页 / 代码页 / 工程页 / 5D / 知识地图
```

---

## 快速开始

```bash
# 1. 安装依赖
pip install openai numpy

# 2. 设置 API Key
# 国内推荐：硅基流动（有免费额度）→ https://siliconflow.cn
export SILICONFLOW_API_KEY="sf-xxx"

# 3. 打开并修改提供商配置
# rag/demo/code/00_配置提供商_先改这个.py
PROVIDER = "siliconflow"   # 可选: siliconflow | zhipu | qwen | openai

# 4. 验证连通性
python rag/demo/code/00_配置提供商_先改这个.py

# 5. 按顺序运行
python rag/demo/code/01_v1_最小RAG循环.py
python rag/demo/code/02_v2_文档分块策略.py
python rag/demo/code/03_v3.5_黄金数据集.py
```

如果你只是看文档，不需要先跑代码；如果你只是查术语，也不需要先通读路线图。

---

## 知识库化约定

这套 docs 现在按“知识库”而不是“散文档”维护，约定如下：

- `index.html` 是唯一统一入口，其他页面不再承担首页职责。
- 核心页面都说明“这页负责什么 / 不负责什么 / 下一步去哪”。
- `ai-knowledge-hub.html` 负责标准知识对象、术语、参数速查和问题索引。
- 高频知识点会逐步收敛成统一对象结构，方便后续做站内检索和 RAG 抽取。
- `knowledge-updates.html` 用于记录结构变化和重要内容更新。

---

## 国内模型选型

| 提供商 | 申请地址 | 环境变量 | 推荐场景 |
|--------|----------|----------|----------|
| 硅基流动 ★ | `siliconflow.cn` | `SILICONFLOW_API_KEY` | 学习首选，一个 key 搞定 embedding + chat |
| 智谱 AI | `open.bigmodel.cn` | `ZHIPU_API_KEY` | GLM-4-Flash 免费 |
| 通义千问 | `dashscope.aliyuncs.com` | `DASHSCOPE_API_KEY` | 企业级稳定 |
| OpenAI | `platform.openai.com` | `OPENAI_API_KEY` | 国际用户 |

---

课程和文档会持续更新。下一步会继续把更多知识点收敛成标准知识对象，并补机器可消费的知识索引。
